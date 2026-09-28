#!/usr/bin/env python3
"""CTest entry for the SPIR-V import profile.

--mode cli: paralyn compile/check/inspect/support on the shipped fixtures and
            negative fixtures (Metal compilation/reflection only; no dispatch).
--mode gpu: compile the fixtures, then run the C ABI driver and the Python
            binding test on the physical GPU with independent CPU references.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

NEGATIVE = {
    "kernel_model.spvasm": "P-SPIRV-KERNEL-MODEL",
    "bad_capability.spvasm": "P-SPIRV-CAPABILITY",
    "invalid_module.spvasm": "P-SPIRV-VALIDATION",
    "binding_mismatch.spvasm": "P-SPIRV-BINDING",
    "workgroup_size_mismatch.spvasm": "P-SPIRV-WORKGROUP-SIZE",
}
MODF_WRITE = Path(__file__).with_name("modf_write.spvasm")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def cli(paralyn, *args, expect=0):
    result = subprocess.run([paralyn, *args, "--json"], capture_output=True, text=True)
    require(result.returncode == expect,
            f"{' '.join(args)} exited {result.returncode}, expected {expect}: {result.stderr}{result.stdout}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    require(len(lines) == 1, f"{' '.join(args)} did not emit exactly one JSON document: {result.stdout}")
    return json.loads(lines[0])


def compile_fixtures(paralyn, examples, assembled, root):
    modules = {}
    for name in ("vector_add", "reduce_sum"):
        text_module, binary_module = root / f"{name}.prx", root / f"{name}-binary.prx"
        a = cli(paralyn, "compile", str(examples / f"{name}.spvasm"), "--output", str(text_module))
        b = cli(paralyn, "compile", str(assembled / f"{name}.spv"), "--output", str(binary_module))
        for report in (a, b):
            require(report["status"] == "compiled" and report["frontend"] == "spirv_vulkan_compute",
                    f"{name}: unexpected compile report")
            require("SPIRV-Tools vulkan-sdk-1.4.363.0" in report["toolchain"], "toolchain identity missing")
        require(a["spirv_sha256"] == b["spirv_sha256"] and
                a["generated_msl_sha256"] == b["generated_msl_sha256"],
                f"{name}: .spvasm (in-process) and .spv (build-time spirv-as) imports differ")
        modules[name] = text_module
    return modules


def reframe_source(container, transform):
    """Rewrite a v2 .prx's generated MSL with a consistent SHA-256 and length,
    as a hand-edited container would be. Returns the new container bytes."""
    import hashlib
    import struct

    def string_at(offset):
        (n,) = struct.unpack_from("<I", container, offset)
        return offset + 4 + n, container[offset + 4:offset + 4 + n]

    require(container[:8] == b"PARALYNX" and struct.unpack_from("<II", container, 8) == (2, 2),
            "not a SPIR-V-derived v2 container")
    at, _ = string_at(16)                   # target
    at += 4                                 # numerical policy
    for _ in range(3):                      # producer, producer version, source name
        at, _ = string_at(at)
    head = container[:at]
    at, _ = string_at(at)                   # source SHA-256
    at, source = string_at(at)
    source = transform(source.decode()).encode()
    digest = hashlib.sha256(source).hexdigest().encode()
    return (head + struct.pack("<I", len(digest)) + digest + struct.pack("<I", len(source)) + source +
            container[at:])


def tampered_containers(args, root, module):
    """A hand-edited v2 container must not pass load-time checks (review finding 3)."""
    original = module.read_bytes()
    identity = root / "identity.prx"
    identity.write_bytes(reframe_source(original, lambda text: text))
    require(identity.read_bytes() == original, "container re-framing is not an identity")
    require(cli(args.paralyn, "check", str(identity))["status"] == "checked", "re-framed control failed")
    cases = {
        "contract-pragma": lambda t: "#pragma clang fp contract(fast)\n" + t,
        "fast-fma": lambda t: t.replace("a.data[gl_GlobalInvocationID.x] + b.data[gl_GlobalInvocationID.x]",
                                        "fast::fma(a.data[gl_GlobalInvocationID.x], 1.0f, "
                                        "b.data[gl_GlobalInvocationID.x])"),
        "define": lambda t: t.replace("using namespace metal;", "#define X 1\nusing namespace metal;"),
    }
    rejected = 0
    for label, transform in cases.items():
        forged = root / f"forged-{label}.prx"
        forged.write_bytes(reframe_source(original, transform))
        require(forged.read_bytes() != original, f"{label}: tamper did not change the container")
        for command in ("check", "inspect"):
            report = cli(args.paralyn, command, str(forged), expect=1)
            require(report["status"] == "failed", f"{label}: {command} accepted a tampered container")
            rejected += 1
    return rejected


def cli_mode(args, root):
    modules = compile_fixtures(args.paralyn, args.examples, args.assembled, root)
    for name, entry, parameters in (("vector_add", "vector_add", ["a", "b", "c", "n"]),
                                    ("reduce_sum", "reduce_sum", ["input", "partial", "n", "scale"])):
        for target in (args.examples / f"{name}.spvasm", modules[name]):
            report = cli(args.paralyn, "check", str(target))
            require(report["status"] == "checked" and report["backend_compilation"] == "passed" and
                    report["gpu_work_submitted"] is False and report["host_code_executed"] is False,
                    f"check {target} did not pass Metal compilation/reflection")
            require(report["frontend"] == "spirv_vulkan_compute", f"check {target}: wrong frontend")
            e = report["entries"][0]
            require(e["name"] == entry and e["required_block"] == [64, 1, 1] and
                    [p["name"] for p in e["parameters"]] == parameters,
                    f"check {target}: reflected descriptor differs")
        inspected = cli(args.paralyn, "inspect", str(args.examples / f"{name}.spvasm"))
        require(inspected["status"] == "checked" and "backend_compilation" not in inspected,
                "inspect must not compile for a device")
    reduce = cli(args.paralyn, "inspect", str(args.examples / "reduce_sum.spvasm"))["entries"][0]
    require(reduce["builtins"] == ["GlobalInvocationId", "LocalInvocationId", "WorkgroupId",
                                   "NumWorkgroups", "WorkgroupSize"], "reduce_sum builtins")
    scale = reduce["parameters"][3]
    require(scale["origin"] == "uniform_member" and scale["block_offset"] == 4 and
            scale["descriptor_binding"] == 2 and scale["metal_slot"] == 2, "uniform member reflection")
    rejected = 0
    for fixture, diagnostic in NEGATIVE.items():
        output = root / (fixture + ".prx")
        for command in (["compile", str(args.negative / fixture), "--output", str(output)],
                        ["check", str(args.negative / fixture)]):
            report = cli(args.paralyn, *command, expect=1)
            require(report["status"] == "failed" and report["diagnostic"]["id"] == diagnostic and
                    report["diagnostic"]["stage"] == "import",
                    f"{fixture}: expected {diagnostic}, got {report.get('diagnostic')}")
            require(not output.exists(), f"{fixture}: a failed import wrote an output module")
            rejected += 1
    rejected += tampered_containers(args, root, modules["vector_add"])
    modf = cli(args.paralyn, "inspect", str(MODF_WRITE))["entries"][0]["parameters"]
    require([p["access_name"] for p in modf] == ["read", "read_write", "write", "read"],
            "modf_write: Modf pointer operand must make b read_write")
    run = cli(args.paralyn, "run", str(args.examples / "vector_add.spvasm"), expect=1)
    require(run["diagnostic"]["id"] == "P-KERNEL-CASE-REQUIRED", "run must not execute a kernel module")
    support = cli(args.paralyn, "support")["support"]["inputs"]
    row = next(r for r in support if r["name"] == "SPIR-V")
    require(row["implementation"] == "partial" and row["full_profile_qualified"] is False and
            "no CUDA/HIP" in row["scope"], "support row for SPIR-V is wrong")
    print(f"SPIR-V CLI: 4 fixture imports, {rejected} stable diagnostics, no GPU dispatch")


def gpu_mode(args, root):
    modules = compile_fixtures(args.paralyn, args.examples, args.assembled, root)
    evidence = root / "evidence"
    modf = root / "modf_write.prx"
    report = cli(args.paralyn, "compile", str(MODF_WRITE), "--output", str(modf))
    require(report["status"] == "compiled", "modf_write did not compile")
    subprocess.run([args.driver, str(modules["vector_add"]), str(modules["reduce_sum"]), str(modf),
                    str(evidence)], check=True)
    require((evidence / "execution.json").is_file(), "driver did not export runtime evidence")
    environment = dict(os.environ, PARALYN_LIBRARY=str(args.library),
                       PYTHONPATH=str(args.python_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    subprocess.run([sys.executable, str(Path(__file__).with_name("test_python.py")),
                    "--vector-add", str(modules["vector_add"]), "--reduce-sum", str(modules["reduce_sum"])],
                   check=True, env=environment)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cli", "gpu"), required=True)
    parser.add_argument("--paralyn", required=True)
    parser.add_argument("--examples", required=True, type=Path)
    parser.add_argument("--negative", required=True, type=Path)
    parser.add_argument("--assembled", required=True, type=Path)
    parser.add_argument("--driver")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--python-path", type=Path)
    parser.add_argument("--keep", type=Path, help="retain modules/evidence in this new directory")
    args = parser.parse_args()
    if args.keep:
        args.keep.mkdir(parents=True, exist_ok=False)
        (cli_mode if args.mode == "cli" else gpu_mode)(args, args.keep.resolve())
        return
    with tempfile.TemporaryDirectory(prefix=f"paralyn-spirv-{args.mode}-") as temporary:
        (cli_mode if args.mode == "cli" else gpu_mode)(args, Path(temporary))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, KeyError, subprocess.SubprocessError) as error:
        print(f"SPIR-V test failed: {error}", file=sys.stderr)
        sys.exit(1)
