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
}


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
    subprocess.run([args.driver, str(modules["vector_add"]), str(modules["reduce_sum"]), str(evidence)],
                   check=True)
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
