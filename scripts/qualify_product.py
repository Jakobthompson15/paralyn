#!/usr/bin/env python3
"""Capture and independently audit the bounded arrays/MSL/CLI product qualification.

This launches real drivers. It is not a performance gate, a whole-portfolio claim,
or a replacement for Gate A/B and the native C ABI qualification. --audit-only
rechecks an existing immutable capture without compiling or submitting GPU work.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys


BINARIES = ("paralyn", "libparalyn_native.dylib", "array_tests", "msl_tests")
ARRAY_LENGTHS = (1, 17, 256, 257, 1003, 1000003)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def output(command):
    return subprocess.check_output([str(x) for x in command], text=True).strip()


def now():
    return datetime.now(timezone.utc).isoformat()


def sources(repo):
    paths = {repo / "CMakeLists.txt", repo / "LICENSE", repo / "THIRD_PARTY_LICENSES.md",
             repo / "tests/cli_tests.py", repo / "scripts/qualify_product.py",
             repo / "tests/product_audit_tests.py"}
    for name in ("include", "runtime", "backends", "compiler", "cli", "bindings/python",
                 "examples/native", "examples/metal", "tests/native", "tests/metal", "third_party"):
        paths.update(path for path in (repo / name).rglob("*") if path.is_file()
                     and "__pycache__" not in path.parts
                     and (path.suffix in (".c", ".cpp", ".mm", ".h", ".hpp", ".py", ".cu",
                                          ".metal", ".json", ".md") or path.name.startswith("LICENSE")))
    return {str(path.relative_to(repo)): sha256(path) for path in sorted(paths)}


def provenance(repo, build, require_clean):
    revision = output(["git", "-C", repo, "rev-parse", "HEAD"])
    require(re.fullmatch(r"[0-9a-f]{40}", revision), "cannot identify full Git revision")
    dirty = bool(output(["git", "-C", repo, "status", "--porcelain"]))
    require(not require_clean or not dirty, "--require-clean needs a clean source checkout")
    version = output([build / "paralyn", "--version"])
    match = re.fullmatch(r"Paralyn v[^ ]+ \(LLVM ([^)]+)\)", version)
    require(match, "cannot derive the selected LLVM version")
    configuration, compilers = {}, {}
    cache = build / "CMakeCache.txt"
    require(cache.is_file(), "CMakeCache.txt is required for build provenance")
    for line in cache.read_text().splitlines():
        if line.startswith(("CMAKE_C_COMPILER:", "CMAKE_CXX_COMPILER:", "CMAKE_OBJCXX_COMPILER:",
                            "CMAKE_BUILD_TYPE:", "CMAKE_OSX_DEPLOYMENT_TARGET:", "CMAKE_CXX_FLAGS:")):
            name, value = line.split("=", 1)
            configuration[name.split(":", 1)[0]] = value
    for name in ("CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_OBJCXX_COMPILER"):
        if name in configuration:
            compilers[name] = output([configuration[name], "--version"])
    return {"paralyn_commit": revision, "paralyn_dirty": dirty, "paralyn_version": version,
            "llvm_version": match[1], "repository": str(repo), "build": str(build),
            "python_version": sys.version, "python_executable": sys.executable,
            "host_platform": platform.platform(), "build_configuration": configuration,
            "host_compiler_versions": compilers, "source_sha256": sources(repo),
            "binary_sha256": {name: sha256(build / name) for name in BINARIES},
            "build_metadata_sha256": {name: sha256(build / name) for name in
                                      ("CMakeCache.txt", "build.ninja", "compile_commands.json")
                                      if (build / name).is_file()}}


def snapshot(repo, build, destination, recorded):
    for category, base, hashes in (("sources", repo, recorded["source_sha256"]),
                                   ("binaries", build, recorded["binary_sha256"]),
                                   ("build", build, recorded["build_metadata_sha256"])):
        for name, digest in hashes.items():
            target = destination / "snapshot" / category / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base / name, target)
            require(sha256(target) == digest, f"snapshot changed while copying: {name}")


def finite(value, label, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), f"invalid {label}")
    return value


def event(value):
    start = finite(value["gpu_start_seconds"], "GPU start", positive=True)
    end = finite(value["gpu_end_seconds"], "GPU end", positive=True)
    require(end > start, "GPU duration must be positive")
    if "completed" in value:
        require(value["completed"] is True, "GPU event is not completed")
    return start, end


def artifact_metadata(inputs):
    """Read container headers independently; runtime loaders still verify bodies."""
    ir = (inputs / "operators.prk").read_bytes()
    require(ir[:8] == b"PARALYN\0" and len(ir) >= 20, "invalid typed-IR header")
    version, policy, count = struct.unpack("<III", ir[8:20])
    data = (inputs / "kernels.prx").read_bytes()
    require(data[:8] == b"PARALYNX" and len(data) <= 16 * 1024 * 1024, "invalid executable header")
    position = 8
    def number():
        nonlocal position
        require(position + 4 <= len(data), "truncated executable integer")
        value = struct.unpack_from("<I", data, position)[0]
        position += 4
        return value
    def string():
        nonlocal position
        length = number()
        require(length <= 8 * 1024 * 1024 and position + length <= len(data), "truncated executable string")
        value = data[position:position + length].decode("utf-8")
        position += length
        return value
    executable = {"container_version": number(), "format": number(), "target": string(),
                  "numerical_policy": number(), "producer": string(), "producer_version": string(),
                  "source_name": string(), "source_sha256": string()}
    source = string()
    executable["entry_count"] = number()
    executable["sha256"] = sha256(inputs / "kernels.prx")
    require(hashlib.sha256(source.encode()).hexdigest() == executable["source_sha256"]
            and source.encode() == (inputs / "kernels.metal").read_bytes(), "executable source hash/body differs")
    return {"operators.prk": {"container_version": version, "numerical_policy": policy,
                               "kernel_count": count, "sha256": sha256(inputs / "operators.prk"),
                               "inspection_scope": "CUDA source and typed IR; public .prk inspection unavailable"},
            "kernels.prx": executable}


def audit_execution(directory, count, recorded, *, buffers_released=False, generated=True):
    execution = read_json(directory / "execution.json")
    require(execution["backend"] == "Metal" and execution["cpu_fallback"] is False,
            "qualification requires physical Metal execution without CPU fallback")
    require(execution["device"] and execution["os"] and execution["registry_id"] > 0,
            "missing physical GPU/OS identity")
    for key in ("paralyn_commit", "paralyn_dirty", "llvm_version"):
        require(execution[key] == recorded[key], f"execution provenance mismatch: {key}")
    require(execution["math_mode"] == "safe" and execution["floating_point_functions"] == "precise",
            "unexpected numerical policy")
    require(len(execution["launches"]) == count, f"wrong GPU record count in {directory}")
    current, peak = (execution[f"runtime_owned_{name}_buffer_bytes"] for name in ("current", "peak"))
    require(type(current) is int and type(peak) is int and 0 <= current <= peak,
            "invalid runtime buffer accounting")
    require(not buffers_released or current == 0, "driver did not release its array/workload buffers")
    for launch in execution["launches"]:
        require(launch["command_status"] == "completed" and not launch["error"], "GPU command failed")
        start, end = event(launch)
        require(launch["gpu_duration_seconds"] == end - start, "GPU duration differs from timestamps")
        for flag in ("gpu_duration_valid", "gpu_timestamps_valid"):
            require(launch.get(flag) is True, f"invalid GPU timing flag: {flag}")
        # Independent commands may have overlapping command-buffer intervals.
        # Dependency correctness comes from the driver's CPU comparisons, not
        # an invented cross-command timestamp ordering requirement.
        finite(launch["pipeline_compile_seconds"], "pipeline compilation time")
        for dimension in ("grid", "block"):
            values = launch[dimension]
            require(len(values) == 3 and all(type(n) is int and 0 < n <= 0xffffffff for n in values),
                    f"invalid {dimension} dimensions")
        source = directory / launch["source_file"]
        require(source.resolve().parent == directory.resolve(), "dispatched source escapes evidence directory")
        text = source.read_text()
        require(text.strip(), "empty dispatched shader source")
        if generated:
            require("#pragma STDC FP_CONTRACT OFF" in text, "missing generated contraction policy")
    require((directory / "generated.metal").read_text().strip(), "missing last dispatched source")
    return execution


def audit_arrays(directory, recorded, module_hash):
    cpp = audit_execution(directory / "arrays-cpp", 19, recorded, buffers_released=True)
    python = audit_execution(directory / "arrays-python", 19, recorded, buffers_released=True)
    expected = [(kernel, [(n + 255) // 256, 1, 1]) for n in ARRAY_LENGTHS
                for kernel in ("array_add", "array_affine", "array_affine")]
    expected.append(("array_add", [1, 1, 1]))
    for execution in (cpp, python):
        for launch, (kernel, grid) in zip(execution["launches"], expected):
            require((launch["kernel"], launch["grid"], launch["block"]) == (kernel, grid, [256, 1, 1]),
                    "array workload/geometry differs from qualified fixture")
    lifetimes = read_json(directory / "arrays-cpp/lifetimes.json")
    require(lifetimes["status"] == "verified" and lifetimes["cpu_fallback"] is False,
            "C++ retained-output tests did not pass")
    require([x["parent"] for x in lifetimes["events"]] == ["explicitly_closed", "wrapper_destroyed"],
            "missing C++ parent ownership cases")
    for value in lifetimes["events"]:
        event(value)
    report = read_json(directory / "arrays-python/array-qualification.json")
    require(report["status"] == "verified" and report["cpu_fallback"] is False
            and report["dtype"] == "float32" and report["shape_contract"] == "one-dimensional contiguous",
            "Python array qualification contract differs")
    require(report["operator_module_sha256"] == module_hash and
            report["native_library_sha256"] == recorded["binary_sha256"]["libparalyn_native.dylib"],
            "Python arrays used an unexpected module/library")
    require(report["device"]["name"] == python["device"] and
            report["device"]["registry_id"] == python["registry_id"], "Python/native device mismatch")
    require(len(report["events"]) == 19, "Python array events incomplete")
    labels = [(n, operation) for n in ARRAY_LENGTHS for operation in ("add", "affine", "dependent_affine")]
    labels.append((3, "same_input_alias"))
    for value, launch, label in zip(report["events"], python["launches"], labels):
        require((value["length"], value["operation"]) == label, "Python workload labels differ")
        require(event(value) == event(launch), "Python/native GPU timestamp mismatch")
    event(report["post_context_close_event"])
    require("Verification: PASS" in (directory / "arrays-python/verification.txt").read_text(),
            "missing Python independent comparison record")
    return [cpp, python]


def audit_msl(directory, recorded):
    report = read_json(directory / "msl/verification.json")
    require(report["verification"] == "PASS" and report["cpu_fallback"] is False,
            "MSL independent comparisons did not pass")
    # Count comes from the driver workload contract (4 add including alias,
    # 3 reduction, 3 transpose), not the manifest's number of entry points.
    require(report["gpu_events"] == len(report["events"]) == 10, "incomplete MSL driver events")
    require(type(report["negative_checks"]) is int and report["negative_checks"] >= 13,
            "MSL rejection qualification incomplete")
    require(report["workloads"] == ["vector_add", "block_reduce", "tiled_transpose"],
            "MSL workload list differs")
    execution = audit_execution(directory / "msl/evidence", 10, recorded,
                                buffers_released=True, generated=False)
    require(Counter(x["kernel"] for x in execution["launches"]) ==
            {"vector_add": 4, "block_reduce": 3, "tiled_transpose": 3}, "MSL kernel counts differ")
    original = (directory / "inputs/kernels.metal").read_bytes()
    for value, launch in zip(report["events"], execution["launches"]):
        require(event(value) == event(launch), "MSL driver/native timestamp mismatch")
        require((directory / "msl/evidence" / launch["source_file"]).read_bytes() == original,
                "MSL dispatched source differs from compiled input")
    return execution


def audit_doctor(directory, recorded):
    report = read_json(directory / "report.json")
    require(report["schema"] == "paralyn.report" and report["schema_version"] == 1
            and report["status"] == "passed" and report["cpu_fallback"] is False,
            "doctor did not report a verified physical probe")
    require(report["verification"]["status"] == "passed" and
            report["verification"]["reference"] == "builtin:vector-add-v1", "wrong doctor reference")
    reference = read_json(directory / "reference.json")
    require(reference["fixture"] == "builtin:vector-add-v1", "unknown probe fixture")
    require(len(reference["a"]) == len(reference["b"]) == len(reference["actual"]) ==
            report["verification"]["elements"] == 257, "incomplete builtin probe values")
    require(len(set(reference["actual"])) > 20, "probe inputs unexpectedly constant")
    for a, b, actual in zip(reference["a"], reference["b"], reference["actual"]):
        require(all(type(x) in (int, float) and math.isfinite(x) for x in (a, b, actual))
                and actual == a + b, "independent doctor CPU reference mismatch")
    execution = audit_execution(directory, 1, recorded)
    timing = report["timing"]
    require(timing["completed"] is True and timing["timestamps_valid"] is True
            and timing["duration_valid"] is True, "doctor timing flags invalid")
    require((timing["start_seconds"], timing["end_seconds"]) == event(execution["launches"][0])
            and timing["duration_seconds"] == timing["end_seconds"] - timing["start_seconds"],
            "doctor/native timing mismatch")
    require((directory / "probe.prk").read_bytes().startswith(b"PARALYN\0"), "missing serialized probe")
    return execution


def audit_capture(directory, *, check_manifest=True):
    capture = read_json(directory / "qualification.json")
    recorded = capture["provenance"]
    require(capture["schema"] == "paralyn.product.qualification" and capture["schema_version"] == 1,
            "unknown product capture schema")
    require(capture["status"] in ("running", "verified"), "capture is failed or incomplete")
    require(re.fullmatch(r"[0-9a-f]{40}", recorded["paralyn_commit"])
            and type(recorded["paralyn_dirty"]) is bool and recorded["llvm_version"] != "unknown",
            "capture lacks exact source/toolchain provenance")
    for category, key in (("sources", "source_sha256"), ("binaries", "binary_sha256"),
                          ("build", "build_metadata_sha256")):
        for name, digest in recorded[key].items():
            path = directory / "snapshot" / category / name
            require(path.resolve().is_relative_to((directory / "snapshot" / category).resolve())
                    and sha256(path) == digest, f"archived provenance mismatch: {category}/{name}")
    require(sha256(directory / "inputs/operators.cu") == recorded["source_sha256"]["bindings/python/operators.cu"],
            "operator input differs from archived source")
    for name in ("kernels.metal", "kernels.json"):
        require(sha256(directory / "inputs" / name) == recorded["source_sha256"]["examples/metal/" + name],
                "MSL input differs from archived source")
    for name, magic in (("operators.prk", b"PARALYN\0"), ("kernels.prx", b"PARALYNX")):
        require((directory / "inputs" / name).read_bytes().startswith(magic), "invalid compiled artifact header")
        require(sha256(directory / "inputs" / name) == capture["module_sha256"][name], "compiled module changed")
        inspected = read_json(directory / "inputs" / (name + ".inspect.json"))
        require(inspected["schema"] == "paralyn.report" and inspected["schema_version"] == 1
                and inspected["status"] == "checked",
                "missing serialized artifact inspection")
    metadata = artifact_metadata(directory / "inputs")
    require(metadata == read_json(directory / "inputs/artifact-metadata.json"), "retained artifact metadata differs")
    typed = metadata["operators.prk"]
    require((typed["container_version"], typed["numerical_policy"], typed["kernel_count"]) == (1, 1, 2),
            "operator artifact version/policy/kernel count differs")
    executable = metadata["kernels.prx"]
    expected_producer = recorded["paralyn_version"].split()[1][1:] + "+" + recorded["paralyn_commit"]
    expected_producer += ".dirty=" + ("true" if recorded["paralyn_dirty"] else "false")
    require(executable["producer"] == "paralyn" and executable["producer_version"] == expected_producer,
            "compiler was not reconfigured/rebuilt for the captured Git revision and dirty state")
    require((executable["container_version"], executable["format"], executable["numerical_policy"],
             executable["target"], executable["entry_count"]) == (1, 1, 1, "metal-msl3.1", 3),
            "executable artifact metadata contract differs")
    array_inspection = read_json(directory / "inputs/operators.prk.inspect.json")
    require(array_inspection["host_code_executed"] is False and array_inspection["gpu_work_submitted"] is False
            and array_inspection["source_sha256"] == sha256(directory / "inputs/operators.cu")
            and [x["name"] for x in array_inspection["inspection"]["kernels"]] == ["array_add", "array_affine"]
            and "paralyn.ir" in array_inspection["inspection"]["ir"],
            "missing nonexecuting operator source/typed-IR inspection")
    msl_inspection = read_json(directory / "inputs/kernels.prx.inspect.json")
    require(msl_inspection["source_sha256"] == sha256(directory / "inputs/kernels.prx")
            and [x["name"] for x in msl_inspection["entries"]] == ["vector_add", "block_reduce", "tiled_transpose"],
            "executable artifact inspection differs")
    processes = capture["processes"]
    require({x["label"] for x in processes} == {"compile-arrays", "inspect-arrays", "compile-msl", "inspect-msl",
            "arrays-cpp", "arrays-python", "msl", "cli", "doctor"}, "incomplete executed process set")
    for process in processes:
        require(process["exit_status"] == 0, "qualification driver did not exit successfully")
        require((directory / process["stdout"]).is_file() and (directory / process["stderr"]).is_file(),
                "missing actual process transcripts")
    for label, expected in (("arrays-cpp", 21), ("arrays-python", 20), ("msl", 10), ("doctor", 1)):
        records = [json.loads(line) for line in (directory / "logs" / (label + ".events.jsonl")).read_text().splitlines()]
        completed = [x for x in records if x["category"] == "completion"]
        require(len(completed) == expected and all(x["status"] == "completed" for x in completed),
                f"runtime completion trace disagrees with {label} event count")
        require(len({(x["context_id"], x["operation_id"]) for x in completed}) == expected,
                f"duplicate runtime completion in {label}")
    for label, marker in (("arrays-cpp", "Verification: PASS native C++ arrays"),
                          ("arrays-python", "Verification: PASS native Python arrays"),
                          ("msl", "Verification: PASS"), ("cli", ": PASS (")):
        process = next(x for x in processes if x["label"] == label)
        require(marker in (directory / process["stdout"]).read_text(), f"missing independent {label} PASS")
    executions = audit_arrays(directory, recorded, capture["module_sha256"]["operators.prk"])
    executions.append(audit_msl(directory, recorded))
    cli = read_json(directory / "cli/cli-test-summary.json")
    require(cli["verification"] == "PASS" and cli["physical_gpu_probe"] is True and
            type(cli["commands"]) is int and cli["commands"] > 0 and
            cli["cli_sha256"] == recorded["binary_sha256"]["paralyn"], "CLI suite did not qualify the captured binary")
    for name in ("python-0", "python-37", "python-live", "large-streams", "native-c"):
        report = read_json(directory / "cli" / name / "report.json")
        require(report["verification"]["status"] == "not_requested" and report["runtime_evidence"] is None,
                "arbitrary application stdout was falsely treated as verification/GPU evidence")
    executions.extend(audit_doctor(directory / path, recorded) for path in ("cli/doctor", "doctor"))
    expected_paths = {"arrays-cpp/execution.json", "arrays-python/execution.json", "msl/evidence/execution.json",
                      "cli/doctor/execution.json", "doctor/execution.json"}
    require({str(path.relative_to(directory)) for path in directory.rglob("execution.json")} == expected_paths,
            "additional or missing unaudited execution records")
    require(len({(x["device"], x["registry_id"], x["os"]) for x in executions}) == 1,
            "qualification unexpectedly used different physical devices")
    summary = {"source_linked_gpu_commands": sum(len(x["launches"]) for x in executions),
               "retained_output_gpu_events": 3, "verified_gpu_events": 53,
               "device": executions[0]["device"], "registry_id": executions[0]["registry_id"],
               "os": executions[0]["os"], "cpu_fallback": False, "cli_commands": cli["commands"]}
    require(summary["source_linked_gpu_commands"] + summary["retained_output_gpu_events"] ==
            summary["verified_gpu_events"], "inconsistent qualification event accounting")
    if check_manifest:
        require(capture["status"] == "verified" and capture["summary"] == summary,
                "capture summary differs from readback audit")
        manifest = read_json(directory / "artifact-sha256.json")
        actual = {str(path.relative_to(directory)): sha256(path) for path in directory.rglob("*")
                  if path.is_file() and path != directory / "artifact-sha256.json"}
        require(actual == manifest, "capture files differ from retained artifact manifest")
    return summary


def run_capture(repo, build, directory, require_clean):
    require(not directory.exists(), "qualification output directory must be new")
    recorded = provenance(repo, build, require_clean)
    directory.mkdir(parents=True)
    capture = {"schema": "paralyn.product.qualification", "schema_version": 1, "status": "running",
               "started_utc": now(), "provenance": recorded, "processes": [], "module_sha256": {},
               "scope": "FP32 arrays, public MSL workloads, CLI contracts and built-in GPU probe; not full portfolio qualification"}
    report = directory / "qualification.json"
    def save():
        write_json(report, capture)
    save()
    environment = dict(os.environ, PARALYN_COMMIT=recorded["paralyn_commit"],
                       PARALYN_SOURCE_DIRTY="true" if recorded["paralyn_dirty"] else "false",
                       PARALYN_LLVM_VERSION=recorded["llvm_version"], PARALYN_DEVICE="metal:0",
                       PARALYN_LIBRARY=str(build / "libparalyn_native.dylib"),
                       PYTHONPATH=str(repo / "bindings/python"), PYTHONDONTWRITEBYTECODE="1",
                       PARALYN_PYTHON=sys.executable)
    for key in ("PARALYN_ARTIFACT_DIR", "PARALYN_OPERATORS", "PARALYN_RUNTIME_LOG", "PARALYN_EVENT_LOG"):
        environment.pop(key, None)
    def run(label, command):
        logs = directory / "logs"
        logs.mkdir(exist_ok=True)
        env = dict(environment, PARALYN_RUNTIME_LOG=str(logs / (label + ".runtime.log")),
                   PARALYN_EVENT_LOG=str(logs / (label + ".events.jsonl")))
        started = now()
        result = subprocess.run([str(x) for x in command], cwd=repo, env=env, text=True,
                                capture_output=True, timeout=300, check=False)
        (logs / (label + ".stdout")).write_text(result.stdout)
        (logs / (label + ".stderr")).write_text(result.stderr)
        capture["processes"].append({"label": label, "command": [str(x) for x in command],
                                     "exit_status": result.returncode, "started_utc": started,
                                     "finished_utc": now(), "stdout": f"logs/{label}.stdout",
                                     "stderr": f"logs/{label}.stderr"})
        save()
        require(result.returncode == 0, f"{label} exited {result.returncode}; see {logs}")
        return result.stdout
    try:
        snapshot(repo, build, directory, recorded)
        inputs = directory / "inputs"
        inputs.mkdir()
        for source, destination in (("bindings/python/operators.cu", "operators.cu"),
                                    ("examples/metal/kernels.metal", "kernels.metal"),
                                    ("examples/metal/kernels.json", "kernels.json")):
            shutil.copyfile(repo / source, inputs / destination)
        cli = build / "paralyn"
        run("compile-arrays", [cli, "compile", inputs / "operators.cu", "--output", inputs / "operators.prk", "--json"])
        run("compile-msl", [cli, "compile", inputs / "kernels.metal", "--manifest", inputs / "kernels.json",
                            "--output", inputs / "kernels.prx", "--json"])
        for label, name in (("arrays", "operators.prk"), ("msl", "kernels.prx")):
            # .prk has no public inspection command yet. Preserve the same
            # compiler's source/typed-IR inspection and audit its serialized
            # header independently; do not present it as artifact reflection.
            inspection_input = inputs / ("operators.cu" if label == "arrays" else name)
            text = run("inspect-" + label, [cli, "inspect", inspection_input, "--json"])
            (inputs / (name + ".inspect.json")).write_text(text)
            capture["module_sha256"][name] = sha256(inputs / name)
        write_json(inputs / "artifact-metadata.json", artifact_metadata(inputs))
        run("arrays-cpp", [build / "array_tests", inputs / "operators.prk", directory / "arrays-cpp"])
        run("arrays-python", [sys.executable, repo / "tests/native/test_arrays.py", "--module",
                               inputs / "operators.prk", "--artifacts", directory / "arrays-python"])
        run("msl", [build / "msl_tests", inputs / "kernels.prx", directory / "msl"])
        run("cli", [sys.executable, repo / "tests/cli_tests.py", "--paralyn", cli,
                     "--source-root", repo, "--artifacts", directory / "cli"])
        run("doctor", [cli, "doctor", "--device", "metal:0", "--json", "--artifacts", directory / "doctor"])
        save()
        summary = audit_capture(directory, check_manifest=False)
        require(output(["git", "-C", repo, "rev-parse", "HEAD"]) == recorded["paralyn_commit"],
                "Git revision changed during qualification")
        require(sources(repo) == recorded["source_sha256"], "source files changed during qualification")
        for name, digest in recorded["binary_sha256"].items():
            require(sha256(build / name) == digest, f"binary changed during qualification: {name}")
        if require_clean:
            command = ["git", "-C", repo, "status", "--porcelain", "--", "."]
            if directory.is_relative_to(repo):
                command.append(":(exclude)" + str(directory.relative_to(repo)))
            require(not output(command), "source checkout changed during clean capture")
        capture.update(status="verified", finished_utc=now(), summary=summary,
                       observation="50 source-linked native command records plus 3 retained-output events after parent-handle release; drivers independently compare arrays/MSL results, and the recorder independently recomputes saved doctor outputs")
        save()
        (directory / "verification.txt").write_text("Verification: PASS\n53 physical GPU events audited; 50 have retained native command/source records. No CPU fallback.\n")
        write_json(directory / "artifact-sha256.json", {str(path.relative_to(directory)): sha256(path)
                   for path in sorted(directory.rglob("*")) if path.is_file()})
        audit_capture(directory)
        print(f"Product qualification: PASS ({summary['verified_gpu_events']} GPU events, {summary['device']}); {directory}")
    except BaseException as error:
        capture.update(status="failed", finished_utc=now(), error=str(error))
        save()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=Path("build"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--audit-only", action="store_true", help="read existing capture; no GPU execution")
    args = parser.parse_args()
    if args.audit_only:
        summary = audit_capture(args.output.resolve())
        print(f"Product readback audit: PASS ({summary['verified_gpu_events']} recorded GPU events; no new GPU work)")
    else:
        run_capture(Path(__file__).resolve().parents[1], args.build.resolve(), args.output.resolve(), args.require_clean)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"Product qualification failed: {error}", file=sys.stderr)
        sys.exit(1)
