#!/usr/bin/env python3
"""Run and independently audit the native C, C++ and Python hardware contract."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


KINDS = ("native-c", "native-cpp", "native-tests", "python-tests", "python-example")
EXPECTED_LAUNCHES = {"native-c": 2, "native-cpp": 1, "native-tests": 8,
                     "python-tests": 8, "python-example": 1}
MARKERS = {"native-c": "Verification: PASS native C vector_add",
           "native-cpp": "Verification: PASS native C++ affine",
           "native-tests": "Verification: PASS",
           "python-tests": "Native Python qualification: PASS",
           "python-example": "Verification: PASS (1003 independently checked values)"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def output(command):
    return subprocess.check_output([str(item) for item in command], text=True).strip()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_files(repo):
    files = {repo / "CMakeLists.txt", repo / "scripts/qualify_native.py"}
    for directory in ("include", "runtime", "backends/metal", "compiler", "cli",
                      "bindings/python/paralyn", "examples/native", "tests/native"):
        files.update(path for path in (repo / directory).rglob("*")
                     if path.is_file() and path.suffix in (".c", ".cpp", ".mm", ".h", ".hpp", ".py", ".cu"))
    return sorted(files)


def gather_provenance(repo, build, *, require_clean=False, require_all=True):
    revision = output(["git", "-C", repo, "rev-parse", "HEAD"])
    require(re.fullmatch(r"[0-9a-f]{40}", revision), "missing full Git revision")
    dirty = bool(output(["git", "-C", repo, "status", "--porcelain"]))
    require(not require_clean or not dirty, "permanent capture requires a clean checkout")
    version = output([build / "paralyn", "--version"])
    match = re.fullmatch(r"Paralyn v[^ ]+ \(LLVM ([^)]+)\)", version)
    require(match, "cannot derive selected LLVM version from paralyn --version")
    library = build / "libparalyn_native.dylib"
    require(library.is_file(), "native shared library is not built")
    names = ("paralyn", "libparalyn_native.dylib", "native_c", "native_cpp", "native_tests")
    binaries = {}
    for name in names:
        path = build / name
        if not path.is_file() and not require_all:
            continue
        require(path.is_file(), f"missing native qualification binary: {path}")
        binaries[name] = sha256(path)
    sources = {str(path.relative_to(repo)): sha256(path) for path in source_files(repo)}
    cache = build / "CMakeCache.txt"
    cache_values = {}
    if cache.is_file():
        for line in cache.read_text().splitlines():
            if line.startswith(("CMAKE_C_COMPILER:", "CMAKE_CXX_COMPILER:",
                                "CMAKE_OBJCXX_COMPILER:", "CMAKE_BUILD_TYPE:",
                                "CMAKE_OSX_DEPLOYMENT_TARGET:")):
                key, value = line.split("=", 1)
                cache_values[key.split(":", 1)[0]] = value
    compiler_versions = {}
    for key in ("CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_OBJCXX_COMPILER"):
        if key in cache_values:
            compiler_versions[key] = output([cache_values[key], "--version"])
    provenance = {
        "paralyn_commit": revision, "paralyn_dirty": dirty, "paralyn_version": version,
        "llvm_version": match.group(1), "native_abi_version": 1,
        "repository": str(repo), "build": str(build), "python_version": sys.version,
        "python_executable": sys.executable, "host_platform": platform.platform(),
        "build_configuration": cache_values, "host_compiler_versions": compiler_versions,
        "cmake_cache_sha256": sha256(cache) if cache.is_file() else None,
        "binary_sha256": binaries, "source_sha256": sources,
    }
    environment = dict(os.environ, PARALYN_LIBRARY=str(library),
                       PYTHONPATH=str(repo / "bindings/python"), PYTHONDONTWRITEBYTECODE="1",
                       PARALYN_COMMIT=revision, PARALYN_SOURCE_DIRTY="true" if dirty else "false",
                       PARALYN_LLVM_VERSION=match.group(1))
    environment.pop("PARALYN_ARTIFACT_DIR", None)
    return provenance, environment


def verify_unchanged(repo, build, provenance, output_directory, *, require_clean=False):
    require(output(["git", "-C", repo, "rev-parse", "HEAD"]) == provenance["paralyn_commit"],
            "Git revision changed during qualification")
    for name, digest in provenance["binary_sha256"].items():
        require(sha256(build / name) == digest, f"binary changed during qualification: {name}")
    current_sources = {str(path.relative_to(repo)): sha256(path) for path in source_files(repo)}
    require(current_sources == provenance["source_sha256"], "native source files changed during qualification")
    if require_clean:
        command = ["git", "-C", repo, "status", "--porcelain", "--", "."]
        if output_directory.is_relative_to(repo):
            command.append(":(exclude)" + str(output_directory.relative_to(repo)))
        require(not output(command), "source checkout changed during clean qualification")


def process(command, log, environment, *, expected_status=0, timeout=120):
    command = [str(value) for value in command]
    completed = subprocess.run(command, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=timeout, check=False)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout)
    require(completed.returncode == expected_status,
            f"command exited {completed.returncode}, expected {expected_status}; see {log}")
    return {"command": command, "exit_status": completed.returncode,
            "log": str(log), "transcript": completed.stdout}


def audit_execution(directory, expected_count, provenance):
    execution = json.loads((directory / "execution.json").read_text())
    require(execution["backend"] == "Metal" and execution["cpu_fallback"] is False,
            "native hardware evidence must use Metal without CPU fallback")
    require(execution["device"] and execution["registry_id"] > 0 and execution["os"],
            "missing physical device identity")
    for key in ("paralyn_commit", "paralyn_dirty", "llvm_version"):
        require(execution[key] == provenance[key], f"native execution provenance differs: {key}")
    require(execution["math_mode"] == "safe" and
            execution["floating_point_functions"] == "precise", "native numerical policy changed")
    launches = execution["launches"]
    require(len(launches) == expected_count,
            f"expected {expected_count} native GPU records, found {len(launches)}")
    previous_end = 0
    for index, launch in enumerate(launches):
        require(launch["command_status"] == "completed" and not launch["error"],
                f"native launch {index} did not complete successfully")
        start, end = launch["gpu_start_seconds"], launch["gpu_end_seconds"]
        require(math.isfinite(start) and math.isfinite(end) and 0 < start < end and start >= previous_end,
                f"native launch {index} lacks ordered positive GPU timestamps")
        require(launch["gpu_duration_seconds"] == end - start, "GPU duration is inconsistent")
        previous_end = end
        for label in ("grid", "block"):
            dimensions = launch[label]
            require(len(dimensions) == 3 and all(type(n) is int and n > 0 for n in dimensions),
                    f"native launch {index} has invalid {label}")
        compile_seconds = launch["pipeline_compile_seconds"]
        require(math.isfinite(compile_seconds) and compile_seconds >= 0, "invalid pipeline compilation time")
        source = directory / launch["source_file"]
        require(source.resolve().parent == directory.resolve(), "source artifact escapes evidence directory")
        text = source.read_text()
        require(text.strip() and "#pragma STDC FP_CONTRACT OFF" in text,
                "missing dispatched MSL or contraction policy")
    require((directory / "generated.metal").read_text().strip(), "missing latest generated MSL")
    current = execution["runtime_owned_current_buffer_bytes"]
    peak = execution["runtime_owned_peak_buffer_bytes"]
    require(type(current) is int and type(peak) is int and 0 <= current <= peak,
            "native buffer accounting is inconsistent")
    return execution


def command_for(kind, repo, build, module, evidence):
    if kind in ("native-c", "native-cpp", "native-tests"):
        binary = {"native-c": "native_c", "native-cpp": "native_cpp", "native-tests": "native_tests"}[kind]
        return [build / binary, module, evidence]
    script = "tests/native/test_python.py" if kind == "python-tests" else "examples/native/vector_add.py"
    return [sys.executable, repo / script, "--module", module, "--artifacts", evidence]


def run_case(kind, repo, build, module, directory, provenance, environment):
    require(kind in KINDS, "unknown native qualification case")
    require(not directory.exists(), f"case directory already exists: {directory}")
    directory.mkdir(parents=True)
    evidence_directory = directory / "evidence"
    record = process(command_for(kind, repo, build, module, evidence_directory),
                     directory / "process.log", environment)
    require(MARKERS[kind] in record["transcript"] and "Verification: FAIL" not in record["transcript"],
            f"{kind} did not report its independent CPU comparison")
    native_directory = evidence_directory / "primary" if kind == "python-tests" else evidence_directory
    execution = audit_execution(native_directory, EXPECTED_LAUNCHES[kind], provenance)
    actual_records = sorted(path.resolve() for path in evidence_directory.rglob("execution.json"))
    require(actual_records == [(native_directory / "execution.json").resolve()],
            f"{kind} has additional unaudited native execution records")
    if kind in ("native-c", "native-cpp"):
        require(execution["runtime_owned_current_buffer_bytes"] == 0,
                f"{kind} did not release its buffers before final capture")
    verified_events = len(execution["launches"])
    if kind == "python-tests":
        report = json.loads((evidence_directory / "python-qualification.json").read_text())
        require(report["status"] == "verified" and report["cpu_fallback"] is False,
                "Python native report does not verify GPU results")
        require(report["device"]["name"] == execution["device"] and
                report["device"]["registry_id"] == execution["registry_id"],
                "Python/native device reports differ")
        events = report["events"]
        require(len(events) == 9, "incomplete Python ownership/operation event set")
        for index, event in enumerate(events):
            start, end = event["gpu_start_seconds"], event["gpu_end_seconds"]
            require(event["completed"] is True and math.isfinite(start) and math.isfinite(end)
                    and 0 < start < end, "Python event lacks physical GPU completion timestamps")
            if index < len(execution["launches"]):
                launch = execution["launches"][index]
                require((start, end) == (launch["gpu_start_seconds"], launch["gpu_end_seconds"]),
                        "Python/native event timestamps differ")
        require(events[-1]["label"] == "released_context_queue_kernel_owners" and
                "context handle" in report["evidence_scope"], "missing final retained-owner event explanation")
        require("Verification: PASS" in (evidence_directory / "verification.txt").read_text(),
                "missing Python verification record")
        verified_events = len(events)
    record.pop("transcript")
    record.update(kind=kind, status="verified", device=execution["device"], os=execution["os"],
                  registry_id=execution["registry_id"], native_command_records=len(execution["launches"]),
                  verified_gpu_events=verified_events, cpu_fallback=False,
                  evidence_directory=str(evidence_directory))
    return record


def run_finalizer_policy(repo, build, directory, environment):
    """Exercise real C ABI handle errors, never label this host policy test GPU execution."""
    require(not directory.exists(), "finalizer policy directory already exists")
    directory.mkdir(parents=True)
    unobserved = (
        "import paralyn as p; context=p.Context(); buffer=context.buffer(4); "
        "buffer._lib.check(buffer._lib.api.pr_release(buffer.handle)); "
        "del buffer; print('UNREACHABLE_SUCCESS')"
    )
    observed = (
        "import paralyn as p\ncontext=p.Context()\nbuffer=context.buffer(4)\n"
        "buffer._lib.check(buffer._lib.api.pr_release(buffer.handle))\n"
        "try:\n buffer.close()\nexcept p.Error as error:\n assert error.code==p.Status.INVALID_HANDLE\n"
        "else:\n raise RuntimeError('explicit close did not report its error')\n"
        "context.close()\nprint('Explicit close error recovered')\n"
    )
    fatal = process([sys.executable, "-c", unobserved], directory / "unobserved.log", environment,
                    expected_status=1)
    require("fatal unhandled Buffer finalizer error" in fatal["transcript"] and
            "INVALID_HANDLE" in fatal["transcript"] and
            "UNREACHABLE_SUCCESS" not in fatal["transcript"], "finalizer failure was not fatal and visible")
    caught = process([sys.executable, "-c", observed], directory / "observed.log", environment)
    require("Explicit close error recovered" in caught["transcript"], "explicit close error was not catchable")
    for result in (fatal, caught):
        result.pop("transcript")
    return {"kind": "python-finalizer", "status": "verified", "gpu_launches": 0,
            "scope": "host exit/error policy through real C ABI invalid-handle failures; no GPU fault injection",
            "unobserved": fatal, "observed": caught}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    build, directory = args.build.resolve(), args.output.resolve()
    require(not directory.exists(), "qualification output directory must not already exist")
    provenance, environment = gather_provenance(repo, build, require_clean=args.require_clean)
    directory.mkdir(parents=True)
    report = {"status": "running", "started_utc": utc_now(), "provenance": provenance,
              "cases": [], "cpu_fallback": False}
    report_path = directory / "qualification.json"
    try:
        source = repo / "examples/native/kernels.cu"
        module = directory / "kernels.prk"
        shutil.copyfile(source, directory / "kernels.cu")
        compiled = process([build / "paralyn", "compile", source, "--output", module],
                           directory / "compile.log", environment)
        require("Compiled 2 verified kernel(s)" in compiled["transcript"] and
                "ERROR: native module" not in compiled["transcript"],
                "module compilation did not preserve the nonexecuting host contract")
        require(module.read_bytes()[:20] == b"PARALYN\0\1\0\0\0\1\0\0\0\2\0\0\0",
                "compiled module does not have the qualified version/two-kernel header")
        inspected = process([build / "paralyn", "inspect", source], directory / "inspect.txt", environment)
        require("vector_add" in inspected["transcript"] and "affine" in inspected["transcript"],
                "inspection does not identify both native fixture kernels")
        for record in (compiled, inspected):
            record.pop("transcript")
        report.update(module_sha256=sha256(module), kernel_source_sha256=sha256(source),
                      compile=compiled, inspect=inspected)
        for kind in KINDS:
            record = run_case(kind, repo, build, module, directory / kind, provenance, environment)
            report["cases"].append(record)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"Verified {kind}: {record['verified_gpu_events']} physical GPU events", flush=True)
        report["host_failure_policy"] = run_finalizer_policy(repo, build,
                                                            directory / "python-finalizer", environment)
        require(sha256(module) == report["module_sha256"] and
                (directory / "kernels.cu").read_bytes() == source.read_bytes(),
                "kernel module/source changed during qualification")
        verify_unchanged(repo, build, provenance, directory, require_clean=args.require_clean)
        devices = {(case["device"], case["registry_id"], case["os"]) for case in report["cases"]}
        require(len(devices) == 1, "native clients unexpectedly used different devices")
        report.update(status="verified", finished_utc=utc_now(),
                      native_command_records=sum(case["native_command_records"] for case in report["cases"]),
                      verified_gpu_events=sum(case["verified_gpu_events"] for case in report["cases"]),
                      observation="Python ownership case intentionally releases the context handle before its ninth launch; its pr_event_wait timestamps are audited separately from eight source-linked native records")
        (directory / "verification.txt").write_text(
            "Verification: PASS\nNative C, C++ and Python clients passed independent CPU comparisons; "
            "21 real GPU events audited, including 20 source-linked command records and one post-context-release event.\n")
        print(f"Native qualification: PASS ({report['verified_gpu_events']} GPU events); {directory}")
    except Exception as error:
        report.update(status="failed", finished_utc=utc_now(), error=str(error))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"Native qualification failed: {error}", file=sys.stderr)
        sys.exit(1)
