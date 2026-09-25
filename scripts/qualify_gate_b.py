#!/usr/bin/env python3
"""Qualify implemented CUDA correctness on physical Metal; never skip hardware.

This is Gate B's correctness suite. Benchmark qualification remains a separate
required gate and is not implied by this script's success.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import subprocess
import sys


POSITIVE_CASES = {
    "exact_arithmetic": 46,
    "xyz_builtins": 3,
    "aliases_and_host": 9,
    "integer_semantics": 5,
    "numerics": 3,
}
NEGATIVE_CASES = {
    "unsupported_shared": "shared_memory",
    "unsupported_double": "device_type_double",
    "unsupported_operator": "operator_-",
}


class HardwareUnavailable(RuntimeError):
    """Unavailable hardware cannot satisfy the qualification gate."""


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def process(arguments, log, timeout):
    result = subprocess.run([str(arg) for arg in arguments], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=timeout, check=False)
    log.write_text(result.stdout)
    return result


def verify_evidence(directory, source, expected_launches, expected_status,
                    require_clean=False, require_pass=True):
    evidence = json.loads((directory / "execution.json").read_text())
    require(evidence.get("backend") == "Metal", "execution backend is not Metal")
    require(evidence.get("cpu_fallback") is False, "CPU fallback contract missing")
    require(bool(evidence.get("device")) and bool(evidence.get("registry_id")),
            "physical GPU identity missing")
    require(bool(evidence.get("os")) and bool(evidence.get("llvm_version")),
            "execution/toolchain identity missing")
    if require_clean:
        revision = evidence.get("paralyn_commit", "")
        require(len(revision) == 40 and all(c in "0123456789abcdef" for c in revision),
                "clean capture requires a full Git revision")
        require(evidence.get("paralyn_dirty") is False, "clean capture used dirty sources")
    launches = evidence.get("launches", [])
    require(len(launches) == expected_launches,
            f"expected {expected_launches} completed launches; found {len(launches)}")
    for index, launch in enumerate(launches):
        require(launch.get("command_status") == "completed" and not launch.get("error"),
                f"launch {index} did not complete successfully")
        start = launch.get("gpu_start_seconds", 0)
        end = launch.get("gpu_end_seconds", 0)
        require(math.isfinite(start) and math.isfinite(end) and 0 < start < end,
                f"launch {index} lacks positive GPU timestamps")
        for dimension in ("grid", "block"):
            shape = launch.get(dimension, [])
            require(len(shape) == 3 and all(isinstance(value, int) and value > 0 for value in shape),
                    f"launch {index} has invalid {dimension} metadata")
        source_file = launch.get("source_file", "")
        require(source_file and Path(source_file).name == source_file,
                f"launch {index} lacks an associated shader artifact")
        require(bool((directory / source_file).read_text().strip()),
                f"launch {index} shader artifact is missing/empty")
    require((directory / "source.cu").read_bytes() == source.read_bytes(),
            "saved source differs from executed fixture")
    for name in ("paralyn-ir.txt", "generated.metal", "host.cpp"):
        require(bool((directory / name).read_text().strip()), f"empty/missing {name}")
    transcript = (directory / "verification.txt").read_text()
    require(f"Host exit status: {expected_status}" in transcript,
            "host exit status was not preserved")
    if require_pass:
        require("Verification: PASS" in transcript and "Verification: FAIL" not in transcript,
                "independent CPU comparison did not pass")
    else:
        require("Verification: PASS" not in transcript, "nonzero host exit incorrectly reported PASS")
    if source.stem == "aliases_and_host":
        alias_sources = {launch["source_file"] for launch in launches if launch["kernel"] == "alias_steps"}
        require(len(alias_sources) == 4, "four alias layouts did not produce four shader artifacts")
        require("Reusing kernel pipeline" in transcript, "alias fixture did not exercise pipeline reuse")
    return {"device": evidence["device"], "os": evidence["os"],
            "revision": evidence.get("paralyn_commit"),
            "dirty": evidence.get("paralyn_dirty"), "launches": len(launches)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paralyn", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--fixtures", type=Path,
                        default=Path(__file__).resolve().parent.parent / "tests" / "qualification")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    binary = args.paralyn.resolve()
    fixtures = args.fixtures.resolve()
    root = args.artifacts.resolve()
    require(binary.is_file(), f"missing Paralyn executable: {binary}")
    require(not root.exists() or not any(root.iterdir()),
            "qualification artifact directory is nonempty; refusing to overwrite evidence")
    root.mkdir(parents=True, exist_ok=True)
    summary = {"suite": "gate-b-correctness", "status": "failing",
               "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "benchmarks_included": False, "hardware_skip_allowed": False,
               "cases": []}
    try:
        for name, expected_launches in POSITIVE_CASES.items():
            source = fixtures / f"{name}.cu"
            require(source.is_file(), f"required fixture missing: {source}")
            directory = root / name
            result = process([binary, "run", source, "--artifacts", directory],
                             root / f"{name}.log", args.timeout)
            if "No physical Metal device is available" in result.stdout:
                raise HardwareUnavailable("no physical Metal device; qualification remains unavailable")
            require(result.returncode == 0,
                    f"{name} failed with exit {result.returncode}; see {name}.log\n{result.stdout[-3000:]}")
            identity = verify_evidence(directory, source, expected_launches, 0, args.require_clean)
            summary["cases"].append({"name": name, "status": "verified", **identity})
            print(f"Verified {name}: {expected_launches} physical-GPU launches", flush=True)

        source = fixtures / "aliases_and_host.cu"
        result = process([binary, "inspect", source], root / "inspect.log", args.timeout)
        require(result.returncode == 0, "inspect failed")
        require("HOST_MAIN_EXECUTED" not in result.stdout and "Executing on GPU" not in result.stdout,
                "inspect executed the host program or dispatched GPU work")
        require("alias_steps" in result.stdout and "unsigned_values" in result.stdout,
                "inspect did not analyze the actual kernels")
        summary["cases"].append({"name": "inspect_without_execution", "status": "verified"})

        directory = root / "host_exit"
        result = process([binary, "run", source, "--artifacts", directory, "--", "exit37"],
                         root / "host_exit.log", args.timeout)
        require(result.returncode == 37, f"host exit 37 was changed to {result.returncode}")
        identity = verify_evidence(directory, source, POSITIVE_CASES["aliases_and_host"], 37,
                                   args.require_clean, require_pass=False)
        summary["cases"].append({"name": "host_exit_propagation", "status": "verified", **identity})

        for name, diagnostic in NEGATIVE_CASES.items():
            source = fixtures / f"{name}.cu"
            result = process([binary, "inspect", source], root / f"{name}.log", args.timeout)
            require(result.returncode != 0, f"unsupported source accepted: {name}")
            require(diagnostic in result.stdout and f"{source}:" in result.stdout,
                    f"{name} lacks its source-located diagnostic")
            require("Verification: PASS" not in result.stdout, f"unsupported source printed PASS: {name}")
            summary["cases"].append({"name": name, "status": "verified_rejection"})
        summary["status"] = "verified"
        print(f"Gate B correctness qualification verified: {root}")
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        if isinstance(error, HardwareUnavailable):
            summary["status"] = "unavailable"
        summary["failure"] = str(error)
        print(f"Gate B correctness qualification failed: {error}", file=sys.stderr)
        return 1
    finally:
        summary["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Gate B correctness qualification failed: {error}", file=sys.stderr)
        sys.exit(1)
