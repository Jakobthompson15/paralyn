#!/usr/bin/env python3
"""Install a local wheel without network access and execute outside the checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import venv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="install/import only; never counts as GPU qualification")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    directory = args.output.resolve()
    if directory.is_relative_to(root):
        raise RuntimeError("installation test output must be outside the source checkout")
    if directory.exists():
        raise RuntimeError("installation test output must be new")
    directory.mkdir(parents=True)
    started = time.perf_counter()
    isolated = directory / "venv"
    venv.EnvBuilder(with_pip=True).create(isolated)
    python = isolated / "bin/python"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for key in ("PYTHONPATH", "PYTHONHOME", "PARALYN_LIBRARY", "PARALYN_OPERATORS", "PARALYN_ARTIFACT_DIR"):
        env.pop(key, None)
    def run(label, command):
        completed = subprocess.run([str(x) for x in command], cwd=directory, env=env,
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   timeout=180, check=False)
        (directory / f"{label}.log").write_text(completed.stdout)
        if completed.returncode:
            raise RuntimeError(f"{label} failed: {completed.stdout}")
        return completed.stdout
    run("install", [python, "-m", "pip", "install", "--no-index", "--no-deps", args.wheel.resolve()])
    discovered = json.loads(run("discovery", [python, "-I", "-c",
        "import json,paralyn; from paralyn.array import _operator_path; "
        "print(json.dumps({'package':paralyn.__file__,'library':paralyn._library().path,"
        "'operators':str(_operator_path()),'abi':paralyn._library().api.pr_abi_version()}))"]))
    for key in ("package", "library", "operators"):
        if not Path(discovered[key]).resolve().is_relative_to(isolated):
            raise RuntimeError(f"{key} escaped the isolated installation: {discovered[key]}")
    if discovered["abi"] != 1:
        raise RuntimeError("installed native ABI mismatch")
    packaged = json.loads((Path(discovered["package"]).parent / "_build.json").read_text())
    for name, key in (("library", "native_library_sha256"), ("operators", "operator_module_sha256")):
        if hashlib.sha256(Path(discovered[name]).read_bytes()).hexdigest() != packaged[key]:
            raise RuntimeError(f"installed {name} differs from the packaged provenance")
    env["PARALYN_COMMIT"] = packaged["packaging_source_commit"]
    env["PARALYN_SOURCE_DIRTY"] = "true" if packaged["packaging_source_dirty"] else "false"
    env["PARALYN_LLVM_VERSION"] = packaged["compiler_version"].split("LLVM ", 1)[1].rstrip(")")
    report = {"status": "installed_import_verified", "gpu_executed": False,
              "python": sys.version, "discovery": discovered,
              "packaging_provenance": packaged,
              "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest()}
    if not args.prepare_only:
        example = directory / "standalone_arrays.py"
        shutil.copyfile(root / "examples/native/arrays.py", example)
        transcript = run("example", [python, "-I", example, "--artifacts", directory / "gpu-evidence"])
        if "Verification: PASS native arrays" not in transcript:
            raise RuntimeError("installed example did not verify its CPU reference")
        evidence = json.loads((directory / "gpu-evidence/execution.json").read_text())
        if evidence["backend"] != "Metal" or evidence["cpu_fallback"] is not False or len(evidence["launches"]) != 2:
            raise RuntimeError("installed example lacks its two physical GPU commands")
        for launch in evidence["launches"]:
            if launch["command_status"] != "completed" or launch["error"] or not (
                    0 < launch["gpu_start_seconds"] < launch["gpu_end_seconds"]):
                raise RuntimeError("installed example GPU execution failed")
            if not (directory / "gpu-evidence" / launch["source_file"]).is_file():
                raise RuntimeError("installed example has no retained dispatched source")
        if evidence["runtime_owned_current_buffer_bytes"] != 0:
            raise RuntimeError("installed array example retained allocations after closing arrays")
        report.update(status="verified", gpu_executed=True, device=evidence["device"],
                      os=evidence["os"], gpu_events=2, cpu_fallback=False)
    report["elapsed_seconds"] = time.perf_counter() - started
    report["timing_scope"] = "local venv creation, offline wheel installation, import and requested example; dependencies already installed"
    (directory / "installation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Installed package: {report['status']}; {directory}")


if __name__ == "__main__":
    main()
