#!/usr/bin/env python3
"""Capture and audit the tensor, kernel-case and SPIR-V suites with retained evidence.

The CTest runners for these suites use temporary directories. This recorder runs
the same drivers into a new output directory and keeps every evidence directory.
It checks that each execution record is completed physical Metal work, has no CPU
fallback, and carries the embedded build revision/dirty state. That revision must
match the checkout. The recorder does not replace product/native/Gate A/B
qualification or the benchmark, and it is not a performance gate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

BINARIES = ("paralyn", "libparalyn_native.dylib", "tensor_tests", "native_mlp",
            "spirv_metal_tests", "spirv_importer_tests", "native-kernels.prk")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def now():
    return datetime.now(timezone.utc).isoformat()


def audit_execution(path, revision, dirty):
    """Return the completed launch count of one execution.json, or raise."""
    record = json.loads(Path(path).read_text())
    launches = record.get("launches") or []
    require(record.get("backend") == "Metal" and record.get("cpu_fallback") is False and launches,
            f"{path}: not physical Metal execution")
    require(record.get("paralyn_commit") == revision and record.get("paralyn_dirty") is dirty,
            f"{path}: revision {record.get('paralyn_commit')}/{record.get('paralyn_dirty')} "
            f"differs from checkout {revision}/{dirty}")
    for launch in launches:
        require(launch.get("command_status") == "completed" and not launch.get("error")
                and launch.get("gpu_duration_valid"), f"{path}: GPU launch did not complete")
        require((Path(path).parent / launch["source_file"]).is_file(), f"{path}: dispatched source missing")
    return len(launches)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    build, out = args.build.resolve(), args.output.resolve()
    require(not out.exists(), "output directory must be new")
    revision = git(repo, "rev-parse", "HEAD")
    dirty = bool(git(repo, "status", "--porcelain"))
    require(not (args.require_clean and dirty), "--require-clean needs a clean source checkout")
    for name in BINARIES:
        require((build / name).is_file(), f"missing build output {name} (configure with -DPARALYN_ENABLE_SPIRV=ON)")
    out.mkdir(parents=True)
    logs = out / "logs"
    logs.mkdir()
    capture = {"schema": "paralyn.suites.qualification", "schema_version": 1, "status": "running",
               "started_utc": now(), "paralyn_commit": revision, "paralyn_dirty": dirty,
               "paralyn_version": subprocess.check_output([str(build / "paralyn"), "--version"], text=True).strip(),
               "python": sys.version, "binary_sha256": {n: sha256(build / n) for n in BINARIES},
               "runs": [], "gpu_events": {}}
    report = out / "qualification.json"
    save = lambda: report.write_text(json.dumps(capture, indent=2) + "\n")
    save()
    env = dict(os.environ, PARALYN_LIBRARY=str(build / "libparalyn_native.dylib"),
               PYTHONPATH=str(repo / "bindings/python"), PYTHONDONTWRITEBYTECODE="1")
    for key in ("PARALYN_ARTIFACT_DIR", "PARALYN_COMMIT", "PARALYN_SOURCE_DIRTY", "PARALYN_OPERATORS",
                "PARALYN_RUNTIME_LOG", "PARALYN_EVENT_LOG"):
        env.pop(key, None)

    def run(label, command, marker="Verification: PASS"):
        started = now()
        result = subprocess.run([str(c) for c in command], cwd=repo, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300, check=False)
        (logs / f"{label}.log").write_text(result.stdout)
        capture["runs"].append({"label": label, "command": [str(c) for c in command], "exit_status": result.returncode,
                                "started_utc": started, "finished_utc": now(), "log": f"logs/{label}.log"})
        save()
        require(result.returncode == 0, f"{label} exited {result.returncode}; see logs/{label}.log")
        require(marker is None or marker in result.stdout, f"{label}: missing '{marker}'")
        return result.stdout

    try:
        py = sys.executable
        run("tensors-cpp", [build / "tensor_tests", out / "tensors/cpp"])
        run("tensors-python", [py, repo / "tests/native/test_tensors.py", "--artifacts", out / "tensors/python"])
        run("mlp-cpp", [build / "native_mlp", "--artifacts", out / "tensors/mlp-cpp"])
        run("mlp-python", [py, repo / "examples/native/mlp.py", "--artifacts", out / "tensors/mlp-python"])
        reports = {k: json.loads((out / f"tensors/{k}/mlp-report.json").read_text()) for k in ("mlp-cpp", "mlp-python")}
        for key in ("provider_artifact_sha256", "hidden_sha256", "output_sha256", "shapes", "seed"):
            require(reports["mlp-cpp"].get(key) is not None and reports["mlp-cpp"][key] == reports["mlp-python"].get(key),
                    f"C++ and Python MLP differ in {key}")
        for kind in ("mlp-cpp", "mlp-python"):
            kernels = [l["kernel"] for l in json.loads((out / f"tensors/{kind}/execution.json").read_text())["launches"]]
            require(kernels == ["paralyn_matmul_f32", "paralyn_bias_activation_f32"] * 2, f"{kind}: unexpected kernels {kernels}")
        spirv = [py, repo / "tests/spirv/run_spirv.py", "--paralyn", build / "paralyn", "--examples", repo / "examples/spirv",
                 "--negative", repo / "tests/spirv/negative", "--assembled", build / "spirv-fixtures"]
        run("spirv-importer", [build / "spirv_importer_tests", repo / "examples/spirv", repo / "tests/spirv/negative",
                               build / "spirv-fixtures"], marker="no GPU work submitted")
        run("spirv-cli", spirv + ["--mode", "cli", "--keep", out / "spirv/cli"], marker="no GPU dispatch")
        text = run("spirv-metal", spirv + ["--mode", "gpu", "--driver", build / "spirv_metal_tests", "--library",
                                           build / "libparalyn_native.dylib", "--python-path", repo / "bindings/python",
                                           "--keep", out / "spirv/metal"])
        python_spirv = re.search(r"Python SPIR-V: (\d+) GPU events", text)
        require(python_spirv, "Python SPIR-V event count missing")
        project = repo / "examples/cases/paralyn.toml"
        cases = re.findall(r"^\[cases\.([A-Za-z0-9_-]+)\]", project.read_text(), re.MULTILINE)
        require(cases, "no cases declared in examples/cases/paralyn.toml")
        for case in cases:
            run(f"case-{case}", [build / "paralyn", "verify", project, "--case", case, "--device", "metal:0",
                                 "--artifacts", out / "cases" / case])
        run("case-ir-affine", [build / "paralyn", "verify", build / "native-kernels.prk", "--case",
                               repo / "examples/cases/ir_affine.toml", "--device", "metal:0",
                               "--artifacts", out / "cases/ir-affine"])
        text = run("kernel-cases-suite", [py, repo / "tests/kernel_cases.py", "--paralyn", build / "paralyn",
                                          "--source", repo, "--prk", build / "native-kernels.prk"], marker="gpu_events")
        suite = json.loads(text.strip().splitlines()[-1])
        events = {str(p.relative_to(out)): audit_execution(p, revision, dirty) for p in sorted(out.rglob("execution.json"))}
        capture["gpu_events"] = {"retained_execution_records": events, "retained_total": sum(events.values()),
                                 "python_spirv_stdout_only": int(python_spirv[1]),
                                 "kernel_cases_suite_stdout_only": suite["gpu_events"],
                                 "kernel_cases_suite_negative_checks": suite["negative_checks"]}
        require(git(repo, "rev-parse", "HEAD") == revision, "Git revision changed during capture")
        require(all(sha256(build / n) == d for n, d in capture["binary_sha256"].items()), "binaries changed during capture")
        if args.require_clean:
            command = ["status", "--porcelain", "--", "."]
            if out.is_relative_to(repo):
                command.append(":(exclude)" + str(out.relative_to(repo)))
            require(not git(repo, *command), "source checkout changed during clean capture")
        capture.update(status="verified", finished_utc=now())
        save()
        (out / "artifact-sha256.json").write_text(json.dumps(
            {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}, indent=1) + "\n")
        print(f"Suites qualification: PASS ({capture['gpu_events']['retained_total']} retained GPU events in "
              f"{len(events)} records; +{int(python_spirv[1])} Python SPIR-V and {suite['gpu_events']} kernel-case-suite "
              f"events reported on stdout); {out}")
    except BaseException as error:
        capture.update(status="failed", finished_utc=now(), error=str(error))
        save()
        raise


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"Suites qualification failed: {error}", file=sys.stderr)
        sys.exit(1)
