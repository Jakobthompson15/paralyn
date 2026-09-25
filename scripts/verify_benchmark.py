#!/usr/bin/env python3
"""Execute and independently audit the complete Gate B benchmark protocol."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


SIZES = (1024, 65536, 1048576, 16777216)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checked_output(command):
    return subprocess.check_output(command, text=True).strip()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nonnegative(value, label, positive=False):
    value = float(value)
    require(math.isfinite(value) and (value > 0 if positive else value >= 0),
            f"invalid {label}: {value}")
    return value


def audit(artifacts, source, revision, dirty, llvm_version, transcript):
    protocol = json.loads((artifacts / "benchmark.json").read_text())
    evidence = json.loads((artifacts / "execution.json").read_text())
    with (artifacts / "samples.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(protocol["status"] == "verified", "benchmark did not verify its CPU comparisons")
    require(protocol["sizes"] == list(SIZES), "wrong benchmark sizes")
    require(protocol["warmups_per_variant_per_size"] == 10 and
            protocol["measured_iterations_per_variant_per_size"] == 100,
            "wrong warmup/measurement protocol")
    require(protocol["cpu_fallback"] is False and evidence["cpu_fallback"] is False,
            "CPU fallback cannot satisfy the hardware benchmark")
    require(evidence["backend"] == "Metal" and evidence["device"] and
            evidence["registry_id"] > 0 and evidence["os"], "missing physical Metal identity")
    require(evidence["paralyn_commit"] == revision and evidence["paralyn_dirty"] is dirty and
            evidence["llvm_version"] == llvm_version, "execution provenance does not match")
    require(evidence["math_mode"] == "safe" and
            evidence["floating_point_functions"] == "precise", "wrong numerical mode")
    require(protocol["pipeline_compilations"] == 2, "expected two cold pipeline compilations")
    require(protocol["completed_gpu_launches"] == protocol["samples"] ==
            len(rows) == len(evidence["launches"]) == 880, "incomplete benchmark sample set")
    expected_peak = max(SIZES) * 3 * 4
    require(protocol["runtime_owned_peak_buffer_bytes"] ==
            evidence["runtime_owned_peak_buffer_bytes"] == expected_peak,
            "unexpected runtime-owned buffer peak")
    require(protocol["runtime_owned_current_buffer_bytes"] ==
            evidence["runtime_owned_current_buffer_bytes"] == 0,
            "runtime buffers were not released")
    nonnegative(protocol["frontend_compile_seconds"], "frontend compilation", positive=True)
    nonnegative(protocol["pipeline_compile_seconds"], "pipeline compilation", positive=True)
    expected_rows = []
    for n in SIZES:
        for phase, count in (("warmup", 10), ("measured", 100)):
            for iteration in range(count):
                for order in range(2):
                    variant = "generated" if (iteration + order) % 2 == 0 else "handwritten"
                    expected_rows.append((n, phase, iteration, order, variant))
    source_files = {"generated": set(), "handwritten": set()}
    compile_total = 0
    previous_end = 0
    for number, (row, launch, expected) in enumerate(zip(rows, evidence["launches"], expected_rows)):
        identity = (int(row["elements"]), row["phase"], int(row["iteration"]),
                    int(row["order"]), row["variant"])
        require(identity == expected, f"sample {number} violates size/phase/alternating order")
        n, phase, iteration, _, variant = expected
        require(launch["kernel"] == ("vector_add" if variant == "generated" else
                                      "handwritten_vector_add"), f"sample {number} kernel mismatch")
        require(launch["grid"] == [(n + 255) // 256, 1, 1] and
                launch["block"] == [256, 1, 1], f"sample {number} geometry mismatch")
        require(launch["command_status"] == "completed" and not launch["error"],
                f"sample {number} Metal command failed")
        start = nonnegative(launch["gpu_start_seconds"], "GPU start", positive=True)
        end = nonnegative(launch["gpu_end_seconds"], "GPU end", positive=True)
        require(end > start >= previous_end, f"sample {number} GPU timestamps are unordered")
        previous_end = end
        gpu = nonnegative(row["gpu_seconds"], "sample GPU time", positive=True)
        require(gpu == launch["gpu_duration_seconds"] == end - start,
                f"sample {number} does not match command-buffer GPU duration")
        for key in ("pipeline_compile_seconds", "h2d_api_seconds", "d2h_api_seconds",
                    "h2d_copy_seconds", "d2h_copy_seconds", "total_seconds"):
            nonnegative(row[key], key)
        require(float(row["total_seconds"]) >= gpu, f"sample {number} total is shorter than GPU time")
        require(float(row["h2d_api_seconds"]) >= float(row["h2d_copy_seconds"]) and
                float(row["d2h_api_seconds"]) >= float(row["d2h_copy_seconds"]),
                f"sample {number} copy timing is inconsistent")
        require(int(row["runtime_owned_peak_buffer_bytes"]) == n * 3 * 4,
                f"sample {number} memory accounting is inconsistent")
        compile_seconds = float(row["pipeline_compile_seconds"])
        require(math.isclose(compile_seconds, launch["pipeline_compile_seconds"],
                             rel_tol=1e-10, abs_tol=1e-10), "compile timing does not match launch")
        cold = n == SIZES[0] and phase == "warmup" and iteration == 0
        require((compile_seconds > 0) == cold, "pipeline compilation outside the cold warmup")
        compile_total += compile_seconds
        source_file = artifacts / launch["source_file"]
        require(source_file.resolve().parent == artifacts.resolve(), "invalid source artifact path")
        text = source_file.read_text()
        require(text.strip() and "#pragma STDC FP_CONTRACT OFF" in text,
                f"sample {number} lacks source or contraction policy")
        source_files[variant].add(launch["source_file"])
    require(math.isclose(compile_total, protocol["pipeline_compile_seconds"],
                         rel_tol=1e-10, abs_tol=1e-10), "compile total is inconsistent")
    require(all(len(paths) == 1 for paths in source_files.values()) and
            source_files["generated"].isdisjoint(source_files["handwritten"]),
            "variants do not have distinct retained sources")
    require((artifacts / "source.cu").read_bytes() == source.read_bytes(),
            "retained CUDA source differs from executed input")
    for name in ("paralyn-ir.txt", "host.cpp", "handwritten.metal", "generated.metal"):
        require((artifacts / name).read_text().strip(), f"empty {name}")
    native_file = artifacts / next(iter(source_files["handwritten"]))
    require(native_file.read_bytes() == (artifacts / "handwritten.metal").read_bytes(),
            "dispatched handwritten source differs from retained reference")
    require("Verification: PASS" in (artifacts / "verification.txt").read_text() and
            "Benchmark verification: PASS; all 880 samples saved" in transcript,
            "benchmark did not report its complete independent CPU comparisons")
    return evidence["device"]


def main():
    parser = argparse.ArgumentParser()
    for name in ("benchmark", "paralyn", "source", "artifacts"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    benchmark, paralyn, source, artifacts = (
        getattr(args, name).resolve() for name in ("benchmark", "paralyn", "source", "artifacts"))
    require(not artifacts.exists(), "artifacts directory must not already exist")
    repo = Path(checked_output(["git", "-C", str(source.parent), "rev-parse", "--show-toplevel"]))
    revision = checked_output(["git", "-C", str(repo), "rev-parse", "HEAD"])
    require(re.fullmatch(r"[0-9a-f]{40}", revision), "missing full Git revision")
    dirty = bool(checked_output(["git", "-C", str(repo), "status", "--porcelain"]))
    require(not args.require_clean or not dirty, "permanent capture requires a clean checkout")
    version = checked_output([str(paralyn), "--version"])
    match = re.fullmatch(r"Paralyn v[^ ]+ \(LLVM ([^)]+)\)", version)
    require(match, f"cannot derive LLVM version from {version!r}")
    llvm_version = match.group(1)
    environment = dict(os.environ, PARALYN_COMMIT=revision,
                       PARALYN_SOURCE_DIRTY="true" if dirty else "false",
                       PARALYN_LLVM_VERSION=llvm_version)
    command = [str(benchmark), "--source", str(source), "--artifacts", str(artifacts)]
    before_hashes = {"benchmark_sha256": sha256(benchmark), "paralyn_sha256": sha256(paralyn),
                     "source_sha256": sha256(source)}
    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(command, env=environment, text=True, capture_output=True,
                               check=False, timeout=300)
    transcript = completed.stdout + completed.stderr
    sys.stdout.write(transcript)
    if artifacts.is_dir():
        (artifacts / "benchmark-transcript.txt").write_text(transcript)
    require(completed.returncode == 0, f"benchmark exited {completed.returncode}")
    require(before_hashes == {"benchmark_sha256": sha256(benchmark),
                              "paralyn_sha256": sha256(paralyn), "source_sha256": sha256(source)},
            "benchmark, compiler or source changed during capture")
    require(checked_output(["git", "-C", str(repo), "rev-parse", "HEAD"]) == revision,
            "Git revision changed during capture")
    if args.require_clean:
        status_command = ["git", "-C", str(repo), "status", "--porcelain", "--", "."]
        if artifacts.is_relative_to(repo):
            status_command.append(":(exclude)" + str(artifacts.relative_to(repo)))
        require(not checked_output(status_command), "source checkout changed during capture")
    device = audit(artifacts, source, revision, dirty, llvm_version, transcript)
    capture = dict(before_hashes, status="verified", command=command, paralyn_version=version,
                   paralyn_commit=revision, paralyn_dirty=dirty, device=device,
                   started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                   benchmark_exit_status=completed.returncode, cpu_fallback=False,
                   audit="880 rows, pairing/order, all sources, numerical policy, real command status/timestamps, buffer release and CPU comparison reports checked")
    (artifacts / "capture.json").write_text(json.dumps(capture, indent=2) + "\n")
    print(f"Benchmark capture: PASS ({device}); {artifacts}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"Benchmark capture failed: {error}", file=sys.stderr)
        sys.exit(1)
