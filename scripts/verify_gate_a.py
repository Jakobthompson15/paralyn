#!/usr/bin/env python3
"""Run the complete CUDA program and require physical-GPU evidence, never skip."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unicuda", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    args = parser.parse_args()
    completed = subprocess.run([
        str(args.unicuda.resolve()), "run", str(args.source.resolve()),
        "--artifacts", str(args.artifacts.resolve()),
    ], check=False)
    if completed.returncode:
        raise RuntimeError(f"complete CUDA program failed: exit {completed.returncode}")
    evidence = json.loads((args.artifacts / "execution.json").read_text())
    if evidence["backend"] != "Metal" or evidence["cpu_fallback"] is not False:
        raise RuntimeError("missing Metal execution contract")
    if not evidence["device"] or not evidence["launches"]:
        raise RuntimeError("missing physical device/launch evidence")
    for launch in evidence["launches"]:
        if launch["command_status"] != "completed" or launch["error"]:
            raise RuntimeError("Metal command did not complete successfully")
        if not 0 < launch["gpu_start_seconds"] < launch["gpu_end_seconds"]:
            raise RuntimeError("missing positive GPU execution timestamps")
    if (args.artifacts / "source.cu").read_bytes() != args.source.read_bytes():
        raise RuntimeError("preserved source differs from the executed input")
    for name in ("unicuda-ir.txt", "generated.metal"):
        if not (args.artifacts / name).read_text().strip():
            raise RuntimeError(f"missing {name}")
    verification = (args.artifacts / "verification.txt").read_text()
    if "Verification: PASS" not in verification or "Host exit status: 0" not in verification:
        raise RuntimeError("independent host verification did not pass")
    print(f"Gate A evidence verified: {args.artifacts}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        print(f"Gate A failed: {error}", file=sys.stderr)
        sys.exit(1)
