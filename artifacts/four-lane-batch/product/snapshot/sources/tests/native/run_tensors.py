#!/usr/bin/env python3
"""Run tensor/matmul/MLP qualification in isolated evidence directories.

Each run must print a verification line produced by an actual comparison and
export runtime evidence whose completed GPU launches match the expected kernels.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--kind", choices=["cpp", "python", "mlp-cpp", "mlp-python", "mlp-match"], required=True)
parser.add_argument("--build", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
build = args.build.resolve()
env = dict(os.environ, PARALYN_LIBRARY=str(build / "libparalyn_native.dylib"),
           PYTHONPATH=str(root / "bindings" / "python"))
env.pop("PARALYN_ARTIFACT_DIR", None)


def run(kind, evidence):
    if kind == "cpp":
        command = [str(build / "tensor_tests"), str(evidence)]
    elif kind == "python":
        command = [sys.executable, str(root / "tests/native/test_tensors.py"), "--artifacts", str(evidence)]
    elif kind == "mlp-cpp":
        command = [str(build / "native_mlp"), "--artifacts", str(evidence)]
    else:
        command = [sys.executable, str(root / "examples/native/mlp.py"), "--artifacts", str(evidence)]
    result = subprocess.run(command, env=env, check=True, stdout=subprocess.PIPE, text=True)
    sys.stdout.write(result.stdout)
    if "Verification: PASS" not in result.stdout:
        raise SystemExit("qualification did not report a verified comparison")
    record = json.loads((evidence / "execution.json").read_text())
    launches = record["launches"]
    if record["backend"] != "Metal" or record["cpu_fallback"] is not False or not launches:
        raise SystemExit("evidence lacks physical Metal execution")
    for launch in launches:
        if launch["command_status"] != "completed" or launch["error"] or not launch["gpu_duration_valid"]:
            raise SystemExit(f"GPU launch did not complete: {launch}")
        if not (evidence / launch["source_file"]).is_file():
            raise SystemExit("dispatched MSL source is missing from evidence")
    report = None
    if kind.startswith("mlp"):
        kernels = [launch["kernel"] for launch in launches]
        expected = ["paralyn_matmul_f32", "paralyn_bias_activation_f32"] * 2
        if kernels != expected:
            raise SystemExit(f"MLP did not run every layer on the GPU in order: {kernels}")
        report = json.loads((evidence / "mlp-report.json").read_text())
        if report["worst_error_to_bound_output"] > 1 or report["cpu_fallback"] is not False:
            raise SystemExit("MLP report is inconsistent with verification")
    print(f"Evidence audit: {len(launches)} completed Metal launches")
    return report


with tempfile.TemporaryDirectory(prefix="paralyn-tensors-") as temporary:
    if args.kind != "mlp-match":
        run(args.kind, Path(temporary) / "proof")
    else:
        # Both applications must compute bit-identical hidden activations and outputs
        # with the same provider artifact; matching error statistics alone is not enough.
        reports = {kind: run(kind, Path(temporary) / kind) for kind in ("mlp-cpp", "mlp-python")}
        cpp, python = reports["mlp-cpp"], reports["mlp-python"]
        for key in ("provider_artifact_sha256", "hidden_sha256", "output_sha256", "shapes", "seed"):
            if cpp.get(key) is None or cpp.get(key) != python.get(key):
                raise SystemExit(f"C++ and Python MLP reports differ in {key}: {cpp.get(key)} != {python.get(key)}")
        print(f"C++/Python MLP comparison: identical hidden and output bytes (output SHA-256 "
              f"{cpp['output_sha256']}), provider artifact {cpp['provider_artifact_sha256']}")
