#!/usr/bin/env python3
"""Run transformer-operator and transformer-block qualification in isolated
evidence directories.

Each run must print a verification line produced by an actual comparison and
export runtime evidence whose completed Metal launches match what the program
observed. The block applications must show exactly the documented 15-launch
kernel sequence (every stage on the GPU, no CPU compute of any stage), and
block-match requires the C++ and Python applications to produce identical
output bytes for every audited stage.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

BLOCK_KERNELS = [
    "paralyn_layer_norm_f32",       # ln1 = LayerNorm(x)
    "paralyn_matmul_f32",           # qkv = ln1 Wqkv
    "paralyn_bias_activation_f32",  #       + bqkv
    "paralyn_batched_matmul_f32",   # scores[h] = Q_h K_h^T (strided head views)
    "paralyn_softmax_rows_f32",     # probs = softmax(scores / sqrt(d_head), causal)
    "paralyn_batched_matmul_f32",   # attn[:, h] = probs[h] V_h (strided output)
    "paralyn_matmul_f32",           # proj = attn Wo
    "paralyn_bias_activation_f32",  #        + bo
    "paralyn_add_f32",              # h1 = x + proj
    "paralyn_layer_norm_f32",       # ln2 = LayerNorm(h1)
    "paralyn_matmul_f32",           # f1 = ln2 W1
    "paralyn_bias_activation_f32",  #      gelu(. + b1)
    "paralyn_matmul_f32",           # f2 = f1 W2
    "paralyn_bias_activation_f32",  #      + b2
    "paralyn_add_f32",              # y = h1 + f2
]

parser = argparse.ArgumentParser()
parser.add_argument("--kind", choices=["ops-cpp", "ops-python", "block-cpp", "block-python", "block-match"],
                    required=True)
parser.add_argument("--build", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
build = args.build.resolve()
env = dict(os.environ, PARALYN_LIBRARY=str(build / "libparalyn_native.dylib"),
           PYTHONPATH=str(root / "bindings" / "python"))
env.pop("PARALYN_ARTIFACT_DIR", None)


def run(kind, evidence, extra=()):
    if kind == "ops-cpp":
        command = [str(build / "transformer_ops_tests"), str(evidence)]
    elif kind == "ops-python":
        command = [sys.executable, str(root / "tests/native/test_transformer_ops.py"), "--artifacts", str(evidence)]
    elif kind == "block-cpp":
        command = [str(build / "native_transformer_block"), "--artifacts", str(evidence), *extra]
    else:
        command = [sys.executable, str(root / "examples/native/transformer_block.py"), "--artifacts", str(evidence),
                   *extra]
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
    if kind.startswith("block"):
        kernels = [launch["kernel"] for launch in launches]
        if kernels != BLOCK_KERNELS:
            raise SystemExit(f"transformer block did not run every stage on the GPU in order: {kernels}")
        report = json.loads((evidence / "transformer-report.json").read_text())
        if (report["cpu_fallback"] is not False or report["gpu_commands"] != len(BLOCK_KERNELS)
                or report["worst_error_to_bound"] > 1):
            raise SystemExit("transformer report is inconsistent with verification")
    print(f"Evidence audit: {len(launches)} completed Metal launches")
    return report


with tempfile.TemporaryDirectory(prefix="paralyn-transformer-") as temporary:
    if args.kind != "block-match":
        run(args.kind, Path(temporary) / "proof")
    else:
        for extra, label in (((), "causal"), (("--no-causal",), "bidirectional")):
            reports = {kind: run(kind, Path(temporary) / f"{kind}-{label}", extra)
                       for kind in ("block-cpp", "block-python")}
            cpp, python = reports["block-cpp"], reports["block-python"]
            for key in ("provider_artifact_sha256", "stage_sha256", "output_sha256", "shapes", "seed", "causal"):
                if cpp.get(key) is None or cpp.get(key) != python.get(key):
                    raise SystemExit(f"C++ and Python transformer reports differ in {key}: "
                                     f"{cpp.get(key)} != {python.get(key)}")
            print(f"C++/Python transformer block ({label}): identical bytes for {len(cpp['stage_sha256'])} "
                  f"audited stages; output SHA-256 {cpp['output_sha256']}")
