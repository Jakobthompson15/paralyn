#!/usr/bin/env python3
"""Two-layer FP32 MLP inference on the GPU: y = relu(x W1 + b1) W2 + b2.

Every layer (two matmuls and two bias/activation passes) runs through the
paralyn.msl.tensor provider on the native runtime; there is no CPU fallback.
The GPU result is compared with an independent float64-accumulated CPU
reference under the a-priori componentwise bound in docs/tensors-matmul.md.
Inputs are seeded and deterministic (same xorshift32 stream as mlp.cpp).
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import paralyn as p

U = 2.0 ** -24


def gamma(n):
    return n * U / (1 - n * U)


def stream(seed):
    state = seed
    while True:
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        yield ((state >> 16) - 32768) / 32768.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default=os.environ.get("PARALYN_ARTIFACT_DIR"))
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--in", dest="inputs", type=int, default=257)
    parser.add_argument("--hidden", type=int, default=130)
    parser.add_argument("--out", type=int, default=11)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not 1 <= args.seed < 1 << 32:
        parser.error("seed must be in [1, 2^32-1]")
    batch, n_in, hidden, n_out = args.batch, args.inputs, args.hidden, args.out
    rng = stream(args.seed)
    # Dyadic values scaled by powers of two are exact in FP32 (same order as mlp.cpp).
    x = [next(rng) for _ in range(batch * n_in)]
    w1 = [next(rng) * 0.0625 for _ in range(n_in * hidden)]
    b1 = [next(rng) * 0.25 for _ in range(hidden)]
    w2 = [next(rng) * 0.0625 for _ in range(hidden * n_out)]
    b2 = [next(rng) * 0.25 for _ in range(n_out)]

    with ExitStack() as owners:
        context = owners.enter_context(p.Context())
        tx = owners.enter_context(p.tensor(x, shape=(batch, n_in), context=context))
        tw1 = owners.enter_context(p.tensor(w1, shape=(n_in, hidden), context=context))
        tb1 = owners.enter_context(p.tensor(b1, shape=(hidden,), context=context))
        tw2 = owners.enter_context(p.tensor(w2, shape=(hidden, n_out), context=context))
        tb2 = owners.enter_context(p.tensor(b2, shape=(n_out,), context=context))
        z1 = owners.enter_context(p.matmul(tx, tw1))            # GPU: x W1
        h = owners.enter_context(p.bias_add(z1, tb1, relu=True))  # GPU: relu(. + b1)
        z2 = owners.enter_context(p.matmul(h, tw2))             # GPU: h W2
        y = owners.enter_context(p.bias_add(z2, tb2))           # GPU: . + b2
        gpu_h, gpu_y = h.to_host(), y.to_host()
        stages = []
        for label, stage in (("matmul1", z1), ("bias_relu1", h), ("matmul2", z2), ("bias2", y)):
            timing = stage.timing()
            if timing is None or not timing.completed or not timing.duration_seconds:
                raise RuntimeError(f"missing completed GPU event for {label}")
            stages.append({"stage": label, "gpu_duration_seconds": timing.duration_seconds})
        device = context.device
        for owner in (y, z2, h, z1, tb2, tw2, tb1, tw1, tx):
            owner.close()
        if args.artifacts:
            context.write_evidence(args.artifacts)

    # Independent CPU reference with float64 accumulation, plus the error bound.
    h64, e1 = [], []
    for i in range(batch):
        row = x[i * n_in:(i + 1) * n_in]
        for j in range(hidden):
            total = magnitude = 0.0
            for q, xv in enumerate(row):
                product = xv * w1[q * hidden + j]
                total += product
                magnitude += abs(product)
            h64.append(max(0.0, total + b1[j]))
            e1.append(gamma(n_in + 1) * (magnitude + abs(b1[j])))
    y64, e2 = [], []
    for i in range(batch):
        for j in range(n_out):
            total = propagated = magnitude = 0.0
            for q in range(hidden):
                wv, hv, ev = w2[q * n_out + j], h64[i * hidden + q], e1[i * hidden + q]
                total += hv * wv
                propagated += ev * abs(wv)
                magnitude += (abs(hv) + ev) * abs(wv)
            y64.append(total + b2[j])
            e2.append(propagated + gamma(hidden + 1) * (magnitude + abs(b2[j])))
    worst_h = worst_y = max_error = 0.0
    for index, value in enumerate(gpu_h):
        error = abs(value - h64[index])
        if not error <= e1[index]:
            raise RuntimeError(f"hidden layer exceeds bound at {index}: {error} > {e1[index]}")
        if e1[index]:
            worst_h = max(worst_h, error / e1[index])
    for index, value in enumerate(gpu_y):
        error = abs(value - y64[index])
        if not error <= e2[index]:
            raise RuntimeError(f"output exceeds bound at {index}: {error} > {e2[index]}")
        if e2[index]:
            worst_y = max(worst_y, error / e2[index])
        max_error = max(max_error, error)

    print(f"Device: {device.name}; provider paralyn.msl.tensor; x[{batch},{n_in}] "
          f"W1[{n_in},{hidden}] W2[{hidden},{n_out}] seed {args.seed}")
    for stage in stages:
        print(f"GPU {stage['stage']}: {stage['gpu_duration_seconds'] * 1e6:.3f} us")
    print(f"Max |error| vs float64 reference: {max_error:.3e}; worst error/bound: "
          f"hidden {worst_h:.3f}, output {worst_y:.3f}")
    if args.artifacts:
        report = {
            "application": "two-layer FP32 MLP inference (Python)",
            "provider": "paralyn.msl.tensor",
            "provider_artifact_sha256": hashlib.sha256(p.tensor_operators_artifact()).hexdigest(),
            "shapes": {"batch": batch, "in": n_in, "hidden": hidden, "out": n_out},
            "seed": args.seed, "gpu_commands": 4, "stages": stages,
            "reference": "float64 accumulation from FP32 inputs",
            "tolerance": "componentwise a-priori bound, docs/tensors-matmul.md",
            "max_abs_error": max_error, "worst_error_to_bound_hidden": worst_h,
            "worst_error_to_bound_output": worst_y, "cpu_fallback": False,
            "output_sha256": hashlib.sha256(gpu_y.tobytes()).hexdigest(),
        }
        Path(args.artifacts, "mlp-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verification: PASS two-layer FP32 MLP ({batch * n_out} outputs within "
          "float64-reference bound; 4 GPU commands)")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"MLP failed: {error}", file=sys.stderr)
        sys.exit(1)
