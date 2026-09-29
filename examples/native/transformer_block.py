#!/usr/bin/env python3
"""One pre-LN transformer encoder block, FP32 inference, every stage on the GPU.

    ln1 = LayerNorm(x)                           layer_norm
    qkv = ln1 Wqkv + bqkv                        matmul, bias_activation
    S_h = Q_h K_h^T         (strided head views) batched_matmul
    P_h = softmax(S_h / sqrt(d_head), causal)    softmax_rows
    A   = concat_h(P_h V_h) (strided output)     batched_matmul
    h1  = x + (A Wo + bo)                        matmul, bias_activation, add
    ln2 = LayerNorm(h1)                          layer_norm
    f   = gelu_tanh(ln2 W1 + b1) W2 + b2         matmul, bias_activation x2, matmul
    y   = h1 + f                                 add

15 launches through the paralyn.msl.tensor provider; the host only uploads
seeded inputs/weights (same xorshift32 stream and order as transformer_block.cpp)
and reads results back. There is no CPU fallback. Every audited stage is
verified against an independent float64 reference with a-priori running error
bounds (transformer_reference.py). Not PyTorch integration, not cuDNN/cuBLAS
compatibility, not training.
"""
import argparse
from array import array
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import paralyn as p
import transformer_reference as ref


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
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--model", type=int, default=48)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff", type=int, default=192)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-causal", dest="causal", action="store_false")
    args = parser.parse_args()
    if not 1 <= args.seed < 1 << 32:
        parser.error("seed must be in [1, 2^32-1]")
    if min(args.tokens, args.model, args.heads, args.ff) < 1 or args.model % args.heads:
        parser.error("tokens, model, heads, ff must be positive and heads must divide model")
    T, D, H, F = args.tokens, args.model, args.heads, args.ff
    Dh, W3 = D // H, 3 * D
    eps = ref.f32(1e-5)
    scale = ref.f32(1.0 / math.sqrt(Dh))

    rng = stream(args.seed)

    def values(count, scale_, offset=0.0):
        return array("f", (next(rng) * scale_ + offset for _ in range(count)))

    x, g1, be1 = values(T * D, 1.0), values(D, 0.25, 1.0), values(D, 0.25)
    wqkv, bqkv = values(D * W3, 0.125), values(W3, 0.25)
    wo, bo = values(D * D, 0.125), values(D, 0.25)
    g2, be2 = values(D, 0.25, 1.0), values(D, 0.25)
    w1, b1 = values(D * F, 0.125), values(F, 0.25)
    w2, b2 = values(F * D, 0.0625), values(D, 0.25)

    Desc = p.TensorDescriptor
    with ExitStack() as owners:
        context = owners.enter_context(p.Context())
        queue = owners.enter_context(context.queue())
        ops = owners.enter_context(p.load_tensor_operators(context))

        def upload(data):
            buffer = owners.enter_context(context.buffer(len(data) * 4))
            buffer.write(data.tobytes())
            return buffer

        def scratch(count):
            return owners.enter_context(context.buffer(count * 4))

        bx, bg1, bbe1, bwqkv, bbqkv, bwo, bbo, bg2, bbe2, bw1, bb1, bw2, bb2 = (
            upload(v) for v in (x, g1, be1, wqkv, bqkv, wo, bo, g2, be2, w1, b1, w2, b2))
        ln1, qkv_raw, qkv = scratch(T * D), scratch(T * W3), scratch(T * W3)
        scores, probs, attn = scratch(H * T * T), scratch(H * T * T), scratch(T * D)
        proj_raw, proj, h1, ln2 = scratch(T * D), scratch(T * D), scratch(T * D), scratch(T * D)
        f1_raw, f1, f2_raw, f2, y = scratch(T * F), scratch(T * F), scratch(T * D), scratch(T * D), scratch(T * D)
        events = []

        def keep(stage, event):
            if event is None:
                raise RuntimeError(f"{stage}: no GPU event for nonempty work")
            events.append((stage, owners.enter_context(event)))

        gelu = p.Activation.GELU_TANH
        keep("ln1", p.layer_norm_into(queue, ops, Desc(bx, (T, D)), Desc(bg1, (D,)), Desc(bbe1, (D,)),
                                      Desc(ln1, (T, D)), epsilon=eps))
        keep("qkv_matmul", p.matmul_into(queue, ops, Desc(ln1, (T, D)), Desc(bwqkv, (D, W3)), Desc(qkv_raw, (T, W3)),
                                         m=T, n=W3, k=D))
        keep("qkv_bias", p.bias_activation_into(queue, ops, Desc(qkv_raw, (T, W3)), Desc(bbqkv, (W3,)),
                                                Desc(qkv, (T, W3))))
        # Head views into qkv [T, 3D]: no copies, only strides and byte offsets.
        q_view = Desc(qkv, (H, T, Dh), 0, (Dh, W3, 1))
        kt_view = Desc(qkv, (H, Dh, T), D * 4, (Dh, 1, W3))
        v_view = Desc(qkv, (H, T, Dh), 2 * D * 4, (Dh, W3, 1))
        keep("scores", p.batched_matmul_into(queue, ops, q_view, kt_view, Desc(scores, (H, T, T)),
                                             batch=H, m=T, n=T, k=Dh))
        keep("softmax", p.softmax_rows_into(queue, ops, Desc(scores, (H, T, T)), Desc(probs, (H, T, T)),
                                            scale=scale, causal=args.causal))
        keep("attention", p.batched_matmul_into(queue, ops, Desc(probs, (H, T, T)), v_view,
                                                Desc(attn, (H, T, Dh), 0, (Dh, D, 1)), batch=H, m=T, n=Dh, k=T))
        keep("out_matmul", p.matmul_into(queue, ops, Desc(attn, (T, D)), Desc(bwo, (D, D)), Desc(proj_raw, (T, D)),
                                         m=T, n=D, k=D))
        keep("out_bias", p.bias_activation_into(queue, ops, Desc(proj_raw, (T, D)), Desc(bbo, (D,)), Desc(proj, (T, D))))
        keep("residual1", p.add_into(queue, ops, Desc(bx, (T, D)), Desc(proj, (T, D)), Desc(h1, (T, D))))
        keep("ln2", p.layer_norm_into(queue, ops, Desc(h1, (T, D)), Desc(bg2, (D,)), Desc(bbe2, (D,)),
                                      Desc(ln2, (T, D)), epsilon=eps))
        keep("ff1_matmul", p.matmul_into(queue, ops, Desc(ln2, (T, D)), Desc(bw1, (D, F)), Desc(f1_raw, (T, F)),
                                         m=T, n=F, k=D))
        keep("ff1_bias_gelu", p.bias_activation_into(queue, ops, Desc(f1_raw, (T, F)), Desc(bb1, (F,)),
                                                     Desc(f1, (T, F)), activation=gelu))
        keep("ff2_matmul", p.matmul_into(queue, ops, Desc(f1, (T, F)), Desc(bw2, (F, D)), Desc(f2_raw, (T, D)),
                                         m=T, n=D, k=F))
        keep("ff2_bias", p.bias_activation_into(queue, ops, Desc(f2_raw, (T, D)), Desc(bb2, (D,)), Desc(f2, (T, D))))
        keep("residual2", p.add_into(queue, ops, Desc(h1, (T, D)), Desc(f2, (T, D)), Desc(y, (T, D))))
        stages = []
        for stage, event in events:
            timing = event.timing()
            if not timing.completed or not timing.duration_seconds:
                raise RuntimeError(f"{stage}: GPU event did not complete with a valid duration")
            stages.append({"stage": stage, "gpu_duration_seconds": timing.duration_seconds})

        def read(buffer):
            out = array("f")
            out.frombytes(buffer.read())
            return out

        gpu = {"ln1": read(ln1), "qkv": read(qkv), "scores": read(scores), "probs": read(probs),
               "attention": read(attn), "projection": read(proj), "h1": read(h1), "ln2": read(ln2),
               "ff1": read(f1), "ff2": read(f2), "output": read(y)}
        device = context.device
        if args.artifacts:
            context.write_evidence(args.artifacts)
            execution = json.loads(Path(args.artifacts, "execution.json").read_text())
            if len(execution["launches"]) != len(events) or execution["cpu_fallback"] is not False:
                raise RuntimeError("exported evidence does not match the observed GPU launches")

    # ---- Independent float64 reference with running error bounds (verification only).
    def bounded(v):
        return [ref.exact(value) for value in v]

    def columns(w, rows, cols):
        return [[ref.exact(w[q * cols + j]) for q in range(rows)] for j in range(cols)]

    def linear(inp, rows, k, w_cols, bias):
        out = []
        for i in range(rows):
            row = inp[i * k:(i + 1) * k]
            out.extend(ref.add(ref.dot(row, column), bias[j]) for j, column in enumerate(w_cols))
        return out

    def layer_norm(xs, rows, cols, g, b):
        out = []
        for i in range(rows):
            out.extend(ref.layer_norm_row(xs[i * cols:(i + 1) * cols], g, b, eps))
        return out

    X = bounded(x)
    r_ln1 = layer_norm(X, T, D, bounded(g1), bounded(be1))
    r_qkv = linear(r_ln1, T, D, columns(wqkv, D, W3), bounded(bqkv))
    r_scores, r_probs = [], []
    for h in range(H):
        for i in range(T):
            q = r_qkv[i * W3 + h * Dh:i * W3 + (h + 1) * Dh]
            row = [ref.dot(q, r_qkv[j * W3 + D + h * Dh:j * W3 + D + (h + 1) * Dh]) for j in range(T)]
            r_scores.extend(row)
            r_probs.extend(ref.softmax_row(row, scale, i + 1 if args.causal else T))
    r_attn = [None] * (T * D)
    for h in range(H):
        v_columns = [[r_qkv[s * W3 + 2 * D + h * Dh + c] for s in range(T)] for c in range(Dh)]
        for i in range(T):
            prow = r_probs[(h * T + i) * T:(h * T + i + 1) * T]
            for c in range(Dh):
                r_attn[i * D + h * Dh + c] = ref.dot(prow, v_columns[c])
    r_proj = linear(r_attn, T, D, columns(wo, D, D), bounded(bo))
    r_h1 = [ref.add(a, b) for a, b in zip(X, r_proj)]
    r_ln2 = layer_norm(r_h1, T, D, bounded(g2), bounded(be2))
    r_f1 = [ref.gelu(v) for v in linear(r_ln2, T, D, columns(w1, D, F), bounded(b1))]
    r_f2 = linear(r_f1, T, F, columns(w2, F, D), bounded(b2))
    r_y = [ref.add(a, b) for a, b in zip(r_h1, r_f2)]
    refs = {"ln1": r_ln1, "qkv": r_qkv, "scores": r_scores, "probs": r_probs, "attention": r_attn,
            "projection": r_proj, "h1": r_h1, "ln2": r_ln2, "ff1": r_f1, "ff2": r_f2, "output": r_y}
    checks, hashes, worst = {}, {}, 0.0
    for stage in sorted(gpu):
        values_, bounds = gpu[stage], refs[stage]
        if len(values_) != len(bounds):
            raise RuntimeError(f"{stage}: size mismatch")
        stage_worst = stage_abs = 0.0
        for index, value in enumerate(values_):
            if not ref.within(value, bounds[index]):
                raise RuntimeError(f"{stage}: GPU value {value} at {index} differs from float64 reference "
                                   f"{bounds[index][0]} beyond bound {bounds[index][1]}")
            stage_worst = max(stage_worst, ref.ratio(value, bounds[index]))
            stage_abs = max(stage_abs, abs(value - bounds[index][0]))
        checks[stage] = {"worst_error_to_bound": stage_worst, "max_abs_error": stage_abs}
        hashes[stage] = hashlib.sha256(values_.tobytes()).hexdigest()
        worst = max(worst, stage_worst)
    max_abs = checks["output"]["max_abs_error"]

    print(f"Device: {device.name}; provider paralyn.msl.tensor; pre-LN encoder block T={T} d_model={D} "
          f"heads={H} d_head={Dh} d_ff={F} causal={'yes' if args.causal else 'no'} seed {args.seed}")
    for stage in stages:
        print(f"GPU {stage['stage']}: {stage['gpu_duration_seconds'] * 1e6:.3f} us")
    print(f"Output max |error| vs float64 reference: {max_abs:.3e}; worst error/bound over {len(gpu)} "
          f"audited stages: {worst:.3f}")
    if args.artifacts:
        report = {
            "application": "pre-LN transformer encoder block, FP32 inference (Python)",
            "provider": "paralyn.msl.tensor",
            "provider_artifact_sha256": hashlib.sha256(p.tensor_operators_artifact()).hexdigest(),
            "shapes": {"tokens": T, "model": D, "heads": H, "head": Dh, "ff": F},
            "seed": args.seed, "causal": args.causal, "epsilon": eps, "softmax_scale": scale,
            "gelu": "tanh approximation, FP32 constants", "gpu_commands": len(events), "stages": stages,
            "reference": "float64 from exact FP32 inputs with a-priori running error bounds",
            "tolerance": "docs/transformer-operators.md", "checks": checks,
            "worst_error_to_bound": worst, "max_abs_error_output": max_abs, "cpu_fallback": False,
            "stage_sha256": hashes, "output_sha256": hashes["output"],
        }
        Path(args.artifacts, "transformer-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verification: PASS pre-LN transformer block ({len(gpu)} stages, {T * D} outputs within float64 "
          f"running-error bounds; {len(events)} GPU commands)")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Transformer block failed: {error}", file=sys.stderr)
        sys.exit(1)
