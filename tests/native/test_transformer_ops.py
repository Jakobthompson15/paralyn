#!/usr/bin/env python3
"""Real-GPU Python qualification of the transformer-block operators of the
paralyn.msl.tensor provider (batched strided matmul, row sum/max, softmax,
LayerNorm, GELU, residual add): low-level *_into calls over TensorDescriptors,
the high-level Tensor API, NaN/Inf policy and negative cases.

References are independent CPU code (examples/native/transformer_reference.py):
bit-exact FP32 emulation where every rounding is fixed by the contract, and a
float64 reference with a-priori running error bounds for every operator."""
import argparse
from array import array
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sys

import paralyn as p

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "native"))
import transformer_reference as ref  # noqa: E402

NAN, INF = math.nan, math.inf


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def rejects(function, status, operation):
    try:
        function()
    except p.Error as error:
        require(error.code == status, f"expected {status!r}, got {error.code!r}: {error}")
        require(error.operation == operation, f"wrong error operation {error.operation} (expected {operation})")
        return
    raise RuntimeError(f"invalid {operation} call unexpectedly succeeded")


def stream(seed):
    state = seed
    while True:
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        yield ((state >> 16) - 32768) / 32768.0


def values(count, seed, scale=1.0):
    g = stream(seed)
    return array("f", (next(g) * scale for _ in range(count)))


class Suite:
    def __init__(self, stack):
        self.stack = stack
        self.context = stack.enter_context(p.Context())
        self.queue = stack.enter_context(self.context.queue())
        self.ops = stack.enter_context(p.load_tensor_operators(self.context))
        self.events = 0
        self.kernels = {}
        self.worst = {}

    def buffer(self, data):
        data = array("f", data)
        b = self.stack.enter_context(self.context.buffer(max(4, len(data) * 4)))
        if len(data):
            b.write(data.tobytes())
        return b

    def output(self, count):
        return self.buffer(array("f", [12345.0] * max(1, count)))

    @staticmethod
    def read(b):
        out = array("f")
        out.frombytes(b.read())
        return out

    def completed(self, event, kernel, label):
        require(event is not None, f"{label}: missing GPU event")
        with event:
            timing = event.timing()
            require(timing.completed and timing.duration_seconds is not None and timing.duration_seconds > 0,
                    f"{label}: event is not a completed GPU interval")
        self.events += 1
        self.kernels[kernel] = self.kernels.get(kernel, 0) + 1

    def note(self, op, value):
        self.worst[op] = max(self.worst.get(op, 0.0), value)


def batched(s):
    seed = 10
    for batch, m, n, k in ((1, 1, 1, 1), (3, 17, 13, 5), (2, 33, 18, 47), (4, 16, 16, 0), (0, 3, 3, 3)):
        a = values(batch * m * k, seed)
        b = values(batch * k * n, seed + 1)
        seed += 2
        # B stored as [batch, n, k] and viewed transposed via strides.
        bt = array("f", [0.0] * len(b))
        for z in range(batch):
            for q in range(k):
                for j in range(n):
                    bt[(z * n + j) * k + q] = b[(z * k + q) * n + j]
        ab, bb, cb = s.buffer(a), s.buffer(bt), s.output(batch * m * n)
        event = p.batched_matmul_into(
            s.queue, s.ops, p.TensorDescriptor(ab, (batch, m, k)),
            p.TensorDescriptor(bb, (batch, k, n), strides=(n * k, 1, k)),
            p.TensorDescriptor(cb, (batch, m, n)), batch=batch, m=m, n=n, k=k)
        label = f"batched {(batch, m, n, k)}"
        if not batch * m * n:
            require(event is None, f"{label}: empty output fabricated an event")
            continue
        s.completed(event, "paralyn_batched_matmul_f32" if k else "paralyn_batched_fill_f32", label)
        out = s.read(cb)
        for z in range(batch):
            for i in range(m):
                for j in range(n):
                    seq = 0.0
                    xs, ys = [], []
                    for q in range(k):
                        x, y = a[(z * m + i) * k + q], b[(z * k + q) * n + j]
                        seq = ref.f32(seq + ref.f32(x * y))
                        xs.append(ref.exact(x))
                        ys.append(ref.exact(y))
                    gpu = out[(z * m + i) * n + j]
                    require(ref.same_bits(gpu, seq), f"{label}: differs from sequential FP32 at {(z, i, j)}")
                    bound = ref.dot(xs, ys)
                    require(ref.within(gpu, bound), f"{label}: float64 bound exceeded")
                    s.note("batched_matmul", ref.ratio(gpu, bound))


def reductions(s):
    for rows, cols, scale in ((1, 1, 1.0), (3, 255, 4.0), (2, 257, 4.0), (4, 0, 1.0), (2, 3001, 1.0)):
        x = values(rows * cols, 200 + cols, scale)
        if cols == 3001:  # non-dyadic, 2^40 dynamic range: exercises rounding
            x = array("f", (v / 3.0 * 2.0 ** (i % 41 - 20) for i, v in enumerate(x)))
        xb = s.buffer(x)
        xd = p.TensorDescriptor(xb, (rows, cols))
        for op in (p.Reduce.SUM, p.Reduce.MAX):
            ob = s.output(rows)
            event = p.reduce_rows_into(s.queue, s.ops, xd, p.TensorDescriptor(ob, (rows,)), op=op)
            label = f"{op.name} {(rows, cols)}"
            s.completed(event, "paralyn_reduce_rows_f32" if cols else "paralyn_fill_f32", label)
            out = s.read(ob)
            for i in range(rows):
                row = x[i * cols:(i + 1) * cols]
                if op == p.Reduce.SUM:
                    require(ref.same_bits(out[i], ref.tree_sum_f32(row)), f"{label}: fixed-order sum differs")
                    bound = ref.tree_sum([ref.exact(v) for v in row])
                    require(ref.within(out[i], bound), f"{label}: float64 bound exceeded")
                    s.note("row_sum", ref.ratio(out[i], bound))
                else:
                    require(ref.same_bits(out[i], ref.tree_max_f32(row)), f"{label}: max differs")
    special = [1.0, NAN, 2.0, -0.0, 0.0, -1.0, INF, -INF, 0.5]
    xb, ob = s.buffer(special), s.output(3)
    s.completed(p.reduce_rows_into(s.queue, s.ops, p.TensorDescriptor(xb, (3, 3)), p.TensorDescriptor(ob, (3,)),
                                   op=p.Reduce.MAX), "paralyn_reduce_rows_f32", "max specials")
    out = s.read(ob)
    require(math.isnan(out[0]) and ref.same_bits(out[1], 0.0) and out[2] == INF, "max NaN/signed-zero/inf policy")


def softmax(s):
    cases = ((1, 1, 1, 1.0, 4.0, False), (1, 3, 17, 0.125, 16.0, False), (2, 5, 12, 1.0, 8.0, True),
             (4, 9, 9, 0.35355339, 8.0, True), (1, 2, 1000, 1.0, 40.0, False), (1, 0, 4, 1.0, 1.0, False))
    for batch, rows, cols, scale, magnitude, causal in cases:
        x = values(batch * rows * cols, 400 + cols, magnitude)
        xb, ob = s.buffer(x), s.output(batch * rows * cols)
        shape = (batch, rows, cols)
        event = p.softmax_rows_into(s.queue, s.ops, p.TensorDescriptor(xb, shape), p.TensorDescriptor(ob, shape),
                                    scale=scale, causal=causal)
        label = f"softmax {shape} scale {scale} causal {causal}"
        if not batch * rows * cols:
            require(event is None, f"{label}: empty softmax fabricated an event")
            continue
        s.completed(event, "paralyn_softmax_rows_f32", label)
        out = s.read(ob)
        fs = ref.f32(scale)
        for z in range(batch):
            for i in range(rows):
                base = (z * rows + i) * cols
                visible = i + (cols - rows) + 1 if causal else cols
                bounds = ref.softmax_row([ref.exact(v) for v in x[base:base + cols]], fs, visible)
                for j in range(cols):
                    gpu = out[base + j]
                    if j >= visible:
                        require(ref.same_bits(gpu, 0.0), f"{label}: masked entry is not +0.0")
                        continue
                    require(ref.within(gpu, bounds[j]), f"{label}: float64 bound exceeded at {(z, i, j)}")
                    s.note("softmax", ref.ratio(gpu, bounds[j]))
    special = [1.0, NAN, 2.0, 1.0, INF, 2.0, -INF, -INF, -INF, 0.0, -INF, 0.0]
    xb, ob = s.buffer(special), s.output(12)
    s.completed(p.softmax_rows_into(s.queue, s.ops, p.TensorDescriptor(xb, (4, 3)), p.TensorDescriptor(ob, (4, 3))),
                "paralyn_softmax_rows_f32", "softmax specials")
    out = s.read(ob)
    require(all(math.isnan(v) for v in out[:9]), "softmax: NaN/+inf/all -inf rows must be all NaN")
    require(ref.same_bits(out[10], 0.0) and not math.isnan(out[9]), "softmax: -inf entry must be +0.0")


def layer_norm(s):
    for rows, cols, center, magnitude, eps in ((1, 1, 0.0, 1.0, 1e-5), (3, 17, 0.5, 2.0, 1e-5),
                                               (2, 257, -3.0, 4.0, 1e-6), (2, 64, 1000.0, 1.0, 1e-5),
                                               (0, 8, 0.0, 1.0, 1e-5)):
        x = array("f", (v + center for v in values(rows * cols, 600 + cols, magnitude)))
        g, b = values(cols, 601 + cols, 2.0), values(cols, 602 + cols, 0.5)
        xb, gb, bb, ob = s.buffer(x), s.buffer(g), s.buffer(b), s.output(rows * cols)
        event = p.layer_norm_into(s.queue, s.ops, p.TensorDescriptor(xb, (rows, cols)), p.TensorDescriptor(gb, (cols,)),
                                  p.TensorDescriptor(bb, (cols,)), p.TensorDescriptor(ob, (rows, cols)), epsilon=eps)
        label = f"layer_norm {(rows, cols)} center {center}"
        if not rows * cols:
            require(event is None, f"{label}: empty output fabricated an event")
            continue
        s.completed(event, "paralyn_layer_norm_f32", label)
        out = s.read(ob)
        gr, br = [ref.exact(v) for v in g], [ref.exact(v) for v in b]
        for i in range(rows):
            bounds = ref.layer_norm_row([ref.exact(v) for v in x[i * cols:(i + 1) * cols]], gr, br, ref.f32(eps))
            for j in range(cols):
                gpu = out[i * cols + j]
                require(ref.within(gpu, bounds[j]), f"{label}: float64 bound exceeded at {(i, j)}")
                s.note("layer_norm", ref.ratio(gpu, bounds[j]))


def gelu_and_add(s):
    x = values(5000, 800, 8.0)
    xb, ob = s.buffer(x), s.output(len(x))
    s.completed(p.bias_activation_into(s.queue, s.ops, p.TensorDescriptor(xb, (50, 100)), None,
                                       p.TensorDescriptor(ob, (50, 100)), activation=p.Activation.GELU_TANH),
                "paralyn_activation_f32", "gelu")
    out = s.read(ob)
    for i, v in enumerate(x):
        bound = ref.gelu(ref.exact(v))
        require(ref.within(out[i], bound), f"gelu: float64 bound exceeded at x={v}")
        s.note("gelu", ref.ratio(out[i], bound))
    special = [NAN, INF, -INF, 1e20, -1e20, 0.0, -0.0]
    sb, so = s.buffer(special), s.output(len(special))
    s.completed(p.bias_activation_into(s.queue, s.ops, p.TensorDescriptor(sb, (1, 7)), None,
                                       p.TensorDescriptor(so, (1, 7)), activation=p.Activation.GELU_TANH),
                "paralyn_activation_f32", "gelu specials")
    o = s.read(so)
    require(math.isnan(o[0]) and o[1] == INF and math.isnan(o[2]) and o[3] == ref.f32(1e20), "gelu specials")
    require(ref.same_bits(o[4], -0.0) and ref.same_bits(o[5], 0.0) and ref.same_bits(o[6], -0.0), "gelu zero signs")
    y = values(5000, 801, 3.0)
    yb, ab = s.buffer(y), s.output(len(y))
    s.completed(p.add_into(s.queue, s.ops, p.TensorDescriptor(xb, (5, 10, 100)), p.TensorDescriptor(yb, (5, 10, 100)),
                           p.TensorDescriptor(ab, (5, 10, 100))), "paralyn_add_f32", "add")
    out = s.read(ab)
    require(all(ref.same_bits(out[i], ref.f32(x[i] + y[i])) for i in range(len(x))), "add differs from FP32 x + y")


def negative(s, stack):
    xb, yb, ob, gb = s.buffer(values(64, 1)), s.buffer(values(64, 2)), s.output(64), s.buffer(values(8, 3))
    x2, y2, o2 = (p.TensorDescriptor(b, (2, 8)) for b in (xb, yb, ob))
    g8 = p.TensorDescriptor(gb, (8,))
    other = stack.enter_context(p.Context())
    other_queue = stack.enter_context(other.queue())
    other_ops = stack.enter_context(p.load_tensor_operators(other))
    foreign = stack.enter_context(other.buffer(256))
    f2 = p.TensorDescriptor(foreign, (2, 8))
    plain = stack.enter_context(p.Module.load(s.context, p.tensor_operators_artifact()))
    calls = {
        "batched_matmul_f32": lambda q, m, **kw: p.batched_matmul_into(
            q, m, kw.get("a", p.TensorDescriptor(xb, (2, 2, 4))), p.TensorDescriptor(yb, (2, 4, 3)),
            kw.get("c", p.TensorDescriptor(ob, (2, 2, 3))), batch=kw.get("batch", 2), m=2, n=3, k=4),
        "reduce_rows_f32": lambda q, m, **kw: p.reduce_rows_into(q, m, kw.get("x", x2), kw.get("out", p.TensorDescriptor(ob, (2,)))),
        "softmax_rows_f32": lambda q, m, **kw: p.softmax_rows_into(q, m, kw.get("x", x2), kw.get("out", o2),
                                                                   scale=kw.get("scale", 1.0), causal=kw.get("causal", False)),
        "layer_norm_f32": lambda q, m, **kw: p.layer_norm_into(q, m, x2, kw.get("gamma", g8), g8, kw.get("out", o2),
                                                               epsilon=kw.get("epsilon", 1e-5)),
        "add_f32": lambda q, m, **kw: p.add_into(q, m, x2, kw.get("y", y2), kw.get("out", o2)),
    }
    bogus_queue = p.Queue._adopt(s.queue._lib, 0xDEADBEEF)  # never released: not a live handle
    for name, call in calls.items():
        rejects(lambda: call(bogus_queue, s.ops), p.Status.INVALID_HANDLE, name)
        rejects(lambda: call(other_queue, s.ops), p.Status.CONTEXT_MISMATCH, name)
        rejects(lambda: call(s.queue, other_ops), p.Status.CONTEXT_MISMATCH, name)
        rejects(lambda: call(s.queue, plain), p.Status.INVALID_ARGUMENT, name)
    rejects(lambda: calls["batched_matmul_f32"](s.queue, s.ops, batch=1), p.Status.INVALID_ARGUMENT, "batched_matmul_f32")
    rejects(lambda: calls["batched_matmul_f32"](s.queue, s.ops, c=p.TensorDescriptor(ob, (2, 2, 3), strides=(0, 3, 1))),
            p.Status.INVALID_ARGUMENT, "batched_matmul_f32")
    rejects(lambda: calls["batched_matmul_f32"](s.queue, s.ops, c=p.TensorDescriptor(xb, (2, 2, 3), 8)),
            p.Status.INVALID_ARGUMENT, "batched_matmul_f32")
    rejects(lambda: calls["batched_matmul_f32"](s.queue, s.ops, a=p.TensorDescriptor(foreign, (2, 2, 4))),
            p.Status.CONTEXT_MISMATCH, "batched_matmul_f32")
    rejects(lambda: calls["reduce_rows_f32"](s.queue, s.ops, out=p.TensorDescriptor(ob, (3,))),
            p.Status.INVALID_ARGUMENT, "reduce_rows_f32")
    rejects(lambda: calls["reduce_rows_f32"](s.queue, s.ops, out=p.TensorDescriptor(xb, (2,), 4 * 60)),
            p.Status.UNSUPPORTED, "reduce_rows_f32")
    rejects(lambda: calls["reduce_rows_f32"](s.queue, s.ops, x=p.TensorDescriptor(xb, (2, 8), strides=(1, 2))),
            p.Status.UNSUPPORTED, "reduce_rows_f32")
    for bad in (0.0, -1.0, NAN, INF):
        rejects(lambda: calls["softmax_rows_f32"](s.queue, s.ops, scale=bad), p.Status.INVALID_ARGUMENT, "softmax_rows_f32")
    rejects(lambda: calls["softmax_rows_f32"](s.queue, s.ops, x=p.TensorDescriptor(xb, (8, 2)),
                                              out=p.TensorDescriptor(ob, (8, 2)), causal=True),
            p.Status.INVALID_ARGUMENT, "softmax_rows_f32")
    rejects(lambda: calls["softmax_rows_f32"](s.queue, s.ops, out=x2), p.Status.INVALID_ARGUMENT, "softmax_rows_f32")
    rejects(lambda: calls["softmax_rows_f32"](s.queue, s.ops, out=f2), p.Status.CONTEXT_MISMATCH, "softmax_rows_f32")
    rejects(lambda: calls["layer_norm_f32"](s.queue, s.ops, gamma=p.TensorDescriptor(gb, (7,))),
            p.Status.INVALID_ARGUMENT, "layer_norm_f32")
    for bad in (0.0, NAN, -1e-5):
        rejects(lambda: calls["layer_norm_f32"](s.queue, s.ops, epsilon=bad), p.Status.INVALID_ARGUMENT, "layer_norm_f32")
    rejects(lambda: calls["add_f32"](s.queue, s.ops, y=p.TensorDescriptor(yb, (8, 2))), p.Status.INVALID_ARGUMENT, "add_f32")
    rejects(lambda: calls["add_f32"](s.queue, s.ops, out=x2), p.Status.INVALID_ARGUMENT, "add_f32")
    rejects(lambda: calls["add_f32"](s.queue, s.ops, out=p.TensorDescriptor(yb, (2, 8), 4 * 32)), p.Status.UNSUPPORTED, "add_f32")
    rejects(lambda: calls["add_f32"](s.queue, s.ops, y=f2), p.Status.CONTEXT_MISMATCH, "add_f32")
    # Zero-work calls still validate ownership.
    rejects(lambda: p.add_into(s.queue, s.ops, p.TensorDescriptor(xb, (0, 3)), p.TensorDescriptor(yb, (0, 3)),
                               p.TensorDescriptor(foreign, (0, 3))), p.Status.CONTEXT_MISMATCH, "add_f32")
    for bad_type in (lambda: p.softmax_rows_into(s.queue, s.ops, x2, o2, scale="1"),
                     lambda: p.softmax_rows_into(s.queue, s.ops, x2, o2, causal=1),
                     lambda: p.layer_norm_into(s.queue, s.ops, x2, g8, g8, o2, epsilon=True)):
        try:
            bad_type()
        except TypeError:
            continue
        raise RuntimeError("wrongly typed argument accepted")
    bogus_queue._handle = 0


def capabilities_and_row_limits(s, stack):
    """The capability record must agree with what the operators accept, and the
    one-threadgroup-per-row limit is rejected from the shape (PR_UNSUPPORTED)."""
    caps = p.tensor_operators_capabilities()
    require(caps.operations == frozenset({"matmul", "bias_activation", "batched_matmul", "reduce_rows",
                                          "softmax_rows", "layer_norm", "add"}), f"operations {caps.operations}")
    require(caps.activations == frozenset(p.Activation) and p.Activation.GELU_TANH in caps.activations,
            f"activations {caps.activations}")
    require(caps.reductions == frozenset(p.Reduce), f"reductions {caps.reductions}")
    limit = (2**32 - 1) // 256
    require((caps.max_tensor_elements, caps.max_row_operator_rows, caps.max_layer_norm_columns)
            == (2**31 - 1, limit, 2**24) and limit == 16777215, "capability limits")
    xb, ob = s.buffer(values(16, 77, 3.0)), s.output(16)
    for activation in caps.activations:  # every advertised activation runs on the GPU
        event = p.bias_activation_into(s.queue, s.ops, p.TensorDescriptor(xb, (2, 8)), None,
                                       p.TensorDescriptor(ob, (2, 8)), activation=activation)
        s.completed(event, "paralyn_activation_f32", f"activation {activation!r}")
    over = limit + 1
    big_x = stack.enter_context(s.context.buffer(over * 4))
    big_o = stack.enter_context(s.context.buffer(over * 4))
    gb = s.buffer([1.0])
    g1, g0 = p.TensorDescriptor(gb, (1,)), p.TensorDescriptor(gb, (0,))

    def too_many(function, operation):
        try:
            function()
        except p.Error as error:
            require(error.code == p.Status.UNSUPPORTED and error.operation == operation and "16777215" in str(error),
                    f"{operation}: wrong rejection {error!r}")
            return
        raise RuntimeError(f"{operation} accepted more than {limit} rows")

    for cols in (1, 0):
        x, o = p.TensorDescriptor(big_x, (over, cols)), p.TensorDescriptor(big_o, (over, cols))
        for op in p.Reduce:
            too_many(lambda: p.reduce_rows_into(s.queue, s.ops, x, p.TensorDescriptor(big_o, (over,)), op=op),
                     "reduce_rows_f32")
        too_many(lambda: p.softmax_rows_into(s.queue, s.ops, x, o), "softmax_rows_f32")
        g = g1 if cols else g0
        too_many(lambda: p.layer_norm_into(s.queue, s.ops, x, g, g, o), "layer_norm_f32")
    batched = p.TensorDescriptor(big_x, (2, over // 2, 1)), p.TensorDescriptor(big_o, (2, over // 2, 1))
    too_many(lambda: p.softmax_rows_into(s.queue, s.ops, *batched), "softmax_rows_f32")
    return caps


def high_level(s):
    context = s.context
    with ExitStack() as owned:
        a = owned.enter_context(p.tensor(values(2 * 5 * 7, 900), shape=(2, 5, 7), context=context))
        b = owned.enter_context(p.tensor(values(2 * 7 * 3, 901), shape=(2, 7, 3), context=context))
        x = owned.enter_context(p.tensor(values(10 * 7, 902), shape=(10, 7), context=context))
        g = owned.enter_context(p.tensor(values(7, 903), shape=(7,), context=context))
        bias = owned.enter_context(p.tensor(values(7, 904), shape=(7,), context=context))
        results = [owned.enter_context(r) for r in (
            p.batched_matmul(a, b), p.row_sum(x), p.row_max(x), p.softmax(x, scale=0.5, causal=False),
            p.layer_norm(x, g, bias, epsilon=1e-5), p.gelu(x), p.bias_gelu(x, bias))]
        results.append(owned.enter_context(p.residual_add(x, results[5])))
        for result in results:
            timing = result.timing()
            require(timing is not None and timing.completed, "high-level operator without a GPU event")
        c, sums, maxes, probs, norm, act, fused, total = (r.to_host() for r in results)
        require(results[0].shape == (2, 5, 3) and results[1].shape == (10,), "high-level result shapes")
        xh, gh, bh = x.to_host(), g.to_host(), bias.to_host()
        for i in range(10):
            row = xh[i * 7:(i + 1) * 7]
            require(ref.same_bits(sums[i], ref.tree_sum_f32(row)) and ref.same_bits(maxes[i], ref.tree_max_f32(row)),
                    "high-level reductions")
            sm = ref.softmax_row([ref.exact(v) for v in row], 0.5, 7)
            ln = ref.layer_norm_row([ref.exact(v) for v in row], [ref.exact(v) for v in gh],
                                    [ref.exact(v) for v in bh], ref.f32(1e-5))
            for j in range(7):
                k = i * 7 + j
                require(ref.within(probs[k], sm[j]) and ref.within(norm[k], ln[j]), "high-level softmax/layer_norm")
                require(ref.within(act[k], ref.gelu(ref.exact(row[j]))), "high-level gelu")
                require(ref.within(fused[k], ref.gelu(ref.exact(ref.f32(row[j] + bh[j])))), "high-level bias_gelu")
                require(ref.same_bits(total[k], ref.f32(row[j] + act[k])), "high-level residual_add")
        try:
            p.residual_add(x, a)
        except p.Error as error:
            require(error.code == p.Status.INVALID_ARGUMENT, "residual_add shape mismatch status")
        else:
            raise RuntimeError("residual_add accepted mismatched shapes")
        try:
            p.batched_matmul(a, a)
        except ValueError:
            pass
        else:
            raise RuntimeError("batched_matmul accepted mismatched shapes")
    return len(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    require(not args.artifacts.exists(), "evidence directory must be new")
    with ExitStack() as stack:
        s = Suite(stack)
        batched(s)
        reductions(s)
        softmax(s)
        layer_norm(s)
        gelu_and_add(s)
        negative(s, stack)
        caps = capabilities_and_row_limits(s, stack)
        high = high_level(s)
        s.context.write_evidence(args.artifacts)
    execution = json.loads((args.artifacts / "execution.json").read_text())
    launches = execution["launches"]
    require(execution["backend"] == "Metal" and execution["cpu_fallback"] is False, "wrong execution policy")
    require(len(launches) == s.events + high, f"launch count {len(launches)} != observed {s.events} + {high}")
    for launch in launches:
        require(launch["command_status"] == "completed" and not launch["error"], "failed launch")
    print(f"Transformer operators (Python): {s.events} low-level GPU commands "
          f"({', '.join(f'{k} {v}' for k, v in sorted(s.kernels.items()))}), {high} high-level events")
    print(f"Capabilities: {len(caps.operations)} operations, activations "
          f"{', '.join(a.name for a in sorted(caps.activations))}; row limit {caps.max_row_operator_rows} enforced")
    for op, worst in sorted(s.worst.items()):
        print(f"worst error/bound {op}: {worst:.4f}")
    print("Verification: PASS Python batched strided matmul, row sum/max, softmax, LayerNorm, GELU, residual add "
          "(bitwise FP32 order references and float64 running-error bounds)")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Transformer operator qualification failed: {error}", file=sys.stderr)
        sys.exit(1)
