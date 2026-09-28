#!/usr/bin/env python3
"""Real-GPU Python qualification of FP32 tensors, matmul, bias/ReLU and the
versioned capability/timing wrappers. References are independent CPU code:
an FP32 sequential reference in the documented accumulation order (bitwise)
and a float64 reference with an a-priori error bound."""
import argparse
from array import array
from contextlib import ExitStack
import json
import math
from pathlib import Path
import struct
import sys
import paralyn as p

U = 2.0 ** -24


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def rejects(function, exception, status=None, operation=None):
    try:
        function()
    except exception as error:
        if status is not None:
            require(error.code == status, f"expected {status!r}, got {error.code!r}: {error}")
        if operation is not None:
            require(error.operation == operation, f"wrong error operation {error.operation}")
        return
    raise RuntimeError("invalid tensor operation unexpectedly succeeded")


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def bits(value):
    return struct.pack("f", value)


def rng(seed):
    state = seed
    while True:
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        yield ((state >> 16) - 32768) / 32768.0


def values(count, seed):
    generator = rng(seed)
    return array("f", (next(generator) for _ in range(count)))


def reference(a, b, m, n, k, ta, tb):
    sequential, exact, bound = [], [], []
    gamma = k * U / (1 - k * U)
    for i in range(m):
        for j in range(n):
            total, precise, magnitude = 0.0, 0.0, 0.0
            for q in range(k):
                x = a[q * m + i] if ta else a[i * k + q]
                y = b[j * k + q] if tb else b[q * n + j]
                total = f32(total + f32(x * y))
                precise += x * y
                magnitude += abs(x * y)
            sequential.append(total)
            exact.append(precise)
            bound.append(gamma * magnitude)
    return sequential, exact, bound


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    require(not args.artifacts.exists(), "evidence directory must be new")
    events = 0
    worst = 0.0
    with p.Context() as context:
        caps = context.capabilities
        require(caps.stable_id.startswith("metal:registry:") and caps.backend == "Metal",
                "Python capability wrapper lacks backend-qualified identity")
        require(caps.artifact_formats == p.ARTIFACT_VERIFIED_IR | p.ARTIFACT_MSL_SOURCE and
                caps.max_buffer_bindings == 31 and all(caps.max_block), "capability record")
        require(p.device_capabilities(0).stable_id == caps.stable_id, "device/context capabilities differ")
        rejects(lambda: p.device_capabilities(1 << 20), p.Error, p.Status.DEVICE_UNAVAILABLE)
        artifact = p.tensor_operators_artifact()
        require(artifact[:8] == b"PARALYNX" and b"paralyn.msl.tensor" in artifact, "provider artifact")

        shapes = [(1, 1, 1), (1, 5, 1), (4, 1, 6), (1, 1, 70), (16, 16, 16), (17, 9, 15),
                  (33, 17, 20), (3, 5, 0), (0, 4, 3), (4, 0, 3)]
        seed = 100
        for m, n, k in shapes:
            for ta in (False, True):
                for tb in (False, True):
                    seed += 1
                    a_host, b_host = values(m * k, seed), values(k * n, seed * 3 + 1)
                    with ExitStack() as owners:
                        a = owners.enter_context(p.tensor(a_host, shape=(k, m) if ta else (m, k), context=context))
                        b = owners.enter_context(p.tensor(b_host, shape=(n, k) if tb else (k, n), context=context))
                        c = owners.enter_context(p.matmul(a, b, transpose_a=ta, transpose_b=tb))
                        a.close()  # the enqueued command retains its inputs
                        label = f"m={m} n={n} k={k} ta={ta} tb={tb}"
                        require(c.shape == (m, n) and c.dtype == p.float32, f"{label}: metadata")
                        timing = c.timing()
                        if m and n:
                            require(timing and timing.completed and timing.duration_seconds > 0 and
                                    timing.clock_domain == p.ClockDomain.METAL_SYSTEM_MACH and
                                    0 < timing.start_seconds < timing.end_seconds, f"{label}: timing")
                            events += 1
                        else:
                            require(timing is None and c.wait() is None, f"{label}: fabricated event")
                        out = c.to_host()
                        seq, exact, bound = reference(a_host, b_host, m, n, k, ta, tb)
                        for index, value in enumerate(out):
                            require(bits(value) == bits(seq[index]), f"{label}: differs from FP32 sequential reference")
                            error = abs(value - exact[index])
                            require(error <= bound[index], f"{label}: float64 bound exceeded")
                            if bound[index]:
                                worst = max(worst, error / bound[index])

        # Bias/ReLU, including exceptional values and activation-only.
        inf, nan = float("inf"), float("nan")
        x_host = [-1.5, 0.0, -0.0, 2.25, nan, -inf, inf, 3.0, -3.0, 0.5, 1e-3, -7.0]
        bias_host = [0.5, -0.0, 0.0, -2.25, 1.0, 0.0]
        with p.tensor(x_host, shape=(2, 6), context=context) as x, \
                p.tensor(bias_host, shape=(6,), context=context) as bias:
            for label, result, with_bias, active in (
                    ("bias", p.bias_add(x, bias), True, False),
                    ("bias_relu", p.bias_add(x, bias, relu=True), True, True),
                    ("relu", p.relu(x), False, True)):
                with result:
                    out = result.to_host()
                    events += 1
                    for index, value in enumerate(out):
                        expected = f32(x_host[index] + bias_host[index % 6]) if with_bias else f32(x_host[index])
                        if active and expected < 0.0:
                            expected = 0.0
                        require(bits(value) == bits(expected) or (math.isnan(value) and math.isnan(expected)),
                                f"{label}: mismatch at {index}")
            rejects(lambda: p.bias_add(x, x), p.Error, p.Status.INVALID_ARGUMENT, "bias_activation_f32")
            rejects(lambda: p.bias_add(x, bias, relu=1), TypeError)

        # Read-only aliasing: G = A A^T binds one allocation twice.
        g_host = values(7 * 9, 7)
        with p.tensor(g_host, shape=(7, 9), context=context) as a, \
                p.matmul(a, a, transpose_b=True) as gram:
            seq, _, _ = reference(g_host, g_host, 7, 7, 9, False, True)
            require([bits(v) for v in gram.to_host()] == [bits(v) for v in seq], "A*A^T alias")
            events += 1

        # High-level validation errors never reach the GPU.
        with p.tensor([1, 2, 3, 4, 5, 6], shape=(2, 3), context=context) as a:
            rejects(lambda: p.matmul(a, a), ValueError)
            rejects(lambda: p.matmul(a, a, transpose_a=1), TypeError)
            with p.tensor([1, 2, 3], shape=(3,), context=context) as vector:
                rejects(lambda: p.matmul(a, vector), ValueError)
            with p.Context() as other, p.tensor([1, 2, 3], shape=(3, 1), context=other) as foreign:
                rejects(lambda: p.matmul(a, foreign), p.Error, p.Status.CONTEXT_MISMATCH)
        rejects(lambda: p.tensor([1, 2, 3], shape=(2, 2), context=context), ValueError)
        rejects(lambda: p.tensor([1], shape=(1,) * 9, context=context), ValueError)
        rejects(lambda: p.tensor([1, 2], shape=(2,), context=context, dtype="float64"), ValueError)
        rejects(lambda: p.tensor(array("d", [1, 2]), shape=(2,), context=context), TypeError)
        rejects(lambda: p.tensor([1], shape=(-1,), context=context), ValueError)

        # Low-level C ABI records: explicit m/n/k, layout and alias rules.
        with ExitStack() as owners:
            ops = owners.enter_context(p.load_tensor_operators(context))
            queue = owners.enter_context(context.queue())
            ab = owners.enter_context(context.buffer(6 * 4))
            bb = owners.enter_context(context.buffer(12 * 4))
            cb = owners.enter_context(context.buffer(8 * 4))
            a_raw, b_raw = values(6, 1), values(12, 2)
            ab.write(a_raw.tobytes())
            bb.write(b_raw.tobytes())
            A = p.TensorDescriptor(ab, (2, 3))
            B = p.TensorDescriptor(bb, (3, 4))
            C = p.TensorDescriptor(cb, (2, 4))
            require(C.required_bytes() == 32, "required bytes")
            with p.matmul_into(queue, ops, A, B, C, m=2, n=4, k=3) as event:
                require(event.timing().completed, "low-level matmul event")
                events += 1
            raw = array("f")
            raw.frombytes(cb.read())
            seq, _, _ = reference(a_raw, b_raw, 2, 4, 3, False, False)
            require([bits(v) for v in raw] == [bits(v) for v in seq], "low-level matmul differs")
            for kwargs, status in (({"m": 3, "n": 4, "k": 3}, p.Status.INVALID_ARGUMENT),
                                   ({"m": 2, "n": 4, "k": 2}, p.Status.INVALID_ARGUMENT)):
                rejects(lambda kwargs=kwargs: p.matmul_into(queue, ops, A, B, C, **kwargs), p.Error, status,
                        "matmul_f32")
            column_major = p.TensorDescriptor(ab, (2, 3), strides=(1, 2))
            require(column_major.required_bytes() == 24, "strided extent")
            rejects(lambda: p.matmul_into(queue, ops, column_major, B, C, m=2, n=4, k=3), p.Error,
                    p.Status.UNSUPPORTED)
            rejects(lambda: p.matmul_into(queue, ops, A, B, p.TensorDescriptor(ab, (2, 4), 0),
                                          m=2, n=4, k=3), p.Error, p.Status.OUT_OF_BOUNDS)
            big = owners.enter_context(context.buffer(16 * 4))
            big.write(values(16, 3).tobytes())
            A_shared = p.TensorDescriptor(big, (2, 3))
            rejects(lambda: p.matmul_into(queue, ops, A_shared, B, p.TensorDescriptor(big, (2, 4), 8),
                                          m=2, n=4, k=3), p.Error, p.Status.INVALID_ARGUMENT)
            rejects(lambda: p.matmul_into(queue, ops, A_shared, B, p.TensorDescriptor(big, (2, 4), 32),
                                          m=2, n=4, k=3), p.Error, p.Status.UNSUPPORTED)
            require(p.matmul_into(queue, ops, p.TensorDescriptor(ab, (0, 3)), B,
                                  p.TensorDescriptor(cb, (0, 4)), m=0, n=4, k=3) is None,
                    "empty low-level matmul fabricated an event")

            # Handles and ownership are validated even when there is no GPU work.
            bogus_queue = p.Queue._adopt(queue._lib, 0xDEADBEEF)  # never released: not a live handle
            try:
                A0, C0 = p.TensorDescriptor(ab, (0, 3)), p.TensorDescriptor(cb, (0, 4))
                for case in ((A, B, C, 2, 4, 3), (A0, B, C0, 0, 4, 3)):  # with work, and m == 0
                    rejects(lambda case=case: p.matmul_into(bogus_queue, ops, *case[:3], m=case[3], n=case[4],
                                                            k=case[5]),
                            p.Error, p.Status.INVALID_HANDLE, "matmul_f32")
                X0, O0 = p.TensorDescriptor(cb, (0, 4)), p.TensorDescriptor(big, (0, 4))
                rejects(lambda: p.bias_activation_into(bogus_queue, ops, X0, None, O0), p.Error,
                        p.Status.INVALID_HANDLE, "bias_activation_f32")
            finally:
                bogus_queue._handle = 0
            with p.Context() as other, other.buffer(8 * 4) as foreign_buffer:
                foreign_c0 = p.TensorDescriptor(foreign_buffer, (0, 4))
                rejects(lambda: p.matmul_into(queue, ops, A0, B, foreign_c0, m=0, n=4, k=3), p.Error,
                        p.Status.CONTEXT_MISMATCH, "matmul_f32")
                rejects(lambda: p.matmul_into(queue, ops, A, B, p.TensorDescriptor(foreign_buffer, (2, 4)),
                                              m=2, n=4, k=3), p.Error, p.Status.CONTEXT_MISMATCH, "matmul_f32")
                rejects(lambda: p.bias_activation_into(queue, ops, X0, None, foreign_c0), p.Error,
                        p.Status.CONTEXT_MISMATCH, "bias_activation_f32")
                foreign_bias = p.TensorDescriptor(foreign_buffer, (4,))
                rejects(lambda: p.bias_activation_into(queue, ops, C, foreign_bias,
                                                       p.TensorDescriptor(big, (2, 4))),
                        p.Error, p.Status.CONTEXT_MISMATCH, "bias_activation_f32")
            # Output aliasing B and bias; dimensions beyond INT32_MAX.
            wide = owners.enter_context(context.buffer(32 * 4))
            wide.write(values(32, 4).tobytes())
            B_shared = p.TensorDescriptor(wide, (3, 4))
            rejects(lambda: p.matmul_into(queue, ops, A, B_shared, p.TensorDescriptor(wide, (2, 4), 16),
                                          m=2, n=4, k=3), p.Error, p.Status.INVALID_ARGUMENT, "matmul_f32")
            rejects(lambda: p.matmul_into(queue, ops, A, B_shared, p.TensorDescriptor(wide, (2, 4), 64),
                                          m=2, n=4, k=3), p.Error, p.Status.UNSUPPORTED, "matmul_f32")
            bias_shared = p.TensorDescriptor(big, (4,))
            rejects(lambda: p.bias_activation_into(queue, ops, C, bias_shared, p.TensorDescriptor(big, (2, 4), 8)),
                    p.Error, p.Status.INVALID_ARGUMENT, "bias_activation_f32")
            rejects(lambda: p.bias_activation_into(queue, ops, C, bias_shared, p.TensorDescriptor(big, (2, 4), 32)),
                    p.Error, p.Status.UNSUPPORTED, "bias_activation_f32")
            rejects(lambda: p.matmul_into(queue, ops, A, B, C, m=1 << 31, n=4, k=3), p.Error,
                    p.Status.UNSUPPORTED, "matmul_f32")
            # Provider identity is the handle from load_tensor_operators, not matching bytes or schema.
            with p.Module.load(context, p.tensor_operators_artifact()) as plain:
                before = cb.read()
                rejects(lambda: p.matmul_into(queue, plain, A, B, C, m=2, n=4, k=3), p.Error,
                        p.Status.INVALID_ARGUMENT, "matmul_f32")
                require(cb.read() == before, "non-provider module wrote the output")
        context.write_evidence(args.artifacts)
        device = context.device

    evidence = json.loads((args.artifacts / "execution.json").read_text())
    require(len(evidence["launches"]) == events, f"launch count {len(evidence['launches'])} != {events}")
    require(evidence["cpu_fallback"] is False and evidence["runtime_owned_current_buffer_bytes"] == 0,
            "tensor cleanup leaked buffers or wrong policy")

    # Retained results survive context close; new work is rejected.
    context = p.Context()
    with p.tensor([1, 2, 3, 4], shape=(2, 2), context=context) as a:
        with p.matmul(a, a) as result:
            context.close()
            require(list(result.to_host()) == [7, 10, 15, 22], "retained matmul after context close")
            rejects(lambda: p.matmul(a, a), p.Error)
    print(f"Tensor Python: {events} GPU commands on {device.name}; worst error/bound {worst:.3f}")
    print("Verification: PASS Python FP32 tensors, matmul, bias/ReLU, capability/timing wrappers")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Tensor qualification failed: {error}", file=sys.stderr)
        sys.exit(1)
