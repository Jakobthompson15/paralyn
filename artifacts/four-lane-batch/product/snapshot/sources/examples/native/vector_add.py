#!/usr/bin/env python3
"""Run a verified kernel artifact through the native Python/C API on a physical GPU."""
import argparse
from array import array
from contextlib import ExitStack

import paralyn as pr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--artifacts")
    args = parser.parse_args()
    n = 1003
    a = array("f", ((i % 31 - 15) * 0.5 for i in range(n)))
    b = array("f", ((i % 17 - 8) * 0.25 for i in range(n)))
    with ExitStack() as owners:
        context = owners.enter_context(pr.Context())
        module = owners.enter_context(pr.Module.load_file(context, args.module))
        kernel = owners.enter_context(module.kernel("vector_add"))
        queue = owners.enter_context(context.queue())
        da = owners.enter_context(context.buffer(n * 4))
        db = owners.enter_context(context.buffer(n * 4))
        output = owners.enter_context(context.buffer(n * 4))
        av = owners.enter_context(da.view(access=pr.Access.READ))
        bv = owners.enter_context(db.view(access=pr.Access.READ))
        cv = owners.enter_context(output.view(access=pr.Access.WRITE))
        da.write(a.tobytes())
        db.write(b.tobytes())
        output.write(array("f", [-9999] * n).tobytes())
        event = owners.enter_context(queue.launch(
            kernel, [av, bv, cv, pr.i32(n)], grid=((n + 63) // 64, 1, 1), block=(64, 1, 1)))
        timing = event.wait()
        actual = array("f")
        actual.frombytes(output.read())
        if not timing.completed or not 0 < timing.gpu_start_seconds < timing.gpu_end_seconds:
            raise RuntimeError("Missing physical GPU completion/timestamps")
        for index, (x, y, result) in enumerate(zip(a, b, actual)):
            if result != x + y:
                raise RuntimeError(f"CPU reference mismatch at {index}")
        if args.artifacts:
            context.write_evidence(args.artifacts)
        print(f"Device: {context.device.name}; backend: {context.device.backend}")
        print(f"GPU duration: {timing.gpu_duration_seconds * 1e6:.3f} microseconds")
    print(f"Verification: PASS ({n} independently checked values)")


if __name__ == "__main__":
    main()
