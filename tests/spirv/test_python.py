#!/usr/bin/env python3
"""SPIR-V-derived modules through the native Python binding on the real Metal GPU.

The kernels execute only on the GPU via the shared C ABI library; the Python
code computes an independent reference and compares every output value.
"""
import argparse
from array import array
import sys

import paralyn as pr


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def floats(data):
    result = array("f")
    result.frombytes(data)
    return list(result)


def completed(event):
    info = event.wait()
    require(info.completed and info.gpu_start_seconds > 0 and info.gpu_duration_seconds >= 0,
            "missing completed GPU event")
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-add", required=True)
    parser.add_argument("--reduce-sum", required=True)
    args = parser.parse_args()
    events = compared = rejected = 0
    with pr.Context() as context:
        queue = context.queue()
        add = pr.Module.load_file(context, args.vector_add).kernel("vector_add")
        reduce = pr.Module.load_file(context, args.reduce_sum).kernel("reduce_sum")
        require([(p.name, p.type, p.is_buffer, p.access) for p in add.parameters] == [
            ("a", pr.Type.F32, True, pr.Access.READ), ("b", pr.Type.F32, True, pr.Access.READ),
            ("c", pr.Type.F32, True, pr.Access.WRITE), ("n", pr.Type.U32, False, pr.Access.READ)],
            "vector_add reflection through Python differs")

        for n in (1, 64, 65, 12345):
            a = array("f", [float((i * 13) % 97 - 48) * 0.25 for i in range(n)])
            b = array("f", [float((i * 7) % 89 - 44) * 0.5 for i in range(n)])
            da, db, dc = context.buffer(n * 4), context.buffer(n * 4), context.buffer(n * 4)
            da.write(a.tobytes())
            db.write(b.tobytes())
            dc.write(array("f", [-1.0] * n).tobytes())
            events += completed(queue.launch(
                add, [da.view(access=pr.Access.READ), db.view(access=pr.Access.READ),
                      dc.view(access=pr.Access.WRITE), pr.u32(n)],
                grid=((n + 63) // 64, 1, 1), block=(64, 1, 1)))
            out = floats(dc.read())
            for i in range(n):
                # float32 inputs with at most 2 fractional bits: the sum is exact.
                require(out[i] == a[i] + b[i], f"vector_add mismatch at {i} of {n}")
            compared += n

        for n, groups, scale in ((5000, 4, 0.5), (65536, 32, 1.0)):
            data = array("f", [float((i * 37 + groups) % 17 - 8) for i in range(n)])
            di, dp = context.buffer(n * 4), context.buffer(groups * 4)
            di.write(data.tobytes())
            events += completed(queue.launch(
                reduce, [di.view(access=pr.Access.READ), dp.view(access=pr.Access.WRITE),
                         pr.u32(n), pr.f32(scale)], grid=(groups, 1, 1), block=(64, 1, 1)))
            out = floats(dp.read())
            expected = [0.0] * groups
            for i, v in enumerate(data):
                expected[(i % (64 * groups)) // 64] += v
            require(out == [e * scale for e in expected], f"reduce_sum mismatch for n={n}")
            compared += groups

        try:
            queue.launch(add, [da.view(access=pr.Access.READ)], grid=(1, 1, 1), block=(64, 1, 1))
        except pr.Error as error:
            require(error.code == pr.Status.INVALID_ARGUMENT, f"wrong error: {error}")
            rejected += 1
        try:
            queue.launch(add, [da.view(access=pr.Access.READ), db.view(access=pr.Access.READ),
                               dc.view(access=pr.Access.WRITE), pr.u32(1)],
                         grid=(1, 1, 1), block=(128, 1, 1))
        except pr.Error as error:
            require(error.code == pr.Status.INVALID_ARGUMENT, f"wrong error: {error}")
            rejected += 1
        require(rejected == 2, "invalid launches were accepted")
    print(f"Python SPIR-V: {events} GPU events, {compared} values independently compared, "
          f"{rejected} rejections")
    print("Verification: PASS")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, pr.Error, OSError) as error:
        print(f"Python SPIR-V test failed: {error}", file=sys.stderr)
        sys.exit(1)
