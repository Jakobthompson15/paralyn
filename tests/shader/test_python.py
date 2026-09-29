#!/usr/bin/env python3
"""GLSL- and HLSL-derived modules through the native Python binding on the real GPU.

The kernels execute only on the GPU via the shared C ABI library; this script
computes independent references and compares every output value.
"""
import argparse
from array import array
import struct
import sys

import paralyn as pr


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def floats(data):
    result = array("f")
    result.frombytes(data)
    return list(result)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def completed(event):
    info = event.wait()
    require(info.completed and info.gpu_start_seconds > 0 and info.gpu_duration_seconds >= 0,
            "missing completed GPU event")
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glsl-blur-rows", required=True)
    parser.add_argument("--glsl-blur-columns", required=True)
    parser.add_argument("--hlsl-vector-add", required=True)
    args = parser.parse_args()
    events = compared = rejected = 0
    with pr.Context() as context:
        queue = context.queue()
        rows = pr.Module.load_file(context, args.glsl_blur_rows).kernel("blur_rows")
        columns = pr.Module.load_file(context, args.glsl_blur_columns).kernel("blur_columns")
        add = pr.Module.load_file(context, args.hlsl_vector_add).kernel("vector_add")
        require([(p.name, p.type, p.is_buffer, p.access) for p in rows.parameters] == [
            ("src", pr.Type.F32, True, pr.Access.READ), ("weights", pr.Type.F32, True, pr.Access.READ),
            ("dst", pr.Type.F32, True, pr.Access.WRITE), ("width", pr.Type.U32, False, pr.Access.READ),
            ("height", pr.Type.U32, False, pr.Access.READ), ("radius", pr.Type.I32, False, pr.Access.READ)],
            "blur_rows reflection through Python differs")

        # Two-dispatch separable blur, both passes queued before waiting.
        width, height, radius = 97, 61, 2
        weights = [f32(v / 16.0) for v in (1.0, 4.0, 6.0, 4.0, 1.0)]
        image = [f32(((i * 53 + 7) % 256) / 256.0) for i in range(width * height)]
        n = width * height
        d_image, d_tmp, d_out, d_w = (context.buffer(n * 4), context.buffer(n * 4),
                                      context.buffer(n * 4), context.buffer(len(weights) * 4))
        d_image.write(array("f", image).tobytes())
        d_w.write(array("f", weights).tobytes())
        grid = ((width + 15) // 16, (height + 15) // 16, 1)
        scalars = [pr.u32(width), pr.u32(height), pr.i32(radius)]
        first = queue.launch(rows, [d_image.view(access=pr.Access.READ), d_w.view(access=pr.Access.READ),
                                    d_tmp.view(access=pr.Access.WRITE), *scalars], grid=grid, block=(16, 16, 1))
        second = queue.launch(columns, [d_tmp.view(access=pr.Access.READ), d_w.view(access=pr.Access.READ),
                                        d_out.view(access=pr.Access.WRITE), *scalars], grid=grid, block=(16, 16, 1))
        events += completed(second) + completed(first)

        def blur(src, horizontal):
            out = []
            for y in range(height):
                for x in range(width):
                    total = 0.0
                    for k in range(-radius, radius + 1):
                        sx = min(max(x + k, 0), width - 1) if horizontal else x
                        sy = y if horizontal else min(max(y + k, 0), height - 1)
                        total = f32(total + f32(weights[k + radius] * src[sy * width + sx]))
                    out.append(total)
            return out

        expected = blur(blur(image, True), False)
        require(floats(d_out.read()) == expected, "two-pass blur differs from the independent reference")
        compared += n

        for count in (1, 65, 10007):
            a = array("f", [float((i * 13) % 97 - 48) * 0.25 for i in range(count)])
            b = array("f", [float((i * 7) % 89 - 44) * 0.5 for i in range(count)])
            da, db, dc = context.buffer(count * 4), context.buffer(count * 4), context.buffer(count * 4)
            da.write(a.tobytes())
            db.write(b.tobytes())
            dc.write(array("f", [-1.0] * count).tobytes())
            events += completed(queue.launch(
                add, [da.view(access=pr.Access.READ), db.view(access=pr.Access.READ),
                      dc.view(access=pr.Access.WRITE), pr.u32(count)],
                grid=((count + 63) // 64, 1, 1), block=(64, 1, 1)))
            out = floats(dc.read())
            require(all(out[i] == a[i] + b[i] for i in range(count)), f"HLSL vector_add mismatch (n={count})")
            compared += count

        try:
            queue.launch(rows, [d_image.view(access=pr.Access.READ), d_w.view(access=pr.Access.READ),
                                d_tmp.view(access=pr.Access.WRITE), *scalars], grid=grid, block=(8, 8, 1))
        except pr.Error as error:
            require(error.code == pr.Status.INVALID_ARGUMENT, f"wrong error: {error}")
            rejected += 1
        require(rejected == 1, "a launch with the wrong workgroup was accepted")
    print(f"Python GLSL/HLSL: {events} GPU events, {compared} values independently compared, "
          f"{rejected} rejection")
    print("Verification: PASS")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, pr.Error, OSError) as error:
        print(f"Python GLSL/HLSL test failed: {error}", file=sys.stderr)
        sys.exit(1)
