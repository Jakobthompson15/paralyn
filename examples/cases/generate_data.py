#!/usr/bin/env python3
"""Regenerate or check the shipped kernel-case data and reference files.

Inputs are deterministic, exactly representable FP32 values. References are
computed here in Python, independently of Paralyn and of the GPU kernels, with
explicit float32 rounding after each operation (contraction off). The case
files pin every file by SHA256, so any regeneration difference is detected.

    python3 examples/cases/generate_data.py          # write missing files, print hashes
    python3 examples/cases/generate_data.py --check  # verify bytes reproduce exactly
"""
import argparse
import hashlib
from pathlib import Path
import struct
import sys

DATA = Path(__file__).resolve().parent / "data"


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def encode(values):
    return b"".join(struct.pack("<f", f32(v)) for v in values)


def series(n, seed, scale):
    return [f32((float((i * 17 + seed) % 127) - 63.0) * scale) for i in range(n)]


def files():
    out = {}
    n = 1003
    a, b = series(n, 3, 0.25), series(n, 29, 0.5)
    out["vector_add_a.f32"] = a
    out["vector_add_b.f32"] = b
    out["vector_add_expected.f32"] = [f32(x + y) for x, y in zip(a, b)]
    # Affine through the CUDA-derived verified IR module: f32(f32(a*scale) + b).
    scale = 0.5
    out["affine_expected.f32"] = [f32(f32(x * scale) + y) for x, y in zip(a, b)]
    out["reduce_input.f32"] = series(n, 7, 1.0)  # small integers: any summation order is exact
    width, height = 33, 19
    t = series(width * height, 11, 1.0)
    out["transpose_input.f32"] = t
    out["transpose_expected.f32"] = [t[y * width + x] for x in range(width) for y in range(height)]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    failures = 0
    for name, values in files().items():
        payload = encode(values)
        path = DATA / name
        if args.check:
            if not path.is_file() or path.read_bytes() != payload:
                print(f"MISMATCH {name}", file=sys.stderr)
                failures += 1
        elif not path.exists():
            path.write_bytes(payload)
        elif path.read_bytes() != payload:
            print(f"refusing to overwrite differing {path}", file=sys.stderr)
            failures += 1
        print(f"{hashlib.sha256(payload).hexdigest()}  {len(values):6d}  {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
