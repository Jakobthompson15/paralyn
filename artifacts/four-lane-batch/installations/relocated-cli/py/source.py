#!/usr/bin/env python3
"""Installed-package FP32 array example; no compiler, module path or checkout required."""
import argparse
import os
from contextlib import ExitStack
import paralyn as p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default=os.environ.get("PARALYN_ARTIFACT_DIR"))
    args = parser.parse_args()
    n = 1003
    host_a = [(i % 31 - 15) * 0.5 for i in range(n)]
    host_b = [(i % 17 - 8) * 0.25 for i in range(n)]
    with ExitStack() as owners:
        context = owners.enter_context(p.Context())
        a = owners.enter_context(p.asarray(host_a, context=context, dtype=p.float32))
        b = owners.enter_context(p.asarray(host_b, context=context, dtype=p.float32))
        summed = owners.enter_context(p.add(a, b))
        result = owners.enter_context(p.affine(summed, b, 0.5))
        actual = result.to_host()
        for index, value in enumerate(actual):
            expected = (host_a[index] + host_b[index]) * 0.5 + host_b[index]
            if value != expected:
                raise RuntimeError(f"CPU reference mismatch at {index}: {value} != {expected}")
        timing = result.wait()
        if timing is None or not timing.completed or not 0 < timing.gpu_start_seconds < timing.gpu_end_seconds:
            raise RuntimeError("Missing physical GPU timing/completion evidence")
        print(f"Device: {result.device.name}; shape={result.shape}; dtype={result.dtype}")
        for owner in (result, summed, a, b):
            owner.close()
        if args.artifacts:
            context.write_evidence(args.artifacts)
    print(f"Verification: PASS native arrays ({n} values, add + affine, independent CPU reference)")


if __name__ == "__main__":
    main()
