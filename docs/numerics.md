# Numerical policy and qualification state

This document separates the behavior demonstrated by Gate A from the broader Gate B contract being implemented. As of the 2026-09-25 mandate adoption, Gate B is **not complete**. No cross-vendor numerical equivalence or performance claim is made.

## Current implementation and proven subset

The scalar kernel IR represents signed/unsigned 32-bit integers, FP32, and internal comparison predicates. FP64 device arithmetic is rejected; it is never silently narrowed. Kernel arithmetic currently includes addition, multiplication, and less-than. Numeric buffer contents are not generally inspected on the host before execution.

The Metal backend requests MSL 3.1, `MTLMathModeSafe`, and `MTLMathFloatingPointFunctionsPrecise`. The native host compiler is invoked with `-fno-fast-math` and `-ffp-contract=off`. These are different compilation paths: the host contraction flag alone does not establish the GPU contraction policy. The implementation now explicitly emits `#pragma STDC FP_CONTRACT OFF` for generated MSL; a discriminating physical-GPU probe is part of Gate B work. This requests separate operations and does not promise every rounding case matches a CPU bit-for-bit.

The canonical vector-add program uses finite exactly representable quarter/half-step FP32 values and compares every GPU result with an independent host addition using exact equality. The preserved [Paralyn evidence](../artifacts/paralyn-gate-a/execution.json) and [verification](../artifacts/paralyn-gate-a/verification.txt) establish that case on Apple M5, macOS 26.5.1, LLVM 21.1.8. They do not establish arbitrary IEEE-754 behavior, every rounding case, signed-zero preservation, subnormal handling, or equivalence to NVIDIA/AMD hardware.

Apple’s [MSL specification, 2026-06-04, §§1.6.3 and 8.1–8.5](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) permits within-statement contraction by default, allows supported rounding toward zero or ties-to-even, and permits flushing subnormal operands/results. The sign of a flushed zero is not guaranteed. Consequently safe/precise mode alone is insufficient for a blanket CUDA/CPU numerical-equivalence claim.

A development run of `tests/qualification/numerics.cu` on the physical Apple M5 passed with 8,192 values per add/multiply operation, maximum observed normal-case ULP error 0, four allowed subnormal-flush variations, matched NaN/infinity classes, preserved tested nonflush signed-zero cases, and a matching noncontracted probe. This is fixture-level development evidence, not a complete Gate B result, a guarantee of globally ties-to-even rounding, or cross-vendor qualification. The full run must be retained with the final Gate B evidence.

## Gate B contract to qualify

| Case | Acceptance policy | Current qualification boundary |
|---|---|---|
| Canonical finite FP32 addition | Exact per-element equality with the independently computed reference | Verified Gate A only |
| Isolated FP32 addition/multiplication on finite normal operands with finite normal results | At most 1 ULP from the independent FP32 reference; retain inputs, policy, and failing examples | Development fixture passed; complete Gate B pending |
| i32 arithmetic | Exact values where all operations/conversions remain in the defined tested range; no signed-overflow claim | Broader conformance pending |
| u32 arithmetic and index math | Preserve unsigned 32-bit semantics; compare exact valid results | Canonical x-axis path verified; broader cases pending |
| Signed zero | Test sign bits separately from ordinary floating equality and document the operation/rounding case | Development nonflush cases passed; flushing is excluded |
| NaN/infinity | Classify inputs/results and document propagation/comparison expectations; do not apply ULP distance to NaNs | Development classification cases passed; payload preservation unclaimed |
| Subnormal inputs/results | Measure and publish backend behavior explicitly; never infer gradual underflow from “safe” settings | Development fixture admits documented flush alternatives; no gradual-underflow promise |
| Multiply-add/contraction | Report the actual GPU contraction policy and compare the intended operation sequence | Explicit OFF pragma and discriminating development probe passed |
| Repeated execution | Compare results across repeated equivalent inputs on the tested backend | Broader determinism pending; no cross-vendor bitwise claim |

The ULP metric must order finite binary32 values consistently, handle the sign transition explicitly, and keep signed-zero tests separate. Reference calculation must not use the generated device expression as its sole oracle. Record dtype, accumulation type, operation sequence, contraction mode, seed, and relevant exceptional-case policy with results.

Unqualified numerical cases are not automatically rejected by the runtime, because ordinary data buffers can contain those values. The distinction is between a documented guarantee and an observed/unqualified case; do not claim validation that code does not perform.

## Later operation/provider policy

Each operator/provider contract must specify element and accumulation dtype, layout, rounding, FMA/contraction, reduced-precision modes, overflow, special values, subnormals, and determinism. A future FP32 matmul must state accumulation precision and tolerance before comparison; the elementwise 1-ULP policy does not automatically apply to reductions or matrix products. Provider names and precision choices must be visible in evidence.

No general tensor/operator layer, provider, alternative dtype, or alternate rounding mode is implied by this policy. Expanding the contract requires positive/negative tests and actual backend execution. A CPU reference remains an oracle, never a fallback kernel implementation.
