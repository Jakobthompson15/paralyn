# Numerical policy and qualification state

CUDA/Metal Gate B passed on 2026-09-25 at clean revision `8af773c9b8925a27041fcca1ab4cc58c82c56edb` on Apple M5, macOS 26.5.1, LLVM 21.1.8. This document distinguishes the tested numerical contract from broader unqualified cases. [Permanent evidence](../artifacts/gate-b/correctness/summary.json) does not establish cross-vendor numerical equivalence.

## Current implementation and proven subset

The scalar kernel IR represents signed/unsigned 32-bit integers, FP32, and internal comparison predicates. FP64 device arithmetic is rejected; it is never silently narrowed. Kernel arithmetic currently includes addition, multiplication, and less-than. Numeric buffer contents are not generally inspected on the host before execution.

The Metal backend requests MSL 3.1, `MTLMathModeSafe`, and `MTLMathFloatingPointFunctionsPrecise`. The native host compiler is invoked with `-fno-fast-math` and `-ffp-contract=off`. These are different compilation paths: the host contraction flag alone does not establish the GPU contraction policy. The implementation now explicitly emits `#pragma STDC FP_CONTRACT OFF` for generated MSL; a discriminating physical-GPU probe passed in the permanent Gate B capture. This requests separate operations and does not promise every rounding case matches a CPU bit-for-bit.

The canonical vector-add program uses finite exactly representable quarter/half-step FP32 values and compares every GPU result with an independent host addition using exact equality. The preserved [Paralyn evidence](../artifacts/paralyn-gate-a/execution.json) and [verification](../artifacts/paralyn-gate-a/verification.txt) establish that case on Apple M5, macOS 26.5.1, LLVM 21.1.8. They do not establish arbitrary IEEE-754 behavior, every rounding case, signed-zero preservation, subnormal handling, or equivalence to NVIDIA/AMD hardware.

Apple’s [MSL specification, 2026-06-04, §§1.6.3 and 8.1–8.5](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) permits within-statement contraction by default, allows supported rounding toward zero or ties-to-even, and permits flushing subnormal operands/results. The sign of a flushed zero is not guaranteed. Consequently safe/precise mode alone is insufficient for a blanket CUDA/CPU numerical-equivalence claim.

The [retained numerical fixture](../artifacts/gate-b/correctness/numerics/verification.txt) passed with 8,192 values per add/multiply operation, maximum observed normal-case ULP error 0, four allowed subnormal-flush variations, matched NaN/infinity classes, preserved tested nonflush signed-zero cases, and a matching noncontracted probe. Its [input](../artifacts/gate-b/correctness/numerics/source.cu), [generated source](../artifacts/gate-b/correctness/numerics/source-0.metal), and [execution record](../artifacts/gate-b/correctness/numerics/execution.json) accompany the result. This establishes those sampled cases and the specified policy, not globally ties-to-even rounding or cross-vendor qualification.

The [integer fixture](../artifacts/gate-b/correctness/integer_semantics/verification.txt) passed five launches with 12 boundary casts and 50 compared values, including signed/unsigned boundary conversion, nonoverflowing signed affine arithmetic, a negative signed bound, and mixed signed/unsigned comparison. The [three-dimensional fixture](../artifacts/gate-b/correctness/xyz_builtins/verification.txt) compared 10,824 outputs across three launches. These are finite conformance cases, not exhaustive integer or indexing proofs.

## Qualified Gate B contract and limits

| Case | Acceptance policy | Current qualification boundary |
|---|---|---|
| Canonical finite FP32 addition | Exact per-element equality with the independently computed reference | Verified in Gate A and repeated in the Gate B capture |
| Isolated FP32 addition/multiplication on finite normal operands with finite normal results | At most 1 ULP from the independent FP32 reference; retain inputs, policy, and failing examples | Gate B fixture passed; observed maximum normal-case error 0 ULP |
| i32 arithmetic | Exact values where all operations/conversions remain in the defined tested range; no signed-overflow claim | Tested boundary conversions and nonoverflowing affine arithmetic passed; signed overflow remains outside the contract |
| u32 arithmetic and index math | Preserve unsigned 32-bit semantics; compare exact valid results | Tested u32 conversions, mixed comparison, and x/y/z builtin geometry passed |
| Signed zero | Test sign bits separately from ordinary floating equality and document the operation/rounding case | Gate B nonflush cases passed; flushing is excluded |
| NaN/infinity | Classify inputs/results and document propagation/comparison expectations; do not apply ULP distance to NaNs | Gate B classification cases passed; payload preservation unclaimed |
| Subnormal inputs/results | Measure and publish backend behavior explicitly; never infer gradual underflow from “safe” settings | Gate B observed four permitted flush variations; no gradual-underflow promise |
| Multiply-add/contraction | Report the actual GPU contraction policy and compare the intended operation sequence | Explicit OFF pragma and discriminating Gate B probe passed |
| Repeated execution | Compare results across repeated equivalent inputs on the tested backend | Repeated benchmark launches all matched the exact CPU reference; no universal determinism or cross-vendor bitwise claim |

The ULP metric must order finite binary32 values consistently, handle the sign transition explicitly, and keep signed-zero tests separate. Reference calculation must not use the generated device expression as its sole oracle. Record dtype, accumulation type, operation sequence, contraction mode, seed, and relevant exceptional-case policy with results.

Unqualified numerical cases are not automatically rejected by the runtime, because ordinary data buffers can contain those values. The distinction is between a documented guarantee and an observed/unqualified case; do not claim validation that code does not perform.

## Later operation/provider policy

Each operator/provider contract must specify element and accumulation dtype, layout, rounding, FMA/contraction, reduced-precision modes, overflow, special values, subnormals, and determinism. A future FP32 matmul must state accumulation precision and tolerance before comparison; the elementwise 1-ULP policy does not automatically apply to reductions or matrix products. Provider names and precision choices must be visible in evidence.

No general tensor/operator layer, provider, alternative dtype, or alternate rounding mode is implied by this policy. Expanding the contract requires positive/negative tests and actual backend execution. A CPU reference remains an oracle, never a fallback kernel implementation.

## Native ABI 1

Native C/C++ and Python modules use the same IR verifier, code generator and safe/precise, contraction-off Metal engine. Artifact numerical-policy identifier 1 rejects incompatible policies on load. Native launch scalars are explicitly i32/u32/f32; Python f32 converts explicitly to binary32 and rejects finite overflow instead of silently producing infinity. Native vector-add/affine qualification uses varied exactly representable inputs and independent per-element CPU comparisons with canary checks. This adds native-interface evidence; it does not expand the Gate B numerical guarantee or qualify reductions/matmul. Native C/C++ test targets disable host fast math and contraction.
