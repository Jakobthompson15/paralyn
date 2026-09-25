# Mandate v1.1 adoption delta

Adopted 2026-09-25. The unmodified source is tracked as [master-mandate-v1.1.md](master-mandate-v1.1.md); its SHA-256 is `15daa7f0a48e4e0fe586a923512b44f8f22542d0d8ad9c61164a9e96c5f44ac0`. This is engineering mandate version 1.1, not a software release change. Paralyn remains 0.0.1.

## Retained foundation and evidence

The repository already has a Clang CUDA AST frontend, immutable typed scalar IR and verifier, native host rewriting, MSL generation, a Metal runtime, explicit checked copies, allocation tokens, alias grouping, ordered submission, diagnostics, and ordinary complete vector-add source. Retain these rather than restart them.

Original UniCUDA-name evidence is preserved in `artifacts/gate-a/`. Renamed Paralyn Gate A passed from clean revision `c823dfcdc4c3d37d8ed1648b4b0d93825cbdb6b1`, on Apple M5/macOS 26.5.1 with LLVM 21.1.8, and is preserved in `artifacts/paralyn-gate-a/`. The baseline at adoption is repository revision `86f406f`. The coordinated baseline reproduction passed all four existing CTest targets; future Gate B results are separate evidence. No Gate B completion is asserted here.

The existing IR/runtime are deliberately limited. Native MSL is only a backend test hook; the native C/C++ allocation/module/view API, Python binding, tensor/operator layer, framework adapters, NVIDIA/AMD backends, and portable input bridges do not exist as qualified products.

## Immediate changes

Preserve the mandate and add concise `AGENTS.md` guidance, the human frontend matrix and machine-readable portfolio ledger, native API design, numerical policy, and execution policy. Replace the stale roadmap chronology with mandate Stages A–I, while preserving Gate B correctness/benchmark obligations. Correct stale rename-pending text using the existing immutable evidence.

Continue Gate B now. Reconcile naming-transition aliases and add transformed host source to future evidence without modifying historical captures. Use the actual numerical policy rather than assuming safe Metal math disables contraction or guarantees every CUDA rounding/subnormal property. Every actual fix/test remains a separate implementation change and result.

## Expanded required scope

All 17 families are mandatory: CUDA C++, native C/C++, native Python, HIP C++, Triton, OpenCL C, SYCL C++, OpenMP target offload, Slang, HLSL compute, GLSL compute, WGSL, SPIR-V, MLIR, Metal source, PTX, and SASS. Numba CUDA, CuPy, and CUDA Python are three additional mandatory client integrations. Keep a first selected PyTorch/JAX/ONNX framework workload separate from those clients and source languages.

The ledger separates required obligation, implementation state, qualification state, explicit backend cells, versions/subsets, proposed useful workloads, dependencies/provenance, evidence, blockers, and next tasks. Null future versions are unresolved requirements to pin, not claims to support any version. A narrowly verified CUDA Gate A record does not mark the full portfolio qualified.

## Architecture changes when required by implementation

Keep kernel representations for kernel work. Introduce tensor/operator representation only with its first real consumer and evaluate established dialects/bridges before inventing a graph format. A mature compiler may provide a validated backend executable without round-tripping through scalar IR, provided arguments, ownership, capabilities, synchronization, errors, and numerics satisfy the shared contract.

Separate frontend adapters, framework adapters, operator providers, and execution backends. Native buffers/views should expose explicit ownership/offsets/access/lifetime without inheriting CUDA token restrictions. First implement the native runtime boundary after existing qualification, then bind that same runtime into Python. Do not prebuild empty registries or claim interfaces because their design is documented.

PTX/SASS retain distinct authorized artifact/ABI/tooling/reference requirements. Hardware absence or an evidenced prerequisite blocker remains incomplete required work; independent ready tracks can proceed. Publication, paid resources, machine enrollment, and remote execution remain separate authorization matters.

The next acceptance gap is complete CUDA/Metal Gate B, including the prescribed benchmark record. The next new interface after that is the smallest native C/C++ runtime slice. No required family is removed at this checkpoint.

## Subsequent checkpoint — CUDA/Metal Gate B qualified

The adoption snapshot above is retained as chronology. On 2026-09-25, clean revision `8af773c9b8925a27041fcca1ab4cc58c82c56edb` passed a fresh configure/build, repeated Gate A, all six CTest targets, 75 correctness GPU launches, and the prescribed 880-launch benchmark on Apple M5/macOS 26.5.1 with LLVM 21.1.8. [Permanent Gate B evidence](../artifacts/gate-b/build-validation.json) records the exact environment, commands, revision, and results; correctness and benchmark reports remain separate. Original Gate A captures are unchanged.

The current next acceptance gap is the native C/C++ runtime slice described in [native-api.md](native-api.md), followed by the same runtime's Python binding. The API remains design only. All 17 frontend/input families, three named clients, the selected first framework workload, and required Metal/CUDA/ROCm qualification remain in the [portfolio ledger](portfolio-ledger.json); this checkpoint does not complete that portfolio.
