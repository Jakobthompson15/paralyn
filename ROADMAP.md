# Paralyn roadmap — adopted mandate v1.1

The [user-adopted mandate](docs/master-mandate-v1.1.md) expands the required portfolio without restarting the repository or changing software version 0.0.1. Stage letters below are sequencing labels, not release versions or completion claims. The [portfolio ledger](docs/portfolio-ledger.json) retains all 17 frontend/input families, all three required clients, separate framework targets, operator providers, and per-backend evidence. Later required tracks are not optional research ideas.

## Stage A — qualified CUDA/Metal foundation

**Gate A passed.** Original working-name evidence is unchanged in `artifacts/gate-a/`, including `unicuda-ir.txt`. Fresh Paralyn evidence from clean revision `c823dfcdc4c3d37d8ed1648b4b0d93825cbdb6b1` is unchanged in `artifacts/paralyn-gate-a/`, including `paralyn-ir.txt`. Actual status and qualification limits are in `docs/status.md`.

The canonical program uses 1,024 elements, distinct allocations, block size 256, and exactly representable FP32 inputs. Never special-case these values or its kernel name. The complete source must traverse CUDA AST extraction, preserved/transformed host C++, verified typed IR, generated MSL, and the physical GPU, followed by independent CPU-reference verification. There is no CPU kernel fallback.

**Gate B passed for the documented CUDA/Metal subset on Apple M5.** Clean revision `8af773c9b8925a27041fcca1ab4cc58c82c56edb` passed a fresh configure/build, repeated Gate A, all six CTest targets, and the full correctness and benchmark captures. [Permanent evidence](artifacts/gate-b/build-validation.json) records 75 correctness GPU launches (66 across positive fixtures plus nine in the expected host-exit-37 case) and 880 benchmark launches. Correctness covers varied lengths through 1,000,003, seeded and changed arithmetic, scalar arguments and dependent launches, supported x/y/z indexing, canaries, same-type aliases, rejected mixed types, integer conversions, explicit failure paths, host behavior, and the [numerical policy](docs/numerics.md). This qualifies the tested current subset, not arbitrary CUDA or the complete frontend portfolio.

The captured benchmark follows the full protocol: 1,024, 65,536, 1,048,576, and 16,777,216 elements; 10 warmups; 100 measured iterations; alternating generated/native order; every sample saved. Report compilation, transfers, GPU command duration, total latency, and peak runtime-owned buffer bytes with exact hardware/toolchain. No fabricated targets or cherry-picked performance claims.

Future evidence also preserves transformed host source, per-launch backend artifacts, real revision/dirty state, device identity, launch dimensions, command status, timestamps, and actual verification output. Do not retrofit the immutable five-file historical captures. Hardware-unavailable results cannot pass a GPU gate. Gate B remains necessary before a qualified release/tag; publication is a separately authorized action, not a side effect of the mandate.

## Stages B–D — native interfaces, another backend, useful operators

**B: Native C/C++ and Python — next ready milestone.** Expose actual device/context ownership, buffers/views, supported module/kernel launch, queue completion and structured errors through a native runtime boundary without CUDA allocation APIs. Add C++ wrappers and bind the same runtime to Python, then the agreed small array/operator surface. `docs/native-api.md` is a design target; the current internal launch/test helpers do not qualify it.

**C: A second hardware backend.** Implement and prove NVIDIA CUDA or AMD HIP/ROCm against available hardware, keeping native-API and CUDA-source evidence separate. The other vendor remains required; lack of hardware is an explicit unavailable cell, not a canceled obligation. Interleave independent ready work when a hardware prerequisite blocks execution, recording the reason.

**D: FP32 matmul provider path.** Define shape/layout/strides/transposition/batching, accumulation precision, tolerance, aliasing, workspace, ordering and errors. Implement a useful operator via a qualified native provider or generated GPU kernel, with visible provider/numerical policy. A matmul operator is not full cuBLAS compatibility. Do not force graph/operator representations through the scalar kernel IR.

## Stages E–H — complete the required frontend and integration portfolio

**E: Source/IR campaign.** Validate extension boundaries with one non-CUDA input, then continue through **HIP C++, Triton, OpenCL C, SYCL C++, OpenMP target offload, Slang, HLSL compute, GLSL compute, WGSL, SPIR-V, MLIR, and Metal source**. HLSL/GLSL and SPIR-V/MLIR are separate deliverables. Use licensed mature compiler bridges where useful, with real resource/reflection, execution, error, and verification integration. MSL is backend-specific unless translation is separately demonstrated. A handwritten-MSL backend smoke does not complete this frontend.

**F: Clients and a first framework workload.** Implement scoped, pinned **Numba CUDA, CuPy, and CUDA Python** integrations separately. Choose and deliver one useful PyTorch, JAX, or ONNX integration using a real extension boundary after the native foundation. Do not conflate custom operators, device backends, graph import, compiler backends, or client compatibility. TensorFlow/MLX/NumPy/Array API/DLPack investigation targets remain distinct; being named in research does not mandate every possible framework surface.

**G: Portable-subset cross-vendor qualification.** Complete both NVIDIA and AMD backends and qualify each designated portable common subset on Metal/Apple, CUDA/NVIDIA, and HIP/ROCm/AMD. Keep frontend × operation × backend × version evidence. Expand CUDA language capabilities, matrix operations, shared memory, barriers, atomics, streams/events, and scoped external projects through separate conformance milestones. A small external CUDA project can enter once the tested subset permits it.

**H: PTX and SASS.** Implement both separate binary tracks after establishing authorized artifacts/tooling, versions or a named NVIDIA ISA generation, ABI subsets, reference hardware, and feasibility. PTX must execute through genuine non-NVIDIA lowering and reference comparison; native NVIDIA loading does not qualify portability. SASS requires actual supported machine-instruction decode/lowering/execution and reference artifacts. An unresolved investigated prerequisite is a visible blocker, never a completion claim or deletion of the track.

Each track requires an explicit input contract, a reference example plus a useful second workload/operation, positive/negative/integration tests, exact dependencies/licenses, numerical/error guarantees, actual GPU evidence, and a concrete next task. Research/scaffolding is incomplete work. A useful checkpoint does not complete the all-frontends project.

## Stage I — distributed systems and specialized accelerators

Defer distributed implementation, dashboards, cloud provisioning, and marketplaces until the preceding foundations justify them. Start with independent jobs or explicitly partitioned tasks. Later graph partitioning needs a supported graph and data-placement contract. Separate control metadata from data transfers; account for locality, bandwidth, latency, bounded queues, budgets, cancellation, version negotiation, failure and safe retry semantics.

Aggregate memory is not one GPU address space. A possible Core ML/ANE or other specialized provider has its own graph/operator contract and public tooling path; it does not imply arbitrary GPU-kernel execution. Do not promise exclusive accelerator placement without evidence or allow hidden CPU execution to satisfy GPU-only qualification.

No new cloud costs, uploads, machine enrollment, or remote service access follow automatically from this roadmap. Preserve applicable user authorization boundaries.

## Ongoing rules

Persistent caching remains deferred until the current path is correct; document/version source dependencies, compiler/IR/runtime, targets, options, numerical policy, and alias layout before implementation. Corrupt/incompatible entries need tests. In-process reuse is sufficient for current work.

Preserve naming-transition compatibility through tested aliases rather than rewriting historical evidence. Document architectural changes with problem, cause, evidence, alternatives, and chosen revision. Keep actual results in status/evidence and resumable obligations in the ledger. Advance to the next ready bounded milestone when authorized and possible; do not describe an invocation's limit as completion of the mandatory portfolio.
