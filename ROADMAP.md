# UniCUDA roadmap

This is an ordered plan, not a support matrix. Actual results are in `docs/status.md`. Gate A is the current implementation boundary; do not add Gate B features merely because they are convenient.

## Gate A: first complete source-to-GPU program

1. Research prior art and record the architecture, IR choice, dependencies, and licenses.
2. Build device discovery, checked memory handling, default queue, and error propagation.
3. Execute handwritten Metal through the same runtime boundary used by generated kernels.
4. Parse complete ordinary `vector_add.cu`, implement `inspect`, preserve host C++, rewrite its launch, verify typed IR, and generate MSL.
5. Execute generated MSL on the physical Apple GPU and compare all outputs against an independent CPU reference. There is no CPU kernel fallback.

The canonical demonstration uses 1,024 elements, distinct allocations, block size 256, and varied exactly representable FP32 inputs. These values must not be special-cased in the compiler or runtime. Add minimal automated tests with each enabled feature.

Preserve the first successful run in `artifacts/gate-a/`: `source.cu`, `unicuda-ir.txt`, `generated.metal`, `execution.json`, and `verification.txt`. Record device, OS, UniCUDA revision/dirty state, LLVM version, grid/block dimensions, command completion status, and GPU timestamps. Print a PASS only after actual comparison. Gate A is an internal working milestone, not permission to tag a broadly qualified public release.

## Gate B: qualification before public tagging

After Gate A, test zero, tiny, boundary, partial-block, and large odd lengths, including 1,000,003; changed arithmetic and randomized inputs; varied scalar arguments; repeated launches; supported x/y/z indexing; sentinels/canaries; same-type alias cases and changing alias layouts; invalid source/IR/memory/geometry and backend failures; and host expression/exit semantics.

Publish a numerical policy with FP32 ULP treatment, signed-zero and subnormal limits, exceptional-value behavior, and exact nonoverflowing integer cases. A GPU-unavailable result cannot satisfy a hardware gate.

After correctness, compare generated and handwritten Metal at 1,024, 65,536, 1,048,576, and 16,777,216 elements. Use 10 warmups and 100 measured iterations, alternating order, retaining all samples. Report compile time, copies, GPU command-buffer duration, total latency, and peak runtime-owned buffer bytes with environment metadata. No performance claim precedes that evidence.

Persistent artifact caching is a stretch goal. Its key/invalidation design is documented now; if implemented later, test dependency/option/toolchain changes and corrupt entries. In-process pipeline reuse is sufficient for Gate A.

## Subsequent device capabilities

Proceed in order, with compiler/runtime/backend and applicable conformance tests at each step:

1. Matrix addition, then matrix multiplication.
2. Shared memory, then barriers, then atomics.
3. Streams and events.
4. One useful, small, license-compatible external CUDA project.
5. NVIDIA backend, retaining native CUDA facilities where appropriate.
6. AMD ROCm/HIP backend.
7. Python API and carefully scoped NumPy/PyTorch/JAX/Triton interoperability experiments.

The cross-vendor gate uses the exact same source and conformance inputs on each physical backend. A backend implementation is not supported merely because it compiles. Version numbers follow demonstrated capability, not calendar promises.

## Distributed experiments, deferred

Only after individual device backends are reliable, investigate explicit placement, device capability advertisements, explainable scheduling, health, and separate control/data planes. Account for memory requirements, compatibility, utilization, transfer cost, bandwidth, and locality. Keep data transfers direct where appropriate. Aggregate device-accessible memory across nodes is not one unified GPU memory space.

## Design changes

When an assumption fails, record the problem, root cause, evidence, potential solutions, and selected revision. Revise the architecture rather than hide a mismatch behind example-specific behavior. Stop expansion at the active gate until it passes.
