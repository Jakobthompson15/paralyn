# Frontend and execution portfolio

Mandate v1.1 requires **all 17 frontend/input families and all three named interoperability clients** below. They remain required when implementation or hardware qualification is unavailable. This is an obligation/evidence ledger, not a support advertisement. The machine-readable source is [portfolio-ledger.json](portfolio-ledger.json).

Software remains v0.0.1. CUDA/Metal Gates A and B are verified for the documented current subset on Apple M5 at clean revision `8af773c9b8925a27041fcca1ab4cc58c82c56edb`. [Gate B evidence](../artifacts/gate-b/build-validation.json) includes six passing CTest targets, 75 correctness GPU launches, and the complete 880-launch generated/handwritten-Metal benchmark. These measurements do not establish universal portability or performance. The [native checkpoint](../artifacts/stage-b/README.md) at clean revision `76217b7708d7b503449bf42e556e4c148bda88e6` additionally qualifies the C/C++/Python byte-buffer/launch subset with 21 GPU events and 14 passing CTests. Research, a CLI registry entry, a printed IR, and a command wrapper do not complete a track.

The [product-foundation checkpoint](../artifacts/product-foundation/README.md) at clean revision `87978b1f99dab214565c232a406635f4222a1a65` adds qualified contiguous FP32 arrays, a bounded public MSL module profile, terminal/process contracts and macOS arm64 installation. All 23 CTests pass. Its product recorder audits 53 GPU events (50 source-linked plus three retained-output events), including 21 C++ array events, 20 Python array events and ten MSL events; the remaining two are doctor probes. The CLI suite has 60 command cases. Separate captures repeat the native and CUDA gates/benchmark and verify complete native CLI applications and installed execution. None of the full frontend/client profiles is marked complete.

## Required frontend/input families

| Track | Stage | Implementation | Full-track qualification | Proven narrower scope |
|---|---|---|---|---|
| CUDA C++ | A | `implementing` | `unavailable` | Documented CUDA/Metal subset, Gates A and B; other vendors unqualified |
| Native C/C++ | B | `implementing` | `unavailable` | Clean ABI1 byte-buffer and C++ contiguous FP32 add/affine array qualification; broader operators/vendors incomplete |
| Native Python | B | `implementing` | `unavailable` | Clean ctypes ABI1 and contiguous FP32 array qualification; offline installed execution on Python 3.9.6/3.14.5 |
| HIP C++ | E | `not_started` | `unavailable` | None |
| Triton | E | `not_started` | `unavailable` | None |
| OpenCL C | E | `not_started` | `unavailable` | None |
| SYCL C++ | E | `not_started` | `unavailable` | None |
| OpenMP target offload | E | `not_started` | `unavailable` | None |
| Slang | E | `not_started` | `unavailable` | None |
| HLSL compute | E | `not_started` | `unavailable` | None |
| GLSL compute | E | `not_started` | `unavailable` | None |
| WGSL | E | `not_started` | `unavailable` | None |
| SPIR-V inputs | E | `not_started` | `unavailable` | None |
| MLIR inputs | E | `not_started` | `unavailable` | None |
| Metal source | E | `implementing` | `unavailable` | Clean bounded MSL module qualification: vector add/read-only alias, threadgroup reduction and tiled transpose; ten GPU events/13 rejections |
| PTX | H | `not_started` | `unavailable` | None |
| SASS | H | `not_started` | `unavailable` | None |

All rows are `required: true`. Each portable common subset must eventually qualify independently on Metal/Apple, CUDA/NVIDIA, and HIP/ROCm/AMD. Native Metal source is explicitly backend-specific; its non-Metal cells are not silently claimed portable. PTX and SASS are distinct required tracks with separate artifact rights, versions/ISA generations, ABI boundaries, tooling, and NVIDIA reference prerequisites. Their current status is not_started, not a fabricated investigated blocker.

The complete version/subset, candidate reference and useful workloads, dependencies/licenses/provenance, execution path, tests, evidence, backend cells, and next implementation task live in each ledger record. Null versions/surfaces identify decisions still required before implementation; they are not wildcard support. Proposed workloads do not imply an agreed or qualified subset.

## Required clients and separate frameworks

| Client | Implementation | Qualification | Next boundary |
|---|---|---|---|
| Numba CUDA | `not_started` | `unavailable` | Pin upstream version and select/approve a bounded integration surface |
| CuPy | `not_started` | `unavailable` | Pin upstream version and select/approve a bounded integration surface |
| CUDA Python | `not_started` | `unavailable` | Pin upstream version and select/approve a bounded integration surface |

Numba CUDA, CuPy, and CUDA Python are clients, not three new kernel languages. A custom-operation demonstration cannot establish full client replacement. Do not falsify CUDA identity or patch availability checks to manufacture compatibility.

The 2026-09-28 complete-platform program now explicitly requires PyTorch (custom operator, device, compiled graph, inference, backward/training), followed by separate JAX and ONNX profiles. Those are distinct achievements; none is implemented yet. DLPack/real device/stream ownership remains required. TensorFlow, MLX, NumPy and general Array API investigations are not silently promoted to extra mandatory frameworks.

## Backend qualification

| Backend | Implementation | Actual verified evidence | Full portfolio |
|---|---|---|---|
| Metal / Apple | `implemented` | CUDA Gates A/B, native C/C++/Python byte-buffer and FP32 arrays, public MSL workloads and runtime smoke on Apple M5 | `unavailable` |
| CUDA / NVIDIA | `not_started` | None; hardware not observed in this environment | `unavailable` |
| HIP/ROCm / AMD | `not_started` | None; hardware not observed in this environment | `unavailable` |

The public MSL path now uses validated source/resource containers and compiled/reflected pipelines; the older smoke hook remains only a regression test. The public native C ABI now allocates without CUDA APIs and Python binds that shared runtime. Explicit source/build/test files are audited in [implementation-audit.md](implementation-audit.md). Contiguous one-dimensional FP32 array add/affine now exist. Matmul, multidimensional tensors, framework/provider integrations and other vendor backends remain unimplemented.

## Evidence and status rules

`implementation` is `not_started`, `implementing`, `implemented`, or `blocked`. `qualification` is `unavailable`, `failing`, or `verified`. Qualification always has a scope: verified Gate A/B subsets are recorded inside the CUDA/Metal cell while the required cross-vendor track remains unqualified. Unavailable reasons distinguish missing implementation, incomplete qualification, and missing hardware. Failed executed tests are failing; an uninvestigated future prerequisite is not yet an evidenced blocker.

The preserved clean Paralyn Gate A revision is `c823dfcdc4c3d37d8ed1648b4b0d93825cbdb6b1`; the ledger references its original source, IR, shader, actual device/timing/status, and CPU verification. The older UniCUDA-name evidence and renamed evidence remain immutable. New tests and evidence are appended as separate records.

For each claimed frontend × operation × backend × version, require actual input, validated compilation/import, resources and arguments, synchronization/errors, physical execution, and independent results. At minimum vary inputs/dimensions, repeat work, reject unsupported input, and verify a reference example plus a second meaningful operation/useful workload. Binary tracks need real executable reference artifacts.

Next: preserve the clean product-foundation qualification while following the ready tasks in [handoff](handoff.md): vendor hardware/backend prerequisites, a pinned SPIR-V bridge, FP32 tensor/matmul and project/kernel-case terminal work. The full [program](complete-platform-program.md) and ledger retain every remaining obligation.
