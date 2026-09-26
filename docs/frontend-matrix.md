# Frontend and execution portfolio

Mandate v1.1 requires **all 17 frontend/input families and all three named interoperability clients** below. They remain required when implementation or hardware qualification is unavailable. This is an obligation/evidence ledger, not a support advertisement. The machine-readable source is [portfolio-ledger.json](portfolio-ledger.json).

Software remains v0.0.1. CUDA/Metal Gates A and B are verified for the documented current subset on Apple M5 at clean revision `8af773c9b8925a27041fcca1ab4cc58c82c56edb`. [Gate B evidence](../artifacts/gate-b/build-validation.json) includes six passing CTest targets, 75 correctness GPU launches, and the complete 880-launch generated/handwritten-Metal benchmark. These measurements do not establish universal portability or performance. The [native checkpoint](../artifacts/stage-b/README.md) at clean revision `76217b7708d7b503449bf42e556e4c148bda88e6` additionally qualifies the C/C++/Python byte-buffer/launch subset with 21 GPU events and 14 passing CTests. Research, a CLI registry entry, a printed IR, and a command wrapper do not complete a track.

## Required frontend/input families

| Track | Stage | Implementation | Full-track qualification | Proven narrower scope |
|---|---|---|---|---|
| CUDA C++ | A | `implementing` | `unavailable` | Documented CUDA/Metal subset, Gates A and B; other vendors unqualified |
| Native C/C++ | B | `implementing` | `unavailable` | ABI 1 byte buffers/views/modules/launch; real Metal vector-add/affine tests |
| Native Python | B | `implementing` | `unavailable` | ctypes binding to ABI 1; real Metal vector-add/affine tests |
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
| Metal source | E | `not_started` | `unavailable` | None; internal MSL smoke is not a frontend |
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

A separate Stage F requirement selects **one first useful PyTorch, JAX, or ONNX integration** after the native foundation. PyTorch custom operators, a device backend, exported graphs, and torch.compile are separate surfaces. JAX requires a real runtime/compiler boundary, not merely StableHLO import; ONNX needs a tested importer/provider and explicit fallback accounting. The ledger lists these candidates independently, plus TensorFlow, MLX, NumPy, Array API, and DLPack investigation targets. It does not incorrectly turn every framework surface into a mandatory implementation. None is implemented or qualified.

## Backend qualification

| Backend | Implementation | Actual verified evidence | Full portfolio |
|---|---|---|---|
| Metal / Apple | `implemented` | CUDA Gates A/B, native C/C++/Python and runtime smoke on Apple M5 | `unavailable` |
| CUDA / NVIDIA | `not_started` | None; hardware not observed in this environment | `unavailable` |
| HIP/ROCm / AMD | `not_started` | None; hardware not observed in this environment | `unavailable` |

The native MSL smoke hook exercises backend submission, but lacks the public module/import contract required for the Metal-source frontend. The public native C ABI now allocates without CUDA APIs and Python binds that shared runtime. Explicit source/build/test files are audited in [implementation-audit.md](implementation-audit.md). There are still no tensor/operator providers or array APIs.

## Evidence and status rules

`implementation` is `not_started`, `implementing`, `implemented`, or `blocked`. `qualification` is `unavailable`, `failing`, or `verified`. Qualification always has a scope: verified Gate A/B subsets are recorded inside the CUDA/Metal cell while the required cross-vendor track remains unqualified. Unavailable reasons distinguish missing implementation, incomplete qualification, and missing hardware. Failed executed tests are failing; an uninvestigated future prerequisite is not yet an evidenced blocker.

The preserved clean Paralyn Gate A revision is `c823dfcdc4c3d37d8ed1648b4b0d93825cbdb6b1`; the ledger references its original source, IR, shader, actual device/timing/status, and CPU verification. The older UniCUDA-name evidence and renamed evidence remain immutable. New tests and evidence are appended as separate records.

For each claimed frontend × operation × backend × version, require actual input, validated compilation/import, resources and arguments, synchronization/errors, physical execution, and independent results. At minimum vary inputs/dimensions, repeat work, reject unsupported input, and verify a reference example plus a second meaningful operation/useful workload. Binary tracks need real executable reference artifacts.

Next: implement the small contiguous FP32 array/operator surface over the now-working native runtime, while establishing authorized hardware prerequisites for Stage C. The new byte-buffer interfaces do not remove any later obligation. The [roadmap](../ROADMAP.md) preserves every later required track.
