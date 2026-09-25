# Status

Updated 2026-09-25. **Gate A passed.** No public release is qualified; Gate B was not started.

- CMake/Ninja build: passed with LLVM/Clang 21.1.8 and Apple Clang 17.
- Typed IR verifier/codegen tests: passed.
- CUDA frontend extraction/diagnostic tests: passed.
- Handwritten Metal runtime smoke: passed on physical Apple M5; 1,024 values matched CPU reference.
- `unicuda devices` and `unicuda inspect examples/vector_add.cu`: verified.
- Complete ordinary CUDA source → GPU → CPU comparison: passed on Apple M5, macOS 26.5.1 (25F80).
- Automated full-pipeline test checks process success, actual command completion, positive GPU timestamps, preserved source/IR/MSL, and the independent verifier transcript.
- Final CTest run: all four targets passed (`ir_and_codegen`, `frontend`, `metal_runtime`, `gate_a`), with no skipped tests.

## First successful complete execution

Source revision: `3f3c960f0eb712869cbc99b81e3fdd84394e8efe`, clean checkout at execution. LLVM/Clang 21.1.8; Apple Clang 17 compiled the transformed host program; macOS SDK 26.2; configured deployment target 26.0.

The source used 1,024 elements, grid `(4,1,1)`, and block `(256,1,1)`. Metal reported command status `completed`, no error, and positive GPU timestamps. The source program read the GPU output, compared each element with its own CPU reference, and returned zero. The canonical sizes/kernel name do not appear as special cases anywhere in the compiler/runtime implementation.

The five original files in `../artifacts/gate-a/` are retained unchanged: `source.cu`, `unicuda-ir.txt`, `generated.metal`, `execution.json`, and `verification.txt`. The captured shader comes from verified IR; the runtime's handwritten-MSL test hook is not used by the CUDA path. GPU timestamps establish execution evidence, not a performance benchmark.

## Boundaries of the result

| Capability | Metal evidence | CUDA / ROCm |
|---|---|---|
| Complete ordinary vector-add source | Gate A passed | Backends not implemented |
| Allocation, H2D/D2H copies, sync, cleanup | Exercised by Gate A and runtime smoke | Not implemented |
| FP32 addition and x-axis i32/u32 indexing | Exact CPU-reference match for canonical data | Not implemented |
| Same-type alias grouping, other index dimensions and arithmetic cases | Mechanism exists; broader GPU qualification deferred | Not implemented |
| FP64, shared memory, barriers, atomics, streams | Unsupported | Not implemented |

The source is written in ordinary CUDA style but has not been compiled/run with `nvcc` here. No NVIDIA hardware/toolkit test is claimed. Source-context macros and certain host preprocessing are explicitly rejected as documented in `cuda-compatibility.md`.

Metal is the only implemented backend. CUDA, ROCm, Python, distributed execution, persistent caching, benchmarks, and Gate B qualification are deferred.
