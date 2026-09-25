# Status

Updated 2026-09-25. **Paralyn's Gate A passed on the physical Apple M5.** A fresh build, all four tests, device/IR inspection, and a clean-revision execution passed after the project rename. No public release is qualified; Gate B was not started.

## Original verified implementation

- CMake/Ninja build: passed with LLVM/Clang 21.1.8 and Apple Clang 17.
- Typed IR verifier/codegen tests: passed.
- CUDA frontend extraction/diagnostic tests: passed.
- Handwritten Metal runtime smoke: passed on physical Apple M5; 1,024 values matched CPU reference.
- The original `unicuda devices` and `unicuda inspect examples/vector_add.cu` commands were verified before the rename.
- Complete ordinary CUDA source → GPU → CPU comparison: passed on Apple M5, macOS 26.5.1 (25F80).
- Automated full-pipeline test checks process success, actual command completion, positive GPU timestamps, preserved source/IR/MSL, and the independent verifier transcript.
- Original pre-rename CTest run: all four targets passed (`ir_and_codegen`, `frontend`, `metal_runtime`, `gate_a`), with no skipped tests.

## First successful complete execution — historical evidence

Source revision: `3f3c960f0eb712869cbc99b81e3fdd84394e8efe`, clean checkout at execution. LLVM/Clang 21.1.8; Apple Clang 17 compiled the transformed host program; macOS SDK 26.2; configured deployment target 26.0.

The source used 1,024 elements, grid `(4,1,1)`, and block `(256,1,1)`. Metal reported command status `completed`, no error, and positive GPU timestamps. The source program read the GPU output, compared each element with its own CPU reference, and returned zero. The canonical sizes/kernel name do not appear as special cases anywhere in the compiler/runtime implementation.

The five original files in `../artifacts/gate-a/` are retained unchanged: `source.cu`, `unicuda-ir.txt`, `generated.metal`, `execution.json`, and `verification.txt`. Their original working-name strings and `unicuda_commit` / `unicuda_dirty` metadata are historical evidence, not stale current interfaces. The captured shader comes from verified IR; the runtime's handwritten-MSL test hook is not used by the CUDA path. GPU timestamps establish execution evidence, not a performance benchmark.

## Paralyn rename verification — passed

Current executable and commands use `build/paralyn`; the C++ namespace is `paralyn`. New runs produce `paralyn-ir.txt` and Paralyn-named execution metadata. The automated evidence verifier accepts `--paralyn`.

The fresh build and all four CTest targets passed without skips. `paralyn devices` and `paralyn inspect examples/vector_add.cu` were checked. A subsequent complete GPU run from clean revision `c823dfcdc4c3d37d8ed1648b4b0d93825cbdb6b1` matched all 1,024 CPU-reference values, with completed Metal status, positive GPU timestamps, and zero host exit status. Its five files are preserved in `../artifacts/paralyn-gate-a/`, using the new `paralyn-ir.txt` filename and `paralyn_commit` / `paralyn_dirty` metadata. Original evidence in `../artifacts/gate-a/` remains unchanged.

## Boundaries of the original result

| Capability | Metal evidence | CUDA / ROCm |
|---|---|---|
| Complete ordinary vector-add source | Gate A passed | Backends not implemented |
| Allocation, H2D/D2H copies, sync, cleanup | Exercised by Gate A and runtime smoke | Not implemented |
| FP32 addition and x-axis i32/u32 indexing | Exact CPU-reference match for canonical data | Not implemented |
| Same-type alias grouping, other index dimensions and arithmetic cases | Mechanism exists; broader GPU qualification deferred | Not implemented |
| FP64, shared memory, barriers, atomics, streams | Unsupported | Not implemented |

The source is written in ordinary CUDA style but has not been compiled/run with `nvcc` here. No NVIDIA hardware/toolkit test is claimed. Source-context macros and certain host preprocessing are explicitly rejected as documented in `cuda-compatibility.md`.

Metal is the only implemented backend. CUDA, ROCm, Python, distributed execution, persistent caching, benchmarks, and Gate B qualification are deferred.
