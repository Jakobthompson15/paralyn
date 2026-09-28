# Native-product implementation update — 2026-09-28

Starting local/remote main: `f3c6c9955daf5a81d764fd0e634eb786a8489bcd`. New actual code implements FP32 arrays and installable modules, a public validated/reflected MSL path, versioned runtime capability/timing queries, conditional/runtime-only builds and structured terminal commands/reports. See [implementation audit](implementation-audit.md), [artifact profile](executable-artifacts.md), [terminal contract](terminal.md), and [complete required program](complete-platform-program.md). The final clean archive is a separate qualification step; original evidence below is unchanged.

Development hardware tests pass on the physical M5: Python arrays20 events; C++ arrays21 including retained output lifetimes; public MSL10 events/13 negatives; terminal58 command cases including a realGPU doctor reference; installed Python3.9.6 and3.14.5 and relocated C++ examples. Gate B caught the moved runtime-log compatibility transcript and passes after preserving that transcript without mixing application streams. Every required portfolio row remains tracked; no full platform or cross-vendor completion is claimed.

# Status

Updated 2026-09-25. **CUDA/Metal Gates A and B passed on the physical Apple M5.** The Gate B checkpoint was captured from clean revision `8af773c9b8925a27041fcca1ab4cc58c82c56edb` and independently audited. All six tests passed with no skips. **The native C/C++ and Python byte-buffer/launch slice also passed clean qualification** at `76217b7708d7b503449bf42e556e4c148bda88e6`: all 14 current CTests, 21 native GPU events, and repeated CUDA Gates A/B plus the full benchmark. [Native checkpoint evidence](../artifacts/stage-b/README.md) is permanent. No release tag has been created; the broader v1.1 frontend portfolio remains incomplete.

## Gate B — qualified documented CUDA/Metal subset

Evidence: [artifacts/gate-b](../artifacts/gate-b/README.md), including [build/test record](../artifacts/gate-b/build-validation.json), [correctness summary](../artifacts/gate-b/correctness/summary.json), [benchmark audit](../artifacts/gate-b/benchmark/capture.json), and all raw samples/shaders/host code.

- Five positive CUDA fixtures executed 66 GPU launches; the deliberate host-exit-37 case executed nine more and returned 37 without reporting success.
- Coverage includes zero/tiny/boundary/partial/large-odd lengths through 1,000,003, seeded exact FP32 arithmetic, varying scalars/blocks, dependent queued launches, sentinels, all xyz builtins, integer boundaries, same-allocation aliases including const/mutable arguments, and alias pipeline reuse.
- Normal FP32 add/multiply observed max 0 ULP under the specified 1-ULP gate. Signed-zero and NaN/Inf-class checks passed; four subnormal variations were allowed flush-to-zero cases. A discriminating probe verifies disabled contraction. These observations are not a universal cross-vendor IEEE-754 guarantee.
- Unsupported source, malformed IR, invalid allocation/copy/launch arguments, mixed-type aliases, missing/invalid Metal entrypoints, and ignored-error process shutdown have rejection tests. This does not simulate every hardware/driver failure.
- The four-size benchmark completed 880 launches: ten warmups and one hundred measured iterations per variant/size, alternating order. Every output was independently checked. Full samples separate compilation, transfers, GPU duration and total latency. Counted runtime-owned buffers peaked at 201,326,592 bytes and returned to zero. No speedup threshold was used.
- A fresh build directory using already-installed dependencies reached its first verified GPU result in 6.698 seconds on this machine; dependency installation and a fresh OS were not measured.

The native C/C++ byte-buffer/context/module API and Python bindings now exist and pass clean physical-M5 qualification. The next bounded gap is the small contiguous FP32 array/operator surface over that runtime; checkpoint evidence is described below. All 17 required frontend/input families and three clients remain tracked in the [portfolio ledger](portfolio-ledger.json). NVIDIA/AMD execution remains unavailable and unqualified.

The sections below preserve the history of Gate A.

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
| Same-type alias grouping, other index dimensions and arithmetic cases | Gate B now exercises same-type/const aliases and all xyz dimensions | Not implemented |
| FP64, shared memory, barriers, atomics, streams | Unsupported | Not implemented |

The source is written in ordinary CUDA style but has not been compiled/run with `nvcc` here. No NVIDIA hardware/toolkit test is claimed. Source-context macros and certain host preprocessing are explicitly rejected as documented in `cuda-compatibility.md`.

Metal is the only implemented backend. Native C/C++ and Python byte-buffer/launch interfaces and contiguous FP32 array add/affine now exist. CUDA/ROCm backends, broader tensor/operators, distributed execution and persistent caching remain unimplemented. Gate B correctness and its benchmark are qualified only for the documented current Metal subset.

## Native runtime checkpoint

Starting local HEAD and remote main were both `0e7041f046ee5d5ff8f2560233ed5b1ba9c812cd`; no newer implementation existed. The [implementation audit](implementation-audit.md) identifies exact new files, targets and tests rather than treating portfolio metadata as code.

C ABI 1, move-only C++ wrappers and standard-library Python ctypes bindings now use the same real Metal engine as the preserved CUDA path. Versioned verified kernel artifacts are produced by `paralyn compile` without running source host code. Native clients do not link LLVM. Tests independently verify vector-add and affine transforms, offset/canary/alias cases, copied scalars, ordered dependent launches, retained resources, handle/context/range/access/schema failures and error detail. The native C API suite exercises eight GPU events and 35 structured negative checks; Python exercises nine GPU events, including work after releasing its original context handle.

The first byte-buffer/launch slice is implemented; the full native frontend tracks remain incomplete until their further operator and cross-vendor obligations are met. All other 14 input families, three interoperability clients, framework integration, NVIDIA/AMD execution and matmul remain required. Real device-loss/timeouts and allocation-exhaustion fault injection remain unqualified. Permanent evidence is in [artifacts/stage-b](../artifacts/stage-b/README.md). Clean revision `76217b7708d7b503449bf42e556e4c148bda88e6` passed a fresh build, all 14 CTests without skips, 21 native GPU events, repeated CUDA Gate A, 75 Gate B correctness launches and the full 880-launch benchmark. CPython 3.14.5 used the same compiled C ABI library as C/C++; all client outputs were independently checked. Twenty native events have source-linked command records; the final Python ownership event has separately audited actual completion timestamps after its context handle is intentionally released.

The archive retains ordinary C/C++/Python sources, verified modules/IR, all dispatched shaders, command timings/status, logs, exact toolchain/revision and source/library/binary hashes. An independent readback audit matched those counts and hashes. Earlier Gates A/B artifacts were not modified. These results establish the documented Metal subset, not full portfolio completion or hardware-fault injection coverage.
