# Architecture: CUDA, native runtime and public executable modules

Updated 2026-09-28. The native-product extension adds versioned capability/timing queries, public MSL compiled modules, FP32 arrays, conditional/runtime-only builds, packaging and structured terminal reports. See [native API](native-api.md), [executable artifacts](executable-artifacts.md), and [terminal contract](terminal.md) for the current additions. The foundational CUDA/IR design below remains in use. This describes the current source interfaces and their limits. Code being present is not hardware acceptance: actual test results belong in `status.md`. The first accepted execution under the original working name UniCUDA is preserved unchanged in `../artifacts/gate-a/`; the separately verified renamed run from clean revision `c823dfcdc4c3d37d8ed1648b4b0d93825cbdb6b1` is preserved in `../artifacts/paralyn-gate-a/`. The adopted [mandate v1.1](master-mandate-v1.1.md) expands future architecture; its [delta](mandate-v1.1-delta.md) and [portfolio](frontend-matrix.md) preserve all required tracks without asserting they exist.

## Goal and implementation boundary

A complete ordinary `examples/vector_add.cu`, including `main`, memory calls, and a triple-chevron launch, passes through Clang AST extraction, native host-code rewriting, verified typed IR, generated MSL, and the physical Apple GPU. The host program independently compares the output to a CPU reference. There is no CPU kernel fallback.

The canonical demonstration uses 1,024 elements, distinct allocations, block size 256, and varied exactly representable FP32 inputs. These values must not be special-cased anywhere in the compiler or runtime.

[CuMetal](https://github.com/Lulzx/cuda-metal) directly overlaps this goal. [IREE](https://iree.dev/guides/deployment-configurations/) already demonstrates portable compiler/runtime boundaries. Paralyn is independently implemented, imports neither CuMetal code nor MetaXuda dependencies, and makes no novelty claim. See `prior-art.md` and `novelty.md`.

```text
complete CUDA source
  |-- Clang host AST --> narrowly rewritten native C++ --> compatibility runtime
  `-- Clang kernel AST --> typed structured IR --> verifier --> Metal codegen
                                                                  |
                                                     runtime MSL compilation
                                                                  |
                                                         physical Apple GPU
```

There is one available GPU implementation, Metal, which is now conditionally built. A no-backend build enumerates zero devices and fails execution; it is not CPU emulation. `include/paralyn/detail/backend.hpp` now defines the internal context/buffer/event boundary consumed by both the CUDA adapter and native C ABI. `backends/metal/engine.mm` implements it; `backends/metal/runtime.mm` retains the CUDA token adapter. Native offset views are implemented. There is no backend registry, plugin ABI, or negotiated multi-backend protocol; adding an abstract interface does not implement NVIDIA/AMD execution.

## Toolchain and commands

The compiler/core use C++17. Objective-C++ and public Metal/Foundation frameworks are confined to Metal integration. The frontend links Clang LibTooling; the generated host executable links the runtime and IR archives without LLVM libraries. A compiler-enabled CLI links the frontend and needs the selected LLVM installation. A runtime-only CLI and native library build without LLVM.

The selected Homebrew LLVM/Clang is **21.1.8**, with matching headers/libraries; integration tests passed on this installation. Other versions require their own build/parser verification. CMake enables C, C++, and Objective-C++ because LLVM's dependency checks need C enabled. CMake/CTest and Ninja handle builds/tests.

The current build deployment target is **macOS 26.0**, tested on macOS 26.5.1 with SDK 26.2. The installed `libclang-cpp.dylib` declares minimum OS 26.0. Safe/precise Metal compilation APIs are available from macOS 15, but that does **not** make this CLI build compatible with macOS 15. The generated host compiler receives the configured deployment target. See `engineering-notes.md` and `../THIRD_PARTY_LICENSES.md`.

```text
paralyn devices
paralyn inspect program.cu
paralyn compile program.cu --output program.prk
paralyn run program.cu [--device auto|INDEX] [--artifacts DIR] [-- program arguments]
```

`devices` enumerates actual Metal devices in stable registry-ID order and reports unified-memory status, maximum buffer bytes, and recommended working-set bytes. It does not report that recommendation as free GPU memory. `auto` selects the first enumerated device; there is no memory-based scheduler.

`inspect` shares parsing, extraction, and verification with `run`. It prints kernel signatures, launch source expressions and line numbers, the current subset's required capabilities, and deterministic verified IR. It neither evaluates host code nor constant-folds displayed grid/block dimensions. Thus `grid=grid` is an expression, not a claim to know a runtime launch size.

`run` compiles rewritten native host C++, executes it with arguments after `--`, and returns the child status, including compilation failures. It launches tools with an argument vector, not through a shell command assembled from source paths.

## Frontend and host-preservation contract

Clang uses CUDA host-only parsing, independent compatibility declarations, no NVIDIA toolkit headers/libraries, and a forced parse header before standard headers. AST visitors extract kernels and launch sites. Range edits replace the supported kernel definitions with immutable IR factories and replace launches with native C++ lambdas. Ordinary host code is otherwise retained as source and compiled by the native host compiler; Paralyn is not a whole-C++ transpiler.

The supported input boundary is a single translation unit with plain top-level, non-template, non-overloaded kernel definitions in the main file. Separate declarations, external kernels, device functions, macro-generated launches, and ambiguous ranges are rejected. Device code rejects features outside the implemented subset, including FP64, volatile types, general pointer expressions, shared/device global storage, loops, returns, else branches, atomics, barriers, and device calls.

Launch rewrites evaluate grid, block, shared-memory size, and stream before arguments, with each expression evaluated once. Typed argument temporaries preserve parameter conversions; arguments are evaluated in source order, one permitted ordering for otherwise indeterminately sequenced ordinary arguments. Nonzero shared memory and nonnull streams are unsupported; constants can be rejected at compile time and dynamic values are checked before argument evaluation/submission.

Preserving ordinary source text alone is insufficient for every C++ program: inserting generated code changes source locations and function contexts, and CUDA-mode preprocessing differs from native C++. Gate A conservatively rejects CUDA-conditioned user preprocessing (`__CUDA_ARCH__`, `__CUDACC__`, `__CUDA__`) and source-location/context expressions such as `__LINE__`, `__FILE__`, and `__func__`. The guard also checks user headers and inactive branches. This is an explicit compatibility restriction, not a promise of transparent preprocessing/reflection support. Future line mapping/context-preserving rewrites can restore selected cases with tests.

Frontend diagnostics retain original source locations. Expressions carry line/column metadata into IR. Native diagnostics arise from the generated host file; comprehensive original-source diagnostic mapping is future work.

## Typed IR and lowering

The implemented portable model contains:

- `ScalarType`: `I32`, `U32`, `F32`, and `Bool`; bool is currently a comparison predicate, not a kernel parameter or local variable type.
- `Expr`: kind, result type, text, operands, and source line/column. Kinds are literal, reference, CUDA builtin, binary operation, cast, and buffer load.
- `Statement`: local declaration, indexed buffer store, or `if` body.
- `Parameter`: name, scalar/pointee type, buffer flag, and read-only flag.
- `Kernel`: name, parameters, and structured body.

The actual operators are addition, multiplication, and less-than. Casts needed for CUDA index arithmetic preserve explicit signed/unsigned 32-bit conversion. Builtins represent x/y/z of `threadIdx`, `blockIdx`, `blockDim`, and `gridDim`; all twelve coordinate/dimension builtin components were checked against a CPU reference across three 3D launch shapes in Gate B. Indexing requires a buffer parameter and an integer index.

`verify` checks types, arity, declarations/references, literal validity, operators, writable buffers, and structured conditions. `dump_ir` is deterministic. `emit_cpp` serializes the verified model as immutable C++ objects embedded in the host program. `emit_msl` takes that model and a `BindingLayout`; it verifies the IR before emitting Metal. There is no text-IR parser at runtime. The separate public MSL source container has its own validated descriptors and retained compiled-pipeline path.

Custom structured IR keeps the initial MSL backend small and CUDA-level indexing visible. LLVM IR offers mature optimization but introduces lower-level target details; MLIR offers reusable dialects and progressive lowering but has a larger integration cost; SPIR-V is a device representation, not CUDA host semantics. Reconsider a hybrid/MLIR path when real optimization or multiple backend needs justify it. The present small IR does not claim those systems' breadth.

## Runtime, pointers, and GPU execution

The portable runtime header exposes `Dim3`, typed `Argument` values, selection/enumeration, launch, synchronization, and shutdown. A Metal-private context owns the device, one queue, allocation registry, pipeline map, and pending commands. Core kernel/argument types contain no Objective-C objects. A test-only entrypoint submits independently written MSL through the same backend execution path.

`cudaMalloc` returns a compatibility-owned allocation token. It is not a Metal mapping or GPU address. Each token object is retained for the process context's lifetime, even after `cudaFree` removes its live allocation, preventing stale token identities from being reused. Tokens support storage, copying, passing, and null checks only. Host dereference, arithmetic, interior pointers, ordering, subtraction, and integer conversion are unsupported. Native allocation-plus-offset views are implemented through a separate API; CUDA base-token semantics remain unchanged.

The compatibility API implements the needed `dim3`, allocation/free, explicit H2D/D2H copy, synchronization, last-error, and error-string operations. It validates live base tokens, allocation size, copy byte count, nonnull nonempty copy endpoints, and mistaken device tokens passed as host pointers. Shared Metal buffers hold data, and checked copies access their mappings privately. Host memory validity and arbitrary kernel bounds are not proved by these API checks.

At launch, the runtime groups same-allocation arguments. `BindingLayout` assigns one Metal slot per unique allocation or scalar. The MSL emitter derives each same-type pointer parameter from that shared slot, preserving const qualification; mixed-pointee-type alias groups are rejected. This avoids relying on independent entrypoint buffer arguments for overlapping memory. See the [MSL specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf). Gate B checks four layouts, layout reuse and const/mutable aliases on the physical Apple M5; native tests additionally cover same-type offset views. Mixed-pointee alias groups remain unsupported; CUDA host interior pointers remain unsupported.

MSL compilation requests language version 3.1, safe math, and precise floating-point functions through public runtime APIs; generated MSL explicitly disables contraction with `#pragma STDC FP_CONTRACT OFF`. See `numerics.md` for the actual qualification limits. Shader/pipeline errors retain Metal details. Pipelines are reused by entrypoint plus emitted MSL source in a map scoped to one device/context with fixed compilation options. Source includes the alias layout. The entrypoint is part of identity so a multi-entrypoint library cannot reuse the wrong pipeline. There is no persistent cache.

The semantic requirement is CUDA-equivalent logical grid execution, including indices and dimensions. The current backend uses `dispatchThreadgroups` with the requested grid/block shape. It checks positive dimensions, device/pipeline block limits, logical-dimension overflow, and total-count overflow. It preserves the kernel's own bounds checks.

Scalar argument bytes are copied into `Argument` and by Metal `setBytes`. Pending commands retain their allocations until completion. Blocking copies, frees, explicit synchronization, and normal process shutdown wait for and inspect pending commands. Error state retains backend detail, and unresolved failure prevents a successful shutdown. The process-lifetime context/error storage avoids destructor-order access after thread-local string teardown; its limits are recorded in the engineering notes. No CPU execution path exists.

## Evidence, tests, and future work

`run` writes a source snapshot and IR into an artifact directory and refuses to reuse a nonempty directory. By default it creates a unique directory under `artifacts/runs`; `--artifacts DIR` selects a new empty evidence directory. The original `artifacts/gate-a/` directory is immutable historical evidence; the post-rename canonical run is reserved for `artifacts/paralyn-gate-a/`. Host output and exit status are written to `verification.txt`. On command completion, the runtime writes `generated.metal` and `execution.json` with device, OS, revision/dirty state, LLVM version, launch dimensions/status, and GPU timestamps. New records also associate each launch with a `source-N.metal` file through `execution.json`; `generated.metal` remains the most recently completed shader for compatibility. New evidence includes transformed `host.cpp` as required by mandate v1.1. Historical five-file directories remain unchanged.

New Paralyn runs preserve `source.cu`, `paralyn-ir.txt`, `generated.metal`, `execution.json`, and `verification.txt`. The original historical set in `artifacts/gate-a/` instead retains its actual `unicuda-ir.txt` filename, original IR text, and `unicuda_commit` / `unicuda_dirty` metadata. Do not rename or rewrite those five historical files. The current verifier takes `--paralyn` and checks the new artifact naming contract. A success string alone is insufficient. The example's actual CPU comparison owns `Verification: PASS`; the CLI must not synthesize it. Positive timing and successful command completion provide GPU evidence alongside the computed result. Unavailable hardware cannot satisfy this gate.

IR/codegen, frontend, handwritten-Metal and shutdown-failure tests accompany the complete-program gates. Gate B now qualifies edge sizes, aliases, dimensions, host semantics and the documented numerical policy, with the full four-size benchmark protocol. Clean-revision evidence is in `../artifacts/gate-b/`; broader frontend/backend support remains in `../ROADMAP.md`.

A future persistent cache key must cover source/dependency contents, compiler/runtime/IR versions, options, SDK, backend target, and applicable alias layout. Atomic writes, malformed-entry handling, invalidation, and bypass behavior need tests before claiming that cache exists.

The native API now consumes the shared owned context/buffer boundary; additional backends should introduce a registry when required. Distributed work is deferred entirely: placement must be explicit, control and data planes separate, aggregate node memory never described as one GPU allocation, and scheduling explanations must account for compatibility, capacity, locality, transfer cost, and expected duration.

When assumptions fail, record the problem, root cause, evidence, alternatives, and chosen revision in `engineering-notes.md`; update the implementation and documented boundary rather than special-case the example.

## Implemented native boundary

`include/paralyn/native.h` defines C ABI 1; `runtime/native.cpp` validates numeric handle identities, reference ownership, byte ranges, view access, typed arguments and contexts before submission. `include/paralyn/native.hpp` adds move-only C++ wrappers. `bindings/python/paralyn/__init__.py` binds that exact shared library with ctypes. Kernel execution is never dispatched through Python or a CPU reference.

`compiler/ir/artifact.cpp` serializes bounded versioned verified IR. `paralyn compile` writes modules without executing input host code. The runtime deserializer/verifier/codegen has no LLVM dependency. A module retains typed entrypoints and numerical policy; source/build provenance is captured by the qualification harness, not embedded as a signed manifest. This new input boundary serves native clients but does not qualify the future public Metal-source frontend.

The native runtime uses one ordered queue per independently created context. Views and child objects retain their owners; commands retain allocations and copied scalars. Same-allocation views lower to a shared binding plus element offsets, including subviews at four-byte offsets. Native and CUDA code use identical backend compilation, execution, timing and completion handling. No second execution engine is introduced. See [native-api.md](native-api.md) for the exact lifetime/error contract and [status.md](status.md) for measured qualification.
