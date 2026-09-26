# Native runtime API — ABI 1

The first native C/C++ and Python byte-buffer/launch slice is implemented in `include/paralyn/native.h`, `include/paralyn/native.hpp`, `runtime/native.cpp`, and `bindings/python/paralyn/__init__.py`. CMake builds `libparalyn_native.dylib`, `native_c`, `native_cpp`, and `native_tests`. The runtime uses the same `backends/metal/engine.mm` as the existing CUDA compatibility adapter. It contains no CPU kernel fallback and does not link LLVM into native clients.

The full native portfolio remains incomplete: array operators, external memory, NVIDIA and AMD execution are still required future work. This interface does not compile arbitrary C++ or Python kernels. Its initial kernel input is a verified, versioned `.prk` module from the existing compiler. Qualification is scoped in [status](status.md) and the [portfolio ledger](portfolio-ledger.json).

## Build and execute

```sh
cmake --build build -j 4
# Build also generates build/native-kernels.prk without executing source main().
build/paralyn compile examples/native/kernels.cu --output /tmp/example-new.prk
build/native_c build/native-kernels.prk artifacts/runs/native-c-new
build/native_cpp build/native-kernels.prk artifacts/runs/native-cpp-new
PYTHONPATH=bindings/python PARALYN_LIBRARY="$PWD/build/libparalyn_native.dylib" \
  python3 examples/native/vector_add.py --module build/native-kernels.prk \
  --artifacts artifacts/runs/native-python-new
python3 scripts/qualify_native.py --build build --output artifacts/runs/native-suite-new
```

Use new output paths. `compile` validates the same CUDA AST/IR as `inspect`/`run`, writes a bounded artifact exclusively, and never runs the input's host program. Native source examples are ordinary C11/C++17, use no CUDA allocation calls, and load that artifact at runtime. Python uses the standard-library `ctypes` binding; `PYTHONPATH` and an explicit library path are the current development installation. No published wheel/package is claimed.

## Device, handles, ownership, memory

`pr_device_count/get` enumerate actual Metal devices. `pr_context_create("auto", ...)` selects the first device in stable registry-ID order; a numeric string selects its index. Each context owns one device and one ordered queue. Multiple contexts are independent. `pr_abi_version()` returns 1; the Python loader rejects a different ABI.

Context, buffer, view, module, kernel, queue and event are opaque process-local numeric handles. Zero is invalid; IDs never recycle. They are not pointers, addresses or transferable capabilities. The registry validates handle kind, liveness and context identity. The native API serializes calls; it does not promise parallel host submission throughput.

`pr_buffer_create` owns a reference-counted allocation. A view retains that allocation and context, byte offset/size, declared alignment and read/write access. A kernel retains its module/context; events retain the context and command resources. Releasing a parent handle does not invalidate children. Submissions copy scalar bytes before returning and retain allocations until completion. C++ wrappers are move-only RAII objects; Python objects support explicit `close()` and context managers.

`pr_release` consumes a valid handle once, including when waiting for completion reports an error. A second C release returns `PR_INVALID_HANDLE`; wrapper `close()` is idempotent after consuming its handle. Context, queue and event release wait for their work. Buffer/view release drops ownership without invalidating submitted work.

Zero-byte buffers and views are valid. Zero-byte host transfers validate handles/ranges and return without accessing host memory. Zero-work callers skip launch; a typed kernel cannot dispatch an empty view. Host copies block and synchronize producer work. Offsets and lengths are checked using overflow-safe arithmetic. View alignments 1, 2 and 4 are accepted; the current i32/u32/f32 kernel subset requires four-byte aligned, nonempty views. Shape, stride, dtype conversion, external borrowing and DLPack are not provided.

Same-allocation same-pointee views, including different offsets, share a Metal binding with derived pointers. Different pointee types in one alias group are rejected. The declared access must cover actual IR reads/writes. These checks do not prove dynamic kernel indices stay inside a view, nor make a data-racing overlap valid. Inputs are trusted local programs with their own kernel bounds guards.

## Modules, arguments and completion

`pr_module_load` copies/parses artifact bytes; `pr_module_load_file` reads a file. The version-1 format has explicit little-endian fields, a numerical-policy identifier, entrypoints, typed IR and source locations. Unknown versions/policies/enums, duplicate entries, truncated/trailing bytes, invalid IR and excessive size/nesting are rejected. Limits are 16 MiB, 64 kernels, 256 parameters per kernel, 100,000 nodes, nesting 64 and 4 KiB strings. The full verifier runs before Metal compilation. The current Metal module loader accepts at most 31 parameters per kernel. Pipeline compilation happens at load for distinct bindings and lazily for additional alias/offset layouts.

The artifact's implemented capabilities are the documented scalar IR subset: i32/u32/f32, reads/writes, addition/multiplication/comparison and guarded indexed work. Numerical policy 1 requests MSL 3.1 safe math, precise functions and disabled FP contraction; [numerics](numerics.md) states its measured limits. `.prk` is an internal versioned interchange boundary, not a stable general-purpose binary standard. It does not embed a signed source/build provenance manifest; the qualification recorder preserves source, IR, artifact and binary hashes with exact build/revision metadata. Future compiler bridges need their own artifact contract and need not pass through CUDA or this small IR.

`pr_module_kernel` selects an entrypoint. `pr_kernel_parameter_count/parameter` expose names, scalar/pointee types, buffer/scalar kinds and inferred access. `pr_launch` requires explicit i32/u32/f32 scalar tags and view handles, validates the schema, context and geometry, and returns an event. All queue handles for a context refer to the same ordered queue. Event identity and retained state are allocated before committing GPU work; a handle-allocation failure cannot orphan an already submitted command.

`pr_event_wait` reports completion and actual GPU start/end timestamps. `pr_event_cancel` returns `PR_UNSUPPORTED`; no fake cancellation is offered. Reads, queue/context synchronization, event waits and release propagate backend completion failures. An execution failure is terminal for the context. After the caller observes it, cleanup consumes remaining handles without repeatedly reporting the same terminal error; further execution/transfer operations still fail.

## Structured errors and shutdown

C functions return `pr_status`; `pr_last_error` copies thread-local status, operation and backend detail without clearing them. Successful operations clear that thread's last error. No C++ exception crosses the C boundary. C++ `Error` and Python `Error` preserve code, operation and message. Unsupported operations fail explicitly.

Use explicit `wait`, synchronization or `close` to catch failures. An unobserved failure during C++ destruction terminates the process; Python finalization also fails the process rather than hiding a release failure. Normal C runtime shutdown waits on remaining contexts and fails the process on an unobserved completion error. Abnormal process termination cannot guarantee completion or evidence flushing.

Positive completion, validation/artifact errors, ownership and shutdown/logging error paths have tests. Actual device-loss, GPU timeout, and allocator-exhaustion fault injection are **not qualified** by this checkpoint. No dangerous out-of-bounds shader is used to manufacture a hardware failure.

## Tests and evidence

`tests/artifact_tests.cpp` and `tests/compile_cli.py` test bounded serialization, verifier reuse, invalid modules, exclusive output creation and compile-without-host-execution. `tests/native/native_tests.cpp`, executable C/C++ examples and `tests/native/test_python.py` verify GPU output independently, offsets/canaries, aliases, copied scalars, ordered dependent work, parent/child lifetimes, empty resources and structured failures. `scripts/qualify_native.py` runs all clients and audits actual device, completed commands, positive GPU timings, generated shaders and CPU verification logs. A success string alone does not pass qualification.

The existing CUDA suite and full benchmark must keep passing after changes to the shared engine. An API declaration or ledger update is never implementation evidence. Array/operator APIs, all remaining frontend/input families, three named interoperability clients and both other backends remain required in the roadmap.
