# Native runtime API — design target, not implemented

Status after CUDA/Metal Gate B, 2026-09-25: **design only; next implementation milestone**. The internal C++ `paralyn::Argument`/`Kernel` launch helpers and handwritten-MSL test hook do not constitute the required public native C/C++ interface. Memory allocation still uses the CUDA compatibility shim. No Python runtime binding, `paralyn.asarray`, or public native module loader is available.

## First native milestone

With [CUDA/Metal Gate B qualified](../artifacts/gate-b/build-validation.json), expose the same actual runtime through a small C-compatible boundary and ergonomic C++ ownership wrappers. Do not build a second execution engine. Initial success is a native application that discovers one GPU, allocates and copies buffers without CUDA API names, launches a supported compiled kernel, observes completion/errors, reads output, and independently verifies vector addition and a second useful operation.

The API must distinguish context, device, owned buffer, borrowed view, compiled module, kernel, typed argument, queue, event, and structured error. These are design responsibilities, not currently exported handle types or a stable ABI promise. Choose exact spellings/ABI versions in the implementation change and make introductory examples executable documentation tests.

## Ownership and memory contract to implement

A context owns one explicitly selected eligible device initially. An owned buffer is a reference-counted device allocation. A view carries its allocation identity, context/device ownership, byte offset, byte size, element alignment, access mode, and retained lifetime. A borrowed external view requires a verified ownership/synchronization bridge; no external borrowing or DLPack claim is part of the first milestone.

Views use explicit offsets and sizes, not CUDA-looking pointer arithmetic. Validate bounds and alignment before submission, reject cross-context/device handles, and retain underlying allocations until dependent work completes. Native buffers must not inherit the CUDA base-token restriction. CUDA adapters continue validating their own tokens separately.

Define zero-byte buffers as valid empty owned buffers in the initial API contract; reading/writing zero bytes is a no-op after handle/context validation, and zero-work applications skip kernel dispatch. Host transfer calls copy bytes explicitly and block initially. Asynchronous copies can arrive later with explicit dependencies and lifetime rules. Shapes, strides, broadcasting, implicit dtype conversion, and external-memory aliasing are not implied by a byte-buffer API.

## Kernel and completion boundary

Reuse validated artifacts from the existing compiler/lowering path. Compiled modules describe entrypoints, typed argument schemas, required capabilities, numerical policy, and provenance. A mature compiler bridge may later supply a backend artifact directly after satisfying the same contract; it need not reconstruct the scalar IR. Do not count a raw shader string/test-only hook as a qualified source frontend.

Initial scalar arguments are explicit i32/u32/f32 values copied at submission. Buffer arguments are owned views with access declarations. Preserve same-allocation alias identity; reject unsupported type/layout overlaps. Launch geometry is explicit and checked against kernel/device limits.

The first queue is ordered. An event represents completion plus possible asynchronous failure and retains submitted resources. A C API returns structured status rather than exceptions across the boundary; C++ wrappers may offer exceptions while preserving the underlying code, operation, and backend detail. Cancellation cannot promise to abort already-submitted GPU work; an initial API may report cancellation unsupported. Host reads require completed producer work.

## Python follows the same runtime

Begin with device discovery, owned buffers, explicit upload/download, already-supported module/kernel launch, and waiting/error propagation. Bind the C/C++ runtime rather than inventing Python allocation/execution semantics. Add an agreed small array/operator surface only afterward. Python is a native client interface here; compiling arbitrary Python or providing a new `@paralyn.kernel` language is not part of this first binding.

The proposed array experience in the mandate remains a design target. NumPy, Array API, MLX, DLPack, and framework integration require separate tested contracts. No CPU fallback or relabeling of device allocations is permitted to make interoperability appear to work.

## Qualification before advertising support

Test create/destroy/release ordering; double/foreign handles; view offsets/bounds/alignment; empty buffers; scalar lifetime; buffer mutation and alias cases; queue/event lifetime; async failures; actual GPU output; and repeated calls. Test equivalent behavior through C, C++ wrappers, and Python when each exists. Use pinned compiler/runtime versions and record a simple reference case plus a useful workload in the portfolio ledger.

The API's design can evolve before implementation evidence establishes its contract. Any implemented subset must be stated explicitly, with unimplemented handles/operations clearly reported rather than success-returning stubs.
