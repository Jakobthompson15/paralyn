# CUDA compatibility: Gate A boundary

Paralyn accepts an explicitly limited CUDA source language. It does not implement CUDA binaries, PTX, the Driver API, CUDA libraries, arbitrary CUDA C++, or vendor floating-point equivalence. Supported here means implemented within these bounds; hardware qualification is listed separately in `status.md`.

## Frontend

A translation unit contains file-scope, non-template `__global__ void` definitions and ordinary host C++17. Host statements remain native C++; only kernels and launch expressions are rewritten. The parser uses independently written declarations without NVIDIA headers or libraries. Kernel declarations separate from definitions, overloads, included-file kernels, device functions, and ambiguous macro launches are not supported.

The initial kernel grammar contains `int`, `unsigned int`, and `float` scalar/buffer parameters; `const` input buffers; literals; initialized scalar locals; explicit integer conversions; `+`, `*`, `<`; buffer indexing, loads and stores; and `if` without `else` or initialization. CUDA index builtins map to unsigned 32-bit values. Predicate values are internal. Unknown operators or nodes receive source-located diagnostics instead of best-effort translation. Signed overflow and invalid kernel memory accesses are outside the valid-program contract, as they are in ordinary C++/CUDA.

FP64, shared memory, barriers, atomics, warp intrinsics, device globals, dynamic parallelism, loops, templates, `return` statements inside kernels, and general local assignment are not implemented. Device local name shadowing is rejected.

Host preprocessing that depends on `__CUDA_ARCH__`, `__CUDACC__`, or `__CUDA__` is rejected, including inactive user branches. Source-context constructs such as `__LINE__`, `__FILE__`, `__COUNTER__`, and function-name/location expressions are rejected explicitly rather than silently changing their values in transformed source. Preserving those constructs requires later source mapping work. This restriction includes user-site expansion through macros. Host code otherwise remains native, not interpreted by Paralyn.

## Launch and memory APIs

`dim3`, typed/untyped `cudaMalloc`, `cudaFree`, explicit host-to-device/device-to-host `cudaMemcpy`, `cudaDeviceSynchronize`, `cudaGetLastError`, and `cudaGetErrorString` comprise the initial shim. Unsupported copy kinds fail. Error codes/messages are the documented subset, not an assertion of complete runtime ABI behavior.

Launch configuration is evaluated before arguments, each expression once. Optional shared-memory and stream expressions must yield zero and null; known unsupported constants fail during compilation and dynamic unsupported values fail at runtime. One ordered default queue is available. Copies/free/synchronization wait and inspect pending GPU commands. An unresolved runtime failure on normal shutdown produces a nonzero process status.

CUDA-visible pointers are opaque allocation tokens. They may be stored, copied, passed to supported APIs, and checked against null. Only live allocation-base tokens are accepted. Dereference, arithmetic, interior pointers, ordering, subtraction, arbitrary integer conversions, and treating tokens as GPU addresses are unsupported. Kernel-side `buffer[index]` remains valid.

Same-allocation arguments with one pointee type are lowered to one Metal binding and derived local references. Mixed pointee types in an alias group are rejected. The implementation mechanism exists; broader aliasing behavior remains unqualified until its planned physical-GPU tests run. No `restrict` assumptions are introduced.

## Numerical and execution limits

Metal compilation uses safe math and precise functions. Gate A tests finite, exactly representable FP32 additions against an independent CPU reference using exact equality. This does not establish complete IEEE-754/CUDA equivalence. Subnormals, exceptional values, rounding/ULP policy, and broader arithmetic are qualification work. FP64 is not silently narrowed.

Launch geometry is validated against device and pipeline limits. The implementation dispatches uniform threadgroups and retains source bounds guards. No kernel runs on the CPU. Missing GPU execution or GPU timing evidence is a failure, not a successful skipped hardware gate.
