# Checkpoint: native C/C++ runtime and Python bindings

The session started at local/remote main `0e7041f046ee5d5ff8f2560233ed5b1ba9c812cd`. The first Stage B byte-buffer/launch slice is implemented and qualified from clean revision `76217b7708d7b503449bf42e556e4c148bda88e6`; software remains 0.0.1. Permanent evidence in `artifacts/stage-b/` records 14 passing CTests, 21 native GPU events, repeated CUDA correctness and the complete 880-launch benchmark. See `docs/implementation-audit.md` for exact files and targets, `docs/native-api.md` for ABI/lifetime/error rules, and `docs/status.md` for actual qualification. The original Gates A/B evidence remains unchanged. No release tag is implied.

## Implemented boundary

C ABI 1 provides real device/context ownership, buffers/views with explicit offsets and access, bounded verified kernel modules, reflection, typed arguments, one ordered queue/context, retained events and structured errors. C++ adds move-only wrappers; Python ctypes uses the same library. The shared Metal engine still serves CUDA. `paralyn compile` creates versioned modules without executing source host code. Native tests independently compare vector-add/affine output, aliases/canaries, copied scalars, queued dependencies and retained lifetimes; invalid input fails explicitly. No CPU kernel fallback exists.

## Exact next implementation task

Complete the next part of Stage B: add a **contiguous one-dimensional FP32 Array** layer in `bindings/python/paralyn/array.py`, backed exclusively by the existing `Buffer`/`View`/`Queue` API. Start with explicit-context `asarray`, `to_host`, elementwise `add` and `affine(scale)` using the already verified kernel module contract. Own/retain context and buffer resources; define copied host input, zero length, explicit binary32 conversion, equal-length/same-context requirements, output ownership and exception behavior before coding. Do not imply arbitrary Python compilation, NumPy protocol support, broadcasting, strides, autograd, DLPack or a CPU fallback.

Add packaged/generated kernel-module handling that works outside the source checkout, then CTest examples/tests for empty, partial and odd lengths, changed inputs/scalars, mismatched length/context, lifetime and failure cases. Capture actual Metal events and independently verify all outputs. Preserve all existing native and CUDA tests, and pin the tested Python version in the evidence. This is an implementation task, not another roadmap-only checkpoint.

In parallel with dependency-ready work, Stage C must establish an **already available and authorized** NVIDIA or AMD machine/public toolchain before a second-backend hardware claim. Only Apple M5 is observed locally. Do not invent a passing vendor cell or provision/access remote resources without authorization. Implement and qualify the other backend when those prerequisites exist; hardware absence does not remove it. Continue the independent required source/frontend/operator campaign when a hardware gate is blocked.

All 17 input families and three named interoperability clients remain required. Fourteen other input families have no implementation yet. Preserve the separate PTX/SASS provenance and reference-hardware gates and the first useful PyTorch/JAX/ONNX requirement. Later work is not optional because this session reaches a checkpoint.

## Reproduce

```sh
cmake --build build -j 4
ctest --test-dir build --output-on-failure
python3 scripts/qualify_native.py --build build --output artifacts/runs/native-new --require-clean
python3 scripts/qualify_gate_b.py --paralyn build/paralyn --artifacts artifacts/runs/cuda-new --require-clean
python3 scripts/verify_benchmark.py --benchmark build/gate_b_benchmark --paralyn build/paralyn \
  --source examples/vector_add.cu --artifacts artifacts/runs/benchmark-new --require-clean
```

Use fresh paths and run GPU captures sequentially. Permanent evidence must be generated from a clean implementation revision under ignored `artifacts/runs/`, then archived unchanged in a later evidence commit. Source, loaded library/module/binary hashes, device/OS/toolchain, command states, timings, shaders and CPU verification logs must agree. A hardware fault injection campaign remains unqualified; ordinary validation failures do not establish device-loss behavior.
