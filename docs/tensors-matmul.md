# FP32 tensor descriptors, matmul provider and MLP inference

Lane: operators/integrations (handoff task 3). Branch `lane/tensors-matmul-mlp`, based on `4f8fb22`. Software remains 0.0.1. This file documents what the branch implements and measured on the recorded Apple M5; it does not update `status.md`, the ledger or the handoff, which are reconciled separately.

**Not claimed:** cuBLAS/BLAS ABI or API compatibility, PyTorch/JAX/ONNX integration, NVIDIA/AMD or any cross-vendor matmul, performance, batched/strided/broadcast tensor algebra, other dtypes, or training. The provider executes only where a backend runs its MSL: currently physical Metal devices. There is no CPU fallback for any operator. CPU code in tests and examples is an independent reference oracle only.

## What exists

| Surface | Location |
|---|---|
| C ABI (additive, versioned; ABI 1 unchanged) | `include/paralyn/tensor.h`, `runtime/tensor.cpp` (in `libparalyn_native`) |
| C++ wrappers | `include/paralyn/tensor.hpp` (`paralyn::tensors`); capability/timing wrappers added to `include/paralyn/native.hpp` |
| Python | `bindings/python/paralyn/tensor.py`; capability/timing wrappers in `paralyn/__init__.py` |
| Applications | `examples/native/mlp.py`, `examples/native/mlp.cpp` (`build/native_mlp`) |
| Tests | `tests/native/tensor_tests.cpp`, `tests/native/test_tensors.py`, runner `tests/native/run_tensors.py`; CTests `tensors_cpp`, `tensors_python`, `tensors_mlp_cpp`, `tensors_mlp_python` |

Shared-file edits are additive: two `target_sources`/test blocks in `CMakeLists.txt`; one internal error-reporting hook appended to `runtime/native.cpp` (declared in the non-installed `runtime/native_internal.hpp`); three wrapper methods in `native.hpp`; ctypes records, three signatures, `Context.capabilities`, `Event.timing`, `device_capabilities` and tensor re-exports in `paralyn/__init__.py`; `Context.close` now also releases cached tensor resources. `native.h`, `.prk` v1, the CUDA path, Gate A/B and `unicuda` aliases are untouched.

## Tensor descriptor v1

`pr_tensor_desc_v1 { struct_size, version, buffer, byte_offset, dtype, rank, shape[8], strides[8] }`. Strides are signed element counts relative to `byte_offset`. `pr_tensor_desc_contiguous` fills a row-major record and initializes `struct_size`/`version`. `pr_tensor_desc_validate` (and every operator) checks, in order:

| Condition | Status |
|---|---|
| null record, `struct_size != sizeof` | `PR_INVALID_ARGUMENT` |
| `version != 1` | `PR_UNSUPPORTED` |
| dtype other than `PR_F32` | `PR_UNSUPPORTED` |
| rank > 8, nonzero unused dimensions, byte offset not a multiple of 4 | `PR_INVALID_ARGUMENT` |
| negative stride; more than INT32_MAX elements | `PR_UNSUPPORTED` |
| element-count or extent overflow; `byte_offset + extent > buffer size` | `PR_OUT_OF_BOUNDS` |
| invalid buffer handle | `PR_INVALID_HANDLE` |

Extent is `(Σ (shape[i]-1)·stride[i] + 1)·4` bytes, or 0 for an empty tensor. A validated non-contiguous layout (for example column-major strides) is legal as a descriptor, but v1 operators reject it with `PR_UNSUPPORTED`; nothing is silently reordered or copied. Dimensions of extent 1 ignore their stride for contiguity. One-dimensional `paralyn.Array`/`paralyn::arrays` are unchanged and independent of this record.

## Provider identity and execution path

The operators are a handwritten MSL module packaged as a PARALYNX1 `.prx` executable: producer `paralyn.msl.tensor`, producer version `0.0.1`, source name `paralyn/tensor_f32.metal`, target `metal-msl3.1`, numerical policy 1. `pr_tensor_operators_artifact` returns its exact bytes (hashable; Python's MLP report records the SHA-256). `pr_tensor_operators_load` passes those bytes to the public `pr_module_load`, so the provider goes through the same container validation, source-signature grammar, Metal compilation with safe math/precise functions/`FP_CONTRACT OFF` prelude, and reflection checks as any public MSL module. The caller owns the returned module handle.

`runtime/tensor.cpp` is an ordinary client of the public C ABI: it validates descriptors, creates exact-extent views with `pr_view_create`, checks that the module's entry schemas are the provider's (a different module with the same entry names is rejected with `PR_INVALID_ARGUMENT`), submits with `pr_launch` on the caller's queue and releases its temporary view/kernel handles. Queue/module/view context mismatches are therefore reported by `pr_launch` as `PR_CONTEXT_MISMATCH`. Errors from inner calls are re-reported with the operator name as `pr_error.operation` (for example `matmul_f32`) and the inner operation prefixed to the message. MPS is not used.

Entrypoints (bindings in order 0..n): `paralyn_matmul_f32(a, b, c, m, n, k, transpose_a, transpose_b)` with required workgroup 16×16×1; `paralyn_bias_activation_f32(x, bias, out, count, columns, activation)`, `paralyn_activation_f32(x, out, count, activation)` and `paralyn_fill_f32(out, count, value)`, each with required workgroup 256×1×1. The dispatched source is exported with every runtime evidence directory.

## Matmul contract

`pr_matmul_f32(queue, operators, &pr_matmul_v1{m, n, k, transpose_a, transpose_b, a, b, c}, &event)` computes `C[m,n] = op(A)·op(B)`. A is stored `[m,k]` (or `[k,m]` when `transpose_a`), B `[k,n]` (or `[n,k]` when `transpose_b`), C `[m,n]`, all rank 2, contiguous row-major FP32. The explicit `m`, `n`, `k` must equal the descriptor shapes; any disagreement, rank ≠ 2 or transpose flag outside {0, 1} is `PR_INVALID_ARGUMENT`. Dimensions above INT32_MAX are `PR_UNSUPPORTED`.

- `m == 0` or `n == 0`: descriptors and the module are validated, nothing is submitted and `*event = 0`. No event is invented.
- `k == 0` with nonempty C: the GPU fill entry writes +0.0 into every element (the empty sum); an event is returned.
- Otherwise one 16×16-tiled dispatch with grid `(ceil(n/16), ceil(m/16), 1)`; each thread owns one output element. Transposition is handled in the tile loads, not by copying.

C++ `tensors::matmul(queue, ops, m, n, k, a, b, c, ta, tb)` and Python `matmul_into(queue, ops, A, B, C, m=, n=, k=, ...)` expose the explicit record; high-level `tensors::matmul(a, b, ta, tb)` / `paralyn.matmul(a, b, transpose_a=, transpose_b=)` derive the dimensions and allocate a new output.

## Bias/activation contract

`pr_bias_activation_f32` computes `out[i,j] = act(x[i,j] + bias[j])` for rank-2 `x`/`out` of equal shape `[rows, columns]` and rank-1 `bias[columns]`; `bias == NULL` applies only the activation. Activation `NONE` or `RELU`; the reserved field must be zero. ReLU replaces only values that compare less than zero with +0.0: NaN and -0.0 pass through unchanged, -inf becomes +0.0. Empty outputs submit nothing. High-level wrappers: `bias_add(x, bias, relu)` and `relu(x)`.

## Alias, ordering and lifetime semantics

- Inputs may alias freely (the same allocation can be bound read-only more than once, e.g. `A·Aᵀ` from one buffer; tested).
- The output must not share an allocation with any input. Overlapping byte ranges: `PR_INVALID_ARGUMENT` (no in-place operators). Disjoint ranges of the same allocation: `PR_UNSUPPORTED`, because the public MSL profile does not rewrite Metal's distinct-pointer assumptions for writable aliases. Buffer handles map one-to-one to allocations, so this check is exact. The rule is allocation-based and applies even to empty tensors.
- All work is enqueued on the context's single ordered queue; a consumer submitted after its producer observes the producer's writes. Host uploads/downloads synchronize as documented for ABI 1.
- `pr_launch` retains every bound allocation until completion; callers may release input tensors, views or buffers right after enqueueing. High-level outputs retain their producing event; `wait()`/`timing()` return nothing for uploads and empty results. Closing a tensor context releases its cached provider module/queue, drains work, and rejects new work; existing results stay readable.

## Numerics addendum

This extends [numerics.md](numerics.md); it does not widen the Gate B elementwise guarantee.

- **Element and accumulation type:** FP32 inputs, FP32 products, one FP32 accumulator per output element. No reduced precision, no split-K, no FP64 on the device.
- **Order:** `sum = 0; for p in 0..k-1: sum = sum + (a(i,p) * b(p,j))` in strictly increasing `p`, identical for every tile size and transposition; padding lanes never enter the sum. Each product and each addition rounds separately (policy 1: safe math, `#pragma STDC FP_CONTRACT OFF`).
- **Observed:** on the tested M5 every nonempty GPU matmul in the suites (38 C ABI plus 2 high-level C++ dispatches and 30 Python dispatches over shapes including 1, 15, 16, 17, 33, 47, 65, 129, 257 and 300, all four transpose combinations, byte offsets and aliasing) was **bit-for-bit equal** to the independent host sequential FP32 reference compiled with `-ffp-contract=off` (C++) or emulated with explicit binary32 rounding (Python). This is evidence of the documented order and of no contraction on this device and toolchain; it is not a cross-vendor or cross-driver bitwise promise.
- **Tolerance against the float64 reference:** each element must satisfy `|c − ĉ| ≤ γ_k · Σ_p |a(i,p)·b(p,j)|`, `γ_k = k·u/(1 − k·u)`, `u = 2⁻²⁴` (Higham, *Accuracy and Stability of Numerical Algorithms*, Thm. 3.1 / §3.1), where `ĉ` is the float64-accumulated exact-input sum. The bound is a-priori, not fitted, and assumes round-to-nearest without underflow; test inputs are 16-bit dyadic values in [−1, 1) so products never approach the subnormal range. Worst observed error/bound ratio: 0.936 (C++ suite, dominated by k = 1 cases where a single product rounding approaches `u`), 0.752 (Python suite).
- **Bias:** `x + bias` is one FP32 rounding (bitwise compared in tests); ReLU is exact.
- **MLP tolerance:** hidden `h = relu(fl(fl(x·W1) + b1))` must satisfy `|h − h₆₄| ≤ e₁ = γ_{in+1}(Σ|x·W1| + |b1|)` (ReLU is 1-Lipschitz); output must satisfy `|y − y₆₄| ≤ Σ_p e₁(p)|W2(p,j)| + γ_{hidden+1}(Σ_p (|h₆₄(p)| + e₁(p))|W2(p,j)| + |b2(j)|)`. Default run (x[64,257], W1[257,130], W2[130,11], seed 2026): max |error| 1.215e-07, worst error/bound 0.010 hidden and 0.0005 output, identical in the C++ and Python applications.
- **Exceptional values:** NaN/inf propagate through MSL arithmetic; only the bias/ReLU cases above (NaN, ±inf, ±0) are compared in tests. Subnormal flushing is permitted by MSL and is not qualified for these operators. Overflow is not detected.
- **Determinism:** the order is fixed per element, so repeated runs on the same device/driver gave identical bits; no universal or cross-vendor determinism is claimed.

## Capability and timing wrappers

The existing C-ABI `pr_device_capabilities_v1`/`pr_event_timing_v1` queries are now exposed in C++ (`native::Context::capabilities()`, `native::device_capabilities(i)`, `native::Event::timing()`, `tensors::Tensor::timing()`) and Python (`Context.capabilities`, `paralyn.device_capabilities(i)`, `Event.timing()`, `Tensor.timing()`, `ClockDomain`, `DeviceCapabilities`, `EventTiming`). Python reports absolute timestamps as `None` unless the backend marks them valid. Per-layer GPU durations printed by the MLP examples are single-run informational event values, not a performance measurement; no benchmark or timing qualification was run in this lane.

## Evidence on this branch

Apple M5, macOS 26, LLVM/Clang 21.1.8, Debug build in the lane worktree. `ctest -j1` full suite result is recorded in the lane report. Tensor CTests audit each exported `execution.json`: backend Metal, `cpu_fallback: false`, every launch completed with valid GPU duration and its dispatched source present, launch count equal to the events the test observed; the MLP runs must show exactly `matmul, bias_activation, matmul, bias_activation` in order. `tensors_cpp`: 47 source-linked commands plus 4 high-level events; `tensors_python`: 37 commands; each MLP: 4 commands. A separate runtime-only configuration (`-DPARALYN_BUILD_COMPILER=OFF`, no LLVM linked) also passed its 13 CTests including all four tensor tests, confirming the provider needs no compiler at runtime. No evidence directory under `artifacts/` was created or modified; a clean-revision evidence capture remains to be archived.

## Remaining work in this area

Strided/view tensors and operator support for non-contiguous layouts; batched matmul and broadcasting; reductions, softmax, normalization, transpose/gather/scatter as tensor operators; a performance-oriented kernel and the separately required benchmark protocol; FP16/BF16 and other accumulation policies; CLI exposure and wheel/installed-package requalification including `tensor.py`; the same provider contract on CUDA/HIP backends; cuBLAS library compatibility; PyTorch custom ops/device backend; transformer inference.
