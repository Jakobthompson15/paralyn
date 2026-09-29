# Transformer-block operators and a pre-LN encoder block

Lane: operators/integrations (handoff task 4: "strided and batched operators, reductions and softmax/normalization, working toward the transformer block"). Branch `lane/transformer-block`, based on `cccb177`. Software remains 0.0.1. This file records what the branch implements and what was measured on the recorded Apple M5. It does not update `status.md`, the handoff or the ledger; those are reconciled separately.

**Not claimed:**

- This is not PyTorch, JAX or ONNX integration.
- It is not cuDNN, cuBLAS or BLAS API/ABI compatibility.
- It is not training: there are no gradients or optimizers.
- It is not a performance result, and no benchmark was run.
- There are no other dtypes, and no NVIDIA/AMD or cross-vendor execution.

The operators execute only where a backend runs the provider's MSL, which today means physical Metal devices. There is no CPU fallback for any operator or for any stage of the block. CPU code in the tests and examples is an independent reference oracle only.

This lane extends the existing `paralyn.msl.tensor` provider and the FP32 tensor descriptor v1 ([tensors-matmul.md](tensors-matmul.md)). It is the same artifact and the same provider-identity, handle, context, zero-work and alias rules. It is not a parallel provider.

## What exists

| Surface | Location |
|---|---|
| C ABI (additive, version 1 records; ABI 1 and `native.h` unchanged) | `include/paralyn/tensor.h`, `runtime/tensor.cpp` |
| C++ wrappers | `include/paralyn/tensor.hpp` (`paralyn::tensors`) |
| Python | `bindings/python/paralyn/tensor.py`, plus re-exports in `paralyn/__init__.py` |
| Float64 reference oracles | `examples/native/transformer_reference.hpp` and `examples/native/transformer_reference.py`, written separately |
| Applications | `examples/native/transformer_block.cpp` (`build/native_transformer_block`) and `examples/native/transformer_block.py` |
| Tests | `tests/native/transformer_ops_tests.cpp`, `tests/native/test_transformer_ops.py`, runner `tests/native/run_transformer.py` |
| CTests | `transformer_ops_cpp`, `transformer_ops_python`, `transformer_block_cpp`, `transformer_block_python`, `transformer_block_match` |

### New C entry points

- `pr_batched_matmul_f32` with `pr_batched_matmul_v1`
- `pr_reduce_rows_f32` with `pr_reduce_rows_v1` and `pr_reduce_op`
- `pr_softmax_rows_f32` with `pr_softmax_rows_v1`
- `pr_layer_norm_f32` with `pr_layer_norm_v1`
- `pr_add_f32` with `pr_add_v1`
- The new enum value `PR_ACTIVATION_GELU_TANH = 2`, accepted by the existing `pr_bias_activation_f32`. With a bias it is a fused bias+GELU; with `bias == NULL` it is GELU alone.
- `pr_tensor_operators_capabilities` with `pr_tensor_operators_capabilities_v1`: a library-level query of the implemented operators, activations and reductions, and of the enforced shape limits (see below).

**Deliberate deviation: GELU is an enum value, not a separate entry point.** The task asked for a new versioned entry point per operator. GELU instead widens the accepted values of the existing version-1 `pr_bias_activation_v1` record, because fused bias+GELU is exactly that record's operation and a second record would duplicate its validation, alias rules and kernel. The record version stays 1, and native ABI 1 is untouched. To avoid trial-and-error detection, the provider now has a versioned capability query:

- `activations` has bit `PR_TENSOR_ACTIVATION_BIT(PR_ACTIVATION_GELU_TANH)` set when GELU is accepted. The C++ wrapper is `paralyn::tensors::supports_activation`, and the Python binding is `paralyn.tensor_operators_capabilities().activations`.
- GELU and the query were added on the same branch. A library that lacks the `pr_tensor_operators_capabilities` symbol therefore also lacks GELU.
- The query reports what this library implements. It does not report device availability; execution still requires Metal.
- Tests check that the record agrees with behavior. Each of the 64 activation values 0–63 is submitted to `pr_bias_activation_f32`. Exactly the three advertised values run on the GPU and are value-checked; the other 61 are `PR_INVALID_ARGUMENT`.

### Shared-file edits

All shared-file edits are additive:

- `CMakeLists.txt` gains two executables and five tests inside the existing tensor block.
- `paralyn/__init__.py` gains one import/`__all__` block.
- `tensor.h`, `tensor.hpp`, `tensor.py` and `runtime/tensor.cpp` gain appended records, functions and kernels.

The existing `paralyn_bias_activation_f32` and `paralyn_activation_f32` kernels now call a shared `paralyn_activate` helper. It has the identical ReLU semantics plus the GELU branch. The existing MLP output bytes are unchanged: the SHA-256 is still `3ecb2b1d...b073`, and `tensors_mlp_match` passes.

`tests/native/tensor_tests.cpp` changed in one assertion: the provider entry count went from 4 to 10.

`runtime/native.cpp`, `native.h`, `cli/main.cpp`, `.prk` v1, the CUDA path, Gate A/B, the `unicuda` aliases and `artifacts/` are untouched.

### Provider artifact

The provider artifact now has 10 entries, in this order:

1. `matmul`
2. `bias_activation`
3. `activation`
4. `fill`
5. `batched_matmul`
6. `batched_fill`
7. `reduce_rows`
8. `softmax_rows`
9. `layer_norm`
10. `add`

The producer (`paralyn.msl.tensor`), producer version, source name and numerical policy 1 are unchanged. The artifact's SHA-256 changed from `2b69286f...e78c` to `3ce8a3156c4089dedab3aa5ca69c73bb489030d607c8a364eb3b21beeacf9803`.

## Common contract

Every new operator follows the tensor v1 contract:

- **Record validation.** Records carry `struct_size` and `version`:
  - An unknown version is `PR_UNSUPPORTED`.
  - A size mismatch is `PR_INVALID_ARGUMENT`.
  - `reserved` fields must be zero.
- **Descriptors.** They are validated as in v1. All operators except batched matmul require contiguous row-major layouts; any other layout is `PR_UNSUPPORTED`. Nothing is reordered or copied.
- **Handles and contexts.** An unknown, released or wrong-kind queue or module is `PR_INVALID_HANDLE`. A module that was not returned by `pr_tensor_operators_load` is `PR_INVALID_ARGUMENT`; this includes the genuine artifact loaded with plain `pr_module_load`. A module or buffer from another context is `PR_CONTEXT_MISMATCH`. All of these are checked before the zero-work decision.
- **Zero work.** When there is no output element, nothing is submitted and `*event = 0`.
- **Row limit.** The row operators (`reduce_rows`, `softmax_rows`, `layer_norm`) launch one 256-lane threadgroup per row. The backends reject a logical grid dimension (threadgroups × lanes) above 2³²−1, so these operators accept at most `PR_TENSOR_ROW_OPERATOR_MAX_ROWS` = ⌊(2³²−1)/256⌋ = **16,777,215** rows. For softmax the limit applies to batch × rows. Larger shapes are `PR_UNSUPPORTED`, with an operator-level message that names the limit. The check uses the shape and runs before the zero-work decision, so `[2²⁴, 0]` is rejected as well. The exact boundary `[16777215, 1]` runs on the GPU in the tests (row sum, row max, softmax and LayerNorm, each with an exact expected value), and `[2²⁴, 1]` is rejected. Before this check, `[2²⁴+5, 1]` failed at `pr_launch` with the backend's `Logical grid dimension exceeds 32-bit indexing`.
- **Other grids are within limits by construction.** Batched matmul launches `(⌈n/16⌉, ⌈m/16⌉, batch)` groups of 16×16×1. With `m`, `n` and `batch` each at most INT32_MAX, every logical dimension stays below 2³². The elementwise and fill launches cover at most INT32_MAX elements in groups of 256.
- **Aliasing.** Inputs may alias each other. An output that overlaps an input is `PR_INVALID_ARGUMENT`. An output in the same allocation as an input but in a disjoint range is `PR_UNSUPPORTED`. The rule is allocation-based and applies to empty tensors too.
- **Errors.** `pr_error.operation` is the entry name: `batched_matmul_f32`, `reduce_rows_f32`, `softmax_rows_f32`, `layer_norm_f32` or `add_f32`. Success clears `pr_last_error`.
- **Numerics.** Policy 1 applies: safe math, precise functions and `FP_CONTRACT OFF`. Each operation rounds separately.
- **Row operators.** Reductions, softmax and LayerNorm use one 256-lane threadgroup per row with a fixed order:
  1. Lane `t` combines `j = t, t+256, ...` in increasing `j`. Sums start from `+0.0` and maxima from `-inf`.
  2. A pairwise tree follows, with `p[t] = p[t] ⊕ p[t+w]` for `w = 128, 64, …, 1`.

  The order does not depend on the device's SIMD width.

### Numerical assumptions behind every tolerance

These come from the MSL specification, Table 8.1 (precise math):

- `+`, `-`, `*` and `sqrt` round to nearest, with relative error at most `u = 2⁻²⁴`.
- `exp` is within 4 ulp and `tanh` within 5 ulp. `k` ulp is at most `k·2⁻²³·|v|`.
- Division is deliberately budgeted at 2.5 ulp. That is the fast-math table's figure, not the precise-mode figure.
- Any subnormal result may be flushed to zero, so each rounding adds 2⁻¹²⁶.
- There is no overflow. Overflow and non-finite inputs are tested separately against exact expectations.

The references propagate `(value, error)` pairs through each floating-point operation of the documented evaluation order:

- **add/sub:** `e = e_a + e_b + u(|v| + e_a + e_b)`
- **mul:** `e = |a|e_b + |b|e_a + e_a e_b + u(...)`
- **div:** `(|a|e_b + |b|e_a)/(|b|(|b| − e_b)) + ε_div(...)`
- **sqrt:** `e_a/(√a + √(a − e_a)) + u(...)`
- **exp:** `v(e^{e_a} − 1) + ε_exp(...)`
- **tanh:** `sech²(|a| − e_a)·e_a + ε_tanh(...)`
- **max:** exact and 1-Lipschitz
- **Fixed-tree sum over n terms:** `γ_L Σ|x|` with `L = ⌈n/256⌉ + 8` (Higham, §4.2)
- **Dot product:** `γ_k Σ|a b|` (Higham, Thm. 3.1)

`value` is exact math in float64 from the exact FP32 inputs. Comparisons add a float64 slack of `2⁻⁴⁵|v| + 2⁻¹²⁶`. These bounds are a-priori and rigorous under the stated assumptions. They are not fitted, and they ignore error cancellation, so they are conservative. NaN never passes a bound check.

## Operator contracts

### Batched strided matmul: `pr_batched_matmul_f32`

The operator computes `C[b] = A[b]·B[b]` for `b < batch`:

- **Shapes.** A is `[batch,m,k]`, B is `[batch,k,n]` and C is `[batch,m,n]`, all rank 3. The explicit `batch`, `m`, `n` and `k` must equal the shapes (`PR_INVALID_ARGUMENT` otherwise).
- **Strides.** Unlike `pr_matmul_f32`, every operand may have arbitrary non-negative element strides. Transposes (`Kᵀ`), head splits of a fused QKV buffer, concatenated-head outputs and batch broadcast (batch stride 0 on an input) are all views; nothing is copied.
- **Limits.** Each operand's extent must be at most INT32_MAX elements (`PR_UNSUPPORTED`), and dimensions must be at most INT32_MAX.
- **Output layout.** The output must be provably injective. With the dimensions ordered by stride, each dimension of extent greater than 1 must step past the span of the smaller ones. Otherwise the call is `PR_INVALID_ARGUMENT`. Examples that are rejected: a zero batch stride on C, and interleaved rows and columns.
- **Order.** The order is identical to `pr_matmul_f32`: FP32 products and one FP32 accumulator per output, in strictly increasing `k`, with a 16×16 tile and grid `(⌈n/16⌉, ⌈m/16⌉, batch)`.
- **`k == 0`.** A strided GPU fill (`paralyn_batched_fill_f32`) writes `+0.0` only to the addressed elements. Zero-byte views cannot be bound, so the fill has its own entry.
- **NaN/Inf.** Values propagate through IEEE arithmetic. A NaN reaches exactly the outputs whose dot product contains it (tested).

**Tolerance.** Every output is compared **bit-for-bit** with a sequential FP32 reference in the documented order, and must also satisfy the float64 dot bound.

### Row reductions: `pr_reduce_rows_f32`

The operator computes `out[i] = ⊕_j x[i,j]` for x `[rows, columns]` and out `[rows]`.

- **SUM** uses the fixed lane/tree order above. The empty sum is `+0.0`. An empty row (`columns == 0`) is written by the GPU fill entry.
- **MAX** is NaN-propagating: NaN if any element is NaN. `+0.0` is preferred over `-0.0`. The maximum of an empty row is `-inf`. Because this maximum is total and order-independent, the result does not depend on scheduling.

**Tolerance.** Both operations are **bit-exact** against a CPU emulation of the fixed order. SUM must also satisfy `γ_L Σ|x|` against float64. The bound is exercised with non-dyadic values over a 2⁴⁰ dynamic range with cancellation, because dyadic test inputs sum exactly.

### Row softmax: `pr_softmax_rows_f32`

The operator computes, over the last dimension of rank-2 `[rows,columns]` or rank-3 `[batch,rows,columns]` inputs:

```
v = fl(scale·x)
m = max over unmasked v
e = exp(v − m)
s = tree sum of e
out = e / s
```

The operator recomputes `exp(v − m)` for the output. It is deterministic.

- **Scale.** `scale` must be finite and greater than 0 (`PR_INVALID_ARGUMENT` otherwise).
- **Causal mask.** `causal = 1` masks column `j` of row `i` when `j > i + (columns − rows)`. The row index is taken within each matrix. Masked outputs are exactly `+0.0`, and masked inputs never enter `m` or `s`, so a NaN in a masked position is ignored (tested). Causal masking requires `columns ≥ rows`, so no row is ever fully masked. `causal` must be 0 or 1.

**NaN/Inf policy.** This is documented and tested, not detected:

- Any NaN among a row's unmasked inputs gives an all-NaN row.
- A `+inf` among them gives an all-NaN row (`inf − inf`).
- A row whose unmasked inputs are all `-inf` gives an all-NaN row.
- If `scale·x` overflows to `+inf`, the row is NaN.
- Individual `-inf` entries alongside finite ones give exactly `+0.0`.

**Tolerance.** The float64 running bound applies, derived through `mul(scale)`, `sub(m)`, `exp`, the tree sum and `div`. The shift by the maximum cancels mathematically, so the bound is conservative.

### LayerNorm: `pr_layer_norm_f32`

For x `[rows, columns]`, with gamma and beta `[columns]` both required:

```
mean     = tree_sum(x) / n
variance = tree_sum((x − mean)²) / n        # two-pass, biased (population)
rstd     = 1 / sqrt(variance + ε)
out      = ((x − mean)·rstd)·gamma + beta
```

- `ε` must be finite and greater than 0. A zero ε would make constant rows NaN.
- `columns ≤ 2²⁴`, so that `float(n)` is exact. Larger values are `PR_UNSUPPORTED` (tested with 2²⁴+1).

**Non-finite policy.** Any NaN or Inf in a row gives an all-NaN row, through the mean. A NaN in gamma or beta affects only its column. A constant row gives exactly `beta` (tested).

**Tolerance.** The float64 running bound is composed through the exact evaluation order above. It is exercised with off-center rows (mean 1000 with unit spread), whose cancellation the bound accounts for.

### GELU: `PR_ACTIVATION_GELU_TANH`

**Formula.** This is the tanh approximation with FP32 constants, evaluated in exactly this order:

```
cube  = (x·x)·x
inner = x + 0.044715f·cube
t     = tanh(0.7978845608028654f·inner)
y     = (0.5·x)·(1 + t)
```

**Why tanh and not erf.** The Metal standard library provides `tanh` (within 5 ulp in precise mode) and has **no `erf`**; `metal_math` on this toolchain declares none. An erf-based GELU would need a hand-written erf approximation with its own separately qualified accuracy. The tanh form has a closed-form error analysis on documented MSL functions, and it is the formulation used by the original BERT/GPT-2 code and by `approximate='tanh'` in common frameworks. The contract is this function. It is not an approximation of erf-GELU, and the reference evaluates the same formula in float64 with the same FP32 constant values.

**Exceptional values.** These are exact expectations, tested:

| Input | Output |
|---|---|
| NaN | NaN |
| `+inf` | `+inf` |
| `-inf` | NaN, because the result is `-inf·0` under this formula (documented, not patched) |
| `±0` | `±0`, sign kept |
| `1e20` | `1e20`, since `x³` overflows and `t = 1` |
| `-1e20` | `-0.0` |

Finite saturation (`±30`) is checked against the bound, because tanh may be up to 5 ulp from `±1`.

**Fused bias+GELU.** `out = gelu(fl(x + bias))`.

### Elementwise add: `pr_add_f32`

The operator computes `out = x + y` for contiguous tensors of equal shape and rank 0–8, with one FP32 rounding. It is **bit-exact** against FP32 `x + y`. IEEE NaN, Inf, signed-zero and overflow behavior is tested.

## The pre-LN encoder block

`examples/native/transformer_block.{cpp,py}`. The defaults are T=24 tokens, d_model=48, 4 heads (d_head=12), d_ff=192, seed 2026, ε=1e-5, and causal masking; `--no-causal` switches the mask off. All sizes are deliberately not multiples of the 16-wide tile.

Weights are seeded xorshift32 values that are exact in FP32. The weight scales are 0.125 and 0.0625, gamma is `1 ± 0.25` and the biases are `±0.25`. The generation order is the same in both applications.

The Q/K/V projections are one fused `[d, 3d]` matmul. The heads are strided views of the `[T, 3d]` result:

- `Q_h`: `[H,T,Dh]` with strides `[Dh,3d,1]`
- `K_hᵀ`: `[H,Dh,T]` with strides `[Dh,1,3d]` at column d
- `V_h`: at column 2d
- Attention output: written through C strides `[Dh,d,1]` into a concatenated `[T,d]` buffer

The softmax scale is `fl32(1/√12)`.

### GPU launches

There are exactly 15 GPU launches, and the runner audits this sequence in the exported `execution.json`:

1. `layer_norm` (ln1)
2. `matmul` (qkv)
3. `bias_activation` (qkv bias)
4. `batched_matmul` (`Q_h K_hᵀ`)
5. `softmax_rows` (causal)
6. `batched_matmul` (`P_h V_h`, concatenated heads)
7. `matmul` (output projection)
8. `bias_activation` (output bias)
9. `add` (residual 1)
10. `layer_norm` (ln2)
11. `matmul` (MLP up-projection)
12. `bias_activation` (bias + GELU)
13. `matmul` (MLP down-projection)
14. `bias_activation` (bias)
15. `add` (residual 2)

The host only uploads inputs and weights, then reads the results back. No stage is computed on the CPU. Each application also checks that its exported evidence has one launch per observed event and `cpu_fallback: false`.

### Verification

Verification covers 11 read-back stages: ln1, qkv, scores, probs, attention, projection, h1, ln2, ff1, ff2 and output. Both modes are enforced in both applications:

- **End to end.** A float64 reference is chained from the exact inputs, with the running bound composed through all stages. The observed output max |error| is 2.62e-7 (causal) and 2.21e-7 (bidirectional), and the worst error/bound over all stages is 0.115, in ln1. The composed bound becomes loose late in the block: the output bound is up to 2.4e-2 against observed errors of about 2.6e-7. It is rigorous, but on its own it would only catch gross errors.
- **Stage local.** Each stage is recomputed in float64 from the GPU's own FP32 inputs to that stage, treated as exact, which gives tight per-stage bounds. The worst error/bound per stage (causal run) is:

  | Stage | Worst error/bound |
  |---|---|
  | scores | 0.197 |
  | ln1 | 0.115 |
  | attention | 0.112 |
  | ln2 | 0.111 |
  | probs | 0.096 |
  | qkv | 0.066 |
  | projection | 0.059 |
  | ff1 | 0.044 |
  | ff2 | 0.014 |

  The two residual adds are single correctly rounded additions: their ratio reaches 0.999, which is the theoretical `u|v|` limit, and they are additionally required to be **bit-exact** FP32 `x + y` of their GPU inputs.

**Mutation checks.** These were run by hand and are not CTests:

- A reference that ignores the causal mask fails at the attention stage in both C++ and Python.
- A V view pointing at the K columns fails as well.
- With the row-limit check disabled, `transformer_ops_cpp` fails at `reduce sum [2^24,1]` with status 1 and `launch: Logical grid dimension exceeds 32-bit indexing`. This reproduces the review finding, and the check was restored afterwards.

**C++/Python comparison.** `transformer_block_match` runs both applications, causal and bidirectional. It requires identical provider artifact hashes, shapes, seed, causal flag and SHA-256 of all 11 stage outputs. The output SHA-256 is `6452f43f1cdacccedbfb27b51b55b060fd7312fa2d54876d56a5d9e4afdedafa` (causal) and `b689479ea51f82de2f68250ba868c1081d0c7750acca7792f31e7729cd16bb48` (bidirectional). Both were identical across C++ and Python on the recorded M5.

Per-stage GPU durations printed by the applications are single-run informational event values, not a measurement.

## Evidence on this branch

Recorded on an Apple M5, macOS 26, LLVM/Clang 21.1.8, Debug, `-DPARALYN_ENABLE_SPIRV=ON`, in the lane worktree. `ctest -j1` passed 41/41, including these five new tests, both in the first round and after the review fixes (row limit and capability query). The fixes add no new CTest; they extend `transformer_ops_cpp` and `transformer_ops_python`. The benchmark and timing qualification were not run, and `artifacts/` was not modified.

**`transformer_ops_cpp`** covers 89 source-linked GPU commands, with per-kernel counts equal to the observed events. This is 82 from the first round, plus 3 activation-consistency launches and 4 row-limit boundary launches:

| Kernel | Commands |
|---|---|
| batched_matmul | 22 |
| batched_fill | 3 |
| reduce_rows | 22 |
| fill | 2 |
| softmax_rows | 14 |
| layer_norm | 10 |
| activation | 8 |
| bias_activation | 1 |
| add | 7 |

It adds 8 high-level wrapper events. The worst error/bound ratios are:

| Operator | Worst error/bound |
|---|---|
| matmul | 0.513 |
| LayerNorm | 0.661 |
| GELU | 0.551 |
| softmax | 0.120 |
| row sum | 0.012 |

The shapes cover:

- 1-sized dimensions
- 15/16/17/33/47/65/129/257 edges
- zero-size batch, m, n, k, rows and columns
- rows 255/256/257/513/1000 wide
- a 100003-wide row for sum/max/softmax/LayerNorm/GELU
- 1,000,003-element add
- byte offsets with canaries
- rank 0, 1, 3 and 8 adds
- transposed, broadcast and attention head-split views

**`transformer_ops_python`** covers 31 low-level plus 8 high-level GPU launches (39 audited). The 31 include 3 launches, one per advertised activation. Its worst error/bound ratios are matmul 0.372, GELU 0.487, LayerNorm 0.155, softmax 0.111 and row sum 0.002.

**Negative coverage, per operator, in C++:**

- invalid queue, wrong-kind queue, invalid module and released provider
- the artifact loaded without `pr_tensor_operators_load`
- foreign queue and foreign module
- null event output
- record version and size, and nonzero `reserved`
- shape, rank and length mismatches
- outputs overlapping an input, or in the same allocation but a disjoint range
- foreign-context buffers, including on zero-work calls
- non-FP32 or non-contiguous inputs, and negative strides
- self-overlapping strided outputs
- dimension limits and the 2²⁴ LayerNorm column limit
- the row limit: `[2²⁴,1]` and `[2²⁴,0]` for sum, max, softmax and LayerNorm; `[2,2²³,1]` and `[2³²,2²⁰,0]` for batched softmax. Each case asserts `PR_UNSUPPORTED`, the operation, a cleared event and a message naming 16777215.
- the capability record: its contents, version 2 (`PR_UNSUPPORTED`), a short record and a null pointer (`PR_INVALID_ARGUMENT`), and agreement with the accepted activation values 0–63
- invalid scale, ε, causal flag, reduce op and activation values
- causal masking with rows > columns

Each case asserts both the status and `pr_error.operation`. The Python tests cover the same families through the bindings, plus Python-side type errors.

**Block tests.** `transformer_block_cpp`, `_python` and `_match` each audit 15 completed Metal launches per application run, in the exact kernel order.

**Runtime-only build.** A separate `-DPARALYN_BUILD_COMPILER=OFF` Debug configuration, with no LLVM linked, passed **22/22** CTests. That includes the five tensor tests and all five transformer tests, which confirms that the operators and the block need no compiler at run time.

## Limits and remaining work

**Limits of the current operators:**

- Strided layouts are accepted only by batched matmul. Other operators require contiguous layouts; there is no general broadcast or elementwise-view algebra.
- Batched matmul is rank 3 only, with no multi-dimensional batch.
- The row operators always use one 256-lane threadgroup per row. Very short rows waste lanes and very long rows run in a single threadgroup. This is a correctness-first design, not a performance one.
- The row operators accept at most 16,777,215 rows (batch × rows for softmax); see the row limit above. A kernel that tiles several rows per threadgroup, or loops over rows, would lift this limit. It has not been written.
- There is no GELU (erf) variant, no RMSNorm, no dropout, no KV cache and no attention fusion (FlashAttention-style). Mask shapes other than causal are not supported.
- There are no FP16/BF16 variants or other accumulation policies.

**Remaining work:**

- CLI exposure.
- Wheel and installed-package requalification including `tensor.py`.
- A clean-revision evidence capture archived under `artifacts/`.
- The same provider contract on CUDA/HIP, which is blocked on hardware.
- PyTorch custom operators or a device backend.
- Inference over more than one block, including embeddings, tokenization and weights loaded from files.
- Training.
