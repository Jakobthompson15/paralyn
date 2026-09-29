#ifndef PARALYN_TENSOR_H
#define PARALYN_TENSOR_H
/* Additive, versioned FP32 tensor descriptors and the first native tensor
 * operator provider ("paralyn.msl.tensor", PARALYNX1 public-MSL executable).
 * ABI 1 records and functions in native.h are unchanged. Every record below
 * carries struct_size/version; callers initialize both (or use the contiguous
 * helper) and unknown versions/sizes fail explicitly.
 *
 * This is not cuBLAS, BLAS or PyTorch compatibility, and it runs only where a
 * backend executes the provider's MSL (currently physical Metal devices).
 * There is no CPU fallback. See docs/tensors-matmul.md for numerical,
 * accumulation-order, provider and alias semantics. */
#include "paralyn/native.h"
#ifdef __cplusplus
extern "C" {
#endif
#define PR_TENSOR_VERSION_1 1u
#define PR_TENSOR_MAX_RANK 8u
/* A tensor is an explicit (buffer, byte offset, dtype, rank, shape, strides)
 * record. Strides are in elements, relative to byte_offset. v1 operators
 * require FP32, four-byte-aligned offsets and contiguous row-major layout;
 * other validated layouts are rejected with PR_UNSUPPORTED, never reordered.
 * Each tensor may address at most INT32_MAX elements. */
typedef struct pr_tensor_desc_v1 {
  uint32_t struct_size, version;
  pr_buffer buffer;
  uint64_t byte_offset;
  pr_type dtype; /* PR_F32 only */
  uint32_t rank; /* 0..PR_TENSOR_MAX_RANK; unused dimensions must be zero */
  uint64_t shape[PR_TENSOR_MAX_RANK];
  int64_t strides[PR_TENSOR_MAX_RANK];
} pr_tensor_desc_v1;
/* PR_ACTIVATION_GELU_TANH (additive): the tanh-form GELU
 * 0.5*x*(1 + tanh(0.7978845608028654f*(x + 0.044715f*x*x*x))) with FP32 constants;
 * MSL has no erf. See docs/transformer-operators.md. */
typedef enum pr_activation {
  PR_ACTIVATION_NONE = 0,
  PR_ACTIVATION_RELU = 1,
  PR_ACTIVATION_GELU_TANH = 2
} pr_activation;
/* C[m,n] = op(A)[m,k] * op(B)[k,n]. A is stored [m,k], or [k,m] when
 * transpose_a != 0; B is stored [k,n], or [n,k] when transpose_b != 0.
 * m, n, k are explicit and must agree with every descriptor. FP32 products
 * are accumulated in FP32 in strictly increasing k order, contraction off. */
typedef struct pr_matmul_v1 {
  uint32_t struct_size, version;
  uint64_t m, n, k;
  uint32_t transpose_a, transpose_b;
  const pr_tensor_desc_v1 *a, *b, *c;
} pr_matmul_v1;
/* out[i,j] = act(x[i,j] + bias[j]) for rank-2 x/out [rows, columns] and rank-1
 * bias [columns]; bias == NULL applies only the activation. ReLU keeps values
 * that are not less than zero (NaN and -0.0 pass through) and writes +0.0 otherwise. */
typedef struct pr_bias_activation_v1 {
  uint32_t struct_size, version;
  pr_activation activation;
  uint32_t reserved; /* must be zero */
  const pr_tensor_desc_v1 *x, *bias, *out;
} pr_bias_activation_v1;

/* Fill a version-1 contiguous row-major descriptor (initializes struct_size/version). */
pr_status pr_tensor_desc_contiguous(pr_buffer buffer, uint64_t byte_offset, pr_type dtype,
                                    uint32_t rank, const uint64_t *shape,
                                    pr_tensor_desc_v1 *out);
/* Validate a descriptor against its live buffer. required_bytes (optional) receives
 * the byte extent from byte_offset (zero for an empty tensor). */
pr_status pr_tensor_desc_validate(const pr_tensor_desc_v1 *desc, uint64_t *required_bytes);
/* Serialized PARALYNX1 provider artifact. With data == NULL only *bytes is set;
 * otherwise capacity must be at least the artifact size. */
pr_status pr_tensor_operators_artifact(void *data, uint64_t capacity, uint64_t *bytes);
/* Load the provider artifact through pr_module_load (validation + Metal reflection).
 * The caller owns the returned module handle and releases it with pr_release.
 * Operators accept only module handles returned by this function: any other
 * module, including the same artifact loaded with pr_module_load, is
 * PR_INVALID_ARGUMENT. */
pr_status pr_tensor_operators_load(pr_context context, pr_module *out);
/* Enqueue on the context's ordered queue. The queue, the provider module and
 * every tensor buffer must belong to one context (PR_INVALID_HANDLE /
 * PR_CONTEXT_MISMATCH otherwise), checked even when there is no GPU work.
 * *out receives an event, or 0 when the operation has no GPU work (m == 0 or
 * n == 0; empty bias/activation output). The output must not share an
 * allocation with an input: overlap is PR_INVALID_ARGUMENT; disjoint ranges of
 * the same allocation are PR_UNSUPPORTED in this MSL profile. Inputs may alias. */
pr_status pr_matmul_f32(pr_queue queue, pr_module operators, const pr_matmul_v1 *op,
                        pr_event *out);
pr_status pr_bias_activation_f32(pr_queue queue, pr_module operators,
                                 const pr_bias_activation_v1 *op, pr_event *out);

/* ---- Transformer-block operators (additive, version 1; same provider) ----
 * Numerical contracts, NaN/Inf policy and tolerances: docs/transformer-operators.md.
 * All share the provider identity, handle/context, zero-work and alias rules above.
 *
 * Batched strided matmul: for every b < batch, C[b] = A[b] * B[b] with A rank 3
 * [batch,m,k], B [batch,k,n], C [batch,m,n]. Unlike pr_matmul_f32, all three
 * descriptors may use arbitrary non-negative element strides (transposes and
 * head splits are strided views; nothing is copied), provided each operand's
 * extent is at most INT32_MAX elements. The output layout must not overlap
 * itself (PR_INVALID_ARGUMENT). A batch stride of 0 broadcasts an input.
 * Accumulation is identical to pr_matmul_f32 (FP32, strictly increasing k);
 * k == 0 writes +0.0. batch, m or n == 0: no work, *out = 0. */
typedef struct pr_batched_matmul_v1 {
  uint32_t struct_size, version;
  uint64_t batch, m, n, k;
  const pr_tensor_desc_v1 *a, *b, *c;
} pr_batched_matmul_v1;
typedef enum pr_reduce_op { PR_REDUCE_SUM = 0, PR_REDUCE_MAX = 1 } pr_reduce_op;
/* out[i] = reduce_j x[i,j] for contiguous x [rows, columns] and out [rows].
 * SUM: fixed 256-lane order (lane t adds j = t, t+256, ... from +0.0, then a
 * pairwise tree); the empty sum is +0.0. MAX: NaN if any element is NaN, +0.0
 * preferred over -0.0, -inf for an empty row. */
typedef struct pr_reduce_rows_v1 {
  uint32_t struct_size, version;
  pr_reduce_op op;
  uint32_t reserved; /* must be zero */
  const pr_tensor_desc_v1 *x, *out;
} pr_reduce_rows_v1;
/* Row operators (reduce_rows, softmax_rows, layer_norm) launch one 256-lane
 * threadgroup per row, and a logical grid dimension may not exceed 2^32-1 lanes, so
 * they accept at most PR_TENSOR_ROW_OPERATOR_MAX_ROWS = floor((2^32-1)/256) rows
 * (for softmax, batch*rows). More rows is PR_UNSUPPORTED, decided from the shape
 * before the zero-work decision (so [2^24, 0] is rejected too). */
#define PR_TENSOR_ROW_OPERATOR_MAX_ROWS 16777215u
/* Row softmax over the last dimension of contiguous x/out, rank 2 [rows,columns]
 * or rank 3 [batch,rows,columns]: out = exp(s*x - max) / sum(exp(s*x - max)),
 * with s = scale (finite, > 0) and the max/sum over unmasked columns. causal = 1
 * masks column j of row i when j > i + (columns - rows) and writes +0.0 there;
 * it requires columns >= rows. causal must be 0 or 1. */
typedef struct pr_softmax_rows_v1 {
  uint32_t struct_size, version;
  uint32_t causal;
  float scale;
  const pr_tensor_desc_v1 *x, *out;
} pr_softmax_rows_v1;
/* LayerNorm over the last dimension: out[i,j] = ((x[i,j]-mean_i) * rstd_i) *
 * gamma[j] + beta[j], rstd_i = 1/sqrt(var_i + epsilon), biased (population)
 * two-pass variance. x/out contiguous [rows, columns], gamma/beta [columns],
 * both required; epsilon finite and > 0; columns <= 2^24. */
typedef struct pr_layer_norm_v1 {
  uint32_t struct_size, version;
  float epsilon;
  uint32_t reserved; /* must be zero */
  const pr_tensor_desc_v1 *x, *gamma, *beta, *out;
} pr_layer_norm_v1;
/* Elementwise out = x + y (one FP32 rounding) for contiguous tensors of equal
 * shape and rank 0..8, e.g. a residual connection. */
typedef struct pr_add_v1 {
  uint32_t struct_size, version;
  const pr_tensor_desc_v1 *x, *y, *out;
} pr_add_v1;
pr_status pr_batched_matmul_f32(pr_queue queue, pr_module operators,
                                const pr_batched_matmul_v1 *op, pr_event *out);
pr_status pr_reduce_rows_f32(pr_queue queue, pr_module operators, const pr_reduce_rows_v1 *op,
                             pr_event *out);
pr_status pr_softmax_rows_f32(pr_queue queue, pr_module operators, const pr_softmax_rows_v1 *op,
                              pr_event *out);
pr_status pr_layer_norm_f32(pr_queue queue, pr_module operators, const pr_layer_norm_v1 *op,
                            pr_event *out);
pr_status pr_add_f32(pr_queue queue, pr_module operators, const pr_add_v1 *op, pr_event *out);

/* ---- Provider capability query (additive, version 1) ----
 * Reports what this library's paralyn.msl.tensor provider implements and the
 * shape limits it enforces. It is a library property, not device availability:
 * execution still needs a backend that runs the provider's MSL (physical Metal).
 * A client detects GELU as PR_TENSOR_ACTIVATION_BIT(PR_ACTIVATION_GELU_TANH) in
 * `activations` instead of by trial; a library without this symbol predates the
 * transformer-block operators (GELU shipped together with this query). Callers set
 * struct_size = sizeof(record), version = PR_TENSOR_VERSION_1; unknown versions are
 * PR_UNSUPPORTED and size mismatches PR_INVALID_ARGUMENT. Bits not defined here are
 * reserved and reported as zero by this version. */
#define PR_TENSOR_OP_MATMUL (1ull << 0)          /* pr_matmul_f32 */
#define PR_TENSOR_OP_BIAS_ACTIVATION (1ull << 1) /* pr_bias_activation_f32 */
#define PR_TENSOR_OP_BATCHED_MATMUL (1ull << 2)  /* pr_batched_matmul_f32 */
#define PR_TENSOR_OP_REDUCE_ROWS (1ull << 3)     /* pr_reduce_rows_f32 */
#define PR_TENSOR_OP_SOFTMAX_ROWS (1ull << 4)    /* pr_softmax_rows_f32 */
#define PR_TENSOR_OP_LAYER_NORM (1ull << 5)      /* pr_layer_norm_f32 */
#define PR_TENSOR_OP_ADD (1ull << 6)             /* pr_add_f32 */
#define PR_TENSOR_ACTIVATION_BIT(activation) (1ull << (unsigned)(activation))
#define PR_TENSOR_REDUCE_BIT(reduce) (1ull << (unsigned)(reduce))
typedef struct pr_tensor_operators_capabilities_v1 {
  uint32_t struct_size, version;
  uint64_t operations;             /* PR_TENSOR_OP_* bits */
  uint64_t activations;            /* PR_TENSOR_ACTIVATION_BIT(v) for each accepted pr_activation */
  uint64_t reductions;             /* PR_TENSOR_REDUCE_BIT(v) for each accepted pr_reduce_op */
  uint64_t max_tensor_elements;    /* per descriptor (INT32_MAX) */
  uint64_t max_row_operator_rows;  /* PR_TENSOR_ROW_OPERATOR_MAX_ROWS */
  uint64_t max_layer_norm_columns; /* 2^24 */
} pr_tensor_operators_capabilities_v1;
pr_status pr_tensor_operators_capabilities(pr_tensor_operators_capabilities_v1 *out);
#ifdef __cplusplus
}
#endif
#endif
