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
typedef enum pr_activation { PR_ACTIVATION_NONE = 0, PR_ACTIVATION_RELU = 1 } pr_activation;
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
#ifdef __cplusplus
}
#endif
#endif
