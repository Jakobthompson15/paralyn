// FP32 tensor descriptors and the "paralyn.msl.tensor" operator provider.
//
// The provider is an ordinary client of the public C ABI: it serializes a
// PARALYNX1 public-MSL executable, loads it with pr_module_load (full container
// validation, source-signature checks and Metal reflection), creates exact-extent
// views and submits with pr_launch on the caller's ordered queue. It owns no
// private backend path and has no CPU implementation of any operator.
#include "paralyn/tensor.h"
#include "paralyn/executable.hpp"
#include "native_internal.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
constexpr std::uint64_t max_elements = static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
constexpr std::uint32_t tile = 16, group = 256;
// Row operators launch one `group`-lane threadgroup per row; the backends reject a
// logical grid dimension (threadgroups x lanes) above UINT32_MAX, so rows are
// limited to floor((2^32 - 1) / 256) = 16,777,215 (PR_TENSOR_ROW_OPERATOR_MAX_ROWS).
constexpr std::uint64_t max_rows = std::numeric_limits<std::uint32_t>::max() / group;
static_assert(max_rows == PR_TENSOR_ROW_OPERATOR_MAX_ROWS, "row limit disagrees with tensor.h");
constexpr std::uint64_t max_layer_norm_columns = std::uint64_t(1) << 24;

// Kernels are self-contained MSL within the public executable source profile
// (no macros, no backslashes). Accumulation order is part of the contract.
constexpr const char *source = R"MSL(#include <metal_stdlib>
using namespace metal;

// C[m,n] = op(A)[m,k] * op(B)[k,n]. One thread per output element, 16x16 tiles.
// FP32 products are added to one FP32 accumulator in strictly increasing k
// order (0, 1, ..., k-1); padding lanes never enter the sum. The executable's
// numerical policy prepends FP_CONTRACT OFF, so product and sum round separately.
kernel void paralyn_matmul_f32(device const float* a [[buffer(0)]],
                               device const float* b [[buffer(1)]],
                               device float* c [[buffer(2)]],
                               constant uint& m [[buffer(3)]],
                               constant uint& n [[buffer(4)]],
                               constant uint& k [[buffer(5)]],
                               constant uint& transpose_a [[buffer(6)]],
                               constant uint& transpose_b [[buffer(7)]],
                               uint2 lane [[thread_position_in_threadgroup]],
                               uint2 group [[threadgroup_position_in_grid]]) {
  threadgroup float tile_a[16][17];
  threadgroup float tile_b[16][17];
  uint row = group.y * 16 + lane.y;
  uint column = group.x * 16 + lane.x;
  float sum = 0.0f;
  for (uint base = 0; base < k; base += 16) {
    uint ka = base + lane.x;
    float va = 0.0f;
    if (row < m && ka < k) va = transpose_a != 0 ? a[ka * m + row] : a[row * k + ka];
    tile_a[lane.y][lane.x] = va;
    uint kb = base + lane.y;
    float vb = 0.0f;
    if (kb < k && column < n) vb = transpose_b != 0 ? b[column * k + kb] : b[kb * n + column];
    tile_b[lane.y][lane.x] = vb;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint count = min(16u, k - base);
    for (uint p = 0; p < count; ++p) {
      float product = tile_a[lane.y][p] * tile_b[p][lane.x];
      sum = sum + product;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < m && column < n) c[row * n + column] = sum;
}

// Tanh-form GELU with FP32 constants, evaluated in this exact order (each
// operation rounds separately): cube = (x*x)*x; inner = x + c1*cube;
// t = tanh(c0*inner); y = (0.5*x)*(1 + t). MSL provides no erf.
float paralyn_gelu_tanh(float x) {
  float cube = (x * x) * x;
  float inner = x + 0.044715f * cube;
  float t = tanh(0.7978845608028654f * inner);
  return (0.5f * x) * (1.0f + t);
}

float paralyn_activate(float value, uint activation) {
  if (activation == 1u && value < 0.0f) return 0.0f;
  if (activation == 2u) return paralyn_gelu_tanh(value);
  return value;
}

// out[i] = act(x[i] + bias[i % columns]); ReLU replaces only values less than zero.
kernel void paralyn_bias_activation_f32(device const float* x [[buffer(0)]],
                                        device const float* bias [[buffer(1)]],
                                        device float* out [[buffer(2)]],
                                        constant uint& count [[buffer(3)]],
                                        constant uint& columns [[buffer(4)]],
                                        constant uint& activation [[buffer(5)]],
                                        uint i [[thread_position_in_grid]]) {
  if (i < count) {
    out[i] = paralyn_activate(x[i] + bias[i % columns], activation);
  }
}

kernel void paralyn_activation_f32(device const float* x [[buffer(0)]],
                                   device float* out [[buffer(1)]],
                                   constant uint& count [[buffer(2)]],
                                   constant uint& activation [[buffer(3)]],
                                   uint i [[thread_position_in_grid]]) {
  if (i < count) {
    out[i] = paralyn_activate(x[i], activation);
  }
}

// Used for k == 0: the empty sum is +0.0 in every output element.
kernel void paralyn_fill_f32(device float* out [[buffer(0)]],
                             constant uint& count [[buffer(1)]],
                             constant float& value [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
  if (i < count) out[i] = value;
}

// ---- Transformer-block operators (docs/transformer-operators.md) ----

// Batched strided matmul: C[z] = A[z] * B[z] for z = threadgroup z. Element
// strides are explicit, so transposed and head-split views need no copies.
// The per-element order is exactly that of paralyn_matmul_f32.
kernel void paralyn_batched_matmul_f32(device const float* a [[buffer(0)]],
                                       device const float* b [[buffer(1)]],
                                       device float* c [[buffer(2)]],
                                       constant uint& m [[buffer(3)]],
                                       constant uint& n [[buffer(4)]],
                                       constant uint& k [[buffer(5)]],
                                       constant uint& a_batch [[buffer(6)]],
                                       constant uint& a_row [[buffer(7)]],
                                       constant uint& a_col [[buffer(8)]],
                                       constant uint& b_batch [[buffer(9)]],
                                       constant uint& b_row [[buffer(10)]],
                                       constant uint& b_col [[buffer(11)]],
                                       constant uint& c_batch [[buffer(12)]],
                                       constant uint& c_row [[buffer(13)]],
                                       constant uint& c_col [[buffer(14)]],
                                       uint3 lane [[thread_position_in_threadgroup]],
                                       uint3 group [[threadgroup_position_in_grid]]) {
  threadgroup float tile_a[16][17];
  threadgroup float tile_b[16][17];
  uint row = group.y * 16 + lane.y;
  uint column = group.x * 16 + lane.x;
  uint a_base = group.z * a_batch;
  uint b_base = group.z * b_batch;
  float sum = 0.0f;
  for (uint base = 0; base < k; base += 16) {
    uint ka = base + lane.x;
    float va = 0.0f;
    if (row < m && ka < k) va = a[a_base + row * a_row + ka * a_col];
    tile_a[lane.y][lane.x] = va;
    uint kb = base + lane.y;
    float vb = 0.0f;
    if (kb < k && column < n) vb = b[b_base + kb * b_row + column * b_col];
    tile_b[lane.y][lane.x] = vb;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint count = min(16u, k - base);
    for (uint p = 0; p < count; ++p) {
      float product = tile_a[lane.y][p] * tile_b[p][lane.x];
      sum = sum + product;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < m && column < n) c[group.z * c_batch + row * c_row + column * c_col] = sum;
}

// Batched strided fill, used for k == 0 (the empty sum is +0.0 everywhere).
kernel void paralyn_batched_fill_f32(device float* c [[buffer(0)]],
                                     constant uint& m [[buffer(1)]],
                                     constant uint& n [[buffer(2)]],
                                     constant uint& c_batch [[buffer(3)]],
                                     constant uint& c_row [[buffer(4)]],
                                     constant uint& c_col [[buffer(5)]],
                                     constant float& value [[buffer(6)]],
                                     uint3 lane [[thread_position_in_threadgroup]],
                                     uint3 group [[threadgroup_position_in_grid]]) {
  uint row = group.y * 16 + lane.y;
  uint column = group.x * 16 + lane.x;
  if (row < m && column < n) c[group.z * c_batch + row * c_row + column * c_col] = value;
}

// Maximum with a total, order-independent result: NaN if either is NaN,
// +0.0 preferred over -0.0, otherwise the larger value.
float paralyn_max2(float a, float b) {
  if (isnan(a)) return a;
  if (isnan(b)) return b;
  if (b > a) return b;
  if (b == a && signbit(a) && !signbit(b)) return b;
  return a;
}

// Fixed 256-lane pairwise tree over threadgroup partials; every lane of the
// threadgroup must call it. Width w = 128, 64, ..., 1: p[t] = p[t] + p[t + w].
float paralyn_tree_sum(threadgroup float* p, float value, uint t) {
  p[t] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint w = 128; w > 0; w >>= 1) {
    if (t < w) p[t] = p[t] + p[t + w];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float result = p[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

float paralyn_tree_max(threadgroup float* p, float value, uint t) {
  p[t] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint w = 128; w > 0; w >>= 1) {
    if (t < w) p[t] = paralyn_max2(p[t], p[t + w]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float result = p[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

// One 256-lane threadgroup per row. Lane t first combines j = t, t+256, ...
// in increasing j (sum from +0.0, max from -inf), then the fixed tree.
kernel void paralyn_reduce_rows_f32(device const float* x [[buffer(0)]],
                                    device float* out [[buffer(1)]],
                                    constant uint& columns [[buffer(2)]],
                                    constant uint& op [[buffer(3)]],
                                    uint t [[thread_position_in_threadgroup]],
                                    uint row [[threadgroup_position_in_grid]]) {
  threadgroup float partial[256];
  uint base = row * columns;
  float result;
  if (op == 1u) {
    float value = -INFINITY;
    for (uint j = t; j < columns; j += 256) value = paralyn_max2(value, x[base + j]);
    result = paralyn_tree_max(partial, value, t);
  } else {
    float value = 0.0f;
    for (uint j = t; j < columns; j += 256) value = value + x[base + j];
    result = paralyn_tree_sum(partial, value, t);
  }
  if (t == 0) out[row] = result;
}

// Numerically stable row softmax: v = scale * x (rounded), m = max over
// unmasked v, e = exp(v - m), s = tree sum of e, out = e / s. Masked (causal)
// columns write +0.0 and never enter m or s.
kernel void paralyn_softmax_rows_f32(device const float* x [[buffer(0)]],
                                     device float* out [[buffer(1)]],
                                     constant uint& columns [[buffer(2)]],
                                     constant uint& rows [[buffer(3)]],
                                     constant uint& causal [[buffer(4)]],
                                     constant float& scale [[buffer(5)]],
                                     uint t [[thread_position_in_threadgroup]],
                                     uint row [[threadgroup_position_in_grid]]) {
  threadgroup float partial[256];
  uint base = row * columns;
  uint visible = columns;
  if (causal != 0u) visible = (row % rows) + (columns - rows) + 1u;
  float m = -INFINITY;
  for (uint j = t; j < visible; j += 256) m = paralyn_max2(m, scale * x[base + j]);
  m = paralyn_tree_max(partial, m, t);
  float s = 0.0f;
  for (uint j = t; j < visible; j += 256) s = s + exp(scale * x[base + j] - m);
  s = paralyn_tree_sum(partial, s, t);
  for (uint j = t; j < columns; j += 256)
    out[base + j] = j < visible ? exp(scale * x[base + j] - m) / s : 0.0f;
}

// LayerNorm over one row: mean = tree_sum(x) / n; var = tree_sum((x - mean)^2) / n;
// rstd = 1 / sqrt(var + epsilon); out = ((x - mean) * rstd) * gamma + beta.
kernel void paralyn_layer_norm_f32(device const float* x [[buffer(0)]],
                                   device const float* gamma [[buffer(1)]],
                                   device const float* beta [[buffer(2)]],
                                   device float* out [[buffer(3)]],
                                   constant uint& columns [[buffer(4)]],
                                   constant float& epsilon [[buffer(5)]],
                                   uint t [[thread_position_in_threadgroup]],
                                   uint row [[threadgroup_position_in_grid]]) {
  threadgroup float partial[256];
  uint base = row * columns;
  float n = float(columns);
  float s = 0.0f;
  for (uint j = t; j < columns; j += 256) s = s + x[base + j];
  float mean = paralyn_tree_sum(partial, s, t) / n;
  float q = 0.0f;
  for (uint j = t; j < columns; j += 256) {
    float d = x[base + j] - mean;
    q = q + d * d;
  }
  float variance = paralyn_tree_sum(partial, q, t) / n;
  float rstd = 1.0f / sqrt(variance + epsilon);
  for (uint j = t; j < columns; j += 256)
    out[base + j] = ((x[base + j] - mean) * rstd) * gamma[j] + beta[j];
}

// Elementwise residual add: out[i] = x[i] + y[i], one FP32 rounding.
kernel void paralyn_add_f32(device const float* x [[buffer(0)]],
                            device const float* y [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            constant uint& count [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
  if (i < count) out[i] = x[i] + y[i];
}
)MSL";

struct Param {
  const char *name;
  paralyn::ScalarType type;
  bool buffer;
  paralyn::ResourceAccess access;
};
struct Entry {
  const char *name;
  std::array<std::uint32_t, 3> block;
  std::vector<Param> params;
};
const std::vector<Entry> &entries() {
  using paralyn::ResourceAccess;
  using paralyn::ScalarType;
  constexpr auto R = ResourceAccess::Read, RW = ResourceAccess::ReadWrite;
  static const std::vector<Entry> value{
      {"paralyn_matmul_f32", {tile, tile, 1},
       {{"a", ScalarType::F32, true, R}, {"b", ScalarType::F32, true, R},
        {"c", ScalarType::F32, true, RW}, {"m", ScalarType::U32, false, R},
        {"n", ScalarType::U32, false, R}, {"k", ScalarType::U32, false, R},
        {"transpose_a", ScalarType::U32, false, R}, {"transpose_b", ScalarType::U32, false, R}}},
      {"paralyn_bias_activation_f32", {group, 1, 1},
       {{"x", ScalarType::F32, true, R}, {"bias", ScalarType::F32, true, R},
        {"out", ScalarType::F32, true, RW}, {"count", ScalarType::U32, false, R},
        {"columns", ScalarType::U32, false, R}, {"activation", ScalarType::U32, false, R}}},
      {"paralyn_activation_f32", {group, 1, 1},
       {{"x", ScalarType::F32, true, R}, {"out", ScalarType::F32, true, RW},
        {"count", ScalarType::U32, false, R}, {"activation", ScalarType::U32, false, R}}},
      {"paralyn_fill_f32", {group, 1, 1},
       {{"out", ScalarType::F32, true, RW}, {"count", ScalarType::U32, false, R},
        {"value", ScalarType::F32, false, R}}},
      {"paralyn_batched_matmul_f32", {tile, tile, 1},
       {{"a", ScalarType::F32, true, R}, {"b", ScalarType::F32, true, R},
        {"c", ScalarType::F32, true, RW}, {"m", ScalarType::U32, false, R},
        {"n", ScalarType::U32, false, R}, {"k", ScalarType::U32, false, R},
        {"a_batch", ScalarType::U32, false, R}, {"a_row", ScalarType::U32, false, R},
        {"a_col", ScalarType::U32, false, R}, {"b_batch", ScalarType::U32, false, R},
        {"b_row", ScalarType::U32, false, R}, {"b_col", ScalarType::U32, false, R},
        {"c_batch", ScalarType::U32, false, R}, {"c_row", ScalarType::U32, false, R},
        {"c_col", ScalarType::U32, false, R}}},
      {"paralyn_batched_fill_f32", {tile, tile, 1},
       {{"c", ScalarType::F32, true, RW}, {"m", ScalarType::U32, false, R},
        {"n", ScalarType::U32, false, R}, {"c_batch", ScalarType::U32, false, R},
        {"c_row", ScalarType::U32, false, R}, {"c_col", ScalarType::U32, false, R},
        {"value", ScalarType::F32, false, R}}},
      {"paralyn_reduce_rows_f32", {group, 1, 1},
       {{"x", ScalarType::F32, true, R}, {"out", ScalarType::F32, true, RW},
        {"columns", ScalarType::U32, false, R}, {"op", ScalarType::U32, false, R}}},
      {"paralyn_softmax_rows_f32", {group, 1, 1},
       {{"x", ScalarType::F32, true, R}, {"out", ScalarType::F32, true, RW},
        {"columns", ScalarType::U32, false, R}, {"rows", ScalarType::U32, false, R},
        {"causal", ScalarType::U32, false, R}, {"scale", ScalarType::F32, false, R}}},
      {"paralyn_layer_norm_f32", {group, 1, 1},
       {{"x", ScalarType::F32, true, R}, {"gamma", ScalarType::F32, true, R},
        {"beta", ScalarType::F32, true, R}, {"out", ScalarType::F32, true, RW},
        {"columns", ScalarType::U32, false, R}, {"epsilon", ScalarType::F32, false, R}}},
      {"paralyn_add_f32", {group, 1, 1},
       {{"x", ScalarType::F32, true, R}, {"y", ScalarType::F32, true, R},
        {"out", ScalarType::F32, true, RW}, {"count", ScalarType::U32, false, R}}}};
  return value;
}
const std::vector<unsigned char> &artifact() {
  static const std::vector<unsigned char> bytes = [] {
    paralyn::ExecutableModule module;
    module.producer = "paralyn.msl.tensor";
    module.producer_version = "0.0.1";
    module.source_name = "paralyn/tensor_f32.metal";
    module.source = source;
    module.source_sha256 = paralyn::source_sha256(module.source);
    for (const auto &e : entries()) {
      paralyn::ExecutableEntry entry;
      entry.name = e.name;
      entry.required_block = e.block;
      std::uint32_t binding = 0;
      for (const auto &p : e.params)
        entry.parameters.push_back({p.name, p.type, p.buffer, p.access, binding++, 4, 4});
      module.entries.push_back(std::move(entry));
    }
    return paralyn::serialize_executable(module);
  }();
  return bytes;
}

struct Fail {
  pr_status code;
  std::string message;
};
[[noreturn]] void fail(pr_status code, const std::string &message) { throw Fail{code, message}; }
void require(bool condition, pr_status code, const char *message) {
  if (!condition)
    fail(code, message);
}
// Captures the inner public call's structured error before cleanup can clear it.
void call(pr_status status) {
  if (status == PR_SUCCESS)
    return;
  pr_error inner{};
  pr_last_error(&inner);
  fail(status, std::string(inner.operation) + ": " + inner.message);
}
template <class F> pr_status boundary(const char *operation, F &&fn) noexcept {
  try {
    fn();
    paralyn_native_clear_error(); // ABI 1 convention: success clears pr_last_error
    return PR_SUCCESS;
  } catch (const Fail &f) {
    return paralyn_native_report_error(f.code, operation, f.message.c_str());
  } catch (const std::bad_alloc &) {
    return paralyn_native_report_error(PR_OUT_OF_MEMORY, operation, "Allocation failed");
  } catch (const std::exception &e) {
    return paralyn_native_report_error(PR_INTERNAL_ERROR, operation, e.what());
  } catch (...) {
    return paralyn_native_report_error(PR_INTERNAL_ERROR, operation, "Unknown operator failure");
  }
}
// Temporary provider handles (views, kernels). Launch retains what it needs.
struct Owned {
  pr_handle handle = 0;
  Owned() = default;
  Owned(const Owned &) = delete;
  Owned &operator=(const Owned &) = delete;
  ~Owned() {
    if (handle)
      pr_release(handle);
  }
};

struct Layout {
  std::uint64_t count = 0, bytes = 0;
  bool contiguous = true;
};
bool mul(std::uint64_t a, std::uint64_t b, std::uint64_t &out) {
  if (a && b > std::numeric_limits<std::uint64_t>::max() / a)
    return false;
  out = a * b;
  return true;
}
Layout validate(const pr_tensor_desc_v1 *d, const char *label) {
  const std::string name(label);
  require(d, PR_INVALID_ARGUMENT, "Missing tensor descriptor");
  require(d->version == PR_TENSOR_VERSION_1, PR_UNSUPPORTED, "Unsupported tensor descriptor version");
  require(d->struct_size == sizeof(pr_tensor_desc_v1), PR_INVALID_ARGUMENT,
          "Tensor descriptor size mismatch");
  if (d->dtype != PR_F32)
    fail(PR_UNSUPPORTED, name + ": only PR_F32 tensors are supported");
  if (d->rank > PR_TENSOR_MAX_RANK)
    fail(PR_INVALID_ARGUMENT, name + ": rank exceeds PR_TENSOR_MAX_RANK");
  if (d->byte_offset % 4)
    fail(PR_INVALID_ARGUMENT, name + ": byte offset must be four-byte aligned");
  Layout layout;
  layout.count = 1;
  for (std::uint32_t i = d->rank; i < PR_TENSOR_MAX_RANK; ++i)
    if (d->shape[i] || d->strides[i])
      fail(PR_INVALID_ARGUMENT, name + ": unused dimensions must be zero");
  for (std::uint32_t i = 0; i < d->rank; ++i) {
    if (d->strides[i] < 0)
      fail(PR_UNSUPPORTED, name + ": negative strides are unsupported");
    if (!mul(layout.count, d->shape[i], layout.count))
      fail(PR_OUT_OF_BOUNDS, name + ": element count overflows");
  }
  if (layout.count > max_elements)
    fail(PR_UNSUPPORTED, name + ": tensors support at most INT32_MAX elements");
  std::uint64_t expected = 1, last = 0;
  for (std::uint32_t r = d->rank; r-- > 0;) {
    if (d->shape[r] != 1 && static_cast<std::uint64_t>(d->strides[r]) != expected)
      layout.contiguous = false;
    expected *= d->shape[r];
    std::uint64_t span = 0;
    if (layout.count && (!mul(d->shape[r] - 1, static_cast<std::uint64_t>(d->strides[r]), span) ||
                         span > std::numeric_limits<std::uint64_t>::max() - last))
      fail(PR_OUT_OF_BOUNDS, name + ": stride extent overflows");
    if (layout.count)
      last += span;
  }
  if (!layout.count)
    layout.contiguous = true;
  if (layout.count) {
    if (last >= std::numeric_limits<std::uint64_t>::max() / 4)
      fail(PR_OUT_OF_BOUNDS, name + ": byte extent overflows");
    layout.bytes = (last + 1) * 4;
  }
  std::uint64_t size = 0;
  call(pr_buffer_size(d->buffer, &size));
  if (d->byte_offset > size || layout.bytes > size - d->byte_offset)
    fail(PR_OUT_OF_BOUNDS, name + ": tensor extent exceeds its buffer");
  return layout;
}
Layout operand(const pr_tensor_desc_v1 *d, const char *label, std::uint32_t rank) {
  auto layout = validate(d, label);
  if (d->rank != rank)
    fail(PR_INVALID_ARGUMENT, std::string(label) + ": expected rank " + std::to_string(rank));
  if (!layout.contiguous)
    fail(PR_UNSUPPORTED, std::string(label) + ": v1 operators require contiguous row-major layout");
  return layout;
}
// Strided operands (batched matmul only): any validated non-negative layout whose
// extent fits the 32-bit element indexing of the kernels.
Layout strided_operand(const pr_tensor_desc_v1 *d, const char *label, std::uint32_t rank) {
  auto layout = validate(d, label);
  if (d->rank != rank)
    fail(PR_INVALID_ARGUMENT, std::string(label) + ": expected rank " + std::to_string(rank));
  if (layout.bytes / 4 > max_elements)
    fail(PR_UNSUPPORTED, std::string(label) + ": strided extent exceeds INT32_MAX elements");
  return layout;
}
// A written layout must address every element at a distinct location. Sufficient
// check: ordered by stride, each dimension of extent > 1 steps past the whole span
// of the smaller ones. Layouts it cannot prove injective are rejected.
void injective(const pr_tensor_desc_v1 *d, const char *label) {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> dims; // (stride, extent)
  for (std::uint32_t i = 0; i < d->rank; ++i) {
    if (!d->shape[i])
      return; // empty tensors write nothing
    if (d->shape[i] > 1)
      dims.emplace_back(static_cast<std::uint64_t>(d->strides[i]), d->shape[i]);
  }
  std::sort(dims.begin(), dims.end());
  std::uint64_t span = 0; // max offset reachable by the smaller dimensions
  for (const auto &dim : dims) {
    if (dim.first <= span)
      fail(PR_INVALID_ARGUMENT, std::string(label) + ": output layout overlaps itself");
    span += (dim.second - 1) * dim.first; // bounded by the validated extent
  }
}
std::string shape(const pr_tensor_desc_v1 *d) {
  std::string s = "[";
  for (std::uint32_t i = 0; i < d->rank; ++i)
    s += (i ? "," : "") + std::to_string(d->shape[i]);
  return s + "]";
}
void expect(const pr_tensor_desc_v1 *d, const char *label, std::uint64_t d0, std::uint64_t d1) {
  if (d->shape[0] != d0 || d->shape[1] != d1)
    fail(PR_INVALID_ARGUMENT, std::string(label) + " shape " + shape(d) + " does not match expected [" +
                                  std::to_string(d0) + "," + std::to_string(d1) + "]");
}
// Output/input allocation rules; buffer handles map one-to-one to allocations.
void no_alias(const pr_tensor_desc_v1 *out, const Layout &o, const pr_tensor_desc_v1 *in,
              const Layout &i, const char *label) {
  if (!in || out->buffer != in->buffer)
    return;
  const bool overlap = o.bytes && i.bytes && out->byte_offset < in->byte_offset + i.bytes &&
                       in->byte_offset < out->byte_offset + o.bytes;
  if (overlap)
    fail(PR_INVALID_ARGUMENT, std::string("Output overlaps input ") + label +
                                  "; in-place tensor operations are not supported");
  fail(PR_UNSUPPORTED, std::string("Output shares an allocation with input ") + label +
                           "; the MSL provider profile requires distinct writable allocations");
}
// Provider identity: exactly the module handles returned by pr_tensor_operators_load.
// Handle identities never recycle, so membership cannot pass to an unrelated module;
// released handles fail as invalid before membership is consulted.
std::mutex registry_mutex;
std::unordered_set<pr_module> &providers() {
  static std::unordered_set<pr_module> value;
  return value;
}
bool is_provider(pr_module module) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  return providers().count(module) != 0;
}
const void *owner(pr_handle handle, paralyn_native_kind kind, const char *label) {
  const void *context = nullptr;
  const auto status = paralyn_native_handle_context(handle, kind, &context);
  if (status != PR_SUCCESS) {
    pr_error inner{};
    pr_last_error(&inner);
    fail(status, std::string(label) + ": " + inner.message);
  }
  return context;
}
// Validates the queue and provider handles and one-context ownership of every tensor
// before any work is submitted or skipped, with the same statuses as pr_launch.
void same_context(pr_queue queue, pr_module operators,
                  std::initializer_list<std::pair<const pr_tensor_desc_v1 *, const char *>> tensors) {
  const void *context = owner(queue, PARALYN_NATIVE_QUEUE, "queue");
  if (owner(operators, PARALYN_NATIVE_MODULE, "operators") != context)
    fail(PR_CONTEXT_MISMATCH, "Queue and operator module belong to different contexts");
  if (!is_provider(operators))
    fail(PR_INVALID_ARGUMENT,
         "Module is not the paralyn.msl.tensor provider (load it with pr_tensor_operators_load)");
  for (const auto &t : tensors)
    if (t.first && owner(t.first->buffer, PARALYN_NATIVE_BUFFER, t.second) != context)
      fail(PR_CONTEXT_MISMATCH, std::string(t.second) + ": tensor buffer belongs to a different context");
}
void kernel(pr_module operators, const char *name, Owned &out) {
  const auto &list = entries();
  const Entry *entry = nullptr;
  for (const auto &e : list)
    if (!std::strcmp(e.name, name))
      entry = &e;
  call(pr_module_kernel(operators, name, &out.handle));
  std::uint32_t count = 0;
  call(pr_kernel_parameter_count(out.handle, &count));
  bool same = entry && count == entry->params.size();
  for (std::uint32_t i = 0; same && i < count; ++i) {
    pr_parameter_info info{};
    call(pr_kernel_parameter(out.handle, i, &info));
    const auto &p = entry->params[i];
    same = !std::strcmp(info.name, p.name) && info.is_buffer == static_cast<std::uint32_t>(p.buffer) &&
           info.type == (p.type == paralyn::ScalarType::F32 ? PR_F32 : PR_U32);
  }
  require(same, PR_INVALID_ARGUMENT,
          "Module is not the paralyn.msl.tensor provider (load it with pr_tensor_operators_load)");
}
void view(const pr_tensor_desc_v1 *d, const Layout &l, pr_access access, Owned &out) {
  call(pr_view_create(d->buffer, d->byte_offset, l.bytes, 4, access, &out.handle));
}
pr_argument buffer(const Owned &v) {
  pr_argument a{};
  a.type = PR_BUFFER;
  a.view = v.handle;
  return a;
}
pr_argument u32(std::uint64_t value) {
  pr_argument a{};
  a.type = PR_U32;
  a.u32 = static_cast<std::uint32_t>(value);
  return a;
}
pr_argument f32(float value) {
  pr_argument a{};
  a.type = PR_F32;
  a.f32 = value;
  return a;
}
void expect_dims(const pr_tensor_desc_v1 *d, const char *label,
                 std::initializer_list<std::uint64_t> dims) {
  std::uint32_t i = 0;
  bool same = d->rank == dims.size();
  std::string text = "[";
  for (auto v : dims) {
    same = same && d->shape[i] == v;
    text += (i++ ? "," : "") + std::to_string(v);
  }
  if (!same)
    fail(PR_INVALID_ARGUMENT, std::string(label) + " shape " + shape(d) + " does not match expected " +
                                  text + "]");
}
template <class Record> void header(const Record *op, const char *name) {
  require(op, PR_INVALID_ARGUMENT, (std::string("Missing ") + name + " record").c_str());
  require(op->version == PR_TENSOR_VERSION_1, PR_UNSUPPORTED,
          (std::string("Unsupported ") + name + " record version").c_str());
  require(op->struct_size == sizeof(Record), PR_INVALID_ARGUMENT,
          (std::string(name) + " record size mismatch").c_str());
}
// Stride as a kernel argument; dimensions of extent <= 1 never multiply it.
pr_argument stride(const pr_tensor_desc_v1 *d, std::uint32_t i) {
  return u32(d->shape[i] > 1 ? static_cast<std::uint64_t>(d->strides[i]) : 0);
}
pr_dim3 linear(std::uint64_t count) {
  return {static_cast<std::uint32_t>((count + group - 1) / group), 1, 1};
}
// Shape-level row limit of the one-threadgroup-per-row operators; checked before the
// zero-work decision so the contract does not depend on whether columns == 0.
void row_limit(std::uint64_t rows, const char *operation) {
  if (rows > max_rows)
    fail(PR_UNSUPPORTED, std::string(operation) + " supports at most " + std::to_string(max_rows) +
                             " rows (one 256-lane threadgroup per row; got " + std::to_string(rows) + ")");
}
} // namespace

extern "C" {
pr_status pr_tensor_desc_contiguous(pr_buffer buffer, uint64_t byte_offset, pr_type dtype,
                                    uint32_t rank, const uint64_t *dims, pr_tensor_desc_v1 *out) {
  return boundary("tensor_desc_contiguous", [&] {
    require(out, PR_INVALID_ARGUMENT, "Missing tensor descriptor output");
    require(rank <= PR_TENSOR_MAX_RANK, PR_INVALID_ARGUMENT, "Rank exceeds PR_TENSOR_MAX_RANK");
    require(dims || !rank, PR_INVALID_ARGUMENT, "Missing tensor shape");
    pr_tensor_desc_v1 d{};
    d.struct_size = sizeof(d);
    d.version = PR_TENSOR_VERSION_1;
    d.buffer = buffer;
    d.byte_offset = byte_offset;
    d.dtype = dtype;
    d.rank = rank;
    std::uint64_t stride = 1;
    for (std::uint32_t r = rank; r-- > 0;) {
      d.shape[r] = dims[r];
      require(stride <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()),
              PR_OUT_OF_BOUNDS, "Contiguous stride overflows");
      d.strides[r] = static_cast<std::int64_t>(stride);
      if (dims[r] && !mul(stride, dims[r], stride))
        fail(PR_OUT_OF_BOUNDS, "Contiguous stride overflows");
    }
    *out = d;
  });
}
pr_status pr_tensor_desc_validate(const pr_tensor_desc_v1 *desc, uint64_t *required_bytes) {
  return boundary("tensor_desc_validate", [&] {
    const auto layout = validate(desc, "tensor");
    if (required_bytes)
      *required_bytes = layout.bytes;
  });
}
pr_status pr_tensor_operators_artifact(void *data, uint64_t capacity, uint64_t *bytes) {
  return boundary("tensor_operators_artifact", [&] {
    require(bytes, PR_INVALID_ARGUMENT, "Missing artifact size output");
    const auto &value = artifact();
    *bytes = value.size();
    if (data) {
      require(capacity >= value.size(), PR_OUT_OF_BOUNDS, "Artifact capacity is too small");
      std::memcpy(data, value.data(), value.size());
    }
  });
}
pr_status pr_tensor_operators_load(pr_context context, pr_module *out) {
  return boundary("tensor_operators_load", [&] {
    require(out, PR_INVALID_ARGUMENT, "Missing module output");
    *out = 0;
    const auto &value = artifact();
    call(pr_module_load(context, value.data(), value.size(), out));
    try {
      std::lock_guard<std::mutex> lock(registry_mutex);
      providers().insert(*out);
    } catch (...) {
      pr_release(*out);
      *out = 0;
      throw;
    }
  });
}
pr_status pr_tensor_operators_capabilities(pr_tensor_operators_capabilities_v1 *out) {
  return boundary("tensor_operators_capabilities", [&] {
    require(out, PR_INVALID_ARGUMENT, "Missing capability record");
    require(out->version == PR_TENSOR_VERSION_1, PR_UNSUPPORTED,
            "Unsupported tensor capability record version");
    require(out->struct_size == sizeof(pr_tensor_operators_capabilities_v1), PR_INVALID_ARGUMENT,
            "Tensor capability record size mismatch");
    pr_tensor_operators_capabilities_v1 caps{};
    caps.struct_size = sizeof(caps);
    caps.version = PR_TENSOR_VERSION_1;
    caps.operations = PR_TENSOR_OP_MATMUL | PR_TENSOR_OP_BIAS_ACTIVATION | PR_TENSOR_OP_BATCHED_MATMUL |
                      PR_TENSOR_OP_REDUCE_ROWS | PR_TENSOR_OP_SOFTMAX_ROWS | PR_TENSOR_OP_LAYER_NORM |
                      PR_TENSOR_OP_ADD;
    caps.activations = PR_TENSOR_ACTIVATION_BIT(PR_ACTIVATION_NONE) |
                       PR_TENSOR_ACTIVATION_BIT(PR_ACTIVATION_RELU) |
                       PR_TENSOR_ACTIVATION_BIT(PR_ACTIVATION_GELU_TANH);
    caps.reductions = PR_TENSOR_REDUCE_BIT(PR_REDUCE_SUM) | PR_TENSOR_REDUCE_BIT(PR_REDUCE_MAX);
    caps.max_tensor_elements = max_elements;
    caps.max_row_operator_rows = max_rows;
    caps.max_layer_norm_columns = max_layer_norm_columns;
    *out = caps;
  });
}
pr_status pr_matmul_f32(pr_queue queue, pr_module operators, const pr_matmul_v1 *op,
                        pr_event *out) {
  return boundary("matmul_f32", [&] {
    require(out, PR_INVALID_ARGUMENT, "Missing event output");
    *out = 0;
    require(op, PR_INVALID_ARGUMENT, "Missing matmul record");
    require(op->version == PR_TENSOR_VERSION_1, PR_UNSUPPORTED, "Unsupported matmul record version");
    require(op->struct_size == sizeof(pr_matmul_v1), PR_INVALID_ARGUMENT, "Matmul record size mismatch");
    require(op->transpose_a <= 1 && op->transpose_b <= 1, PR_INVALID_ARGUMENT,
            "Transpose flags must be 0 or 1");
    require(op->a && op->b && op->c, PR_INVALID_ARGUMENT, "Missing matmul tensor");
    const auto a = operand(op->a, "A", 2), b = operand(op->b, "B", 2), c = operand(op->c, "C", 2);
    const auto m = op->m, n = op->n, k = op->k;
    require(m <= max_elements && n <= max_elements && k <= max_elements, PR_UNSUPPORTED,
            "Matmul dimensions must not exceed INT32_MAX");
    if (op->transpose_a) expect(op->a, "A (transposed storage [k,m])", k, m);
    else expect(op->a, "A [m,k]", m, k);
    if (op->transpose_b) expect(op->b, "B (transposed storage [n,k])", n, k);
    else expect(op->b, "B [k,n]", k, n);
    expect(op->c, "C [m,n]", m, n);
    no_alias(op->c, c, op->a, a, "A");
    no_alias(op->c, c, op->b, b, "B");
    same_context(queue, operators, {{op->a, "A"}, {op->b, "B"}, {op->c, "C"}});
    Owned fn;
    kernel(operators, k ? "paralyn_matmul_f32" : "paralyn_fill_f32", fn);
    if (!m || !n)
      return; // No output elements: nothing is submitted and no event is invented.
    Owned cv;
    view(op->c, c, PR_READ_WRITE, cv);
    if (!k) {
      const std::array<pr_argument, 3> args{buffer(cv), u32(m * n), f32(0.0f)};
      call(pr_launch(queue, fn.handle, linear(m * n), {group, 1, 1}, args.data(), 3, out));
      return;
    }
    Owned av, bv;
    view(op->a, a, PR_READ, av);
    view(op->b, b, PR_READ, bv);
    const std::array<pr_argument, 8> args{buffer(av), buffer(bv), buffer(cv), u32(m), u32(n), u32(k),
                                          u32(op->transpose_a), u32(op->transpose_b)};
    const pr_dim3 grid{static_cast<std::uint32_t>((n + tile - 1) / tile),
                       static_cast<std::uint32_t>((m + tile - 1) / tile), 1};
    call(pr_launch(queue, fn.handle, grid, {tile, tile, 1}, args.data(), 8, out));
  });
}
pr_status pr_bias_activation_f32(pr_queue queue, pr_module operators,
                                 const pr_bias_activation_v1 *op, pr_event *out) {
  return boundary("bias_activation_f32", [&] {
    require(out, PR_INVALID_ARGUMENT, "Missing event output");
    *out = 0;
    require(op, PR_INVALID_ARGUMENT, "Missing bias/activation record");
    require(op->version == PR_TENSOR_VERSION_1, PR_UNSUPPORTED,
            "Unsupported bias/activation record version");
    require(op->struct_size == sizeof(pr_bias_activation_v1), PR_INVALID_ARGUMENT,
            "Bias/activation record size mismatch");
    require(op->activation == PR_ACTIVATION_NONE || op->activation == PR_ACTIVATION_RELU ||
                op->activation == PR_ACTIVATION_GELU_TANH,
            PR_INVALID_ARGUMENT, "Unknown activation");
    require(op->reserved == 0, PR_INVALID_ARGUMENT, "Reserved field must be zero");
    require(op->x && op->out, PR_INVALID_ARGUMENT, "Missing input or output tensor");
    const auto x = operand(op->x, "x", 2), o = operand(op->out, "out", 2);
    const auto rows = op->x->shape[0], columns = op->x->shape[1];
    expect(op->out, "out", rows, columns);
    Layout bias;
    if (op->bias) {
      bias = operand(op->bias, "bias", 1);
      if (op->bias->shape[0] != columns)
        fail(PR_INVALID_ARGUMENT, "bias shape " + shape(op->bias) + " does not match expected [" +
                                      std::to_string(columns) + "]");
    }
    no_alias(op->out, o, op->x, x, "x");
    no_alias(op->out, o, op->bias, bias, "bias");
    same_context(queue, operators, {{op->x, "x"}, {op->bias, "bias"}, {op->out, "out"}});
    Owned fn;
    kernel(operators, op->bias ? "paralyn_bias_activation_f32" : "paralyn_activation_f32", fn);
    const auto count = rows * columns;
    if (!count)
      return;
    Owned xv, ov, bv;
    view(op->x, x, PR_READ, xv);
    view(op->out, o, PR_READ_WRITE, ov);
    if (op->bias) {
      view(op->bias, bias, PR_READ, bv);
      const std::array<pr_argument, 6> args{buffer(xv), buffer(bv), buffer(ov), u32(count),
                                            u32(columns), u32(op->activation)};
      call(pr_launch(queue, fn.handle, linear(count), {group, 1, 1}, args.data(), 6, out));
    } else {
      const std::array<pr_argument, 4> args{buffer(xv), buffer(ov), u32(count), u32(op->activation)};
      call(pr_launch(queue, fn.handle, linear(count), {group, 1, 1}, args.data(), 4, out));
    }
  });
}
// ---- Transformer-block operators (docs/transformer-operators.md) ----
pr_status pr_batched_matmul_f32(pr_queue queue, pr_module operators,
                                const pr_batched_matmul_v1 *op, pr_event *out) {
  return boundary("batched_matmul_f32", [&] {
    require(out, PR_INVALID_ARGUMENT, "Missing event output");
    *out = 0;
    header(op, "batched matmul");
    require(op->a && op->b && op->c, PR_INVALID_ARGUMENT, "Missing batched matmul tensor");
    const auto a = strided_operand(op->a, "A", 3), b = strided_operand(op->b, "B", 3),
               c = strided_operand(op->c, "C", 3);
    const auto batch = op->batch, m = op->m, n = op->n, k = op->k;
    require(batch <= max_elements && m <= max_elements && n <= max_elements && k <= max_elements,
            PR_UNSUPPORTED, "Batched matmul dimensions must not exceed INT32_MAX");
    expect_dims(op->a, "A [batch,m,k]", {batch, m, k});
    expect_dims(op->b, "B [batch,k,n]", {batch, k, n});
    expect_dims(op->c, "C [batch,m,n]", {batch, m, n});
    injective(op->c, "C");
    no_alias(op->c, c, op->a, a, "A");
    no_alias(op->c, c, op->b, b, "B");
    same_context(queue, operators, {{op->a, "A"}, {op->b, "B"}, {op->c, "C"}});
    Owned fn;
    kernel(operators, k ? "paralyn_batched_matmul_f32" : "paralyn_batched_fill_f32", fn);
    if (!batch || !m || !n)
      return; // No output elements: nothing is submitted and no event is invented.
    Owned cv;
    view(op->c, c, PR_READ_WRITE, cv);
    const pr_dim3 grid{static_cast<std::uint32_t>((n + tile - 1) / tile),
                       static_cast<std::uint32_t>((m + tile - 1) / tile),
                       static_cast<std::uint32_t>(batch)};
    if (!k) { // The empty sum: +0.0 in every (strided) output element, on the GPU.
      const std::array<pr_argument, 7> args{buffer(cv),       u32(m),           u32(n),
                                            stride(op->c, 0), stride(op->c, 1), stride(op->c, 2),
                                            f32(0.0f)};
      call(pr_launch(queue, fn.handle, grid, {tile, tile, 1}, args.data(), 7, out));
      return;
    }
    Owned av, bv;
    view(op->a, a, PR_READ, av);
    view(op->b, b, PR_READ, bv);
    const std::array<pr_argument, 15> args{
        buffer(av),       buffer(bv),       buffer(cv),       u32(m),
        u32(n),           u32(k),           stride(op->a, 0), stride(op->a, 1),
        stride(op->a, 2), stride(op->b, 0), stride(op->b, 1), stride(op->b, 2),
        stride(op->c, 0), stride(op->c, 1), stride(op->c, 2)};
    call(pr_launch(queue, fn.handle, grid, {tile, tile, 1}, args.data(), 15, out));
  });
}
pr_status pr_reduce_rows_f32(pr_queue queue, pr_module operators, const pr_reduce_rows_v1 *op,
                             pr_event *out) {
  return boundary("reduce_rows_f32", [&] {
    require(out, PR_INVALID_ARGUMENT, "Missing event output");
    *out = 0;
    header(op, "row reduction");
    require(op->op == PR_REDUCE_SUM || op->op == PR_REDUCE_MAX, PR_INVALID_ARGUMENT,
            "Unknown reduction");
    require(op->reserved == 0, PR_INVALID_ARGUMENT, "Reserved field must be zero");
    require(op->x && op->out, PR_INVALID_ARGUMENT, "Missing input or output tensor");
    const auto x = operand(op->x, "x", 2), o = operand(op->out, "out", 1);
    const auto rows = op->x->shape[0], columns = op->x->shape[1];
    expect_dims(op->out, "out [rows]", {rows});
    row_limit(rows, "row reduction");
    no_alias(op->out, o, op->x, x, "x");
    same_context(queue, operators, {{op->x, "x"}, {op->out, "out"}});
    Owned fn;
    kernel(operators, columns ? "paralyn_reduce_rows_f32" : "paralyn_fill_f32", fn);
    if (!rows)
      return;
    Owned ov;
    view(op->out, o, PR_READ_WRITE, ov);
    if (!columns) { // Empty rows: +0.0 (sum) or -inf (max), written on the GPU.
      const float identity =
          op->op == PR_REDUCE_MAX ? -std::numeric_limits<float>::infinity() : 0.0f;
      const std::array<pr_argument, 3> args{buffer(ov), u32(rows), f32(identity)};
      call(pr_launch(queue, fn.handle, linear(rows), {group, 1, 1}, args.data(), 3, out));
      return;
    }
    Owned xv;
    view(op->x, x, PR_READ, xv);
    const std::array<pr_argument, 4> args{buffer(xv), buffer(ov), u32(columns), u32(op->op)};
    call(pr_launch(queue, fn.handle, {static_cast<std::uint32_t>(rows), 1, 1}, {group, 1, 1},
                   args.data(), 4, out));
  });
}
pr_status pr_softmax_rows_f32(pr_queue queue, pr_module operators, const pr_softmax_rows_v1 *op,
                              pr_event *out) {
  return boundary("softmax_rows_f32", [&] {
    require(out, PR_INVALID_ARGUMENT, "Missing event output");
    *out = 0;
    header(op, "softmax");
    require(op->causal <= 1, PR_INVALID_ARGUMENT, "causal must be 0 or 1");
    require(std::isfinite(op->scale) && op->scale > 0.0f, PR_INVALID_ARGUMENT,
            "scale must be finite and greater than zero");
    require(op->x && op->out, PR_INVALID_ARGUMENT, "Missing input or output tensor");
    const auto rank = op->x->rank;
    require(rank == 2 || rank == 3, PR_INVALID_ARGUMENT,
            "x: softmax input must have rank 2 [rows,columns] or rank 3 [batch,rows,columns]");
    const auto x = operand(op->x, "x", rank), o = operand(op->out, "out", rank);
    for (std::uint32_t i = 0; i < rank; ++i)
      if (op->out->shape[i] != op->x->shape[i])
        fail(PR_INVALID_ARGUMENT,
             "out shape " + shape(op->out) + " does not match x shape " + shape(op->x));
    const auto columns = op->x->shape[rank - 1], rows = op->x->shape[rank - 2];
    if (op->causal && columns < rows)
      fail(PR_INVALID_ARGUMENT,
           "causal softmax requires columns >= rows (got " + shape(op->x) + ")");
    std::uint64_t all_rows = rows; // batch * rows for rank 3 (overflow is already rejected by validate)
    if (rank == 3 && !mul(all_rows, op->x->shape[0], all_rows))
      all_rows = std::numeric_limits<std::uint64_t>::max();
    row_limit(all_rows, "softmax (batch * rows)");
    no_alias(op->out, o, op->x, x, "x");
    same_context(queue, operators, {{op->x, "x"}, {op->out, "out"}});
    Owned fn;
    kernel(operators, "paralyn_softmax_rows_f32", fn);
    if (!x.count)
      return;
    const auto total_rows = x.count / columns;
    Owned xv, ov;
    view(op->x, x, PR_READ, xv);
    view(op->out, o, PR_READ_WRITE, ov);
    const std::array<pr_argument, 6> args{buffer(xv), buffer(ov),      u32(columns),
                                          u32(rows),  u32(op->causal), f32(op->scale)};
    call(pr_launch(queue, fn.handle, {static_cast<std::uint32_t>(total_rows), 1, 1},
                   {group, 1, 1}, args.data(), 6, out));
  });
}
pr_status pr_layer_norm_f32(pr_queue queue, pr_module operators, const pr_layer_norm_v1 *op,
                            pr_event *out) {
  return boundary("layer_norm_f32", [&] {
    require(out, PR_INVALID_ARGUMENT, "Missing event output");
    *out = 0;
    header(op, "layer norm");
    require(std::isfinite(op->epsilon) && op->epsilon > 0.0f, PR_INVALID_ARGUMENT,
            "epsilon must be finite and greater than zero");
    require(op->reserved == 0, PR_INVALID_ARGUMENT, "Reserved field must be zero");
    require(op->x && op->gamma && op->beta && op->out, PR_INVALID_ARGUMENT,
            "Missing layer norm tensor (x, gamma, beta and out are required)");
    const auto x = operand(op->x, "x", 2), g = operand(op->gamma, "gamma", 1),
               b = operand(op->beta, "beta", 1), o = operand(op->out, "out", 2);
    const auto rows = op->x->shape[0], columns = op->x->shape[1];
    expect_dims(op->out, "out", {rows, columns});
    expect_dims(op->gamma, "gamma", {columns});
    expect_dims(op->beta, "beta", {columns});
    require(columns <= max_layer_norm_columns, PR_UNSUPPORTED,
            "layer norm supports at most 2^24 columns (exact FP32 column count)");
    row_limit(rows, "layer norm");
    no_alias(op->out, o, op->x, x, "x");
    no_alias(op->out, o, op->gamma, g, "gamma");
    no_alias(op->out, o, op->beta, b, "beta");
    same_context(queue, operators,
                 {{op->x, "x"}, {op->gamma, "gamma"}, {op->beta, "beta"}, {op->out, "out"}});
    Owned fn;
    kernel(operators, "paralyn_layer_norm_f32", fn);
    if (!x.count)
      return;
    Owned xv, gv, bv, ov;
    view(op->x, x, PR_READ, xv);
    view(op->gamma, g, PR_READ, gv);
    view(op->beta, b, PR_READ, bv);
    view(op->out, o, PR_READ_WRITE, ov);
    const std::array<pr_argument, 6> args{buffer(xv), buffer(gv),   buffer(bv),
                                          buffer(ov), u32(columns), f32(op->epsilon)};
    call(pr_launch(queue, fn.handle, {static_cast<std::uint32_t>(rows), 1, 1}, {group, 1, 1},
                   args.data(), 6, out));
  });
}
pr_status pr_add_f32(pr_queue queue, pr_module operators, const pr_add_v1 *op, pr_event *out) {
  return boundary("add_f32", [&] {
    require(out, PR_INVALID_ARGUMENT, "Missing event output");
    *out = 0;
    header(op, "add");
    require(op->x && op->y && op->out, PR_INVALID_ARGUMENT, "Missing input or output tensor");
    const auto rank = op->x->rank;
    const auto x = operand(op->x, "x", rank), y = operand(op->y, "y", rank),
               o = operand(op->out, "out", rank);
    for (std::uint32_t i = 0; i < rank; ++i)
      if (op->y->shape[i] != op->x->shape[i] || op->out->shape[i] != op->x->shape[i])
        fail(PR_INVALID_ARGUMENT, "add shapes differ: x " + shape(op->x) + ", y " +
                                      shape(op->y) + ", out " + shape(op->out));
    no_alias(op->out, o, op->x, x, "x");
    no_alias(op->out, o, op->y, y, "y");
    same_context(queue, operators, {{op->x, "x"}, {op->y, "y"}, {op->out, "out"}});
    Owned fn;
    kernel(operators, "paralyn_add_f32", fn);
    if (!x.count)
      return;
    Owned xv, yv, ov;
    view(op->x, x, PR_READ, xv);
    view(op->y, y, PR_READ, yv);
    view(op->out, o, PR_READ_WRITE, ov);
    const std::array<pr_argument, 4> args{buffer(xv), buffer(yv), buffer(ov), u32(x.count)};
    call(pr_launch(queue, fn.handle, linear(x.count), {group, 1, 1}, args.data(), 4, out));
  });
}
} // extern "C"
