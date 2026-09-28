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
#include <array>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::uint64_t max_elements = static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
constexpr std::uint32_t tile = 16, group = 256;

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

// out[i] = act(x[i] + bias[i % columns]); ReLU replaces only values less than zero.
kernel void paralyn_bias_activation_f32(device const float* x [[buffer(0)]],
                                        device const float* bias [[buffer(1)]],
                                        device float* out [[buffer(2)]],
                                        constant uint& count [[buffer(3)]],
                                        constant uint& columns [[buffer(4)]],
                                        constant uint& activation [[buffer(5)]],
                                        uint i [[thread_position_in_grid]]) {
  if (i < count) {
    float value = x[i] + bias[i % columns];
    if (activation == 1u && value < 0.0f) value = 0.0f;
    out[i] = value;
  }
}

kernel void paralyn_activation_f32(device const float* x [[buffer(0)]],
                                   device float* out [[buffer(1)]],
                                   constant uint& count [[buffer(2)]],
                                   constant uint& activation [[buffer(3)]],
                                   uint i [[thread_position_in_grid]]) {
  if (i < count) {
    float value = x[i];
    if (activation == 1u && value < 0.0f) value = 0.0f;
    out[i] = value;
  }
}

// Used for k == 0: the empty sum is +0.0 in every output element.
kernel void paralyn_fill_f32(device float* out [[buffer(0)]],
                             constant uint& count [[buffer(1)]],
                             constant float& value [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
  if (i < count) out[i] = value;
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
        {"value", ScalarType::F32, false, R}}}};
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
pr_dim3 linear(std::uint64_t count) {
  return {static_cast<std::uint32_t>((count + group - 1) / group), 1, 1};
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
    require(op->activation == PR_ACTIVATION_NONE || op->activation == PR_ACTIVATION_RELU,
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
} // extern "C"
