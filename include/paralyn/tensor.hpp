#pragma once
// C++ wrappers for the versioned FP32 tensor descriptor and the
// paralyn.msl.tensor provider (include/paralyn/tensor.h). Owned tensors live on
// one explicit context and its single ordered queue; operators enqueue real GPU
// work and never fall back to the CPU. Not cuBLAS/BLAS/PyTorch compatibility.
#include "paralyn/native.hpp"
#include "paralyn/tensor.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace paralyn::tensors {
using Shape = std::vector<std::uint64_t>;

inline pr_tensor_desc_v1 contiguous(const native::Buffer &buffer, const Shape &shape,
                                    std::uint64_t byte_offset = 0) {
  pr_tensor_desc_v1 d{};
  native::check(pr_tensor_desc_contiguous(buffer.get(), byte_offset, PR_F32,
                                          static_cast<std::uint32_t>(shape.size()), shape.data(), &d));
  return d;
}
inline std::uint64_t required_bytes(const pr_tensor_desc_v1 &desc) {
  std::uint64_t n = 0;
  native::check(pr_tensor_desc_validate(&desc, &n));
  return n;
}
inline std::vector<unsigned char> operators_artifact() {
  std::uint64_t n = 0;
  native::check(pr_tensor_operators_artifact(nullptr, 0, &n));
  std::vector<unsigned char> bytes(static_cast<std::size_t>(n));
  native::check(pr_tensor_operators_artifact(bytes.data(), bytes.size(), &n));
  return bytes;
}
inline native::Module load_operators(const native::Context &context) {
  pr_module m = 0;
  native::check(pr_tensor_operators_load(context.get(), &m));
  return native::Module(m);
}
inline std::optional<native::Event> adopt(pr_event e) {
  if (!e)
    return std::nullopt; // No output elements: no GPU work and no invented event.
  return native::Event(e);
}
// Explicit m, n, k; the provider checks them against every descriptor.
inline std::optional<native::Event>
matmul(const native::Queue &queue, const native::Module &operators, std::uint64_t m, std::uint64_t n,
       std::uint64_t k, const pr_tensor_desc_v1 &a, const pr_tensor_desc_v1 &b,
       const pr_tensor_desc_v1 &c, bool transpose_a = false, bool transpose_b = false) {
  pr_matmul_v1 op{};
  op.struct_size = sizeof(op);
  op.version = PR_TENSOR_VERSION_1;
  op.m = m;
  op.n = n;
  op.k = k;
  op.transpose_a = transpose_a;
  op.transpose_b = transpose_b;
  op.a = &a;
  op.b = &b;
  op.c = &c;
  pr_event e = 0;
  native::check(pr_matmul_f32(queue.get(), operators.get(), &op, &e));
  return adopt(e);
}
inline std::optional<native::Event>
bias_activation(const native::Queue &queue, const native::Module &operators, const pr_tensor_desc_v1 &x,
                const pr_tensor_desc_v1 *bias, const pr_tensor_desc_v1 &out,
                pr_activation activation = PR_ACTIVATION_NONE) {
  pr_bias_activation_v1 op{};
  op.struct_size = sizeof(op);
  op.version = PR_TENSOR_VERSION_1;
  op.activation = activation;
  op.x = &x;
  op.bias = bias;
  op.out = &out;
  pr_event e = 0;
  native::check(pr_bias_activation_f32(queue.get(), operators.get(), &op, &e));
  return adopt(e);
}

namespace detail {
struct State {
  native::Context context;
  pr_device_info device;
  std::unique_ptr<native::Module> operators;
  std::unique_ptr<native::Queue> queue;
  bool closed = false;
  explicit State(const std::string &selector)
      : context(selector.empty() ? native::Context() : native::Context(selector)),
        device(context.device()) {}
  void require_open() const {
    if (closed)
      throw std::runtime_error("Tensor context is closed");
  }
  void load() {
    require_open();
    if (operators)
      return;
    auto loaded = std::make_unique<native::Module>(load_operators(context));
    auto ordered = std::make_unique<native::Queue>(context.queue());
    operators = std::move(loaded);
    queue = std::move(ordered);
  }
  void close() {
    if (closed)
      return;
    closed = true;
    std::exception_ptr failure;
    auto release = [&](auto &owner) {
      if (owner) {
        try { owner->close(); } catch (...) { if (!failure) failure = std::current_exception(); }
        owner.reset();
      }
    };
    release(queue);
    release(operators);
    try { context.close(); } catch (...) { if (!failure) failure = std::current_exception(); }
    if (failure)
      std::rethrow_exception(failure);
  }
};
} // namespace detail

class Tensor;
class Context {
  std::shared_ptr<detail::State> state_;
  friend Tensor from_host(Context &, const Shape &, const std::vector<float> &);

public:
  explicit Context(const std::string &selector = "")
      : state_(std::make_shared<detail::State>(selector)) {}
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&) noexcept = default;
  Context &operator=(Context &&) noexcept = default;
  void close() { if (state_) state_->close(); }
  const native::Context &native() const {
    if (!state_) throw std::runtime_error("Tensor context was moved");
    return state_->context;
  }
  pr_device_info device() const {
    if (!state_) throw std::runtime_error("Tensor context was moved");
    return state_->device;
  }
  pr_device_capabilities_v1 capabilities() const { return native().capabilities(); }
  void evidence(const std::string &directory) const {
    if (!state_) throw std::runtime_error("Tensor context was moved");
    state_->require_open();
    state_->context.evidence(directory);
  }
};

// Owned, contiguous row-major FP32 tensor. Operators allocate distinct outputs.
class Tensor {
  std::shared_ptr<detail::State> state_;
  Shape shape_;
  std::uint64_t count_ = 0;
  native::Buffer buffer_;
  std::unique_ptr<native::Event> event_;
  Tensor(std::shared_ptr<detail::State> s, Shape shape, std::uint64_t count, native::Buffer b,
         std::optional<native::Event> e = std::nullopt)
      : state_(std::move(s)), shape_(std::move(shape)), count_(count), buffer_(std::move(b)) {
    if (e) event_ = std::make_unique<native::Event>(std::move(*e));
  }
  void require_open() const {
    if (!buffer_.get()) throw std::runtime_error("Tensor is closed or moved");
  }
  static std::uint64_t elements(const Shape &shape) {
    std::uint64_t n = 1;
    for (auto d : shape) {
      if (d && n > static_cast<std::uint64_t>(INT32_MAX) / d)
        throw std::invalid_argument("FP32 tensors support at most INT32_MAX elements");
      n *= d;
    }
    return n;
  }
  friend Tensor from_host(Context &, const Shape &, const std::vector<float> &);
  friend Tensor matmul(const Tensor &, const Tensor &, bool, bool);
  friend Tensor bias_activation(const Tensor &, const Tensor *, pr_activation);

public:
  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;
  Tensor(Tensor &&) noexcept = default;
  Tensor &operator=(Tensor &&other) {
    if (this != &other) {
      close();
      state_ = std::move(other.state_);
      shape_ = std::move(other.shape_);
      count_ = other.count_;
      buffer_ = std::move(other.buffer_);
      event_ = std::move(other.event_);
    }
    return *this;
  }
  const Shape &shape() const noexcept { return shape_; }
  std::uint64_t size() const noexcept { return count_; }
  std::uint64_t nbytes() const noexcept { return count_ * sizeof(float); }
  const char *dtype() const noexcept { return "float32"; }
  bool closed() const noexcept { return !buffer_.get(); }
  pr_tensor_desc_v1 desc() const {
    require_open();
    return contiguous(buffer_, shape_);
  }
  // Uploads and empty results have no producing GPU event.
  std::optional<pr_event_info> wait() const {
    require_open();
    if (event_) return event_->wait();
    return std::nullopt;
  }
  std::optional<pr_event_timing_v1> timing() const {
    require_open();
    if (event_) return event_->timing();
    return std::nullopt;
  }
  std::vector<float> to_host() const {
    wait();
    std::vector<float> out(static_cast<std::size_t>(count_));
    buffer_.read(out.data(), nbytes());
    return out;
  }
  void close() {
    std::exception_ptr failure;
    if (event_) {
      try { event_->close(); } catch (...) { failure = std::current_exception(); }
      event_.reset();
    }
    try { buffer_.close(); } catch (...) { if (!failure) failure = std::current_exception(); }
    if (failure) std::rethrow_exception(failure);
  }
};

inline Tensor from_host(Context &context, const Shape &shape, const std::vector<float> &values) {
  static_assert(sizeof(float) == 4, "FP32 tensors require a four-byte float");
  if (!context.state_) throw std::runtime_error("Tensor context was moved");
  context.state_->require_open();
  if (shape.size() > PR_TENSOR_MAX_RANK) throw std::invalid_argument("Tensor rank exceeds 8");
  const auto count = Tensor::elements(shape);
  if (count != values.size()) throw std::invalid_argument("Value count does not match tensor shape");
  auto buffer = context.state_->context.buffer(count * sizeof(float));
  buffer.write(values.data(), count * sizeof(float));
  return Tensor(context.state_, shape, count, std::move(buffer));
}
inline Tensor matmul(const Tensor &a, const Tensor &b, bool transpose_a = false,
                     bool transpose_b = false) {
  a.require_open();
  b.require_open();
  if (a.state_ != b.state_) throw std::invalid_argument("Tensor contexts must match");
  if (a.shape_.size() != 2 || b.shape_.size() != 2)
    throw std::invalid_argument("matmul requires rank-2 tensors");
  auto &state = *a.state_;
  state.load();
  const auto m = transpose_a ? a.shape_[1] : a.shape_[0];
  const auto k = transpose_a ? a.shape_[0] : a.shape_[1];
  const auto n = transpose_b ? b.shape_[0] : b.shape_[1];
  const auto kb = transpose_b ? b.shape_[1] : b.shape_[0];
  if (k != kb) throw std::invalid_argument("matmul inner dimensions differ");
  Shape shape{m, n};
  const auto count = Tensor::elements(shape);
  auto output = state.context.buffer(count * sizeof(float));
  const auto ad = a.desc(), bd = b.desc(), cd = contiguous(output, shape);
  auto event = tensors::matmul(*state.queue, *state.operators, m, n, k, ad, bd, cd, transpose_a,
                               transpose_b);
  return Tensor(a.state_, shape, count, std::move(output), std::move(event));
}
// out = act(x + bias) with bias broadcast along rows; bias == nullptr applies act only.
inline Tensor bias_activation(const Tensor &x, const Tensor *bias,
                              pr_activation activation = PR_ACTIVATION_NONE) {
  x.require_open();
  if (bias) {
    bias->require_open();
    if (bias->state_ != x.state_) throw std::invalid_argument("Tensor contexts must match");
  }
  auto &state = *x.state_;
  state.load();
  auto output = state.context.buffer(x.nbytes());
  const auto xd = x.desc(), od = contiguous(output, x.shape_);
  std::optional<pr_tensor_desc_v1> bd;
  if (bias) bd = bias->desc();
  auto event = tensors::bias_activation(*state.queue, *state.operators, xd, bd ? &*bd : nullptr, od,
                                        activation);
  return Tensor(x.state_, x.shape_, x.count_, std::move(output), std::move(event));
}
inline Tensor bias_add(const Tensor &x, const Tensor &bias, bool relu = false) {
  return bias_activation(x, &bias, relu ? PR_ACTIVATION_RELU : PR_ACTIVATION_NONE);
}
inline Tensor relu(const Tensor &x) { return bias_activation(x, nullptr, PR_ACTIVATION_RELU); }
} // namespace paralyn::tensors
