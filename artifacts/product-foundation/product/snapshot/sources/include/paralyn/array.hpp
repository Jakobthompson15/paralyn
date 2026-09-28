#pragma once
#include "paralyn/native.hpp"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#define PARALYN_ARRAY_RESTORE_NOMINMAX
#endif
#include <windows.h>
#ifdef PARALYN_ARRAY_RESTORE_NOMINMAX
#undef NOMINMAX
#undef PARALYN_ARRAY_RESTORE_NOMINMAX
#endif
#else
#include <dlfcn.h>
#endif

namespace paralyn::arrays {
namespace detail {
inline std::string operator_module_path() {
  if (const char *path = std::getenv("PARALYN_OPERATORS")) {
    if (*path)
      return path;
  }
#ifdef _WIN32
  HMODULE module = nullptr;
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCWSTR>(&pr_abi_version), &module))
    throw std::runtime_error("Cannot locate the native library; set PARALYN_OPERATORS");
  std::wstring path(32768, L'\0');
  const auto length = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
  if (!length || length >= path.size())
    throw std::runtime_error("Cannot locate the native library path; set PARALYN_OPERATORS");
  path.resize(length);
  auto library = std::filesystem::path(path).parent_path();
#else
  Dl_info info{};
  if (!dladdr(reinterpret_cast<const void *>(&pr_abi_version), &info) || !info.dli_fname)
    throw std::runtime_error("Cannot locate the native library; set PARALYN_OPERATORS");
  auto library = std::filesystem::absolute(info.dli_fname).parent_path();
#endif
  for (const auto &path : {library / "operators.prk",
                           library / ".." / "share" / "paralyn" / "operators.prk",
                           library / ".." / "_data" / "operators.prk"})
    if (std::filesystem::is_regular_file(path))
      return std::filesystem::canonical(path).string();
  throw std::runtime_error("Array operator module is missing; install it or set PARALYN_OPERATORS");
}
struct State {
  native::Context context;
  pr_device_info device;
  std::string module_path;
  std::unique_ptr<native::Module> module;
  std::unique_ptr<native::Kernel> add, affine;
  std::unique_ptr<native::Queue> queue;
  bool closed = false;
  State(const std::string &selector, std::string path)
      : context(selector.empty() ? native::Context() : native::Context(selector)),
        device(context.device()), module_path(std::move(path)) {}
  void require_open() const {
    if (closed)
      throw std::runtime_error("Array context is closed");
  }
  void operators() {
    require_open();
    if (module)
      return;
    auto loaded = std::make_unique<native::Module>(context.module(
        module_path.empty() ? operator_module_path() : module_path));
    auto sum = std::make_unique<native::Kernel>(loaded->kernel("array_add"));
    auto scaled = std::make_unique<native::Kernel>(loaded->kernel("array_affine"));
    auto ordered = std::make_unique<native::Queue>(context.queue());
    module = std::move(loaded);
    add = std::move(sum);
    affine = std::move(scaled);
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
    release(affine);
    release(add);
    release(module);
    try { context.close(); } catch (...) { if (!failure) failure = std::current_exception(); }
    if (failure)
      std::rethrow_exception(failure);
  }
};
inline void check_size(std::size_t size) {
  if (size > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
    throw std::invalid_argument("FP32 arrays support at most INT32_MAX elements");
}
} // namespace detail

class Array;
class Context {
  std::shared_ptr<detail::State> state_;
  friend Array asarray(Context &, const std::vector<float> &);
public:
  explicit Context(const std::string &selector = "", const std::string &module_path = "")
      : state_(std::make_shared<detail::State>(selector, module_path)) {}
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&) noexcept = default;
  Context &operator=(Context &&) noexcept = default;
  void close() { if (state_) state_->close(); }
  pr_device_info device() const {
    if (!state_) throw std::runtime_error("Array context was moved");
    return state_->device;
  }
  void evidence(const std::string &directory) const {
    if (!state_) throw std::runtime_error("Array context was moved");
    state_->require_open();
    state_->context.evidence(directory);
  }
};

class Array {
  std::shared_ptr<detail::State> state_;
  std::size_t size_;
  native::Buffer buffer_;
  std::unique_ptr<native::Event> event_;
  Array(std::shared_ptr<detail::State> state, std::size_t n, native::Buffer buffer,
        std::unique_ptr<native::Event> event = {})
      : state_(std::move(state)), size_(n), buffer_(std::move(buffer)), event_(std::move(event)) {}
  void require_open() const {
    if (!buffer_.get()) throw std::runtime_error("Array is closed or moved");
  }
  friend Array asarray(Context &, const std::vector<float> &);
  friend Array binary(const Array &, const Array &, std::optional<float>);
public:
  Array(const Array &) = delete;
  Array &operator=(const Array &) = delete;
  Array(Array &&) noexcept = default;
  Array &operator=(Array &&other) {
    if (this != &other) {
      close();
      state_ = std::move(other.state_);
      size_ = other.size_;
      buffer_ = std::move(other.buffer_);
      event_ = std::move(other.event_);
    }
    return *this;
  }
  std::size_t size() const noexcept { return size_; }
  std::size_t nbytes() const noexcept { return size_ * sizeof(float); }
  std::array<std::size_t, 1> shape() const noexcept { return {size_}; }
  const char *dtype() const noexcept { return "float32"; }
  bool closed() const noexcept { return !buffer_.get(); }
  pr_device_info device() const {
    require_open();
    return state_->device;
  }
  std::optional<pr_event_info> wait() const {
    require_open();
    if (event_) return event_->wait();
    return std::nullopt; // Uploaded and empty arrays do not invent a GPU event.
  }
  std::vector<float> to_host() const {
    wait();
    std::vector<float> result(size_);
    buffer_.read(result.data(), nbytes());
    return result;
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

inline Array asarray(Context &context, const std::vector<float> &values) {
  static_assert(sizeof(float) == 4, "Native FP32 arrays require a four-byte float");
  if (!context.state_) throw std::runtime_error("Array context was moved");
  context.state_->require_open();
  detail::check_size(values.size());
  auto buffer = context.state_->context.buffer(values.size() * sizeof(float));
  buffer.write(values.data(), values.size() * sizeof(float));
  return Array(context.state_, values.size(), std::move(buffer));
}
inline Array binary(const Array &a, const Array &b, std::optional<float> scale) {
  a.require_open();
  b.require_open();
  if (a.state_ != b.state_) throw std::invalid_argument("Array contexts must match");
  if (a.size_ != b.size_) throw std::invalid_argument("Array lengths must match; no broadcasting");
  a.state_->require_open();
  auto &state = *a.state_;
  if (!a.size_)
    return Array(a.state_, 0, state.context.buffer(0));
  state.operators();
  auto output = state.context.buffer(a.nbytes());
  auto av = a.buffer_.view(0, a.nbytes(), PR_READ);
  auto bv = b.buffer_.view(0, b.nbytes(), PR_READ);
  auto ov = output.view(0, a.nbytes(), PR_WRITE);
  std::vector<native::Argument> args{native::Argument::buffer(av), native::Argument::buffer(bv),
                                    native::Argument::buffer(ov),
                                    native::Argument::u32(static_cast<std::uint32_t>(a.size_))};
  if (scale) args.push_back(native::Argument::f32(*scale));
  auto event = std::make_unique<native::Event>(state.queue->launch(
      scale ? *state.affine : *state.add,
      {static_cast<std::uint32_t>((a.size_ + 255) / 256), 1, 1}, {256, 1, 1}, args));
  return Array(a.state_, a.size_, std::move(output), std::move(event));
}
inline Array add(const Array &a, const Array &b) { return binary(a, b, std::nullopt); }
inline Array affine(const Array &a, const Array &b, float scale) { return binary(a, b, scale); }
} // namespace paralyn::arrays
