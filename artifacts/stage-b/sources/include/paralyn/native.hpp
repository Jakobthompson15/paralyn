#pragma once
#include "paralyn/native.h"
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace paralyn::native {
class Error : public std::runtime_error {
public:
  pr_status code;
  std::string operation;
  explicit Error(const pr_error &e)
      : std::runtime_error(e.message), code(e.code), operation(e.operation) {}
};
inline void check(pr_status status) {
  if (status == PR_SUCCESS)
    return;
  pr_error error{};
  pr_last_error(&error);
  if (error.code == PR_SUCCESS)
    error.code = status;
  throw Error(error);
}
class Handle {
protected:
  pr_handle handle_ = 0;
  explicit Handle(pr_handle h) : handle_(h) {}

public:
  Handle() = default;
  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;
  Handle(Handle &&other) noexcept : handle_(std::exchange(other.handle_, 0)) {}
  Handle &operator=(Handle &&other) {
    if (this != &other) {
      close();
      handle_ = std::exchange(other.handle_, 0);
    }
    return *this;
  }
  void close() {
    auto h = std::exchange(handle_, 0);
    if (h)
      check(pr_release(h));
  }
  ~Handle() noexcept {
    if (handle_ && pr_release(handle_) != PR_SUCCESS) {
      pr_error error{};
      pr_last_error(&error);
      std::fprintf(stderr, "Paralyn resource destructor failed: %s: %s\n", error.operation,
                   error.message);
      std::terminate(); // Explicit wait/close provides catchable error propagation.
    }
  }
  pr_handle get() const noexcept { return handle_; }
};
class View final : public Handle {
public:
  explicit View(pr_view h) : Handle(h) {}
};
class Buffer final : public Handle {
public:
  explicit Buffer(pr_buffer h) : Handle(h) {}
  uint64_t size() const {
    uint64_t n = 0;
    check(pr_buffer_size(handle_, &n));
    return n;
  }
  View view(uint64_t offset, uint64_t n, pr_access access = PR_READ_WRITE,
            uint32_t alignment = 4) const {
    pr_view h = 0;
    check(pr_view_create(handle_, offset, n, alignment, access, &h));
    return View(h);
  }
  void write(const void *data, uint64_t n, uint64_t offset = 0) const {
    check(pr_buffer_write(handle_, offset, data, n));
  }
  void read(void *data, uint64_t n, uint64_t offset = 0) const {
    check(pr_buffer_read(handle_, offset, data, n));
  }
};
class Argument {
  pr_argument value_{};

public:
  static Argument buffer(const View &v) {
    Argument a;
    a.value_.type = PR_BUFFER;
    a.value_.view = v.get();
    return a;
  }
  static Argument i32(int32_t v) {
    Argument a;
    a.value_.type = PR_I32;
    a.value_.i32 = v;
    return a;
  }
  static Argument u32(uint32_t v) {
    Argument a;
    a.value_.type = PR_U32;
    a.value_.u32 = v;
    return a;
  }
  static Argument f32(float v) {
    Argument a;
    a.value_.type = PR_F32;
    a.value_.f32 = v;
    return a;
  }
  const pr_argument &raw() const { return value_; }
};
class Event final : public Handle {
public:
  explicit Event(pr_event h) : Handle(h) {}
  pr_event_info wait() const {
    pr_event_info info{};
    check(pr_event_wait(handle_, &info));
    return info;
  }
};
class Kernel final : public Handle {
public:
  explicit Kernel(pr_kernel h) : Handle(h) {}
  std::vector<pr_parameter_info> parameters() const {
    uint32_t n = 0;
    check(pr_kernel_parameter_count(handle_, &n));
    std::vector<pr_parameter_info> out(n);
    for (uint32_t i = 0; i < n; ++i)
      check(pr_kernel_parameter(handle_, i, &out[i]));
    return out;
  }
};
class Module final : public Handle {
public:
  explicit Module(pr_module h) : Handle(h) {}
  Kernel kernel(const std::string &name) const {
    pr_kernel k = 0;
    check(pr_module_kernel(handle_, name.c_str(), &k));
    return Kernel(k);
  }
};
class Queue final : public Handle {
public:
  explicit Queue(pr_queue h) : Handle(h) {}
  Event launch(const Kernel &kernel, pr_dim3 grid, pr_dim3 block,
               const std::vector<Argument> &args) const {
    std::vector<pr_argument> raw;
    raw.reserve(args.size());
    for (const auto &a : args)
      raw.push_back(a.raw());
    pr_event e = 0;
    check(pr_launch(handle_, kernel.get(), grid, block, raw.data(),
                    static_cast<uint32_t>(raw.size()), &e));
    return Event(e);
  }
  void synchronize() const { check(pr_queue_synchronize(handle_)); }
};
class Context final : public Handle {
public:
  explicit Context(const std::string &selector = "auto") {
    check(pr_context_create(selector.c_str(), &handle_));
  }
  pr_device_info device() const {
    pr_device_info d{};
    check(pr_context_device(handle_, &d));
    return d;
  }
  Buffer buffer(uint64_t bytes) const {
    pr_buffer b = 0;
    check(pr_buffer_create(handle_, bytes, &b));
    return Buffer(b);
  }
  Module module(const std::string &path) const {
    pr_module m = 0;
    check(pr_module_load_file(handle_, path.c_str(), &m));
    return Module(m);
  }
  Queue queue() const {
    pr_queue q = 0;
    check(pr_queue_get(handle_, &q));
    return Queue(q);
  }
  void synchronize() const { check(pr_context_synchronize(handle_)); }
  void evidence(const std::string &directory) const {
    check(pr_context_write_evidence(handle_, directory.c_str()));
  }
};
inline std::vector<pr_device_info> devices() {
  uint32_t count = 0;
  check(pr_device_count(&count));
  std::vector<pr_device_info> out(count);
  for (uint32_t i = 0; i < count; ++i)
    check(pr_device_get(i, &out[i]));
  return out;
}
} // namespace paralyn::native
