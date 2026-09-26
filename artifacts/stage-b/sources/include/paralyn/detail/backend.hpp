#pragma once
#include "paralyn/runtime.hpp"
#include <memory>
#include <stdexcept>

namespace paralyn::backend {
enum class ErrorCode {
  invalid_value,
  invalid_device,
  invalid_handle,
  out_of_memory,
  unsupported,
  compilation,
  execution,
  internal
};
struct Error : std::runtime_error {
  ErrorCode code;
  Error(ErrorCode c, const std::string &message) : std::runtime_error(message), code(c) {}
};
struct DeviceInfo {
  std::string name, backend, os;
  std::uint64_t registry_id = 0, max_buffer_bytes = 0;
  bool unified_memory = false;
  std::uint64_t recommended_working_set_bytes = 0;
};
struct Buffer {
  virtual ~Buffer() = default;
  virtual std::size_t size() const = 0;
};
struct BoundArgument {
  bool is_buffer = false;
  std::shared_ptr<Buffer> allocation;
  std::size_t offset = 0, size = 0;
  ScalarType type = ScalarType::I32;
  std::array<unsigned char, 4> bytes{};
};
struct EventInfo {
  bool completed = false;
  double gpu_start_seconds = 0, gpu_end_seconds = 0;
};
struct Event {
  virtual ~Event() = default;
  virtual EventInfo wait() = 0;
};
class Context {
public:
  virtual ~Context() = default;
  virtual DeviceInfo device_info() const = 0;
  virtual std::shared_ptr<Buffer> allocate(std::size_t bytes) = 0;
  virtual void write(const std::shared_ptr<Buffer> &, std::size_t offset, const void *,
                     std::size_t bytes) = 0;
  virtual void read(const std::shared_ptr<Buffer> &, std::size_t offset, void *,
                    std::size_t bytes) = 0;
  virtual void prepare(const Kernel &) = 0;
  virtual std::shared_ptr<Event> submit(const Kernel &, Dim3 grid, Dim3 block,
                                        const std::vector<BoundArgument> &,
                                        const std::string *handwritten = nullptr) = 0;
  virtual void synchronize() = 0;
  virtual RuntimeStatistics statistics() const = 0;
  virtual void write_evidence(const std::string &directory) = 0;
};
std::vector<DeviceInfo> devices();
std::shared_ptr<Context> create_context(const std::string &selector);
} // namespace paralyn::backend
