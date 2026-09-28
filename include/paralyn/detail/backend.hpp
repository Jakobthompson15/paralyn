#pragma once
#include "paralyn/runtime.hpp"
#include "paralyn/executable.hpp"
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
  std::string stable_id;
  std::uint64_t max_threadgroup_memory_bytes = 0;
  Dim3 max_block{0, 0, 0};
};
struct CompiledExecutable { virtual ~CompiledExecutable() = default; };
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
  bool duration_valid = false, timestamps_valid = false;
  // 0 unavailable, 1 duration only, 2 Metal system mach time; never wall time.
  std::uint32_t clock_domain = 0;
  double duration_seconds = 0;
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
  virtual std::shared_ptr<CompiledExecutable> prepare(const ExecutableModule &) = 0;
  virtual std::shared_ptr<Event> submit(const std::shared_ptr<CompiledExecutable> &,
                                       std::size_t entry, Dim3 grid, Dim3 block,
                                       const std::vector<BoundArgument> &) = 0;
  virtual std::shared_ptr<Event> submit(const Kernel &, Dim3 grid, Dim3 block,
                                        const std::vector<BoundArgument> &,
                                        const std::string *handwritten = nullptr) = 0;
  virtual void synchronize() = 0;
  virtual RuntimeStatistics statistics() const = 0;
  virtual void write_evidence(const std::string &directory) = 0;
};
std::vector<DeviceInfo> devices();
std::shared_ptr<Context> create_context(const std::string &selector);
// Progress goes to an explicit runtime sidecar when requested, otherwise stderr.
// Log failures are propagated before GPU submission, never silently discarded.
void progress(const std::string &message);
void runtime_event(const std::string &category, const std::string &status,
                   const std::string &detail, std::uint64_t context_id = 0,
                   std::uint64_t operation_id = 0, std::uint64_t bytes = 0);
} // namespace paralyn::backend
