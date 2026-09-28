// CUDA compatibility adapter. Physical GPU state lives only in backend::Context.
#include <cuda_runtime.h>
#include "paralyn/detail/backend.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace paralyn {
namespace {
struct RuntimeError : std::runtime_error {
  cudaError_t code;
  RuntimeError(cudaError_t value, const std::string &text) : std::runtime_error(text), code(value) {}
};
thread_local cudaError_t last_error = cudaSuccess;
thread_local cudaError_t last_detail_code = cudaSuccess;
// Kept alive through atexit; C++ destroys thread-local objects before exit callbacks.
thread_local std::string *last_detail = new std::string;
std::mutex runtime_mutex;
std::string environment(const char *key, const char *fallback = "unknown") {
  const char *value = std::getenv(key);
  return value && *value ? value : fallback;
}
cudaError_t cuda_code(backend::ErrorCode code) {
  switch (code) {
  case backend::ErrorCode::invalid_value: return cudaErrorInvalidValue;
  case backend::ErrorCode::invalid_device: return cudaErrorInvalidDevice;
  case backend::ErrorCode::invalid_handle: return cudaErrorInvalidDevicePointer;
  case backend::ErrorCode::out_of_memory: return cudaErrorMemoryAllocation;
  case backend::ErrorCode::unsupported: return cudaErrorNotSupported;
  case backend::ErrorCode::compilation:
  case backend::ErrorCode::execution: return cudaErrorLaunchFailure;
  default: return cudaErrorUnknown;
  }
}
struct Token { std::uint64_t identity; };
struct Adapter {
  std::shared_ptr<backend::Context> engine;
  std::string selector;
  std::vector<std::unique_ptr<Token>> tokens;
  std::unordered_map<void *, std::shared_ptr<backend::Buffer>> allocations;
  bool announced = false;
  explicit Adapter(const std::string &choice) : engine(backend::create_context(choice)), selector(choice) {}
  void announce_device() {
    if (announced) return;
    backend::progress("Device: " + engine->device_info().name + "\nBackend: Metal\nSelection: " +
                      (selector == "auto" ? "auto; first device in stable registry-ID order"
                                          : "explicit device index " + selector) + '\n');
    announced = true;
  }
  std::shared_ptr<backend::Buffer> allocation(const void *token) const {
    const auto found = allocations.find(const_cast<void *>(token));
    if (found == allocations.end())
      throw RuntimeError(cudaErrorInvalidDevicePointer,
                         "Expected a live cudaMalloc base token; interior, foreign, and freed pointers are unsupported");
    return found->second;
  }
  void sync() {
    // The adapter alone opts into the CUDA CLI's environment-selected evidence path.
    const auto directory = environment("PARALYN_ARTIFACT_DIR", "");
    if (directory.empty()) engine->synchronize();
    else engine->write_evidence(directory);
  }
};
Adapter *active_context = nullptr;
void exit_shutdown();
void ensure_shutdown_handler() {
  static const bool installed = std::atexit(exit_shutdown) == 0;
  if (!installed) {
    try { backend::runtime_event("process_failure", "failed", "Cannot install GPU shutdown handler"); }
    catch (...) { std::fputs("Paralyn shutdown event log also failed\n", stderr); }
    std::cerr << "Paralyn could not install its GPU shutdown handler\n";
    std::_Exit(EXIT_FAILURE);
  }
}
Adapter &context() {
  if (!active_context) {
    ensure_shutdown_handler();
    auto owned = std::make_unique<Adapter>(environment("PARALYN_DEVICE", "auto"));
    active_context = owned.release();
  }
  return *active_context;
}
void remember(cudaError_t code, const char *message) noexcept {
  ensure_shutdown_handler();
  last_error = code;
  last_detail_code = code;
  try { *last_detail = message; } catch (...) { last_detail->clear(); }
  try {
    backend::runtime_event("api_failure", "failed", std::string("CUDA compatibility: ") + message);
  } catch (...) {
    // The original compatibility error remains the returned result; failure of
    // its diagnostic channel is part of the detail rather than a silent drop.
    try { *last_detail += " [runtime event log also failed]"; } catch (...) {}
  }
}
cudaError_t remember_exception() noexcept {
  try {
    throw;
  } catch (const RuntimeError &e) {
    remember(e.code, e.what());
  } catch (const backend::Error &e) {
    remember(cuda_code(e.code), e.what());
  } catch (const std::bad_alloc &e) {
    remember(cudaErrorMemoryAllocation, e.what());
  } catch (const std::exception &e) {
    remember(cudaErrorUnknown, e.what());
  } catch (...) {
    remember(cudaErrorUnknown, "Unknown Paralyn runtime failure");
  }
  return last_error;
}

void exit_shutdown() {
  try {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    if (active_context) active_context->sync();
    if (last_error != cudaSuccess) throw RuntimeError(last_error, *last_detail);
  } catch (const std::exception &error) {
    try { backend::runtime_event("process_failure", "failed", error.what()); }
    catch (...) { std::fputs("Paralyn shutdown event log also failed\n", stderr); }
    std::cerr << "Paralyn shutdown failed: " << error.what() << '\n';
    std::cerr.flush(); std::cout.flush();
    std::_Exit(EXIT_FAILURE);
  }
}
void submit(const Kernel &kernel, Dim3 grid, Dim3 block, const std::vector<Argument> &arguments,
            const std::string *handwritten) {
  verify(kernel);
  if (arguments.size() != kernel.parameters.size())
    throw RuntimeError(cudaErrorInvalidValue, "Kernel argument count mismatch");
  auto &ctx = context();
  std::vector<backend::BoundArgument> bound;
  for (std::size_t i = 0; i < arguments.size(); ++i) {
    const auto &argument = arguments[i];
    if (argument.buffer != kernel.parameters[i].buffer)
      throw RuntimeError(cudaErrorInvalidValue,
                         "Kernel argument kind mismatch for " + kernel.parameters[i].name);
    backend::BoundArgument value;
    value.is_buffer = argument.buffer;
    value.type = argument.type; value.bytes = argument.bytes;
    if (argument.buffer) {
      value.allocation = ctx.allocation(argument.token);
      value.size = value.allocation->size();
    }
    bound.push_back(std::move(value));
  }
  ctx.announce_device();
  ctx.engine->submit(kernel, grid, block, bound, handwritten);
}
} // namespace
Argument Argument::from_buffer(const void *value) {
  Argument a;
  a.buffer = true;
  a.token = const_cast<void *>(value);
  return a;
}
Argument Argument::from_i32(std::int32_t value) {
  Argument a;
  a.type = ScalarType::I32;
  std::memcpy(a.bytes.data(), &value, 4);
  return a;
}
Argument Argument::from_u32(std::uint32_t value) {
  Argument a;
  a.type = ScalarType::U32;
  std::memcpy(a.bytes.data(), &value, 4);
  return a;
}
Argument Argument::from_f32(float value) {
  Argument a;
  a.type = ScalarType::F32;
  std::memcpy(a.bytes.data(), &value, 4);
  return a;
}

RuntimeStatistics runtime_statistics() {
  std::lock_guard<std::mutex> lock(runtime_mutex);
  return active_context ? active_context->engine->statistics() : RuntimeStatistics{};
}
void select_device(const std::string &selector) {
  std::lock_guard<std::mutex> lock(runtime_mutex);
  // Validate the selector without changing an initialized compatibility context.
  auto selected = backend::create_context(selector);
  if (active_context && active_context->engine->device_info().registry_id != selected->device_info().registry_id)
    throw RuntimeError(cudaErrorNotSupported, "Changing devices after runtime initialization is unsupported");
  if (!active_context) {
    ensure_shutdown_handler();
    active_context = new Adapter(selector);
  }
  active_context->announce_device();
}
std::string devices_text() {
  auto found = backend::devices();
  if (found.empty()) return "No Metal devices available. CPU fallback is unavailable.\n";
  std::ostringstream out;
  for (std::size_t i = 0; i < found.size(); ++i) {
    const auto &device = found[i];
    out << i << ": " << device.name << " [Metal, registry " << device.registry_id << "]\n"
        << "  Unified memory: " << (device.unified_memory ? "yes" : "no") << '\n'
        << "  Maximum buffer bytes: " << device.max_buffer_bytes << '\n'
        << "  Recommended working-set bytes: " << device.recommended_working_set_bytes
        << " (recommendation, not free memory)\n";
  }
  return out.str();
}
void launch(const Kernel &kernel, Dim3 grid, Dim3 block, const std::vector<Argument> &arguments) {
  std::lock_guard<std::mutex> lock(runtime_mutex);
  submit(kernel, grid, block, arguments, nullptr);
}
void launch_checked(const Kernel &kernel, Dim3 grid, Dim3 block,
                    const std::vector<Argument> &arguments) noexcept {
  try { launch(kernel, grid, block, arguments); } catch (...) { remember_exception(); }
}
bool validate_launch_configuration(std::size_t shared, const void *stream) noexcept {
  if (shared || stream) {
    remember(cudaErrorNotSupported, "Only zero dynamic shared memory and the default null stream are supported");
    return false;
  }
  return true;
}
void launch_msl_for_test(const Kernel &kernel, const std::string &source, Dim3 grid, Dim3 block,
                         const std::vector<Argument> &arguments) {
  std::lock_guard<std::mutex> lock(runtime_mutex);
  submit(kernel, grid, block, arguments, &source);
}
void synchronize() {
  std::lock_guard<std::mutex> lock(runtime_mutex);
  context().sync();
}
void shutdown() {
  std::lock_guard<std::mutex> lock(runtime_mutex);
  if (active_context) active_context->sync();
  if (last_error != cudaSuccess) throw RuntimeError(last_error, *last_detail);
}
} // namespace paralyn

extern "C" cudaError_t cudaMalloc(void **pointer, std::size_t bytes) {
  try {
    std::lock_guard<std::mutex> lock(paralyn::runtime_mutex);
    if (!pointer) throw paralyn::RuntimeError(cudaErrorInvalidValue, "cudaMalloc requires an output pointer");
    *pointer = nullptr;
    if (!bytes) return cudaSuccess;
    auto &ctx = paralyn::context();
    auto allocation = ctx.engine->allocate(bytes);
    auto token = std::make_unique<paralyn::Token>();
    token->identity = ctx.tokens.size() + 1;
    void *identity = token.get();
    ctx.tokens.push_back(std::move(token));
    ctx.allocations.emplace(identity, std::move(allocation));
    *pointer = identity;
    return cudaSuccess;
  } catch (...) { return paralyn::remember_exception(); }
}
extern "C" cudaError_t cudaFree(void *pointer) {
  try {
    std::lock_guard<std::mutex> lock(paralyn::runtime_mutex);
    auto &ctx = paralyn::context();
    ctx.sync();
    if (pointer) { ctx.allocation(pointer); ctx.allocations.erase(pointer); }
    return cudaSuccess;
  } catch (...) { return paralyn::remember_exception(); }
}
extern "C" cudaError_t cudaMemcpy(void *destination, const void *source, std::size_t bytes,
                                   cudaMemcpyKind kind) {
  try {
    std::lock_guard<std::mutex> lock(paralyn::runtime_mutex);
    if (kind != cudaMemcpyHostToDevice && kind != cudaMemcpyDeviceToHost)
      throw paralyn::RuntimeError(cudaErrorInvalidMemcpyDirection, "Only explicit H2D and D2H copies are supported");
    auto &ctx = paralyn::context();
    ctx.sync();
    if (!bytes) return cudaSuccess;
    if (!destination || !source)
      throw paralyn::RuntimeError(cudaErrorInvalidValue, "Nonempty cudaMemcpy requires nonnull pointers");
    const bool to_device = kind == cudaMemcpyHostToDevice;
    auto allocation = ctx.allocation(to_device ? destination : source);
    if (bytes > allocation->size())
      throw paralyn::RuntimeError(cudaErrorInvalidValue, "cudaMemcpy byte count exceeds allocation");
    const void *host_pointer = to_device ? source : destination;
    if (std::any_of(ctx.tokens.begin(), ctx.tokens.end(),
                    [&](const auto &token) { return token.get() == host_pointer; }))
      throw paralyn::RuntimeError(cudaErrorInvalidValue, "Host side of cudaMemcpy is a device allocation token");
    if (to_device) ctx.engine->write(allocation, 0, source, bytes);
    else ctx.engine->read(allocation, 0, destination, bytes);
    return cudaSuccess;
  } catch (...) { return paralyn::remember_exception(); }
}
extern "C" cudaError_t cudaDeviceSynchronize() {
  try {
    paralyn::synchronize();
    return cudaSuccess;
  } catch (...) {
    return paralyn::remember_exception();
  }
}
extern "C" cudaError_t cudaGetLastError() {
  const auto result = paralyn::last_error;
  paralyn::last_error = cudaSuccess;
  return result;
}
extern "C" const char *cudaGetErrorString(cudaError_t error) {
  if (error != cudaSuccess && error == paralyn::last_detail_code && !paralyn::last_detail->empty())
    return paralyn::last_detail->c_str();
  switch (error) {
  case cudaSuccess:
    return "success";
  case cudaErrorInvalidValue:
    return "invalid value";
  case cudaErrorMemoryAllocation:
    return "allocation failed";
  case cudaErrorInitializationError:
    return "Metal initialization failed";
  case cudaErrorInvalidDevice:
    return "invalid device";
  case cudaErrorInvalidDevicePointer:
    return "invalid allocation base token";
  case cudaErrorInvalidMemcpyDirection:
    return "unsupported copy direction";
  case cudaErrorLaunchFailure:
    return "Metal launch failed";
  case cudaErrorNotSupported:
    return "unsupported operation";
  default:
    return "unknown Paralyn runtime error";
  }
}
