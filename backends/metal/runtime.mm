#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace unicuda {
namespace {
struct RuntimeError : std::runtime_error {
  cudaError_t code;
  RuntimeError(cudaError_t value, const std::string &text)
      : std::runtime_error(text), code(value) {}
};
thread_local cudaError_t last_error = cudaSuccess;
thread_local cudaError_t last_detail_code = cudaSuccess;
// Kept alive through atexit; C++ destroys thread-local objects before exit callbacks.
thread_local std::string *last_detail = new std::string;
std::mutex runtime_mutex;

std::string utf8(NSString *text) { return text ? std::string([text UTF8String]) : std::string(); }
std::string metal_error(NSError *error) {
  if (!error)
    return "Metal supplied no NSError details";
  return utf8(error.domain) + " (" + std::to_string(error.code) +
         "): " + utf8(error.localizedDescription);
}
std::string environment(const char *key, const char *fallback = "unknown") {
  const char *value = std::getenv(key);
  return value && *value ? value : fallback;
}
std::string json(const std::string &text) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : text) {
    if (c == '"' || c == '\\')
      out << '\\' << char(c);
    else if (c == '\n')
      out << "\\n";
    else if (c == '\r')
      out << "\\r";
    else if (c == '\t')
      out << "\\t";
    else if (c < 32)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
    else
      out << char(c);
  }
  out << '"';
  return out.str();
}
std::vector<id<MTLDevice>> enumerate_devices() {
  NSArray<id<MTLDevice>> *found = MTLCopyAllDevices();
  std::vector<id<MTLDevice>> result;
  for (id<MTLDevice> device in found)
    result.push_back(device);
  std::sort(result.begin(), result.end(),
            [](id<MTLDevice> a, id<MTLDevice> b) { return a.registryID < b.registryID; });
  return result;
}
id<MTLDevice> choose_device(const std::string &selector) {
  auto devices = enumerate_devices();
  if (devices.empty())
    throw RuntimeError(cudaErrorInitializationError,
                       "No physical Metal device is available; UniCUDA has no CPU fallback");
  std::size_t index = 0;
  if (selector != "auto") {
    if (selector.empty() || selector.find_first_not_of("0123456789") != std::string::npos)
      throw RuntimeError(cudaErrorInvalidDevice, "Device must be auto or a numeric index");
    try {
      index = std::stoull(selector);
    } catch (...) {
      throw RuntimeError(cudaErrorInvalidDevice, "Device index is out of range");
    }
  }
  if (index >= devices.size())
    throw RuntimeError(cudaErrorInvalidDevice, "Device index is out of range");
  return devices[index];
}
struct Token {
  std::uint64_t identity;
};
struct Allocation {
  id<MTLBuffer> buffer;
  std::size_t size;
};
struct Pending {
  id<MTLCommandBuffer> command;
  std::vector<std::shared_ptr<Allocation>> resources;
  std::string kernel;
  std::string source;
  Dim3 grid;
  Dim3 block;
};
struct Execution {
  std::string kernel;
  Dim3 grid;
  Dim3 block;
  double start;
  double end;
  bool complete;
  std::string error;
};
struct Context;
Context *active_context = nullptr;
void exit_shutdown();
void ensure_shutdown_handler() {
  static const bool installed = std::atexit(exit_shutdown) == 0;
  if (!installed) {
    std::cerr << "UniCUDA could not install its GPU shutdown handler\n";
    std::_Exit(EXIT_FAILURE);
  }
}

struct Context {
  std::string device_selector;
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  std::vector<std::unique_ptr<Token>> tokens;
  std::unordered_map<void *, std::shared_ptr<Allocation>> allocations;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
  std::vector<Pending> pending;
  std::vector<Execution> executions;
  std::string terminal_failure;
  std::string latest_source;
  bool announced = false;

  explicit Context(const std::string &selector)
      : device_selector(selector), device(choose_device(selector)) {
    queue = [device newCommandQueue];
    if (!queue)
      throw RuntimeError(cudaErrorInitializationError, "Metal failed to create a command queue");
  }
  void announce_device() {
    if (announced)
      return;
    std::cout << "Device: " << utf8(device.name) << "\nBackend: Metal\nSelection: "
              << (device_selector == "auto" ? "auto; first device in stable registry-ID order"
                                            : "explicit device index " + device_selector)
              << '\n';
    announced = true;
  }
  std::shared_ptr<Allocation> allocation(const void *token) {
    auto it = allocations.find(const_cast<void *>(token));
    if (it == allocations.end())
      throw RuntimeError(cudaErrorInvalidDevicePointer,
                         "Expected a live cudaMalloc base token; interior, foreign, and freed "
                         "pointers are unsupported");
    return it->second;
  }
  void write_artifacts() {
    const auto directory = environment("UNICUDA_ARTIFACT_DIR", "");
    if (directory.empty() || executions.empty())
      return;
    std::filesystem::create_directories(directory);
    std::ofstream metal(std::filesystem::path(directory) / "generated.metal");
    metal << latest_source;
    metal.close();
    if (!metal)
      throw RuntimeError(cudaErrorUnknown, "Cannot write generated.metal");
    std::ofstream output(std::filesystem::path(directory) / "execution.json");
    output << std::setprecision(17)
           << "{\n  \"backend\": \"Metal\",\n  \"device\": " << json(utf8(device.name))
           << ",\n  \"registry_id\": " << device.registryID
           << ",\n  \"os\": " << json(utf8(NSProcessInfo.processInfo.operatingSystemVersionString))
           << ",\n  \"unicuda_commit\": " << json(environment("UNICUDA_COMMIT"))
           << ",\n  \"unicuda_dirty\": "
           << (environment("UNICUDA_SOURCE_DIRTY") == "true"    ? "true"
               : environment("UNICUDA_SOURCE_DIRTY") == "false" ? "false"
                                                                : "null")
           << ",\n  \"llvm_version\": " << json(environment("UNICUDA_LLVM_VERSION"))
           << ",\n  \"math_mode\": \"safe\",\n  \"floating_point_functions\": \"precise\","
           << "\n  \"cpu_fallback\": false,\n  \"launches\": [\n";
    for (std::size_t i = 0; i < executions.size(); ++i) {
      const auto &e = executions[i];
      output << "    {\"kernel\": " << json(e.kernel) << ", \"grid\": [" << e.grid.x << ','
             << e.grid.y << ',' << e.grid.z << "], \"block\": [" << e.block.x << ',' << e.block.y
             << ',' << e.block.z
             << "], \"command_status\": " << json(e.complete ? "completed" : "failed")
             << ", \"gpu_start_seconds\": " << e.start << ", \"gpu_end_seconds\": " << e.end
             << ", \"gpu_duration_seconds\": " << (e.end - e.start)
             << ", \"error\": " << json(e.error) << '}'
             << (i + 1 == executions.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    output.close();
    if (!output)
      throw RuntimeError(cudaErrorUnknown, "Cannot write execution.json");
  }
  void sync() {
    // Inspect every pending command, even after an earlier command failed.
    for (auto &work : pending) {
      [work.command waitUntilCompleted];
      const bool completed = work.command.status == MTLCommandBufferStatusCompleted;
      std::string failure;
      if (!completed)
        failure =
            "Metal command failed for " + work.kernel + ": " + metal_error(work.command.error);
      double start = work.command.GPUStartTime;
      double end = work.command.GPUEndTime;
      if (completed && (!(start > 0) || !(end > start)))
        failure = "Metal completed " + work.kernel + " without positive GPU timing evidence";
      executions.push_back({work.kernel, work.grid, work.block, start, end, completed, failure});
      latest_source = work.source;
      if (!failure.empty() && terminal_failure.empty())
        terminal_failure = failure;
    }
    pending.clear();
    write_artifacts();
    if (!terminal_failure.empty())
      throw RuntimeError(cudaErrorLaunchFailure, terminal_failure);
  }
};

Context &context() {
  if (!active_context) {
    ensure_shutdown_handler();
    auto owned = std::make_unique<Context>(environment("UNICUDA_DEVICE", "auto"));
    active_context = owned.release();
  }
  return *active_context;
}
void remember(cudaError_t code, const std::string &message) {
  ensure_shutdown_handler();
  last_error = code;
  last_detail_code = code;
  *last_detail = message;
}
cudaError_t remember_exception() noexcept {
  try {
    throw;
  } catch (const RuntimeError &e) {
    remember(e.code, e.what());
  } catch (const std::bad_alloc &e) {
    remember(cudaErrorMemoryAllocation, e.what());
  } catch (const std::exception &e) {
    remember(cudaErrorUnknown, e.what());
  } catch (...) {
    remember(cudaErrorUnknown, "Unknown UniCUDA runtime failure");
  }
  return last_error;
}
void exit_shutdown() {
  try {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    @autoreleasepool {
      if (active_context)
        active_context->sync();
      if (last_error != cudaSuccess)
        throw RuntimeError(last_error, *last_detail);
    }
  } catch (const std::exception &e) {
    std::cerr << "UniCUDA shutdown failed: " << e.what() << '\n';
    std::cerr.flush();
    std::cout.flush();
    std::_Exit(EXIT_FAILURE);
  }
}
struct Bindings {
  BindingLayout layout;
  std::vector<std::shared_ptr<Allocation>> allocations;
};
Bindings resolve(Context &ctx, const Kernel &kernel, const std::vector<Argument> &arguments) {
  if (arguments.size() != kernel.parameters.size())
    throw RuntimeError(cudaErrorInvalidValue, "Kernel argument count mismatch");
  Bindings result;
  std::unordered_map<void *, std::pair<unsigned, ScalarType>> seen;
  unsigned next = 0;
  for (std::size_t i = 0; i < arguments.size(); ++i) {
    const auto &argument = arguments[i];
    const auto &parameter = kernel.parameters[i];
    if (argument.buffer != parameter.buffer)
      throw RuntimeError(cudaErrorInvalidValue,
                         "Kernel argument kind mismatch for " + parameter.name);
    if (parameter.buffer) {
      auto allocation = ctx.allocation(argument.token);
      auto inserted = seen.emplace(argument.token, std::make_pair(next, parameter.type));
      if (inserted.second) {
        ++next;
        result.allocations.push_back(allocation);
      } else if (inserted.first->second.second != parameter.type) {
        throw RuntimeError(cudaErrorNotSupported,
                           "Mixed pointee types for aliased allocation at parameter " +
                               parameter.name);
      }
      result.layout.push_back(inserted.first->second.first);
    } else {
      if (argument.type != parameter.type || argument.type == ScalarType::Bool)
        throw RuntimeError(cudaErrorInvalidValue,
                           "Kernel scalar type mismatch for " + parameter.name);
      result.layout.push_back(next++);
    }
  }
  if (next > 31)
    throw RuntimeError(cudaErrorNotSupported, "Kernel exceeds the Metal buffer argument limit");
  return result;
}
void validate_geometry(Context &ctx, id<MTLComputePipelineState> pipeline, Dim3 grid, Dim3 block) {
  if (!grid.x || !grid.y || !grid.z || !block.x || !block.y || !block.z)
    throw RuntimeError(cudaErrorInvalidValue, "Grid and block dimensions must be positive");
  const MTLSize limit = ctx.device.maxThreadsPerThreadgroup;
  if (block.x > limit.width || block.y > limit.height || block.z > limit.depth)
    throw RuntimeError(cudaErrorInvalidValue, "Block dimension exceeds Metal device limits");
  const std::uint64_t threads = std::uint64_t(block.x) * block.y * block.z;
  if (threads > pipeline.maxTotalThreadsPerThreadgroup)
    throw RuntimeError(cudaErrorInvalidValue, "Block thread count exceeds Metal pipeline limits");
  const std::uint64_t x = std::uint64_t(grid.x) * block.x;
  const std::uint64_t y = std::uint64_t(grid.y) * block.y;
  const std::uint64_t z = std::uint64_t(grid.z) * block.z;
  if (x > UINT32_MAX || y > UINT32_MAX || z > UINT32_MAX)
    throw RuntimeError(cudaErrorInvalidValue, "Logical grid dimension exceeds 32-bit indexing");
  if (x > UINT64_MAX / y || x * y > UINT64_MAX / z)
    throw RuntimeError(cudaErrorInvalidValue, "Logical grid thread count overflows 64 bits");
}
void submit(const Kernel &kernel, Dim3 grid, Dim3 block, const std::vector<Argument> &arguments,
            const std::string *handwritten) {
  verify(kernel);
  Context &ctx = context();
  if (!ctx.terminal_failure.empty())
    throw RuntimeError(cudaErrorLaunchFailure, ctx.terminal_failure);
  Bindings bindings = resolve(ctx, kernel, arguments);
  const std::string source = handwritten ? *handwritten : emit_msl(kernel, bindings.layout);
  ctx.announce_device();
  std::cout << "Kernel: " << kernel.name << "\nGrid: " << grid.x << " × " << grid.y << " × "
            << grid.z << "\nBlock: " << block.x << " × " << block.y << " × " << block.z << '\n';
  auto found = ctx.pipelines.find(source);
  id<MTLComputePipelineState> pipeline;
  if (found == ctx.pipelines.end()) {
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    options.languageVersion = MTLLanguageVersion3_1;
    if (@available(macOS 15.0, *)) {
      options.mathMode = MTLMathModeSafe;
      options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    } else {
      throw RuntimeError(cudaErrorNotSupported,
                         "Safe/precise Metal compilation requires macOS 15 or newer");
    }
    NSError *error = nil;
    NSString *text = [[NSString alloc] initWithBytes:source.data()
                                              length:source.size()
                                            encoding:NSUTF8StringEncoding];
    std::cout << "Compiling kernel...\n" << std::flush;
    id<MTLLibrary> library = [ctx.device newLibraryWithSource:text options:options error:&error];
    if (!library)
      throw RuntimeError(cudaErrorLaunchFailure,
                         "Metal shader compilation failed: " + metal_error(error));
    if (error)
      std::cerr << "Metal compiler: " << metal_error(error) << '\n';
    NSString *name = [NSString stringWithUTF8String:kernel.name.c_str()];
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function)
      throw RuntimeError(cudaErrorLaunchFailure,
                         "Generated Metal library has no entrypoint named " + kernel.name);
    pipeline = [ctx.device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline)
      throw RuntimeError(cudaErrorLaunchFailure,
                         "Metal pipeline creation failed: " + metal_error(error));
    ctx.pipelines.emplace(source, pipeline);
  } else {
    std::cout << "Reusing kernel pipeline...\n" << std::flush;
    pipeline = found->second;
  }
  validate_geometry(ctx, pipeline, grid, block);
  id<MTLCommandBuffer> command = [ctx.queue commandBuffer];
  if (!command)
    throw RuntimeError(cudaErrorLaunchFailure, "Metal could not create a command buffer");
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (!encoder)
    throw RuntimeError(cudaErrorLaunchFailure, "Metal could not create a compute encoder");
  [encoder setComputePipelineState:pipeline];
  for (std::size_t i = 0; i < arguments.size(); ++i) {
    const auto &a = arguments[i];
    if (a.buffer) {
      auto allocation = ctx.allocation(a.token);
      [encoder setBuffer:allocation->buffer offset:0 atIndex:bindings.layout[i]];
    } else {
      // Metal copies setBytes data immediately; no caller scalar lifetime is retained.
      [encoder setBytes:a.bytes.data() length:a.bytes.size() atIndex:bindings.layout[i]];
    }
  }
  [encoder dispatchThreadgroups:MTLSizeMake(grid.x, grid.y, grid.z)
          threadsPerThreadgroup:MTLSizeMake(block.x, block.y, block.z)];
  [encoder endEncoding];
  ctx.pending.push_back(
      {command, std::move(bindings.allocations), kernel.name, source, grid, block});
  std::cout << "Executing on GPU...\n" << std::flush;
  [command commit];
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
void select_device(const std::string &selector) {
  std::lock_guard<std::mutex> lock(runtime_mutex);
  @autoreleasepool {
    id<MTLDevice> selected = choose_device(selector);
    if (active_context && active_context->device.registryID != selected.registryID)
      throw RuntimeError(cudaErrorNotSupported,
                         "Changing devices after runtime initialization is unsupported");
    if (!active_context) {
      ensure_shutdown_handler();
      active_context = new Context(selector);
    }
    active_context->announce_device();
  }
}
std::string devices_text() {
  std::lock_guard<std::mutex> lock(runtime_mutex);
  @autoreleasepool {
    auto devices = enumerate_devices();
    if (devices.empty())
      return "No Metal devices available. CPU fallback is unavailable.\n";
    std::ostringstream out;
    for (std::size_t i = 0; i < devices.size(); ++i) {
      id<MTLDevice> d = devices[i];
      out << i << ": " << utf8(d.name) << " [Metal, registry " << d.registryID << "]\n"
          << "  Unified memory: " << (d.hasUnifiedMemory ? "yes" : "no") << '\n'
          << "  Maximum buffer bytes: " << d.maxBufferLength << '\n'
          << "  Recommended working-set bytes: " << d.recommendedMaxWorkingSetSize
          << " (recommendation, not free memory)\n";
    }
    return out.str();
  }
}
void launch(const Kernel &kernel, Dim3 grid, Dim3 block, const std::vector<Argument> &args) {
  std::lock_guard<std::mutex> lock(runtime_mutex);
  @autoreleasepool {
    submit(kernel, grid, block, args, nullptr);
  }
}
void launch_checked(const Kernel &kernel, Dim3 grid, Dim3 block,
                    const std::vector<Argument> &args) noexcept {
  try {
    launch(kernel, grid, block, args);
  } catch (...) {
    remember_exception();
  }
}
bool validate_launch_configuration(std::size_t shared, const void *stream) noexcept {
  if (shared || stream) {
    remember(cudaErrorNotSupported,
             "Only zero dynamic shared memory and the default null stream are supported");
    return false;
  }
  return true;
}
void launch_msl_for_test(const Kernel &kernel, const std::string &source, Dim3 grid, Dim3 block,
                         const std::vector<Argument> &args) {
  std::lock_guard<std::mutex> lock(runtime_mutex);
  @autoreleasepool {
    submit(kernel, grid, block, args, &source);
  }
}
void synchronize() {
  std::lock_guard<std::mutex> lock(runtime_mutex);
  @autoreleasepool {
    context().sync();
  }
}
void shutdown() {
  std::lock_guard<std::mutex> lock(runtime_mutex);
  @autoreleasepool {
    if (active_context)
      active_context->sync();
    if (last_error != cudaSuccess)
      throw RuntimeError(last_error, *last_detail);
  }
}
} // namespace unicuda

extern "C" cudaError_t cudaMalloc(void **pointer, std::size_t bytes) {
  try {
    std::lock_guard<std::mutex> lock(unicuda::runtime_mutex);
    @autoreleasepool {
      if (!pointer)
        throw unicuda::RuntimeError(cudaErrorInvalidValue, "cudaMalloc requires an output pointer");
      *pointer = nullptr;
      if (!bytes)
        return cudaSuccess;
      auto &ctx = unicuda::context();
      if (bytes > ctx.device.maxBufferLength)
        throw unicuda::RuntimeError(cudaErrorMemoryAllocation,
                                    "cudaMalloc size exceeds Metal maximum buffer length");
      id<MTLBuffer> buffer = [ctx.device newBufferWithLength:bytes
                                                     options:MTLResourceStorageModeShared];
      if (!buffer)
        throw unicuda::RuntimeError(cudaErrorMemoryAllocation, "Metal buffer allocation failed");
      auto token = std::make_unique<unicuda::Token>();
      token->identity = ctx.tokens.size() + 1;
      void *identity = token.get();
      ctx.tokens.push_back(std::move(token));
      ctx.allocations.emplace(
          identity, std::make_shared<unicuda::Allocation>(unicuda::Allocation{buffer, bytes}));
      *pointer = identity;
      return cudaSuccess;
    }
  } catch (...) {
    return unicuda::remember_exception();
  }
}
extern "C" cudaError_t cudaFree(void *pointer) {
  try {
    std::lock_guard<std::mutex> lock(unicuda::runtime_mutex);
    @autoreleasepool {
      auto &ctx = unicuda::context();
      ctx.sync();
      if (pointer) {
        ctx.allocation(pointer);
        ctx.allocations.erase(pointer);
      }
      return cudaSuccess;
    }
  } catch (...) {
    return unicuda::remember_exception();
  }
}
extern "C" cudaError_t cudaMemcpy(void *destination, const void *source, std::size_t bytes,
                                  cudaMemcpyKind kind) {
  try {
    std::lock_guard<std::mutex> lock(unicuda::runtime_mutex);
    @autoreleasepool {
      if (kind != cudaMemcpyHostToDevice && kind != cudaMemcpyDeviceToHost)
        throw unicuda::RuntimeError(cudaErrorInvalidMemcpyDirection,
                                    "Only explicit H2D and D2H copies are supported");
      auto &ctx = unicuda::context();
      ctx.sync();
      if (!bytes)
        return cudaSuccess;
      if (!destination || !source)
        throw unicuda::RuntimeError(cudaErrorInvalidValue,
                                    "Nonempty cudaMemcpy requires nonnull pointers");
      const bool to_device = kind == cudaMemcpyHostToDevice;
      auto allocation = ctx.allocation(to_device ? destination : source);
      if (bytes > allocation->size)
        throw unicuda::RuntimeError(cudaErrorInvalidValue,
                                    "cudaMemcpy byte count exceeds allocation");
      const void *host_pointer = to_device ? source : destination;
      if (std::any_of(ctx.tokens.begin(), ctx.tokens.end(),
                      [&](const auto &token) { return token.get() == host_pointer; }))
        throw unicuda::RuntimeError(cudaErrorInvalidValue,
                                    "Host side of cudaMemcpy is a device allocation token");
      if (to_device)
        std::memcpy(allocation->buffer.contents, source, bytes);
      else
        std::memcpy(destination, allocation->buffer.contents, bytes);
      return cudaSuccess;
    }
  } catch (...) {
    return unicuda::remember_exception();
  }
}
extern "C" cudaError_t cudaDeviceSynchronize() {
  try {
    unicuda::synchronize();
    return cudaSuccess;
  } catch (...) {
    return unicuda::remember_exception();
  }
}
extern "C" cudaError_t cudaGetLastError() {
  const auto result = unicuda::last_error;
  unicuda::last_error = cudaSuccess;
  return result;
}
extern "C" const char *cudaGetErrorString(cudaError_t error) {
  if (error != cudaSuccess && error == unicuda::last_detail_code && !unicuda::last_detail->empty())
    return unicuda::last_detail->c_str();
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
    return "unknown UniCUDA runtime error";
  }
}
