#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
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

namespace paralyn {
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
                       "No physical Metal device is available; Paralyn has no CPU fallback");
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
  std::size_t source_index;
  double compile_seconds;
};
struct Execution {
  std::string kernel;
  Dim3 grid;
  Dim3 block;
  double start;
  double end;
  bool complete;
  std::string error;
  std::size_t source_index;
  double compile_seconds;
};
struct Context;
Context *active_context = nullptr;
void exit_shutdown();
void ensure_shutdown_handler() {
  static const bool installed = std::atexit(exit_shutdown) == 0;
  if (!installed) {
    std::cerr << "Paralyn could not install its GPU shutdown handler\n";
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
  std::vector<std::string> sources;
  RuntimeStatistics statistics;
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
    const auto directory = environment("PARALYN_ARTIFACT_DIR", "");
    if (directory.empty() || executions.empty())
      return;
    std::filesystem::create_directories(directory);
    std::ofstream metal(std::filesystem::path(directory) / "generated.metal");
    metal << latest_source;
    metal.close();
    if (!metal)
      throw RuntimeError(cudaErrorUnknown, "Cannot write generated.metal");
    for (std::size_t i = 0; i < sources.size(); ++i) {
      const auto name = "source-" + std::to_string(i) + ".metal";
      std::ofstream source(std::filesystem::path(directory) / name);
      source << sources[i];
      source.close();
      if (!source)
        throw RuntimeError(cudaErrorUnknown, "Cannot write " + name);
    }
    std::ofstream output(std::filesystem::path(directory) / "execution.json");
    output << std::setprecision(17)
           << "{\n  \"backend\": \"Metal\",\n  \"device\": " << json(utf8(device.name))
           << ",\n  \"registry_id\": " << device.registryID
           << ",\n  \"os\": " << json(utf8(NSProcessInfo.processInfo.operatingSystemVersionString))
           << ",\n  \"physical_memory_bytes\": " << NSProcessInfo.processInfo.physicalMemory
           << ",\n  \"unified_memory\": " << (device.hasUnifiedMemory ? "true" : "false")
           << ",\n  \"metal_language_version\": \"3.1\""
           << ",\n  \"thermal_state_at_capture\": " << NSProcessInfo.processInfo.thermalState
           << ",\n  \"low_power_mode_at_capture\": "
           << (NSProcessInfo.processInfo.lowPowerModeEnabled ? "true" : "false")
           << ",\n  \"paralyn_commit\": " << json(environment("PARALYN_COMMIT"))
           << ",\n  \"paralyn_dirty\": "
           << (environment("PARALYN_SOURCE_DIRTY") == "true"    ? "true"
               : environment("PARALYN_SOURCE_DIRTY") == "false" ? "false"
                                                                : "null")
           << ",\n  \"llvm_version\": " << json(environment("PARALYN_LLVM_VERSION"))
           << ",\n  \"math_mode\": \"safe\",\n  \"floating_point_functions\": \"precise\","
           << "\n  \"cpu_fallback\": false,"
           << "\n  \"runtime_owned_current_buffer_bytes\": " << statistics.current_buffer_bytes
           << ",\n  \"runtime_owned_peak_buffer_bytes\": " << statistics.peak_buffer_bytes
           << ",\n  \"buffer_accounting\": \"requested MTLBuffer lengths; excludes host vectors, pipelines, driver allocations\","
           << "\n  \"launches\": [\n";
    for (std::size_t i = 0; i < executions.size(); ++i) {
      const auto &e = executions[i];
      output << "    {\"kernel\": " << json(e.kernel) << ", \"grid\": [" << e.grid.x << ','
             << e.grid.y << ',' << e.grid.z << "], \"block\": [" << e.block.x << ',' << e.block.y
             << ',' << e.block.z
             << "], \"command_status\": " << json(e.complete ? "completed" : "failed")
             << ", \"gpu_start_seconds\": " << e.start << ", \"gpu_end_seconds\": " << e.end
             << ", \"gpu_duration_seconds\": " << (e.end - e.start)
             << ", \"pipeline_compile_seconds\": " << e.compile_seconds
             << ", \"source_file\": " << json("source-" + std::to_string(e.source_index) + ".metal")
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
      executions.push_back({work.kernel, work.grid, work.block, start, end, completed, failure,
                            work.source_index, work.compile_seconds});
      if (completed && failure.empty()) {
        statistics.last_gpu_seconds = end - start;
        statistics.gpu_seconds += end - start;
        ++statistics.completed_launches;
      }
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
    auto owned = std::make_unique<Context>(environment("PARALYN_DEVICE", "auto"));
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
    remember(cudaErrorUnknown, "Unknown Paralyn runtime failure");
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
    std::cerr << "Paralyn shutdown failed: " << e.what() << '\n';
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
  const std::string entrypoint = handwritten ? kernel.name : "uc_kernel_" + kernel.name;
  const std::string pipeline_key = entrypoint + '\n' + source;
  auto found = ctx.pipelines.find(pipeline_key);
  double compile_seconds = 0;
  id<MTLComputePipelineState> pipeline;
  if (found == ctx.pipelines.end()) {
    const auto compile_start = std::chrono::steady_clock::now();
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
    NSString *name = [NSString stringWithUTF8String:entrypoint.c_str()];
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function)
      throw RuntimeError(cudaErrorLaunchFailure,
                         "Generated Metal library has no entrypoint named " + kernel.name);
    pipeline = [ctx.device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline)
      throw RuntimeError(cudaErrorLaunchFailure,
                         "Metal pipeline creation failed: " + metal_error(error));
    ctx.pipelines.emplace(pipeline_key, pipeline);
    compile_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - compile_start).count();
    ctx.statistics.pipeline_compile_seconds += compile_seconds;
    ++ctx.statistics.pipeline_compilations;
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
  auto source_it = std::find(ctx.sources.begin(), ctx.sources.end(), source);
  const std::size_t source_index = std::distance(ctx.sources.begin(), source_it);
  if (source_it == ctx.sources.end())
    ctx.sources.push_back(source);
  ctx.pending.push_back(
      {command, std::move(bindings.allocations), kernel.name, source, grid, block, source_index,
       compile_seconds});
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
RuntimeStatistics runtime_statistics() {
  std::lock_guard<std::mutex> lock(runtime_mutex);
  return active_context ? active_context->statistics : RuntimeStatistics{};
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
} // namespace paralyn

extern "C" cudaError_t cudaMalloc(void **pointer, std::size_t bytes) {
  try {
    std::lock_guard<std::mutex> lock(paralyn::runtime_mutex);
    @autoreleasepool {
      if (!pointer)
        throw paralyn::RuntimeError(cudaErrorInvalidValue, "cudaMalloc requires an output pointer");
      *pointer = nullptr;
      if (!bytes)
        return cudaSuccess;
      auto &ctx = paralyn::context();
      if (bytes > ctx.device.maxBufferLength)
        throw paralyn::RuntimeError(cudaErrorMemoryAllocation,
                                    "cudaMalloc size exceeds Metal maximum buffer length");
      id<MTLBuffer> buffer = [ctx.device newBufferWithLength:bytes
                                                     options:MTLResourceStorageModeShared];
      if (!buffer)
        throw paralyn::RuntimeError(cudaErrorMemoryAllocation, "Metal buffer allocation failed");
      auto token = std::make_unique<paralyn::Token>();
      token->identity = ctx.tokens.size() + 1;
      void *identity = token.get();
      ctx.tokens.push_back(std::move(token));
      ctx.allocations.emplace(
          identity, std::make_shared<paralyn::Allocation>(paralyn::Allocation{buffer, bytes}));
      ctx.statistics.current_buffer_bytes += bytes;
      ctx.statistics.peak_buffer_bytes =
          std::max(ctx.statistics.peak_buffer_bytes, ctx.statistics.current_buffer_bytes);
      *pointer = identity;
      return cudaSuccess;
    }
  } catch (...) {
    return paralyn::remember_exception();
  }
}
extern "C" cudaError_t cudaFree(void *pointer) {
  try {
    std::lock_guard<std::mutex> lock(paralyn::runtime_mutex);
    @autoreleasepool {
      auto &ctx = paralyn::context();
      ctx.sync();
      if (pointer) {
        ctx.statistics.current_buffer_bytes -= ctx.allocation(pointer)->size;
        ctx.allocations.erase(pointer);
      }
      return cudaSuccess;
    }
  } catch (...) {
    return paralyn::remember_exception();
  }
}
extern "C" cudaError_t cudaMemcpy(void *destination, const void *source, std::size_t bytes,
                                  cudaMemcpyKind kind) {
  try {
    std::lock_guard<std::mutex> lock(paralyn::runtime_mutex);
    @autoreleasepool {
      if (kind != cudaMemcpyHostToDevice && kind != cudaMemcpyDeviceToHost)
        throw paralyn::RuntimeError(cudaErrorInvalidMemcpyDirection,
                                    "Only explicit H2D and D2H copies are supported");
      auto &ctx = paralyn::context();
      ctx.sync();
      if (!bytes)
        return cudaSuccess;
      if (!destination || !source)
        throw paralyn::RuntimeError(cudaErrorInvalidValue,
                                    "Nonempty cudaMemcpy requires nonnull pointers");
      const bool to_device = kind == cudaMemcpyHostToDevice;
      auto allocation = ctx.allocation(to_device ? destination : source);
      if (bytes > allocation->size)
        throw paralyn::RuntimeError(cudaErrorInvalidValue,
                                    "cudaMemcpy byte count exceeds allocation");
      const void *host_pointer = to_device ? source : destination;
      if (std::any_of(ctx.tokens.begin(), ctx.tokens.end(),
                      [&](const auto &token) { return token.get() == host_pointer; }))
        throw paralyn::RuntimeError(cudaErrorInvalidValue,
                                    "Host side of cudaMemcpy is a device allocation token");
      const auto copy_start = std::chrono::steady_clock::now();
      if (to_device)
        std::memcpy(allocation->buffer.contents, source, bytes);
      else
        std::memcpy(destination, allocation->buffer.contents, bytes);
      const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - copy_start).count();
      if (to_device) {
        ctx.statistics.host_to_device_seconds += seconds;
        ctx.statistics.host_to_device_bytes += bytes;
      } else {
        ctx.statistics.device_to_host_seconds += seconds;
        ctx.statistics.device_to_host_bytes += bytes;
      }
      return cudaSuccess;
    }
  } catch (...) {
    return paralyn::remember_exception();
  }
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
