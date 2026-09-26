#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "paralyn/detail/backend.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace paralyn::backend {
namespace {
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
  std::string choice = selector;
  if (choice.rfind("metal:", 0) == 0) choice.erase(0, 6);
  if (devices.empty())
    throw Error(ErrorCode::invalid_device,
                       "No physical Metal device is available; Paralyn has no CPU fallback");
  std::size_t index = 0;
  if (choice != "auto") {
    if (choice.empty() || choice.find_first_not_of("0123456789") != std::string::npos)
      throw Error(ErrorCode::invalid_device, "Device must be auto, metal:INDEX, or a numeric index");
    try {
      index = std::stoull(choice);
    } catch (...) {
      throw Error(ErrorCode::invalid_device, "Device index is out of range");
    }
  }
  if (index >= devices.size())
    throw Error(ErrorCode::invalid_device, "Device index is out of range");
  return devices[index];
}

struct Accounting {
  std::atomic<std::uint64_t> current{0}, peak{0};
};
struct MetalBuffer final : Buffer {
  id<MTLBuffer> metal;
  std::size_t bytes;
  std::shared_ptr<const int> owner;
  std::shared_ptr<Accounting> accounting;
  MetalBuffer(id<MTLBuffer> buffer, std::size_t length, std::shared_ptr<const int> identity,
              std::shared_ptr<Accounting> counters)
      : metal(buffer), bytes(length), owner(std::move(identity)), accounting(std::move(counters)) {
    const auto current = accounting->current.fetch_add(bytes) + bytes;
    auto previous = accounting->peak.load();
    while (previous < current && !accounting->peak.compare_exchange_weak(previous, current)) {}
  }
  ~MetalBuffer() override { accounting->current.fetch_sub(bytes); }
  std::size_t size() const override { return bytes; }
};
struct Work {
  id<MTLCommandBuffer> command;
  std::vector<std::shared_ptr<MetalBuffer>> resources;
  std::string kernel, source;
  Dim3 grid, block;
  std::size_t source_index;
  double compile_seconds;
  bool recorded = false;
  EventInfo info;
  std::string failure;
};
struct Execution {
  std::string kernel;
  Dim3 grid, block;
  double start, end;
  bool complete;
  std::string error;
  std::size_t source_index;
  double compile_seconds;
};
struct MetalContext;
struct MetalEvent final : Event {
  // Context holds Work, never Events. Keeping the context here does not form a cycle.
  std::shared_ptr<MetalContext> context;
  std::shared_ptr<Work> work;
  MetalEvent(std::shared_ptr<MetalContext> c, std::shared_ptr<Work> w)
      : context(std::move(c)), work(std::move(w)) {}
  EventInfo wait() override;
};
struct Bindings {
  BindingLayout layout;
  std::vector<std::size_t> offsets;
  std::vector<std::shared_ptr<MetalBuffer>> allocations;
};
struct MetalContext final : Context, std::enable_shared_from_this<MetalContext> {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  mutable std::mutex mutex;
  const std::shared_ptr<const int> identity = std::make_shared<const int>(0);
  const std::shared_ptr<Accounting> accounting = std::make_shared<Accounting>();
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
  std::vector<std::shared_ptr<Work>> pending;
  std::vector<Execution> executions;
  std::string terminal_failure, latest_source;
  std::vector<std::string> sources;
  RuntimeStatistics counters;

  explicit MetalContext(const std::string &selector) : device(choose_device(selector)) {
    queue = [device newCommandQueue];
    if (!queue) throw Error(ErrorCode::execution, "Metal failed to create a command queue");
  }
  ~MetalContext() override {
    // Public close/wait reports errors; destruction still completes retained GPU work.
    try { synchronize(); } catch (...) {}
  }
  DeviceInfo device_info() const override {
    @autoreleasepool {
      return {utf8(device.name), "Metal", utf8(NSProcessInfo.processInfo.operatingSystemVersionString),
              device.registryID, device.maxBufferLength, bool(device.hasUnifiedMemory),
              device.recommendedMaxWorkingSetSize};
    }
  }
  std::shared_ptr<MetalBuffer> buffer(const std::shared_ptr<Buffer> &value) const {
    auto result = std::dynamic_pointer_cast<MetalBuffer>(value);
    if (!result || result->owner != identity)
      throw Error(ErrorCode::invalid_handle, "Buffer belongs to another context or backend");
    return result;
  }
  void range(const MetalBuffer &allocation, std::size_t offset, std::size_t bytes) const {
    if (offset > allocation.bytes || bytes > allocation.bytes - offset)
      throw Error(ErrorCode::invalid_value, "Buffer view or copy range exceeds allocation");
  }
  std::shared_ptr<Buffer> allocate(std::size_t bytes) override {
    std::lock_guard<std::mutex> lock(mutex);
    @autoreleasepool {
      healthy();
      if (bytes > device.maxBufferLength)
        throw Error(ErrorCode::out_of_memory, "Allocation size exceeds Metal maximum buffer length");
      id<MTLBuffer> allocation = bytes ? [device newBufferWithLength:bytes options:MTLResourceStorageModeShared] : nil;
      if (bytes && !allocation)
        throw Error(ErrorCode::out_of_memory, "Metal buffer allocation failed");
      return std::make_shared<MetalBuffer>(allocation, bytes, identity, accounting);
    }
  }
  void write(const std::shared_ptr<Buffer> &value, std::size_t offset,
             const void *source, std::size_t bytes) override {
    std::lock_guard<std::mutex> lock(mutex);
    @autoreleasepool {
      auto allocation = buffer(value);
      range(*allocation, offset, bytes);
      if (bytes && !source) throw Error(ErrorCode::invalid_value, "Nonempty write requires a host pointer");
      sync_locked();
      const auto start = std::chrono::steady_clock::now();
      if (bytes) std::memcpy(static_cast<unsigned char *>(allocation->metal.contents) + offset, source, bytes);
      counters.host_to_device_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      counters.host_to_device_bytes += bytes;
    }
  }
  void read(const std::shared_ptr<Buffer> &value, std::size_t offset,
            void *destination, std::size_t bytes) override {
    std::lock_guard<std::mutex> lock(mutex);
    @autoreleasepool {
      auto allocation = buffer(value);
      range(*allocation, offset, bytes);
      if (bytes && !destination) throw Error(ErrorCode::invalid_value, "Nonempty read requires a host pointer");
      sync_locked();
      const auto start = std::chrono::steady_clock::now();
      if (bytes) std::memcpy(destination, static_cast<unsigned char *>(allocation->metal.contents) + offset, bytes);
      counters.device_to_host_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      counters.device_to_host_bytes += bytes;
    }
  }
  RuntimeStatistics snapshot() const {
    auto result = counters;
    result.current_buffer_bytes = accounting->current.load();
    result.peak_buffer_bytes = accounting->peak.load();
    return result;
  }
  RuntimeStatistics statistics() const override {
    std::lock_guard<std::mutex> lock(mutex);
    return snapshot();
  }
  void healthy() const {
    if (!terminal_failure.empty()) throw Error(ErrorCode::execution, terminal_failure);
  }
  void finish(const std::shared_ptr<Work> &work) {
    if (work->recorded) return;
    [work->command waitUntilCompleted];
    const bool completed = work->command.status == MTLCommandBufferStatusCompleted;
    if (!completed)
      work->failure = "Metal command failed for " + work->kernel + ": " + metal_error(work->command.error);
    work->info = {completed, work->command.GPUStartTime, work->command.GPUEndTime};
    if (completed && (!(work->info.gpu_start_seconds > 0) ||
                      !(work->info.gpu_end_seconds > work->info.gpu_start_seconds)))
      work->failure = "Metal completed " + work->kernel + " without positive GPU timing evidence";
    // Complete all allocating copies before mutating counters/record state. A
    // failed host allocation must not make a later wait count this command twice.
    std::string completed_source = work->source;
    const bool first_failure = !work->failure.empty() && terminal_failure.empty();
    std::string failure_copy = first_failure ? work->failure : std::string();
    Execution execution{work->kernel, work->grid, work->block, work->info.gpu_start_seconds,
                        work->info.gpu_end_seconds, completed, work->failure,
                        work->source_index, work->compile_seconds};
    executions.push_back(std::move(execution));
    if (completed && work->failure.empty()) {
      counters.last_gpu_seconds = work->info.gpu_end_seconds - work->info.gpu_start_seconds;
      counters.gpu_seconds += counters.last_gpu_seconds;
      ++counters.completed_launches;
    }
    latest_source.swap(completed_source);
    if (first_failure) terminal_failure.swap(failure_copy);
    work->recorded = true;
    work->resources.clear();
  }
  void sync_locked() {
    // Inspect every submitted command even when a previous command failed.
    for (const auto &work : pending) finish(work);
    pending.clear();
    healthy();
  }
  void synchronize() override {
    std::lock_guard<std::mutex> lock(mutex);
    @autoreleasepool { sync_locked(); }
  }
  EventInfo wait(const std::shared_ptr<Work> &work) {
    std::lock_guard<std::mutex> lock(mutex);
    @autoreleasepool {
      // A queue event includes earlier commands, not commands submitted after this event.
      auto end = std::find(pending.begin(), pending.end(), work);
      if (end != pending.end()) {
        ++end;
        for (auto it = pending.begin(); it != end; ++it) finish(*it);
        pending.erase(pending.begin(), end);
      }
      healthy();
      if (!work->recorded) throw Error(ErrorCode::internal, "Event has no recorded command");
      return work->info;
    }
  }
  Bindings resolve(const Kernel &kernel, const std::vector<BoundArgument> &arguments) const {
    if (arguments.size() != kernel.parameters.size())
      throw Error(ErrorCode::invalid_value, "Kernel argument count mismatch");
    Bindings result;
    std::unordered_map<MetalBuffer *, std::pair<unsigned, ScalarType>> seen;
    unsigned next = 0;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
      const auto &argument = arguments[i];
      const auto &parameter = kernel.parameters[i];
      if (argument.is_buffer != parameter.buffer)
        throw Error(ErrorCode::invalid_value, "Kernel argument kind mismatch for " + parameter.name);
      if (parameter.buffer) {
        auto allocation = buffer(argument.allocation);
        range(*allocation, argument.offset, argument.size);
        if (!argument.size || argument.size % 4 || argument.offset % 4 ||
            argument.offset / 4 > UINT32_MAX || argument.size / 4 > UINT32_MAX)
          throw Error(ErrorCode::invalid_value, "Kernel buffer views require nonempty aligned 32-bit elements");
        auto inserted = seen.emplace(allocation.get(), std::make_pair(next, parameter.type));
        if (inserted.second) { ++next; result.allocations.push_back(allocation); }
        else if (inserted.first->second.second != parameter.type)
          throw Error(ErrorCode::unsupported, "Mixed pointee types for aliased allocation at parameter " + parameter.name);
        result.layout.push_back(inserted.first->second.first);
        result.offsets.push_back(argument.offset);
      } else {
        if (argument.type != parameter.type || argument.type == ScalarType::Bool)
          throw Error(ErrorCode::invalid_value, "Kernel scalar type mismatch for " + parameter.name);
        result.layout.push_back(next++);
        result.offsets.push_back(0);
      }
    }
    if (next > 31) throw Error(ErrorCode::unsupported, "Kernel exceeds the Metal buffer argument limit");
    return result;
  }
  void geometry(id<MTLComputePipelineState> pipeline, Dim3 grid, Dim3 block) const {
    if (!grid.x || !grid.y || !grid.z || !block.x || !block.y || !block.z)
      throw Error(ErrorCode::invalid_value, "Grid and block dimensions must be positive");
    const MTLSize limit = device.maxThreadsPerThreadgroup;
    if (block.x > limit.width || block.y > limit.height || block.z > limit.depth)
      throw Error(ErrorCode::invalid_value, "Block dimension exceeds Metal device limits");
    const std::uint64_t threads = std::uint64_t(block.x) * block.y * block.z;
    if (threads > pipeline.maxTotalThreadsPerThreadgroup)
      throw Error(ErrorCode::invalid_value, "Block thread count exceeds Metal pipeline limits");
    const std::uint64_t x = std::uint64_t(grid.x) * block.x;
    const std::uint64_t y = std::uint64_t(grid.y) * block.y;
    const std::uint64_t z = std::uint64_t(grid.z) * block.z;
    if (x > UINT32_MAX || y > UINT32_MAX || z > UINT32_MAX)
      throw Error(ErrorCode::invalid_value, "Logical grid dimension exceeds 32-bit indexing");
    if (x > UINT64_MAX / y || x * y > UINT64_MAX / z)
      throw Error(ErrorCode::invalid_value, "Logical grid thread count overflows 64 bits");
  }
  id<MTLComputePipelineState> pipeline(const std::string &entrypoint, const std::string &source,
                                      double &seconds) {
    const std::string key = entrypoint + '\n' + source;
    const auto found = pipelines.find(key);
    if (found != pipelines.end()) {
      seconds = 0;
      std::cout << "Reusing kernel pipeline...\n" << std::flush;
      return found->second;
    }
    const auto start = std::chrono::steady_clock::now();
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    options.languageVersion = MTLLanguageVersion3_1;
    if (@available(macOS 15.0, *)) {
      options.mathMode = MTLMathModeSafe;
      options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    } else throw Error(ErrorCode::unsupported, "Safe/precise Metal compilation requires macOS 15 or newer");
    NSError *error = nil;
    NSString *text = [[NSString alloc] initWithBytes:source.data() length:source.size()
                                          encoding:NSUTF8StringEncoding];
    std::cout << "Compiling kernel...\n" << std::flush;
    id<MTLLibrary> library = [device newLibraryWithSource:text options:options error:&error];
    if (!library) throw Error(ErrorCode::compilation, "Metal shader compilation failed: " + metal_error(error));
    if (error) std::cerr << "Metal compiler: " << metal_error(error) << '\n';
    NSString *name = [NSString stringWithUTF8String:entrypoint.c_str()];
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) throw Error(ErrorCode::compilation, "Generated Metal library has no entrypoint named " + entrypoint);
    id<MTLComputePipelineState> result = [device newComputePipelineStateWithFunction:function error:&error];
    if (!result) throw Error(ErrorCode::compilation, "Metal pipeline creation failed: " + metal_error(error));
    pipelines.emplace(key, result);
    seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    counters.pipeline_compile_seconds += seconds;
    ++counters.pipeline_compilations;
    return result;
  }
  void prepare(const Kernel &kernel) override {
    std::lock_guard<std::mutex> lock(mutex);
    @autoreleasepool {
      healthy();
      verify(kernel);
      if (kernel.parameters.size() > 31)
        throw Error(ErrorCode::unsupported, "Kernel exceeds the Metal buffer argument limit");
      BindingLayout layout;
      for (unsigned i = 0; i < kernel.parameters.size(); ++i) layout.push_back(i);
      double seconds = 0;
      pipeline("uc_kernel_" + kernel.name, emit_msl(kernel, layout), seconds);
    }
  }
  std::shared_ptr<Event> submit(const Kernel &kernel, Dim3 grid, Dim3 block,
                                const std::vector<BoundArgument> &arguments,
                                const std::string *handwritten) override {
    std::lock_guard<std::mutex> lock(mutex);
    @autoreleasepool {
      healthy();
      verify(kernel);
      Bindings bindings = resolve(kernel, arguments);
      if (handwritten && std::any_of(bindings.offsets.begin(), bindings.offsets.end(), [](auto n){ return n != 0; }))
        throw Error(ErrorCode::unsupported, "Handwritten test kernels do not support buffer-view offsets");
      const std::string source = handwritten ? *handwritten : emit_msl(kernel, bindings.layout, bindings.offsets);
      const std::string entrypoint = handwritten ? kernel.name : "uc_kernel_" + kernel.name;
      double compile_seconds = 0;
      auto state = pipeline(entrypoint, source, compile_seconds);
      geometry(state, grid, block);
      std::cout << "Kernel: " << kernel.name << "\nGrid: " << grid.x << " × " << grid.y << " × "
                << grid.z << "\nBlock: " << block.x << " × " << block.y << " × " << block.z << '\n';
      id<MTLCommandBuffer> command = [queue commandBuffer];
      if (!command) throw Error(ErrorCode::execution, "Metal could not create a command buffer");
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      if (!encoder) throw Error(ErrorCode::execution, "Metal could not create a compute encoder");
      [encoder setComputePipelineState:state];
      std::vector<bool> bound(31, false);
      for (std::size_t i = 0; i < arguments.size(); ++i) {
        const auto &argument = arguments[i];
        const unsigned slot = bindings.layout[i];
        if (bound[slot]) continue;
        if (argument.is_buffer) {
          auto allocation = buffer(argument.allocation);
          [encoder setBuffer:allocation->metal offset:0 atIndex:slot];
        } else [encoder setBytes:argument.bytes.data() length:argument.bytes.size() atIndex:slot];
        bound[slot] = true;
      }
      [encoder dispatchThreadgroups:MTLSizeMake(grid.x, grid.y, grid.z)
              threadsPerThreadgroup:MTLSizeMake(block.x, block.y, block.z)];
      [encoder endEncoding];
      const auto source_it = std::find(sources.begin(), sources.end(), source);
      const std::size_t source_index = std::distance(sources.begin(), source_it);
      if (source_it == sources.end()) sources.push_back(source);
      auto work = std::make_shared<Work>();
      work->command = command;
      work->resources = std::move(bindings.allocations);
      work->kernel = kernel.name; work->source = source;
      work->grid = grid; work->block = block;
      work->source_index = source_index; work->compile_seconds = compile_seconds;
      // Allocate event and pending storage before committing any physical work.
      auto event = std::make_shared<MetalEvent>(shared_from_this(), work);
      std::cout << "Executing on GPU...\n" << std::flush;
      // Logging can throw if a host application enables iostream exceptions.
      // Never retain an uncommitted command across a potentially throwing log.
      pending.push_back(work);
      [command commit];
      return event;
    }
  }
  void write_artifacts(const std::string &directory) {
    const auto statistics = snapshot();
    if (directory.empty() || executions.empty())
      return;
    std::filesystem::create_directories(directory);
    std::ofstream metal(std::filesystem::path(directory) / "generated.metal");
    metal << latest_source;
    metal.close();
    if (!metal)
      throw Error(ErrorCode::internal, "Cannot write generated.metal");
    for (std::size_t i = 0; i < sources.size(); ++i) {
      const auto name = "source-" + std::to_string(i) + ".metal";
      std::ofstream source(std::filesystem::path(directory) / name);
      source << sources[i];
      source.close();
      if (!source)
        throw Error(ErrorCode::internal, "Cannot write " + name);
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
      throw Error(ErrorCode::internal, "Cannot write execution.json");
  }

  void write_evidence(const std::string &directory) override {
    std::lock_guard<std::mutex> lock(mutex);
    @autoreleasepool {
      // Preserve failed command evidence as well as successful execution records.
      std::exception_ptr failure;
      try { sync_locked(); } catch (...) { failure = std::current_exception(); }
      write_artifacts(directory);
      if (failure) std::rethrow_exception(failure);
    }
  }
};
EventInfo MetalEvent::wait() { return context->wait(work); }
} // namespace
std::vector<DeviceInfo> devices() {
  @autoreleasepool {
    std::vector<DeviceInfo> result;
    for (id<MTLDevice> device : enumerate_devices())
      result.push_back({utf8(device.name), "Metal", utf8(NSProcessInfo.processInfo.operatingSystemVersionString),
                        device.registryID, device.maxBufferLength, bool(device.hasUnifiedMemory),
                        device.recommendedMaxWorkingSetSize});
    return result;
  }
}
std::shared_ptr<Context> create_context(const std::string &selector) {
  @autoreleasepool { return std::make_shared<MetalContext>(selector); }
}
} // namespace paralyn::backend
