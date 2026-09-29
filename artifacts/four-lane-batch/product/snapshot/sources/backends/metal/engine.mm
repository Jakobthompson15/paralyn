#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "paralyn/detail/backend.hpp"
#include "paralyn_build_info.h" // Generated at build time by cmake/build_info.cmake.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
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
DeviceInfo describe(id<MTLDevice> device) {
  DeviceInfo result{utf8(device.name), "Metal", utf8(NSProcessInfo.processInfo.operatingSystemVersionString),
                    device.registryID, device.maxBufferLength, bool(device.hasUnifiedMemory),
                    device.recommendedMaxWorkingSetSize, {}, 0, {0, 0, 0}};
  std::ostringstream identity;
  identity << "metal:registry:" << std::hex << std::setw(16) << std::setfill('0') << device.registryID;
  result.stable_id = identity.str();
  result.max_threadgroup_memory_bytes = device.maxThreadgroupMemoryLength;
  const auto limit = device.maxThreadsPerThreadgroup;
  result.max_block = {static_cast<std::uint32_t>(limit.width), static_cast<std::uint32_t>(limit.height),
                      static_cast<std::uint32_t>(limit.depth)};
  return result;
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
  std::shared_ptr<CompiledExecutable> executable;
  std::vector<std::shared_ptr<MetalBuffer>> resources;
  std::string kernel, source;
  Dim3 grid, block;
  std::size_t source_index;
  std::uint64_t operation_id = 0;
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
struct Pipeline {
  id<MTLComputePipelineState> state;
  MTLComputePipelineReflection *reflection;
};
struct MetalExecutable final : CompiledExecutable {
  std::shared_ptr<const int> owner;
  ExecutableModule description;
  std::vector<Pipeline> entries;
  // SpirvMsl only: reflected byte size of each packed scalar block, by slot.
  std::vector<std::map<std::uint32_t, std::size_t>> blocks;
};
struct MetalContext final : Context, std::enable_shared_from_this<MetalContext> {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  mutable std::mutex mutex;
  const std::shared_ptr<const int> identity = std::make_shared<const int>(0);
  const std::shared_ptr<Accounting> accounting = std::make_shared<Accounting>();
  std::unordered_map<std::string, Pipeline> pipelines;
  std::vector<std::shared_ptr<Work>> pending;
  std::vector<Execution> executions;
  std::string terminal_failure, latest_source;
  std::vector<std::string> sources;
  RuntimeStatistics counters;
  const std::uint64_t trace_id;
  std::uint64_t next_operation_id = 1;

  static std::uint64_t next_context_id() {
    static std::atomic<std::uint64_t> next{1};
    return next.fetch_add(1);
  }
  explicit MetalContext(const std::string &selector)
      : device(choose_device(selector)), trace_id(next_context_id()) {
    queue = [device newCommandQueue];
    if (!queue) throw Error(ErrorCode::execution, "Metal failed to create a command queue");
    runtime_event("context", "created", describe(device).stable_id, trace_id);
  }
  ~MetalContext() override {
    // Public close/wait reports errors; destruction still completes retained GPU work.
    try { synchronize(); } catch (...) {}
  }
  DeviceInfo device_info() const override {
    @autoreleasepool {
      return describe(device);
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
      runtime_event("transfer", "completed", "host_to_device", trace_id, 0, bytes);
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
      runtime_event("transfer", "completed", "device_to_host", trace_id, 0, bytes);
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
    if (completed && (!std::isfinite(work->info.gpu_start_seconds) ||
                      !std::isfinite(work->info.gpu_end_seconds) ||
                      !(work->info.gpu_start_seconds > 0) ||
                      !(work->info.gpu_end_seconds > work->info.gpu_start_seconds)))
      work->failure = "Metal completed " + work->kernel + " without positive GPU timing evidence";
    if (completed && work->failure.empty()) {
      work->info.duration_valid = work->info.timestamps_valid = true;
      work->info.clock_domain = 2;
      work->info.duration_seconds = work->info.gpu_end_seconds - work->info.gpu_start_seconds;
    }
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
    // Record actual completion before diagnostics: an I/O failure must not make
    // a later wait double-count the command or retry an already-consumed event.
    runtime_event("completion", work->failure.empty() ? "completed" : "failed",
                  work->failure.empty() ? work->kernel : work->failure, trace_id, work->operation_id);
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
  void log_launch(const std::string &name, Dim3 grid, Dim3 block) const {
    std::ostringstream log;
    log << "Kernel: " << name << "\nGrid: " << grid.x << " × " << grid.y << " × " << grid.z
        << "\nBlock: " << block.x << " × " << block.y << " × " << block.z << '\n';
    progress(log.str());
  }
  Pipeline pipeline(const std::string &entrypoint, const std::string &source, double &seconds) {
    const std::string key = entrypoint + '\n' + source;
    const auto found = pipelines.find(key);
    if (found != pipelines.end()) {
      seconds = 0;
      progress("Reusing kernel pipeline...\n");
      runtime_event("compilation", "reused", entrypoint, trace_id);
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
    if (!text) throw Error(ErrorCode::compilation, "Metal shader source is not valid UTF-8");
    progress("Compiling kernel...\n");
    runtime_event("compilation", "started", entrypoint, trace_id);
    id<MTLLibrary> library = [device newLibraryWithSource:text options:options error:&error];
    if (!library) throw Error(ErrorCode::compilation, "Metal shader compilation failed: " + metal_error(error));
    if (error) progress("Metal compiler: " + metal_error(error) + '\n');
    NSString *name = [NSString stringWithUTF8String:entrypoint.c_str()];
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) throw Error(ErrorCode::compilation, "Generated Metal library has no entrypoint named " + entrypoint);
    MTLComputePipelineReflection *reflection = nil;
    id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function
                         options:MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo
                      reflection:&reflection error:&error];
    if (!state) throw Error(ErrorCode::compilation, "Metal pipeline creation failed: " + metal_error(error));
    Pipeline result{state, reflection};
    pipelines.emplace(key, result);
    seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    counters.pipeline_compile_seconds += seconds;
    ++counters.pipeline_compilations;
    runtime_event("compilation", "completed", entrypoint, trace_id);
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
  void validate_reflection(const ExecutableEntry &entry, const Pipeline &pipeline) const {
    if (!pipeline.reflection)
      throw Error(ErrorCode::compilation, "Metal did not return executable argument reflection");
    std::set<unsigned> reflected;
    for (id<MTLBinding> binding in pipeline.reflection.bindings) {
      if (!binding.used) continue;
      if (binding.type != MTLBindingTypeBuffer)
        throw Error(ErrorCode::unsupported, "MSL executable profile only accepts buffer/scalar resources");
      id<MTLBufferBinding> argument = (id<MTLBufferBinding>)binding;
      const auto found = std::find_if(entry.parameters.begin(), entry.parameters.end(),
          [&](const auto &parameter) { return parameter.binding == argument.index; });
      if (found == entry.parameters.end())
        throw Error(ErrorCode::compilation, "MSL manifest omits active buffer binding " + std::to_string(argument.index));
      const auto &p = *found;
      const MTLDataType expected = p.type == ScalarType::I32 ? MTLDataTypeInt :
                                   p.type == ScalarType::U32 ? MTLDataTypeUInt : MTLDataTypeFloat;
      if (utf8(argument.name) != p.name || argument.bufferDataType != expected ||
          argument.bufferDataSize != 4 ||
          (argument.bufferPointerType && argument.bufferPointerType.elementIsArgumentBuffer))
        throw Error(ErrorCode::compilation, "MSL manifest argument type/name mismatch at " + p.name);
      const unsigned actual_access = argument.access == MTLBindingAccessReadOnly ? 1u :
                                      argument.access == MTLBindingAccessWriteOnly ? 2u : 3u;
      if ((static_cast<unsigned>(p.access) & actual_access) != actual_access ||
          (!p.buffer && actual_access != 1))
        throw Error(ErrorCode::compilation, "MSL manifest understates reflected access at " + p.name);
      if (!argument.bufferAlignment || p.alignment % argument.bufferAlignment ||
          p.minimum_bytes < argument.bufferDataSize)
        throw Error(ErrorCode::compilation, "MSL manifest understates reflected alignment/size at " + p.name);
      reflected.insert(p.binding);
    }
    if (reflected.size() != entry.parameters.size())
      throw Error(ErrorCode::compilation, "MSL manifest contains missing or inactive buffer arguments");
    if (pipeline.state.staticThreadgroupMemoryLength > device.maxThreadgroupMemoryLength)
      throw Error(ErrorCode::unsupported, "MSL static threadgroup memory exceeds device capacity");
    if (entry.required_block[0])
      geometry(pipeline.state, {1, 1, 1}, {entry.required_block[0], entry.required_block[1], entry.required_block[2]});
  }
  // SPIR-V-derived MSL wraps each resource in a SPIRV-Cross block structure, so
  // the check is structural: every active slot must match the reflected
  // descriptor's element type/offset/access, and every descriptor must be active.
  std::map<std::uint32_t, std::size_t> validate_spirv_reflection(const ExecutableEntry &entry,
                                                                 const Pipeline &pipeline) const {
    if (!pipeline.reflection)
      throw Error(ErrorCode::compilation, "Metal did not return executable argument reflection");
    std::map<std::uint32_t, std::size_t> blocks;
    std::set<std::uint32_t> reflected, declared;
    for (const auto &p : entry.parameters) declared.insert(p.binding);
    for (id<MTLBinding> binding in pipeline.reflection.bindings) {
      if (!binding.used) continue;
      if (binding.type != MTLBindingTypeBuffer)
        throw Error(ErrorCode::unsupported, "SPIR-V executable profile only accepts buffer resources");
      id<MTLBufferBinding> argument = (id<MTLBufferBinding>)binding;
      const auto slot = static_cast<std::uint32_t>(argument.index);
      std::vector<const ExecutableParameter *> at;
      for (const auto &p : entry.parameters)
        if (p.binding == slot) at.push_back(&p);
      if (at.empty())
        throw Error(ErrorCode::compilation, "SPIR-V descriptor omits active Metal buffer slot " + std::to_string(slot));
      MTLStructType *layout = argument.bufferStructType;
      if (argument.bufferDataType != MTLDataTypeStruct || !layout ||
          (argument.bufferPointerType && argument.bufferPointerType.elementIsArgumentBuffer))
        throw Error(ErrorCode::compilation, "SPIR-V Metal slot " + std::to_string(slot) + " is not a block structure");
      const unsigned actual_access = argument.access == MTLBindingAccessReadOnly ? 1u :
                                      argument.access == MTLBindingAccessWriteOnly ? 2u : 3u;
      auto expected = [](ScalarType t) {
        return t == ScalarType::I32 ? MTLDataTypeInt : t == ScalarType::U32 ? MTLDataTypeUInt : MTLDataTypeFloat;
      };
      if (at.front()->buffer) {
        const auto &p = *at.front();
        MTLStructMember *array = nil;
        for (MTLStructMember *m in layout.members)
          if (m.offset == 0) array = m;
        if (at.size() != 1 || !array || array.dataType != MTLDataTypeArray || !array.arrayType ||
            array.arrayType.elementType != expected(p.type) || array.arrayType.stride != 4)
          throw Error(ErrorCode::compilation, "SPIR-V storage buffer layout differs from descriptor at " + p.name);
        // SPIRV-Cross emits storage buffers as non-const `device` (descriptor
        // aliasing), so Metal reports qualifier-based read-write access here.
        // verify_executable re-derives the actual access from the retained
        // SPIR-V and checks the descriptor against it before this point.
        if (!argument.bufferAlignment || p.alignment % argument.bufferAlignment)
          throw Error(ErrorCode::compilation, "SPIR-V descriptor understates reflected alignment at " + p.name);
      } else {
        if (actual_access != 1)
          throw Error(ErrorCode::compilation, "SPIR-V scalar block at Metal slot " + std::to_string(slot) + " is not read-only");
        const std::size_t size = argument.bufferDataSize;
        if (!size || size > 4096)
          throw Error(ErrorCode::compilation, "SPIR-V scalar block size is outside the inline-bytes limit");
        for (const auto *p : at) {
          bool found = false;
          for (MTLStructMember *m in layout.members)
            found |= m.offset == p->block_offset && m.dataType == expected(p->type);
          if (p->buffer || !found || p->block_offset + 4 > size)
            throw Error(ErrorCode::compilation, "SPIR-V scalar block member differs from descriptor at " + p->name);
        }
        blocks[slot] = size;
      }
      reflected.insert(slot);
    }
    if (reflected != declared)
      throw Error(ErrorCode::compilation, "SPIR-V descriptor contains resources Metal reports inactive");
    if (pipeline.state.staticThreadgroupMemoryLength > device.maxThreadgroupMemoryLength)
      throw Error(ErrorCode::unsupported, "SPIR-V workgroup memory exceeds device capacity");
    geometry(pipeline.state, {1, 1, 1}, {entry.required_block[0], entry.required_block[1], entry.required_block[2]});
    return blocks;
  }
  std::shared_ptr<CompiledExecutable> prepare(const ExecutableModule &module) override {
    std::lock_guard<std::mutex> lock(mutex);
    @autoreleasepool {
      healthy();
      verify_executable(module);
      auto result = std::make_shared<MetalExecutable>();
      result->owner = identity;
      result->description = module;
      const auto source = std::string("#pragma STDC FP_CONTRACT OFF\n") + module.source;
      for (const auto &entry : module.entries) {
        double seconds = 0;
        auto compiled = pipeline(entry.name, source, seconds);
        if (module.format == ExecutableFormat::SpirvMsl)
          result->blocks.push_back(validate_spirv_reflection(entry, compiled));
        else
          validate_reflection(entry, compiled);
        result->entries.push_back(compiled);
      }
      return result;
    }
  }
  std::shared_ptr<Event> submit(const std::shared_ptr<CompiledExecutable> &value,
                              std::size_t entry_index, Dim3 grid, Dim3 block,
                              const std::vector<BoundArgument> &arguments) override {
    std::lock_guard<std::mutex> lock(mutex);
    @autoreleasepool {
      healthy();
      auto executable = std::dynamic_pointer_cast<MetalExecutable>(value);
      if (!executable || executable->owner != identity)
        throw Error(ErrorCode::invalid_handle, "Executable belongs to another context or backend");
      if (entry_index >= executable->description.entries.size())
        throw Error(ErrorCode::invalid_value, "Executable entrypoint index is out of range");
      const auto &entry = executable->description.entries[entry_index];
      const auto state = executable->entries[entry_index].state;
      if (arguments.size() != entry.parameters.size())
        throw Error(ErrorCode::invalid_value, "MSL kernel argument count mismatch");
      geometry(state, grid, block);
      if (entry.required_block[0] && (entry.required_block[0] != block.x ||
          entry.required_block[1] != block.y || entry.required_block[2] != block.z))
        throw Error(ErrorCode::invalid_value, "Launch block differs from MSL executable's required workgroup shape");
      std::unordered_map<MetalBuffer *, ResourceAccess> seen;
      std::vector<std::shared_ptr<MetalBuffer>> retained;
      for (std::size_t i = 0; i < arguments.size(); ++i) {
        const auto &a = arguments[i];
        const auto &p = entry.parameters[i];
        if (a.is_buffer != p.buffer || a.type != p.type)
          throw Error(ErrorCode::invalid_value, "MSL argument kind/type mismatch at " + p.name);
        if (p.buffer) {
          auto allocation = buffer(a.allocation);
          range(*allocation, a.offset, a.size);
          if (a.size < p.minimum_bytes || a.offset % p.alignment || a.size % 4)
            throw Error(ErrorCode::invalid_value, "MSL view violates descriptor size/alignment at " + p.name);
          // Multiple read-only bindings cannot observe one another's writes.
          // Writable aliases require shared-pointer lowering, which arbitrary
          // MSL source does not provide as generated IR does.
          const auto inserted = seen.emplace(allocation.get(), p.access);
          if (!inserted.second && (p.access != ResourceAccess::Read ||
                                   inserted.first->second != ResourceAccess::Read))
            throw Error(ErrorCode::unsupported, "MSL repeated-allocation arguments require every binding to be read-only");
          if (inserted.second) retained.push_back(std::move(allocation));
        }
      }
      log_launch(entry.name, grid, block);
      id<MTLCommandBuffer> command = [queue commandBuffer];
      if (!command) throw Error(ErrorCode::execution, "Metal could not create a command buffer");
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      if (!encoder) throw Error(ErrorCode::execution, "Metal could not create a compute encoder");
      [encoder setComputePipelineState:state];
      const bool packed = executable->description.format == ExecutableFormat::SpirvMsl;
      // SPIR-V scalars are members of one reflected block per slot; copy each
      // typed argument to its declared offset, then bind the block once.
      std::map<std::uint32_t, std::vector<unsigned char>> blocks;
      if (packed)
        for (const auto &[slot, size] : executable->blocks[entry_index]) blocks[slot].assign(size, 0);
      for (std::size_t i = 0; i < arguments.size(); ++i) {
        const auto &a = arguments[i];
        const auto slot = entry.parameters[i].binding;
        if (a.is_buffer) {
          auto allocation = buffer(a.allocation);
          [encoder setBuffer:allocation->metal offset:a.offset atIndex:slot];
        } else if (packed) {
          auto &block = blocks.at(slot);
          std::memcpy(block.data() + entry.parameters[i].block_offset, a.bytes.data(), a.bytes.size());
        } else [encoder setBytes:a.bytes.data() length:a.bytes.size() atIndex:slot];
      }
      for (const auto &[slot, bytes] : blocks)
        [encoder setBytes:bytes.data() length:bytes.size() atIndex:slot];
      [encoder dispatchThreadgroups:MTLSizeMake(grid.x, grid.y, grid.z)
              threadsPerThreadgroup:MTLSizeMake(block.x, block.y, block.z)];
      [encoder endEncoding];
      const auto source = std::string("#pragma STDC FP_CONTRACT OFF\n") + executable->description.source;
      const auto source_it = std::find(sources.begin(), sources.end(), source);
      const std::size_t source_index = std::distance(sources.begin(), source_it);
      if (source_it == sources.end()) sources.push_back(source);
      auto work = std::make_shared<Work>();
      work->command = command;
      work->executable = executable;
      work->resources = std::move(retained);
      work->kernel = entry.name; work->source = source;
      work->grid = grid; work->block = block;
      work->source_index = source_index; work->compile_seconds = 0;
      work->operation_id = next_operation_id++;
      auto event = std::make_shared<MetalEvent>(shared_from_this(), work);
      progress("Executing on GPU...\n");
      runtime_event("submission", "prepared", entry.name, trace_id, work->operation_id);
      pending.push_back(work);
      [command commit];
      return event;
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
      geometry(state.state, grid, block);
      log_launch(kernel.name, grid, block);
      id<MTLCommandBuffer> command = [queue commandBuffer];
      if (!command) throw Error(ErrorCode::execution, "Metal could not create a command buffer");
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      if (!encoder) throw Error(ErrorCode::execution, "Metal could not create a compute encoder");
      [encoder setComputePipelineState:state.state];
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
      work->operation_id = next_operation_id++;
      // Allocate event and pending storage before committing any physical work.
      auto event = std::make_shared<MetalEvent>(shared_from_this(), work);
      progress("Executing on GPU...\n");
      runtime_event("submission", "prepared", kernel.name, trace_id, work->operation_id);
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
           << ",\n  \"stable_device_id\": " << json(describe(device).stable_id)
           << ",\n  \"os\": " << json(utf8(NSProcessInfo.processInfo.operatingSystemVersionString))
           << ",\n  \"physical_memory_bytes\": " << NSProcessInfo.processInfo.physicalMemory
           << ",\n  \"unified_memory\": " << (device.hasUnifiedMemory ? "true" : "false")
           << ",\n  \"metal_language_version\": \"3.1\""
           << ",\n  \"thermal_state_at_capture\": " << NSProcessInfo.processInfo.thermalState
           << ",\n  \"low_power_mode_at_capture\": "
           << (NSProcessInfo.processInfo.lowPowerModeEnabled ? "true" : "false")
           // A caller-supplied revision (CLI, qualification harness) takes precedence;
           // direct native clients fall back to the runtime's embedded build identity.
           << ",\n  \"paralyn_commit\": "
           << json(environment("PARALYN_COMMIT", PARALYN_EMBEDDED_REVISION))
           << ",\n  \"paralyn_dirty\": "
           << (environment("PARALYN_SOURCE_DIRTY", PARALYN_EMBEDDED_DIRTY) == "true"    ? "true"
               : environment("PARALYN_SOURCE_DIRTY", PARALYN_EMBEDDED_DIRTY) == "false" ? "false"
                                                                                       : "null")
           << ",\n  \"paralyn_revision_source\": "
           << json(std::getenv("PARALYN_COMMIT") && *std::getenv("PARALYN_COMMIT")
                       ? "environment"
                       : "embedded_build")
           << ",\n  \"paralyn_runtime_build\": {\"revision\": " << json(PARALYN_EMBEDDED_REVISION)
           << ", \"dirty\": "
           << (std::string(PARALYN_EMBEDDED_DIRTY) == "true"    ? "true"
               : std::string(PARALYN_EMBEDDED_DIRTY) == "false" ? "false"
                                                                : "null")
           << ", \"changes_sha256\": "
           << (*PARALYN_EMBEDDED_CHANGES_SHA256 ? json(PARALYN_EMBEDDED_CHANGES_SHA256) : "null")
           << ", \"captured\": \"build\"}"
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
             << ", \"gpu_duration_valid\": " << (e.complete && e.error.empty() ? "true" : "false")
             << ", \"gpu_timestamps_valid\": " << (e.complete && e.error.empty() ? "true" : "false")
             << ", \"gpu_clock_domain\": \"metal_system_mach\""
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
      result.push_back(describe(device));
    return result;
  }
}
std::shared_ptr<Context> create_context(const std::string &selector) {
  @autoreleasepool { return std::make_shared<MetalContext>(selector); }
}
} // namespace paralyn::backend
