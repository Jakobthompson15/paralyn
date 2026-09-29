// CUDA Driver API + NVRTC execution backend behind the shared Context boundary.
// Kernels are generated from verified Paralyn IR (compiler/codegen/cuda.cpp),
// compiled to PTX by NVRTC for an explicit virtual architecture, loaded by the
// driver JIT and launched on one ordered stream per context. There is no CPU
// execution path: absence of the driver, NVRTC or a device is an error.
#include "engine.hpp"
#include "paralyn/cuda_codegen.hpp"
#include "paralyn/detail/cuda_backend.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <utility>
#ifndef _WIN32
#include <sys/utsname.h>
#endif

namespace paralyn::backend::cuda {
ErrorCode map_result(CUresult code) {
  switch (code) {
  case CUDA_SUCCESS:
    return ErrorCode::internal; // callers never map success
  case CUDA_ERROR_INVALID_VALUE:
    return ErrorCode::invalid_value;
  case CUDA_ERROR_OUT_OF_MEMORY:
    return ErrorCode::out_of_memory;
  case CUDA_ERROR_NOT_INITIALIZED:
  case CUDA_ERROR_DEINITIALIZED:
  case CUDA_ERROR_STUB_LIBRARY:
  case CUDA_ERROR_INSUFFICIENT_DRIVER:
  case CUDA_ERROR_NO_DEVICE:
  case CUDA_ERROR_INVALID_DEVICE:
  case CUDA_ERROR_SYSTEM_NOT_READY:
  case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:
  case CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:
    return ErrorCode::invalid_device;
  case CUDA_ERROR_INVALID_IMAGE:
  case CUDA_ERROR_NO_BINARY_FOR_GPU:
  case CUDA_ERROR_INVALID_PTX:
  case CUDA_ERROR_JIT_COMPILER_NOT_FOUND:
  case CUDA_ERROR_UNSUPPORTED_PTX_VERSION:
  case CUDA_ERROR_INVALID_SOURCE:
  case CUDA_ERROR_NOT_FOUND:
    return ErrorCode::compilation;
  case CUDA_ERROR_INVALID_HANDLE:
  case CUDA_ERROR_INVALID_CONTEXT:
    return ErrorCode::internal;
  case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
    return ErrorCode::invalid_value;
  case CUDA_ERROR_NOT_PERMITTED:
  case CUDA_ERROR_NOT_SUPPORTED:
    return ErrorCode::unsupported;
  default:
    return ErrorCode::execution;
  }
}
// Codes NVIDIA documents as leaving the context unusable ("the context cannot be
// used anymore, and must be destroyed") or the process in an inconsistent state
// ("any further CUDA work will return the same error"), plus a destroyed
// current context, uncorrectable ECC and the conservative CUDA_ERROR_UNKNOWN.
// Reviewed against the CUresult reference (cuda-python driver bindings, CUDA 13).
bool is_sticky(CUresult code) {
  switch (code) {
  case CUDA_ERROR_ECC_UNCORRECTABLE:
  case CUDA_ERROR_CONTAINED:
  case CUDA_ERROR_ILLEGAL_ADDRESS:
  case CUDA_ERROR_LAUNCH_TIMEOUT:
  case CUDA_ERROR_CONTEXT_IS_DESTROYED:
  case CUDA_ERROR_ASSERT:
  case CUDA_ERROR_TENSOR_MEMORY_LEAK:
  case CUDA_ERROR_MPS_CLIENT_TERMINATED:
  case CUDA_ERROR_EXTERNAL_DEVICE:
  case CUDA_ERROR_HARDWARE_STACK_ERROR:
  case CUDA_ERROR_ILLEGAL_INSTRUCTION:
  case CUDA_ERROR_MISALIGNED_ADDRESS:
  case CUDA_ERROR_INVALID_ADDRESS_SPACE:
  case CUDA_ERROR_INVALID_PC:
  case CUDA_ERROR_LAUNCH_FAILED:
  case CUDA_ERROR_UNKNOWN:
    return true;
  default:
    return false;
  }
}
std::uint32_t parse_ordinal(const std::string &selector) {
  const std::string prefix = "cuda:";
  if (selector.compare(0, prefix.size(), prefix) != 0)
    throw Error(ErrorCode::invalid_device, "CUDA selectors have the form cuda:INDEX");
  const auto digits = selector.substr(prefix.size());
  if (digits.empty() || digits.size() > 10 || digits.find_first_not_of("0123456789") != std::string::npos)
    throw Error(ErrorCode::invalid_device, "CUDA selectors have the form cuda:INDEX (decimal device ordinal)");
  const auto value = std::stoull(digits);
  if (value > static_cast<unsigned long long>(std::numeric_limits<int>::max()))
    throw Error(ErrorCode::invalid_device, "CUDA device ordinal is out of range for cuda:INDEX");
  return static_cast<std::uint32_t>(value);
}
std::string choose_architecture(const NvrtcApi &nvrtc, int major, int minor) {
  const int wanted = major * 10 + minor;
  if (wanted <= 0) throw Error(ErrorCode::invalid_device, "CUDA device reports no compute capability");
  int count = 0;
  if (nvrtc.nvrtcGetNumSupportedArchs && nvrtc.nvrtcGetSupportedArchs &&
      nvrtc.nvrtcGetNumSupportedArchs(&count) == NVRTC_SUCCESS && count > 0) {
    std::vector<int> archs(static_cast<std::size_t>(count));
    if (nvrtc.nvrtcGetSupportedArchs(archs.data()) == NVRTC_SUCCESS) {
      // PTX for a lower virtual architecture is forward compatible with the
      // device; never select one above the device's capability.
      int best = 0;
      for (int arch : archs)
        if (arch <= wanted && arch > best) best = arch;
      if (!best)
        throw Error(ErrorCode::unsupported,
                    "This NVRTC supports no virtual architecture at or below compute capability " +
                        std::to_string(major) + "." + std::to_string(minor));
      return "compute_" + std::to_string(best);
    }
  }
  return "compute_" + std::to_string(wanted);
}

namespace {
[[noreturn]] void fail(const DriverApi &d, CUresult code, const std::string &operation) {
  throw Error(map_result(code), "CUDA " + operation + " failed: " + result_text(d, code));
}
void check(const DriverApi &d, CUresult code, const char *operation) {
  if (code != CUDA_SUCCESS) fail(d, code, operation);
}
std::string os_description() {
#ifdef _WIN32
  return "Windows";
#else
  struct utsname name {};
  if (uname(&name) == 0) return std::string(name.sysname) + " " + name.release;
  return "unknown";
#endif
}
std::string json(const std::string &text) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : text) {
    if (c == '"' || c == '\\') out << '\\' << char(c);
    else if (c == '\n') out << "\\n";
    else if (c == '\r') out << "\\r";
    else if (c == '\t') out << "\\t";
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
    else out << char(c);
  }
  out << '"';
  return out.str();
}
int attribute(const DriverApi &d, CUdevice device, int which) {
  int value = 0;
  check(d, d.cuDeviceGetAttribute(&value, which, device), "cuDeviceGetAttribute");
  return value;
}
struct Properties {
  DeviceInfo info;
  int cc_major = 0, cc_minor = 0, max_threads = 0;
  Dim3 max_grid{0, 0, 0};
};
Properties describe(const Api &api, std::uint32_t ordinal) {
  const auto &d = api.driver;
  CUdevice device = 0;
  check(d, d.cuDeviceGet(&device, static_cast<int>(ordinal)), "cuDeviceGet");
  Properties p;
  char name[256] = {};
  check(d, d.cuDeviceGetName(name, sizeof(name) - 1, device), "cuDeviceGetName");
  std::size_t memory = 0;
  check(d, d.cuDeviceTotalMem(&memory, device), "cuDeviceTotalMem");
  p.info.name = name;
  p.info.backend = "CUDA";
  p.info.os = os_description();
  // Legacy ABI field (Metal registry ID on Metal). CUDA has no registry ID, and
  // consumers treat 0 as "no physical device", so CUDA reports a nonzero,
  // backend-tagged enumeration value: 0x43554441'00000000 ("CUDA") | (ordinal + 1).
  // It is not a hardware identity; stable_id (UUID or PCI location) is.
  p.info.registry_id = cuda_registry_id(ordinal);
  // Total device memory: an upper bound, not free memory or a guaranteed single allocation.
  p.info.max_buffer_bytes = memory;
  p.info.unified_memory = attribute(d, device, CU_DEVICE_ATTRIBUTE_INTEGRATED) != 0;
  p.info.recommended_working_set_bytes = 0;
  p.info.max_threadgroup_memory_bytes =
      static_cast<std::uint64_t>(attribute(d, device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK));
  p.info.max_block = {static_cast<std::uint32_t>(attribute(d, device, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)),
                      static_cast<std::uint32_t>(attribute(d, device, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)),
                      static_cast<std::uint32_t>(attribute(d, device, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z))};
  p.max_grid = {static_cast<std::uint32_t>(attribute(d, device, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)),
                static_cast<std::uint32_t>(attribute(d, device, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)),
                static_cast<std::uint32_t>(attribute(d, device, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z))};
  p.max_threads = attribute(d, device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
  p.cc_major = attribute(d, device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
  p.cc_minor = attribute(d, device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
  std::ostringstream id;
  CUuuid uuid{};
  if (d.cuDeviceGetUuid && d.cuDeviceGetUuid(&uuid, device) == CUDA_SUCCESS) {
    id << "cuda:uuid:" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
      if (i == 4 || i == 6 || i == 8 || i == 10) id << '-';
      id << std::setw(2) << unsigned(uuid.bytes[i]);
    }
  } else {
    id << "cuda:pci:" << std::hex << std::setfill('0') << std::setw(4)
       << attribute(d, device, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID) << ':' << std::setw(2)
       << attribute(d, device, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID) << ':' << std::setw(2)
       << attribute(d, device, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID);
  }
  p.info.stable_id = id.str();
  return p;
}

// Owns the retained primary context. Buffers keep it alive past their Context.
struct DeviceState {
  std::shared_ptr<const Api> api;
  CUdevice device = 0;
  CUcontext primary = nullptr;
  ~DeviceState() {
    if (primary) api->driver.cuDevicePrimaryCtxRelease(device);
  }
};
// Makes the primary context current for one operation and restores the
// caller's current context (whatever it was, including none) afterwards.
class Current {
  const DriverApi &d;
  CUcontext expected;
  bool active = false;

public:
  explicit Current(const DeviceState &state) : d(state.api->driver), expected(state.primary) {
    check(d, d.cuCtxPushCurrent(expected), "cuCtxPushCurrent");
    active = true;
  }
  // Normal path: report a failure to restore the caller's context.
  void restore() {
    active = false;
    CUcontext popped = nullptr;
    const auto result = d.cuCtxPopCurrent(&popped);
    if (result != CUDA_SUCCESS) fail(d, result, "cuCtxPopCurrent");
    if (popped != expected)
      throw Error(ErrorCode::internal, "CUDA context stack was modified during a Paralyn operation");
  }
  ~Current() {
    if (active) {
      CUcontext popped = nullptr;
      d.cuCtxPopCurrent(&popped); // exceptional path; the primary error is already propagating
    }
  }
  Current(const Current &) = delete;
  Current &operator=(const Current &) = delete;
};
template <class F> auto with_context(const DeviceState &state, F &&fn) -> decltype(fn()) {
  Current current(state);
  if constexpr (std::is_void_v<decltype(fn())>) {
    fn();
    current.restore();
  } else {
    auto value = fn();
    current.restore();
    return value;
  }
}

struct Accounting {
  std::atomic<std::uint64_t> current{0}, peak{0};
};
struct CudaBuffer final : Buffer {
  std::shared_ptr<DeviceState> state;
  CUdeviceptr pointer = 0;
  std::size_t bytes = 0;
  std::shared_ptr<const int> owner;
  std::shared_ptr<Accounting> accounting;
  CudaBuffer(std::shared_ptr<DeviceState> s, CUdeviceptr p, std::size_t n,
             std::shared_ptr<const int> identity, std::shared_ptr<Accounting> counters)
      : state(std::move(s)), pointer(p), bytes(n), owner(std::move(identity)),
        accounting(std::move(counters)) {
    const auto current = accounting->current.fetch_add(bytes) + bytes;
    auto previous = accounting->peak.load();
    while (previous < current && !accounting->peak.compare_exchange_weak(previous, current)) {}
  }
  ~CudaBuffer() override {
    accounting->current.fetch_sub(bytes);
    if (!pointer) return;
    try {
      Current current(*state);
      state->api->driver.cuMemFree(pointer);
    } catch (...) {
      // Destruction cannot report; a failed context push leaves the allocation
      // to the driver's context teardown.
    }
  }
  std::size_t size() const override { return bytes; }
};
struct Compiled {
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  std::string entrypoint, source, ptx, log;
  int max_threads = 0;
  double compile_seconds = 0;
};
struct Work {
  CUevent start = nullptr, end = nullptr;
  std::vector<std::shared_ptr<CudaBuffer>> resources;
  std::string kernel;
  Dim3 grid, block;
  std::size_t source_index = 0;
  std::uint64_t operation_id = 0;
  double compile_seconds = 0;
  bool recorded = false;
  EventInfo info;
  std::string failure;
};
struct Execution {
  std::string kernel;
  Dim3 grid, block;
  double duration;
  bool complete;
  std::string error;
  std::size_t source_index;
  double compile_seconds;
};
struct CudaContext;
struct CudaEvent final : Event {
  std::shared_ptr<CudaContext> context;
  std::shared_ptr<Work> work;
  CudaEvent(std::shared_ptr<CudaContext> c, std::shared_ptr<Work> w)
      : context(std::move(c)), work(std::move(w)) {}
  EventInfo wait() override;
};
struct CudaExecutable final : CompiledExecutable {};

struct CudaContext final : Context, std::enable_shared_from_this<CudaContext> {
  std::shared_ptr<DeviceState> state;
  const DriverApi &d;
  const NvrtcApi &nvrtc;
  Properties properties;
  std::string architecture;
  std::vector<std::string> options;
  CUstream stream = nullptr;
  mutable std::mutex mutex;
  const std::shared_ptr<const int> identity = std::make_shared<const int>(0);
  const std::shared_ptr<Accounting> accounting = std::make_shared<Accounting>();
  std::unordered_map<std::string, std::shared_ptr<Compiled>> modules;
  std::vector<std::shared_ptr<Compiled>> sources; // evidence order
  std::vector<std::shared_ptr<Work>> pending;
  std::vector<Execution> executions;
  std::string terminal_failure;
  RuntimeStatistics counters;
  const std::uint64_t trace_id;
  std::uint64_t next_operation_id = 1;
  std::uint32_t ordinal;

  static std::uint64_t next_context_id() {
    // Distinct range from Metal context identities in shared event logs.
    static std::atomic<std::uint64_t> next{1};
    return (std::uint64_t(1) << 32) | next.fetch_add(1);
  }
  CudaContext(std::shared_ptr<const Api> api, std::uint32_t index)
      : state(std::make_shared<DeviceState>()), d(api->driver), nvrtc(api->nvrtc),
        trace_id(next_context_id()), ordinal(index) {
    state->api = std::move(api);
    properties = describe(*state->api, ordinal);
    check(d, d.cuDeviceGet(&state->device, static_cast<int>(ordinal)), "cuDeviceGet");
    // The primary context is the one the CUDA runtime API uses, so allocations
    // interoperate with a caller's CUDA runtime code on the same device.
    check(d, d.cuDevicePrimaryCtxRetain(&state->primary, state->device), "cuDevicePrimaryCtxRetain");
    architecture = choose_architecture(nvrtc, properties.cc_major, properties.cc_minor);
    options = {"--gpu-architecture=" + architecture};
    for (const auto &option : cuda_numerical_options()) options.push_back(option);
    with_context(*state, [&] { check(d, d.cuStreamCreate(&stream, CU_STREAM_DEFAULT), "cuStreamCreate"); });
    runtime_event("context", "created", properties.info.stable_id, trace_id);
  }
  ~CudaContext() override {
    try { synchronize(); } catch (...) {}
    try {
      Current current(*state);
      for (auto &work : pending) release_events(*work);
      if (stream) d.cuStreamDestroy(stream);
      for (auto &entry : modules)
        if (entry.second->module) d.cuModuleUnload(entry.second->module);
    } catch (...) {}
  }
  void release_events(Work &work) {
    if (work.start) d.cuEventDestroy(work.start);
    if (work.end) d.cuEventDestroy(work.end);
    work.start = work.end = nullptr;
  }
  DeviceInfo device_info() const override { return properties.info; }
  void healthy() const {
    if (!terminal_failure.empty()) throw Error(ErrorCode::execution, terminal_failure);
  }
  // A sticky driver fault makes this context unusable; record it before throwing.
  [[noreturn]] void driver_failure(CUresult code, const std::string &operation) {
    const std::string text = "CUDA " + operation + " failed: " + result_text(d, code);
    if (is_sticky(code) && terminal_failure.empty()) terminal_failure = text;
    throw Error(map_result(code), text);
  }
  std::shared_ptr<CudaBuffer> buffer(const std::shared_ptr<Buffer> &value) const {
    auto result = std::dynamic_pointer_cast<CudaBuffer>(value);
    if (!result || result->owner != identity)
      throw Error(ErrorCode::invalid_handle, "Buffer belongs to another context or backend");
    return result;
  }
  static void range(const CudaBuffer &allocation, std::size_t offset, std::size_t bytes) {
    if (offset > allocation.bytes || bytes > allocation.bytes - offset)
      throw Error(ErrorCode::invalid_value, "Buffer view or copy range exceeds allocation");
  }
  std::shared_ptr<Buffer> allocate(std::size_t bytes) override {
    std::lock_guard<std::mutex> lock(mutex);
    healthy();
    CUdeviceptr pointer = 0;
    if (bytes) {
      Current current(*state);
      const auto result = d.cuMemAlloc(&pointer, bytes);
      if (result != CUDA_SUCCESS) driver_failure(result, "cuMemAlloc");
      current.restore();
    }
    try {
      return std::make_shared<CudaBuffer>(state, pointer, bytes, identity, accounting);
    } catch (...) {
      if (pointer) {
        Current current(*state);
        d.cuMemFree(pointer);
      }
      throw;
    }
  }
  void write(const std::shared_ptr<Buffer> &value, std::size_t offset, const void *source,
             std::size_t bytes) override {
    std::lock_guard<std::mutex> lock(mutex);
    auto allocation = buffer(value);
    range(*allocation, offset, bytes);
    if (bytes && !source) throw Error(ErrorCode::invalid_value, "Nonempty write requires a host pointer");
    sync_locked();
    const auto start = std::chrono::steady_clock::now();
    if (bytes) {
      Current current(*state);
      const auto result = d.cuMemcpyHtoD(allocation->pointer + offset, source, bytes);
      if (result != CUDA_SUCCESS) driver_failure(result, "cuMemcpyHtoD");
      current.restore();
    }
    counters.host_to_device_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    counters.host_to_device_bytes += bytes;
    runtime_event("transfer", "completed", "host_to_device", trace_id, 0, bytes);
  }
  void read(const std::shared_ptr<Buffer> &value, std::size_t offset, void *destination,
            std::size_t bytes) override {
    std::lock_guard<std::mutex> lock(mutex);
    auto allocation = buffer(value);
    range(*allocation, offset, bytes);
    if (bytes && !destination) throw Error(ErrorCode::invalid_value, "Nonempty read requires a host pointer");
    sync_locked();
    const auto start = std::chrono::steady_clock::now();
    if (bytes) {
      Current current(*state);
      const auto result = d.cuMemcpyDtoH(destination, allocation->pointer + offset, bytes);
      if (result != CUDA_SUCCESS) driver_failure(result, "cuMemcpyDtoH");
      current.restore();
    }
    counters.device_to_host_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    counters.device_to_host_bytes += bytes;
    runtime_event("transfer", "completed", "device_to_host", trace_id, 0, bytes);
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
  std::string nvrtc_error(nvrtcResult code) const {
    const char *text = nvrtc.nvrtcGetErrorString ? nvrtc.nvrtcGetErrorString(code) : nullptr;
    return std::string(text ? text : "NVRTC error") + " (" + std::to_string(code) + ")";
  }
  std::shared_ptr<Compiled> compile(const std::string &entrypoint, const std::string &source,
                                    double &seconds) {
    const auto found = modules.find(source);
    if (found != modules.end()) {
      seconds = 0;
      progress("Reusing kernel pipeline...\n");
      runtime_event("compilation", "reused", entrypoint, trace_id);
      return found->second;
    }
    const auto start = std::chrono::steady_clock::now();
    progress("Compiling kernel...\n");
    runtime_event("compilation", "started", entrypoint, trace_id);
    auto compiled = std::make_shared<Compiled>();
    compiled->entrypoint = entrypoint;
    compiled->source = source;
    nvrtcProgram program = nullptr;
    auto result = nvrtc.nvrtcCreateProgram(&program, source.c_str(), "paralyn_generated.cu", 0, nullptr, nullptr);
    if (result != NVRTC_SUCCESS)
      throw Error(ErrorCode::compilation, "NVRTC could not create a program: " + nvrtc_error(result));
    struct Destroy {
      const NvrtcApi &api;
      nvrtcProgram &program;
      ~Destroy() { api.nvrtcDestroyProgram(&program); }
    } destroy{nvrtc, program};
    std::vector<const char *> arguments;
    for (const auto &option : options) arguments.push_back(option.c_str());
    const auto compiled_status = nvrtc.nvrtcCompileProgram(program, static_cast<int>(arguments.size()), arguments.data());
    std::size_t log_size = 0;
    if (nvrtc.nvrtcGetProgramLogSize(program, &log_size) == NVRTC_SUCCESS && log_size > 1) {
      std::string log(log_size, '\0');
      if (nvrtc.nvrtcGetProgramLog(program, log.data()) == NVRTC_SUCCESS) {
        log.resize(std::strlen(log.c_str()));
        compiled->log = log;
      }
    }
    if (compiled_status != NVRTC_SUCCESS)
      throw Error(ErrorCode::compilation, "NVRTC compilation of " + entrypoint + " failed: " +
                                              nvrtc_error(compiled_status) + "\n" + compiled->log);
    std::size_t ptx_size = 0;
    result = nvrtc.nvrtcGetPTXSize(program, &ptx_size);
    if (result != NVRTC_SUCCESS || !ptx_size)
      throw Error(ErrorCode::compilation, "NVRTC returned no PTX: " + nvrtc_error(result));
    compiled->ptx.assign(ptx_size, '\0');
    result = nvrtc.nvrtcGetPTX(program, compiled->ptx.data());
    if (result != NVRTC_SUCCESS) throw Error(ErrorCode::compilation, "NVRTC PTX retrieval failed: " + nvrtc_error(result));
    compiled->ptx.resize(std::strlen(compiled->ptx.c_str()));
    {
      Current current(*state);
      char error_log[8192] = {}, info_log[8192] = {};
      CUjit_option keys[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                             CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES};
      void *values[] = {error_log, reinterpret_cast<void *>(std::uintptr_t(sizeof(error_log) - 1)),
                        info_log, reinterpret_cast<void *>(std::uintptr_t(sizeof(info_log) - 1))};
      const auto loaded = d.cuModuleLoadDataEx(&compiled->module, compiled->ptx.c_str(), 4, keys, values);
      if (loaded != CUDA_SUCCESS) {
        std::string text = "CUDA cuModuleLoadDataEx failed for " + entrypoint + " (" + architecture +
                           ", NVRTC PTX): " + result_text(d, loaded);
        if (loaded == CUDA_ERROR_UNSUPPORTED_PTX_VERSION)
          text += ". The installed driver is older than NVRTC; install a newer NVIDIA driver or "
                  "select an NVRTC no newer than the driver (PARALYN_NVRTC_LIBRARY)";
        if (*error_log) text += "\nJIT log: " + std::string(error_log);
        if (is_sticky(loaded) && terminal_failure.empty()) terminal_failure = text;
        throw Error(map_result(loaded), text);
      }
      if (*info_log) compiled->log += std::string(compiled->log.empty() ? "" : "\n") + info_log;
      const auto function = d.cuModuleGetFunction(&compiled->function, compiled->module, entrypoint.c_str());
      if (function != CUDA_SUCCESS) {
        d.cuModuleUnload(compiled->module);
        compiled->module = nullptr;
        driver_failure(function, "cuModuleGetFunction(" + entrypoint + ")");
      }
      const auto limit = d.cuFuncGetAttribute(&compiled->max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                              compiled->function);
      if (limit != CUDA_SUCCESS) {
        d.cuModuleUnload(compiled->module);
        compiled->module = nullptr;
        driver_failure(limit, "cuFuncGetAttribute");
      }
      current.restore();
    }
    seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    compiled->compile_seconds = seconds;
    try {
      modules.emplace(source, compiled);
    } catch (...) {
      Current current(*state);
      d.cuModuleUnload(compiled->module);
      throw;
    }
    counters.pipeline_compile_seconds += seconds;
    ++counters.pipeline_compilations;
    runtime_event("compilation", "completed", entrypoint, trace_id);
    return compiled;
  }
  void prepare(const Kernel &kernel) override {
    std::lock_guard<std::mutex> lock(mutex);
    healthy();
    verify(kernel);
    if (kernel.parameters.size() > 31)
      throw Error(ErrorCode::unsupported, "Kernel exceeds the portable 31-parameter limit");
    double seconds = 0;
    compile(cuda_entrypoint(kernel), emit_cuda(kernel), seconds);
  }
  std::shared_ptr<CompiledExecutable> prepare(const ExecutableModule &) override {
    throw Error(ErrorCode::unsupported,
                "MSL executable modules are Metal-specific; the CUDA backend accepts verified IR "
                "modules only");
  }
  std::shared_ptr<Event> submit(const std::shared_ptr<CompiledExecutable> &, std::size_t, Dim3, Dim3,
                                const std::vector<BoundArgument> &) override {
    throw Error(ErrorCode::invalid_handle, "Executable belongs to another backend");
  }
  void geometry(const Compiled &compiled, Dim3 grid, Dim3 block) const {
    if (!grid.x || !grid.y || !grid.z || !block.x || !block.y || !block.z)
      throw Error(ErrorCode::invalid_value, "Grid and block dimensions must be positive");
    const auto &limit = properties.info.max_block;
    if (block.x > limit.x || block.y > limit.y || block.z > limit.z)
      throw Error(ErrorCode::invalid_value, "Block dimension exceeds CUDA device limits");
    const std::uint64_t threads = std::uint64_t(block.x) * block.y * block.z;
    if (threads > std::uint64_t(std::max(0, properties.max_threads)) ||
        threads > std::uint64_t(std::max(0, compiled.max_threads)))
      throw Error(ErrorCode::invalid_value, "Block thread count exceeds CUDA device/function limits");
    if (grid.x > properties.max_grid.x || grid.y > properties.max_grid.y || grid.z > properties.max_grid.z)
      throw Error(ErrorCode::invalid_value, "Grid dimension exceeds CUDA device limits");
    const std::uint64_t x = std::uint64_t(grid.x) * block.x;
    const std::uint64_t y = std::uint64_t(grid.y) * block.y;
    const std::uint64_t z = std::uint64_t(grid.z) * block.z;
    if (x > UINT32_MAX || y > UINT32_MAX || z > UINT32_MAX)
      throw Error(ErrorCode::invalid_value, "Logical grid dimension exceeds 32-bit indexing");
    if (x > UINT64_MAX / y || x * y > UINT64_MAX / z)
      throw Error(ErrorCode::invalid_value, "Logical grid thread count overflows 64 bits");
  }
  std::shared_ptr<Event> submit(const Kernel &kernel, Dim3 grid, Dim3 block,
                                const std::vector<BoundArgument> &arguments,
                                const std::string *handwritten) override {
    std::lock_guard<std::mutex> lock(mutex);
    healthy();
    if (handwritten)
      throw Error(ErrorCode::unsupported, "Handwritten MSL test kernels are Metal-only");
    verify(kernel);
    if (arguments.size() != kernel.parameters.size())
      throw Error(ErrorCode::invalid_value, "Kernel argument count mismatch");
    if (kernel.parameters.size() > 31)
      throw Error(ErrorCode::unsupported, "Kernel exceeds the portable 31-parameter limit");
    // Argument storage must stay valid until cuLaunchKernel returns.
    std::vector<CUdeviceptr> pointers(arguments.size(), 0);
    std::vector<std::array<unsigned char, 4>> scalars(arguments.size());
    std::vector<void *> parameters(arguments.size(), nullptr);
    std::vector<std::shared_ptr<CudaBuffer>> retained;
    std::unordered_map<CudaBuffer *, ScalarType> seen;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
      const auto &argument = arguments[i];
      const auto &parameter = kernel.parameters[i];
      if (argument.is_buffer != parameter.buffer)
        throw Error(ErrorCode::invalid_value, "Kernel argument kind mismatch for " + parameter.name);
      if (parameter.buffer) {
        auto allocation = buffer(argument.allocation);
        range(*allocation, argument.offset, argument.size);
        if (!argument.size || argument.size % 4 || argument.offset % 4)
          throw Error(ErrorCode::invalid_value, "Kernel buffer views require nonempty aligned 32-bit elements");
        // Same portable contract as Metal: same-allocation aliases must share a pointee type.
        const auto inserted = seen.emplace(allocation.get(), parameter.type);
        if (!inserted.second && inserted.first->second != parameter.type)
          throw Error(ErrorCode::unsupported, "Mixed pointee types for aliased allocation at parameter " + parameter.name);
        pointers[i] = allocation->pointer + argument.offset;
        parameters[i] = &pointers[i];
        if (inserted.second) retained.push_back(std::move(allocation));
      } else {
        if (argument.type != parameter.type || argument.type == ScalarType::Bool)
          throw Error(ErrorCode::invalid_value, "Kernel scalar type mismatch for " + parameter.name);
        scalars[i] = argument.bytes;
        parameters[i] = scalars[i].data();
      }
    }
    const auto source = emit_cuda(kernel);
    const auto entrypoint = cuda_entrypoint(kernel);
    double compile_seconds = 0;
    auto compiled = compile(entrypoint, source, compile_seconds);
    geometry(*compiled, grid, block);
    std::ostringstream log;
    log << "Kernel: " << kernel.name << "\nGrid: " << grid.x << " × " << grid.y << " × " << grid.z
        << "\nBlock: " << block.x << " × " << block.y << " × " << block.z << '\n';
    progress(log.str());
    const auto source_it = std::find(sources.begin(), sources.end(), compiled);
    const std::size_t source_index = std::distance(sources.begin(), source_it);
    if (source_it == sources.end()) sources.push_back(compiled);
    auto work = std::make_shared<Work>();
    work->resources = std::move(retained);
    work->kernel = kernel.name;
    work->grid = grid;
    work->block = block;
    work->source_index = source_index;
    work->compile_seconds = compile_seconds;
    work->operation_id = next_operation_id++;
    auto event = std::make_shared<CudaEvent>(shared_from_this(), work);
    pending.reserve(pending.size() + 1); // no allocation after the launch is enqueued
    progress(state->api->test_double ? "Submitting to CUDA driver test double (nothing executes)...\n"
                                     : "Executing on GPU...\n");
    runtime_event("submission", "prepared", kernel.name, trace_id, work->operation_id);
    Current current(*state);
    auto created = d.cuEventCreate(&work->start, CU_EVENT_DEFAULT);
    if (created == CUDA_SUCCESS) created = d.cuEventCreate(&work->end, CU_EVENT_DEFAULT);
    if (created != CUDA_SUCCESS) {
      release_events(*work);
      driver_failure(created, "cuEventCreate");
    }
    auto result = d.cuEventRecord(work->start, stream);
    if (result != CUDA_SUCCESS) {
      release_events(*work);
      driver_failure(result, "cuEventRecord");
    }
    result = d.cuLaunchKernel(compiled->function, grid.x, grid.y, grid.z, block.x, block.y, block.z,
                              0, stream, parameters.data(), nullptr);
    if (result != CUDA_SUCCESS) {
      // The start event may already be enqueued; wait for it before destroying.
      d.cuEventSynchronize(work->start);
      release_events(*work);
      driver_failure(result, "cuLaunchKernel(" + entrypoint + ")");
    }
    // From here the launch is enqueued: always track it so completion is observed.
    pending.push_back(work);
    result = d.cuEventRecord(work->end, stream);
    if (result != CUDA_SUCCESS) {
      // Without an end event this command's completion cannot be observed
      // individually; synchronize the stream and fail the context honestly.
      const auto text = "CUDA cuEventRecord after launch failed: " + result_text(d, result);
      d.cuStreamSynchronize(stream);
      if (terminal_failure.empty()) terminal_failure = text;
      work->failure = text;
      throw Error(map_result(result), text);
    }
    current.restore();
    return event;
  }
  void finish(const std::shared_ptr<Work> &work) {
    if (work->recorded) return;
    std::string failure = work->failure;
    double milliseconds = 0;
    bool completed = false;
    if (failure.empty()) {
      Current current(*state);
      auto result = d.cuEventSynchronize(work->end);
      if (result != CUDA_SUCCESS)
        failure = "CUDA command failed for " + work->kernel + ": " + result_text(d, result);
      else {
        completed = true;
        float ms = 0;
        result = d.cuEventElapsedTime(&ms, work->start, work->end);
        if (result != CUDA_SUCCESS)
          failure = "CUDA completed " + work->kernel + " without GPU duration evidence: " + result_text(d, result);
        else if (!std::isfinite(ms) || ms < 0)
          failure = "CUDA completed " + work->kernel + " with invalid GPU duration";
        else
          milliseconds = ms;
      }
      release_events(*work);
      current.restore();
    } else if (work->start || work->end) {
      // Submission already failed (end-event record after launch). The events
      // still belong to the primary context, so destroy them with it current.
      // Best effort: the command's failure is already recorded and reported.
      try {
        Current current(*state);
        release_events(*work);
        current.restore();
      } catch (...) {
      }
    }
    work->info = {};
    work->info.completed = completed;
    if (completed && failure.empty()) {
      // Duration only: CUDA events have no documented absolute clock domain.
      work->info.duration_valid = true;
      work->info.timestamps_valid = false;
      work->info.clock_domain = 1;
      work->info.duration_seconds = milliseconds / 1000.0;
    }
    const bool first_failure = !failure.empty() && terminal_failure.empty();
    std::string failure_copy = first_failure ? failure : std::string();
    Execution execution{work->kernel, work->grid, work->block, work->info.duration_seconds,
                        completed, failure, work->source_index, work->compile_seconds};
    executions.push_back(std::move(execution));
    if (completed && failure.empty()) {
      counters.last_gpu_seconds = work->info.duration_seconds;
      counters.gpu_seconds += counters.last_gpu_seconds;
      ++counters.completed_launches;
    }
    work->failure = failure;
    if (first_failure) terminal_failure.swap(failure_copy);
    work->recorded = true;
    work->resources.clear();
    runtime_event("completion", work->failure.empty() ? "completed" : "failed",
                  work->failure.empty() ? work->kernel : work->failure, trace_id, work->operation_id);
  }
  void sync_locked() {
    for (const auto &work : pending) finish(work);
    pending.clear();
    healthy();
  }
  void synchronize() override {
    std::lock_guard<std::mutex> lock(mutex);
    sync_locked();
  }
  EventInfo wait(const std::shared_ptr<Work> &work) {
    std::lock_guard<std::mutex> lock(mutex);
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
  void write_artifacts(const std::string &directory) {
    const auto statistics = snapshot();
    if (directory.empty() || executions.empty()) return;
    namespace fs = std::filesystem;
    fs::create_directories(directory);
    auto save = [&](const std::string &name, const std::string &text) {
      std::ofstream file(fs::path(directory) / name, std::ios::binary);
      file << text;
      file.close();
      if (!file) throw Error(ErrorCode::internal, "Cannot write " + name);
    };
    for (std::size_t i = 0; i < sources.size(); ++i) {
      save("source-" + std::to_string(i) + ".cu", sources[i]->source);
      save("source-" + std::to_string(i) + ".ptx", sources[i]->ptx);
      if (!sources[i]->log.empty()) save("source-" + std::to_string(i) + ".log", sources[i]->log);
    }
    const auto &api = *state->api;
    int nvrtc_major = 0, nvrtc_minor = 0, driver = 0;
    if (nvrtc.nvrtcVersion) nvrtc.nvrtcVersion(&nvrtc_major, &nvrtc_minor);
    if (d.cuDriverGetVersion) d.cuDriverGetVersion(&driver);
    std::ostringstream out;
    out << std::setprecision(17) << "{\n  \"backend\": \"CUDA\",\n  \"device\": " << json(properties.info.name)
        << ",\n  \"device_ordinal\": " << ordinal
        << ",\n  \"stable_device_id\": " << json(properties.info.stable_id)
        << ",\n  \"compute_capability\": " << json(std::to_string(properties.cc_major) + "." + std::to_string(properties.cc_minor))
        << ",\n  \"nvrtc_architecture\": " << json(architecture)
        << ",\n  \"nvrtc_options\": [";
    for (std::size_t i = 0; i < options.size(); ++i) out << (i ? ", " : "") << json(options[i]);
    out << "],\n  \"cuda_driver_version\": " << driver
        << ",\n  \"nvrtc_version\": " << json(std::to_string(nvrtc_major) + "." + std::to_string(nvrtc_minor))
        << ",\n  \"driver_library\": " << json(api.driver_path)
        << ",\n  \"nvrtc_library\": " << json(api.nvrtc_path)
        << ",\n  \"test_double\": " << (api.test_double ? "true" : "false")
        << ",\n  \"os\": " << json(properties.info.os)
        << ",\n  \"context\": \"device primary context; caller current context restored after each operation\""
        << ",\n  \"cpu_fallback\": false"
        << ",\n  \"runtime_owned_current_buffer_bytes\": " << statistics.current_buffer_bytes
        << ",\n  \"runtime_owned_peak_buffer_bytes\": " << statistics.peak_buffer_bytes
        << ",\n  \"buffer_accounting\": \"requested cuMemAlloc sizes; excludes driver/module allocations\""
        << ",\n  \"launches\": [\n";
    for (std::size_t i = 0; i < executions.size(); ++i) {
      const auto &e = executions[i];
      const bool valid = e.complete && e.error.empty();
      out << "    {\"kernel\": " << json(e.kernel) << ", \"grid\": [" << e.grid.x << ',' << e.grid.y << ','
          << e.grid.z << "], \"block\": [" << e.block.x << ',' << e.block.y << ',' << e.block.z
          << "], \"command_status\": " << json(e.complete ? "completed" : "failed")
          << ", \"gpu_duration_seconds\": ";
      if (valid) out << e.duration; else out << "null";
      out << ", \"gpu_duration_valid\": " << (valid ? "true" : "false")
          << ", \"gpu_start_seconds\": null, \"gpu_end_seconds\": null"
          << ", \"gpu_timestamps_valid\": false, \"gpu_clock_domain\": \"duration_only\""
          << ", \"pipeline_compile_seconds\": " << e.compile_seconds
          << ", \"source_file\": " << json("source-" + std::to_string(e.source_index) + ".cu")
          << ", \"ptx_file\": " << json("source-" + std::to_string(e.source_index) + ".ptx")
          << ", \"error\": " << json(e.error) << '}' << (i + 1 == executions.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    save("execution.json", out.str());
  }
  void write_evidence(const std::string &directory) override {
    std::lock_guard<std::mutex> lock(mutex);
    std::exception_ptr failure;
    try { sync_locked(); } catch (...) { failure = std::current_exception(); }
    write_artifacts(directory);
    if (failure) std::rethrow_exception(failure);
  }
};
EventInfo CudaEvent::wait() { return context->wait(work); }
} // namespace

std::vector<DeviceInfo> devices(const Api &api) {
  int count = 0;
  check(api.driver, api.driver.cuDeviceGetCount(&count), "cuDeviceGetCount");
  std::vector<DeviceInfo> result;
  for (int i = 0; i < count; ++i) result.push_back(describe(api, static_cast<std::uint32_t>(i)).info);
  return result;
}
std::shared_ptr<Context> create_context(std::shared_ptr<const Api> api, std::uint32_t ordinal) {
  if (!api) throw Error(ErrorCode::invalid_device, "CUDA backend is unavailable");
  int count = 0;
  check(api->driver, api->driver.cuDeviceGetCount(&count), "cuDeviceGetCount");
  if (ordinal >= static_cast<std::uint32_t>(std::max(count, 0)))
    throw Error(ErrorCode::invalid_device, "CUDA device index is out of range (" + std::to_string(count) +
                                               " device(s) reported by the driver)");
  return std::make_shared<CudaContext>(std::move(api), ordinal);
}

Status status() {
  const auto &loaded = cached_load();
  Status s;
  s.driver_loaded = loaded.driver_loaded;
  s.nvrtc_loaded = loaded.nvrtc_loaded;
  s.initialized = loaded.initialized;
  s.available = loaded.available();
  s.driver_version = loaded.driver_version;
  s.nvrtc_major = loaded.nvrtc_major;
  s.nvrtc_minor = loaded.nvrtc_minor;
  s.device_count = loaded.device_count;
  s.driver_library = loaded.driver_path;
  s.nvrtc_library = loaded.nvrtc_path;
  s.reason = loaded.reason;
  s.attempts = loaded.attempts;
  return s;
}
std::vector<DeviceInfo> devices() {
  const auto &loaded = cached_load();
  if (!loaded.available()) return {};
  return devices(*loaded.api);
}
std::shared_ptr<Context> create_context(const std::string &selector) {
  const auto ordinal = parse_ordinal(selector);
  const auto &loaded = cached_load();
  if (!loaded.available())
    throw Error(ErrorCode::invalid_device, "CUDA backend unavailable: " + loaded.reason +
                                               ". Paralyn has no CPU fallback.");
  return create_context(loaded.api, ordinal);
}
} // namespace paralyn::backend::cuda
