// TEST DOUBLE ONLY: see fake_driver.hpp. Never executes kernels.
#include "fake_driver.hpp"
#include <cstring>

namespace fake_cuda {
namespace {
State global;
thread_local std::vector<CUcontext> stack;
int injected(const char *name) {
  auto it = global.fail_next.find(name);
  if (it == global.fail_next.end()) return CUDA_SUCCESS;
  auto skip = global.fail_after.find(name);
  if (skip != global.fail_after.end() && skip->second > 0) {
    --skip->second; // let this call succeed; fail a later one
    return CUDA_SUCCESS;
  }
  const int code = it->second;
  global.fail_next.erase(it);
  return code;
}
// Every operation that needs a context must run with the primary current.
bool current(const char *name) {
  if (stack.empty() || stack.back() != global.primary) {
    global.context_violations.push_back(name);
    return false;
  }
  return true;
}
#define FAKE_INJECT(name)                                                                          \
  if (int r = injected(name)) return r
#define FAKE_CURRENT(name)                                                                         \
  if (!current(name)) return CUDA_ERROR_INVALID_CONTEXT

CUresult init(unsigned) { FAKE_INJECT("cuInit"); return CUDA_SUCCESS; }
CUresult driver_version(int *v) { *v = 12080; return CUDA_SUCCESS; }
CUresult error_name(CUresult code, const char **out) {
  switch (code) {
  case CUDA_ERROR_OUT_OF_MEMORY: *out = "CUDA_ERROR_OUT_OF_MEMORY"; break;
  case CUDA_ERROR_INVALID_PTX: *out = "CUDA_ERROR_INVALID_PTX"; break;
  case CUDA_ERROR_ILLEGAL_ADDRESS: *out = "CUDA_ERROR_ILLEGAL_ADDRESS"; break;
  case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: *out = "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES"; break;
  default: return CUDA_ERROR_INVALID_VALUE;
  }
  return CUDA_SUCCESS;
}
CUresult error_string(CUresult, const char **out) { *out = "test-double error description"; return CUDA_SUCCESS; }
CUresult device_count(int *n) { FAKE_INJECT("cuDeviceGetCount"); *n = global.device_count; return CUDA_SUCCESS; }
CUresult device_get(CUdevice *d, int ordinal) {
  if (ordinal < 0 || ordinal >= global.device_count) return CUDA_ERROR_INVALID_DEVICE;
  *d = ordinal;
  return CUDA_SUCCESS;
}
CUresult device_name(char *name, int length, CUdevice) {
  std::strncpy(name, "Paralyn fake CUDA test double", static_cast<std::size_t>(length));
  return CUDA_SUCCESS;
}
CUresult total_mem(std::size_t *bytes, CUdevice) { *bytes = std::size_t(1) << 30; return CUDA_SUCCESS; }
CUresult attribute(int *value, int which, CUdevice) {
  switch (which) {
  case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: *value = 1024; break;
  case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: *value = 1024; break;
  case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: *value = 64; break;
  case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: *value = 2147483647; break;
  case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: *value = 65535; break;
  case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: *value = 49152; break;
  case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: *value = global.cc_major; break;
  case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: *value = global.cc_minor; break;
  default: *value = 0;
  }
  return CUDA_SUCCESS;
}
CUresult uuid(CUuuid *out, CUdevice d) {
  for (int i = 0; i < 16; ++i) out->bytes[i] = static_cast<unsigned char>(0xA0 + i + d);
  return CUDA_SUCCESS;
}
CUresult retain(CUcontext *out, CUdevice) {
  FAKE_INJECT("cuDevicePrimaryCtxRetain");
  *out = global.primary;
  ++global.primary_retained;
  ++global.primary_retains;
  return CUDA_SUCCESS;
}
CUresult release(CUdevice) { --global.primary_retained; ++global.primary_releases; return CUDA_SUCCESS; }
CUresult push(CUcontext c) { FAKE_INJECT("cuCtxPushCurrent"); stack.push_back(c); return CUDA_SUCCESS; }
CUresult pop(CUcontext *out) {
  if (stack.empty()) return CUDA_ERROR_INVALID_CONTEXT;
  *out = stack.back();
  stack.pop_back();
  return CUDA_SUCCESS;
}
CUresult mem_alloc(CUdeviceptr *p, std::size_t n) {
  FAKE_CURRENT("cuMemAlloc");
  FAKE_INJECT("cuMemAlloc");
  *p = global.next_pointer;
  global.next_pointer += (n + 255) / 256 * 256 + 256;
  global.memory[*p].assign(n, 0xEE);
  return CUDA_SUCCESS;
}
CUresult mem_free(CUdeviceptr p) {
  FAKE_CURRENT("cuMemFree");
  return global.memory.erase(p) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}
std::vector<unsigned char> *find(CUdeviceptr p, std::size_t n, std::size_t &offset) {
  for (auto &entry : global.memory)
    if (p >= entry.first && p - entry.first + n <= entry.second.size()) {
      offset = p - entry.first;
      return &entry.second;
    }
  return nullptr;
}
CUresult htod(CUdeviceptr p, const void *src, std::size_t n) {
  FAKE_CURRENT("cuMemcpyHtoD");
  FAKE_INJECT("cuMemcpyHtoD");
  std::size_t offset = 0;
  auto *target = find(p, n, offset);
  if (!target) return CUDA_ERROR_INVALID_VALUE;
  std::memcpy(target->data() + offset, src, n);
  return CUDA_SUCCESS;
}
CUresult dtoh(void *dst, CUdeviceptr p, std::size_t n) {
  FAKE_CURRENT("cuMemcpyDtoH");
  FAKE_INJECT("cuMemcpyDtoH");
  std::size_t offset = 0;
  auto *source = find(p, n, offset);
  if (!source) return CUDA_ERROR_INVALID_VALUE;
  std::memcpy(dst, source->data() + offset, n);
  return CUDA_SUCCESS;
}
CUresult module_load(CUmodule *m, const void *image, unsigned count, CUjit_option *keys, void **values) {
  FAKE_CURRENT("cuModuleLoadDataEx");
  if (int r = injected("cuModuleLoadDataEx")) {
    for (unsigned i = 0; i < count; ++i)
      if (keys[i] == CU_JIT_ERROR_LOG_BUFFER)
        std::strcpy(static_cast<char *>(values[i]), "ptxas fatal: test-double JIT failure");
    return r;
  }
  global.loaded_images.push_back(static_cast<const char *>(image));
  *m = reinterpret_cast<CUmodule>(std::uintptr_t(0x4000 + ++global.modules));
  return CUDA_SUCCESS;
}
CUresult module_unload(CUmodule) { FAKE_CURRENT("cuModuleUnload"); --global.modules; return CUDA_SUCCESS; }
std::map<CUfunction, std::string> functions;
CUresult get_function(CUfunction *f, CUmodule, const char *name) {
  FAKE_CURRENT("cuModuleGetFunction");
  FAKE_INJECT("cuModuleGetFunction");
  *f = reinterpret_cast<CUfunction>(std::uintptr_t(0x8000 + functions.size() + 1));
  functions[*f] = name;
  return CUDA_SUCCESS;
}
CUresult func_attribute(int *value, int, CUfunction) {
  FAKE_CURRENT("cuFuncGetAttribute");
  FAKE_INJECT("cuFuncGetAttribute");
  *value = 1024;
  return CUDA_SUCCESS;
}
CUresult launch(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                unsigned shared, CUstream, void **params, void **extra) {
  FAKE_CURRENT("cuLaunchKernel");
  FAKE_INJECT("cuLaunchKernel");
  if (shared || extra) return CUDA_ERROR_INVALID_VALUE;
  // Records the marshalled arguments only. Deliberately computes nothing.
  Launch record{functions[f], {gx, gy, gz}, {bx, by, bz}, {}};
  for (std::size_t i = 0; i < global.parameter_sizes.size(); ++i) {
    const auto *bytes = static_cast<const unsigned char *>(params[i]);
    record.parameters.emplace_back(bytes, bytes + global.parameter_sizes[i]);
  }
  global.launches.push_back(std::move(record));
  return CUDA_SUCCESS;
}
CUresult stream_create(CUstream *s, unsigned) {
  FAKE_CURRENT("cuStreamCreate");
  FAKE_INJECT("cuStreamCreate");
  *s = reinterpret_cast<CUstream>(std::uintptr_t(0x5000 + ++global.streams));
  return CUDA_SUCCESS;
}
CUresult stream_destroy(CUstream) { FAKE_CURRENT("cuStreamDestroy"); --global.streams; return CUDA_SUCCESS; }
CUresult stream_sync(CUstream) { FAKE_CURRENT("cuStreamSynchronize"); return CUDA_SUCCESS; }
CUresult event_create(CUevent *e, unsigned) {
  FAKE_CURRENT("cuEventCreate");
  FAKE_INJECT("cuEventCreate");
  *e = reinterpret_cast<CUevent>(std::uintptr_t(0x6000 + ++global.events));
  return CUDA_SUCCESS;
}
CUresult event_destroy(CUevent) { FAKE_CURRENT("cuEventDestroy"); --global.events; return CUDA_SUCCESS; }
CUresult event_record(CUevent, CUstream) {
  FAKE_CURRENT("cuEventRecord");
  FAKE_INJECT("cuEventRecord");
  return CUDA_SUCCESS;
}
CUresult event_sync(CUevent) { FAKE_CURRENT("cuEventSynchronize"); FAKE_INJECT("cuEventSynchronize"); return CUDA_SUCCESS; }
CUresult elapsed(float *ms, CUevent, CUevent) { FAKE_CURRENT("cuEventElapsedTime"); *ms = global.elapsed_ms; return CUDA_SUCCESS; }

struct Program {
  std::string source;
  bool compiled = false;
};
nvrtcResult nv_version(int *major, int *minor) { *major = 12; *minor = 8; return NVRTC_SUCCESS; }
const char *nv_error(nvrtcResult) { return "NVRTC_ERROR_COMPILATION (test double)"; }
nvrtcResult nv_create(nvrtcProgram *p, const char *source, const char *, int, const char *const *, const char *const *) {
  auto *program = new Program{source, false};
  global.sources.push_back(source);
  ++global.programs;
  *p = reinterpret_cast<nvrtcProgram>(program);
  return NVRTC_SUCCESS;
}
nvrtcResult nv_destroy(nvrtcProgram *p) {
  delete reinterpret_cast<Program *>(*p);
  *p = nullptr;
  --global.programs;
  return NVRTC_SUCCESS;
}
nvrtcResult nv_compile(nvrtcProgram p, int count, const char *const *options) {
  global.nvrtc_options.assign(options, options + count);
  if (int r = injected("nvrtcCompileProgram")) return r;
  reinterpret_cast<Program *>(p)->compiled = true;
  return NVRTC_SUCCESS;
}
nvrtcResult nv_log_size(nvrtcProgram, std::size_t *n) { *n = global.nvrtc_log.size() + 1; return NVRTC_SUCCESS; }
nvrtcResult nv_log(nvrtcProgram, char *out) { std::strcpy(out, global.nvrtc_log.c_str()); return NVRTC_SUCCESS; }
const char *fake_ptx = "// TEST DOUBLE: not PTX produced by NVRTC\n";
nvrtcResult nv_ptx_size(nvrtcProgram p, std::size_t *n) {
  if (!reinterpret_cast<Program *>(p)->compiled) return 6;
  *n = std::strlen(fake_ptx) + 1;
  return NVRTC_SUCCESS;
}
nvrtcResult nv_ptx(nvrtcProgram, char *out) { std::strcpy(out, fake_ptx); return NVRTC_SUCCESS; }
nvrtcResult nv_arch_count(int *n) { *n = static_cast<int>(global.nvrtc_archs.size()); return NVRTC_SUCCESS; }
nvrtcResult nv_archs(int *out) {
  for (std::size_t i = 0; i < global.nvrtc_archs.size(); ++i) out[i] = global.nvrtc_archs[i];
  return NVRTC_SUCCESS;
}
} // namespace

State &state() { return global; }
void reset() {
  global = State{};
  stack.clear();
  functions.clear();
}
std::vector<CUcontext> &context_stack() { return stack; }
DriverApi driver_table() {
  DriverApi d;
  d.cuInit = init;
  d.cuDriverGetVersion = driver_version;
  d.cuGetErrorName = error_name;
  d.cuGetErrorString = error_string;
  d.cuDeviceGetCount = device_count;
  d.cuDeviceGet = device_get;
  d.cuDeviceGetName = device_name;
  d.cuDeviceTotalMem = total_mem;
  d.cuDeviceGetAttribute = attribute;
  d.cuDeviceGetUuid = uuid;
  d.cuDevicePrimaryCtxRetain = retain;
  d.cuDevicePrimaryCtxRelease = release;
  d.cuCtxPushCurrent = push;
  d.cuCtxPopCurrent = pop;
  d.cuMemAlloc = mem_alloc;
  d.cuMemFree = mem_free;
  d.cuMemcpyHtoD = htod;
  d.cuMemcpyDtoH = dtoh;
  d.cuModuleLoadDataEx = module_load;
  d.cuModuleUnload = module_unload;
  d.cuModuleGetFunction = get_function;
  d.cuFuncGetAttribute = func_attribute;
  d.cuLaunchKernel = launch;
  d.cuStreamCreate = stream_create;
  d.cuStreamDestroy = stream_destroy;
  d.cuStreamSynchronize = stream_sync;
  d.cuEventCreate = event_create;
  d.cuEventDestroy = event_destroy;
  d.cuEventRecord = event_record;
  d.cuEventSynchronize = event_sync;
  d.cuEventElapsedTime = elapsed;
  return d;
}
NvrtcApi nvrtc_table() {
  NvrtcApi n;
  n.nvrtcVersion = nv_version;
  n.nvrtcGetErrorString = nv_error;
  n.nvrtcCreateProgram = nv_create;
  n.nvrtcDestroyProgram = nv_destroy;
  n.nvrtcCompileProgram = nv_compile;
  n.nvrtcGetProgramLogSize = nv_log_size;
  n.nvrtcGetProgramLog = nv_log;
  n.nvrtcGetPTXSize = nv_ptx_size;
  n.nvrtcGetPTX = nv_ptx;
  n.nvrtcGetNumSupportedArchs = nv_arch_count;
  n.nvrtcGetSupportedArchs = nv_archs;
  return n;
}
} // namespace fake_cuda
