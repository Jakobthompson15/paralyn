#pragma once
// Minimal, independently written declarations of the documented public CUDA
// Driver API and NVRTC C ABI subset used by Paralyn. No NVIDIA header or
// library is needed to build; both libraries are loaded at run time.
// Values follow NVIDIA's published CUDA Driver API / NVRTC reference.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace paralyn::backend::cuda {
using CUresult = int;
using CUdevice = int;
using CUdeviceptr = unsigned long long; // 64-bit hosts only (checked in loader)
struct CUctx_st;
struct CUmod_st;
struct CUfunc_st;
struct CUstream_st;
struct CUevent_st;
using CUcontext = CUctx_st *;
using CUmodule = CUmod_st *;
using CUfunction = CUfunc_st *;
using CUstream = CUstream_st *;
using CUevent = CUevent_st *;
using CUjit_option = int;
using nvrtcResult = int;
struct _nvrtcProgram;
using nvrtcProgram = _nvrtcProgram *;
struct CUuuid {
  unsigned char bytes[16];
};

enum : CUresult {
  CUDA_SUCCESS = 0,
  CUDA_ERROR_INVALID_VALUE = 1,
  CUDA_ERROR_OUT_OF_MEMORY = 2,
  CUDA_ERROR_NOT_INITIALIZED = 3,
  CUDA_ERROR_DEINITIALIZED = 4,
  CUDA_ERROR_STUB_LIBRARY = 34,
  CUDA_ERROR_INSUFFICIENT_DRIVER = 35,
  CUDA_ERROR_NO_DEVICE = 100,
  CUDA_ERROR_INVALID_DEVICE = 101,
  CUDA_ERROR_INVALID_IMAGE = 200,
  CUDA_ERROR_INVALID_CONTEXT = 201,
  CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
  CUDA_ERROR_ECC_UNCORRECTABLE = 214,
  CUDA_ERROR_INVALID_PTX = 218,
  CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
  CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
  CUDA_ERROR_INVALID_SOURCE = 300,
  CUDA_ERROR_INVALID_HANDLE = 400,
  CUDA_ERROR_NOT_FOUND = 500,
  CUDA_ERROR_NOT_READY = 600,
  CUDA_ERROR_ILLEGAL_ADDRESS = 700,
  CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
  CUDA_ERROR_LAUNCH_TIMEOUT = 702,
  CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
  CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
  CUDA_ERROR_MISALIGNED_ADDRESS = 716,
  CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
  CUDA_ERROR_INVALID_PC = 718,
  CUDA_ERROR_LAUNCH_FAILED = 719,
  CUDA_ERROR_NOT_PERMITTED = 800,
  CUDA_ERROR_NOT_SUPPORTED = 801,
  CUDA_ERROR_SYSTEM_NOT_READY = 802,
  CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
  CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
  CUDA_ERROR_UNKNOWN = 999
};
enum : int {
  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
  CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
  CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
  CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
  CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
};
enum : int { CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0 };
enum : CUjit_option {
  CU_JIT_INFO_LOG_BUFFER = 3,
  CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
  CU_JIT_ERROR_LOG_BUFFER = 5,
  CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
};
enum : unsigned { CU_STREAM_DEFAULT = 0, CU_EVENT_DEFAULT = 0 };
enum : nvrtcResult { NVRTC_SUCCESS = 0 };

// Function table. Production fills it from the dynamically loaded libraries.
// Tests may fill it with an explicitly labelled in-process test double; no
// production path constructs such a table.
struct DriverApi {
  CUresult (*cuInit)(unsigned) = nullptr;
  CUresult (*cuDriverGetVersion)(int *) = nullptr;
  CUresult (*cuGetErrorName)(CUresult, const char **) = nullptr;
  CUresult (*cuGetErrorString)(CUresult, const char **) = nullptr;
  CUresult (*cuDeviceGetCount)(int *) = nullptr;
  CUresult (*cuDeviceGet)(CUdevice *, int) = nullptr;
  CUresult (*cuDeviceGetName)(char *, int, CUdevice) = nullptr;
  CUresult (*cuDeviceTotalMem)(std::size_t *, CUdevice) = nullptr; // cuDeviceTotalMem_v2
  CUresult (*cuDeviceGetAttribute)(int *, int, CUdevice) = nullptr;
  CUresult (*cuDeviceGetUuid)(CUuuid *, CUdevice) = nullptr; // optional
  CUresult (*cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice) = nullptr;
  CUresult (*cuDevicePrimaryCtxRelease)(CUdevice) = nullptr; // _v2
  CUresult (*cuCtxPushCurrent)(CUcontext) = nullptr;         // _v2
  CUresult (*cuCtxPopCurrent)(CUcontext *) = nullptr;        // _v2
  CUresult (*cuMemAlloc)(CUdeviceptr *, std::size_t) = nullptr;               // _v2
  CUresult (*cuMemFree)(CUdeviceptr) = nullptr;                               // _v2
  CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void *, std::size_t) = nullptr; // _v2
  CUresult (*cuMemcpyDtoH)(void *, CUdeviceptr, std::size_t) = nullptr;       // _v2
  CUresult (*cuModuleLoadDataEx)(CUmodule *, const void *, unsigned, CUjit_option *,
                                 void **) = nullptr;
  CUresult (*cuModuleUnload)(CUmodule) = nullptr;
  CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *) = nullptr;
  CUresult (*cuFuncGetAttribute)(int *, int, CUfunction) = nullptr;
  CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                             unsigned, unsigned, CUstream, void **, void **) = nullptr;
  CUresult (*cuStreamCreate)(CUstream *, unsigned) = nullptr;
  CUresult (*cuStreamDestroy)(CUstream) = nullptr; // _v2
  CUresult (*cuStreamSynchronize)(CUstream) = nullptr;
  CUresult (*cuEventCreate)(CUevent *, unsigned) = nullptr;
  CUresult (*cuEventDestroy)(CUevent) = nullptr; // _v2
  CUresult (*cuEventRecord)(CUevent, CUstream) = nullptr;
  CUresult (*cuEventSynchronize)(CUevent) = nullptr;
  CUresult (*cuEventElapsedTime)(float *, CUevent, CUevent) = nullptr;
};
struct NvrtcApi {
  nvrtcResult (*nvrtcVersion)(int *, int *) = nullptr;
  const char *(*nvrtcGetErrorString)(nvrtcResult) = nullptr;
  nvrtcResult (*nvrtcCreateProgram)(nvrtcProgram *, const char *, const char *, int,
                                    const char *const *, const char *const *) = nullptr;
  nvrtcResult (*nvrtcDestroyProgram)(nvrtcProgram *) = nullptr;
  nvrtcResult (*nvrtcCompileProgram)(nvrtcProgram, int, const char *const *) = nullptr;
  nvrtcResult (*nvrtcGetProgramLogSize)(nvrtcProgram, std::size_t *) = nullptr;
  nvrtcResult (*nvrtcGetProgramLog)(nvrtcProgram, char *) = nullptr;
  nvrtcResult (*nvrtcGetPTXSize)(nvrtcProgram, std::size_t *) = nullptr;
  nvrtcResult (*nvrtcGetPTX)(nvrtcProgram, char *) = nullptr;
  nvrtcResult (*nvrtcGetNumSupportedArchs)(int *) = nullptr; // optional (NVRTC >= 11.2)
  nvrtcResult (*nvrtcGetSupportedArchs)(int *) = nullptr;    // optional
};

// Keeps loaded libraries alive for as long as any backend object uses the table.
struct LibraryHandle {
  virtual ~LibraryHandle() = default;
};
struct Api {
  DriverApi driver;
  NvrtcApi nvrtc;
  std::string driver_path, nvrtc_path;
  bool test_double = false;
  std::vector<std::shared_ptr<LibraryHandle>> libraries;
};

// Result of attempting to load and initialize the CUDA driver and NVRTC.
struct LoadResult {
  std::shared_ptr<const Api> api; // non-null only when both libraries and all symbols loaded
  bool driver_loaded = false, nvrtc_loaded = false, initialized = false;
  std::string driver_path, nvrtc_path;
  int driver_version = 0, nvrtc_major = 0, nvrtc_minor = 0, device_count = 0;
  std::vector<std::string> attempts; // every candidate tried and why it failed
  std::string reason;               // empty when available
  bool available() const { return api && initialized && device_count > 0 && reason.empty(); }
};
struct LoadRequest {
  // Explicit paths (from PARALYN_CUDA_DRIVER_LIBRARY / PARALYN_NVRTC_LIBRARY in
  // production). An explicit path is never replaced by a silent fallback.
  std::string driver_path, nvrtc_path;
};
// Uncached; used by tests. Production uses cached_load().
LoadResult load(const LoadRequest &request);
// Process-wide, loaded once from the environment/default platform names.
const LoadResult &cached_load();
std::vector<std::string> default_driver_candidates();
std::vector<std::string> default_nvrtc_candidates();
// Complete a load result from an already populated table (production and tests).
void initialize(LoadResult &result);
std::string result_text(const DriverApi &api, CUresult code);
} // namespace paralyn::backend::cuda
