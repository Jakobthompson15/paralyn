#include "api.hpp"
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <utility>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace paralyn::backend::cuda {
namespace {
static_assert(sizeof(void *) == 8, "The CUDA backend requires a 64-bit host (CUdeviceptr width)");

struct Library final : LibraryHandle {
#ifdef _WIN32
  HMODULE handle = nullptr;
  ~Library() override {
    if (handle) FreeLibrary(handle);
  }
#else
  void *handle = nullptr;
  ~Library() override {
    if (handle) dlclose(handle);
  }
#endif
  void *symbol(const char *name) const {
#ifdef _WIN32
    return reinterpret_cast<void *>(GetProcAddress(handle, name));
#else
    return dlsym(handle, name);
#endif
  }
};

#ifdef _WIN32
std::wstring wide(const std::string &s) {
  if (s.empty()) return {};
  const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), int(s.size()), nullptr, 0);
  if (n <= 0) return {};
  std::wstring w(static_cast<std::size_t>(n), L'\0');
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), int(s.size()), w.data(), n);
  return w;
}
std::string last_error_text() {
  const DWORD code = GetLastError();
  char *message = nullptr;
  FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                     FORMAT_MESSAGE_IGNORE_INSERTS,
                 nullptr, code, 0, reinterpret_cast<char *>(&message), 0, nullptr);
  std::string text = "Windows error " + std::to_string(code);
  if (message) {
    text += ": ";
    text += message;
    LocalFree(message);
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
  }
  return text;
}
bool absolute_path(const std::string &path) {
  return path.find('\\') != std::string::npos || path.find('/') != std::string::npos;
}
#endif

std::shared_ptr<Library> open(const std::string &path, std::string &error) {
  auto library = std::make_shared<Library>();
#ifdef _WIN32
  const auto name = wide(path);
  if (name.empty()) {
    error = "library path is not valid UTF-8";
    return nullptr;
  }
  // The display driver's nvcuda.dll is loaded from System32 only. Explicit
  // paths (including %CUDA_PATH% candidates) also resolve dependencies such as
  // nvrtc-builtins beside the DLL. Other bare names use the standard Windows
  // search order, which includes PATH as configured by the CUDA installer.
  DWORD flags = 0;
  if (path == "nvcuda.dll")
    flags = LOAD_LIBRARY_SEARCH_SYSTEM32;
  else if (absolute_path(path))
    flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
  library->handle = LoadLibraryExW(name.c_str(), nullptr, flags);
  if (!library->handle) {
    error = last_error_text();
    return nullptr;
  }
#else
  library->handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!library->handle) {
    const char *detail = dlerror();
    error = detail ? detail : "dlopen failed without detail";
    return nullptr;
  }
#endif
  return library;
}

template <class T>
bool bind(const Library &library, T &slot, std::initializer_list<const char *> names,
          std::string &missing, bool required = true) {
  for (const char *name : names)
    if (void *address = library.symbol(name)) {
      slot = reinterpret_cast<T>(address);
      return true;
    }
  if (required) {
    if (!missing.empty()) missing += ", ";
    missing += *names.begin();
  }
  return !required;
}

std::string environment(const char *key) {
  const char *value = std::getenv(key);
  return value && *value ? value : "";
}

std::shared_ptr<Library> first(const std::vector<std::string> &candidates,
                               std::vector<std::string> &attempts, std::string &chosen) {
  for (const auto &candidate : candidates) {
    std::string error;
    if (auto library = open(candidate, error)) {
      chosen = candidate;
      return library;
    }
    attempts.push_back(candidate + ": " + error);
  }
  return nullptr;
}

bool bind_driver(const Library &l, DriverApi &d, std::string &missing) {
  bool ok = true;
  ok &= bind(l, d.cuInit, {"cuInit"}, missing);
  ok &= bind(l, d.cuDriverGetVersion, {"cuDriverGetVersion"}, missing);
  ok &= bind(l, d.cuGetErrorName, {"cuGetErrorName"}, missing);
  ok &= bind(l, d.cuGetErrorString, {"cuGetErrorString"}, missing);
  ok &= bind(l, d.cuDeviceGetCount, {"cuDeviceGetCount"}, missing);
  ok &= bind(l, d.cuDeviceGet, {"cuDeviceGet"}, missing);
  ok &= bind(l, d.cuDeviceGetName, {"cuDeviceGetName"}, missing);
  ok &= bind(l, d.cuDeviceTotalMem, {"cuDeviceTotalMem_v2"}, missing);
  ok &= bind(l, d.cuDeviceGetAttribute, {"cuDeviceGetAttribute"}, missing);
  bind(l, d.cuDeviceGetUuid, {"cuDeviceGetUuid_v2", "cuDeviceGetUuid"}, missing, false);
  ok &= bind(l, d.cuDevicePrimaryCtxRetain, {"cuDevicePrimaryCtxRetain"}, missing);
  ok &= bind(l, d.cuDevicePrimaryCtxRelease, {"cuDevicePrimaryCtxRelease_v2"}, missing);
  ok &= bind(l, d.cuCtxPushCurrent, {"cuCtxPushCurrent_v2"}, missing);
  ok &= bind(l, d.cuCtxPopCurrent, {"cuCtxPopCurrent_v2"}, missing);
  ok &= bind(l, d.cuMemAlloc, {"cuMemAlloc_v2"}, missing);
  ok &= bind(l, d.cuMemFree, {"cuMemFree_v2"}, missing);
  ok &= bind(l, d.cuMemcpyHtoD, {"cuMemcpyHtoD_v2"}, missing);
  ok &= bind(l, d.cuMemcpyDtoH, {"cuMemcpyDtoH_v2"}, missing);
  ok &= bind(l, d.cuModuleLoadDataEx, {"cuModuleLoadDataEx"}, missing);
  ok &= bind(l, d.cuModuleUnload, {"cuModuleUnload"}, missing);
  ok &= bind(l, d.cuModuleGetFunction, {"cuModuleGetFunction"}, missing);
  ok &= bind(l, d.cuFuncGetAttribute, {"cuFuncGetAttribute"}, missing);
  ok &= bind(l, d.cuLaunchKernel, {"cuLaunchKernel"}, missing);
  ok &= bind(l, d.cuStreamCreate, {"cuStreamCreate"}, missing);
  ok &= bind(l, d.cuStreamDestroy, {"cuStreamDestroy_v2"}, missing);
  ok &= bind(l, d.cuStreamSynchronize, {"cuStreamSynchronize"}, missing);
  ok &= bind(l, d.cuEventCreate, {"cuEventCreate"}, missing);
  ok &= bind(l, d.cuEventDestroy, {"cuEventDestroy_v2"}, missing);
  ok &= bind(l, d.cuEventRecord, {"cuEventRecord"}, missing);
  ok &= bind(l, d.cuEventSynchronize, {"cuEventSynchronize"}, missing);
  ok &= bind(l, d.cuEventElapsedTime, {"cuEventElapsedTime_v2", "cuEventElapsedTime"}, missing);
  return ok;
}
bool bind_nvrtc(const Library &l, NvrtcApi &n, std::string &missing) {
  bool ok = true;
  ok &= bind(l, n.nvrtcVersion, {"nvrtcVersion"}, missing);
  ok &= bind(l, n.nvrtcGetErrorString, {"nvrtcGetErrorString"}, missing);
  ok &= bind(l, n.nvrtcCreateProgram, {"nvrtcCreateProgram"}, missing);
  ok &= bind(l, n.nvrtcDestroyProgram, {"nvrtcDestroyProgram"}, missing);
  ok &= bind(l, n.nvrtcCompileProgram, {"nvrtcCompileProgram"}, missing);
  ok &= bind(l, n.nvrtcGetProgramLogSize, {"nvrtcGetProgramLogSize"}, missing);
  ok &= bind(l, n.nvrtcGetProgramLog, {"nvrtcGetProgramLog"}, missing);
  ok &= bind(l, n.nvrtcGetPTXSize, {"nvrtcGetPTXSize"}, missing);
  ok &= bind(l, n.nvrtcGetPTX, {"nvrtcGetPTX"}, missing);
  bind(l, n.nvrtcGetNumSupportedArchs, {"nvrtcGetNumSupportedArchs"}, missing, false);
  bind(l, n.nvrtcGetSupportedArchs, {"nvrtcGetSupportedArchs"}, missing, false);
  return ok;
}
} // namespace

std::string result_text(const DriverApi &api, CUresult code) {
  const char *name = nullptr, *text = nullptr;
  if (api.cuGetErrorName && api.cuGetErrorName(code, &name) != CUDA_SUCCESS) name = nullptr;
  if (api.cuGetErrorString && api.cuGetErrorString(code, &text) != CUDA_SUCCESS) text = nullptr;
  std::ostringstream out;
  out << (name ? name : "CUresult") << " (" << code << ")";
  if (text) out << ": " << text;
  return out.str();
}

std::vector<std::string> default_driver_candidates() {
#ifdef _WIN32
  return {"nvcuda.dll"};
#elif defined(__APPLE__)
  // NVIDIA has not shipped a macOS CUDA driver since CUDA 10.2; the attempt is
  // recorded so that "unavailable" states exactly what was tried.
  return {"libcuda.dylib", "/usr/local/cuda/lib/libcuda.dylib"};
#else
  return {"libcuda.so.1", "libcuda.so"};
#endif
}
std::vector<std::string> default_nvrtc_candidates() {
  std::vector<std::string> names;
#ifdef _WIN32
  const std::vector<std::string> dlls = {"nvrtc64_130_0.dll", "nvrtc64_120_0.dll",
                                         "nvrtc64_112_0.dll"};
  const auto root = environment("CUDA_PATH");
  if (!root.empty())
    for (const char *sub : {"\\bin\\x64\\", "\\bin\\"})
      for (const auto &dll : dlls) names.push_back(root + sub + dll);
  names.insert(names.end(), dlls.begin(), dlls.end());
#elif defined(__APPLE__)
  names = {"libnvrtc.dylib", "/usr/local/cuda/lib/libnvrtc.dylib"};
#else
  names = {"libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11.2", "libnvrtc.so"};
  const auto root = environment("CUDA_HOME");
  if (!root.empty())
    for (const char *name : {"/lib64/libnvrtc.so", "/lib/libnvrtc.so"}) names.push_back(root + name);
#endif
  return names;
}

void initialize(LoadResult &result) {
  const auto &d = result.api->driver;
  const auto init = d.cuInit(0);
  if (init != CUDA_SUCCESS) {
    result.reason = "cuInit failed: " + result_text(d, init);
    return;
  }
  result.initialized = true;
  int version = 0;
  if (d.cuDriverGetVersion(&version) == CUDA_SUCCESS) result.driver_version = version;
  int major = 0, minor = 0;
  if (result.api->nvrtc.nvrtcVersion(&major, &minor) == NVRTC_SUCCESS) {
    result.nvrtc_major = major;
    result.nvrtc_minor = minor;
  }
  int count = 0;
  const auto counted = d.cuDeviceGetCount(&count);
  if (counted != CUDA_SUCCESS) {
    result.reason = "cuDeviceGetCount failed: " + result_text(d, counted);
    return;
  }
  result.device_count = count;
  if (count <= 0) result.reason = "The CUDA driver reports no devices";
}

LoadResult load(const LoadRequest &request) {
  LoadResult result;
  auto api = std::make_shared<Api>();
  const auto driver_candidates =
      request.driver_path.empty() ? default_driver_candidates() : std::vector<std::string>{request.driver_path};
  const auto nvrtc_candidates =
      request.nvrtc_path.empty() ? default_nvrtc_candidates() : std::vector<std::string>{request.nvrtc_path};
  std::string reasons;
  auto driver = first(driver_candidates, result.attempts, result.driver_path);
  if (!driver)
    reasons = "CUDA driver library not found (NVIDIA driver not installed or not on the loader path)";
  else {
    std::string missing;
    if (bind_driver(*driver, api->driver, missing)) {
      result.driver_loaded = true;
      api->libraries.push_back(driver);
    } else
      reasons = "CUDA driver library " + result.driver_path + " lacks required symbols: " + missing;
  }
  auto nvrtc = first(nvrtc_candidates, result.attempts, result.nvrtc_path);
  if (!nvrtc) {
    if (!reasons.empty()) reasons += "; ";
    reasons += "NVRTC library not found (CUDA Toolkit runtime compiler is required to build kernels)";
  } else {
    std::string missing;
    if (bind_nvrtc(*nvrtc, api->nvrtc, missing)) {
      result.nvrtc_loaded = true;
      api->libraries.push_back(nvrtc);
    } else {
      if (!reasons.empty()) reasons += "; ";
      reasons += "NVRTC library " + result.nvrtc_path + " lacks required symbols: " + missing;
    }
  }
  if (!result.driver_loaded || !result.nvrtc_loaded) {
    result.reason = reasons;
    return result;
  }
  api->driver_path = result.driver_path;
  api->nvrtc_path = result.nvrtc_path;
  result.api = api;
  initialize(result);
  return result;
}

const LoadResult &cached_load() {
  static std::once_flag once;
  static LoadResult result;
  std::call_once(once, [] {
    LoadRequest request{environment("PARALYN_CUDA_DRIVER_LIBRARY"),
                        environment("PARALYN_NVRTC_LIBRARY")};
    result = load(request);
  });
  return result;
}
} // namespace paralyn::backend::cuda
