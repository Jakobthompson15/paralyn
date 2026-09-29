#include "api.hpp"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
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
#endif

#ifndef _WIN32
// Linux default sonames: glibc's dlopen resolves a name without '/' through
// LD_LIBRARY_PATH, DT_RUNPATH, ld.so.cache and the trusted system directories,
// never the current working directory (unlike macOS dyld, which does search it
// for bare names). Only this fixed set of built-in names is accepted bare, and
// only on Linux.
bool system_soname(const std::string &path) {
#if defined(__APPLE__)
  (void)path;
  return false;
#else
  for (const char *name : {"libcuda.so.1", "libcuda.so", "libnvrtc.so.13", "libnvrtc.so.12",
                           "libnvrtc.so.11.2", "libnvrtc.so"})
    if (path == name) return true;
  return false;
#endif
}
#endif

// Only fully qualified paths are handed to the platform loader. A bare or
// relative name would let dyld (macOS) or the default Windows search order pick
// a same-named library from the current working directory, i.e. run untrusted
// code from whatever directory Paralyn is started in. The only exceptions are
// the system-directory-only loads documented below.
std::shared_ptr<Library> open(const std::string &path, std::string &error) {
#ifdef _WIN32
  DWORD flags = 0;
  if (path == "nvcuda.dll")
    flags = LOAD_LIBRARY_SEARCH_SYSTEM32; // the display driver's DLL lives in System32 only
  else if (is_absolute_library_path(path))
    // Dependencies (e.g. nvrtc-builtins) resolve beside the DLL, then in the
    // application directory and System32. Never the current directory or PATH.
    flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
  else {
    error = "refused: not a fully qualified path (the current directory is never searched)";
    return nullptr;
  }
  const auto name = wide(path);
  if (name.empty()) {
    error = "library path is not valid UTF-8";
    return nullptr;
  }
  auto library = std::make_shared<Library>();
  library->handle = LoadLibraryExW(name.c_str(), nullptr, flags);
  if (!library->handle) {
    error = last_error_text();
    return nullptr;
  }
#else
  if (!is_absolute_library_path(path) && !system_soname(path)) {
    error = "refused: not an absolute path (the current directory is never searched)";
    return nullptr;
  }
  auto library = std::make_shared<Library>();
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

bool is_absolute_library_path(const std::string &path) {
#ifdef _WIN32
  // Drive-absolute ("C:\..." or "C:/...") or UNC ("\\server\share\..."). Drive-
  // relative ("C:foo"), rooted-without-drive ("\foo") and device ("\\?\",
  // "\\.\") forms are refused.
  if (path.size() >= 3 && std::isalpha(static_cast<unsigned char>(path[0])) && path[1] == ':' &&
      (path[2] == '\\' || path[2] == '/'))
    return true;
  return path.size() >= 3 && (path[0] == '\\' || path[0] == '/') && (path[1] == '\\' || path[1] == '/') &&
         path[2] != '\\' && path[2] != '/' && path[2] != '?' && path[2] != '.';
#else
  return !path.empty() && path[0] == '/';
#endif
}

std::vector<std::string> default_driver_candidates() {
#ifdef _WIN32
  return {"nvcuda.dll"}; // loaded with LOAD_LIBRARY_SEARCH_SYSTEM32 only
#elif defined(__APPLE__)
  // NVIDIA has shipped no macOS CUDA driver since CUDA 10.2, so nothing is
  // loaded by default: a bare name would make dyld search the current working
  // directory. An explicit absolute PARALYN_CUDA_DRIVER_LIBRARY still works.
  return {};
#else
  return {"libcuda.so.1", "libcuda.so"};
#endif
}
std::vector<std::string> default_nvrtc_candidates() {
  std::vector<std::string> names;
#ifdef _WIN32
  // Fully qualified candidates only: %CUDA_PATH% (set by the CUDA Toolkit
  // installer) and absolute PATH directories that actually contain the DLL.
  // Relative PATH entries and the current directory are never searched.
  const std::vector<std::string> dlls = {"nvrtc64_130_0.dll", "nvrtc64_120_0.dll",
                                         "nvrtc64_112_0.dll"};
  auto join = [](std::string directory, const std::string &file) {
    while (!directory.empty() && (directory.back() == '\\' || directory.back() == '/')) directory.pop_back();
    return directory + "\\" + file;
  };
  const auto root = environment("CUDA_PATH");
  if (is_absolute_library_path(root))
    for (const char *sub : {"\\bin\\x64", "\\bin"})
      for (const auto &dll : dlls) names.push_back(join(root + sub, dll));
  const auto search = environment("PATH");
  std::size_t begin = 0;
  while (begin <= search.size()) {
    auto end = search.find(';', begin);
    if (end == std::string::npos) end = search.size();
    std::string entry = search.substr(begin, end - begin);
    if (entry.size() >= 2 && entry.front() == '"' && entry.back() == '"') entry = entry.substr(1, entry.size() - 2);
    if (is_absolute_library_path(entry))
      for (const auto &dll : dlls) {
        const auto candidate = join(entry, dll);
        std::error_code ignored;
        if (std::find(names.begin(), names.end(), candidate) == names.end() &&
            std::filesystem::is_regular_file(std::filesystem::path(wide(candidate)), ignored))
          names.push_back(candidate);
      }
    begin = end + 1;
  }
#elif defined(__APPLE__)
  // No default on macOS; see default_driver_candidates().
#else
  names = {"libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11.2", "libnvrtc.so"};
  const auto root = environment("CUDA_HOME");
  if (is_absolute_library_path(root))
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
  if (!driver && driver_candidates.empty())
    reasons = "CUDA driver library not found (NVIDIA ships no CUDA driver for this platform, so no "
              "library is loaded by default; set PARALYN_CUDA_DRIVER_LIBRARY to an absolute path)";
  else if (!driver)
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
    reasons += nvrtc_candidates.empty()
                   ? "NVRTC library not found (no default NVRTC location on this platform; set "
                     "PARALYN_NVRTC_LIBRARY to an absolute path)"
                   : "NVRTC library not found (CUDA Toolkit runtime compiler is required to build kernels)";
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
