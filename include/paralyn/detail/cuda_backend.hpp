#pragma once
// Internal: dynamically loaded CUDA Driver API + NVRTC backend and the small
// selector registry shared by the native ABI and CLI. Not a public API.
#include "paralyn/detail/backend.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace paralyn::backend {
namespace cuda {
struct Status {
  bool implemented = true;                   // code is compiled into this build
  bool driver_loaded = false, nvrtc_loaded = false, initialized = false;
  bool available = false;                    // loaded, initialized, and at least one device
  int driver_version = 0;                    // cuDriverGetVersion encoding (1000*major+10*minor)
  int nvrtc_major = 0, nvrtc_minor = 0;
  int device_count = 0;
  std::string driver_library, nvrtc_library, reason;
  std::vector<std::string> attempts;         // every failed candidate with its loader error
};
// Loads libcuda/nvcuda and NVRTC once per process. Never throws for absence.
Status status();
// Empty unless status().available. Never fabricates a device.
std::vector<DeviceInfo> devices();
// DeviceInfo::registry_id reported for CUDA ordinal N: 0x43554441'00000000
// ("CUDA") | (N + 1). Nonzero and backend-tagged; not a hardware identity.
constexpr std::uint64_t registry_id_tag = 0x4355444100000000ULL;
constexpr std::uint64_t cuda_registry_id(std::uint32_t ordinal) {
  return registry_id_tag | (std::uint64_t(ordinal) + 1);
}
// selector must be "cuda:INDEX"; throws Error(invalid_device) when unavailable.
std::shared_ptr<Context> create_context(const std::string &selector);
} // namespace cuda

enum class SelectorKind { automatic, metal, cuda, hip };
struct Selector {
  SelectorKind kind = SelectorKind::automatic;
  // cuda: parsed ordinal. metal: the original text is passed unchanged to the
  // Metal backend so its existing validation/messages are preserved.
  std::uint32_t index = 0;
  std::string text;
};
// Throws Error(invalid_device) for malformed cuda:/hip:/rocm: selectors.
Selector parse_selector(const std::string &selector);
// Metal devices first (unchanged order/indices), then available CUDA devices.
std::vector<DeviceInfo> registry_devices();
// "cuda:N" -> CUDA; "hip:N"/"rocm:N" -> not implemented error; everything else
// -> Metal exactly as before. "auto" uses the first Metal device; only when no
// Metal device exists does it use cuda:0 if the CUDA backend is available.
std::shared_ptr<Context> registry_create_context(const std::string &selector);
} // namespace paralyn::backend
