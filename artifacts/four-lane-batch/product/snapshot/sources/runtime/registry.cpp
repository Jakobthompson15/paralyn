// Selector routing across the execution backends compiled into this build.
// Metal keeps its existing selectors, indices and messages unchanged; CUDA is
// reached with cuda:N; HIP/ROCm is reported as not implemented.
#include "paralyn/detail/cuda_backend.hpp"

namespace paralyn::backend {
namespace {
bool starts_with(const std::string &text, const char *prefix) {
  return text.rfind(prefix, 0) == 0;
}
} // namespace
Selector parse_selector(const std::string &selector) {
  Selector result;
  result.text = selector;
  if (starts_with(selector, "cuda:")) {
    result.kind = SelectorKind::cuda;
    // Validates the complete selector before any library is loaded.
    const std::string digits = selector.substr(5);
    if (digits.empty() || digits.size() > 10 ||
        digits.find_first_not_of("0123456789") != std::string::npos)
      throw Error(ErrorCode::invalid_device, "CUDA selectors have the form cuda:INDEX (decimal device ordinal)");
    const auto value = std::stoull(digits);
    if (value > 0x7fffffffULL) throw Error(ErrorCode::invalid_device, "CUDA device ordinal is out of range for cuda:INDEX");
    result.index = static_cast<std::uint32_t>(value);
    return result;
  }
  if (starts_with(selector, "hip:") || starts_with(selector, "rocm:")) {
    result.kind = SelectorKind::hip;
    return result;
  }
  result.kind = selector == "auto" ? SelectorKind::automatic : SelectorKind::metal;
  return result;
}
std::vector<DeviceInfo> registry_devices() {
  auto result = devices();
  for (auto &device : cuda::devices()) result.push_back(std::move(device));
  return result;
}
std::shared_ptr<Context> registry_create_context(const std::string &selector) {
  const auto parsed = parse_selector(selector);
  switch (parsed.kind) {
  case SelectorKind::cuda:
    return cuda::create_context(selector);
  case SelectorKind::hip:
    throw Error(ErrorCode::invalid_device,
                "The HIP/ROCm backend is not implemented. Installing ROCm alone cannot enable it.");
  case SelectorKind::automatic:
    if (devices().empty() && cuda::status().available) return cuda::create_context("cuda:0");
    return create_context(selector);
  case SelectorKind::metal:
    break;
  }
  return create_context(selector);
}
} // namespace paralyn::backend
