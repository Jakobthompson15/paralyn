#pragma once
// Private engine interface. The explicit-table entry points exist so tests can
// exercise host-side logic with an in-process test double; production code
// always passes the table produced by the dynamic loader (cached_load()).
#include "api.hpp"
#include "paralyn/detail/backend.hpp"

namespace paralyn::backend::cuda {
ErrorCode map_result(CUresult code);
// Errors after which the CUDA context is unusable (sticky launch/device faults).
bool is_sticky(CUresult code);
std::uint32_t parse_ordinal(const std::string &selector); // "cuda:N"
std::vector<DeviceInfo> devices(const Api &api);
std::shared_ptr<Context> create_context(std::shared_ptr<const Api> api, std::uint32_t ordinal);
// "compute_XY" chosen from device capability and NVRTC's supported list.
std::string choose_architecture(const NvrtcApi &nvrtc, int major, int minor);
} // namespace paralyn::backend::cuda
