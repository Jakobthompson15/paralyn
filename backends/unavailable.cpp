#include "paralyn/detail/backend.hpp"
namespace paralyn::backend {
std::vector<DeviceInfo> devices() { return {}; }
std::shared_ptr<Context> create_context(const std::string &) {
  throw Error(ErrorCode::invalid_device,
              "This build contains no Metal backend and no available CUDA device was found for this "
              "selector (use cuda:INDEX with an NVIDIA driver and NVRTC; see paralyn devices). "
              "HIP/ROCm is not implemented. There is no CPU fallback.");
}
} // namespace paralyn::backend
