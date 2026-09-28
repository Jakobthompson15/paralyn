#include "paralyn/detail/backend.hpp"
namespace paralyn::backend {
std::vector<DeviceInfo> devices() { return {}; }
std::shared_ptr<Context> create_context(const std::string &) {
  throw Error(ErrorCode::invalid_device,
              "This build contains no GPU execution backend. CUDA/NVIDIA and HIP/ROCm backends are "
              "not implemented; installing an SDK alone cannot enable them.");
}
} // namespace paralyn::backend
