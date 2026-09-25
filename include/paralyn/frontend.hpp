#pragma once
#include "paralyn/ir.hpp"
#include <string>
#include <vector>

namespace paralyn {
struct LaunchInfo {
  std::string kernel;
  std::string grid_expression;
  std::string block_expression;
  unsigned line = 0;
};
struct FrontendResult {
  std::vector<Kernel> kernels;
  std::vector<LaunchInfo> launches;
  std::string rewritten_host;
};
FrontendResult compile_source(const std::string &path);
} // namespace paralyn
