#pragma once
#include <string>
#include <utility>
#include <vector>
namespace paralyn::cli {
struct Process {
  int status = 0;
  std::string out, err;
  bool interrupted = false;
};
// No shell evaluation. Each stream retains its identity, including under redirection.
Process execute(const std::vector<std::string> &arguments, bool live = false,
                const std::vector<std::pair<std::string, std::string>> &environment = {});
unsigned long process_id();
} // namespace paralyn::cli
