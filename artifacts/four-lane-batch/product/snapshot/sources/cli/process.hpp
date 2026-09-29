#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>
namespace paralyn::cli {
struct Process {
  int status = 0;
  std::string out, err;
  bool interrupted = false;
  // Populated by the Capture overload; the original overload leaves them zero.
  std::uint64_t out_bytes = 0, err_bytes = 0;         // total bytes the child wrote
  std::uint64_t out_persisted = 0, err_persisted = 0; // bytes stored in the sidecar files
  bool out_file_truncated = false, err_file_truncated = false;
  bool out_memory_truncated = false, err_memory_truncated = false;
  bool streamed = false; // true when sidecars were written while the child ran
};
// Bounded capture: each stream is appended to its own new sidecar file as it
// arrives (created exclusively, never overwritten) up to file_limit bytes; only
// the first memory_limit bytes are retained in Process::out/err. Bytes beyond a
// limit are counted and reported, never silently claimed as captured. Sidecars
// written before a failure remain on disk.
struct Capture {
  std::string stdout_path, stderr_path;
  std::uint64_t file_limit = std::numeric_limits<std::uint64_t>::max();
  std::size_t memory_limit = std::numeric_limits<std::size_t>::max();
};
// No shell evaluation. Each stream retains its identity, including under redirection.
Process execute(const std::vector<std::string> &arguments, bool live = false,
                const std::vector<std::pair<std::string, std::string>> &environment = {});
Process execute(const std::vector<std::string> &arguments, bool live,
                const std::vector<std::pair<std::string, std::string>> &environment,
                const Capture &capture);
unsigned long process_id();
} // namespace paralyn::cli
