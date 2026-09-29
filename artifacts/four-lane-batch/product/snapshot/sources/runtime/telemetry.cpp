#include "paralyn/detail/backend.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace paralyn::backend {
namespace {
std::mutex log_mutex;
std::uint64_t sequence = 0;
std::string quote(const std::string &text) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : text) {
    if (c == '"' || c == '\\')
      out << '\\' << char(c);
    else if (c < 32)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
    else
      out << char(c);
  }
  out << '"';
  return out.str();
}
} // namespace
void progress(const std::string &message) {
  std::lock_guard<std::mutex> lock(log_mutex);
  const char *path = std::getenv("PARALYN_RUNTIME_LOG");
  if (path && *path) {
    std::ofstream output(path, std::ios::app);
    output << message;
    output.close();
    if (!output)
      throw Error(ErrorCode::internal, "Cannot write PARALYN_RUNTIME_LOG");
  } else {
    std::cerr << message << std::flush;
    if (!std::cerr)
      throw Error(ErrorCode::internal, "Cannot write runtime progress to stderr");
  }
}
void runtime_event(const std::string &category, const std::string &status,
                   const std::string &detail, std::uint64_t context_id, std::uint64_t operation_id,
                   std::uint64_t bytes) {
  const char *path = std::getenv("PARALYN_EVENT_LOG");
  if (!path || !*path)
    return;
  std::lock_guard<std::mutex> lock(log_mutex);
  const auto host_time =
      std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  std::ostringstream record;
  record << std::setprecision(17)
         << "{\"schema\":\"paralyn.runtime.event\",\"schema_version\":1,\"sequence\":" << ++sequence
         << ",\"category\":" << quote(category) << ",\"status\":" << quote(status)
         << ",\"detail\":" << quote(detail) << ",\"context_id\":" << context_id
         << ",\"operation_id\":" << operation_id << ",\"bytes\":" << bytes
         << ",\"host_monotonic_seconds\":" << host_time << "}\n";
  const auto line = record.str();
  auto *file = std::fopen(path, "ab");
  if (!file)
    throw Error(ErrorCode::internal, "Cannot open PARALYN_EVENT_LOG");
  bool okay = std::fwrite(line.data(), 1, line.size(), file) == line.size();
  if (std::fflush(file))
    okay = false;
#ifdef _WIN32
  if (_commit(_fileno(file)))
    okay = false;
#else
  if (fsync(fileno(file)))
    okay = false;
#endif
  if (std::fclose(file))
    okay = false;
  if (!okay)
    throw Error(ErrorCode::internal, "Cannot durably append PARALYN_EVENT_LOG");
}
} // namespace paralyn::backend
