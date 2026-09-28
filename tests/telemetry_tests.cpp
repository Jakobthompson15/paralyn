#include "paralyn/detail/backend.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
void set(const char *key, const std::string &value) {
#ifdef _WIN32
  if (_putenv_s(key, value.c_str())) throw std::runtime_error("set environment failed");
#else
  if (setenv(key, value.c_str(), 1)) throw std::runtime_error("set environment failed");
#endif
}
void check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
template<class F> void fails(F &&operation) {
  try { operation(); }
  catch (const paralyn::backend::Error &error) {
    check(error.code == paralyn::backend::ErrorCode::internal, "Wrong logging failure code");
    return;
  }
  throw std::runtime_error("Unwritable log was silently accepted");
}
std::string contents(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), {}};
}
}
int main() {
  namespace fs = std::filesystem;
  try {
    const auto root = fs::temp_directory_path() / ("paralyn-telemetry-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(root);
    set("PARALYN_RUNTIME_LOG", (root / "runtime.log").string());
    paralyn::backend::progress("first\n");
    paralyn::backend::progress("second\n");
    check(contents(root / "runtime.log") == "first\nsecond\n", "Progress did not append exactly");
    set("PARALYN_RUNTIME_LOG", "");
    std::ostringstream error, output;
    auto *saved_error = std::cerr.rdbuf(error.rdbuf());
    auto *saved_output = std::cout.rdbuf(output.rdbuf());
    try { paralyn::backend::progress("stderr only\n"); }
    catch (...) { std::cerr.rdbuf(saved_error); std::cout.rdbuf(saved_output); throw; }
    std::cerr.rdbuf(saved_error);
    std::cout.rdbuf(saved_output);
    check(output.str().empty() && error.str() == "stderr only\n", "Runtime progress polluted stdout");
    set("PARALYN_EVENT_LOG", (root / "events.ndjson").string());
    const std::string detail = "quote=\" slash=\\ newline=\n tab=\t utf8=Δ";
    paralyn::backend::runtime_event("test", "completed", detail, 42, 7, 4096);
    std::vector<std::thread> workers;
    for (unsigned i = 0; i < 4; ++i) workers.emplace_back([i] {
      for (unsigned n = 0; n < 5; ++n)
        paralyn::backend::runtime_event("thread_test", "completed", std::to_string(i), i, n);
    });
    for (auto &worker : workers) worker.join();
    std::ifstream input(root / "events.ndjson");
    std::string line;
    std::uint64_t previous = 0;
    unsigned count = 0;
    while (std::getline(input, line)) {
      const auto event = nlohmann::json::parse(line);
      check(event.at("schema") == "paralyn.runtime.event" && event.at("schema_version") == 1,
            "Wrong runtime event schema");
      const auto sequence = event.at("sequence").get<std::uint64_t>();
      check(sequence > previous, "Concurrent event records are out of order or interleaved");
      previous = sequence;
      if (count++ == 0)
        check(event.at("detail") == detail && event.at("context_id") == 42 &&
              event.at("operation_id") == 7 && event.at("bytes") == 4096,
              "Structured event lost details or escaped incorrectly");
    }
    check(count == 21, "Structured log lost records");
    set("PARALYN_RUNTIME_LOG", root.string());
    fails([] { paralyn::backend::progress("cannot append to directory"); });
    set("PARALYN_EVENT_LOG", root.string());
    fails([] { paralyn::backend::runtime_event("test", "failed", "unwritable destination"); });
    set("PARALYN_RUNTIME_LOG", "");
    set("PARALYN_EVENT_LOG", "");
    fs::remove_all(root);
    std::cout << "Runtime stream isolation, durable structured logs and failure propagation: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Telemetry tests failed: " << error.what() << '\n';
    return 1;
  }
}
