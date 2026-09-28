#include "process.hpp"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <iostream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
void check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
std::string pattern(unsigned channel) {
  std::string result(1024 * 1024, '\0');
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = char((i + channel * 17) % 251);
  return result;
}
void write_all(int fd, const std::string &data) {
  std::size_t done = 0;
  while (done < data.size()) {
    const auto n = ::write(fd, data.data() + done, data.size() - done);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) std::_Exit(2);
    done += static_cast<std::size_t>(n);
  }
}
std::string encode(const std::string &value) { return std::to_string(value.size()) + ":" + value; }
unsigned descriptor_count() {
  unsigned count = 0;
  for (int fd = 0; fd < 512; ++fd) if (fcntl(fd, F_GETFD) >= 0) ++count;
  return count;
}
volatile std::sig_atomic_t restored_signal = 0;
void saved_handler(int signal) { restored_signal = signal; }
class ThrowingOutput : public std::streambuf {
  std::streamsize xsputn(const char *, std::streamsize) override {
    throw std::runtime_error("deliberate process-output failure");
  }
  int overflow(int) override { throw std::runtime_error("deliberate process-output failure"); }
};
}
int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "--streams") {
    std::thread a([] { write_all(STDOUT_FILENO, pattern(0)); });
    std::thread b([] { write_all(STDERR_FILENO, pattern(1)); });
    a.join(); b.join();
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "--arguments") {
    for (int i = 2; i < argc; ++i) std::cout << encode(argv[i]);
    const char *value = std::getenv("PARALYN_PROCESS_TEST");
    std::cerr << encode(value ? value : "<missing>");
    return 37;
  }
  if (argc == 3 && std::string(argv[1]) == "--wait") {
    std::ofstream ready(argv[2]); ready << getpid(); ready.close();
    write_all(STDOUT_FILENO, "child is ready\n");
    alarm(5); // A broken forwarding test must never leave an unbounded child.
    for (;;) pause();
  }
  try {
    using paralyn::cli::execute;
    const auto self = std::filesystem::absolute(argv[0]).string();
    const auto streamed = execute({self, "--streams"});
    check(streamed.status == 0 && streamed.out == pattern(0) && streamed.err == pattern(1),
          "Simultaneous binary stdout/stderr were lost, mixed or deadlocked");
    {
      // Bounded streaming sidecars: exact prefixes on disk, counted excess, bounded memory.
      const auto root = std::filesystem::temp_directory_path() / ("paralyn-capture-" + std::to_string(getpid()));
      std::filesystem::create_directory(root);
      auto slurp = [](const std::filesystem::path &p) {
        std::ifstream f(p, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(f), {});
      };
      paralyn::cli::Capture bounded;
      bounded.stdout_path = (root / "bounded.stdout").string();
      bounded.stderr_path = (root / "bounded.stderr").string();
      bounded.file_limit = 300000;
      bounded.memory_limit = 1000;
      const auto limited = execute({self, "--streams"}, false, {}, bounded);
      check(limited.status == 0 && limited.streamed && limited.out_bytes == pattern(0).size() &&
                limited.err_bytes == pattern(1).size(), "Capture byte counts are wrong");
      check(limited.out_persisted == 300000 && limited.err_persisted == 300000 &&
                limited.out_file_truncated && limited.err_file_truncated,
            "Capture file limit was not enforced or reported");
      check(slurp(bounded.stdout_path) == pattern(0).substr(0, 300000) &&
                slurp(bounded.stderr_path) == pattern(1).substr(0, 300000),
            "Capture sidecars do not hold the exact stream prefixes");
      check(limited.out == pattern(0).substr(0, 1000) && limited.err == pattern(1).substr(0, 1000) &&
                limited.out_memory_truncated && limited.err_memory_truncated,
            "Capture memory limit was not enforced or reported");
      paralyn::cli::Capture full;
      full.stdout_path = (root / "full.stdout").string();
      full.stderr_path = (root / "full.stderr").string();
      const auto complete = execute({self, "--streams"}, false, {}, full);
      check(complete.out == pattern(0) && slurp(full.stdout_path) == pattern(0) &&
                slurp(full.stderr_path) == pattern(1) && !complete.out_file_truncated &&
                !complete.out_memory_truncated, "Unbounded capture lost data");
      const auto count = descriptor_count();
      bool refused = false;
      try { execute({self, "--streams"}, false, {}, full); } catch (const std::exception &) { refused = true; }
      check(refused && slurp(full.stdout_path) == pattern(0), "Capture overwrote an existing sidecar");
      check(descriptor_count() == count, "Refused capture leaked descriptors");
      std::filesystem::remove_all(root);
    }
    const std::vector<std::string> exact{"", "white space", "quote\"backslash\\", "line\nbreak", "Unicode Δ 🍎", "--not-an-option"};
    std::vector<std::string> command{self, "--arguments"}; command.insert(command.end(), exact.begin(), exact.end());
    const std::string environment = "environment = Unicode Δ\nquote\"";
    const auto arguments = execute(command, false, {{"PARALYN_PROCESS_TEST", environment}});
    std::string expected; for (const auto &value : exact) expected += encode(value);
    check(arguments.status == 37 && arguments.out == expected && arguments.err == encode(environment),
          "Argument/environment boundaries or application exit status changed");
    const auto before = descriptor_count();
    for (unsigned i = 0; i < 3; ++i) {
      bool rejected = false;
      try { execute({"/nonexistent/paralyn/process-test"}); } catch (const std::exception &) { rejected = true; }
      check(rejected, "Spawn failure was accepted");
    }
    check(descriptor_count() == before, "Spawn failure leaked pipe descriptors");
    const auto directory = std::filesystem::temp_directory_path() / ("paralyn-process-" + std::to_string(getpid()));
    std::filesystem::create_directory(directory);
    const auto ready = directory / "signal-ready";
    struct sigaction custom{}, previous{}; custom.sa_handler = saved_handler; sigemptyset(&custom.sa_mask);
    check(sigaction(SIGINT, &custom, &previous) == 0, "Cannot install test handler");
    std::atomic<bool> sent{false};
    std::thread interrupter([&] {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
      while (!std::filesystem::exists(ready) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      if (std::filesystem::exists(ready)) { sent = true; kill(getpid(), SIGINT); }
    });
    paralyn::cli::Process interrupted;
    try { interrupted = execute({self, "--wait", ready.string()}); }
    catch (...) { interrupter.join(); sigaction(SIGINT, &previous, nullptr); throw; }
    interrupter.join();
    struct sigaction after{}; sigaction(SIGINT, nullptr, &after);
    const bool restored = after.sa_handler == saved_handler;
    raise(SIGINT);
    sigaction(SIGINT, &previous, nullptr);
    check(sent && interrupted.interrupted && interrupted.status == 128 + SIGINT,
          "Interruption did not reach the child process group or preserve status");
    check(restored && restored_signal == SIGINT, "Original signal handler was not restored");
    const auto failing_ready = directory / "failure-ready";
    ThrowingOutput bad;
    auto *original = std::cout.rdbuf(&bad);
    const auto old_exceptions = std::cout.exceptions();
    std::cout.exceptions(std::ios::badbit | std::ios::failbit);
    bool rejected = false;
    try { execute({self, "--wait", failing_ready.string()}, true); }
    catch (const std::exception &) { rejected = true; }
    std::cout.exceptions(std::ios::goodbit); std::cout.clear(); std::cout.rdbuf(original); std::cout.exceptions(old_exceptions);
    check(rejected, "Output-forwarding failure was ignored");
    pid_t child = -1; std::ifstream(failing_ready) >> child;
    check(child > 0 && kill(child, 0) == -1 && errno == ESRCH,
          "Output failure left the child alive");
    check(waitpid(child, nullptr, WNOHANG) == -1 && errno == ECHILD,
          "Output failure left a zombie child");
    check(descriptor_count() == before, "Process execution leaked descriptors");
    std::filesystem::remove_all(directory);
    std::cout << "POSIX process streams, exact argv/env/exit, failure cleanup and SIGINT forwarding: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Process tests failed: " << error.what() << '\n';
    return 1;
  }
}
