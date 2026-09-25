#include "unicuda/frontend.hpp"
#include "unicuda/runtime.hpp"
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
namespace {
struct Process {
  int status;
  std::string output;
};
Process execute(const std::vector<std::string> &args, bool echo = false,
                const std::vector<std::pair<std::string, std::string>> &env = {}) {
  int pipes[2];
  if (pipe(pipes))
    throw std::runtime_error(std::strerror(errno));
  auto pid = fork();
  if (pid < 0) {
    close(pipes[0]);
    close(pipes[1]);
    throw std::runtime_error("could not fork compiler/program");
  }
  if (pid == 0) {
    close(pipes[0]);
    dup2(pipes[1], STDOUT_FILENO);
    dup2(pipes[1], STDERR_FILENO);
    close(pipes[1]);
    for (const auto &[key, value] : env)
      setenv(key.c_str(), value.c_str(), 1);
    std::vector<char *> argv;
    for (const auto &a : args)
      argv.push_back(const_cast<char *>(a.c_str()));
    argv.push_back(nullptr);
    execvp(argv[0], argv.data());
    std::cerr << "UniCUDAError: cannot execute " << args[0] << ": " << std::strerror(errno) << "\n";
    _exit(127);
  }
  close(pipes[1]);
  Process result{};
  char buffer[4096];
  while (true) {
    ssize_t n = read(pipes[0], buffer, sizeof(buffer));
    if (n == 0)
      break;
    if (n < 0) {
      if (errno == EINTR)
        continue;
      break;
    }
    result.output.append(buffer, static_cast<std::size_t>(n));
    if (echo) {
      std::cout.write(buffer, n);
      std::cout.flush();
    }
  }
  close(pipes[0]);
  int status = 0;
  while (waitpid(pid, &status, 0) < 0)
    if (errno != EINTR)
      throw std::runtime_error("waitpid failed");
  result.status = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
  return result;
}
void write(const fs::path &p, const std::string &s) {
  std::ofstream out(p);
  if (!out)
    throw std::runtime_error("cannot write " + p.string());
  out << s;
  if (!out)
    throw std::runtime_error("write failed: " + p.string());
}
std::string trim(std::string s) {
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r'))
    s.pop_back();
  return s;
}
std::string ir_text(const unicuda::FrontendResult &r) {
  std::string s;
  for (const auto &k : r.kernels)
    s += unicuda::dump_ir(k) + "\n";
  return s;
}
void usage() {
  std::cout << "UniCUDA v0.0.1\nUsage:\n  unicuda devices\n  unicuda inspect program.cu\n  unicuda "
               "run program.cu [--device auto|INDEX] [--artifacts DIR] [-- program arguments]\n";
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      usage();
      return 2;
    }
    std::string command = argv[1];
    if (command == "--version") {
      std::cout << "UniCUDA v0.0.1 (LLVM " << UNICUDA_LLVM_VERSION << ")\n";
      return 0;
    }
    if (command == "devices") {
      if (argc != 2)
        throw std::runtime_error("devices accepts no arguments");
      std::cout << unicuda::devices_text();
      return 0;
    }
    if ((command != "run" && command != "inspect") || argc < 3) {
      usage();
      return 2;
    }
    auto source = fs::absolute(argv[2]);
    std::string device = "auto";
    fs::path artifact;
    std::vector<std::string> program_args;
    for (int i = 3; i < argc; ++i) {
      std::string a = argv[i];
      if (command == "inspect")
        throw std::runtime_error("inspect accepts one source file");
      if (a == "--") {
        for (++i; i < argc; ++i)
          program_args.emplace_back(argv[i]);
        break;
      }
      if ((a == "--device" || a == "--artifacts") && i + 1 < argc) {
        if (a == "--device")
          device = argv[++i];
        else
          artifact = fs::absolute(argv[++i]);
      } else
        throw std::runtime_error("unknown/incomplete option: " + a);
    }
    auto frontend = unicuda::compile_source(source.string());
    if (command == "inspect") {
      std::cout << "Detected kernels:\n";
      for (const auto &k : frontend.kernels) {
        std::cout << "  " << k.name << "(";
        for (std::size_t i = 0; i < k.parameters.size(); ++i) {
          const auto &p = k.parameters[i];
          if (i)
            std::cout << ", ";
          if (p.read_only)
            std::cout << "const ";
          std::cout << unicuda::type_name(p.type) << (p.buffer ? "* " : " ") << p.name;
        }
        std::cout << ")\n";
      }
      std::cout << "\nLaunches (expressions; runtime values are not evaluated):\n";
      for (const auto &l : frontend.launches)
        std::cout << "  " << l.kernel << " at line " << l.line << ": grid=" << l.grid_expression
                  << ", block=" << l.block_expression << "\n";
      std::cout << "\nRequired backend capabilities: typed buffers, i32/u32/f32, index "
                   "builtins\n\nUniCUDA IR:\n"
                << ir_text(frontend);
      return 0;
    }
    auto stamp = std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count();
    auto id = std::to_string(stamp) + "-" + std::to_string(getpid());
    fs::path work = fs::path(UNICUDA_BINARY_DIR) / "runs" / id;
    fs::create_directories(work);
    if (artifact.empty())
      artifact = fs::path(UNICUDA_SOURCE_DIR) / "artifacts" / "runs" / id;
    if (fs::exists(artifact) && !fs::is_empty(artifact))
      throw std::runtime_error(
          "artifact directory is not empty; refusing to overwrite execution evidence");
    fs::create_directories(artifact);
    fs::copy_file(source, artifact / "source.cu");
    write(artifact / "unicuda-ir.txt", ir_text(frontend));
    auto host = work / "host.cpp";
    write(host, frontend.rewritten_host);
    auto executable = work / "program";
    std::vector<std::string> compile = {UNICUDA_HOST_CXX,
                                        "-std=c++17",
                                        "-O0",
                                        "-g",
                                        "-fno-fast-math",
                                        "-ffp-contract=off",
                                        "-mmacosx-version-min=" UNICUDA_DEPLOYMENT_TARGET,
                                        "-I",
                                        UNICUDA_INCLUDE_DIR,
                                        "-iquote",
                                        source.parent_path().string(),
                                        host.string(),
                                        UNICUDA_RUNTIME_ARCHIVE,
                                        UNICUDA_IR_ARCHIVE,
                                        "-framework",
                                        "Metal",
                                        "-framework",
                                        "Foundation",
                                        "-o",
                                        executable.string()};
    auto compiled = execute(compile);
    if (compiled.status) {
      std::cerr << compiled.output;
      write(artifact / "verification.txt", "Host compilation failed\n" + compiled.output);
      return compiled.status;
    }
    auto commit = execute({"git", "-C", UNICUDA_SOURCE_DIR, "rev-parse", "HEAD"});
    auto dirty =
        execute({"git", "-C", UNICUDA_SOURCE_DIR, "status", "--porcelain", "--untracked-files=no"});
    std::vector<std::string> run = {executable.string()};
    run.insert(run.end(), program_args.begin(), program_args.end());
    std::cout << "UniCUDA v0.0.1\n\n" << std::flush;
    auto result = execute(run, true,
                          {{"UNICUDA_DEVICE", device},
                           {"UNICUDA_ARTIFACT_DIR", artifact.string()},
                           {"UNICUDA_COMMIT", commit.status ? "uncommitted" : trim(commit.output)},
                           {"UNICUDA_LLVM_VERSION", UNICUDA_LLVM_VERSION},
                           {"UNICUDA_SOURCE_DIRTY", dirty.output.empty() ? "false" : "true"}});
    write(artifact / "verification.txt",
          result.output + "\nHost exit status: " + std::to_string(result.status) + "\n");
    if (result.status)
      std::cerr << "UniCUDAError: program exited with status " << result.status << "\n";
    return result.status;
  } catch (const std::exception &e) {
    std::cerr << "UniCUDAError: " << e.what() << "\n";
    return 1;
  }
}
