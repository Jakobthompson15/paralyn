#include "process.hpp"
#include <cerrno>
#include <algorithm>
#include <csignal>
#include <fcntl.h>
#include <cstring>
#include <iostream>
#include <map>
#include <poll.h>
#include <spawn.h>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
namespace paralyn::cli {
namespace {
volatile std::sig_atomic_t child = 0;
volatile std::sig_atomic_t interrupted = 0;
void interrupt(int signal) {
  interrupted = signal;
  if (child > 0)
    ::kill(-child, signal);
}
struct Descriptor {
  int fd = -1;
  ~Descriptor() {
    if (fd >= 0)
      close(fd);
  }
  void reset() {
    if (fd >= 0)
      close(fd);
    fd = -1;
  }
};
struct SpawnSetup {
  posix_spawn_file_actions_t files;
  posix_spawnattr_t attributes;
  SpawnSetup() {
    int e = posix_spawn_file_actions_init(&files);
    if (e)
      throw std::runtime_error(std::strerror(e));
    e = posix_spawnattr_init(&attributes);
    if (e) {
      posix_spawn_file_actions_destroy(&files);
      throw std::runtime_error(std::strerror(e));
    }
  }
  ~SpawnSetup() {
    posix_spawn_file_actions_destroy(&files);
    posix_spawnattr_destroy(&attributes);
  }
};
void persist(int fd, const char *data, std::size_t size) {
  while (size) {
    const auto n = ::write(fd, data, size);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      throw std::runtime_error(std::string("cannot write application capture: ") +
                               std::strerror(errno));
    data += n;
    size -= static_cast<std::size_t>(n);
  }
}
void checked(int status) {
  if (status)
    throw std::runtime_error(std::strerror(status));
}
struct ChildGuard {
  pid_t pid = -1;
  sigset_t previous_mask{};
  struct sigaction previous_int{}, previous_term{};
  bool masks = false, handlers = false, reaped = false;
  ~ChildGuard() {
    if (pid > 0 && !reaped) {
      kill(-pid, SIGKILL);
      while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {
      }
    }
    child = 0;
    if (handlers) {
      sigaction(SIGINT, &previous_int, nullptr);
      sigaction(SIGTERM, &previous_term, nullptr);
    }
    if (masks)
      sigprocmask(SIG_SETMASK, &previous_mask, nullptr);
  }
};
} // namespace
unsigned long process_id() { return static_cast<unsigned long>(getpid()); }
Process execute(const std::vector<std::string> &args, bool live,
                const std::vector<std::pair<std::string, std::string>> &env) {
  return execute(args, live, env, Capture{});
}
Process execute(const std::vector<std::string> &args, bool live,
                const std::vector<std::pair<std::string, std::string>> &env,
                const Capture &capture) {
  if (args.empty())
    throw std::runtime_error("empty process command");
  // Sidecars exist before the child starts; O_EXCL never truncates prior evidence.
  Descriptor sidecars[2];
  const std::string *paths[2] = {&capture.stdout_path, &capture.stderr_path};
  for (unsigned i = 0; i < 2; ++i)
    if (!paths[i]->empty()) {
      sidecars[i].fd = ::open(paths[i]->c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
      if (sidecars[i].fd < 0)
        throw std::runtime_error("cannot create application capture " + *paths[i] + ": " +
                                 std::strerror(errno));
    }
  std::map<std::string, std::string> variables;
  for (char **p = environ; *p; ++p) {
    std::string entry = *p;
    auto equals = entry.find('=');
    if (equals != std::string::npos)
      variables[entry.substr(0, equals)] = entry.substr(equals + 1);
  }
  for (const auto &value : env)
    variables[value.first] = value.second;
  std::vector<std::string> environment;
  for (const auto &value : variables)
    environment.push_back(value.first + "=" + value.second);
  std::vector<char *> envp, argv;
  for (auto &value : environment)
    envp.push_back(value.data());
  envp.push_back(nullptr);
  for (const auto &value : args)
    argv.push_back(const_cast<char *>(value.c_str()));
  argv.push_back(nullptr);
  Descriptor out_read, out_write, err_read, err_write;
  int pipe_fds[2];
  if (pipe(pipe_fds))
    throw std::runtime_error(std::strerror(errno));
  out_read.fd = pipe_fds[0];
  out_write.fd = pipe_fds[1];
  if (pipe(pipe_fds))
    throw std::runtime_error(std::strerror(errno));
  err_read.fd = pipe_fds[0];
  err_write.fd = pipe_fds[1];
  SpawnSetup spawn;
  checked(posix_spawn_file_actions_adddup2(&spawn.files, out_write.fd, STDOUT_FILENO));
  checked(posix_spawn_file_actions_adddup2(&spawn.files, err_write.fd, STDERR_FILENO));
  for (auto *fd : {&out_read, &out_write, &err_read, &err_write})
    checked(posix_spawn_file_actions_addclose(&spawn.files, fd->fd));
  ChildGuard guard;
  sigset_t blocked;
  sigemptyset(&blocked);
  sigaddset(&blocked, SIGINT);
  sigaddset(&blocked, SIGTERM);
  if (sigprocmask(SIG_BLOCK, &blocked, &guard.previous_mask))
    throw std::runtime_error("cannot mask interruption during process start");
  guard.masks = true;
  sigset_t defaults;
  sigemptyset(&defaults);
  sigaddset(&defaults, SIGINT);
  sigaddset(&defaults, SIGTERM);
  checked(posix_spawnattr_setsigdefault(&spawn.attributes, &defaults));
  checked(posix_spawnattr_setsigmask(&spawn.attributes, &guard.previous_mask));
  checked(posix_spawnattr_setpgroup(&spawn.attributes, 0));
  checked(posix_spawnattr_setflags(
      &spawn.attributes, POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK));
  // No fork-time allocation/setenv after Metal has created runtime threads.
  pid_t spawned = -1;
  checked(
      posix_spawnp(&spawned, argv[0], &spawn.files, &spawn.attributes, argv.data(), envp.data()));
  guard.pid = spawned;
  out_write.reset();
  err_write.reset();
  child = guard.pid;
  interrupted = 0;
  struct sigaction action{};
  action.sa_handler = interrupt;
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGINT, &action, &guard.previous_int))
    throw std::runtime_error("cannot install interrupt handler");
  if (sigaction(SIGTERM, &action, &guard.previous_term)) {
    sigaction(SIGINT, &guard.previous_int, nullptr);
    throw std::runtime_error("cannot install termination handler");
  }
  guard.handlers = true;
  if (sigprocmask(SIG_SETMASK, &guard.previous_mask, nullptr))
    throw std::runtime_error("cannot restore signal mask");
  Process result;
  result.streamed = sidecars[0].fd >= 0 || sidecars[1].fd >= 0;
  pollfd descriptors[2]{{out_read.fd, POLLIN, 0}, {err_read.fd, POLLIN, 0}};
  unsigned open = 2;
  while (open) {
    if (poll(descriptors, 2, -1) < 0) {
      if (errno == EINTR)
        continue;
      throw std::runtime_error(std::strerror(errno));
    }
    for (unsigned i = 0; i < 2; ++i) {
      auto &fd = descriptors[i];
      if (fd.fd < 0 || !fd.revents)
        continue;
      char buffer[8192];
      const auto count = read(fd.fd, buffer, sizeof(buffer));
      if (count > 0) {
        const auto size = static_cast<std::size_t>(count);
        (i ? result.err_bytes : result.out_bytes) += size;
        if (sidecars[i].fd >= 0) {
          auto &stored = i ? result.err_persisted : result.out_persisted;
          const auto room = static_cast<std::size_t>(
              std::min<std::uint64_t>(size, capture.file_limit - stored));
          persist(sidecars[i].fd, buffer, room);
          stored += room;
          if (room < size)
            (i ? result.err_file_truncated : result.out_file_truncated) = true;
        }
        auto &text = i ? result.err : result.out;
        const auto keep = std::min(size, capture.memory_limit - std::min(capture.memory_limit, text.size()));
        text.append(buffer, keep);
        if (keep < size)
          (i ? result.err_memory_truncated : result.out_memory_truncated) = true;
        if (live) {
          auto &stream = i ? std::cerr : std::cout;
          stream.write(buffer, count);
          stream.flush();
          if (!stream)
            throw std::runtime_error("cannot forward application output");
        }
      } else if (!count) {
        (i ? err_read : out_read).reset();
        fd.fd = -1;
        --open;
      } else if (errno != EINTR && errno != EAGAIN)
        throw std::runtime_error(std::strerror(errno));
    }
  }
  int status = 0;
  pid_t waited;
  do {
    waited = waitpid(guard.pid, &status, 0);
  } while (waited < 0 && errno == EINTR);
  if (waited < 0)
    throw std::runtime_error("cannot collect child exit status");
  guard.reaped = true;
  result.interrupted = interrupted != 0;
  result.status = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
  return result;
}
} // namespace paralyn::cli
