// GLSL (glslang) and HLSL (DXC) compute frontends on top of the pinned SPIR-V
// importer. Each compiler runs as a separate worker process built from a
// pinned source archive (cmake/ParalynShaderCompilers.cmake); no compiler
// library is linked into Paralyn, so DXC's LLVM 3.7 fork never shares an
// address space or ABI with the Clang 21 CUDA frontend.
#include "paralyn/shader.hpp"
#include "paralyn_shader_toolchain.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <poll.h>
#include <regex>
#include <spawn.h>
#include <sstream>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace paralyn::shader {
namespace {
constexpr std::size_t max_source_bytes = 1u << 20;
constexpr std::size_t max_stream_bytes = 1u << 20;
constexpr int worker_timeout_seconds = 120;

const char *label(Language l) { return l == Language::Glsl ? "GLSL" : "HLSL"; }

[[noreturn]] void fail(Language l, const std::string &suffix, const std::string &detail,
                       std::vector<SourceDiagnostic> diagnostics = {}) {
  throw CompileError(std::string(language_name(l)) + "." + suffix,
                     std::string("ParalynError: ") + label(l) + " frontend [" + language_name(l) +
                         "." + suffix + "]: " + detail,
                     std::move(diagnostics));
}

bool utf8(const std::string &s) {
  std::size_t i = 0;
  while (i < s.size()) {
    const auto c = static_cast<unsigned char>(s[i]);
    std::size_t n = c < 0x80 ? 0 : (c >> 5) == 6 ? 1 : (c >> 4) == 14 ? 2 : (c >> 3) == 30 ? 3 : 9;
    if (n == 9 || i + n >= s.size() + (n ? 0 : 1))
      return false;
    std::uint32_t cp = n == 0 ? c : n == 1 ? (c & 0x1f) : n == 2 ? (c & 0x0f) : (c & 0x07);
    for (std::size_t k = 1; k <= n; ++k) {
      const auto d = static_cast<unsigned char>(s[i + k]);
      if ((d & 0xc0) != 0x80)
        return false;
      cp = (cp << 6) | (d & 0x3f);
    }
    if ((n == 1 && cp < 0x80) || (n == 2 && cp < 0x800) || (n == 3 && cp < 0x10000) ||
        cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff))
      return false;
    i += n + 1;
  }
  return true;
}

bool identifier(const std::string &s) {
  static const std::regex pattern("[A-Za-z_][A-Za-z0-9_]{0,63}");
  return std::regex_match(s, pattern);
}

// One view of the source after translation phases 2-3: backslash-newline
// splicing (strict: backslash immediately before the newline; lenient: also
// with trailing blanks, as clang accepts) and comments replaced by one space,
// optionally honouring "..." / '...' literals (clang does; glslang has no
// character literals). Each kept character remembers its physical line.
struct CleanChar {
  char c;
  std::uint32_t line;
};
std::vector<CleanChar> clean_view(const std::string &s, bool lenient_splice, bool literals) {
  // Phase 1/2: newlines normalized (\r\n and lone \r count as one newline), splices removed.
  std::vector<CleanChar> spliced;
  spliced.reserve(s.size());
  std::uint32_t line = 1;
  auto newline_at = [&](std::size_t i) -> std::size_t { // length of a newline at i, or 0
    if (i >= s.size())
      return 0;
    if (s[i] == '\r')
      return i + 1 < s.size() && s[i + 1] == '\n' ? 2 : 1;
    return s[i] == '\n' ? 1 : 0;
  };
  for (std::size_t i = 0; i < s.size();) {
    if (s[i] == '\\') {
      std::size_t j = i + 1;
      if (lenient_splice)
        while (j < s.size() && (s[j] == ' ' || s[j] == '\t'))
          ++j;
      if (const auto n = newline_at(j)) {
        i = j + n;
        ++line;
        continue;
      }
    }
    if (const auto n = newline_at(i)) {
      spliced.push_back({'\n', line++});
      i += n;
      continue;
    }
    spliced.push_back({s[i], line});
    ++i;
  }
  // Phase 3: comments become one space (a block comment may span newlines and
  // then joins the lines around it, exactly as in the C preprocessor).
  std::vector<CleanChar> out;
  out.reserve(spliced.size());
  for (std::size_t i = 0; i < spliced.size();) {
    const char c = spliced[i].c;
    const char next = i + 1 < spliced.size() ? spliced[i + 1].c : '\0';
    if (c == '/' && next == '/') {
      out.push_back({' ', spliced[i].line});
      while (i < spliced.size() && spliced[i].c != '\n')
        ++i;
      continue;
    }
    if (c == '/' && next == '*') {
      out.push_back({' ', spliced[i].line});
      i += 2;
      while (i < spliced.size() && !(spliced[i].c == '*' && i + 1 < spliced.size() && spliced[i + 1].c == '/'))
        ++i;
      i = std::min(spliced.size(), i + 2);
      continue;
    }
    if (literals && (c == '"' || c == '\'')) {
      out.push_back(spliced[i++]);
      while (i < spliced.size() && spliced[i].c != '\n') {
        const char d = spliced[i].c;
        out.push_back(spliced[i++]);
        if (d == '\\' && i < spliced.size() && spliced[i].c != '\n')
          out.push_back(spliced[i++]);
        else if (d == c)
          break;
      }
      continue;
    }
    out.push_back(spliced[i++]);
  }
  return out;
}

bool blank(char c) { return c == ' ' || c == '\t' || c == '\v' || c == '\f'; }
bool ident_char(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
}

// First directive in one clean view that reads another file (#include,
// #include_next, #import; kind 0) and first GLSL #extension that enables
// include support (kind 1). hits[kind] = {line or 0, spelling}.
using Hit = std::pair<std::uint32_t, std::string>;
void scan_view(const std::vector<CleanChar> &v, Hit hits[2]) {
  bool line_start = true;
  for (std::size_t i = 0; i < v.size(); ++i) {
    const char c = v[i].c;
    if (c == '\n') {
      line_start = true;
      continue;
    }
    if (blank(c))
      continue;
    if (c == '#' && line_start) {
      std::size_t j = i + 1;
      auto word = [&] {
        while (j < v.size() && blank(v[j].c))
          ++j;
        std::string w;
        while (j < v.size() && ident_char(v[j].c))
          w += v[j++].c;
        return w;
      };
      const auto name = word();
      if ((name == "include" || name == "include_next" || name == "import") && !hits[0].first)
        hits[0] = {v[i].line, "#" + name};
      if (name == "extension" && !hits[1].first) {
        const auto ext = word();
        if (ext == "GL_GOOGLE_include_directive" || ext == "GL_ARB_shading_language_include")
          hits[1] = {v[i].line, "#extension " + ext};
      }
    }
    line_start = false;
  }
}

std::string file_bytes(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream s;
  s << in.rdbuf();
  return s.str();
}

// Characters a Make-style dependency list never escapes; the private
// temporary directory is placed where every path consists only of these.
bool plain_path(const std::string &p) {
  return !p.empty() && std::all_of(p.begin(), p.end(), [](char c) {
    return ident_char(c) || c == '/' || c == '.' || c == '-' || c == '+';
  });
}

// Worker identity: SHA-256 of the executable (and of DXC's libdxcompiler,
// which the dxc driver loads), computed once per process.
std::string worker_identity(Language l) {
  static std::mutex lock;
  static std::map<Language, std::string> cache;
  std::lock_guard<std::mutex> guard(lock);
  if (auto it = cache.find(l); it != cache.end())
    return it->second;
  std::string id;
  if (l == Language::Glsl) {
    id = std::string(PARALYN_GLSLANG_TOOLCHAIN) + "; worker glslang sha256 " +
         source_sha256(file_bytes(PARALYN_GLSLANG_EXECUTABLE));
  } else {
    const std::string exe = PARALYN_DXC_EXECUTABLE;
    const auto bin = exe.substr(0, exe.find_last_of('/'));
    const auto lib = bin.substr(0, bin.find_last_of('/')) + "/lib/libdxcompiler.dylib";
    id = std::string(PARALYN_DXC_TOOLCHAIN) + "; worker dxc sha256 " +
         source_sha256(file_bytes(exe)) + ", libdxcompiler sha256 " +
         source_sha256(file_bytes(lib));
  }
  cache[l] = id;
  return id;
}

struct TempDir {
  std::string path;
  TempDir() {
    const char *base = std::getenv("TMPDIR");
    std::string pattern = std::string(base && *base ? base : "/tmp");
    if (pattern.back() != '/')
      pattern += '/';
    // The worker's dependency list is compared path by path, so the directory
    // must not need Make-style escaping (spaces, '$', '#', ':', ...).
    if (!plain_path(pattern))
      pattern = "/tmp/";
    pattern += "paralyn-shader-XXXXXX";
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    if (!mkdtemp(buffer.data()))
      throw std::runtime_error(std::string("cannot create a private temporary directory: ") +
                               std::strerror(errno));
    path = buffer.data();
  }
  ~TempDir() {
    for (const char *name : {"source", "out.spv", "deps.d"}) {
      const auto p = path + "/" + name;
      ::unlink(p.c_str());
    }
    ::rmdir(path.c_str());
  }
  TempDir(const TempDir &) = delete;
  TempDir &operator=(const TempDir &) = delete;
};

struct WorkerResult {
  int status = -1;
  bool timed_out = false, signaled = false;
  int signal = 0;
  std::string out, err;
};

// posix_spawn with an argument vector (no shell), an empty environment, stdin
// from /dev/null, bounded stdout/stderr capture and a wall-clock timeout.
WorkerResult run_worker(const std::vector<std::string> &arguments) {
  int out_pipe[2], err_pipe[2];
  if (pipe(out_pipe) != 0)
    throw std::runtime_error("pipe failed");
  if (pipe(err_pipe) != 0) {
    close(out_pipe[0]);
    close(out_pipe[1]);
    throw std::runtime_error("pipe failed");
  }
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0);
  posix_spawn_file_actions_adddup2(&actions, out_pipe[1], 1);
  posix_spawn_file_actions_adddup2(&actions, err_pipe[1], 2);
  for (int fd : {out_pipe[0], out_pipe[1], err_pipe[0], err_pipe[1]})
    posix_spawn_file_actions_addclose(&actions, fd);
  std::vector<char *> argv;
  for (const auto &a : arguments)
    argv.push_back(const_cast<char *>(a.c_str()));
  argv.push_back(nullptr);
  char *envp[] = {nullptr};
  pid_t pid = 0;
  const int spawned = posix_spawn(&pid, argv[0], &actions, nullptr, argv.data(), envp);
  posix_spawn_file_actions_destroy(&actions);
  close(out_pipe[1]);
  close(err_pipe[1]);
  WorkerResult result;
  if (spawned != 0) {
    close(out_pipe[0]);
    close(err_pipe[0]);
    throw std::runtime_error("cannot start " + arguments.front() + ": " + std::strerror(spawned));
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(worker_timeout_seconds);
  pollfd fds[2] = {{out_pipe[0], POLLIN, 0}, {err_pipe[0], POLLIN, 0}};
  std::string *sinks[2] = {&result.out, &result.err};
  int open_streams = 2;
  char chunk[8192];
  while (open_streams > 0) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                          deadline - std::chrono::steady_clock::now())
                          .count();
    if (left <= 0) {
      result.timed_out = true;
      kill(pid, SIGKILL);
      break;
    }
    const int ready = poll(fds, 2, static_cast<int>(std::min<long long>(left, 1000)));
    if (ready < 0 && errno != EINTR)
      break;
    for (int k = 0; k < 2; ++k) {
      if (fds[k].fd < 0 || !(fds[k].revents & (POLLIN | POLLHUP | POLLERR)))
        continue;
      const auto n = read(fds[k].fd, chunk, sizeof chunk);
      if (n <= 0) {
        close(fds[k].fd);
        fds[k].fd = -1;
        --open_streams;
      } else if (sinks[k]->size() < max_stream_bytes) {
        sinks[k]->append(chunk, std::min<std::size_t>(static_cast<std::size_t>(n),
                                                      max_stream_bytes - sinks[k]->size()));
      }
    }
  }
  for (auto &f : fds)
    if (f.fd >= 0)
      close(f.fd);
  int status = 0;
  while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
  if (WIFEXITED(status))
    result.status = WEXITSTATUS(status);
  else if (WIFSIGNALED(status)) {
    result.signaled = true;
    result.signal = WTERMSIG(status);
  }
  return result;
}

void replace_all(std::string &text, const std::string &from, const std::string &to) {
  if (from.empty())
    return;
  for (std::size_t at = 0; (at = text.find(from, at)) != std::string::npos; at += to.size())
    text.replace(at, from.size(), to);
}

std::uint32_t number(const std::string &s) {
  return s.empty() ? 0 : static_cast<std::uint32_t>(std::min<unsigned long>(std::stoul(s), 0xffffffffu));
}

// Parse compiler messages into source-located diagnostics. Lines that name
// the compiled source file are located; other error lines are kept unlocated.
std::vector<SourceDiagnostic> parse_messages(Language l, const std::string &text,
                                             const std::string &source_name) {
  std::vector<SourceDiagnostic> result;
  std::istringstream in(text);
  std::string line;
  // glslang: "ERROR: file:12:7: 'x' : undeclared identifier" (column with --error-column)
  static const std::regex glslang_line(R"(^(ERROR|WARNING): (.+?):(\d+):(?:(\d+):)? (.*)$)");
  static const std::regex glslang_plain(R"(^(ERROR|WARNING): (.*)$)");
  // DXC (clang style): "file:12:7: error: message"
  static const std::regex dxc_line(R"(^(.+?):(\d+):(\d+): (fatal error|error|warning): (.*)$)");
  static const std::regex dxc_plain(R"(^(?:.*?: )?(fatal error|error): (.*)$)");
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    std::smatch m;
    SourceDiagnostic d;
    if (l == Language::Glsl) {
      if (std::regex_match(line, m, glslang_line) && m[2].str() == source_name) {
        d.severity = m[1].str() == "ERROR" ? "error" : "warning";
        d.file = m[2].str();
        d.line = number(m[3].str());
        d.column = number(m[4].str());
        d.message = m[5].str();
      } else if (std::regex_match(line, m, glslang_plain)) {
        const auto message = m[2].str();
        // Summary lines ("1 compilation errors.  No code generated.") and the
        // generic terminator are not diagnostics of their own.
        if (message.find("compilation errors") != std::string::npos ||
            message.find("compilation terminated") != std::string::npos)
          continue;
        d.severity = m[1].str() == "ERROR" ? "error" : "warning";
        d.message = message;
      } else
        continue;
      if (d.message.find("compilation terminated") != std::string::npos)
        continue;
    } else {
      if (std::regex_match(line, m, dxc_line) && m[1].str() == source_name) {
        d.severity = m[4].str() == "warning" ? "warning" : "error";
        d.file = m[1].str();
        d.line = number(m[2].str());
        d.column = number(m[3].str());
        d.message = m[5].str();
      } else if (std::regex_match(line, m, dxc_plain)) {
        d.severity = "error";
        d.message = m[2].str();
      } else
        continue;
    }
    result.push_back(std::move(d));
  }
  return result;
}

std::string render(const std::vector<SourceDiagnostic> &diagnostics) {
  std::string s;
  for (const auto &d : diagnostics) {
    if (d.severity != "error")
      continue;
    if (!s.empty())
      s += "; ";
    if (!d.file.empty())
      s += d.file + ":" + std::to_string(d.line) + (d.column ? ":" + std::to_string(d.column) : "") + ": ";
    s += d.message;
  }
  return s;
}

void write_file(const std::string &path, const std::string &bytes) {
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  if (fd < 0)
    throw std::runtime_error("cannot create " + path + ": " + std::strerror(errno));
  std::size_t done = 0;
  while (done < bytes.size()) {
    const auto n = ::write(fd, bytes.data() + done, bytes.size() - done);
    if (n <= 0) {
      ::close(fd);
      throw std::runtime_error("cannot write " + path);
    }
    done += static_cast<std::size_t>(n);
  }
  ::close(fd);
}

std::string base_name(const std::string &path) {
  const auto slash = path.find_last_of('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}
} // namespace

CompileError::CompileError(std::string diagnostic, const std::string &detail,
                           std::vector<SourceDiagnostic> located)
    : std::runtime_error(detail), code(std::move(diagnostic)), diagnostics(std::move(located)) {}

const char *language_name(Language language) {
  return language == Language::Glsl ? "glsl" : "hlsl";
}

bool available(Language language) {
  return language == Language::Glsl ? PARALYN_HAS_GLSL != 0 : PARALYN_HAS_HLSL != 0;
}

std::string toolchain(Language language) {
  return available(language) ? (language == Language::Glsl ? PARALYN_GLSLANG_TOOLCHAIN
                                                           : PARALYN_DXC_TOOLCHAIN)
                             : "";
}

CompileResult compile(const std::string &source, const CompileOptions &options) {
  const auto l = options.language;
  if (!available(l))
    fail(l, "unavailable",
         std::string("this build does not include the ") + label(l) +
             " frontend; reconfigure with -DPARALYN_ENABLE_SPIRV=ON -DPARALYN_ENABLE_" + label(l) +
             "=ON (see docs/hlsl-glsl-frontends.md)");
  if (source.empty() || source.size() > max_source_bytes)
    fail(l, "input", "source must be nonempty and at most 1 MiB");
  if (source.find('\0') != std::string::npos || !utf8(source))
    fail(l, "input", "source must be UTF-8 text without NUL bytes");
  std::string directive;
  if (const auto line = include_directive_line(source, &directive))
    fail(l, "include",
         directive + " on line " + std::to_string(line) +
             ": #include is outside the profile: exactly the given source bytes are compiled and "
             "hashed, so included files cannot be resolved",
         {{"error", options.source_name, line, 0, directive + " is not supported"}});
  std::string entry = options.entry;
  if (l == Language::Hlsl) {
    if (entry.empty())
      fail(l, "entry", "HLSL compilation requires an entry function (--entry NAME)");
    static const std::regex compute_profile("cs_6_[0-8]");
    if (options.profile.empty())
      fail(l, "profile", "HLSL compilation requires a compute profile (--profile cs_6_0 .. cs_6_8)");
    if (!std::regex_match(options.profile, compute_profile))
      fail(l, "profile",
           "profile '" + options.profile +
               "' is not accepted; the frontend compiles compute shaders for cs_6_0 .. cs_6_8 only");
  } else if (!options.profile.empty()) {
    fail(l, "profile", "GLSL compute shaders take no --profile; the stage is always compute "
                       "and the target is Vulkan 1.1 / SPIR-V 1.3");
  }
  if (!entry.empty() && !identifier(entry))
    fail(l, "entry", "entry-point name '" + entry + "' must be an identifier of at most 64 characters");

  const std::string display = base_name(options.source_name.empty() ? "shader" : options.source_name);
  TempDir dir;
  const auto source_path = dir.path + "/source";
  const auto output_path = dir.path + "/out.spv";
  const auto deps_path = dir.path + "/deps.d";
  try {
    write_file(source_path, source);
  } catch (const std::exception &e) {
    fail(l, "worker", e.what());
  }
  std::vector<std::string> args;
  if (l == Language::Glsl) {
    // --depfile: glslang lists every file its preprocessor read (checked below).
    args = {PARALYN_GLSLANG_EXECUTABLE, "--target-env", "vulkan1.1", "-S", "comp", "--error-column",
            "--quiet", "--depfile", deps_path};
    if (!entry.empty()) {
      args.insert(args.end(), {"-e", entry, "--source-entrypoint", "main"});
    }
    args.insert(args.end(), {"-o", output_path, source_path});
  } else {
    args = {PARALYN_DXC_EXECUTABLE, "-spirv", "-fspv-target-env=vulkan1.1", "-T", options.profile,
            "-E", entry, "-HV", "2021", "-Fo", output_path, source_path};
  }
  WorkerResult worker;
  try {
    worker = run_worker(args);
  } catch (const std::exception &e) {
    fail(l, "worker", e.what());
  }
  CompileResult result;
  for (auto a : args) {
    replace_all(a, source_path, display);
    replace_all(a, output_path, "<output>.spv");
    replace_all(a, deps_path, "<output>.d");
    result.arguments.push_back(a);
  }
  std::string messages = worker.out + (worker.out.empty() || worker.err.empty() ? "" : "\n") + worker.err;
  replace_all(messages, source_path, display);
  replace_all(messages, output_path, "<output>.spv");
  auto diagnostics = parse_messages(l, messages, display);
  if (worker.timed_out)
    fail(l, "worker", "the compiler worker exceeded " + std::to_string(worker_timeout_seconds) +
                          " seconds and was terminated");
  if (worker.signaled)
    fail(l, "worker", "the compiler worker terminated with signal " + std::to_string(worker.signal));
  if (worker.status != 0) {
    const bool located = std::any_of(diagnostics.begin(), diagnostics.end(),
                                     [](const auto &d) { return d.severity == "error"; });
    std::string detail = located ? render(diagnostics) : messages;
    while (!detail.empty() && (detail.back() == '\n' || detail.back() == ' '))
      detail.pop_back();
    fail(l, "compile",
         std::string(l == Language::Glsl ? "glslang" : "DXC") + " rejected " + display +
             (detail.empty() ? std::string(" (exit status ") + std::to_string(worker.status) + ")"
                             : ": " + detail),
         diagnostics);
  }
  // Compiler-enforced inclusion check: the pinned compiler itself reports
  // every file its preprocessor read; anything but the source is rejected, so
  // no preprocessor spelling of #include can smuggle unhashed code in.
  std::string dependency_list;
  if (l == Language::Glsl) {
    dependency_list = file_bytes(deps_path);
  } else {
    auto dep_args = args; // identical options, -M: list dependencies, compile nothing
    dep_args.insert(dep_args.end() - 1, "-M");
    WorkerResult deps;
    try {
      deps = run_worker(dep_args);
    } catch (const std::exception &e) {
      fail(l, "worker", e.what());
    }
    if (deps.timed_out || deps.signaled || deps.status != 0)
      fail(l, "worker", "the DXC dependency listing (-M) failed after a successful compilation");
    dependency_list = deps.out;
  }
  bool well_formed = false;
  auto extra = unexpected_dependencies(dependency_list, output_path, source_path, &well_formed);
  if (!well_formed)
    fail(l, "worker", "could not verify which files the compiler read: its dependency list does not "
                      "name exactly the output and the source");
  if (!extra.empty()) {
    std::string names;
    for (auto &x : extra) {
      replace_all(x, source_path, display);
      names += (names.empty() ? "" : ", ") + x;
    }
    fail(l, "include",
         std::string(l == Language::Glsl ? "glslang" : "DXC") + " read " + std::to_string(extra.size()) +
             " file(s) besides " + display + " (" + names +
             "): #include is outside the profile: exactly the given source bytes are compiled and hashed",
         {{"error", options.source_name, 0, 0, "the compiler included another file"}});
  }
  struct stat info {};
  if (::stat(output_path.c_str(), &info) != 0 || info.st_size <= 0)
    fail(l, "worker", "the compiler worker reported success but wrote no SPIR-V");
  if (static_cast<std::uint64_t>(info.st_size) > executable_max_spirv_bytes)
    fail(l, "worker", "the compiler worker produced more than 4 MiB of SPIR-V");
  const auto bytes = file_bytes(output_path);
  result.compiler_spirv_sha256 = source_sha256(bytes);
  result.words = spirv::binary_words(bytes.data(), bytes.size());
  if (l == Language::Hlsl)
    result.words = normalize_storage_buffers(result.words, &result.normalization);
  for (auto &d : diagnostics)
    if (d.severity == "warning")
      result.warnings.push_back(d);
  result.toolchain = worker_identity(l) + "; files read: source only (checked with " +
                     (l == Language::Glsl ? "glslang --depfile" : "dxc -M") + ")";
  return result;
}

std::uint32_t include_directive_line(const std::string &source, std::string *directive) {
  Hit first[2] = {{0, ""}, {0, ""}};
  auto consider = [&](int kind, const Hit &hit) {
    if (hit.first && (!first[kind].first || hit.first < first[kind].first))
      first[kind] = hit;
  };
  // Every combination of splicing and literal handling the pinned
  // preprocessors (clang-based DXC, glslang) might apply; flag the union.
  for (bool lenient : {false, true})
    for (bool literals : {false, true}) {
      Hit hits[2] = {{0, ""}, {0, ""}};
      scan_view(clean_view(source, lenient, literals), hits);
      consider(0, hits[0]);
      consider(1, hits[1]);
    }
  // Also, conservatively, an #include at the start of any physical line, even
  // inside a comment.
  static const std::regex raw(R"(^[ \t]*#[ \t]*include\b)");
  std::istringstream in(source);
  std::string text;
  for (std::uint32_t n = 1; std::getline(in, text); ++n)
    if (std::regex_search(text, raw)) {
      consider(0, {n, "#include"});
      break;
    }
  // Report the file-reading directive itself when there is one, else the
  // #extension that would enable one.
  const auto &hit = first[0].first ? first[0] : first[1];
  if (directive)
    *directive = hit.second;
  return hit.first;
}

std::vector<std::string> unexpected_dependencies(const std::string &dependency_list, const std::string &target,
                                                 const std::string &source, bool *well_formed) {
  *well_formed = false;
  std::string text = dependency_list;
  replace_all(text, "\\\r\n", " ");
  replace_all(text, "\\\n", " ");
  std::vector<std::string> tokens;
  std::istringstream in(text);
  for (std::string t; in >> t;)
    tokens.push_back(t);
  std::vector<std::string> extra;
  if (tokens.empty() || target.empty() || source.empty())
    return extra;
  // "TARGET:" or "TARGET :" first.
  std::size_t k = 0;
  if (tokens[0] == target + ":")
    k = 1;
  else if (tokens.size() > 1 && tokens[0] == target && tokens[1] == ":")
    k = 2;
  else
    return extra;
  bool named = false;
  for (; k < tokens.size(); ++k) {
    if (tokens[k] == source)
      named = true;
    else
      extra.push_back(tokens[k]);
  }
  *well_formed = named;
  return extra;
}

std::vector<std::uint32_t> normalize_storage_buffers(const std::vector<std::uint32_t> &words,
                                                     std::string *summary) {
  // SPIR-V opcodes/enumerants used here (SPIR-V 1.3 unified specification).
  enum : std::uint32_t {
    OpName = 5, OpFunction = 54, OpFunctionParameter = 55, OpFunctionCall = 57,
    OpTypeStruct = 30, OpTypePointer = 32, OpVariable = 59, OpLoad = 61, OpStore = 62,
    OpCopyMemory = 63, OpCopyMemorySized = 64, OpAccessChain = 65, OpInBoundsAccessChain = 66,
    OpPtrAccessChain = 67, OpArrayLength = 68, OpInBoundsPtrAccessChain = 70, OpDecorate = 71,
    OpMemberDecorate = 72, OpCopyObject = 83, OpMemberName = 6, OpEntryPoint = 15,
    StorageUniform = 2, StorageBuffer = 12, DecorationBlock = 2, DecorationBufferBlock = 3
  };
  const auto hl = Language::Hlsl;
  if (words.size() < 5)
    fail(hl, "legalization", "SPIR-V module is truncated");
  struct Inst {
    std::size_t at;
    std::uint32_t op, count;
  };
  std::vector<Inst> list;
  for (std::size_t i = 5; i < words.size();) {
    const std::uint32_t count = words[i] >> 16, op = words[i] & 0xffff;
    if (!count || count > words.size() - i)
      fail(hl, "legalization", "SPIR-V instruction stream is truncated");
    list.push_back({i, op, count});
    i += count;
  }
  std::set<std::uint32_t> buffer_blocks;
  for (const auto &x : list)
    if (x.op == OpDecorate && x.count >= 3 && words[x.at + 2] == DecorationBufferBlock)
      buffer_blocks.insert(words[x.at + 1]);
  // Pointer types: id -> (storage class, pointee), plus declaration position.
  std::map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>> pointers;
  std::map<std::uint32_t, std::uint32_t> storage_buffer_pointer; // pointee -> existing SB pointer
  std::set<std::uint32_t> root_types;
  for (const auto &x : list)
    if (x.op == OpTypePointer && x.count == 4) {
      const auto id = words[x.at + 1], sc = words[x.at + 2], pointee = words[x.at + 3];
      pointers[id] = {sc, pointee};
      if (sc == StorageBuffer && !storage_buffer_pointer.count(pointee))
        storage_buffer_pointer[pointee] = id;
      if (sc == StorageUniform && buffer_blocks.count(pointee))
        root_types.insert(id);
    }
  // Converted variables and every pointer value derived from them.
  std::set<std::uint32_t> rooted;
  std::set<std::uint32_t> converted_structs;
  std::vector<std::string> converted_names;
  std::map<std::uint32_t, std::string> names;
  for (const auto &x : list)
    if (x.op == OpName && x.count >= 3) {
      const char *text = reinterpret_cast<const char *>(&words[x.at + 2]);
      names[words[x.at + 1]] = std::string(text, strnlen(text, (x.count - 2) * 4));
    }
  for (const auto &x : list)
    if (x.op == OpVariable && x.count >= 4 && root_types.count(words[x.at + 1])) {
      if (words[x.at + 3] != StorageUniform)
        fail(hl, "legalization", "a BufferBlock variable is not in the Uniform storage class");
      rooted.insert(words[x.at + 2]);
      converted_structs.insert(pointers[words[x.at + 1]].second);
      const auto n = names.find(words[x.at + 2]);
      converted_names.push_back(n == names.end() ? "%" + std::to_string(words[x.at + 2]) : n->second);
    }
  if (rooted.empty())
    return words;
  // Derived pointer result types that must move to the StorageBuffer class.
  std::map<std::uint32_t, std::uint32_t> retype; // instruction position -> new result type
  std::map<std::uint32_t, std::uint32_t> needed;  // pointee -> StorageBuffer pointer id (0 = create)
  std::size_t derived = 0;
  bool in_function = false;
  for (std::size_t k = 0; k < list.size(); ++k) {
    const auto &x = list[k];
    auto w = [&](std::uint32_t j) { return words[x.at + j]; };
    if (x.op == OpFunction)
      in_function = true;
    if (!in_function)
      continue;
    switch (x.op) {
    case OpAccessChain:
    case OpInBoundsAccessChain:
    case OpPtrAccessChain:
    case OpInBoundsPtrAccessChain:
    case OpCopyObject:
      if (x.count >= 4 && rooted.count(w(3))) {
        for (std::uint32_t j = 4; j < x.count; ++j)
          if (rooted.count(w(j)))
            fail(hl, "legalization", "a buffer pointer is used as an access-chain index");
        const auto type = pointers.find(w(1));
        if (type == pointers.end() || type->second.first != StorageUniform)
          fail(hl, "legalization", "a pointer derived from a BufferBlock variable is not a Uniform pointer");
        rooted.insert(w(2));
        needed.emplace(type->second.second, 0);
        retype[static_cast<std::uint32_t>(k)] = type->second.second; // pointee; resolved below
        ++derived;
        continue;
      }
      break;
    case OpLoad:
    case OpArrayLength:
      // Pointer operand only; the result is a value (or a length).
      if (x.count >= 4 && rooted.count(w(1)))
        fail(hl, "legalization", "unexpected pointer-typed load result");
      continue;
    case OpStore:
      if (x.count >= 3 && rooted.count(w(2)))
        fail(hl, "legalization", "a buffer pointer value is stored to memory");
      continue;
    case OpCopyMemory:
    case OpCopyMemorySized:
      continue;
    default:
      break;
    }
    // Fail closed: any other instruction that names a rooted pointer (a call
    // argument, OpPhi/OpSelect, an atomic, an extended instruction, ...)
    // would need its own types rewritten; leave that to an explicit extension.
    const bool atomic = (x.op >= 227 && x.op <= 242) || x.op == 318 || x.op == 319 ||
                        x.op == 5614 || x.op == 5615 || x.op == 6035;
    for (std::uint32_t j = 1; j < x.count; ++j)
      if (atomic && rooted.count(w(j)))
        // Same stable code and wording as the importer's own atomic rejection.
        throw spirv::ImportError("spirv.instruction",
                                 "atomic instructions are not qualified in this profile");
    for (std::uint32_t j = 1; j < x.count; ++j)
      if (rooted.count(w(j)) && !(x.op == OpVariable && j == 2))
        fail(hl, "legalization",
             "a pointer into a structured buffer is used by SPIR-V opcode " + std::to_string(x.op) +
                 ", which the StorageBuffer normalization does not rewrite");
  }
  (void)OpFunctionParameter;
  (void)OpFunctionCall;
  (void)OpMemberDecorate;
  (void)OpMemberName;
  (void)OpEntryPoint;
  (void)OpTypeStruct;
  std::uint32_t bound = words[3];
  // New StorageBuffer pointer types are declared right after the Uniform
  // pointer type they replace (its pointee is necessarily declared earlier).
  std::map<std::uint32_t, std::vector<std::uint32_t>> insert_after; // position -> new pointer ids
  std::map<std::uint32_t, std::uint32_t> pointee_of_new;
  for (auto &[pointee, id] : needed) {
    if (auto existing = storage_buffer_pointer.find(pointee); existing != storage_buffer_pointer.end()) {
      id = existing->second;
      continue;
    }
    id = bound++;
    pointee_of_new[id] = pointee;
    for (std::size_t k = 0; k < list.size(); ++k)
      if (list[k].op == OpTypePointer && list[k].count == 4 &&
          words[list[k].at + 2] == StorageUniform && words[list[k].at + 3] == pointee) {
        insert_after[static_cast<std::uint32_t>(k)].push_back(id);
        break;
      }
  }
  std::vector<std::uint32_t> out(words.begin(), words.begin() + 5);
  out[3] = bound;
  for (std::size_t k = 0; k < list.size(); ++k) {
    const auto &x = list[k];
    const auto start = out.size();
    out.insert(out.end(), words.begin() + static_cast<std::ptrdiff_t>(x.at),
               words.begin() + static_cast<std::ptrdiff_t>(x.at + x.count));
    if (x.op == OpDecorate && x.count >= 3 && out[start + 2] == DecorationBufferBlock &&
        converted_structs.count(out[start + 1]))
      out[start + 2] = DecorationBlock;
    if (x.op == OpTypePointer && x.count == 4 && root_types.count(out[start + 1]))
      out[start + 2] = StorageBuffer;
    if (x.op == OpVariable && x.count >= 4 && rooted.count(out[start + 2]))
      out[start + 3] = StorageBuffer;
    if (auto r = retype.find(static_cast<std::uint32_t>(k)); r != retype.end())
      out[start + 1] = needed.at(r->second);
    if (auto add = insert_after.find(static_cast<std::uint32_t>(k)); add != insert_after.end())
      for (auto id : add->second)
        out.insert(out.end(), {(4u << 16) | OpTypePointer, id, StorageBuffer, pointee_of_new.at(id)});
  }
  // A converted struct must not also back a Uniform (cbuffer) variable.
  for (const auto &x : list)
    if (x.op == OpVariable && x.count >= 4 && words[x.at + 3] == StorageUniform &&
        !rooted.count(words[x.at + 2])) {
      const auto type = pointers.find(words[x.at + 1]);
      if (type != pointers.end() && converted_structs.count(type->second.second))
        fail(hl, "legalization", "a BufferBlock struct type also backs a Uniform variable");
    }
  if (summary) {
    std::string list_names;
    for (const auto &n : converted_names)
      list_names += (list_names.empty() ? "" : ", ") + n;
    *summary = "DXC Vulkan 1.1 Uniform+BufferBlock structured buffers rewritten to StorageBuffer+Block: " +
               std::to_string(converted_names.size()) + " variable(s) (" + list_names + "), " +
               std::to_string(derived) + " derived pointer(s), " +
               std::to_string(pointee_of_new.size()) + " new pointer type(s)";
  }
  return out;
}

ExecutableModule import_source(const std::string &source, const CompileOptions &options,
                               const spirv::ImportOptions &import_options, CompileResult *out) {
  auto compiled = compile(source, options);
  auto module = spirv::import_module(compiled.words, import_options);
  std::string args;
  for (std::size_t i = 1; i < compiled.arguments.size(); ++i)
    args += (i > 1 ? " " : "") + compiled.arguments[i];
  module.toolchain += "; " + std::string(label(options.language)) + " frontend: " +
                      compiled.toolchain + "; source " + base_name(options.source_name) +
                      " sha256 " + source_sha256(source) + "; compiler SPIR-V sha256 " +
                      compiled.compiler_spirv_sha256 +
                      (compiled.normalization.empty() ? std::string()
                                                      : "; normalized: " + compiled.normalization) +
                      "; worker arguments: " + args;
  verify_executable(module);
  if (out)
    *out = std::move(compiled);
  return module;
}
} // namespace paralyn::shader
