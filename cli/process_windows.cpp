#include "process.hpp"
#define NOMINMAX
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <map>
#include <stdexcept>
#include <thread>
#include <windows.h>
namespace paralyn::cli {
namespace {
std::wstring wide(const std::string &s) {
  int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()),
                              nullptr, 0);
  if (!n && !s.empty())
    throw std::runtime_error("invalid UTF-8 process argument");
  std::wstring w(n, L'\0');
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()), w.data(),
                      n);
  return w;
}
std::wstring quote(const std::string &s) {
  std::wstring out = L"\"";
  unsigned slashes = 0;
  for (wchar_t c : wide(s)) {
    if (c == L'\\') {
      ++slashes;
      continue;
    }
    out.append(c == L'"' ? 2 * slashes + 1 : slashes, L'\\');
    slashes = 0;
    out += c;
  }
  out.append(2 * slashes, L'\\');
  return out + L'"';
}
struct Handle {
  HANDLE h = nullptr;
  ~Handle() {
    if (h && h != INVALID_HANDLE_VALUE)
      CloseHandle(h);
  }
};
volatile DWORD active = 0;
BOOL WINAPI interrupted(DWORD event) {
  if (event != CTRL_C_EVENT && event != CTRL_BREAK_EVENT)
    return FALSE;
  if (active)
    GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, active);
  return TRUE;
}
} // namespace
unsigned long process_id() { return GetCurrentProcessId(); }
Process execute(const std::vector<std::string> &args, bool live,
                const std::vector<std::pair<std::string, std::string>> &env) {
  if (args.empty())
    throw std::runtime_error("empty process command");
  SECURITY_ATTRIBUTES security{sizeof(security), nullptr, TRUE};
  Handle out_read, out_write, err_read, err_write;
  if (!CreatePipe(&out_read.h, &out_write.h, &security, 0) ||
      !CreatePipe(&err_read.h, &err_write.h, &security, 0))
    throw std::runtime_error("cannot create child output pipes");
  SetHandleInformation(out_read.h, HANDLE_FLAG_INHERIT, 0);
  SetHandleInformation(err_read.h, HANDLE_FLAG_INHERIT, 0);
  std::map<std::wstring, std::wstring> vars;
  auto inherited = GetEnvironmentStringsW();
  if (!inherited)
    throw std::runtime_error("cannot read process environment");
  for (auto p = inherited; *p; p += wcslen(p) + 1) {
    std::wstring entry = p;
    auto i = entry.find(L'=', entry[0] == L'=' ? 1 : 0);
    if (i != std::wstring::npos)
      vars[entry.substr(0, i)] = entry.substr(i + 1);
  }
  FreeEnvironmentStringsW(inherited);
  for (const auto &item : env)
    vars[wide(item.first)] = wide(item.second);
  std::wstring environment;
  for (const auto &item : vars) {
    environment += item.first + L"=" + item.second;
    environment += L'\0';
  }
  environment += L'\0';
  std::wstring command;
  for (const auto &arg : args) {
    if (!command.empty())
      command += L' ';
    command += quote(arg);
  }
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESTDHANDLES;
  startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  startup.hStdOutput = out_write.h;
  startup.hStdError = err_write.h;
  PROCESS_INFORMATION process{};
  if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE,
                      CREATE_NEW_PROCESS_GROUP | CREATE_UNICODE_ENVIRONMENT, environment.data(),
                      nullptr, &startup, &process))
    throw std::runtime_error("cannot start program (Windows error " +
                             std::to_string(GetLastError()) + ")");
  Handle process_handle{process.hProcess}, thread_handle{process.hThread};
  CloseHandle(out_write.h);
  out_write.h = nullptr;
  CloseHandle(err_write.h);
  err_write.h = nullptr;
  active = process.dwProcessId;
  SetConsoleCtrlHandler(interrupted, TRUE);
  Process result;
  DWORD read_errors[2]{};
  auto reader = [&](HANDLE handle, std::string &target, std::ostream &output, unsigned index) {
    char buffer[8192];
    DWORD n = 0;
    while (ReadFile(handle, buffer, sizeof(buffer), &n, nullptr) && n) {
      target.append(buffer, n);
      if (live) {
        output.write(buffer, n);
        output.flush();
      }
    }
    auto e = GetLastError();
    if (e != ERROR_BROKEN_PIPE && e != ERROR_SUCCESS)
      read_errors[index] = e;
  };
  std::thread a(reader, out_read.h, std::ref(result.out), std::ref(std::cout), 0);
  std::thread b(reader, err_read.h, std::ref(result.err), std::ref(std::cerr), 1);
  WaitForSingleObject(process_handle.h, INFINITE);
  a.join();
  b.join();
  active = 0;
  SetConsoleCtrlHandler(interrupted, FALSE);
  DWORD code = 0;
  if (!GetExitCodeProcess(process_handle.h, &code))
    throw std::runtime_error("cannot read program exit status");
  if (read_errors[0] || read_errors[1])
    throw std::runtime_error("cannot read program output");
  result.status = static_cast<int>(code);
  result.interrupted = code == 0xC000013A;
  return result;
}
// Unqualified on Windows: output is persisted after the child exits (streamed =
// false), with the same exclusive-creation and byte limits as the POSIX runner.
Process execute(const std::vector<std::string> &args, bool live,
                const std::vector<std::pair<std::string, std::string>> &env,
                const Capture &capture) {
  const std::string *paths[2] = {&capture.stdout_path, &capture.stderr_path};
  for (auto *path : paths)
    if (!path->empty()) {
      auto *file = std::fopen(path->c_str(), "wbx");
      if (!file)
        throw std::runtime_error("cannot create application capture " + *path);
      std::fclose(file);
    }
  auto result = execute(args, live, env);
  std::string *streams[2] = {&result.out, &result.err};
  for (unsigned i = 0; i < 2; ++i) {
    auto &text = *streams[i];
    (i ? result.err_bytes : result.out_bytes) = text.size();
    if (!paths[i]->empty()) {
      const auto stored = std::min<std::uint64_t>(text.size(), capture.file_limit);
      auto *file = std::fopen(paths[i]->c_str(), "wb");
      bool okay = file && std::fwrite(text.data(), 1, static_cast<std::size_t>(stored), file) == stored;
      if (file && std::fclose(file))
        okay = false;
      if (!okay)
        throw std::runtime_error("cannot write application capture " + *paths[i]);
      (i ? result.err_persisted : result.out_persisted) = stored;
      (i ? result.err_file_truncated : result.out_file_truncated) = stored < text.size();
    }
    if (text.size() > capture.memory_limit) {
      text.resize(capture.memory_limit);
      (i ? result.err_memory_truncated : result.out_memory_truncated) = true;
    }
  }
  return result;
}
} // namespace paralyn::cli
