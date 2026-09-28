// Windows process runner. UNBUILT AND UNTESTED: no Windows toolchain or machine
// was available when this was written; it is kept correct by inspection only
// and must be compiled and exercised (tests/process_tests.cpp equivalent) on
// Windows before any Windows support claim. See docs/cuda-backend.md.
#include "process.hpp"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <atomic>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
#include <windows.h>
namespace paralyn::cli {
namespace {
std::wstring wide(const std::string &s) {
  if (s.empty())
    return {};
  int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()),
                              nullptr, 0);
  if (!n)
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
std::runtime_error failure(const char *what) {
  return std::runtime_error(std::string(what) + " (Windows error " + std::to_string(GetLastError()) + ")");
}
struct Handle {
  HANDLE h = nullptr;
  Handle() = default;
  explicit Handle(HANDLE value) : h(value) {}
  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;
  ~Handle() { reset(); }
  void reset() {
    if (h && h != INVALID_HANDLE_VALUE)
      CloseHandle(h);
    h = nullptr;
  }
};
// Environment names are case-insensitive on Windows ("Path" and "PATH" are one
// variable) and the block must be sorted case-insensitively.
struct NameLess {
  bool operator()(const std::wstring &a, const std::wstring &b) const {
    return CompareStringOrdinal(a.c_str(), static_cast<int>(a.size()), b.c_str(),
                                static_cast<int>(b.size()), TRUE) == CSTR_LESS_THAN;
  }
};
std::atomic<DWORD> active{0};
BOOL WINAPI interrupted(DWORD event) {
  if (event != CTRL_C_EVENT && event != CTRL_BREAK_EVENT)
    return FALSE;
  if (const DWORD pid = active.load())
    GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid);
  return TRUE;
}
struct AttributeList {
  std::vector<unsigned char> storage;
  LPPROC_THREAD_ATTRIBUTE_LIST list = nullptr;
  explicit AttributeList(DWORD count) {
    SIZE_T size = 0;
    InitializeProcThreadAttributeList(nullptr, count, 0, &size);
    storage.resize(size);
    list = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
    if (!InitializeProcThreadAttributeList(list, count, 0, &size))
      throw failure("cannot initialize process attributes");
  }
  ~AttributeList() { DeleteProcThreadAttributeList(list); }
};
// Unwinding always leaves no running child, no reader thread and no installed
// console handler, mirroring the POSIX ChildGuard.
struct ChildGuard {
  Handle job, process, thread;
  std::vector<std::thread> readers;
  bool handler = false, reaped = false;
  ~ChildGuard() {
    if (process.h && !reaped) {
      if (!job.h || !TerminateJobObject(job.h, 1))
        TerminateProcess(process.h, 1);
      WaitForSingleObject(process.h, INFINITE);
    }
    // Termination closes the child's pipe ends, so readers reach end-of-file.
    for (auto &reader : readers)
      if (reader.joinable())
        reader.join();
    active.store(0);
    if (handler)
      SetConsoleCtrlHandler(interrupted, FALSE);
  }
};
} // namespace
unsigned long process_id() { return GetCurrentProcessId(); }
Process execute(const std::vector<std::string> &args, bool live,
                const std::vector<std::pair<std::string, std::string>> &env) {
  if (args.empty())
    throw std::runtime_error("empty process command");
  SECURITY_ATTRIBUTES security{sizeof(security), nullptr, TRUE};
  Handle out_read, out_write, err_read, err_write, input;
  if (!CreatePipe(&out_read.h, &out_write.h, &security, 0) ||
      !CreatePipe(&err_read.h, &err_write.h, &security, 0))
    throw failure("cannot create child output pipes");
  if (!SetHandleInformation(out_read.h, HANDLE_FLAG_INHERIT, 0) ||
      !SetHandleInformation(err_read.h, HANDLE_FLAG_INHERIT, 0))
    throw failure("cannot restrict pipe inheritance");
  // Give the child an explicitly inheritable duplicate of our stdin, or NUL.
  const HANDLE parent_input = GetStdHandle(STD_INPUT_HANDLE);
  if (!parent_input || parent_input == INVALID_HANDLE_VALUE ||
      !DuplicateHandle(GetCurrentProcess(), parent_input, GetCurrentProcess(), &input.h, 0, TRUE,
                       DUPLICATE_SAME_ACCESS)) {
    input.h = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &security,
                          OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (input.h == INVALID_HANDLE_VALUE) {
      input.h = nullptr;
      throw failure("cannot open NUL for child input");
    }
  }
  std::map<std::wstring, std::wstring, NameLess> vars;
  {
    std::unique_ptr<wchar_t, decltype(&FreeEnvironmentStringsW)> inherited(GetEnvironmentStringsW(),
                                                                         &FreeEnvironmentStringsW);
    if (!inherited)
      throw failure("cannot read process environment");
    for (auto p = inherited.get(); *p; p += wcslen(p) + 1) {
      std::wstring entry = p;
      auto i = entry.find(L'=', entry[0] == L'=' ? 1 : 0);
      if (i != std::wstring::npos)
        vars[entry.substr(0, i)] = entry.substr(i + 1);
    }
  }
  for (const auto &item : env) {
    const auto name = wide(item.first);
    vars.erase(name); // replace any differently cased spelling
    vars[name] = wide(item.second);
  }
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
  if (command.size() >= 32767)
    throw std::runtime_error("command line exceeds the Windows 32767-character limit");
  // Inherit exactly the three standard handles, never unrelated inheritable handles.
  HANDLE inherit[] = {input.h, out_write.h, err_write.h};
  AttributeList attributes(1);
  if (!UpdateProcThreadAttribute(attributes.list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit,
                                 sizeof(inherit), nullptr, nullptr))
    throw failure("cannot restrict inherited handles");
  STARTUPINFOEXW startup{};
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = input.h;
  startup.StartupInfo.hStdOutput = out_write.h;
  startup.StartupInfo.hStdError = err_write.h;
  startup.lpAttributeList = attributes.list;
  // Everything the reader threads touch (output strings, error slots) is
  // declared BEFORE the guard: locals are destroyed in reverse order, so on any
  // exception ~ChildGuard terminates the child and joins the readers while
  // these targets are still alive. (The pipe read handles above outlive the
  // guard for the same reason.)
  Process result;
  DWORD read_errors[2]{};
  ChildGuard guard;
  // A job lets an unwinding caller terminate the child's whole process tree.
  guard.job.h = CreateJobObjectW(nullptr, nullptr);
  PROCESS_INFORMATION process{};
  if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE,
                      CREATE_NEW_PROCESS_GROUP | CREATE_UNICODE_ENVIRONMENT | CREATE_SUSPENDED |
                          EXTENDED_STARTUPINFO_PRESENT,
                      environment.data(), nullptr, &startup.StartupInfo, &process))
    throw failure("cannot start program");
  guard.process.h = process.hProcess;
  guard.thread.h = process.hThread;
  if (guard.job.h && !AssignProcessToJobObject(guard.job.h, process.hProcess))
    guard.job.reset(); // e.g. restricted nesting: fall back to TerminateProcess on unwind
  // The child owns its copies; our write ends must close for readers to see EOF.
  out_write.reset();
  err_write.reset();
  input.reset();
  active.store(process.dwProcessId);
  guard.handler = SetConsoleCtrlHandler(interrupted, TRUE) != FALSE;
  auto reader = [&](HANDLE handle, std::string &target, std::ostream &output, unsigned index) {
    try {
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
    } catch (...) {
      read_errors[index] = ERROR_NOT_ENOUGH_MEMORY; // never let an exception escape a thread
    }
  };
  guard.readers.reserve(2);
  guard.readers.emplace_back(reader, out_read.h, std::ref(result.out), std::ref(std::cout), 0);
  guard.readers.emplace_back(reader, err_read.h, std::ref(result.err), std::ref(std::cerr), 1);
  if (ResumeThread(process.hThread) == static_cast<DWORD>(-1))
    throw failure("cannot resume program");
  if (WaitForSingleObject(guard.process.h, INFINITE) != WAIT_OBJECT_0)
    throw failure("cannot wait for program");
  guard.reaped = true;
  for (auto &thread : guard.readers)
    thread.join();
  DWORD code = 0;
  if (!GetExitCodeProcess(guard.process.h, &code))
    throw failure("cannot read program exit status");
  if (read_errors[0] || read_errors[1])
    throw std::runtime_error("cannot read program output");
  result.status = static_cast<int>(code);
  result.interrupted = code == 0xC000013A; // STATUS_CONTROL_C_EXIT
  return result;
}
} // namespace paralyn::cli
