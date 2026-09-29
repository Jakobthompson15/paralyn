#pragma once
// Platform naming and host-compiler command construction for the CLI.
// The Windows branches are UNBUILT AND UNTESTED (no Windows toolchain was
// available); they follow documented MSVC/clang-cl conventions by inspection.
#include <string>
#include <vector>

namespace paralyn::cli::platform {
#ifdef _WIN32
inline constexpr char path_list_separator = ';';
inline constexpr const char *native_library = "paralyn_native.dll";
// MSVC-style linkers consume the import library, never the DLL itself.
inline constexpr const char *native_link_library = "paralyn_native.lib";
inline constexpr const char *executable_suffix = ".exe";
inline std::string static_library(const std::string &base) { return base + ".lib"; }
#else
inline constexpr char path_list_separator = ':';
#ifdef __APPLE__
inline constexpr const char *native_library = "libparalyn_native.dylib";
#else
inline constexpr const char *native_library = "libparalyn_native.so";
#endif
inline constexpr const char *native_link_library = native_library;
inline constexpr const char *executable_suffix = "";
inline std::string static_library(const std::string &base) { return "lib" + base + ".a"; }
#endif

// Prepend one entry to a PATH-like list (PYTHONPATH, PATH) with the platform separator.
inline std::string prepend_path_list(const std::string &entry, const char *existing) {
  return existing && *existing ? entry + path_list_separator + existing : entry;
}

enum class CompilerStyle { gnu, msvc };
// variant: CMAKE_CXX_COMPILER_FRONTEND_VARIANT ("MSVC" for cl.exe and clang-cl).
inline CompilerStyle compiler_style(const std::string &variant, const std::string &id) {
  return variant == "MSVC" || (variant.empty() && id == "MSVC") ? CompilerStyle::msvc : CompilerStyle::gnu;
}

struct HostCompile {
  std::string compiler;
  bool c_language = false;
  CompilerStyle style = CompilerStyle::gnu;
  std::string include_dir, quote_dir, input, output, object_dir;
  std::vector<std::string> link_inputs;  // libraries/import libraries to link
  std::vector<std::string> runtime_dirs; // rpath entries (ELF/Mach-O only)
  std::vector<std::string> extra;        // platform frameworks/flags before the output
};
// Strict FP: no fast math and no contraction on either compiler family.
inline std::vector<std::string> host_compile_command(const HostCompile &h) {
  std::vector<std::string> command{h.compiler};
  if (h.style == CompilerStyle::msvc) {
    // /fp:precise without /fp:contract does not contract (VS 2022+); /Od keeps debug semantics.
    command.insert(command.end(), {"/nologo", h.c_language ? "/std:c11" : "/std:c++17", "/Od", "/Zi",
                                   "/fp:precise", "/EHsc", "/utf-8", "/I", h.include_dir, "/I",
                                   h.quote_dir, h.input});
    if (!h.object_dir.empty()) {
      command.push_back("/Fo" + h.object_dir + "\\");
      command.push_back("/Fd" + h.object_dir + "\\");
    }
    command.insert(command.end(), h.extra.begin(), h.extra.end());
    command.push_back("/Fe" + h.output);
    command.push_back("/link");
    command.insert(command.end(), h.link_inputs.begin(), h.link_inputs.end());
    return command;
  }
  command.insert(command.end(), {h.c_language ? "-std=c11" : "-std=c++17", "-O0", "-g",
                                 "-fno-fast-math", "-ffp-contract=off", "-I", h.include_dir,
                                 "-iquote", h.quote_dir, h.input});
  command.insert(command.end(), h.link_inputs.begin(), h.link_inputs.end());
#ifndef _WIN32
  for (const auto &dir : h.runtime_dirs) command.push_back("-Wl,-rpath," + dir);
#endif
  command.insert(command.end(), h.extra.begin(), h.extra.end());
  command.insert(command.end(), {"-o", h.output});
  return command;
}
} // namespace paralyn::cli::platform
