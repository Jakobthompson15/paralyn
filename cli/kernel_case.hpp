#pragma once
// Declarative paralyn.toml projects and typed kernel cases (schema version 1).
// Parsing is strict: unknown fields, wrong TOML types, missing required values,
// inexact numeric conversions and ambiguous selections are rejected with stable
// diagnostic identifiers. Nothing here invents inputs, sizes, geometry or references.
#include "diagnostic.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace paralyn::cli {
enum class DType { F32, I32, U32 };
const char *dtype_name(DType type);

struct CaseScalar {
  std::string name;
  DType type = DType::U32;
  std::uint32_t bits = 0; // Exact 32-bit little-endian payload.
};
struct CaseBuffer {
  std::string name;
  DType type = DType::F32;
  std::uint64_t length = 0;
  std::vector<std::uint32_t> initial; // Exactly `length` declared words.
  std::string origin;                 // "values", "fill" or "file"
  std::filesystem::path file;         // Resolved relative to the case file.
  std::string sha256;                 // Declared and verified for files.
  bool output = false;
};
struct Tolerance {
  enum class Kind { Exact, AbsoluteRelative, Ulp } kind = Kind::Exact;
  double absolute = 0, relative = 0;
  std::uint64_t ulp = 0;
};
struct CaseCheck {
  std::string buffer;
  std::string reference_kind; // "file", "values" or "builtin"
  std::vector<std::uint32_t> expected; // file/values references
  std::filesystem::path file;
  std::string sha256;
  std::string builtin;
  std::map<std::string, std::string> roles; // builtin role -> case argument name
  std::map<std::string, std::uint64_t> constants; // builtin integer parameters
  Tolerance tolerance;
};
struct KernelCase {
  std::filesystem::path path;
  std::string sha256, name, entry, description;
  std::string bytes; // Exact parsed bytes; sha256 is their digest (evidence copies these).
  std::array<std::uint32_t, 3> grid{0, 0, 0}, block{0, 0, 0};
  std::vector<CaseScalar> scalars;
  std::vector<CaseBuffer> buffers;
  std::vector<CaseCheck> checks;
};
KernelCase load_case(const std::filesystem::path &path);

struct ProjectModule {
  std::string name;
  std::filesystem::path source, manifest; // manifest only for .metal sources
};
struct ProjectCase {
  std::string name, module;
  std::filesystem::path file;
};
struct ProjectProgram {
  std::string name;
  std::filesystem::path source;
  std::vector<std::string> arguments;
};
struct Project {
  std::filesystem::path path;
  std::string sha256, name, default_target;
  std::map<std::string, ProjectModule> modules;
  std::map<std::string, ProjectCase> cases;
  std::map<std::string, ProjectProgram> programs;
};
Project load_project(const std::filesystem::path &path);
// Resolves exactly one case or program. Both selectors, an unknown name, or no
// selector with several targets and no declared default are rejected.
struct ProjectSelection {
  std::optional<ProjectCase> kernel_case;
  std::optional<ProjectProgram> program;
  const ProjectModule *module = nullptr;
};
ProjectSelection select_target(const Project &project, const std::string &case_name,
                               const std::string &program_name);

// Host-side reference computation and comparison. Builtins are explicit CPU
// references computed from the declared case inputs, never from GPU results.
struct CheckResult {
  nlohmann::ordered_json record;
  bool passed = false;
  std::uint64_t compared = 0;
};
CheckResult evaluate_check(const KernelCase &kernel_case, const CaseCheck &check,
                           const std::vector<std::uint32_t> &actual);
std::string sha256_bytes(const void *data, std::size_t size);
std::string read_bounded(const std::filesystem::path &path, std::size_t limit,
                         const std::string &error_id);
} // namespace paralyn::cli
