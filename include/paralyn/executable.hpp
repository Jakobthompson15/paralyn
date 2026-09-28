#pragma once
#include "paralyn/ir.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace paralyn {
// Trusted-local native shaders. This container validates the resource contract;
// Metal compilation/reflection validates the program and its actual signature.
// It is not a shader sandbox, nor a portable translation of arbitrary MSL.
enum class ExecutableFormat : std::uint32_t { MslSource = 1 };
enum class ResourceAccess : std::uint32_t { Read = 1, Write = 2, ReadWrite = 3 };
struct ExecutableParameter {
  std::string name;
  ScalarType type = ScalarType::F32;
  bool buffer = true;
  ResourceAccess access = ResourceAccess::Read;
  std::uint32_t binding = 0;
  std::uint32_t alignment = 4;
  std::uint64_t minimum_bytes = 4;
};
struct ExecutableEntry {
  std::string name;
  std::vector<ExecutableParameter> parameters;
  // All zero: any device-valid block. Otherwise the launch must match exactly.
  std::array<std::uint32_t, 3> required_block{0, 0, 0};
};
struct ExecutableModule {
  ExecutableFormat format = ExecutableFormat::MslSource;
  std::string target = "metal-msl3.1";
  // Safe math, precise functions, contraction off. Explicit MSL operations
  // retain their source semantics; this does not promise CUDA equivalence.
  std::uint32_t numerical_policy = 1;
  std::string producer;
  std::string producer_version;
  std::string source_name;
  std::string source_sha256;
  std::string source;
  std::vector<ExecutableEntry> entries;
};
inline constexpr std::size_t executable_max_bytes = 16 * 1024 * 1024;
inline constexpr std::size_t executable_max_source_bytes = 8 * 1024 * 1024;
std::string source_sha256(const std::string &source);
void verify_executable(const ExecutableModule &module);
std::vector<unsigned char> serialize_executable(const ExecutableModule &module);
ExecutableModule deserialize_executable(const void *data, std::size_t size);
bool is_executable_artifact(const void *data, std::size_t size) noexcept;
} // namespace paralyn
