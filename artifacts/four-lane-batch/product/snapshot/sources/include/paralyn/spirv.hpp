#pragma once
// Optional SPIR-V importer (built only with -DPARALYN_ENABLE_SPIRV=ON).
//
// Profile: SPIR-V 1.0-1.3 validated for Vulkan 1.1, one GLCompute entry point,
// Logical addressing with the GLSL450 memory model, a fixed literal LocalSize,
// StorageBuffer blocks holding one 32-bit f32/i32/u32 array, and 32-bit scalar
// members of Uniform/push-constant blocks. OpenCL Kernel-model SPIR-V is a
// distinct profile and is always rejected, never reinterpreted. The importer
// lowers through the pinned SPIRV-Cross MSL backend into a PARALYNX container
// (ExecutableFormat::SpirvMsl) that executes on the existing Metal engine. It
// provides no CUDA or HIP lowering. See docs/spirv-import.md.
#include "paralyn/executable.hpp"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace paralyn::spirv {
// Stable diagnostic codes (the `code` member), e.g. "spirv.kernel-model",
// "spirv.capability", "spirv.validation". what() is
// "ParalynError: SPIR-V import [<code>]: <detail>".
struct ImportError : std::runtime_error {
  std::string code;
  ImportError(std::string diagnostic, const std::string &detail);
};
struct ImportOptions {
  std::string source_name = "module.spv";
  std::string producer_version = "0.0.1";
};
// Pinned SPIRV-Tools / SPIRV-Cross / SPIRV-Headers identities.
std::string toolchain();
// Assemble SPIR-V text with the pinned SPIRV-Tools assembler (Vulkan 1.1
// target environment, so the header version is SPIR-V 1.3 unless the text
// requests otherwise). Throws spirv.assembly.
std::vector<std::uint32_t> assemble(const std::string &text);
// Copy a little-endian SPIR-V binary into words. Throws spirv.invalid-binary.
std::vector<std::uint32_t> binary_words(const void *data, std::size_t size);
// Profile-check, validate (spirv-val, Vulkan 1.1), reflect and lower to MSL.
ExecutableModule import_module(const std::vector<std::uint32_t> &words,
                               const ImportOptions &options = {});
} // namespace paralyn::spirv
