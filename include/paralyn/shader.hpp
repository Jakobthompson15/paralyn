#pragma once
// Optional GLSL and HLSL compute frontends (built only with
// -DPARALYN_ENABLE_GLSL=ON and/or -DPARALYN_ENABLE_HLSL=ON, which require
// -DPARALYN_ENABLE_SPIRV=ON).
//
// Flow: GLSL/HLSL source -> pinned compiler worker process (glslang
// vulkan-sdk-1.4.363.0 or DXC v1.9.2607, run as a separate process, never
// linked into Paralyn) -> SPIR-V 1.3 (Vulkan 1.1, GLCompute) -> the existing
// SPIR-V importer (profile scan, spirv-val, reflection, SPIRV-Cross MSL) ->
// PARALYNX container v2 -> Metal. There is no GLSL/HLSL -> CUDA/HIP path.
// See docs/hlsl-glsl-frontends.md.
#include "paralyn/executable.hpp"
#include "paralyn/spirv.hpp"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace paralyn::shader {
enum class Language { Glsl, Hlsl };
const char *language_name(Language language); // "glsl" | "hlsl"

// One compiler message, located in the caller's source file. line/column are
// 1-based; 0 means the compiler did not report one.
struct SourceDiagnostic {
  std::string severity; // "error" | "warning"
  std::string file;
  std::uint32_t line = 0, column = 0;
  std::string message;
};

// Stable codes (the `code` member):
//   <lang>.unavailable  frontend not built into this binary
//   <lang>.input        unreadable, empty, oversized, non-UTF-8 or NUL-containing source
//   <lang>.include      #include is outside the profile (the exact source bytes are compiled alone)
//   <lang>.entry        missing/invalid entry-point name
//   hlsl.profile        profile is not a compute profile cs_6_0 .. cs_6_8
//   <lang>.compile      the pinned compiler rejected the source (see diagnostics)
//   <lang>.worker       the compiler worker could not run, crashed, timed out or produced no SPIR-V
//   hlsl.legalization   DXC output uses buffer pointers in a form the StorageBuffer
//                       normalization cannot rewrite exactly (fails closed)
// what() is "ParalynError: <GLSL|HLSL> frontend [<code>]: <detail>".
struct CompileError : std::runtime_error {
  std::string code;
  std::vector<SourceDiagnostic> diagnostics;
  CompileError(std::string diagnostic, const std::string &detail,
               std::vector<SourceDiagnostic> located = {});
};

struct CompileOptions {
  Language language = Language::Glsl;
  // File name used in diagnostics and provenance (the display path of the source).
  std::string source_name = "shader.comp";
  // GLSL: optional SPIR-V/MSL entry-point name (the GLSL function is always
  // main). HLSL: required entry function name.
  std::string entry;
  // HLSL only: cs_6_0 .. cs_6_8.
  std::string profile;
};

struct CompileResult {
  std::vector<std::uint32_t> words;   // SPIR-V handed to the importer
  std::string toolchain;              // pinned compiler identity + worker executable SHA-256
  std::vector<std::string> arguments; // exact worker arguments (source path shown as source_name)
  std::vector<SourceDiagnostic> warnings;
  std::string compiler_spirv_sha256;  // SHA-256 of the SPIR-V exactly as the worker wrote it
  // Non-empty when Paralyn rewrote the worker's SPIR-V before import (HLSL:
  // DXC's Vulkan 1.1 Uniform+BufferBlock structured buffers become the
  // equivalent StorageBuffer+Block form the importer profile accepts).
  std::string normalization;
};

bool available(Language language);
// Pinned identity, e.g. "glslang vulkan-sdk-1.4.363.0 (e1b562a8...)". Empty when unavailable.
std::string toolchain(Language language);
// Compile exact source bytes with the pinned worker. Throws CompileError.
CompileResult compile(const std::string &source, const CompileOptions &options);
// compile() then spirv::import_module(); the container's toolchain field
// records the SPIR-V toolchain, the frontend worker and the source SHA-256.
// Throws CompileError or spirv::ImportError.
ExecutableModule import_source(const std::string &source, const CompileOptions &options,
                               const spirv::ImportOptions &import_options = {},
                               CompileResult *compiled = nullptr);
// HLSL normalization, exposed for tests: rewrites Uniform-class variables whose
// struct type is decorated BufferBlock (and every pointer derived from them)
// to the StorageBuffer class with Block, leaving Uniform+Block (cbuffer) data
// untouched. Throws CompileError(hlsl.legalization) on any pointer use it
// cannot retype exactly. Returns the input unchanged when nothing matches.
std::vector<std::uint32_t> normalize_storage_buffers(const std::vector<std::uint32_t> &words,
                                                     std::string *summary = nullptr);
} // namespace paralyn::shader
