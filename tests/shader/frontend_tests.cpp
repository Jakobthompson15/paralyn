// CPU-only tests of the GLSL/HLSL frontends: pinned worker compilation,
// SPIR-V import of every shipped example, the DXC StorageBuffer normalization
// (positive and fail-closed), source-located diagnostics and stable codes.
// No GPU work is submitted here; tests/shader/metal_tests.cpp executes.
// Usage: shader_frontend_tests EXAMPLES_DIR NEGATIVE_DIR
#include "paralyn/shader.hpp"
#include <algorithm>
#include <array>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sh = paralyn::shader;
namespace {
unsigned checks = 0, rejections = 0;
void require(bool ok, const std::string &message) {
  if (!ok)
    throw std::runtime_error(message);
  ++checks;
}
std::string read(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot read " + path);
  std::ostringstream s;
  s << in.rdbuf();
  return s.str();
}
sh::CompileOptions glsl(const std::string &name, const std::string &entry = "") {
  sh::CompileOptions o;
  o.language = sh::Language::Glsl;
  o.source_name = name;
  o.entry = entry;
  return o;
}
sh::CompileOptions hlsl(const std::string &name, const std::string &entry, const std::string &profile) {
  sh::CompileOptions o;
  o.language = sh::Language::Hlsl;
  o.source_name = name;
  o.entry = entry;
  o.profile = profile;
  return o;
}
// Expect a CompileError (or, with code "spirv.*", an ImportError) with this code.
void rejects(const std::string &code, const std::function<void()> &f, const std::string &label,
             std::uint32_t line = 0, std::uint32_t column = 0) {
  try {
    f();
  } catch (const sh::CompileError &e) {
    require(e.code == code, label + ": expected " + code + ", got " + e.code + " (" + e.what() + ")");
    require(std::string(e.what()).find("[" + code + "]") != std::string::npos, label + ": what() lacks code");
    if (line) {
      const auto hit = std::find_if(e.diagnostics.begin(), e.diagnostics.end(), [&](const auto &d) {
        return d.severity == "error" && d.line == line && (!column || d.column == column) && !d.file.empty();
      });
      require(hit != e.diagnostics.end(), label + ": no diagnostic located at line " + std::to_string(line) +
                                              (column ? ":" + std::to_string(column) : ""));
    }
    ++rejections;
    return;
  } catch (const paralyn::spirv::ImportError &e) {
    require(e.code == code, label + ": expected " + code + ", got " + e.code + " (" + e.what() + ")");
    ++rejections;
    return;
  }
  throw std::runtime_error(label + ": was accepted, expected " + code);
}

struct Expect {
  std::string file, entry, profile, msl_entry;
  std::array<std::uint32_t, 3> block;
  std::vector<std::string> names;
  std::vector<paralyn::ResourceAccess> access;
};

// A hand-written Vulkan 1.1 module in DXC's legacy form: structured buffers as
// Uniform + BufferBlock, plus a cbuffer (Uniform + Block) that must stay put.
const char *legacy_add = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "legacy_add" %gid
               OpExecutionMode %main LocalSize 64 1 1
               OpName %a "a"
               OpName %c "c"
               OpName %cb "cb"
               OpMemberName %CB 0 "n"
               OpDecorate %gid BuiltIn GlobalInvocationId
               OpDecorate %floats ArrayStride 4
               OpDecorate %SB BufferBlock
               OpMemberDecorate %SB 0 Offset 0
               OpDecorate %CB Block
               OpMemberDecorate %CB 0 Offset 0
               OpDecorate %a DescriptorSet 0
               OpDecorate %a Binding 0
               OpDecorate %c DescriptorSet 0
               OpDecorate %c Binding 1
               OpDecorate %cb DescriptorSet 0
               OpDecorate %cb Binding 2
       %void = OpTypeVoid
       %func = OpTypeFunction %void
       %bool = OpTypeBool
      %float = OpTypeFloat 32
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
     %v3uint = OpTypeVector %uint 3
     %floats = OpTypeRuntimeArray %float
         %SB = OpTypeStruct %floats
         %CB = OpTypeStruct %uint
     %ptr_sb = OpTypePointer Uniform %SB
     %ptr_cb = OpTypePointer Uniform %CB
     %ptr_uf = OpTypePointer Uniform %float
     %ptr_uu = OpTypePointer Uniform %uint
      %ptr_i = OpTypePointer Input %v3uint
      %int_0 = OpConstant %int 0
          %a = OpVariable %ptr_sb Uniform
          %c = OpVariable %ptr_sb Uniform
         %cb = OpVariable %ptr_cb Uniform
        %gid = OpVariable %ptr_i Input
       %main = OpFunction %void None %func
      %entry = OpLabel
         %id = OpLoad %v3uint %gid
          %i = OpCompositeExtract %uint %id 0
         %pn = OpAccessChain %ptr_uu %cb %int_0
          %n = OpLoad %uint %pn
         %ok = OpULessThan %bool %i %n
               OpSelectionMerge %done None
               OpBranchConditional %ok %body %done
       %body = OpLabel
         %pa = OpAccessChain %ptr_uf %a %int_0 %i
          %x = OpLoad %float %pa
          %y = OpFAdd %float %x %x
         %pc = OpAccessChain %ptr_uf %c %int_0 %i
               OpStore %pc %y
               OpBranch %done
       %done = OpLabel
               OpReturn
               OpFunctionEnd
)";
std::string with(const std::string &text, const std::string &from, const std::string &to) {
  auto s = text;
  const auto at = s.find(from);
  if (at == std::string::npos)
    throw std::runtime_error("fixture edit anchor missing: " + from);
  s.replace(at, from.size(), to);
  return s;
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("Usage: shader_frontend_tests EXAMPLES_DIR NEGATIVE_DIR");
    const std::string examples = argv[1], negative = argv[2];
    require(sh::available(sh::Language::Glsl) && sh::available(sh::Language::Hlsl),
            "both frontends must be built for this test");
    require(sh::toolchain(sh::Language::Glsl).find("e1b562a8bed273a02f30b59b66a5d499793cede5") != std::string::npos,
            "glslang identity");
    require(sh::toolchain(sh::Language::Hlsl).find("0d3ee6b551b8fa768fbf825300ebab81047ef6a8") != std::string::npos,
            "DXC identity");

    using A = paralyn::ResourceAccess;
    const std::vector<Expect> positive{
        {"glsl/vector_add.comp", "vector_add", "", "vector_add", {64, 1, 1}, {"a", "b", "c", "n"},
         {A::Read, A::Read, A::Write, A::Read}},
        {"glsl/blur_rows.comp", "blur_rows", "", "blur_rows", {16, 16, 1},
         {"src", "weights", "dst", "width", "height", "radius"},
         {A::Read, A::Read, A::Write, A::Read, A::Read, A::Read}},
        {"glsl/blur_columns.comp", "blur_columns", "", "blur_columns", {16, 16, 1},
         {"src", "weights", "dst", "width", "height", "radius"},
         {A::Read, A::Read, A::Write, A::Read, A::Read, A::Read}},
        {"glsl/vector_add.comp", "", "", "main0", {64, 1, 1}, {"a", "b", "c", "n"},
         {A::Read, A::Read, A::Write, A::Read}},
        {"hlsl/vector_add.hlsl", "vector_add", "cs_6_0", "vector_add", {64, 1, 1}, {"a", "b", "c", "n"},
         {A::Read, A::Read, A::Write, A::Read}},
        {"hlsl/vector_add.hlsl", "vector_add", "cs_6_6", "vector_add", {64, 1, 1}, {"a", "b", "c", "n"},
         {A::Read, A::Read, A::Write, A::Read}},
        {"hlsl/vector_add.hlsl", "vector_add", "cs_6_8", "vector_add", {64, 1, 1}, {"a", "b", "c", "n"},
         {A::Read, A::Read, A::Write, A::Read}},
        {"hlsl/transpose.hlsl", "transpose_tiled", "cs_6_0", "transpose_tiled", {16, 16, 1},
         {"src", "dst", "width", "height"}, {A::Read, A::Write, A::Read, A::Read}},
        {"hlsl/reduce_sum.hlsl", "reduce_sum", "cs_6_0", "reduce_sum", {256, 1, 1},
         {"input", "partial", "n", "scale"}, {A::Read, A::Write, A::Read, A::Read}},
    };
    for (const auto &x : positive) {
      const bool is_hlsl = !x.profile.empty();
      const auto path = examples + "/" + x.file;
      const auto source = read(path);
      const auto name = x.file.substr(x.file.find('/') + 1);
      const auto options = is_hlsl ? hlsl(name, x.entry, x.profile) : glsl(name, x.entry);
      const auto label = x.file + " [" + x.entry + (is_hlsl ? " " + x.profile : "") + "]";
      sh::CompileResult compiled;
      const auto module = sh::import_source(source, options, {}, &compiled);
      require(module.format == paralyn::ExecutableFormat::SpirvMsl && module.entries.size() == 1,
              label + ": not a single-entry SPIR-V-derived module");
      const auto &e = module.entries.front();
      require(e.name == x.msl_entry, label + ": MSL entry " + e.name + " != " + x.msl_entry);
      require(e.required_block == x.block, label + ": workgroup");
      require(e.parameters.size() == x.names.size(), label + ": parameter count");
      for (std::size_t i = 0; i < x.names.size(); ++i)
        require(e.parameters[i].name == x.names[i] && e.parameters[i].access == x.access[i],
                label + ": parameter " + x.names[i]);
      require(module.toolchain.find("SPIRV-Tools vulkan-sdk-1.4.363.0") != std::string::npos &&
                  module.toolchain.find(is_hlsl ? "HLSL frontend: DXC v1.9.2607" : "GLSL frontend: glslang vulkan-sdk-1.4.363.0") !=
                      std::string::npos &&
                  module.toolchain.find("source " + name + " sha256 " + paralyn::source_sha256(source)) !=
                      std::string::npos &&
                  module.toolchain.find("worker " + std::string(is_hlsl ? "dxc" : "glslang") + " sha256 ") !=
                      std::string::npos,
              label + ": container toolchain provenance: " + module.toolchain);
      require(compiled.compiler_spirv_sha256.size() == 64, label + ": compiler SPIR-V hash");
      require(is_hlsl == !compiled.normalization.empty(),
              label + ": normalization expected only for DXC output");
      require((module.spirv_sha256 == compiled.compiler_spirv_sha256) == !is_hlsl,
              label + ": retained SPIR-V must be the worker's bytes exactly unless normalized");
      // The normalization is idempotent: normalized words contain no BufferBlock.
      require(sh::normalize_storage_buffers(compiled.words) == compiled.words, label + ": normalization idempotence");
      // Reproducible worker output for identical input.
      const auto again = sh::compile(source, options);
      require(again.words == compiled.words && again.compiler_spirv_sha256 == compiled.compiler_spirv_sha256,
              label + ": worker output is not reproducible");
      require(compiled.arguments.back() == name, label + ": worker arguments must show the source name");
    }

    // DXC-legacy form: rejected by the importer as-is, accepted after normalization,
    // with the cbuffer left in the Uniform class.
    {
      const auto words = paralyn::spirv::assemble(legacy_add);
      rejects("spirv.resource", [&] { (void)paralyn::spirv::import_module(words); }, "legacy BufferBlock import");
      std::string summary;
      const auto normalized = sh::normalize_storage_buffers(words, &summary);
      require(normalized != words && summary.find("2 variable(s) (a, c)") != std::string::npos &&
                  summary.find("2 derived pointer(s)") != std::string::npos &&
                  summary.find("1 new pointer type(s)") != std::string::npos,
              "normalization summary: " + summary);
      const auto module = paralyn::spirv::import_module(normalized);
      const auto &p = module.entries.front().parameters;
      require(p.size() == 3 && p[0].name == "a" && p[0].access == A::Read &&
                  p[0].origin == paralyn::ResourceOrigin::StorageBuffer && p[1].name == "c" &&
                  p[1].access == A::Write && p[2].name == "n" &&
                  p[2].origin == paralyn::ResourceOrigin::UniformMember,
              "normalized legacy module descriptor");
      // Fail closed on pointer uses the rewrite does not retype.
      const auto call = with(with(legacy_add, "       %main = OpFunction %void None %func",
                                  "     %helpty = OpTypeFunction %void %ptr_sb\n"
                                  "     %helper = OpFunction %void None %helpty\n"
                                  "      %param = OpFunctionParameter %ptr_sb\n"
                                  "     %hentry = OpLabel\n"
                                  "               OpReturn\n"
                                  "               OpFunctionEnd\n"
                                  "       %main = OpFunction %void None %func"),
                             "         %pa = OpAccessChain", "      %callr = OpFunctionCall %void %helper %a\n"
                                                           "         %pa = OpAccessChain");
      rejects("hlsl.legalization",
              [&] { (void)sh::normalize_storage_buffers(paralyn::spirv::assemble(call)); },
              "buffer pointer passed to a function");
      const auto select = with(legacy_add, "         %pa = OpAccessChain %ptr_uf %a %int_0 %i",
                               "        %sel = OpSelect %ptr_sb %ok %a %c\n"
                               "         %pa = OpAccessChain %ptr_uf %sel %int_0 %i");
      rejects("hlsl.legalization",
              [&] { (void)sh::normalize_storage_buffers(paralyn::spirv::assemble(select)); },
              "OpSelect of buffer pointers");
      const auto atomic = with(with(legacy_add, "%ptr_uf = OpTypePointer Uniform %float",
                                    "%ptr_uf = OpTypePointer Uniform %float\n"
                                    "     %uint_1 = OpConstant %uint 1\n"
                                    "     %uint_0 = OpConstant %uint 0"),
                               "          %x = OpLoad %float %pa",
                               "          %x = OpLoad %float %pa\n"
                               "         %pu = OpAccessChain %ptr_uf %c %int_0 %i\n"
                               "         %at = OpAtomicLoad %float %pu %uint_1 %uint_0");
      rejects("spirv.instruction",
              [&] { (void)sh::normalize_storage_buffers(paralyn::spirv::assemble(atomic)); },
              "atomic on a structured buffer");
    }

    // Source-located compiler diagnostics and stable codes.
    rejects("glsl.compile", [&] { (void)sh::compile(read(negative + "/compile_error.comp"), glsl("compile_error.comp")); },
            "GLSL undeclared identifier", 9, 23);
    rejects("hlsl.compile",
            [&] { (void)sh::compile(read(negative + "/compile_error.hlsl"), hlsl("compile_error.hlsl", "main_cs", "cs_6_0")); },
            "HLSL undeclared identifier", 7, 24);
    rejects("hlsl.compile",
            [&] { (void)sh::compile(read(negative + "/vertex.hlsl"), hlsl("vertex.hlsl", "vs_main", "cs_6_0")); },
            "vertex entry with a compute profile", 3, 8);
    rejects("glsl.include", [&] { (void)sh::compile(read(negative + "/include.comp"), glsl("include.comp")); },
            "GLSL #include", 4);
    rejects("hlsl.include",
            [&] { (void)sh::compile(read(negative + "/include.hlsl"), hlsl("include.hlsl", "main_cs", "cs_6_0")); },
            "HLSL #include", 2);
    const auto add_hlsl = read(examples + "/hlsl/vector_add.hlsl");
    for (const char *profile : {"ps_6_0", "vs_6_0", "cs_5_0", "cs_5_1", "cs_6_9", "cs_7_0", "CS_6_0", "lib_6_3", ""})
      rejects("hlsl.profile", [&] { (void)sh::compile(add_hlsl, hlsl("vector_add.hlsl", "vector_add", profile)); },
              std::string("profile '") + profile + "'");
    rejects("hlsl.entry", [&] { (void)sh::compile(add_hlsl, hlsl("vector_add.hlsl", "", "cs_6_0")); }, "HLSL entry required");
    rejects("hlsl.entry", [&] { (void)sh::compile(add_hlsl, hlsl("vector_add.hlsl", "9lives", "cs_6_0")); }, "HLSL entry identifier");
    rejects("hlsl.compile", [&] { (void)sh::compile(add_hlsl, hlsl("vector_add.hlsl", "missing", "cs_6_0")); }, "HLSL missing entry");
    const auto add_glsl = read(examples + "/glsl/vector_add.comp");
    rejects("glsl.profile", [&] {
      auto o = glsl("vector_add.comp");
      o.profile = "cs_6_0";
      (void)sh::compile(add_glsl, o);
    }, "GLSL profile");
    rejects("glsl.entry", [&] { (void)sh::compile(add_glsl, glsl("vector_add.comp", "bad-name")); }, "GLSL entry identifier");
    rejects("glsl.input", [&] { (void)sh::compile("", glsl("empty.comp")); }, "empty source");
    rejects("glsl.input", [&] { (void)sh::compile(std::string("#version 450\n\0", 14), glsl("nul.comp")); }, "NUL byte");
    rejects("glsl.input", [&] { (void)sh::compile("#version 450\n// \xff\xfe\n", glsl("bad.comp")); }, "invalid UTF-8");
    rejects("glsl.input", [&] { (void)sh::compile(std::string((1u << 20) + 1, ' '), glsl("big.comp")); }, "oversized source");
    rejects("hlsl.input", [&] { (void)sh::compile("\xc0\xaf", hlsl("overlong.hlsl", "main", "cs_6_0")); }, "overlong UTF-8");

    // Unsupported features are rejected by the unchanged SPIR-V profile.
    struct Unsupported {
      std::string file, code;
      bool is_hlsl;
    };
    for (const auto &u : std::vector<Unsupported>{{"image.comp", "spirv.resource", false},
                                                   {"float64.comp", "spirv.capability", false},
                                                   {"subgroup.comp", "spirv.capability", false},
                                                   {"atomic.comp", "spirv.instruction", false},
                                                   {"texture.hlsl", "spirv.resource", true},
                                                   {"float64.hlsl", "spirv.capability", true},
                                                   {"int64.hlsl", "spirv.capability", true},
                                                   {"wave.hlsl", "spirv.capability", true},
                                                   {"atomic.hlsl", "spirv.instruction", true}}) {
      const auto source = read(negative + "/" + u.file);
      rejects(u.code, [&] {
        (void)sh::import_source(source, u.is_hlsl ? hlsl(u.file, "main_cs", "cs_6_0") : glsl(u.file));
      }, u.file);
    }
    std::cout << "GLSL/HLSL frontends: " << positive.size() << " example compilations imported, "
              << rejections << " stable rejections, " << checks << " checks\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "shader_frontend_tests failed: " << e.what() << "\n";
    return 1;
  }
}
