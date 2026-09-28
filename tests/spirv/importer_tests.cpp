// SPIR-V importer and PARALYNX v2 container tests. No GPU work is submitted.
// Usage: spirv_importer_tests EXAMPLES_DIR NEGATIVE_DIR ASSEMBLED_DIR
#include "paralyn/executable.hpp"
#include "paralyn/spirv.hpp"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using paralyn::ExecutableModule;
using paralyn::ResourceAccess;
using paralyn::ResourceOrigin;
using paralyn::ScalarType;
namespace spv = paralyn::spirv;

namespace {
unsigned checks = 0, rejections = 0;
void require(bool ok, const std::string &message) {
  if (!ok)
    throw std::runtime_error(message);
  ++checks;
}
std::string read(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  require(bool(in), "cannot read " + path.string());
  return {std::istreambuf_iterator<char>(in), {}};
}
std::string replace(std::string text, const std::string &from, const std::string &to) {
  const auto at = text.find(from);
  require(at != std::string::npos, "fixture mutation anchor missing: " + from);
  return text.replace(at, from.size(), to);
}
ExecutableModule import_text(const std::string &text, const std::string &name = "fixture.spvasm") {
  spv::ImportOptions options;
  options.source_name = name;
  options.producer_version = "test";
  return spv::import_module(spv::assemble(text), options);
}
void rejects(const std::string &code, const std::function<void()> &action, const std::string &label) {
  try {
    action();
  } catch (const spv::ImportError &e) {
    const std::string what = e.what();
    require(e.code == code, label + ": expected " + code + ", got " + e.code + " (" + what + ")");
    require(what.rfind("ParalynError: SPIR-V import [" + code + "]: ", 0) == 0,
            label + ": unstable diagnostic prefix: " + what);
    ++rejections;
    return;
  }
  throw std::runtime_error(label + ": unexpectedly accepted");
}
void container_rejects(const std::function<void()> &action, const std::string &needle,
                       const std::string &label) {
  try {
    action();
  } catch (const spv::ImportError &) {
    throw std::runtime_error(label + ": importer error where container error expected");
  } catch (const std::exception &e) {
    require(std::string(e.what()).find(needle) != std::string::npos,
            label + ": wrong container diagnostic: " + e.what());
    ++rejections;
    return;
  }
  throw std::runtime_error(label + ": container unexpectedly accepted");
}
const paralyn::ExecutableParameter &param(const ExecutableModule &m, std::size_t i) {
  require(m.entries.size() == 1 && i < m.entries[0].parameters.size(), "parameter index");
  return m.entries[0].parameters[i];
}
void expect_buffer(const ExecutableModule &m, std::size_t i, const char *name, ResourceAccess access,
                   std::uint32_t binding) {
  const auto &p = param(m, i);
  require(p.name == name && p.buffer && p.type == ScalarType::F32 && p.access == access &&
              p.binding == binding && p.source_binding == binding && p.source_set == 0 &&
              p.origin == ResourceOrigin::StorageBuffer && p.block_offset == 0 &&
              p.minimum_bytes == 4 && p.alignment == 4,
          std::string("storage buffer reflection mismatch at ") + name);
}
void expect_scalar(const ExecutableModule &m, std::size_t i, const char *name, ScalarType type,
                   ResourceOrigin origin, std::uint32_t slot, std::uint32_t offset) {
  const auto &p = param(m, i);
  require(p.name == name && !p.buffer && p.type == type && p.access == ResourceAccess::Read &&
              p.origin == origin && p.binding == slot && p.block_offset == offset &&
              p.minimum_bytes == 4,
          std::string("scalar reflection mismatch at ") + name);
}
std::vector<unsigned char> bytes_of(const std::vector<std::uint32_t> &words) {
  std::vector<unsigned char> out(words.size() * 4);
  for (std::size_t i = 0; i < words.size(); ++i)
    for (unsigned b = 0; b < 4; ++b)
      out[4 * i + b] = static_cast<unsigned char>(words[i] >> (8 * b));
  return out;
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 4, "usage: spirv_importer_tests EXAMPLES NEGATIVE ASSEMBLED");
    const fs::path examples = argv[1], negative = argv[2], assembled = argv[3];
    const auto add_text = read(examples / "vector_add.spvasm");
    const auto reduce_text = read(examples / "reduce_sum.spvasm");
    std::cout << "Toolchain: " << spv::toolchain() << "\n";
    require(spv::toolchain().find("SPIRV-Tools vulkan-sdk-1.4.363.0 (ef96ed763b43b59b33b31b362f09a02b729fa1c9)") !=
                std::string::npos, "toolchain identity is not the pinned release");

    // Positive reflection: vector_add.
    const auto add = import_text(add_text, "vector_add.spvasm");
    require(add.format == paralyn::ExecutableFormat::SpirvMsl && add.target == "metal-msl3.1" &&
                add.numerical_policy == 1 && add.producer == "paralyn-spirv-import" &&
                add.toolchain == spv::toolchain(), "vector_add module identity");
    require(add.entries.size() == 1 && add.entries[0].name == "vector_add", "vector_add entry name");
    require(add.entries[0].required_block == std::array<std::uint32_t, 3>{64, 1, 1}, "vector_add LocalSize");
    require(add.entries[0].builtins == paralyn::BuiltinGlobalInvocationId, "vector_add builtins");
    require(add.entries[0].parameters.size() == 4, "vector_add parameter count");
    expect_buffer(add, 0, "a", ResourceAccess::Read, 0);
    expect_buffer(add, 1, "b", ResourceAccess::Read, 1);
    expect_buffer(add, 2, "c", ResourceAccess::Write, 2);
    expect_scalar(add, 3, "n", ScalarType::U32, ResourceOrigin::PushConstantMember,
                  paralyn::spirv_push_constant_slot, 0);
    require(add.source.find("kernel void vector_add(") != std::string::npos &&
                add.source.find("[[buffer(30)]]") != std::string::npos,
            "generated MSL lacks the reflected entry/slots");

    // Positive reflection: reduce_sum (uniform block with two members).
    const auto reduce = import_text(reduce_text, "reduce_sum.spvasm");
    require(reduce.entries[0].name == "reduce_sum" &&
                reduce.entries[0].required_block == std::array<std::uint32_t, 3>{64, 1, 1},
            "reduce_sum entry");
    require(reduce.entries[0].builtins ==
                (paralyn::BuiltinGlobalInvocationId | paralyn::BuiltinLocalInvocationId |
                 paralyn::BuiltinWorkgroupId | paralyn::BuiltinNumWorkgroups |
                 paralyn::BuiltinWorkgroupSize),
            "reduce_sum builtins");
    require(reduce.entries[0].parameters.size() == 4, "reduce_sum parameter count");
    expect_buffer(reduce, 0, "input", ResourceAccess::Read, 0);
    expect_buffer(reduce, 1, "partial", ResourceAccess::Write, 1);
    expect_scalar(reduce, 2, "n", ScalarType::U32, ResourceOrigin::UniformMember, 2, 0);
    expect_scalar(reduce, 3, "scale", ScalarType::F32, ResourceOrigin::UniformMember, 2, 4);
    require(reduce.source.find("threadgroup_barrier") != std::string::npos,
            "reduction barrier missing from generated MSL");

    // Build-time spirv-as output equals in-process assembly; .spv import is identical.
    for (const char *name : {"vector_add", "reduce_sum"}) {
      const auto binary = read(assembled / (std::string(name) + ".spv"));
      const auto words = spv::binary_words(binary.data(), binary.size());
      const auto text = read(examples / (std::string(name) + ".spvasm"));
      require(words == spv::assemble(text), std::string(name) + ": build-time spirv-as differs");
      require(words[1] == 0x00010300, std::string(name) + ": fixture is not SPIR-V 1.3");
      spv::ImportOptions options;
      options.source_name = std::string(name) + ".spvasm";
      options.producer_version = "test";
      const auto from_binary = spv::import_module(words, options);
      const auto &from_text = std::string(name) == "vector_add" ? add : reduce;
      require(from_binary.source == from_text.source && from_binary.spirv == from_text.spirv &&
                  from_binary.spirv_sha256 == from_text.spirv_sha256,
              std::string(name) + ": .spv and .spvasm imports differ");
    }

    // PARALYNX v2 round trip and wire identity.
    for (const auto *m : {&add, &reduce}) {
      const auto wire = paralyn::serialize_executable(*m);
      require(std::memcmp(wire.data(), "PARALYNX", 8) == 0 && wire[8] == 2 && !wire[9] &&
                  !wire[10] && !wire[11] && wire[12] == 2,
              "container version/kind words");
      const auto back = paralyn::deserialize_executable(wire.data(), wire.size());
      require(paralyn::serialize_executable(back) == wire && back.spirv == m->spirv &&
                  back.entries[0].builtins == m->entries[0].builtins,
              "v2 round trip is not exact");
      for (std::size_t cut : {std::size_t(9), std::size_t(64), wire.size() / 2, wire.size() - 1})
        container_rejects([&] { paralyn::deserialize_executable(wire.data(), cut); }, "invalid executable",
                          "truncated v2 container");
      auto kind = wire;
      kind[12] = 1; // v2 header claiming MSL source payload
      container_rejects([&] { paralyn::deserialize_executable(kind.data(), kind.size()); },
                        "payload kind does not match container version", "v2/kind mismatch");
      auto version = wire;
      version[8] = 1; // v1 header with SPIR-V payload kind
      container_rejects([&] { paralyn::deserialize_executable(version.data(), version.size()); },
                        "payload kind does not match container version", "v1/kind mismatch");
    }

    // Descriptor tampering is rejected by the container verifier (runtime path).
    auto tampered = add;
    tampered.entries[0].parameters[2].access = ResourceAccess::Read;
    container_rejects([&] { paralyn::serialize_executable(tampered); },
                      "understates the retained SPIR-V's storage-buffer access", "understated access");
    tampered = add;
    tampered.spirv[40] ^= 1;
    container_rejects([&] { paralyn::serialize_executable(tampered); }, "SPIR-V SHA-256 mismatch",
                      "retained SPIR-V hash");
    tampered = add;
    tampered.entries[0].parameters[0].binding = 5; // slot no longer equals Vulkan binding
    container_rejects([&] { paralyn::serialize_executable(tampered); }, "unique Metal slot",
                      "slot/binding mismatch");
    tampered = add;
    tampered.entries[0].required_block = {0, 0, 0};
    container_rejects([&] { paralyn::serialize_executable(tampered); }, "fixed LocalSize",
                      "missing workgroup size");
    tampered = add;
    tampered.entries[0].parameters[3].binding = 2; // push constant onto a buffer slot
    container_rejects([&] { paralyn::serialize_executable(tampered); }, "reserved Metal slot",
                      "push-constant slot");
    tampered = add;
    tampered.format = paralyn::ExecutableFormat::MslSource;
    container_rejects([&] { paralyn::serialize_executable(tampered); }, "invalid executable",
                      "MSL source carrying SPIR-V fields");

    // Named negative fixtures.
    rejects("spirv.kernel-model", [&] { import_text(read(negative / "kernel_model.spvasm")); }, "kernel model");
    rejects("spirv.capability", [&] { import_text(read(negative / "bad_capability.spvasm")); }, "bad capability");
    rejects("spirv.validation", [&] { import_text(read(negative / "invalid_module.spvasm")); }, "invalid module");
    rejects("spirv.binding", [&] { import_text(read(negative / "binding_mismatch.spvasm")); }, "binding mismatch");

    // Further profile boundaries, each a single mutation of the valid vector_add fixture.
    rejects("spirv.binding", [&] {
      import_text(replace(add_text, "OpDecorate %b DescriptorSet 0", "OpDecorate %b DescriptorSet 1"));
    }, "descriptor set 1");
    rejects("spirv.binding", [&] {
      import_text(replace(add_text, "OpDecorate %c Binding 2", "OpDecorate %c Binding 30"));
    }, "binding beyond Metal slots");
    rejects("spirv.access", [&] {
      import_text(replace(add_text, "               OpStore %c_ptr %sum", "               OpStore %a_ptr %sum"));
    }, "store to NonWritable");
    rejects("spirv.specialization", [&] {
      import_text(replace(add_text, "%int_0 = OpConstant %int 0", "%int_0 = OpSpecConstant %int 0"));
    }, "specialization constant");
    rejects("spirv.workgroup-size", [&] {
      import_text(replace(add_text, "               OpExecutionMode %main LocalSize 64 1 1\n", ""));
    }, "missing LocalSize");
    rejects("spirv.capability", [&] {
      import_text(replace(add_text, "               OpCapability Shader\n",
                          "               OpCapability Shader\n               OpCapability Int64\n"));
    }, "Int64 capability");
    rejects("spirv.extension", [&] {
      import_text(replace(add_text, "               OpMemoryModel",
                          "               OpExtension \"SPV_KHR_variable_pointers\"\n               OpMemoryModel"));
    }, "extension");
    rejects("spirv.resource", [&] {
      import_text(replace(add_text, "               OpName %a \"a\"\n", ""));
    }, "unnamed storage buffer");
    rejects("spirv.resource", [&] {
      auto text = replace(add_text, "               OpDecorate %c Binding 2\n",
                          "               OpDecorate %c Binding 2\n               OpDecorate %d DescriptorSet 0\n"
                          "               OpDecorate %d Binding 3\n");
      text = replace(text, "               OpName %c \"c\"\n",
                     "               OpName %c \"c\"\n               OpName %d \"d\"\n");
      import_text(replace(text, "          %c = OpVariable %ptr_out_sb StorageBuffer\n",
                          "          %c = OpVariable %ptr_out_sb StorageBuffer\n"
                          "          %d = OpVariable %ptr_out_sb StorageBuffer\n"));
    }, "declared but unused storage buffer");
    rejects("spirv.execution-model", [&] {
      import_text(replace(add_text, "OpEntryPoint GLCompute %main", "OpEntryPoint Fragment %main"));
    }, "fragment entry point");
    rejects("spirv.builtin", [&] {
      import_text(replace(add_text, "OpDecorate %gid BuiltIn GlobalInvocationId",
                          "OpDecorate %gid BuiltIn SubgroupLocalInvocationId"));
    }, "unsupported builtin");
    rejects("spirv.assembly", [&] { spv::assemble("OpCapability Shader\nOpNotAnInstruction\n"); },
            "malformed assembly");

    // Binary framing.
    auto words = spv::assemble(add_text);
    rejects("spirv.version", [&] {
      auto w = words;
      w[1] = 0x00010400;
      spv::import_module(w);
    }, "SPIR-V 1.4 header");
    rejects("spirv.invalid-binary", [&] {
      auto b = bytes_of(words);
      spv::binary_words(b.data(), b.size() - 2);
    }, "misaligned binary");
    rejects("spirv.invalid-binary", [&] {
      auto b = bytes_of(words);
      std::swap(b[0], b[3]);
      std::swap(b[1], b[2]);
      spv::binary_words(b.data(), b.size());
    }, "big-endian binary");
    rejects("spirv.invalid-binary", [&] {
      auto w = words;
      w.resize(w.size() - 1);
      w.back() = (5u << 16) | 62u; // OpStore claiming words past the end
      spv::import_module(w);
    }, "truncated instruction");

    std::cout << "SPIR-V importer: " << checks << " checks, " << rejections
              << " stable rejections, no GPU work submitted\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "spirv_importer_tests failed: " << e.what() << "\n";
    return 1;
  }
}
