// SPIR-V 1.3 / Vulkan 1.1 GLCompute importer. The pinned SPIRV-Tools validator
// decides validity; this file adds an explicit, narrower Paralyn profile and
// reflects the module into the PARALYNX executable descriptor. Lowering uses
// the pinned SPIRV-Cross MSL backend; execution uses the existing Metal engine.
#define SPV_ENABLE_UTILITY_CODE
#include "paralyn/spirv.hpp"
#include "paralyn_spirv_toolchain.h"
#include <spirv_msl.hpp>
#include <spirv-tools/libspirv.hpp>
#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <sstream>

namespace paralyn::spirv {
namespace {
namespace sc = spirv_cross;
constexpr std::uint32_t magic = 0x07230203;
constexpr std::uint32_t max_version = 0x00010300; // SPIR-V 1.3 (Vulkan 1.1)

[[noreturn]] void fail(const char *code, const std::string &detail) { throw ImportError(code, detail); }

std::string literal(const std::vector<std::uint32_t> &w, std::size_t begin, std::size_t end) {
  std::string s;
  for (std::size_t i = begin; i < end; ++i)
    for (unsigned b = 0; b < 32; b += 8) {
      const char c = static_cast<char>((w[i] >> b) & 0xff);
      if (!c)
        return s;
      s += c;
    }
  fail("spirv.invalid-binary", "unterminated literal string");
}
std::string capability_name(std::uint32_t value) {
  std::string name = spv::CapabilityToString(static_cast<spv::Capability>(value));
  return (name == "Unknown" ? std::string("capability") : name) + " (" + std::to_string(value) + ")";
}
std::string version_text(std::uint32_t v) {
  return std::to_string((v >> 16) & 0xff) + "." + std::to_string((v >> 8) & 0xff);
}
struct Scan {
  std::array<std::uint32_t, 3> local_size{0, 0, 0};
  bool workgroup_size_constant = false;
};

// Structural profile checks that must be decided before (and independently of)
// the Vulkan validator. In particular, OpenCL Kernel-model modules receive one
// stable diagnostic instead of whatever a Vulkan-environment validator reports.
Scan scan(const std::vector<std::uint32_t> &w) {
  if (w.size() < 5 || w[0] != magic)
    fail("spirv.invalid-binary", "missing SPIR-V header or little-endian magic number");
  if (w[4] != 0)
    fail("spirv.invalid-binary", "nonzero reserved schema word");
  std::vector<std::string> kernel_reasons;
  std::vector<std::uint32_t> capabilities;
  std::vector<std::string> other;
  std::size_t entries = 0;
  bool memory_model = false;
  std::set<std::uint32_t> gl_entries;
  Scan result;
  bool local_size = false;
  std::map<std::uint32_t, std::uint32_t> builtin_targets; // id -> builtin
  std::map<std::uint32_t, std::vector<std::uint32_t>> constant_composites; // id -> constituents
  std::map<std::uint32_t, std::uint32_t> scalar_constants;                   // 32-bit OpConstant
  for (std::size_t i = 5; i < w.size();) {
    const auto count = w[i] >> 16, op = w[i] & 0xffff;
    if (!count || count > w.size() - i)
      fail("spirv.invalid-binary", "truncated or zero-length instruction at word " + std::to_string(i));
    const auto end = i + count;
    auto need = [&](std::uint32_t n) {
      if (count < n)
        fail("spirv.invalid-binary", "instruction " + std::to_string(op) + " is too short");
    };
    switch (op) {
    case 17: // OpCapability
      need(2);
      if (w[i + 1] == static_cast<std::uint32_t>(spv::CapabilityKernel))
        kernel_reasons.push_back("OpCapability Kernel");
      else if (w[i + 1] != static_cast<std::uint32_t>(spv::CapabilityShader) &&
               w[i + 1] != static_cast<std::uint32_t>(spv::CapabilityMatrix))
        capabilities.push_back(w[i + 1]);
      break;
    case 10: { // OpExtension
      need(2);
      const auto name = literal(w, i + 1, end);
      if (name != "SPV_KHR_storage_buffer_storage_class")
        other.push_back("spirv.extension|unsupported extension " + name);
      break;
    }
    case 11: { // OpExtInstImport
      need(3);
      const auto name = literal(w, i + 2, end);
      if (name == "OpenCL.std")
        kernel_reasons.push_back("OpExtInstImport OpenCL.std");
      else if (name != "GLSL.std.450")
        other.push_back("spirv.extension|unsupported extended instruction set " + name);
      break;
    }
    case 14: // OpMemoryModel
      need(3);
      memory_model = true;
      if (w[i + 2] == static_cast<std::uint32_t>(spv::MemoryModelOpenCL))
        kernel_reasons.push_back("OpenCL memory model");
      if (w[i + 1] != static_cast<std::uint32_t>(spv::AddressingModelLogical))
        other.push_back("spirv.addressing|only Logical addressing is supported (found addressing model " +
                        std::to_string(w[i + 1]) + ")");
      else if (w[i + 2] != static_cast<std::uint32_t>(spv::MemoryModelGLSL450) &&
               w[i + 2] != static_cast<std::uint32_t>(spv::MemoryModelOpenCL))
        other.push_back("spirv.addressing|only the GLSL450 memory model is supported");
      break;
    case 15: // OpEntryPoint
      need(4);
      ++entries;
      if (w[i + 1] == static_cast<std::uint32_t>(spv::ExecutionModelKernel))
        kernel_reasons.push_back("OpEntryPoint Kernel");
      else if (w[i + 1] != static_cast<std::uint32_t>(spv::ExecutionModelGLCompute))
        other.push_back(std::string("spirv.execution-model|only GLCompute entry points are supported (found ") +
                        spv::ExecutionModelToString(static_cast<spv::ExecutionModel>(w[i + 1])) + ")");
      else
        gl_entries.insert(w[i + 2]);
      break;
    case 16: // OpExecutionMode
      need(3);
      if (w[i + 2] == static_cast<std::uint32_t>(spv::ExecutionModeLocalSize)) {
        need(6);
        local_size = true;
        result.local_size = {w[i + 3], w[i + 4], w[i + 5]};
      } else if (w[i + 2] == static_cast<std::uint32_t>(spv::ExecutionModeLocalSizeId))
        other.push_back("spirv.workgroup-size|LocalSizeId is outside the fixed-LocalSize profile");
      else
        other.push_back(std::string("spirv.execution-mode|unsupported execution mode ") +
                        spv::ExecutionModeToString(static_cast<spv::ExecutionMode>(w[i + 2])));
      break;
    case 331: // OpExecutionModeId
      other.push_back("spirv.workgroup-size|OpExecutionModeId is outside the fixed-LocalSize profile");
      break;
    case 48: case 49: case 50: case 51: case 52: // OpSpecConstant*
      other.push_back("spirv.specialization|specialization constants are not supported");
      break;
    case 68: // OpArrayLength
      other.push_back("spirv.instruction|OpArrayLength needs a buffer-size side channel outside this profile");
      break;
    case 39: // OpTypeForwardPointer
      other.push_back("spirv.addressing|forward pointers are outside the Logical profile");
      break;
    case 43: // OpConstant
      need(4);
      if (count == 4)
        scalar_constants[w[i + 2]] = w[i + 3];
      break;
    case 44: // OpConstantComposite
      need(3);
      constant_composites[w[i + 2]].assign(w.begin() + i + 3, w.begin() + end);
      break;
    case 71: // OpDecorate
      need(3);
      if (w[i + 2] == static_cast<std::uint32_t>(spv::DecorationBuiltIn)) {
        need(4);
        builtin_targets[w[i + 1]] = w[i + 3];
      }
      break;
    case 72: // OpMemberDecorate
      need(4);
      if (w[i + 3] == static_cast<std::uint32_t>(spv::DecorationBuiltIn))
        other.push_back("spirv.builtin|block-member builtins are outside the compute profile");
      break;
    default:
      if ((op >= 227 && op <= 242) || op == 318 || op == 319)
        other.push_back("spirv.instruction|atomic instructions are not qualified in this profile");
      break;
    }
    i = end;
  }
  if (!kernel_reasons.empty()) {
    std::string why;
    for (const auto &r : kernel_reasons)
      why += (why.empty() ? "" : ", ") + r;
    fail("spirv.kernel-model",
         "OpenCL Kernel-execution-model SPIR-V (" + why +
             ") is a distinct profile; Paralyn never reinterprets it as Vulkan GLCompute");
  }
  if ((w[1] & 0xff0000ffu) || w[1] < 0x00010000 || w[1] > max_version)
    fail("spirv.version", "SPIR-V " + version_text(w[1]) +
                              " is outside the supported SPIR-V 1.0-1.3 (Vulkan 1.1) profile");
  if (!capabilities.empty()) {
    std::string list;
    for (auto c : capabilities)
      list += (list.empty() ? "" : ", ") + capability_name(c);
    fail("spirv.capability", "unsupported capability " + list + "; the profile permits only Shader");
  }
  if (!other.empty()) {
    const auto bar = other.front().find('|');
    throw ImportError(other.front().substr(0, bar), other.front().substr(bar + 1));
  }
  if (!memory_model)
    fail("spirv.invalid-binary", "missing OpMemoryModel");
  if (entries != 1 || gl_entries.size() != 1)
    fail("spirv.entry-point", "the profile requires exactly one GLCompute entry point (found " +
                                  std::to_string(entries) + ")");
  if (!local_size)
    fail("spirv.workgroup-size", "the entry point must declare a fixed LocalSize execution mode");
  static const std::set<std::uint32_t> allowed{
      spv::BuiltInGlobalInvocationId, spv::BuiltInLocalInvocationId,
      spv::BuiltInLocalInvocationIndex, spv::BuiltInWorkgroupId, spv::BuiltInNumWorkgroups,
      spv::BuiltInWorkgroupSize};
  for (const auto &[id, builtin] : builtin_targets) {
    if (!allowed.count(builtin))
      fail("spirv.builtin", std::string("unsupported builtin ") +
                                spv::BuiltInToString(static_cast<spv::BuiltIn>(builtin)));
    if (builtin == spv::BuiltInWorkgroupSize) {
      const auto composite = constant_composites.find(id);
      if (composite == constant_composites.end())
        fail("spirv.workgroup-size", "WorkgroupSize must decorate a non-specialization constant");
      // The WorkgroupSize constant takes precedence over LocalSize in SPIR-V;
      // the profile requires both to agree so the enforced launch block is the
      // workgroup the shader computes with.
      std::array<std::uint32_t, 3> size{0, 0, 0};
      if (composite->second.size() != 3)
        fail("spirv.workgroup-size", "WorkgroupSize must be a three-component constant");
      for (std::size_t k = 0; k < 3; ++k) {
        const auto value = scalar_constants.find(composite->second[k]);
        if (value == scalar_constants.end())
          fail("spirv.workgroup-size", "WorkgroupSize components must be literal 32-bit OpConstant values");
        size[k] = value->second;
      }
      if (size != result.local_size)
        fail("spirv.workgroup-size",
             "WorkgroupSize constant (" + std::to_string(size[0]) + ", " + std::to_string(size[1]) + ", " +
                 std::to_string(size[2]) + ") differs from LocalSize " + std::to_string(result.local_size[0]) +
                 " " + std::to_string(result.local_size[1]) + " " + std::to_string(result.local_size[2]) +
                 "; WorkgroupSize would take precedence over the launch block Paralyn enforces");
      result.workgroup_size_constant = true;
    }
  }
  return result;
}

void validate(const std::vector<std::uint32_t> &w) {
  spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_1);
  std::string messages;
  tools.SetMessageConsumer([&](spv_message_level_t, const char *, const spv_position_t &position,
                               const char *message) {
    if (!messages.empty())
      messages += "; ";
    messages += message;
    (void)position;
  });
  if (!tools.IsValid())
    fail("spirv.validation", "SPIRV-Tools could not create a Vulkan 1.1 validator");
  spvtools::ValidatorOptions options;
  if (!tools.Validate(w.data(), w.size(), options))
    fail("spirv.validation", "spirv-val (Vulkan 1.1) rejected the module: " +
                                 (messages.empty() ? std::string("no detail") : messages));
}

bool identifier(const std::string &s) {
  if (s.empty() || s.size() > 255 || (s[0] >= '0' && s[0] <= '9'))
    return false;
  return std::all_of(s.begin(), s.end(), [](unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
  });
}
ScalarType scalar(const sc::SPIRType &t, const std::string &where) {
  if (t.width != 32 || t.vecsize != 1 || t.columns != 1 || !t.array.empty() || t.pointer)
    fail("spirv.resource", where + " must be a 32-bit scalar (f32, i32 or u32)");
  switch (t.basetype) {
  case sc::SPIRType::Float:
    return ScalarType::F32;
  case sc::SPIRType::Int:
    return ScalarType::I32;
  case sc::SPIRType::UInt:
    return ScalarType::U32;
  default:
    fail("spirv.resource", where + " must be f32, i32 or u32");
  }
}
std::string display(const sc::CompilerMSL &c, const sc::Resource &r) {
  const auto &name = c.get_name(r.id);
  return "resource " + (name.empty() ? "%" + std::to_string(std::uint32_t(r.id)) : "'" + name + "'");
}

struct Reflected {
  std::vector<ExecutableParameter> parameters;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> descriptor_bindings; // (set, binding)
  bool push_constants = false;
};

Reflected reflect(sc::CompilerMSL &c, const std::vector<unsigned char> &binary) {
  const auto active = c.get_active_interface_variables();
  const auto all = c.get_shader_resources();
  const auto used = c.get_shader_resources(active);
  auto reject_any = [&](const sc::SmallVector<sc::Resource> &list, const char *kind) {
    if (!list.empty())
      fail("spirv.resource", std::string(kind) + " are outside the buffer/scalar profile (" +
                                 display(c, list.front()) + ")");
  };
  reject_any(all.stage_inputs, "non-builtin stage inputs");
  reject_any(all.stage_outputs, "stage outputs");
  reject_any(all.subpass_inputs, "subpass inputs");
  reject_any(all.storage_images, "storage images");
  reject_any(all.sampled_images, "sampled images");
  reject_any(all.separate_images, "separate images");
  reject_any(all.separate_samplers, "samplers");
  reject_any(all.atomic_counters, "atomic counters");
  reject_any(all.acceleration_structures, "acceleration structures");
  reject_any(all.gl_plain_uniforms, "plain uniforms");
  reject_any(all.tensors, "tensors");
  reject_any(all.shader_record_buffers, "shader record buffers");
  auto is_used = [&](const sc::Resource &r) { return active.count(r.id) != 0; };
  for (const auto *list : {&all.storage_buffers, &all.uniform_buffers, &all.push_constant_buffers})
    for (const auto &r : *list)
      if (!is_used(r))
        fail("spirv.resource", display(c, r) +
                                   " is declared but not statically used by the entry point; "
                                   "the profile reflects every descriptor it binds");

  struct Descriptor {
    std::uint32_t binding;
    const sc::Resource *resource;
    bool storage;
  };
  std::vector<Descriptor> descriptors;
  std::set<std::uint32_t> bindings;
  auto descriptor = [&](const sc::Resource &r, bool storage) {
    const auto set = c.get_decoration(r.id, spv::DecorationDescriptorSet);
    const auto binding = c.get_decoration(r.id, spv::DecorationBinding);
    if (set != 0)
      fail("spirv.binding", display(c, r) + " uses descriptor set " + std::to_string(set) +
                                "; the profile maps only set 0");
    if (binding >= spirv_push_constant_slot)
      fail("spirv.binding", display(c, r) + " binding " + std::to_string(binding) +
                                " exceeds the Metal slot range 0-29");
    if (!bindings.insert(binding).second)
      fail("spirv.binding", "descriptor set 0 binding " + std::to_string(binding) +
                                " is declared more than once");
    if (!c.get_type(r.type_id).array.empty())
      fail("spirv.resource", display(c, r) + " is a descriptor array; arrays of blocks are unsupported");
    descriptors.push_back({binding, &r, storage});
  };
  for (const auto &r : used.storage_buffers)
    descriptor(r, true);
  for (const auto &r : used.uniform_buffers)
    descriptor(r, false);
  std::sort(descriptors.begin(), descriptors.end(),
            [](const Descriptor &a, const Descriptor &b) { return a.binding < b.binding; });
  // Bindings are unique and in set 0 here, so the static use analysis (shared
  // with the runtime's container verifier) can key storage buffers by binding.
  std::map<std::uint32_t, std::uint32_t> use;
  try {
    use = spirv_storage_buffer_access(binary);
  } catch (const std::exception &e) {
    fail("spirv.resource", e.what());
  }

  Reflected result;
  std::set<std::string> names;
  auto add = [&](ExecutableParameter p) {
    if (!names.insert(p.name).second)
      fail("spirv.resource", "parameter name '" + p.name +
                                 "' is not unique across the entry point's reflected resources");
    result.parameters.push_back(std::move(p));
  };
  auto scalar_members = [&](const sc::Resource &r, ResourceOrigin origin, std::uint32_t slot,
                            std::uint32_t binding) {
    const auto &block = c.get_type(r.base_type_id);
    if (block.basetype != sc::SPIRType::Struct || block.member_types.empty())
      fail("spirv.resource", display(c, r) + " must be a nonempty block of scalars");
    for (std::uint32_t m = 0; m < block.member_types.size(); ++m) {
      ExecutableParameter p;
      auto name = c.get_member_name(r.base_type_id, m);
      const std::string where = display(c, r) + " member " + std::to_string(m);
      if (!identifier(name))
        fail("spirv.resource", where + " needs an OpMemberName that is a valid identifier");
      p.name = name;
      p.type = scalar(c.get_type(block.member_types[m]), where);
      p.buffer = false;
      p.access = ResourceAccess::Read;
      p.binding = slot;
      p.alignment = 4;
      p.minimum_bytes = 4;
      p.block_offset = c.type_struct_member_offset(block, m);
      if (p.block_offset % 4)
        fail("spirv.resource", where + " offset is not four-byte aligned");
      p.origin = origin;
      p.source_binding = binding;
      add(std::move(p));
    }
  };
  for (const auto &d : descriptors) {
    const auto &r = *d.resource;
    if (!d.storage) {
      scalar_members(r, ResourceOrigin::UniformMember, d.binding, d.binding);
      result.descriptor_bindings.emplace_back(0, d.binding);
      continue;
    }
    if (c.get_storage_class(r.id) != spv::StorageClassStorageBuffer)
      fail("spirv.resource", display(c, r) +
                                 " uses the legacy Uniform+BufferBlock form; use the StorageBuffer "
                                 "storage class (SPIR-V 1.3)");
    const auto &block = c.get_type(r.base_type_id);
    if (block.basetype != sc::SPIRType::Struct || block.member_types.size() != 1)
      fail("spirv.resource", display(c, r) + " must be a block with exactly one array member");
    const auto &array = c.get_type(block.member_types[0]);
    if (array.array.size() != 1 || array.pointer)
      fail("spirv.resource", display(c, r) + " member must be a one-dimensional array");
    const sc::TypeID element_id = array.parent_type ? sc::TypeID(array.parent_type) : sc::TypeID(array.self);
    const auto &element = c.get_type(element_id);
    if (element.array.size())
      fail("spirv.resource", display(c, r) + " member must be an array of scalars");
    ExecutableParameter p;
    auto name = c.get_name(r.id);
    if (!identifier(name))
      fail("spirv.resource", display(c, r) + " needs an OpName that is a valid identifier");
    p.name = name;
    p.type = scalar(element, display(c, r) + " element");
    if (c.type_struct_member_offset(block, 0) != 0 || c.type_struct_member_array_stride(block, 0) != 4)
      fail("spirv.resource", display(c, r) + " array must start at offset 0 with ArrayStride 4");
    const auto flags = c.get_buffer_block_flags(r.id);
    const auto variable = c.get_decoration_bitset(r.id);
    if (flags.get(spv::DecorationVolatile) || flags.get(spv::DecorationCoherent) ||
        variable.get(spv::DecorationVolatile) || variable.get(spv::DecorationCoherent))
      fail("spirv.resource", display(c, r) + " uses Volatile/Coherent, outside the memory profile");
    // Access is the statically derived use, which the runtime re-derives from
    // the retained binary. Decorations may only narrow it, never contradict it.
    const bool readonly = flags.get(spv::DecorationNonWritable) || variable.get(spv::DecorationNonWritable);
    const bool writeonly = flags.get(spv::DecorationNonReadable) || variable.get(spv::DecorationNonReadable);
    const auto found = use.find(d.binding);
    const std::uint32_t bits = found == use.end() ? 0 : found->second;
    if (!bits)
      fail("spirv.resource", display(c, r) + " is never loaded or stored");
    if (readonly && (bits & 2))
      fail("spirv.access", display(c, r) + " is decorated NonWritable but the module writes to it (or uses its pointer outside loads and stores)");
    if (writeonly && (bits & 1))
      fail("spirv.access", display(c, r) + " is decorated NonReadable but the module loads from it");
    p.access = static_cast<ResourceAccess>(bits);
    p.buffer = true;
    p.binding = d.binding;
    p.alignment = 4;
    const bool runtime = !array.array_size_literal.empty() && array.array_size_literal[0] && array.array[0] == 0;
    if (!array.array_size_literal.empty() && !array.array_size_literal[0])
      fail("spirv.specialization", display(c, r) + " array length depends on a specialization constant");
    p.minimum_bytes = runtime ? 4 : std::uint64_t(array.array[0]) * 4;
    p.origin = ResourceOrigin::StorageBuffer;
    p.source_binding = d.binding;
    add(std::move(p));
    result.descriptor_bindings.emplace_back(0, d.binding);
  }
  if (used.push_constant_buffers.size() > 1)
    fail("spirv.resource", "more than one push-constant block");
  for (const auto &r : used.push_constant_buffers) {
    scalar_members(r, ResourceOrigin::PushConstantMember, spirv_push_constant_slot, 0);
    result.push_constants = true;
  }
  return result;
}

std::uint32_t builtin_mask(const sc::CompilerMSL &c, const Scan &s) {
  std::uint32_t mask = s.workgroup_size_constant ? BuiltinWorkgroupSize : 0;
  const auto used = c.get_shader_resources(c.get_active_interface_variables());
  for (const auto &b : used.builtin_inputs) {
    switch (b.builtin) {
    case spv::BuiltInGlobalInvocationId:
      mask |= BuiltinGlobalInvocationId;
      break;
    case spv::BuiltInLocalInvocationId:
      mask |= BuiltinLocalInvocationId;
      break;
    case spv::BuiltInLocalInvocationIndex:
      mask |= BuiltinLocalInvocationIndex;
      break;
    case spv::BuiltInWorkgroupId:
      mask |= BuiltinWorkgroupId;
      break;
    case spv::BuiltInNumWorkgroups:
      mask |= BuiltinNumWorkgroups;
      break;
    default:
      fail("spirv.builtin", std::string("unsupported builtin ") + spv::BuiltInToString(b.builtin));
    }
  }
  return mask;
}
} // namespace

ImportError::ImportError(std::string diagnostic, const std::string &detail)
    : std::runtime_error("ParalynError: SPIR-V import [" + diagnostic + "]: " + detail),
      code(std::move(diagnostic)) {}

std::string toolchain() { return PARALYN_SPIRV_TOOLCHAIN; }

std::vector<std::uint32_t> assemble(const std::string &text) {
  spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_1);
  std::string messages;
  tools.SetMessageConsumer([&](spv_message_level_t, const char *, const spv_position_t &position,
                               const char *message) {
    if (!messages.empty())
      messages += "; ";
    messages += "line " + std::to_string(position.line + 1) + ", column " +
                std::to_string(position.column + 1) + ": " + message;
  });
  std::vector<std::uint32_t> binary;
  if (!tools.Assemble(text, &binary, SPV_TEXT_TO_BINARY_OPTION_NONE))
    fail("spirv.assembly", "spirv-as (Vulkan 1.1) rejected the assembly: " +
                               (messages.empty() ? std::string("no detail") : messages));
  return binary;
}

std::vector<std::uint32_t> binary_words(const void *data, std::size_t size) {
  if (!data || size < 20 || size % 4 || size > executable_max_spirv_bytes)
    fail("spirv.invalid-binary", "SPIR-V binaries are nonempty, word-aligned and at most 4 MiB");
  const auto *bytes = static_cast<const unsigned char *>(data);
  std::vector<std::uint32_t> words(size / 4);
  for (std::size_t i = 0; i < words.size(); ++i)
    words[i] = std::uint32_t(bytes[4 * i]) | std::uint32_t(bytes[4 * i + 1]) << 8 |
               std::uint32_t(bytes[4 * i + 2]) << 16 | std::uint32_t(bytes[4 * i + 3]) << 24;
  if (words[0] != magic)
    fail("spirv.invalid-binary", words[0] == 0x03022307
                                     ? "big-endian SPIR-V is not accepted"
                                     : "missing SPIR-V magic number");
  return words;
}

ExecutableModule import_module(const std::vector<std::uint32_t> &words, const ImportOptions &options) {
  if (words.size() * 4 > executable_max_spirv_bytes)
    fail("spirv.invalid-binary", "SPIR-V binary exceeds 4 MiB");
  const auto profile = scan(words);
  validate(words);
  ExecutableModule module;
  module.format = ExecutableFormat::SpirvMsl;
  module.target = "metal-msl3.1";
  module.numerical_policy = 1;
  module.producer = "paralyn-spirv-import";
  module.producer_version = options.producer_version;
  module.source_name = options.source_name;
  module.toolchain = toolchain();
  try {
    sc::CompilerMSL compiler(words);
    const auto points = compiler.get_entry_points_and_stages();
    if (points.size() != 1 || points.front().execution_model != spv::ExecutionModelGLCompute)
      fail("spirv.entry-point", "the profile requires exactly one GLCompute entry point");
    compiler.set_entry_point(points.front().name, spv::ExecutionModelGLCompute);
    std::vector<unsigned char> bytes(words.size() * 4);
    for (std::size_t i = 0; i < words.size(); ++i)
      for (unsigned b = 0; b < 4; ++b)
        bytes[4 * i + b] = static_cast<unsigned char>(words[i] >> (8 * b));
    auto reflected = reflect(compiler, bytes);
    ExecutableEntry entry;
    entry.required_block = profile.local_size;
    entry.builtins = builtin_mask(compiler, profile);
    for (const auto &[set, binding] : reflected.descriptor_bindings) {
      sc::MSLResourceBinding b;
      b.stage = spv::ExecutionModelGLCompute;
      b.desc_set = set;
      b.binding = binding;
      b.count = 1;
      b.msl_buffer = binding;
      compiler.add_msl_resource_binding(b);
    }
    if (reflected.push_constants) {
      sc::MSLResourceBinding b;
      b.stage = spv::ExecutionModelGLCompute;
      b.desc_set = sc::kPushConstDescSet;
      b.binding = sc::kPushConstBinding;
      b.count = 1;
      b.msl_buffer = spirv_push_constant_slot;
      compiler.add_msl_resource_binding(b);
    }
    auto msl = compiler.get_msl_options();
    msl.platform = sc::CompilerMSL::Options::macOS;
    msl.set_msl_version(3, 1);
    msl.argument_buffers = false;
    msl.enable_decoration_binding = false;
    compiler.set_msl_options(msl);
    module.source = compiler.compile();
    if (compiler.needs_buffer_size_buffer() || compiler.needs_swizzle_buffer() ||
        compiler.needs_output_buffer() || compiler.needs_dispatch_base_buffer() ||
        compiler.needs_view_mask_buffer() || compiler.needs_input_threadgroup_mem())
      fail("spirv.instruction", "lowering requires an auxiliary Metal side-channel buffer outside the profile");
    for (const auto &[set, binding] : reflected.descriptor_bindings)
      if (!compiler.is_msl_resource_binding_used(spv::ExecutionModelGLCompute, set, binding))
        fail("spirv.binding", "SPIRV-Cross did not bind set " + std::to_string(set) + " binding " +
                                  std::to_string(binding) + " as reflected");
    if (reflected.push_constants &&
        !compiler.is_msl_resource_binding_used(spv::ExecutionModelGLCompute, sc::kPushConstDescSet,
                                               sc::kPushConstBinding))
      fail("spirv.binding", "SPIRV-Cross did not bind the push-constant block as reflected");
    entry.name = compiler.get_cleansed_entry_point_name(points.front().name, spv::ExecutionModelGLCompute);
    entry.parameters = std::move(reflected.parameters);
    module.entries.push_back(std::move(entry));
  } catch (const sc::CompilerError &e) {
    fail("spirv.cross", std::string("SPIRV-Cross MSL lowering failed: ") + e.what());
  }
  // Numerical policy 1 forbids source-level overrides of contraction/fast math.
  for (const char *forbidden : {"fast::", "FP_CONTRACT", "fp_contract", "_Pragma", "#pragma metal", "#pragma METAL"})
    if (module.source.find(forbidden) != std::string::npos)
      fail("spirv.numerics", std::string("generated MSL contains '") + forbidden +
                                 "', which would override numerical policy 1");
  module.source_sha256 = source_sha256(module.source);
  module.spirv.resize(words.size() * 4);
  for (std::size_t i = 0; i < words.size(); ++i)
    for (unsigned b = 0; b < 4; ++b)
      module.spirv[4 * i + b] = static_cast<unsigned char>(words[i] >> (8 * b));
  module.spirv_sha256 = source_sha256(std::string(module.spirv.begin(), module.spirv.end()));
  try {
    verify_executable(module);
  } catch (const std::exception &e) {
    fail("spirv.descriptor", e.what());
  }
  return module;
}
} // namespace paralyn::spirv
