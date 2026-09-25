#include "paralyn/ir.hpp"
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>

namespace paralyn {
namespace {
const char *msl_type(ScalarType t) {
  switch (t) {
  case ScalarType::I32:
    return "int";
  case ScalarType::U32:
    return "uint";
  case ScalarType::F32:
    return "float";
  case ScalarType::Bool:
    return "bool";
  }
  throw std::runtime_error("ParalynError: unknown MSL scalar type");
}
using Names = std::map<std::string, std::string>;
std::string expr(const Expr &e, const Names &names) {
  switch (e.kind) {
  case ExprKind::Literal:
    if (e.type == ScalarType::F32) {
      float f = std::strtof(e.text.c_str(), nullptr);
      std::uint32_t bits;
      std::memcpy(&bits, &f, 4);
      return "as_type<float>(" + std::to_string(bits) + "u)";
    }
    if (e.type == ScalarType::U32)
      return std::to_string(std::strtoull(e.text.c_str(), nullptr, 10)) + "u";
    if (e.type == ScalarType::Bool)
      return e.text == "1" ? "true" : "false";
    return "int(" + std::to_string(std::strtoll(e.text.c_str(), nullptr, 10)) + ")";
  case ExprKind::Ref:
    return names.at(e.text);
  case ExprKind::Builtin: {
    const std::map<std::string, std::string> builtins = {{"threadIdx", "uc_tid"},
                                                         {"blockIdx", "uc_bid"},
                                                         {"blockDim", "uc_bdim"},
                                                         {"gridDim", "uc_gdim"}};
    auto dot = e.text.find('.');
    return builtins.at(e.text.substr(0, dot)) + e.text.substr(dot);
  }
  case ExprKind::Cast:
    return std::string(msl_type(e.type)) + "(" + expr(e.operands[0], names) + ")";
  case ExprKind::Binary:
    return "(" + expr(e.operands[0], names) + " " + e.text + " " + expr(e.operands[1], names) + ")";
  case ExprKind::Load:
    return expr(e.operands[0], names) + "[" + expr(e.operands[1], names) + "]";
  }
  throw std::runtime_error("ParalynError: unknown MSL expression");
}
void body(std::ostream &out, const std::vector<Statement> &statements, Names names, unsigned indent,
          unsigned &locals) {
  for (const auto &s : statements) {
    out << std::string(indent, ' ');
    if (s.kind == StmtKind::Let) {
      auto value = expr(s.expression, names);
      auto name = "uc_local_" + std::to_string(locals++);
      names[s.name] = name;
      out << msl_type(s.type) << " " << name << " = " << value << ";\n";
    } else if (s.kind == StmtKind::Store)
      out << expr(s.target, names) << " = " << expr(s.expression, names) << ";\n";
    else {
      out << "if (" << expr(s.expression, names) << ") {\n";
      body(out, s.body, names, indent + 2, locals);
      out << std::string(indent, ' ') << "}\n";
    }
  }
}
} // namespace
std::string emit_msl(const Kernel &k, const BindingLayout &layout) {
  verify(k);
  if (layout.size() != k.parameters.size())
    throw std::runtime_error("ParalynError: MSL binding layout length mismatch");
  struct Slot {
    ScalarType type;
    bool buffer;
    bool writable;
  };
  std::map<unsigned, Slot> slots;
  for (std::size_t i = 0; i < layout.size(); ++i) {
    const auto &p = k.parameters[i];
    auto pos = slots.find(layout[i]);
    if (pos == slots.end())
      slots.emplace(layout[i], Slot{p.type, p.buffer, !p.read_only});
    else {
      if (!p.buffer || !pos->second.buffer || p.type != pos->second.type)
        throw std::runtime_error("ParalynError: mixed-type or scalar alias binding is unsupported");
      pos->second.writable |= !p.read_only;
    }
  }
  unsigned next = 0;
  for (const auto &entry : slots)
    if (entry.first != next++)
      throw std::runtime_error("ParalynError: binding slots must be contiguous");
  std::ostringstream out;
  out << "#include <metal_stdlib>\nusing namespace metal;\n#pragma STDC FP_CONTRACT OFF\n\n// Generated exclusively from "
         "verified Paralyn IR.\n";
  out << "kernel void uc_kernel_" << k.name << "(\n";
  for (const auto &[index, s] : slots) {
    out << "  ";
    if (s.buffer)
      out << (s.writable ? "device " : "const device ") << msl_type(s.type) << "* uc_slot_"
          << index;
    else
      out << "constant " << msl_type(s.type) << "& uc_slot_" << index;
    out << " [[buffer(" << index << ")]],\n";
  }
  out << "  uint3 uc_tid [[thread_position_in_threadgroup]],\n"
      << "  uint3 uc_bid [[threadgroup_position_in_grid]],\n"
      << "  uint3 uc_bdim [[threads_per_threadgroup]],\n"
      << "  uint3 uc_gdim [[threadgroups_per_grid]]) {\n";
  Names names;
  for (std::size_t i = 0; i < k.parameters.size(); ++i) {
    const auto &p = k.parameters[i];
    auto name = "uc_arg_" + std::to_string(i);
    names[p.name] = name;
    out << "  ";
    if (p.buffer)
      out << (p.read_only ? "const device " : "device ") << msl_type(p.type) << "* ";
    else
      out << msl_type(p.type) << " ";
    out << name << " = uc_slot_" << layout[i] << ";\n";
  }
  unsigned locals = 0;
  body(out, k.body, names, 2, locals);
  out << "}\n";
  return out.str();
}
} // namespace paralyn
