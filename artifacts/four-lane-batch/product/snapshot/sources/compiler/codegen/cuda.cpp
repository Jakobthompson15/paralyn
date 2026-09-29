#include "paralyn/cuda_codegen.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>

namespace paralyn {
namespace {
const char *cuda_type(ScalarType t) {
  switch (t) {
  case ScalarType::I32:
    return "int";
  case ScalarType::U32:
    return "unsigned int";
  case ScalarType::F32:
    return "float";
  case ScalarType::Bool:
    return "bool";
  }
  throw std::runtime_error("ParalynError: unknown CUDA scalar type");
}
using Names = std::map<std::string, std::string>;
std::string expr(const Expr &e, const Names &names) {
  switch (e.kind) {
  case ExprKind::Literal:
    if (e.type == ScalarType::F32) {
      // Exact bit pattern: never depends on decimal parsing in NVRTC.
      const float f = std::strtof(e.text.c_str(), nullptr);
      std::uint32_t bits;
      std::memcpy(&bits, &f, 4);
      return "__uint_as_float(" + std::to_string(bits) + "u)";
    }
    if (e.type == ScalarType::U32)
      return std::to_string(std::strtoull(e.text.c_str(), nullptr, 10)) + "u";
    if (e.type == ScalarType::Bool)
      return e.text == "1" ? "true" : "false";
    // Decimal IR literal; a 64-bit intermediate keeps INT32_MIN well-formed.
    return "static_cast<int>(" + std::to_string(std::strtoll(e.text.c_str(), nullptr, 10)) + "ll)";
  case ExprKind::Ref:
    return names.at(e.text);
  case ExprKind::Builtin:
    // Verified: threadIdx/blockIdx/blockDim/gridDim with x/y/z; CUDA builtins are unsigned.
    return e.text;
  case ExprKind::Cast:
    return std::string("static_cast<") + cuda_type(e.type) + ">(" + expr(e.operands[0], names) + ")";
  case ExprKind::Binary: {
    const auto lhs = expr(e.operands[0], names), rhs = expr(e.operands[1], names);
    if (e.operands[0].type == ScalarType::F32 && e.text == "+")
      return "__fadd_rn(" + lhs + ", " + rhs + ")";
    if (e.operands[0].type == ScalarType::F32 && e.text == "*")
      return "__fmul_rn(" + lhs + ", " + rhs + ")";
    return "(" + lhs + " " + e.text + " " + rhs + ")";
  }
  case ExprKind::Load:
    return expr(e.operands[0], names) + "[" + expr(e.operands[1], names) + "]";
  }
  throw std::runtime_error("ParalynError: unknown CUDA expression");
}
void body(std::ostream &out, const std::vector<Statement> &statements, Names names, unsigned indent,
          unsigned &locals) {
  for (const auto &s : statements) {
    out << std::string(indent, ' ');
    if (s.kind == StmtKind::Let) {
      auto value = expr(s.expression, names);
      auto name = "uc_local_" + std::to_string(locals++);
      names[s.name] = name;
      out << cuda_type(s.type) << " " << name << " = " << value << ";\n";
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
std::string cuda_entrypoint(const Kernel &k) { return "uc_kernel_" + k.name; }
std::vector<std::string> cuda_numerical_options() {
  return {"--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"};
}
std::string emit_cuda(const Kernel &k) {
  verify(k);
  std::ostringstream out;
  out << "// Generated exclusively from verified Paralyn IR (CUDA C++ device code for NVRTC).\n"
         "// Numerical policy 1: __fadd_rn/__fmul_rn, --fmad=false, --ftz=false, IEEE div/sqrt.\n"
         "extern \"C\" __global__ void "
      << cuda_entrypoint(k) << "(";
  Names names;
  for (std::size_t i = 0; i < k.parameters.size(); ++i) {
    const auto &p = k.parameters[i];
    const auto name = "uc_arg_" + std::to_string(i);
    names[p.name] = name;
    out << (i ? ",\n    " : "\n    ");
    if (p.buffer)
      out << (p.read_only ? "const " : "") << cuda_type(p.type) << " *" << name;
    else
      out << cuda_type(p.type) << " " << name;
  }
  out << ") {\n";
  unsigned locals = 0;
  body(out, k.body, names, 2, locals);
  out << "}\n";
  return out.str();
}
} // namespace paralyn
