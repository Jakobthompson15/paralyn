#include "paralyn/ir.hpp"
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace paralyn {
const char *type_name(ScalarType t) {
  switch (t) {
  case ScalarType::I32:
    return "i32";
  case ScalarType::U32:
    return "u32";
  case ScalarType::F32:
    return "f32";
  case ScalarType::Bool:
    return "bool";
  }
  throw std::runtime_error("ParalynError: invalid IR scalar type");
}
namespace {
[[noreturn]] void bad(const std::string &why) {
  throw std::runtime_error("ParalynError: invalid IR: " + why);
}
bool integer(ScalarType t) { return t == ScalarType::I32 || t == ScalarType::U32; }
bool identifier(const std::string &n) {
  if (n.empty() || !(std::isalpha(static_cast<unsigned char>(n[0])) || n[0] == '_'))
    return false;
  for (unsigned char c : n)
    if (!std::isalnum(c) && c != '_')
      return false;
  return true;
}
struct Symbol {
  ScalarType type;
  bool buffer;
  bool read_only;
};
using Symbols = std::map<std::string, Symbol>;
void literal(const Expr &e) {
  if (e.text.empty())
    bad("empty literal");
  char *end = nullptr;
  errno = 0;
  if (e.type == ScalarType::F32) {
    float v = std::strtof(e.text.c_str(), &end);
    if (!std::isfinite(v) || errno == ERANGE)
      bad("non-finite or out-of-range float literal");
  } else if (e.type == ScalarType::I32) {
    long long v = std::strtoll(e.text.c_str(), &end, 10);
    if (errno || v < INT32_MIN || v > INT32_MAX)
      bad("out-of-range i32 literal");
  } else if (e.type == ScalarType::U32) {
    auto v = std::strtoull(e.text.c_str(), &end, 10);
    if (errno || e.text[0] == '-' || v > UINT32_MAX)
      bad("out-of-range u32 literal");
  } else {
    if (e.text != "0" && e.text != "1")
      bad("invalid predicate literal");
    return;
  }
  if (!end || *end != '\0')
    bad("invalid literal spelling");
}
void expression(const Expr &e, const Symbols &symbols, bool allow_buffer = false) {
  type_name(e.type);
  auto count = [&](std::size_t n) {
    if (e.operands.size() != n)
      bad("incorrect expression arity");
  };
  switch (e.kind) {
  case ExprKind::Literal:
    count(0);
    literal(e);
    break;
  case ExprKind::Ref: {
    count(0);
    auto it = symbols.find(e.text);
    if (it == symbols.end())
      bad("undefined symbol " + e.text);
    if (it->second.type != e.type)
      bad("reference type mismatch for " + e.text);
    if (it->second.buffer && !allow_buffer)
      bad("buffer used as scalar " + e.text);
    if (!it->second.buffer && allow_buffer)
      bad("load/store requires buffer");
    break;
  }
  case ExprKind::Builtin: {
    count(0);
    auto p = e.text.find('.');
    if (e.type != ScalarType::U32 || p == std::string::npos)
      bad("builtin type/name");
    auto base = e.text.substr(0, p), dim = e.text.substr(p + 1);
    if ((base != "threadIdx" && base != "blockIdx" && base != "blockDim" && base != "gridDim") ||
        (dim != "x" && dim != "y" && dim != "z"))
      bad("unknown index builtin");
    break;
  }
  case ExprKind::Cast: {
    count(1);
    expression(e.operands[0], symbols);
    if (e.type != e.operands[0].type && !(integer(e.type) && integer(e.operands[0].type)))
      bad("unsupported scalar conversion");
    break;
  }
  case ExprKind::Binary: {
    count(2);
    for (const auto &x : e.operands)
      expression(x, symbols);
    auto t = e.operands[0].type;
    if (t != e.operands[1].type || t == ScalarType::Bool)
      bad("binary operand type mismatch");
    if (e.text == "<") {
      if (e.type != ScalarType::Bool)
        bad("comparison result must be predicate");
    } else if (e.text == "+" || e.text == "*") {
      if (e.type != t)
        bad("arithmetic result type mismatch");
    } else
      bad("unsupported operator " + e.text);
    break;
  }
  case ExprKind::Load: {
    count(2);
    if (e.operands[0].kind != ExprKind::Ref)
      bad("buffer base must be parameter reference");
    expression(e.operands[0], symbols, true);
    expression(e.operands[1], symbols);
    if (!integer(e.operands[1].type) || e.type != e.operands[0].type)
      bad("invalid load type");
    break;
  }
  default:
    bad("unknown expression kind");
  }
}
void statements(const std::vector<Statement> &body, Symbols symbols) {
  for (const auto &s : body) {
    switch (s.kind) {
    case StmtKind::Let:
      if (!identifier(s.name) || symbols.count(s.name))
        bad("duplicate/invalid local " + s.name);
      expression(s.expression, symbols);
      if (s.type != s.expression.type || s.type == ScalarType::Bool)
        bad("invalid local type");
      symbols.emplace(s.name, Symbol{s.type, false, false});
      break;
    case StmtKind::Store:
      if (s.target.kind != ExprKind::Load)
        bad("store target is not indexed buffer");
      expression(s.target, symbols);
      expression(s.expression, symbols);
      if (s.target.type != s.expression.type)
        bad("store type mismatch");
      if (symbols.at(s.target.operands[0].text).read_only)
        bad("store to read-only buffer");
      break;
    case StmtKind::If:
      expression(s.expression, symbols);
      if (s.expression.type != ScalarType::Bool)
        bad("if condition must be predicate");
      statements(s.body, symbols);
      break;
    default:
      bad("unknown statement kind");
    }
  }
}
std::string expr_text(const Expr &e) {
  switch (e.kind) {
  case ExprKind::Literal:
    return e.text + ":" + type_name(e.type);
  case ExprKind::Ref:
  case ExprKind::Builtin:
    return e.text;
  case ExprKind::Cast:
    return std::string(type_name(e.type)) + "(" + expr_text(e.operands[0]) + ")";
  case ExprKind::Binary:
    return "(" + expr_text(e.operands[0]) + " " + e.text + " " + expr_text(e.operands[1]) + ")";
  case ExprKind::Load:
    return "load " + expr_text(e.operands[0]) + "[" + expr_text(e.operands[1]) + "]";
  }
  bad("unknown expression kind");
}
void dump_body(std::ostream &out, const std::vector<Statement> &body, unsigned indent) {
  for (const auto &s : body) {
    out << std::string(indent, ' ');
    if (s.kind == StmtKind::Let)
      out << "let " << s.name << ":" << type_name(s.type) << " = " << expr_text(s.expression)
          << "\n";
    else if (s.kind == StmtKind::Store)
      out << "store " << s.target.operands[0].text << "[" << expr_text(s.target.operands[1])
          << "] = " << expr_text(s.expression) << "\n";
    else {
      out << "if " << expr_text(s.expression) << " {\n";
      dump_body(out, s.body, indent + 2);
      out << std::string(indent, ' ') << "}\n";
    }
  }
}
const char *cpp_type(ScalarType t) {
  switch (t) {
  case ScalarType::I32:
    return "paralyn::ScalarType::I32";
  case ScalarType::U32:
    return "paralyn::ScalarType::U32";
  case ScalarType::F32:
    return "paralyn::ScalarType::F32";
  case ScalarType::Bool:
    return "paralyn::ScalarType::Bool";
  }
  bad("unknown type");
}
void cpp_expr(std::ostream &out, const Expr &e) {
  const char *names[] = {"Literal", "Ref", "Builtin", "Binary", "Cast", "Load"};
  out << "paralyn::Expr{paralyn::ExprKind::" << names[static_cast<unsigned>(e.kind)] << ","
      << cpp_type(e.type) << "," << std::quoted(e.text) << ",{";
  for (const auto &x : e.operands) {
    cpp_expr(out, x);
    out << ",";
  }
  out << "}," << e.line << "," << e.column << "}";
}
void cpp_body(std::ostream &out, const std::vector<Statement> &body) {
  const char *names[] = {"Let", "Store", "If"};
  out << "{";
  for (const auto &s : body) {
    out << "paralyn::Statement{paralyn::StmtKind::" << names[static_cast<unsigned>(s.kind)] << ","
        << std::quoted(s.name) << "," << cpp_type(s.type) << ",";
    cpp_expr(out, s.expression);
    out << ",";
    cpp_expr(out, s.target);
    out << ",";
    cpp_body(out, s.body);
    out << "},";
  }
  out << "}";
}
} // namespace
void verify(const Kernel &k) {
  if (!identifier(k.name))
    bad("invalid kernel name");
  Symbols symbols;
  for (const auto &p : k.parameters) {
    type_name(p.type);
    if (!identifier(p.name) || symbols.count(p.name))
      bad("duplicate/invalid parameter");
    if (p.type == ScalarType::Bool)
      bad("predicate kernel arguments unsupported");
    symbols.emplace(p.name, Symbol{p.type, p.buffer, p.read_only});
  }
  statements(k.body, symbols);
}
std::string dump_ir(const Kernel &k) {
  verify(k);
  std::ostringstream out;
  out << "paralyn.ir v0\nkernel " << k.name << "(";
  for (std::size_t i = 0; i < k.parameters.size(); ++i) {
    const auto &p = k.parameters[i];
    if (i)
      out << ", ";
    out << p.name << ": ";
    if (p.buffer)
      out << (p.read_only ? "read " : "read_write ") << "buffer<";
    out << type_name(p.type);
    if (p.buffer)
      out << ">";
  }
  out << ") {\n";
  dump_body(out, k.body, 2);
  out << "}\n";
  return out.str();
}
std::string emit_cpp(const Kernel &k) {
  verify(k);
  std::ostringstream out;
  out << "paralyn::Kernel{" << std::quoted(k.name) << ",{";
  for (const auto &p : k.parameters)
    out << "paralyn::Parameter{" << std::quoted(p.name) << "," << cpp_type(p.type) << ","
        << p.buffer << "," << p.read_only << "},";
  out << "},";
  cpp_body(out, k.body);
  out << "}";
  return out.str();
}
} // namespace paralyn
