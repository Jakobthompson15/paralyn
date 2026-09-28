#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace paralyn {
enum class ScalarType { I32, U32, F32, Bool };
enum class ExprKind { Literal, Ref, Builtin, Binary, Cast, Load };
struct Expr {
  ExprKind kind = ExprKind::Literal;
  ScalarType type = ScalarType::I32;
  std::string text;
  std::vector<Expr> operands;
  unsigned line = 0;
  unsigned column = 0;
};
enum class StmtKind { Let, Store, If };
struct Statement {
  StmtKind kind = StmtKind::Let;
  std::string name;
  ScalarType type = ScalarType::I32;
  Expr expression;
  Expr target;
  std::vector<Statement> body;
};
struct Parameter {
  std::string name;
  ScalarType type = ScalarType::I32;
  bool buffer = false;
  bool read_only = false;
};
struct Kernel {
  std::string name;
  std::vector<Parameter> parameters;
  std::vector<Statement> body;
};
// One Metal argument slot per distinct allocation or scalar. Aliases share a slot.
using BindingLayout = std::vector<unsigned>;
const char *type_name(ScalarType type);
void verify(const Kernel &kernel);
std::string dump_ir(const Kernel &kernel);
std::string emit_cpp(const Kernel &kernel);
std::string emit_msl(const Kernel &kernel, const BindingLayout &layout);
// Byte offsets belong to buffer views; aliased views still share one Metal binding.
std::string emit_msl(const Kernel &kernel, const BindingLayout &layout,
                     const std::vector<std::size_t> &offsets);
} // namespace paralyn
