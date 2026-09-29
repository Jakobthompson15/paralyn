#include "paralyn/artifact.hpp"
#include <algorithm>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>

namespace paralyn {
namespace {
constexpr unsigned char magic[] = {'P', 'A', 'R', 'A', 'L', 'Y', 'N', 0};
constexpr std::uint32_t format_version = 1;
// i32/u32 + safe/precise f32, with contraction disabled. A changed numerical
// contract must receive a new policy identifier, not silently reuse this one.
constexpr std::uint32_t numerical_policy = 1;
constexpr std::uint32_t max_kernels = 64;
constexpr std::uint32_t max_parameters = 256;
constexpr std::uint32_t max_nodes = 100000;
constexpr std::uint32_t max_depth = 64;
constexpr std::uint32_t max_string = 4096;
constexpr std::uint32_t max_statements = 65536;

[[noreturn]] void bad(const std::string &message) {
  throw std::runtime_error("ParalynError: invalid module artifact: " + message);
}
std::uint32_t scalar_code(ScalarType type) {
  switch (type) {
  case ScalarType::I32:
    return 0;
  case ScalarType::U32:
    return 1;
  case ScalarType::F32:
    return 2;
  case ScalarType::Bool:
    return 3;
  }
  bad("unknown scalar type");
}
ScalarType scalar_type(std::uint32_t code) {
  switch (code) {
  case 0:
    return ScalarType::I32;
  case 1:
    return ScalarType::U32;
  case 2:
    return ScalarType::F32;
  case 3:
    return ScalarType::Bool;
  default:
    bad("unknown scalar type");
  }
}
std::uint32_t expression_code(ExprKind kind) {
  switch (kind) {
  case ExprKind::Literal:
    return 0;
  case ExprKind::Ref:
    return 1;
  case ExprKind::Builtin:
    return 2;
  case ExprKind::Binary:
    return 3;
  case ExprKind::Cast:
    return 4;
  case ExprKind::Load:
    return 5;
  }
  bad("unknown expression kind");
}
ExprKind expression_kind(std::uint32_t code) {
  switch (code) {
  case 0:
    return ExprKind::Literal;
  case 1:
    return ExprKind::Ref;
  case 2:
    return ExprKind::Builtin;
  case 3:
    return ExprKind::Binary;
  case 4:
    return ExprKind::Cast;
  case 5:
    return ExprKind::Load;
  default:
    bad("unknown expression kind");
  }
}
struct Budget {
  std::uint32_t nodes = 0;
  void node(unsigned depth) {
    if (depth > max_depth)
      bad("nesting depth exceeds limit");
    if (++nodes > max_nodes)
      bad("node count exceeds limit");
  }
};
struct Writer : Budget {
  std::vector<unsigned char> bytes;
  void byte(unsigned char value) {
    if (bytes.size() == artifact_max_bytes)
      bad("artifact exceeds byte limit");
    bytes.push_back(value);
  }
  void u32(std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8)
      byte(static_cast<unsigned char>(value >> shift));
  }
  void count(std::size_t size, std::uint32_t limit) {
    if (size > limit)
      bad("collection count exceeds limit");
    u32(static_cast<std::uint32_t>(size));
  }
  void string(const std::string &value) {
    count(value.size(), max_string);
    for (unsigned char c : value) {
      if (!c)
        bad("embedded NUL in string");
      byte(c);
    }
  }
  void expression(const Expr &value, unsigned depth) {
    node(depth);
    u32(expression_code(value.kind));
    u32(scalar_code(value.type));
    string(value.text);
    u32(value.line);
    u32(value.column);
    count(value.operands.size(), 2);
    for (const auto &operand : value.operands)
      expression(operand, depth + 1);
  }
  void body(const std::vector<Statement> &statements, unsigned depth) {
    count(statements.size(), max_statements);
    for (const auto &statement : statements) {
      node(depth);
      // Inactive Statement fields are deliberately omitted. This mirrors the
      // verifier and canonicalizes objects without serializing unverified data.
      switch (statement.kind) {
      case StmtKind::Let:
        u32(0);
        string(statement.name);
        u32(scalar_code(statement.type));
        expression(statement.expression, depth + 1);
        break;
      case StmtKind::Store:
        u32(1);
        u32(scalar_code(statement.type));
        expression(statement.expression, depth + 1);
        expression(statement.target, depth + 1);
        break;
      case StmtKind::If:
        u32(2);
        expression(statement.expression, depth + 1);
        body(statement.body, depth + 1);
        break;
      default:
        bad("unknown statement kind");
      }
    }
  }
};
struct Reader : Budget {
  const unsigned char *bytes;
  std::size_t size;
  std::size_t position = 0;
  Reader(const void *data, std::size_t length)
      : bytes(static_cast<const unsigned char *>(data)), size(length) {
    if (!data || size > artifact_max_bytes)
      bad("null input or artifact exceeds byte limit");
  }
  void need(std::size_t count) const {
    if (count > size - position)
      bad("truncated input");
  }
  unsigned char byte() {
    need(1);
    return bytes[position++];
  }
  std::uint32_t u32() {
    need(4);
    std::uint32_t value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8)
      value |= static_cast<std::uint32_t>(bytes[position++]) << shift;
    return value;
  }
  std::uint32_t count(std::uint32_t limit, std::size_t minimum_bytes) {
    auto value = u32();
    if (value > limit)
      bad("collection count exceeds limit");
    if (minimum_bytes && value > (size - position) / minimum_bytes)
      bad("truncated collection");
    return value;
  }
  std::string string() {
    auto length = count(max_string, 1);
    std::string value(reinterpret_cast<const char *>(bytes + position), length);
    position += length;
    if (value.find('\0') != std::string::npos)
      bad("embedded NUL in string");
    return value;
  }
  Expr expression(unsigned depth) {
    node(depth);
    Expr result;
    result.kind = expression_kind(u32());
    result.type = scalar_type(u32());
    result.text = string();
    result.line = u32();
    result.column = u32();
    auto size = count(2, 24);
    for (std::uint32_t i = 0; i < size; ++i)
      result.operands.push_back(expression(depth + 1));
    return result;
  }
  std::vector<Statement> body(unsigned depth) {
    auto size = count(max_statements, 4);
    std::vector<Statement> result;
    for (std::uint32_t i = 0; i < size; ++i) {
      node(depth);
      Statement statement;
      switch (u32()) {
      case 0:
        statement.kind = StmtKind::Let;
        statement.name = string();
        statement.type = scalar_type(u32());
        statement.expression = expression(depth + 1);
        break;
      case 1:
        statement.kind = StmtKind::Store;
        statement.type = scalar_type(u32());
        statement.expression = expression(depth + 1);
        statement.target = expression(depth + 1);
        break;
      case 2:
        statement.kind = StmtKind::If;
        statement.expression = expression(depth + 1);
        statement.body = body(depth + 1);
        break;
      default:
        bad("unknown statement kind");
      }
      result.push_back(std::move(statement));
    }
    return result;
  }
};
} // namespace

std::vector<unsigned char> serialize_module(const std::vector<Kernel> &kernels) {
  if (kernels.empty())
    bad("module contains no kernels");
  Writer writer;
  for (auto c : magic)
    writer.byte(c);
  writer.u32(format_version);
  writer.u32(numerical_policy);
  writer.count(kernels.size(), max_kernels);
  std::set<std::string> names;
  for (const auto &kernel : kernels) {
    if (!names.insert(kernel.name).second)
      bad("duplicate entrypoint " + kernel.name);
    writer.string(kernel.name);
    writer.count(kernel.parameters.size(), max_parameters);
    for (const auto &parameter : kernel.parameters) {
      writer.string(parameter.name);
      writer.u32(scalar_code(parameter.type));
      writer.u32((parameter.buffer ? 1u : 0u) | (parameter.read_only ? 2u : 0u));
    }
    writer.body(kernel.body, 0);
    // Enforce recursion limits before invoking the IR verifier.
    verify(kernel);
  }
  return std::move(writer.bytes);
}

std::vector<Kernel> deserialize_module(const void *data, std::size_t size) {
  Reader reader(data, size);
  for (auto c : magic)
    if (reader.byte() != c)
      bad("incorrect magic");
  if (reader.u32() != format_version)
    bad("unsupported format version");
  if (reader.u32() != numerical_policy)
    bad("unsupported numerical policy");
  auto kernel_count = reader.count(max_kernels, 12);
  if (!kernel_count)
    bad("module contains no kernels");
  std::vector<Kernel> kernels;
  std::set<std::string> names;
  for (std::uint32_t i = 0; i < kernel_count; ++i) {
    Kernel kernel;
    kernel.name = reader.string();
    if (!names.insert(kernel.name).second)
      bad("duplicate entrypoint " + kernel.name);
    auto parameter_count = reader.count(max_parameters, 12);
    for (std::uint32_t p = 0; p < parameter_count; ++p) {
      Parameter parameter;
      parameter.name = reader.string();
      parameter.type = scalar_type(reader.u32());
      auto flags = reader.u32();
      if (flags > 3)
        bad("unknown parameter flags");
      parameter.buffer = (flags & 1) != 0;
      parameter.read_only = (flags & 2) != 0;
      kernel.parameters.push_back(std::move(parameter));
    }
    kernel.body = reader.body(0);
    verify(kernel);
    kernels.push_back(std::move(kernel));
  }
  if (reader.position != reader.size)
    bad("trailing data");
  return kernels;
}
} // namespace paralyn
