#include "paralyn/artifact.hpp"
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace paralyn;
namespace {
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
void rejects(const std::function<void()> &function) {
  bool rejected = false;
  try {
    function();
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  require(rejected, "invalid module artifact was accepted");
}
void put(std::vector<unsigned char> &bytes, std::uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8)
    bytes.push_back(static_cast<unsigned char>(value >> shift));
}
void string(std::vector<unsigned char> &bytes, const std::string &value) {
  put(bytes, static_cast<std::uint32_t>(value.size()));
  bytes.insert(bytes.end(), value.begin(), value.end());
}
void patch(std::vector<unsigned char> &bytes, std::size_t offset, std::uint32_t value) {
  require(offset + 4 <= bytes.size(), "invalid test patch");
  for (unsigned shift = 0; shift < 32; shift += 8)
    bytes[offset++] = static_cast<unsigned char>(value >> shift);
}
void rejected_bytes(std::vector<unsigned char> bytes) {
  rejects([&] { deserialize_module(bytes.data(), bytes.size()); });
}
Kernel sample() {
  Expr index{ExprKind::Builtin, ScalarType::U32, "threadIdx.x", {}, 8, 7};
  Expr a{ExprKind::Ref, ScalarType::F32, "input", {}, 8, 12};
  Expr b{ExprKind::Ref, ScalarType::F32, "output", {}, 8, 20};
  Expr read{ExprKind::Load, ScalarType::F32, "", {a, index}, 9, 4};
  Expr target{ExprKind::Load, ScalarType::F32, "", {b, index}, 9, 11};
  Statement store{StmtKind::Store, "", ScalarType::F32, read, target, {}};
  Expr limit{ExprKind::Literal, ScalarType::U32, "17", {}};
  Expr condition{ExprKind::Binary, ScalarType::Bool, "<", {index, limit}};
  Statement branch{StmtKind::If, "", ScalarType::I32, condition, {}, {store}};
  return {"copy_values",
          {{"input", ScalarType::F32, true, true}, {"output", ScalarType::F32, true, false}},
          {branch}};
}
} // namespace

int main() {
  try {
    auto kernel = sample();
    auto second = kernel;
    second.name = "copy_again";
    auto bytes = serialize_module({kernel, second});
    auto loaded = deserialize_module(bytes.data(), bytes.size());
    require(loaded.size() == 2, "module lost an entrypoint");
    require(emit_cpp(loaded[0]) == emit_cpp(kernel), "round trip lost IR or source locations");
    require(dump_ir(loaded[1]) == dump_ir(second), "round trip changed second kernel");
    require(serialize_module(loaded) == bytes, "serialization is not deterministic");
    require(emit_msl(loaded[0], {0, 1}) == emit_msl(kernel, {0, 1}),
            "artifact changed generated shader");

    // Independent byte fixture fixes version-1 little-endian schema. It contains
    // one scalar local with a literal and no parameters, independent of Writer.
    std::vector<unsigned char> literal = {'P', 'A', 'R', 'A', 'L', 'Y', 'N', 0};
    put(literal, 1); // format version
    put(literal, 1); // numerical policy
    put(literal, 1); // kernel count
    string(literal, "literal");
    const auto parameter_count = literal.size();
    put(literal, 0);
    const auto statement_count = literal.size();
    put(literal, 1);
    const auto statement_kind = literal.size();
    put(literal, 0); // Let
    string(literal, "x");
    const auto local_type = literal.size();
    put(literal, 0); // i32
    const auto expression_kind = literal.size();
    put(literal, 0); // Literal
    const auto expression_type = literal.size();
    put(literal, 0); // i32
    const auto expression_text = literal.size();
    string(literal, "7");
    put(literal, 0x01020304); // source line demonstrates little-endian encoding
    put(literal, 11);
    const auto operand_count = literal.size();
    put(literal, 0);
    auto literal_module = deserialize_module(literal.data(), literal.size());
    require(literal_module[0].body[0].expression.line == 0x01020304,
            "source coordinate byte order changed");
    require(serialize_module(literal_module) == literal, "version-1 byte schema changed");

    for (std::size_t size = 0; size < bytes.size(); ++size)
      rejects([&] { deserialize_module(bytes.data(), size); });
    rejects([&] { deserialize_module(nullptr, 0); });
    rejects([&] { deserialize_module(bytes.data(), artifact_max_bytes + 1); });
    auto bad = literal;
    bad.push_back(0);
    rejected_bytes(bad);
    bad = literal;
    bad[0] = 'X';
    rejected_bytes(bad);
    for (auto offset : {std::size_t(8), std::size_t(12), std::size_t(16), std::size_t(20),
                        parameter_count, statement_count, statement_kind, local_type,
                        expression_kind, expression_type, expression_text, operand_count}) {
      bad = literal;
      patch(bad, offset, 0xffffffff);
      rejected_bytes(bad);
    }
    bad = literal;
    patch(bad, 16, 0);
    bad.resize(20);
    rejected_bytes(bad);
    bad = literal;
    bad[expression_text + 4] = 0;
    rejected_bytes(bad);
    bad = literal;
    patch(bad, expression_kind, 1); // Undefined Ref must fail semantic verification.
    rejected_bytes(bad);
    bad = literal;
    patch(bad, local_type, 2); // f32 local initialized from i32.
    rejected_bytes(bad);

    bad = literal;
    patch(bad, 16, 2);
    bad.insert(bad.end(), literal.begin() + 20, literal.end());
    rejected_bytes(bad); // duplicate entrypoint names
    std::vector<unsigned char> flags = {'P', 'A', 'R', 'A', 'L', 'Y', 'N', 0};
    put(flags, 1);
    put(flags, 1);
    put(flags, 1);
    string(flags, "empty");
    put(flags, 1);
    string(flags, "a");
    put(flags, 2);
    put(flags, 4); // unknown parameter flag
    put(flags, 0);
    rejected_bytes(flags);

    rejects([&] { serialize_module({}); });
    rejects([&] { serialize_module({kernel, kernel}); });
    auto broken = kernel;
    broken.parameters[1].read_only = true;
    rejects([&] { serialize_module({broken}); });
    broken = kernel;
    broken.parameters[0].name = std::string(4097, 'x');
    rejects([&] { serialize_module({broken}); });
    broken = kernel;
    broken.parameters[0].type = static_cast<ScalarType>(999);
    rejects([&] { serialize_module({broken}); });
    broken = literal_module[0];
    broken.body[0].expression.kind = static_cast<ExprKind>(999);
    rejects([&] { serialize_module({broken}); });
    broken = literal_module[0];
    for (int i = 0; i < 70; ++i) {
      Expr nested{ExprKind::Cast, ScalarType::I32, "", {broken.body[0].expression}};
      broken.body[0].expression = std::move(nested);
    }
    rejects([&] { serialize_module({broken}); });

    // Decode-side depth limits also apply before the recursive IR verifier.
    auto deep = literal;
    deep.resize(expression_kind);
    for (int i = 0; i < 70; ++i) {
      put(deep, 4); // Cast
      put(deep, 0);
      string(deep, "");
      put(deep, 0);
      put(deep, 0);
      put(deep, 1);
    }
    deep.insert(deep.end(), literal.begin() + expression_kind, literal.end());
    rejected_bytes(deep);
    auto nodes = literal;
    nodes.resize(statement_count);
    put(nodes, 60000);
    for (int i = 0; i < 60000; ++i)
      nodes.insert(nodes.end(), literal.begin() + statement_kind, literal.end());
    rejected_bytes(nodes); // 120,000 expressions/statements exceeds node budget.

    // The verifier ignores a Let target; module encoding must not touch it.
    broken = literal_module[0];
    broken.body[0].target.kind = static_cast<ExprKind>(999);
    require(serialize_module({broken}) == literal, "inactive target changed artifact");
    std::cout << "Versioned module artifacts, limits, and IR rejection: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
