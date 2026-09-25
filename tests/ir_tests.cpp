#include "paralyn/ir.hpp"
#include <functional>
#include <iostream>
#include <stdexcept>

using namespace paralyn;
namespace {
void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
void rejects(const std::function<void()> &f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  require(rejected, "invalid IR was accepted");
}
Kernel sample() {
  Expr index{ExprKind::Builtin, ScalarType::U32, "threadIdx.x", {}};
  Expr input{ExprKind::Ref, ScalarType::F32, "input", {}};
  Expr output{ExprKind::Ref, ScalarType::F32, "output", {}};
  Expr load{ExprKind::Load, ScalarType::F32, "", {input, index}};
  Expr target{ExprKind::Load, ScalarType::F32, "", {output, index}};
  Statement store{StmtKind::Store, "", ScalarType::F32, load, target, {}};
  return {"copy_values",
          {{"input", ScalarType::F32, true, true}, {"output", ScalarType::F32, true, false}},
          {store}};
}
} // namespace
int main() {
  try {
    auto k = sample();
    verify(k);
    require(!dump_ir(k).empty(), "IR dump missing");
    require(!emit_cpp(k).empty(), "host serialization missing");
    auto msl = emit_msl(k, {0, 1});
    require(msl.find("kernel void uc_kernel_copy_values") != std::string::npos, "kernel name missing");
    auto reserved_name = k;
    reserved_name.name = "kernel";
    require(emit_msl(reserved_name, {0, 1}).find("kernel void uc_kernel_kernel") != std::string::npos,
            "CUDA identifier collides with MSL keyword");
    auto alias = emit_msl(k, {0, 0});
    require(alias.find("[[buffer(1)]]") == std::string::npos, "aliased allocation bound twice");
    auto broken = k;
    broken.parameters[1].read_only = true;
    rejects([&] { verify(broken); });
    broken = k;
    broken.body[0].expression.operands[0].text = "undefined";
    rejects([&] { verify(broken); });
    broken = k;
    broken.body[0].expression.type = ScalarType::I32;
    rejects([&] { verify(broken); });
    broken = k;
    broken.body[0].type = ScalarType::I32;
    rejects([&] { verify(broken); });
    broken = k;
    broken.body[0].body = k.body;
    rejects([&] { verify(broken); });
    broken = k;
    broken.body[0].expression.kind = static_cast<ExprKind>(999);
    rejects([&] { emit_cpp(broken); });
    auto local = Statement{StmtKind::Let, "value", ScalarType::I32,
                           {ExprKind::Literal, ScalarType::I32, "1", {}}, {}, {}};
    local.target.kind = static_cast<ExprKind>(999); // Inactive field must never reach serializer.
    broken = k;
    broken.body.insert(broken.body.begin(), local);
    require(!emit_cpp(broken).empty(), "serializer read inactive target field");
    local.expression.text = "00012";
    broken.body[0] = local;
    require(emit_msl(broken, {0, 1}).find("int(12)") != std::string::npos,
            "decimal IR literal must not become an octal Metal literal");
    broken.body[0].expression.text = "\n12";
    rejects([&] { emit_cpp(broken); });
    broken = k;
    broken.body[0].target.operands[1].text = "threadIdx.w";
    rejects([&] { verify(broken); });
    rejects([&] { emit_msl(k, {0}); });
    rejects([&] { emit_msl(k, {1, 2}); });
    Kernel mixed{
        "mixed", {{"a", ScalarType::F32, true, true}, {"b", ScalarType::I32, true, true}}, {}};
    rejects([&] { emit_msl(mixed, {0, 0}); });
    std::cout << "IR verification and binding-layout tests: PASS\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
