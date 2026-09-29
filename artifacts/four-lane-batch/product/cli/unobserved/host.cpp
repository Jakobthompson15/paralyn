#include <paralyn/runtime.hpp>
#include <cuda_runtime.h>
#include <cuda_runtime.h>
static const paralyn::Kernel& __paralyn_generated_kernel_unused() {
  static const paralyn::Kernel kernel = paralyn::Kernel{"unused",{paralyn::Parameter{"p",paralyn::ScalarType::F32,1,0},},{paralyn::Statement{paralyn::StmtKind::Store,"",paralyn::ScalarType::F32,paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::F32,"1.0E+0",{},2,51},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"p",{},2,36},paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"threadIdx.x",{},2,48},},2,36},{}},}};
  return kernel;
}

int main() { cudaFree(reinterpret_cast<void*>(1)); return 0; }
