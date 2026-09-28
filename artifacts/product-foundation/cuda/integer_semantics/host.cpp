#include <paralyn/runtime.hpp>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <vector>

static const paralyn::Kernel& __paralyn_generated_kernel_signed_affine() {
  static const paralyn::Kernel kernel = paralyn::Kernel{"signed_affine",{paralyn::Parameter{"input",paralyn::ScalarType::I32,1,1},paralyn::Parameter{"output",paralyn::ScalarType::I32,1,0},paralyn::Parameter{"n",paralyn::ScalarType::I32,0,0},paralyn::Parameter{"scale",paralyn::ScalarType::I32,0,0},paralyn::Parameter{"bias",paralyn::ScalarType::I32,0,0},},{paralyn::Statement{paralyn::StmtKind::Let,"i",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Cast,paralyn::ScalarType::I32,"",{paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"threadIdx.x",{},9,21},},9,11},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"",{},0,0},{}},paralyn::Statement{paralyn::StmtKind::If,"",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::Bool,"<",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},10,7},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"n",{},10,11},},10,9},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"",{},0,0},{paralyn::Statement{paralyn::StmtKind::Store,"",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::I32,"+",{paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::I32,"*",{paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::I32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"input",{},11,17},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},11,23},},11,17},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"scale",{},11,28},},11,26},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"bias",{},11,36},},11,34},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::I32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"output",{},11,5},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},11,12},},11,5},{}},}},}};
  return kernel;
}

static const paralyn::Kernel& __paralyn_generated_kernel_unsigned_to_signed() {
  static const paralyn::Kernel kernel = paralyn::Kernel{"unsigned_to_signed",{paralyn::Parameter{"input",paralyn::ScalarType::U32,1,1},paralyn::Parameter{"output",paralyn::ScalarType::I32,1,0},paralyn::Parameter{"n",paralyn::ScalarType::I32,0,0},},{paralyn::Statement{paralyn::StmtKind::Let,"i",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Cast,paralyn::ScalarType::I32,"",{paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"threadIdx.x",{},14,21},},14,11},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"",{},0,0},{}},paralyn::Statement{paralyn::StmtKind::If,"",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::Bool,"<",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},15,7},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"n",{},15,11},},15,9},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"",{},0,0},{paralyn::Statement{paralyn::StmtKind::Store,"",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Cast,paralyn::ScalarType::I32,"",{paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::U32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::U32,"input",{},16,34},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},16,40},},16,34},},16,34},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::I32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"output",{},16,5},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},16,12},},16,5},{}},}},}};
  return kernel;
}

static const paralyn::Kernel& __paralyn_generated_kernel_signed_to_unsigned() {
  static const paralyn::Kernel kernel = paralyn::Kernel{"signed_to_unsigned",{paralyn::Parameter{"input",paralyn::ScalarType::I32,1,1},paralyn::Parameter{"output",paralyn::ScalarType::U32,1,0},paralyn::Parameter{"n",paralyn::ScalarType::I32,0,0},},{paralyn::Statement{paralyn::StmtKind::Let,"i",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Cast,paralyn::ScalarType::I32,"",{paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"threadIdx.x",{},19,21},},19,11},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"",{},0,0},{}},paralyn::Statement{paralyn::StmtKind::If,"",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::Bool,"<",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},20,7},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"n",{},20,11},},20,9},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"",{},0,0},{paralyn::Statement{paralyn::StmtKind::Store,"",paralyn::ScalarType::U32,paralyn::Expr{paralyn::ExprKind::Cast,paralyn::ScalarType::U32,"",{paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::I32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"input",{},21,43},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},21,49},},21,43},},21,43},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::U32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::U32,"output",{},21,5},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},21,12},},21,5},{}},}},}};
  return kernel;
}

static const paralyn::Kernel& __paralyn_generated_kernel_unsigned_comparison() {
  static const paralyn::Kernel kernel = paralyn::Kernel{"unsigned_comparison",{paralyn::Parameter{"output",paralyn::ScalarType::I32,1,0},paralyn::Parameter{"bound",paralyn::ScalarType::I32,0,0},},{paralyn::Statement{paralyn::StmtKind::If,"",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::Bool,"<",{paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"threadIdx.x",{},24,17},paralyn::Expr{paralyn::ExprKind::Cast,paralyn::ScalarType::U32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"bound",{},24,21},},24,21},},24,19},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"",{},0,0},{paralyn::Statement{paralyn::StmtKind::Store,"",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"1",{},25,27},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::I32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"output",{},25,5},paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"threadIdx.x",{},25,22},},25,5},{}},}},}};
  return kernel;
}

static void check(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}
static void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
int main() {
  try {
    constexpr int n = 6, count = 10, canary = -991357;
    const std::size_t bytes = count * sizeof(std::uint32_t);
    const std::vector<int> signed_input = {std::numeric_limits<int>::min(),
                                           std::numeric_limits<int>::min() + 1,
                                           -1,
                                           0,
                                           1,
                                           std::numeric_limits<int>::max(),
                                           canary,
                                           canary,
                                           canary,
                                           canary};
    const std::vector<unsigned int> unsigned_input = {
        0u,          1u,          0x7fffffffu, 0x80000000u, 0x80000001u,
        0xffffffffu, 0xfefefefeu, 0xfefefefeu, 0xfefefefeu, 0xfefefefeu};
    int *signed_device = nullptr, *signed_output_device = nullptr;
    unsigned int *unsigned_device = nullptr, *unsigned_output_device = nullptr;
    check(cudaMalloc(&signed_device, bytes));
    check(cudaMalloc(&signed_output_device, bytes));
    check(cudaMalloc(&unsigned_device, bytes));
    check(cudaMalloc(&unsigned_output_device, bytes));
    check(cudaMemcpy(signed_device, signed_input.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(unsigned_device, unsigned_input.data(), bytes, cudaMemcpyHostToDevice));
    std::vector<int> signed_actual(count, canary), signed_expected(count, canary);
    std::vector<unsigned int> unsigned_actual(count, 0xfefefefeu),
        unsigned_expected(count, 0xfefefefeu);
    check(cudaMemcpy(signed_output_device, signed_actual.data(), bytes, cudaMemcpyHostToDevice));
    check(
        cudaMemcpy(unsigned_output_device, unsigned_actual.data(), bytes, cudaMemcpyHostToDevice));
    ([&]() {
  const dim3 __paralyn_generated_grid = (1);
  const dim3 __paralyn_generated_block = (8);
  const std::size_t __paralyn_generated_shared = (0);
  const cudaStream_t __paralyn_generated_stream = (0);
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  const unsigned int* __paralyn_generated_argument_0 = (unsigned_device);
  int* __paralyn_generated_argument_1 = (signed_output_device);
  int __paralyn_generated_argument_2 = (n);
  paralyn::launch_checked(__paralyn_generated_kernel_unsigned_to_signed(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_buffer(__paralyn_generated_argument_1), paralyn::Argument::from_i32(__paralyn_generated_argument_2)});
}());
    check(cudaGetLastError());
    ([&]() {
  const dim3 __paralyn_generated_grid = (1);
  const dim3 __paralyn_generated_block = (8);
  const std::size_t __paralyn_generated_shared = (0);
  const cudaStream_t __paralyn_generated_stream = (0);
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  const int* __paralyn_generated_argument_0 = (signed_device);
  unsigned int* __paralyn_generated_argument_1 = (unsigned_output_device);
  int __paralyn_generated_argument_2 = (n);
  paralyn::launch_checked(__paralyn_generated_kernel_signed_to_unsigned(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_buffer(__paralyn_generated_argument_1), paralyn::Argument::from_i32(__paralyn_generated_argument_2)});
}());
    check(cudaGetLastError());
    check(cudaMemcpy(signed_actual.data(), signed_output_device, bytes, cudaMemcpyDeviceToHost));
    check(
        cudaMemcpy(unsigned_actual.data(), unsigned_output_device, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
      signed_expected[i] = static_cast<int>(unsigned_input[i]);
      unsigned_expected[i] = static_cast<unsigned int>(signed_input[i]);
    }
    require(signed_actual == signed_expected, "u32 to i32 boundary conversion mismatch");
    require(unsigned_actual == unsigned_expected, "i32 to u32 boundary conversion mismatch");

    std::vector<int> arithmetic = {-1000000, -37,    -1,     0,      29,
                                   1000000,  canary, canary, canary, canary};
    check(cudaMemcpy(signed_device, arithmetic.data(), bytes, cudaMemcpyHostToDevice));
    ([&]() {
  const dim3 __paralyn_generated_grid = (1);
  const dim3 __paralyn_generated_block = (8);
  const std::size_t __paralyn_generated_shared = (0);
  const cudaStream_t __paralyn_generated_stream = (0);
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  const int* __paralyn_generated_argument_0 = (signed_device);
  int* __paralyn_generated_argument_1 = (signed_output_device);
  int __paralyn_generated_argument_2 = (n);
  int __paralyn_generated_argument_3 = (37);
  int __paralyn_generated_argument_4 = (-19);
  paralyn::launch_checked(__paralyn_generated_kernel_signed_affine(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_buffer(__paralyn_generated_argument_1), paralyn::Argument::from_i32(__paralyn_generated_argument_2), paralyn::Argument::from_i32(__paralyn_generated_argument_3), paralyn::Argument::from_i32(__paralyn_generated_argument_4)});
}());
    check(cudaGetLastError());
    check(cudaMemcpy(signed_actual.data(), signed_output_device, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i)
      signed_expected[i] = arithmetic[i] * 37 - 19;
    require(signed_actual == signed_expected, "nonoverflowing signed arithmetic mismatch");
    // The signed comparison must reject every nonnegative thread index.
    ([&]() {
  const dim3 __paralyn_generated_grid = (1);
  const dim3 __paralyn_generated_block = (8);
  const std::size_t __paralyn_generated_shared = (0);
  const cudaStream_t __paralyn_generated_stream = (0);
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  const int* __paralyn_generated_argument_0 = (signed_device);
  int* __paralyn_generated_argument_1 = (signed_output_device);
  int __paralyn_generated_argument_2 = (-1);
  int __paralyn_generated_argument_3 = (1);
  int __paralyn_generated_argument_4 = (0);
  paralyn::launch_checked(__paralyn_generated_kernel_signed_affine(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_buffer(__paralyn_generated_argument_1), paralyn::Argument::from_i32(__paralyn_generated_argument_2), paralyn::Argument::from_i32(__paralyn_generated_argument_3), paralyn::Argument::from_i32(__paralyn_generated_argument_4)});
}());
    check(cudaGetLastError());
    check(cudaMemcpy(signed_actual.data(), signed_output_device, bytes, cudaMemcpyDeviceToHost));
    require(signed_actual == signed_expected, "signed negative bound changed output");
    // Usual arithmetic conversions make -1 an unsigned maximum here.
    ([&]() {
  const dim3 __paralyn_generated_grid = (1);
  const dim3 __paralyn_generated_block = (6);
  const std::size_t __paralyn_generated_shared = (0);
  const cudaStream_t __paralyn_generated_stream = (0);
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  int* __paralyn_generated_argument_0 = (signed_output_device);
  int __paralyn_generated_argument_1 = (-1);
  paralyn::launch_checked(__paralyn_generated_kernel_unsigned_comparison(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_i32(__paralyn_generated_argument_1)});
}());
    check(cudaGetLastError());
    check(cudaMemcpy(signed_actual.data(), signed_output_device, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i)
      signed_expected[i] = 1;
    require(signed_actual == signed_expected, "mixed signed/unsigned comparison mismatch");
    check(cudaFree(signed_device));
    check(cudaFree(signed_output_device));
    check(cudaFree(unsigned_device));
    check(cudaFree(unsigned_output_device));
    std::puts("Verification: PASS integer_semantics launches=5 boundary_casts=12 compared=50");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Verification: FAIL integer_semantics: %s\n", error.what());
    return 1;
  }
}
