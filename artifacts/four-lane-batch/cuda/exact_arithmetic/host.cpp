#include <paralyn/runtime.hpp>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

static const paralyn::Kernel& __paralyn_generated_kernel_guarded_affine() {
  static const paralyn::Kernel kernel = paralyn::Kernel{"guarded_affine",{paralyn::Parameter{"a",paralyn::ScalarType::F32,1,1},paralyn::Parameter{"b",paralyn::ScalarType::F32,1,1},paralyn::Parameter{"output",paralyn::ScalarType::F32,1,0},paralyn::Parameter{"n",paralyn::ScalarType::I32,0,0},paralyn::Parameter{"scale",paralyn::ScalarType::F32,0,0},paralyn::Parameter{"guard",paralyn::ScalarType::I32,0,0},},{paralyn::Statement{paralyn::StmtKind::Let,"i",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Cast,paralyn::ScalarType::I32,"",{paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::U32,"+",{paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::U32,"*",{paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"blockIdx.x",{},10,20},paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"blockDim.x",{},10,33},},10,22},paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"threadIdx.x",{},10,47},},10,35},},10,11},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"",{},0,0},{}},paralyn::Statement{paralyn::StmtKind::If,"",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::Bool,"<",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},11,7},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"n",{},11,11},},11,9},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"",{},0,0},{paralyn::Statement{paralyn::StmtKind::Store,"",paralyn::ScalarType::F32,paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::F32,"+",{paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::F32,"*",{paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"a",{},12,25},paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::I32,"+",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},12,27},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"guard",{},12,31},},12,29},},12,25},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"scale",{},12,40},},12,38},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"b",{},12,48},paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::I32,"+",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},12,50},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"guard",{},12,54},},12,52},},12,48},},12,46},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"output",{},12,5},paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::I32,"+",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},12,12},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"guard",{},12,16},},12,14},},12,5},{}},}},}};
  return kernel;
}


static void check(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}
static std::uint32_t random_bits(std::uint32_t &state) {
  state = state * 1664525u + 1013904223u;
  return state;
}
static bool same_bits(float left, float right) {
  return std::memcmp(&left, &right, sizeof(float)) == 0;
}

int main() {
  try {
    // Zero-length allocation/copies are checked independently of the guarded
    // n=0 GPU launch below, which must execute without modifying a canary.
    float *zero = nullptr;
    check(cudaMalloc(&zero, 0));
    if (zero != nullptr)
      throw std::runtime_error("zero allocation did not return null");
    check(cudaMemcpy(nullptr, nullptr, 0, cudaMemcpyHostToDevice));
    check(cudaMemcpy(nullptr, nullptr, 0, cudaMemcpyDeviceToHost));
    check(cudaFree(zero));
    const int sizes[] = {0,   1,   2,   31,  32,  33,  63,   64,   65,   127,  128,    129,
                         255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 4099, 1000003};
    const int blocks[] = {1, 7, 32, 64, 128, 256};
    const float scales[] = {0.5f, 2.0f, -1.0f};
    constexpr int guard = 16;
    constexpr float canary = -8191.25f;
    std::uint32_t random_state = 0x9e3779b9u;
    std::size_t compared = 0;
    int cases = 0;
    for (int n : sizes) {
      const int block = blocks[cases % 6];
      const float scale = scales[cases % 3];
      const float next_scale = scales[(cases + 1) % 3];
      const int count = n + 2 * guard;
      const std::size_t bytes = std::size_t(count) * sizeof(float);
      std::vector<float> a(count, canary), b(count, canary), output(count, canary);
      std::vector<float> first(count, canary), second(count, canary);
      for (int i = guard; i < guard + n; ++i) {
        a[i] = float(int(random_bits(random_state) % 511u) - 255) * 0.25f;
        b[i] = float(int(random_bits(random_state) % 255u) - 127) * 0.5f;
        first[i] = a[i] * scale + b[i];
        second[i] = first[i] * next_scale + b[i];
      }
      float *da = nullptr, *db = nullptr, *dc = nullptr;
      check(cudaMalloc(&da, bytes));
      check(cudaMalloc(&db, bytes));
      check(cudaMalloc(&dc, bytes));
      check(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
      check(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
      check(cudaMemcpy(dc, output.data(), bytes, cudaMemcpyHostToDevice));
      const int grid = n == 0 ? 1 : (n + block - 1) / block;
      ([&]() {
  const dim3 __paralyn_generated_grid = (grid);
  const dim3 __paralyn_generated_block = (block);
  const std::size_t __paralyn_generated_shared = (0);
  const cudaStream_t __paralyn_generated_stream = (0);
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  const float* __paralyn_generated_argument_0 = (da);
  const float* __paralyn_generated_argument_1 = (db);
  float* __paralyn_generated_argument_2 = (dc);
  int __paralyn_generated_argument_3 = (n);
  float __paralyn_generated_argument_4 = (scale);
  int __paralyn_generated_argument_5 = (guard);
  paralyn::launch_checked(__paralyn_generated_kernel_guarded_affine(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_buffer(__paralyn_generated_argument_1), paralyn::Argument::from_buffer(__paralyn_generated_argument_2), paralyn::Argument::from_i32(__paralyn_generated_argument_3), paralyn::Argument::from_f32(__paralyn_generated_argument_4), paralyn::Argument::from_i32(__paralyn_generated_argument_5)});
}());
      check(cudaGetLastError());
      // Dependent launches are deliberately queued without a host wait.
      ([&]() {
  const dim3 __paralyn_generated_grid = (grid);
  const dim3 __paralyn_generated_block = (block);
  const std::size_t __paralyn_generated_shared = (0);
  const cudaStream_t __paralyn_generated_stream = (0);
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  const float* __paralyn_generated_argument_0 = (dc);
  const float* __paralyn_generated_argument_1 = (db);
  float* __paralyn_generated_argument_2 = (da);
  int __paralyn_generated_argument_3 = (n);
  float __paralyn_generated_argument_4 = (next_scale);
  int __paralyn_generated_argument_5 = (guard);
  paralyn::launch_checked(__paralyn_generated_kernel_guarded_affine(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_buffer(__paralyn_generated_argument_1), paralyn::Argument::from_buffer(__paralyn_generated_argument_2), paralyn::Argument::from_i32(__paralyn_generated_argument_3), paralyn::Argument::from_f32(__paralyn_generated_argument_4), paralyn::Argument::from_i32(__paralyn_generated_argument_5)});
}());
      check(cudaGetLastError());
      // Blocking readback must establish completion and queue ordering.
      check(cudaMemcpy(output.data(), dc, bytes, cudaMemcpyDeviceToHost));
      check(cudaMemcpy(a.data(), da, bytes, cudaMemcpyDeviceToHost));
      for (int i = 0; i < count; ++i) {
        if (!same_bits(output[i], first[i]) || !same_bits(a[i], second[i])) {
          std::fprintf(stderr, "Exact arithmetic/canary mismatch n=%d block=%d i=%d\n", n, block,
                       i);
          return 1;
        }
        compared += 2;
      }
      check(cudaFree(da));
      check(cudaFree(db));
      check(cudaFree(dc));
      ++cases;
    }
    check(cudaDeviceSynchronize());
    std::printf(
        "Verification: PASS exact_arithmetic cases=%d launches=%d compared=%zu seed=2654435769\n",
        cases, cases * 2, compared);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Verification: FAIL exact_arithmetic: %s\n", error.what());
    return 1;
  }
}
