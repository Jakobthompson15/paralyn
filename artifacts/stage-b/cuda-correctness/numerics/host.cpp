#include <paralyn/runtime.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

static const paralyn::Kernel& __paralyn_generated_kernel_numeric_add() {
  static const paralyn::Kernel kernel = paralyn::Kernel{"numeric_add",{paralyn::Parameter{"a",paralyn::ScalarType::F32,1,1},paralyn::Parameter{"b",paralyn::ScalarType::F32,1,1},paralyn::Parameter{"out",paralyn::ScalarType::F32,1,0},paralyn::Parameter{"n",paralyn::ScalarType::I32,0,0},},{paralyn::Statement{paralyn::StmtKind::Let,"i",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Cast,paralyn::ScalarType::I32,"",{paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::U32,"+",{paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::U32,"*",{paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"blockIdx.x",{},10,20},paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"blockDim.x",{},10,33},},10,22},paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"threadIdx.x",{},10,47},},10,35},},10,11},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"",{},0,0},{}},paralyn::Statement{paralyn::StmtKind::If,"",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::Bool,"<",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},11,7},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"n",{},11,11},},11,9},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"",{},0,0},{paralyn::Statement{paralyn::StmtKind::Store,"",paralyn::ScalarType::F32,paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::F32,"+",{paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"a",{},12,14},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},12,16},},12,14},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"b",{},12,21},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},12,23},},12,21},},12,19},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"out",{},12,5},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},12,9},},12,5},{}},}},}};
  return kernel;
}

static const paralyn::Kernel& __paralyn_generated_kernel_numeric_mul() {
  static const paralyn::Kernel kernel = paralyn::Kernel{"numeric_mul",{paralyn::Parameter{"a",paralyn::ScalarType::F32,1,1},paralyn::Parameter{"b",paralyn::ScalarType::F32,1,1},paralyn::Parameter{"out",paralyn::ScalarType::F32,1,0},paralyn::Parameter{"n",paralyn::ScalarType::I32,0,0},},{paralyn::Statement{paralyn::StmtKind::Let,"i",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Cast,paralyn::ScalarType::I32,"",{paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::U32,"+",{paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::U32,"*",{paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"blockIdx.x",{},15,20},paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"blockDim.x",{},15,33},},15,22},paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"threadIdx.x",{},15,47},},15,35},},15,11},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"",{},0,0},{}},paralyn::Statement{paralyn::StmtKind::If,"",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::Bool,"<",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},16,7},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"n",{},16,11},},16,9},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"",{},0,0},{paralyn::Statement{paralyn::StmtKind::Store,"",paralyn::ScalarType::F32,paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::F32,"*",{paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"a",{},17,14},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},17,16},},17,14},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"b",{},17,21},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},17,23},},17,21},},17,19},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"out",{},17,5},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},17,9},},17,5},{}},}},}};
  return kernel;
}

static const paralyn::Kernel& __paralyn_generated_kernel_numeric_separate() {
  static const paralyn::Kernel kernel = paralyn::Kernel{"numeric_separate",{paralyn::Parameter{"a",paralyn::ScalarType::F32,1,1},paralyn::Parameter{"b",paralyn::ScalarType::F32,1,1},paralyn::Parameter{"c",paralyn::ScalarType::F32,1,1},paralyn::Parameter{"out",paralyn::ScalarType::F32,1,0},paralyn::Parameter{"n",paralyn::ScalarType::I32,0,0},},{paralyn::Statement{paralyn::StmtKind::Let,"i",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Cast,paralyn::ScalarType::I32,"",{paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::U32,"+",{paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::U32,"*",{paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"blockIdx.x",{},21,20},paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"blockDim.x",{},21,33},},21,22},paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"threadIdx.x",{},21,47},},21,35},},21,11},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"",{},0,0},{}},paralyn::Statement{paralyn::StmtKind::If,"",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::Bool,"<",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},22,7},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"n",{},22,11},},22,9},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"",{},0,0},{paralyn::Statement{paralyn::StmtKind::Store,"",paralyn::ScalarType::F32,paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::F32,"+",{paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::F32,"*",{paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"a",{},23,14},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},23,16},},23,14},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"b",{},23,21},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},23,23},},23,21},},23,19},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"c",{},23,28},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},23,30},},23,28},},23,26},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"out",{},23,5},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},23,9},},23,5},{}},}},}};
  return kernel;
}

static float from_bits(std::uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
static std::uint32_t bits(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
static std::uint32_t order(float f) {
  auto u = bits(f);
  return u & 0x80000000u ? ~u : u | 0x80000000u;
}
static void check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
static float ref(float a, float b, bool multiply) {
  volatile float result = multiply ? a * b : a + b;
  return result;
}
static float flush(float f) {
  return std::fpclassify(f) == FP_SUBNORMAL ? std::copysign(0.0f, f) : f;
}
static bool same(float actual, float expected) {
  if (std::isnan(expected))
    return std::isnan(actual);
  return bits(actual) == bits(expected);
}
int main() {
  try {
    const int n = 8192;
    std::vector<float> a(n), b(n), c(n), out(n);
    std::uint32_t state = 0x39284912u;
    auto random = [&]() {
      state = state * 1664525u + 1013904223u;
      return state;
    };
    for (int i = 0; i < n; ++i) {
      auto x = random(), y = random();
      a[i] = from_bits((x & 0x807fffffu) | ((120u + (x >> 24) % 12u) << 23));
      b[i] = from_bits((y & 0x807fffffu) | ((120u + (y >> 24) % 12u) << 23));
      c[i] = -1.0f;
    }
    // Host-side bit fixtures independently specify exceptional and rounding inputs.
    const std::uint32_t special[][2] = {{0u, 0u},
                                        {0x80000000u, 0x80000000u},
                                        {0u, 0x80000000u},
                                        {0x7f800000u, 0x3f800000u},
                                        {0xff800000u, 0x3f800000u},
                                        {0x7f800000u, 0xff800000u},
                                        {0x7fc01234u, 0x3f800000u},
                                        {0x7f800000u, 0u},
                                        {1u, 0x3f800000u},
                                        {0x80000001u, 0x3f800000u},
                                        {0x00800000u, 0x3f000000u},
                                        {0x80800000u, 0x3f000000u},
                                        {0x7f7fffffu, 0x40000000u},
                                        {0x3f800001u, 0x3f7ffffeu}};
    const int special_n = sizeof(special) / sizeof(special[0]);
    for (int i = 0; i < special_n; ++i) {
      a[i] = from_bits(special[i][0]);
      b[i] = from_bits(special[i][1]);
    }
    float *da = nullptr, *db = nullptr, *dc = nullptr, *dout = nullptr;
    check(cudaMalloc(&da, n * 4));
    check(cudaMalloc(&db, n * 4));
    check(cudaMalloc(&dc, n * 4));
    check(cudaMalloc(&dout, n * 4));
    check(cudaMemcpy(da, a.data(), n * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(db, b.data(), n * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dc, c.data(), n * 4, cudaMemcpyHostToDevice));
    unsigned max_ulp = 0, subnormal_variations = 0;
    for (int op = 0; op < 2; ++op) {
      if (op)
        ([&]() {
  const dim3 __paralyn_generated_grid = ((n + 127) / 128);
  const dim3 __paralyn_generated_block = (128);
  const std::size_t __paralyn_generated_shared = (0);
  const cudaStream_t __paralyn_generated_stream = (0);
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  const float* __paralyn_generated_argument_0 = (da);
  const float* __paralyn_generated_argument_1 = (db);
  float* __paralyn_generated_argument_2 = (dout);
  int __paralyn_generated_argument_3 = (n);
  paralyn::launch_checked(__paralyn_generated_kernel_numeric_mul(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_buffer(__paralyn_generated_argument_1), paralyn::Argument::from_buffer(__paralyn_generated_argument_2), paralyn::Argument::from_i32(__paralyn_generated_argument_3)});
}());
      else
        ([&]() {
  const dim3 __paralyn_generated_grid = ((n + 127) / 128);
  const dim3 __paralyn_generated_block = (128);
  const std::size_t __paralyn_generated_shared = (0);
  const cudaStream_t __paralyn_generated_stream = (0);
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  const float* __paralyn_generated_argument_0 = (da);
  const float* __paralyn_generated_argument_1 = (db);
  float* __paralyn_generated_argument_2 = (dout);
  int __paralyn_generated_argument_3 = (n);
  paralyn::launch_checked(__paralyn_generated_kernel_numeric_add(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_buffer(__paralyn_generated_argument_1), paralyn::Argument::from_buffer(__paralyn_generated_argument_2), paralyn::Argument::from_i32(__paralyn_generated_argument_3)});
}());
      check(cudaGetLastError());
      check(cudaDeviceSynchronize());
      check(cudaMemcpy(out.data(), dout, n * 4, cudaMemcpyDeviceToHost));
      for (int i = 0; i < n; ++i) {
        float expected = ref(a[i], b[i], op != 0);
        bool sub = std::fpclassify(a[i]) == FP_SUBNORMAL || std::fpclassify(b[i]) == FP_SUBNORMAL ||
                   std::fpclassify(expected) == FP_SUBNORMAL;
        bool ok = false;
        if (sub) {
          for (int mask = 0; mask < 4; ++mask) {
            float candidate =
                ref(mask & 1 ? flush(a[i]) : a[i], mask & 2 ? flush(b[i]) : b[i], op != 0);
            // Metal permits either sign when a subnormal operand/result is flushed.
            bool flushed_zero = out[i] == 0.0f && flush(candidate) == 0.0f;
            ok = ok || same(out[i], candidate) || same(out[i], flush(candidate)) || flushed_zero;
          }
          if (!same(out[i], expected))
            ++subnormal_variations;
        } else if (std::isfinite(expected) && expected != 0.0f && std::isfinite(out[i])) {
          auto x = order(expected), y = order(out[i]);
          unsigned ulp = x > y ? x - y : y - x;
          if (ulp > max_ulp)
            max_ulp = ulp;
          ok = ulp <= 1;
        } else
          ok = same(out[i], expected);
        if (!ok) {
          std::fprintf(stderr, "numeric mismatch op=%d index=%d got=0x%08x expected=0x%08x\n", op,
                       i, bits(out[i]), bits(expected));
          return 1;
        }
      }
    }
    ([&]() {
  const dim3 __paralyn_generated_grid = (1);
  const dim3 __paralyn_generated_block = (32);
  const std::size_t __paralyn_generated_shared = (0);
  const cudaStream_t __paralyn_generated_stream = (0);
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  const float* __paralyn_generated_argument_0 = (da);
  const float* __paralyn_generated_argument_1 = (db);
  const float* __paralyn_generated_argument_2 = (dc);
  float* __paralyn_generated_argument_3 = (dout);
  int __paralyn_generated_argument_4 = (special_n);
  paralyn::launch_checked(__paralyn_generated_kernel_numeric_separate(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_buffer(__paralyn_generated_argument_1), paralyn::Argument::from_buffer(__paralyn_generated_argument_2), paralyn::Argument::from_buffer(__paralyn_generated_argument_3), paralyn::Argument::from_i32(__paralyn_generated_argument_4)});
}());
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(out.data(), dout, n * 4, cudaMemcpyDeviceToHost));
    int probe = special_n - 1;
    float rounded_product = ref(a[probe], b[probe], true);
    float separate = ref(rounded_product, c[probe], false);
    float fused = std::fma(a[probe], b[probe], c[probe]);
    if (same(separate, fused) || !same(out[probe], separate)) {
      std::fprintf(stderr, "contraction probe failed: actual=%08x separate=%08x fused=%08x\n",
                   bits(out[probe]), bits(separate), bits(fused));
      return 1;
    }
    check(cudaFree(da));
    check(cudaFree(db));
    check(cudaFree(dc));
    check(cudaFree(dout));
    std::printf("Numerics: normal_max_ulp=%u subnormal_flush_variations=%u contraction=off "
                "signed_zero=preserved exceptional_classes=matched\n",
                max_ulp, subnormal_variations);
    std::puts(
        "Verification: PASS (FP32 add/multiply, exceptional values, subnormals and contraction)");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "Numerics failed: %s\n", e.what());
    return 1;
  }
}
