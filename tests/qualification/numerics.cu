#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

__global__ void numeric_add(const float *a, const float *b, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] + b[i];
}
__global__ void numeric_mul(const float *a, const float *b, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] * b[i];
}
__global__ void numeric_separate(const float *a, const float *b, const float *c, float *out,
                                 int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] * b[i] + c[i];
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
        numeric_mul<<<(n + 127) / 128, 128>>>(da, db, dout, n);
      else
        numeric_add<<<(n + 127) / 128, 128>>>(da, db, dout, n);
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
    numeric_separate<<<1, 32>>>(da, db, dc, dout, special_n);
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
