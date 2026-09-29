#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

__global__ void guarded_affine(const float *a, const float *b, float *output, int n, float scale,
                               int guard) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    output[i + guard] = a[i + guard] * scale + b[i + guard];
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
      guarded_affine<<<grid, block>>>(da, db, dc, n, scale, guard);
      check(cudaGetLastError());
      // Dependent launches are deliberately queued without a host wait.
      guarded_affine<<<grid, block>>>(dc, db, da, n, next_scale, guard);
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
