#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <vector>

__global__ void signed_affine(const int *input, int *output, int n, int scale, int bias) {
  int i = threadIdx.x;
  if (i < n)
    output[i] = input[i] * scale + bias;
}
__global__ void unsigned_to_signed(const unsigned int *input, int *output, int n) {
  int i = threadIdx.x;
  if (i < n)
    output[i] = static_cast<int>(input[i]);
}
__global__ void signed_to_unsigned(const int *input, unsigned int *output, int n) {
  int i = threadIdx.x;
  if (i < n)
    output[i] = static_cast<unsigned int>(input[i]);
}
__global__ void unsigned_comparison(int *output, int bound) {
  if (threadIdx.x < bound)
    output[threadIdx.x] = 1;
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
    unsigned_to_signed<<<1, 8>>>(unsigned_device, signed_output_device, n);
    check(cudaGetLastError());
    signed_to_unsigned<<<1, 8>>>(signed_device, unsigned_output_device, n);
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
    signed_affine<<<1, 8>>>(signed_device, signed_output_device, n, 37, -19);
    check(cudaGetLastError());
    check(cudaMemcpy(signed_actual.data(), signed_output_device, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i)
      signed_expected[i] = arithmetic[i] * 37 - 19;
    require(signed_actual == signed_expected, "nonoverflowing signed arithmetic mismatch");
    // The signed comparison must reject every nonnegative thread index.
    signed_affine<<<1, 8>>>(signed_device, signed_output_device, -1, 1, 0);
    check(cudaGetLastError());
    check(cudaMemcpy(signed_actual.data(), signed_output_device, bytes, cudaMemcpyDeviceToHost));
    require(signed_actual == signed_expected, "signed negative bound changed output");
    // Usual arithmetic conversions make -1 an unsigned maximum here.
    unsigned_comparison<<<1, 6>>>(signed_output_device, -1);
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
