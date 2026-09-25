#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}
} // namespace
int main() {
  try {
    constexpr int n = 1024;
    std::vector<float> a(n), b(n), actual(n, -9999.0f);
    for (int i = 0; i < n; ++i) {
      a[i] = float((i % 31) - 15) * 0.5f;
      b[i] = float((i % 17) - 8) * 0.25f;
    }
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    require(cudaMalloc(&da, n * sizeof(float)));
    require(cudaMalloc(&db, n * sizeof(float)));
    require(cudaMalloc(&dc, n * sizeof(float)));
    require(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    require(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    require(cudaMemcpy(dc, actual.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    using namespace unicuda;
    Kernel signature{"runtime_smoke",
                     {{"a", ScalarType::F32, true, true},
                      {"b", ScalarType::F32, true, true},
                      {"c", ScalarType::F32, true, false},
                      {"n", ScalarType::I32, false, false}},
                     {}};
    const char *handwritten = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void runtime_smoke(const device float* a [[buffer(0)]],
                          const device float* b [[buffer(1)]],
                          device float* c [[buffer(2)]],
                          constant int& n [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
  if (i < uint(n)) c[i] = a[i] + b[i];
}
)MSL";
    launch_msl_for_test(signature, handwritten, {4, 1, 1}, {256, 1, 1},
                        {Argument::from_buffer(da), Argument::from_buffer(db),
                         Argument::from_buffer(dc), Argument::from_i32(n)});
    require(cudaDeviceSynchronize());
    require(cudaMemcpy(actual.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
      if (actual[i] != a[i] + b[i])
        throw std::runtime_error("Handwritten Metal result differs from CPU reference at index " +
                                 std::to_string(i));
    }
    if (cudaMemcpy(actual.data(), dc, (n + 1) * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaErrorInvalidValue)
      throw std::runtime_error("Oversized copy was not rejected");
    if (cudaGetLastError() != cudaErrorInvalidValue || cudaGetLastError() != cudaSuccess)
      throw std::runtime_error("Last-error reset failed");
    require(cudaFree(da));
    require(cudaFree(db));
    require(cudaFree(dc));
    if (cudaFree(dc) != cudaErrorInvalidDevicePointer)
      throw std::runtime_error("Repeated free was not rejected");
    cudaGetLastError();
    shutdown();
    std::cout << "Handwritten Metal smoke: " << n << " values matched the CPU reference\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Runtime smoke failed: " << error.what() << '\n';
    return 1;
  }
}
