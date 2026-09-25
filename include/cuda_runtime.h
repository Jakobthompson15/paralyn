#pragma once
#include "paralyn/runtime.hpp"
#include <cstddef>
#include <cstdint>

struct uint3 {
  unsigned int x, y, z;
};
struct dim3 {
  unsigned int x, y, z;
  constexpr dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1)
      : x(x_), y(y_), z(z_) {}
  constexpr dim3(uint3 value) : x(value.x), y(value.y), z(value.z) {}
  constexpr operator paralyn::Dim3() const { return {x, y, z}; }
};
using cudaStream_t = void *;
enum cudaError_t {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInitializationError = 3,
  cudaErrorInvalidDevice = 10,
  cudaErrorInvalidDevicePointer = 17,
  cudaErrorInvalidMemcpyDirection = 21,
  cudaErrorLaunchFailure = 719,
  cudaErrorNotSupported = 801,
  cudaErrorUnknown = 999
};
enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
};
extern "C" {
cudaError_t cudaMalloc(void **pointer, std::size_t bytes);
cudaError_t cudaFree(void *pointer);
cudaError_t cudaMemcpy(void *destination, const void *source, std::size_t bytes,
                       cudaMemcpyKind kind);
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaGetLastError();
const char *cudaGetErrorString(cudaError_t error);
}
template <class T> inline cudaError_t cudaMalloc(T **pointer, std::size_t bytes) {
  return cudaMalloc(reinterpret_cast<void **>(pointer), bytes);
}
