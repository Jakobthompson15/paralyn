#include <cstdio>
#include <cuda_runtime.h>

// Compile with `paralyn compile kernels.cu --output kernels.plyn`.
// This is the existing CUDA frontend feeding the native module API; it does not
// introduce a new kernel language or require CUDA APIs in native applications.
__global__ void vector_add(const float *a, const float *b, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

__global__ void affine(const float *a, const float *b, float *out, int n, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] * scale + b[i];
  }
}

int main() {
  std::puts("ERROR: native module compilation must never execute this host main");
  return 93;
}
