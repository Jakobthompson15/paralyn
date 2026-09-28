#include <cuda_runtime.h>

__global__ void array_add(const float *a, const float *b, float *out, unsigned int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

__global__ void array_affine(const float *a, const float *b, float *out, unsigned int n,
                             float scale) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] * scale + b[i];
  }
}
