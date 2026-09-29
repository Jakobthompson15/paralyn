#include <cuda_runtime.h>
__global__ void unused(float* p) { p[threadIdx.x]=1.0f; }
int main() { cudaFree(reinterpret_cast<void*>(1)); return 0; }
