#pragma once
// Standalone Clang CUDA declarations: no NVIDIA headers or toolkit are required.
#ifndef __host__
#define __host__ __attribute__((host))
#endif
#ifndef __device__
#define __device__ __attribute__((device))
#endif
#ifndef __global__
#define __global__ __attribute__((global))
#endif
#ifndef __shared__
#define __shared__ __attribute__((shared))
#endif
#ifndef __constant__
#define __constant__ __attribute__((constant))
#endif
#include <cuda_runtime.h>
extern const __device__ uint3 threadIdx;
extern const __device__ uint3 blockIdx;
extern const __device__ uint3 blockDim;
extern const __device__ uint3 gridDim;
extern "C" cudaError_t cudaConfigureCall(dim3 grid, dim3 block, std::size_t shared = 0,
                                         cudaStream_t stream = nullptr);
extern "C" cudaError_t cudaSetupArgument(const void *, std::size_t, std::size_t);
extern "C" cudaError_t cudaLaunch(const void *);
