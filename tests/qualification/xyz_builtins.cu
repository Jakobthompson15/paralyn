#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

__global__ void record_coordinates(int *output, int guard) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int width = gridDim.x * blockDim.x;
  int height = gridDim.y * blockDim.y;
  int base = ((z * height + y) * width + x) * 12 + guard;
  output[base + 0] = threadIdx.x;
  output[base + 1] = threadIdx.y;
  output[base + 2] = threadIdx.z;
  output[base + 3] = blockIdx.x;
  output[base + 4] = blockIdx.y;
  output[base + 5] = blockIdx.z;
  output[base + 6] = blockDim.x;
  output[base + 7] = blockDim.y;
  output[base + 8] = blockDim.z;
  output[base + 9] = gridDim.x;
  output[base + 10] = gridDim.y;
  output[base + 11] = gridDim.z;
}
static void check(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}
int main() {
  try {
    const dim3 grids[] = {dim3(2, 3, 2), dim3(3, 2, 4), dim3(1, 1, 1)};
    const dim3 blocks[] = {dim3(3, 2, 2), dim3(5, 3, 2), dim3(2, 3, 5)};
    constexpr int guard = 16, canary = -12345678;
    std::size_t compared = 0;
    for (int test = 0; test < 3; ++test) {
      dim3 grid = grids[test], block = blocks[test];
      const int width = grid.x * block.x, height = grid.y * block.y, depth = grid.z * block.z;
      const int count = width * height * depth * 12 + 2 * guard;
      std::vector<int> expected(count, canary), actual(count, canary);
      for (int z = 0; z < depth; ++z)
        for (int y = 0; y < height; ++y)
          for (int x = 0; x < width; ++x) {
            const int base = ((z * height + y) * width + x) * 12 + guard;
            const int values[] = {x % int(block.x), y % int(block.y), z % int(block.z),
                                  x / int(block.x), y / int(block.y), z / int(block.z),
                                  int(block.x),     int(block.y),     int(block.z),
                                  int(grid.x),      int(grid.y),      int(grid.z)};
            for (int j = 0; j < 12; ++j)
              expected[base + j] = values[j];
          }
      int *device = nullptr;
      const std::size_t bytes = actual.size() * sizeof(int);
      check(cudaMalloc(&device, bytes));
      check(cudaMemcpy(device, actual.data(), bytes, cudaMemcpyHostToDevice));
      record_coordinates<<<grid, block>>>(device, guard);
      check(cudaGetLastError());
      check(cudaMemcpy(actual.data(), device, bytes, cudaMemcpyDeviceToHost));
      if (actual != expected)
        throw std::runtime_error("xyz builtin value or canary mismatch");
      compared += actual.size();
      check(cudaFree(device));
    }
    std::printf("Verification: PASS xyz_builtins cases=3 launches=3 compared=%zu\n", compared);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Verification: FAIL xyz_builtins: %s\n", error.what());
    return 1;
  }
}
