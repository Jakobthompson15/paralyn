#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

static bool check(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return true;
    std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(error));
    return false;
}

int main() {
    // Canonical demonstration values; the compiler/runtime must not special-case them.
    const int n = 1024;
    const std::size_t bytes = n * sizeof(float);
    std::vector<float> a(n), b(n), c(n, -9999.0f);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i % 97) * 0.25f;
        b[i] = static_cast<float>((i % 31) - 15) * 0.5f;
    }
    float *device_a = nullptr, *device_b = nullptr, *device_c = nullptr;
    auto cleanup = [&]() {
        bool ok = check(cudaFree(device_a), "cudaFree(a)");
        ok = check(cudaFree(device_b), "cudaFree(b)") && ok;
        ok = check(cudaFree(device_c), "cudaFree(c)") && ok;
        return ok;
    };
    if (!check(cudaMalloc(&device_a, bytes), "cudaMalloc(a)") ||
        !check(cudaMalloc(&device_b, bytes), "cudaMalloc(b)") ||
        !check(cudaMalloc(&device_c, bytes), "cudaMalloc(c)")) {
        cleanup(); return 1;
    }
    if (!check(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice), "copy a") ||
        !check(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice), "copy b") ||
        !check(cudaMemcpy(device_c, c.data(), bytes, cudaMemcpyHostToDevice), "initialize c")) {
        cleanup(); return 1;
    }
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    vector_add<<<grid, block>>>(device_a, device_b, device_c, n);
    if (!check(cudaGetLastError(), "kernel launch") ||
        !check(cudaDeviceSynchronize(), "GPU synchronization") ||
        !check(cudaMemcpy(c.data(), device_c, bytes, cudaMemcpyDeviceToHost), "read c")) {
        cleanup(); return 1;
    }
    // Independent host verification after the GPU has finished; not a runtime fallback.
    for (int i = 0; i < n; ++i) {
        const float expected = a[i] + b[i];
        if (c[i] != expected) {
            std::fprintf(stderr, "Verification: FAIL at %d (GPU=%g, CPU=%g)\n", i, c[i], expected);
            cleanup(); return 1;
        }
    }
    if (!cleanup()) return 1;
    std::printf("\nVerification: PASS (%d independently checked elements)\n", n);
    return 0;
}
