#include <paralyn/runtime.hpp>
#include <cuda_runtime.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

static const paralyn::Kernel& __paralyn_generated_kernel_vector_add() {
  static const paralyn::Kernel kernel = paralyn::Kernel{"vector_add",{paralyn::Parameter{"a",paralyn::ScalarType::F32,1,1},paralyn::Parameter{"b",paralyn::ScalarType::F32,1,1},paralyn::Parameter{"c",paralyn::ScalarType::F32,1,0},paralyn::Parameter{"n",paralyn::ScalarType::I32,0,0},},{paralyn::Statement{paralyn::StmtKind::Let,"i",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Cast,paralyn::ScalarType::I32,"",{paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::U32,"+",{paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::U32,"*",{paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"blockIdx.x",{},6,22},paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"blockDim.x",{},6,35},},6,24},paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"threadIdx.x",{},6,49},},6,37},},6,13},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"",{},0,0},{}},paralyn::Statement{paralyn::StmtKind::If,"",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::Bool,"<",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},7,9},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"n",{},7,13},},7,11},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"",{},0,0},{paralyn::Statement{paralyn::StmtKind::Store,"",paralyn::ScalarType::F32,paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::F32,"+",{paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"a",{},8,16},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},8,18},},8,16},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"b",{},8,23},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},8,25},},8,23},},8,21},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"c",{},8,9},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},8,11},},8,9},{}},}},}};
  return kernel;
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
    ([&]() {
  const dim3 __paralyn_generated_grid = (grid);
  const dim3 __paralyn_generated_block = (block);
  const std::size_t __paralyn_generated_shared = (0);
  const cudaStream_t __paralyn_generated_stream = (0);
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  const float* __paralyn_generated_argument_0 = (device_a);
  const float* __paralyn_generated_argument_1 = (device_b);
  float* __paralyn_generated_argument_2 = (device_c);
  int __paralyn_generated_argument_3 = (n);
  paralyn::launch_checked(__paralyn_generated_kernel_vector_add(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_buffer(__paralyn_generated_argument_1), paralyn::Argument::from_buffer(__paralyn_generated_argument_2), paralyn::Argument::from_i32(__paralyn_generated_argument_3)});
}());
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
