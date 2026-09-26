#include <paralyn/runtime.hpp>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

static const paralyn::Kernel& __paralyn_generated_kernel_alias_steps() {
  static const paralyn::Kernel kernel = paralyn::Kernel{"alias_steps",{paralyn::Parameter{"a",paralyn::ScalarType::F32,1,0},paralyn::Parameter{"b",paralyn::ScalarType::F32,1,0},paralyn::Parameter{"c",paralyn::ScalarType::F32,1,0},paralyn::Parameter{"n",paralyn::ScalarType::I32,0,0},paralyn::Parameter{"scale",paralyn::ScalarType::F32,0,0},},{paralyn::Statement{paralyn::StmtKind::Let,"i",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Cast,paralyn::ScalarType::I32,"",{paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::U32,"+",{paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::U32,"*",{paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"blockIdx.x",{},9,20},paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"blockDim.x",{},9,33},},9,22},paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"threadIdx.x",{},9,47},},9,35},},9,11},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"",{},0,0},{}},paralyn::Statement{paralyn::StmtKind::If,"",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::Bool,"<",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},10,7},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"n",{},10,11},},10,9},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"",{},0,0},{paralyn::Statement{paralyn::StmtKind::Store,"",paralyn::ScalarType::F32,paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::F32,"+",{paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"a",{},11,12},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},11,14},},11,12},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"b",{},11,19},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},11,21},},11,19},},11,17},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"a",{},11,5},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},11,7},},11,5},{}},paralyn::Statement{paralyn::StmtKind::Store,"",paralyn::ScalarType::F32,paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::F32,"+",{paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::F32,"*",{paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"a",{},12,12},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},12,14},},12,12},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"scale",{},12,19},},12,17},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"b",{},12,27},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},12,29},},12,27},},12,25},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"c",{},12,5},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},12,7},},12,5},{}},}},}};
  return kernel;
}

static const paralyn::Kernel& __paralyn_generated_kernel_const_alias() {
  static const paralyn::Kernel kernel = paralyn::Kernel{"const_alias",{paralyn::Parameter{"input",paralyn::ScalarType::F32,1,1},paralyn::Parameter{"output",paralyn::ScalarType::F32,1,0},paralyn::Parameter{"n",paralyn::ScalarType::I32,0,0},},{paralyn::Statement{paralyn::StmtKind::Let,"i",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Cast,paralyn::ScalarType::I32,"",{paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::U32,"+",{paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::U32,"*",{paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"blockIdx.x",{},16,20},paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"blockDim.x",{},16,33},},16,22},paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"threadIdx.x",{},16,47},},16,35},},16,11},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"",{},0,0},{}},paralyn::Statement{paralyn::StmtKind::If,"",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::Bool,"<",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},17,7},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"n",{},17,11},},17,9},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"",{},0,0},{paralyn::Statement{paralyn::StmtKind::Store,"",paralyn::ScalarType::F32,paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::F32,"+",{paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"input",{},18,17},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},18,23},},18,17},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::F32,"1.0E+0",{},18,28},},18,26},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"output",{},18,5},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},18,12},},18,5},{}},paralyn::Statement{paralyn::StmtKind::Store,"",paralyn::ScalarType::F32,paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::F32,"+",{paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"input",{},19,17},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},19,23},},19,17},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::F32,"2.0E+0",{},19,28},},19,26},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::F32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::F32,"output",{},19,5},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},19,12},},19,5},{}},}},}};
  return kernel;
}

static const paralyn::Kernel& __paralyn_generated_kernel_mixed_pointees() {
  static const paralyn::Kernel kernel = paralyn::Kernel{"mixed_pointees",{paralyn::Parameter{"a",paralyn::ScalarType::F32,1,0},paralyn::Parameter{"b",paralyn::ScalarType::I32,1,0},},{paralyn::Statement{paralyn::StmtKind::Store,"",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"1",{},22,59},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::I32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"b",{},22,52},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"0",{},22,54},},22,52},{}},}};
  return kernel;
}

static const paralyn::Kernel& __paralyn_generated_kernel_unsigned_values() {
  static const paralyn::Kernel kernel = paralyn::Kernel{"unsigned_values",{paralyn::Parameter{"output",paralyn::ScalarType::U32,1,0},paralyn::Parameter{"bias",paralyn::ScalarType::U32,0,0},},{paralyn::Statement{paralyn::StmtKind::Let,"i",paralyn::ScalarType::I32,paralyn::Expr{paralyn::ExprKind::Cast,paralyn::ScalarType::I32,"",{paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"threadIdx.x",{},24,21},},24,11},paralyn::Expr{paralyn::ExprKind::Literal,paralyn::ScalarType::I32,"",{},0,0},{}},paralyn::Statement{paralyn::StmtKind::Store,"",paralyn::ScalarType::U32,paralyn::Expr{paralyn::ExprKind::Binary,paralyn::ScalarType::U32,"+",{paralyn::Expr{paralyn::ExprKind::Builtin,paralyn::ScalarType::U32,"threadIdx.x",{},25,25},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::U32,"bias",{},25,29},},25,27},paralyn::Expr{paralyn::ExprKind::Load,paralyn::ScalarType::U32,"",{paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::U32,"output",{},25,3},paralyn::Expr{paralyn::ExprKind::Ref,paralyn::ScalarType::I32,"i",{},25,10},},25,3},{}},}};
  return kernel;
}


static void check(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}
static void require(bool condition, const char *text) {
  if (!condition)
    throw std::runtime_error(text);
}
static void expect_error(cudaError_t actual, cudaError_t expected) {
  require(actual == expected, "runtime returned wrong error category");
  require(cudaGetLastError() == expected, "last-error value differs from API error");
  require(cudaGetLastError() == cudaSuccess, "last error did not reset");
}
static void expect_launch_error(cudaError_t expected) {
  require(cudaGetLastError() == expected, "launch returned wrong error category");
  require(cudaGetLastError() == cudaSuccess, "launch error did not reset");
}
static int configurations[4] = {0, 0, 0, 0};
static int arguments[5] = {0, 0, 0, 0, 0};
static bool ordering_valid = true;
static dim3 grid_expression() {
  ++configurations[0];
  return dim3(1);
}
static dim3 block_expression() {
  ++configurations[1];
  return dim3(7);
}
static std::size_t shared_expression() {
  ++configurations[2];
  return 0;
}
static cudaStream_t stream_expression() {
  ++configurations[3];
  return nullptr;
}
static void argument_seen(int index) {
  for (int count : configurations)
    if (count != 1)
      ordering_valid = false;
  ++arguments[index];
}
static float *pointer_expression(float *value, int index) {
  argument_seen(index);
  return value;
}
static unsigned short count_expression() {
  argument_seen(3);
  return 7;
}
static double scalar_expression() {
  argument_seen(4);
  return 2.25;
}
static int rejected_argument_calls = 0;
static float *rejected_argument(float *value) {
  ++rejected_argument_calls;
  return value;
}
static std::size_t invalid_shared_expression() { return 4; }
static cudaStream_t invalid_stream_expression(float *value) { return static_cast<void *>(value); }

int main(int argc, char **argv) {
  std::puts("HOST_MAIN_EXECUTED");
  try {
    constexpr int n = 19, tail = 13;
    constexpr float canary = -4093.5f;
    const std::size_t bytes = (n + tail) * sizeof(float);
    float *device[3] = {nullptr, nullptr, nullptr};
    for (auto &allocation : device)
      check(cudaMalloc(&allocation, bytes));
    const int layouts[][3] = {{0, 1, 2}, {0, 1, 0}, {0, 0, 2}, {0, 0, 0}, {0, 1, 2}};
    int launches = 0;
    for (int test = 0; test < 5; ++test) {
      std::vector<std::vector<float>> expected(3, std::vector<float>(n + tail, canary));
      for (int buffer = 0; buffer < 3; ++buffer) {
        for (int i = 0; i < n; ++i)
          expected[buffer][i] = float((i + buffer * 7) % 23 - 11) * 0.5f;
        check(cudaMemcpy(device[buffer], expected[buffer].data(), bytes, cudaMemcpyHostToDevice));
      }
      const int a = layouts[test][0], b = layouts[test][1], c = layouts[test][2];
      const float scale = test % 2 ? 0.5f : 2.0f;
      for (int i = 0; i < n; ++i) {
        expected[a][i] = expected[a][i] + expected[b][i];
        expected[c][i] = expected[a][i] * scale + expected[b][i];
      }
      ([&]() {
  const dim3 __paralyn_generated_grid = (3);
  const dim3 __paralyn_generated_block = (7);
  const std::size_t __paralyn_generated_shared = (0);
  const cudaStream_t __paralyn_generated_stream = (0);
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  float* __paralyn_generated_argument_0 = (device[a]);
  float* __paralyn_generated_argument_1 = (device[b]);
  float* __paralyn_generated_argument_2 = (device[c]);
  int __paralyn_generated_argument_3 = (n);
  float __paralyn_generated_argument_4 = (scale);
  paralyn::launch_checked(__paralyn_generated_kernel_alias_steps(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_buffer(__paralyn_generated_argument_1), paralyn::Argument::from_buffer(__paralyn_generated_argument_2), paralyn::Argument::from_i32(__paralyn_generated_argument_3), paralyn::Argument::from_f32(__paralyn_generated_argument_4)});
}());
      check(cudaGetLastError());
      ++launches;
      for (int buffer = 0; buffer < 3; ++buffer) {
        std::vector<float> actual(n + tail);
        check(cudaMemcpy(actual.data(), device[buffer], bytes, cudaMemcpyDeviceToHost));
        require(actual == expected[buffer],
                "alias layout, dependent store, or tail canary mismatch");
      }
    }

    std::vector<float> input(n + tail, 1.0f), actual(n + tail, 0.0f);
    for (int i = 0; i < 3; ++i)
      check(cudaMemcpy(device[i], input.data(), bytes, cudaMemcpyHostToDevice));
    ([&]() {
  const dim3 __paralyn_generated_grid = (grid_expression());
  const dim3 __paralyn_generated_block = (block_expression());
  const std::size_t __paralyn_generated_shared = (shared_expression());
  const cudaStream_t __paralyn_generated_stream = (stream_expression());
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  float* __paralyn_generated_argument_0 = (pointer_expression(device[0], 0));
  float* __paralyn_generated_argument_1 = (pointer_expression(device[1], 1));
  float* __paralyn_generated_argument_2 = (pointer_expression(device[2], 2));
  int __paralyn_generated_argument_3 = (count_expression());
  float __paralyn_generated_argument_4 = (scalar_expression());
  paralyn::launch_checked(__paralyn_generated_kernel_alias_steps(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_buffer(__paralyn_generated_argument_1), paralyn::Argument::from_buffer(__paralyn_generated_argument_2), paralyn::Argument::from_i32(__paralyn_generated_argument_3), paralyn::Argument::from_f32(__paralyn_generated_argument_4)});
}());
    check(cudaGetLastError());
    ++launches;
    require(ordering_valid, "kernel argument ran before launch configuration completed");
    for (int count : configurations)
      require(count == 1, "configuration evaluated other than once");
    for (int count : arguments)
      require(count == 1, "argument evaluated other than once");
    check(cudaMemcpy(actual.data(), device[2], bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n + tail; ++i)
      require(actual[i] == (i < 7 ? 5.5f : 1.0f), "host parameter conversion or output mismatch");

    ([&]() {
  const dim3 __paralyn_generated_grid = (1);
  const dim3 __paralyn_generated_block = (1);
  const std::size_t __paralyn_generated_shared = (invalid_shared_expression());
  const cudaStream_t __paralyn_generated_stream = (0);
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  float* __paralyn_generated_argument_0 = (rejected_argument(device[0]));
  float* __paralyn_generated_argument_1 = (device[1]);
  float* __paralyn_generated_argument_2 = (device[2]);
  int __paralyn_generated_argument_3 = (1);
  float __paralyn_generated_argument_4 = (1.0f);
  paralyn::launch_checked(__paralyn_generated_kernel_alias_steps(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_buffer(__paralyn_generated_argument_1), paralyn::Argument::from_buffer(__paralyn_generated_argument_2), paralyn::Argument::from_i32(__paralyn_generated_argument_3), paralyn::Argument::from_f32(__paralyn_generated_argument_4)});
}());
    expect_launch_error(cudaErrorNotSupported);
    require(rejected_argument_calls == 0, "unsupported configuration evaluated kernel arguments");
    ([&]() {
  const dim3 __paralyn_generated_grid = (1);
  const dim3 __paralyn_generated_block = (1);
  const std::size_t __paralyn_generated_shared = (0);
  const cudaStream_t __paralyn_generated_stream = (invalid_stream_expression(device[0]));
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  float* __paralyn_generated_argument_0 = (rejected_argument(device[0]));
  float* __paralyn_generated_argument_1 = (device[1]);
  float* __paralyn_generated_argument_2 = (device[2]);
  int __paralyn_generated_argument_3 = (1);
  float __paralyn_generated_argument_4 = (1.0f);
  paralyn::launch_checked(__paralyn_generated_kernel_alias_steps(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_buffer(__paralyn_generated_argument_1), paralyn::Argument::from_buffer(__paralyn_generated_argument_2), paralyn::Argument::from_i32(__paralyn_generated_argument_3), paralyn::Argument::from_f32(__paralyn_generated_argument_4)});
}());
    expect_launch_error(cudaErrorNotSupported);
    require(rejected_argument_calls == 0, "unsupported stream evaluated kernel arguments");

    ([&]() {
  const dim3 __paralyn_generated_grid = (0);
  const dim3 __paralyn_generated_block = (1);
  const std::size_t __paralyn_generated_shared = (0);
  const cudaStream_t __paralyn_generated_stream = (0);
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  float* __paralyn_generated_argument_0 = (device[0]);
  float* __paralyn_generated_argument_1 = (device[1]);
  float* __paralyn_generated_argument_2 = (device[2]);
  int __paralyn_generated_argument_3 = (1);
  float __paralyn_generated_argument_4 = (1.0f);
  paralyn::launch_checked(__paralyn_generated_kernel_alias_steps(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_buffer(__paralyn_generated_argument_1), paralyn::Argument::from_buffer(__paralyn_generated_argument_2), paralyn::Argument::from_i32(__paralyn_generated_argument_3), paralyn::Argument::from_f32(__paralyn_generated_argument_4)});
}());
    expect_launch_error(cudaErrorInvalidValue);
    ([&]() {
  const dim3 __paralyn_generated_grid = (1);
  const dim3 __paralyn_generated_block = (0);
  const std::size_t __paralyn_generated_shared = (0);
  const cudaStream_t __paralyn_generated_stream = (0);
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  float* __paralyn_generated_argument_0 = (device[0]);
  float* __paralyn_generated_argument_1 = (device[1]);
  float* __paralyn_generated_argument_2 = (device[2]);
  int __paralyn_generated_argument_3 = (1);
  float __paralyn_generated_argument_4 = (1.0f);
  paralyn::launch_checked(__paralyn_generated_kernel_alias_steps(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_buffer(__paralyn_generated_argument_1), paralyn::Argument::from_buffer(__paralyn_generated_argument_2), paralyn::Argument::from_i32(__paralyn_generated_argument_3), paralyn::Argument::from_f32(__paralyn_generated_argument_4)});
}());
    expect_launch_error(cudaErrorInvalidValue);
    ([&]() {
  const dim3 __paralyn_generated_grid = (1);
  const dim3 __paralyn_generated_block = (1048576);
  const std::size_t __paralyn_generated_shared = (0);
  const cudaStream_t __paralyn_generated_stream = (0);
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  float* __paralyn_generated_argument_0 = (device[0]);
  float* __paralyn_generated_argument_1 = (device[1]);
  float* __paralyn_generated_argument_2 = (device[2]);
  int __paralyn_generated_argument_3 = (1);
  float __paralyn_generated_argument_4 = (1.0f);
  paralyn::launch_checked(__paralyn_generated_kernel_alias_steps(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_buffer(__paralyn_generated_argument_1), paralyn::Argument::from_buffer(__paralyn_generated_argument_2), paralyn::Argument::from_i32(__paralyn_generated_argument_3), paralyn::Argument::from_f32(__paralyn_generated_argument_4)});
}());
    expect_launch_error(cudaErrorInvalidValue);
    ([&]() {
  const dim3 __paralyn_generated_grid = (1);
  const dim3 __paralyn_generated_block = (1);
  const std::size_t __paralyn_generated_shared = (0);
  const cudaStream_t __paralyn_generated_stream = (0);
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  float* __paralyn_generated_argument_0 = (device[0]);
  int* __paralyn_generated_argument_1 = (reinterpret_cast<int *>(device[0]));
  paralyn::launch_checked(__paralyn_generated_kernel_mixed_pointees(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_buffer(__paralyn_generated_argument_1)});
}());
    expect_launch_error(cudaErrorNotSupported);
    ([&]() {
  const dim3 __paralyn_generated_grid = (1);
  const dim3 __paralyn_generated_block = (1);
  const std::size_t __paralyn_generated_shared = (0);
  const cudaStream_t __paralyn_generated_stream = (0);
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  float* __paralyn_generated_argument_0 = (nullptr);
  float* __paralyn_generated_argument_1 = (device[1]);
  float* __paralyn_generated_argument_2 = (device[2]);
  int __paralyn_generated_argument_3 = (1);
  float __paralyn_generated_argument_4 = (1.0f);
  paralyn::launch_checked(__paralyn_generated_kernel_alias_steps(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_buffer(__paralyn_generated_argument_1), paralyn::Argument::from_buffer(__paralyn_generated_argument_2), paralyn::Argument::from_i32(__paralyn_generated_argument_3), paralyn::Argument::from_f32(__paralyn_generated_argument_4)});
}());
    expect_launch_error(cudaErrorInvalidDevicePointer);
    expect_error(
        cudaMemcpy(actual.data(), device[0], bytes + sizeof(float), cudaMemcpyDeviceToHost),
        cudaErrorInvalidValue);
    expect_error(cudaMemcpy(device[0], input.data(), bytes + sizeof(float), cudaMemcpyHostToDevice),
                 cudaErrorInvalidValue);
    expect_error(cudaMemcpy(actual.data(), input.data(), sizeof(float), cudaMemcpyHostToHost),
                 cudaErrorInvalidMemcpyDirection);
    expect_error(cudaMalloc(nullptr, 4), cudaErrorInvalidValue);
    expect_error(cudaFree(input.data()), cudaErrorInvalidDevicePointer);

    unsigned int *unsigned_device = nullptr;
    check(cudaMalloc(&unsigned_device, 8 * sizeof(unsigned int)));
    ([&]() {
  const dim3 __paralyn_generated_grid = (1);
  const dim3 __paralyn_generated_block = (8);
  const std::size_t __paralyn_generated_shared = (0);
  const cudaStream_t __paralyn_generated_stream = (0);
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  unsigned int* __paralyn_generated_argument_0 = (unsigned_device);
  unsigned int __paralyn_generated_argument_1 = (std::uint32_t(0xfffffffc));
  paralyn::launch_checked(__paralyn_generated_kernel_unsigned_values(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_u32(__paralyn_generated_argument_1)});
}());
    check(cudaGetLastError());
    ++launches;
    std::vector<unsigned int> unsigned_actual(8);
    check(cudaMemcpy(unsigned_actual.data(), unsigned_device, 8 * sizeof(unsigned int),
                     cudaMemcpyDeviceToHost));
    for (unsigned int i = 0; i < 8; ++i)
      require(unsigned_actual[i] == std::uint32_t(0xfffffffcu + i), "u32 wrap/argument mismatch");
    check(cudaFree(unsigned_device));

    // A const-qualified read must observe writes through an alias of the same allocation.
    std::vector<float> const_input(n + tail, canary);
    for (int i = 0; i < n; ++i)
      const_input[i] = float(i - 9) * 0.25f;
    check(cudaMemcpy(device[0], const_input.data(), bytes, cudaMemcpyHostToDevice));
    ([&]() {
  const dim3 __paralyn_generated_grid = (3);
  const dim3 __paralyn_generated_block = (7);
  const std::size_t __paralyn_generated_shared = (0);
  const cudaStream_t __paralyn_generated_stream = (0);
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  const float* __paralyn_generated_argument_0 = (device[0]);
  float* __paralyn_generated_argument_1 = (device[0]);
  int __paralyn_generated_argument_2 = (n);
  paralyn::launch_checked(__paralyn_generated_kernel_const_alias(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_buffer(__paralyn_generated_argument_1), paralyn::Argument::from_i32(__paralyn_generated_argument_2)});
}());
    check(cudaGetLastError());
    ++launches;
    check(cudaMemcpy(actual.data(), device[0], bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n + tail; ++i)
      require(actual[i] == (i < n ? const_input[i] + 3.0f : canary),
              "const/mutable alias dependent read or tail canary mismatch");

    // A free must wait for a launch that still retains this allocation.
    ([&]() {
  const dim3 __paralyn_generated_grid = (1);
  const dim3 __paralyn_generated_block = (7);
  const std::size_t __paralyn_generated_shared = (0);
  const cudaStream_t __paralyn_generated_stream = (0);
  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, __paralyn_generated_stream)) return;
  float* __paralyn_generated_argument_0 = (device[0]);
  float* __paralyn_generated_argument_1 = (device[1]);
  float* __paralyn_generated_argument_2 = (device[2]);
  int __paralyn_generated_argument_3 = (7);
  float __paralyn_generated_argument_4 = (0.5f);
  paralyn::launch_checked(__paralyn_generated_kernel_alias_steps(), __paralyn_generated_grid, __paralyn_generated_block, {paralyn::Argument::from_buffer(__paralyn_generated_argument_0), paralyn::Argument::from_buffer(__paralyn_generated_argument_1), paralyn::Argument::from_buffer(__paralyn_generated_argument_2), paralyn::Argument::from_i32(__paralyn_generated_argument_3), paralyn::Argument::from_f32(__paralyn_generated_argument_4)});
}());
    check(cudaGetLastError());
    ++launches;
    check(cudaFree(device[0]));
    expect_error(cudaFree(device[0]), cudaErrorInvalidDevicePointer);
    expect_error(cudaMemcpy(actual.data(), device[0], sizeof(float), cudaMemcpyDeviceToHost),
                 cudaErrorInvalidDevicePointer);
    check(cudaFree(device[1]));
    check(cudaFree(device[2]));
    check(cudaDeviceSynchronize());
    if (argc > 1 && std::strcmp(argv[1], "exit37") == 0)
      return 37;
    std::printf(
        "Verification: PASS aliases_and_host cases=5 const_alias=1 launches=%d runtime_errors=14\n",
        launches);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Verification: FAIL aliases_and_host: %s\n", error.what());
    return 1;
  }
}
