#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

__global__ void alias_steps(float *a, float *b, float *c, int n, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    a[i] = a[i] + b[i];
    c[i] = a[i] * scale + b[i];
  }
}
__global__ void const_alias(const float *input, float *output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = input[i] + 1.0f;
    output[i] = input[i] + 2.0f;
  }
}
__global__ void mixed_pointees(float *a, int *b) { b[0] = 1; }
__global__ void unsigned_values(unsigned int *output, unsigned int bias) {
  int i = threadIdx.x;
  output[i] = threadIdx.x + bias;
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
      alias_steps<<<3, 7>>>(device[a], device[b], device[c], n, scale);
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
    alias_steps<<<grid_expression(), block_expression(), shared_expression(),
                  stream_expression()>>>(
        pointer_expression(device[0], 0), pointer_expression(device[1], 1),
        pointer_expression(device[2], 2), count_expression(), scalar_expression());
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

    alias_steps<<<1, 1, invalid_shared_expression()>>>(rejected_argument(device[0]), device[1],
                                                       device[2], 1, 1.0f);
    expect_launch_error(cudaErrorNotSupported);
    require(rejected_argument_calls == 0, "unsupported configuration evaluated kernel arguments");
    alias_steps<<<1, 1, 0, invalid_stream_expression(device[0])>>>(rejected_argument(device[0]),
                                                                   device[1], device[2], 1, 1.0f);
    expect_launch_error(cudaErrorNotSupported);
    require(rejected_argument_calls == 0, "unsupported stream evaluated kernel arguments");

    alias_steps<<<0, 1>>>(device[0], device[1], device[2], 1, 1.0f);
    expect_launch_error(cudaErrorInvalidValue);
    alias_steps<<<1, 0>>>(device[0], device[1], device[2], 1, 1.0f);
    expect_launch_error(cudaErrorInvalidValue);
    alias_steps<<<1, 1048576>>>(device[0], device[1], device[2], 1, 1.0f);
    expect_launch_error(cudaErrorInvalidValue);
    mixed_pointees<<<1, 1>>>(device[0], reinterpret_cast<int *>(device[0]));
    expect_launch_error(cudaErrorNotSupported);
    alias_steps<<<1, 1>>>(nullptr, device[1], device[2], 1, 1.0f);
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
    unsigned_values<<<1, 8>>>(unsigned_device, std::uint32_t(0xfffffffc));
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
    const_alias<<<3, 7>>>(device[0], device[0], n);
    check(cudaGetLastError());
    ++launches;
    check(cudaMemcpy(actual.data(), device[0], bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n + tail; ++i)
      require(actual[i] == (i < n ? const_input[i] + 3.0f : canary),
              "const/mutable alias dependent read or tail canary mismatch");

    // A free must wait for a launch that still retains this allocation.
    alias_steps<<<1, 7>>>(device[0], device[1], device[2], 7, 0.5f);
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
