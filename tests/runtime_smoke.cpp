#include <cuda_runtime.h>
#include <iostream>
#include <functional>
#include <limits>
#include <streambuf>
#include <stdexcept>
#include <vector>

namespace {
class ThrowingSubmissionLog : public std::streambuf {
  std::streambuf *original;
public:
  explicit ThrowingSubmissionLog(std::streambuf *out) : original(out) {}
  std::streamsize xsputn(const char *text, std::streamsize count) override {
    if (std::string(text, static_cast<std::size_t>(count)).find("Executing on GPU") !=
        std::string::npos)
      throw std::runtime_error("Deliberate host log failure");
    return original->sputn(text, count);
  }
  int overflow(int value) override {
    return traits_type::eq_int_type(value, traits_type::eof())
               ? traits_type::not_eof(value)
               : original->sputc(traits_type::to_char_type(value));
  }
  int sync() override { return original->pubsync(); }
};
void require(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}
void expect(cudaError_t actual, cudaError_t expected, const char *message) {
  if (actual != expected)
    throw std::runtime_error(message);
  if (cudaGetLastError() != expected || cudaGetLastError() != cudaSuccess)
    throw std::runtime_error("Runtime failure was not preserved/reset correctly");
}
void rejects(const std::function<void()> &action, const char *detail) {
  try {
    action();
  } catch (const std::exception &error) {
    if (std::string(error.what()).find(detail) != std::string::npos)
      return;
    throw std::runtime_error(std::string("Wrong failure: ") + error.what());
  }
  throw std::runtime_error("Invalid runtime operation was accepted");
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--throwing-log") {
      paralyn::Kernel kernel{"logging_failure", {}, {}};
      unsetenv("PARALYN_RUNTIME_LOG");
      auto *original = std::cerr.rdbuf();
      const auto exceptions = std::cerr.exceptions();
      ThrowingSubmissionLog failing(original);
      std::cerr.rdbuf(&failing);
      std::cerr.exceptions(std::ios::badbit | std::ios::failbit);
      bool rejected = false;
      try {
        paralyn::launch(kernel, {1, 1, 1}, {1, 1, 1}, {});
      } catch (const std::exception &) {
        rejected = true;
      }
      std::cerr.exceptions(std::ios::goodbit);
      std::cerr.clear();
      std::cerr.rdbuf(original);
      std::cerr.exceptions(exceptions);
      if (!rejected || paralyn::runtime_statistics().completed_launches != 0)
        throw std::runtime_error("Throwing log unexpectedly submitted or completed GPU work");
      // A retained uncommitted command would make this synchronization hang/fail.
      paralyn::synchronize();
      paralyn::launch(kernel, {1, 1, 1}, {1, 1, 1}, {});
      paralyn::synchronize();
      if (paralyn::runtime_statistics().completed_launches != 1)
        throw std::runtime_error("Runtime did not recover after host log exception");
      paralyn::shutdown();
      std::cout << "Throwing host log: submission remained atomic and GPU recovery passed\n";
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--unobserved-error") {
      // Returning zero must not hide a compatibility-runtime error ignored by the host.
      cudaFree(reinterpret_cast<void *>(1));
      return 0;
    }
    constexpr int n = 1024;
    std::vector<float> a(n), b(n), actual(n, -9999.0f);
    for (int i = 0; i < n; ++i) {
      a[i] = float((i % 31) - 15) * 0.5f;
      b[i] = float((i % 17) - 8) * 0.25f;
    }
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    require(cudaMalloc(&da, n * sizeof(float)));
    require(cudaMalloc(&db, n * sizeof(float)));
    require(cudaMalloc(&dc, n * sizeof(float)));
    require(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    require(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    require(cudaMemcpy(dc, actual.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    using namespace paralyn;
    Kernel signature{"runtime_smoke",
                     {{"a", ScalarType::F32, true, true},
                      {"b", ScalarType::F32, true, true},
                      {"c", ScalarType::F32, true, false},
                      {"n", ScalarType::I32, false, false}},
                     {}};
    const char *handwritten = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void runtime_smoke(const device float* a [[buffer(0)]],
                          const device float* b [[buffer(1)]],
                          device float* c [[buffer(2)]],
                          constant int& n [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
  if (i < uint(n)) c[i] = a[i] + b[i];
}
)MSL";
    launch_msl_for_test(signature, handwritten, {4, 1, 1}, {256, 1, 1},
                        {Argument::from_buffer(da), Argument::from_buffer(db),
                         Argument::from_buffer(dc), Argument::from_i32(n)});
    require(cudaDeviceSynchronize());
    require(cudaMemcpy(actual.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
      if (actual[i] != a[i] + b[i])
        throw std::runtime_error("Handwritten Metal result differs from CPU reference at index " +
                                 std::to_string(i));
    }
    auto stats = runtime_statistics();
    if (stats.completed_launches != 1 || !(stats.last_gpu_seconds > 0) ||
        stats.pipeline_compilations != 1 || !(stats.pipeline_compile_seconds > 0) ||
        stats.current_buffer_bytes != 3 * n * sizeof(float) ||
        stats.peak_buffer_bytes != 3 * n * sizeof(float))
      throw std::runtime_error("Runtime timing/buffer statistics are inconsistent");
    expect(cudaMemcpy(actual.data(), dc, (n + 1) * sizeof(float), cudaMemcpyDeviceToHost),
           cudaErrorInvalidValue, "Oversized copy was not rejected");
    expect(cudaMalloc(nullptr, 4), cudaErrorInvalidValue, "Null allocation output was accepted");
    void *empty = reinterpret_cast<void *>(1);
    require(cudaMalloc(&empty, 0));
    if (empty != nullptr)
      throw std::runtime_error("Zero allocation did not return a null token");
    require(cudaFree(nullptr));
    require(cudaMemcpy(nullptr, nullptr, 0, cudaMemcpyHostToDevice));
    void *huge = nullptr;
    expect(cudaMalloc(&huge, std::numeric_limits<std::size_t>::max()),
           cudaErrorMemoryAllocation, "Impossible allocation was accepted");
    expect(cudaMemcpy(nullptr, da, 4, cudaMemcpyDeviceToHost), cudaErrorInvalidValue,
           "Null nonempty copy was accepted");
    expect(cudaMemcpy(da, db, 4, cudaMemcpyHostToDevice), cudaErrorInvalidValue,
           "Device token was accepted as a host pointer");
    expect(cudaMemcpy(actual.data(), da, 4, cudaMemcpyDefault), cudaErrorInvalidMemcpyDirection,
           "Implicit copy direction was accepted");
    expect(cudaMemcpy(db, da, 4, cudaMemcpyDeviceToDevice), cudaErrorInvalidMemcpyDirection,
           "Unsupported device copy was accepted");
    auto interior = reinterpret_cast<void *>(reinterpret_cast<std::uintptr_t>(da) + 1);
    expect(cudaMemcpy(actual.data(), interior, 4, cudaMemcpyDeviceToHost),
           cudaErrorInvalidDevicePointer, "Interior token was accepted");
    expect(cudaFree(actual.data()), cudaErrorInvalidDevicePointer, "Foreign free was accepted");
    const std::vector<Argument> arguments{
        Argument::from_buffer(da), Argument::from_buffer(db), Argument::from_buffer(dc),
        Argument::from_i32(n)};
    // Unlike expect(), launch errors are retrieved once because launch_checked returns void.
    auto launch_error = [&](const Kernel &kernel, Dim3 grid, Dim3 block,
                            const std::vector<Argument> &args, cudaError_t expected) {
      launch_checked(kernel, grid, block, args);
      if (cudaGetLastError() != expected || cudaGetLastError() != cudaSuccess)
        throw std::runtime_error("Invalid launch did not preserve/reset its error");
    };
    launch_error(signature, {1, 1, 1}, {1, 1, 1}, {}, cudaErrorInvalidValue);
    auto bad_arguments = arguments;
    bad_arguments[3] = Argument::from_u32(n);
    launch_error(signature, {1, 1, 1}, {1, 1, 1}, bad_arguments, cudaErrorInvalidValue);
    bad_arguments = arguments;
    bad_arguments[0] = Argument::from_i32(1);
    launch_error(signature, {1, 1, 1}, {1, 1, 1}, bad_arguments, cudaErrorInvalidValue);
    bad_arguments = arguments;
    bad_arguments[0] = Argument::from_buffer(interior);
    launch_error(signature, {1, 1, 1}, {1, 1, 1}, bad_arguments, cudaErrorInvalidDevicePointer);
    Kernel mixed{"mixed", {{"a", ScalarType::F32, true, true},
                            {"b", ScalarType::I32, true, true}}, {}};
    launch_error(mixed, {1, 1, 1}, {1, 1, 1},
                 {Argument::from_buffer(da), Argument::from_buffer(da)}, cudaErrorNotSupported);
    for (auto geometry : std::vector<std::pair<Dim3, Dim3>>{
             {{0, 1, 1}, {256, 1, 1}}, {{1, 0, 1}, {1, 1, 1}},
             {{1, 1, 1}, {1, 1, 0}}, {{1, 1, 1}, {UINT32_MAX, 1, 1}},
             {{UINT32_MAX, 1, 1}, {256, 1, 1}}, {{UINT32_MAX, UINT32_MAX, 2}, {1, 1, 1}}})
      launch_error(signature, geometry.first, geometry.second, arguments, cudaErrorInvalidValue);
    if (validate_launch_configuration(1, nullptr) || cudaGetLastError() != cudaErrorNotSupported)
      throw std::runtime_error("Dynamic shared memory was silently accepted");
    if (validate_launch_configuration(0, da) || cudaGetLastError() != cudaErrorNotSupported)
      throw std::runtime_error("Nondefault stream was silently accepted");
    rejects([&] { select_device("999999999999999999999999999"); }, "out of range");
    rejects([&] { select_device("invalid"); }, "numeric index");
    rejects([&] { launch_msl_for_test(signature, "this is not Metal", {1, 1, 1},
                                     {1, 1, 1}, arguments); }, "compilation failed");
    auto wrong_entrypoint = signature;
    wrong_entrypoint.name = "missing_entrypoint";
    rejects([&] { launch_msl_for_test(wrong_entrypoint, handwritten, {1, 1, 1},
                                     {1, 1, 1}, arguments); }, "no entrypoint");
    if (runtime_statistics().completed_launches != 1)
      throw std::runtime_error("A rejected operation submitted GPU work");
    require(cudaFree(da));
    require(cudaFree(db));
    require(cudaFree(dc));
    expect(cudaFree(dc), cudaErrorInvalidDevicePointer, "Repeated free was not rejected");
    expect(cudaMemcpy(actual.data(), dc, 4, cudaMemcpyDeviceToHost), cudaErrorInvalidDevicePointer,
           "Use after free was accepted");
    if (runtime_statistics().current_buffer_bytes != 0)
      throw std::runtime_error("Freed buffers remain counted as runtime owned");
    shutdown();
    std::cout << "Handwritten Metal smoke: " << n << " values matched the CPU reference\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Runtime smoke failed: " << error.what() << '\n';
    return 1;
  }
}
