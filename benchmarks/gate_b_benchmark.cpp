#include "paralyn/frontend.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <streambuf>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
void check(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}
struct QuietBuffer : std::streambuf {
  int overflow(int c) override { return traits_type::not_eof(c); }
};
struct QuietRuntime {
  QuietBuffer buffer;
  std::streambuf *previous_out = std::cout.rdbuf(&buffer);
  std::streambuf *previous_error = std::cerr.rdbuf(&buffer);
  ~QuietRuntime() {
    std::cerr.rdbuf(previous_error);
    std::cout.rdbuf(previous_out);
  }
};
struct Sample {
  int n;
  const char *phase;
  int iteration;
  int order;
  const char *variant;
  double compile, h2d_api, d2h_api, h2d_copy, d2h_copy, gpu, total;
  std::uint64_t peak_bytes;
};
constexpr const char *native_source = R"MSL(
#include <metal_stdlib>
#pragma STDC FP_CONTRACT OFF
using namespace metal;
kernel void handwritten_vector_add(const device float* a [[buffer(0)]],
                                   const device float* b [[buffer(1)]],
                                   device float* c [[buffer(2)]],
                                   constant int& n [[buffer(3)]],
                                   uint i [[thread_position_in_grid]]) {
  if (i < uint(n)) c[i] = a[i] + b[i];
}
)MSL";
void write_file(const std::filesystem::path &path, const std::string &content) {
  std::ofstream out(path);
  out << content;
  out.close();
  if (!out)
    throw std::runtime_error("Cannot write " + path.string());
}
void write_samples(const std::filesystem::path &directory, const std::vector<Sample> &samples,
                   double frontend_seconds, const paralyn::RuntimeStatistics &statistics) {
  std::ofstream out(directory / "samples.csv");
  out << std::setprecision(17)
      << "elements,phase,iteration,order,variant,pipeline_compile_seconds,h2d_api_seconds,"
         "d2h_api_seconds,h2d_copy_seconds,d2h_copy_seconds,gpu_seconds,total_seconds,"
         "runtime_owned_peak_buffer_bytes\n";
  for (const auto &s : samples)
    out << s.n << ',' << s.phase << ',' << s.iteration << ',' << s.order << ',' << s.variant << ','
        << s.compile << ',' << s.h2d_api << ',' << s.d2h_api << ',' << s.h2d_copy << ','
        << s.d2h_copy << ',' << s.gpu << ',' << s.total << ',' << s.peak_bytes << '\n';
  out.close();
  if (!out)
    throw std::runtime_error("Cannot write samples.csv");
  std::ofstream protocol(directory / "benchmark.json");
  protocol << std::setprecision(17)
           << "{\n  \"status\": \"verified\",\n  \"sizes\": [1024, 65536, 1048576, 16777216],"
           << "\n  \"warmups_per_variant_per_size\": 10,"
           << "\n  \"measured_iterations_per_variant_per_size\": 100,"
           << "\n  \"order\": \"generated first on even iterations, handwritten first on odd "
              "iterations\","
           << "\n  \"frontend_compile_seconds\": " << frontend_seconds
           << ",\n  \"pipeline_compile_seconds\": " << statistics.pipeline_compile_seconds
           << ",\n  \"pipeline_compilations\": " << statistics.pipeline_compilations
           << ",\n  \"completed_gpu_launches\": " << statistics.completed_launches
           << ",\n  \"samples\": " << samples.size()
           << ",\n  \"runtime_owned_peak_buffer_bytes\": " << statistics.peak_buffer_bytes
           << ",\n  \"runtime_owned_current_buffer_bytes\": " << statistics.current_buffer_bytes
           << ",\n  \"gpu_environment_and_launch_evidence\": \"execution.json\","
           << "\n  \"sample_file\": \"samples.csv\","
           << "\n  \"total_definition\": \"H2D of both inputs, launch, explicit synchronization, "
              "D2H; excludes allocation, input generation and CPU verification\","
           << "\n  \"copy_definition\": \"copy_seconds times memcpy to/from shared MTLBuffer "
              "contents; api_seconds also includes validation and synchronization\","
           << "\n  \"compile_definition\": \"frontend timed once; Metal library and pipeline "
              "compilation attributed to cold warmup launch, including any Metal driver cache "
              "behavior; per-launch IR verification and MSL emission remain in total latency; no "
              "persistent Paralyn cache\","
           << "\n  \"memory_definition\": \"requested lengths of live runtime-owned MTLBuffers; "
              "excludes host vectors, pipeline and driver allocations\","
           << "\n  \"verification\": \"every element after every warmup and measured launch "
              "compared exactly with independently computed CPU FP32 sum; varied dyadic inputs and "
              "output sentinel each run\","
           << "\n  \"verification_timing\": \"CPU comparison and sentinel initialization occur "
              "outside measured total\","
           << "\n  \"progress_policy\": \"Launch progress is discarded through in-memory stdout/stderr "
              "sinks; runtime/event log files are disabled; command evidence is exported after measurement\","
           << "\n  \"cpu_fallback\": false\n}\n";
  protocol.close();
  if (!protocol)
    throw std::runtime_error("Cannot write benchmark.json");
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 5 || std::string(argv[1]) != "--source" || std::string(argv[3]) != "--artifacts")
      throw std::runtime_error(
          "Usage: gate_b_benchmark --source examples/vector_add.cu --artifacts NEW_DIRECTORY");
    const auto source_path = std::filesystem::absolute(argv[2]);
    const auto directory = std::filesystem::absolute(argv[4]);
    if (std::filesystem::exists(directory))
      throw std::runtime_error("Benchmark artifacts directory must not already exist");
    std::filesystem::create_directories(directory);
    // Avoid evidence-file I/O in measured work. Runtime retains all launch records in memory.
    unsetenv("PARALYN_ARTIFACT_DIR");
    unsetenv("PARALYN_RUNTIME_LOG");
    unsetenv("PARALYN_EVENT_LOG");
    const auto frontend_start = Clock::now();
    const auto frontend = paralyn::compile_source(source_path.string());
    const double frontend_seconds = seconds(frontend_start);
    if (frontend.kernels.size() != 1 || frontend.kernels[0].name != "vector_add")
      throw std::runtime_error("Benchmark expects the canonical vector_add source");
    const auto &generated = frontend.kernels[0];
    paralyn::verify(generated);
    if (generated.parameters.size() != 4)
      throw std::runtime_error("Benchmark vector_add signature changed");
    auto native = generated;
    native.name = "handwritten_vector_add";
    std::filesystem::copy_file(source_path, directory / "source.cu");
    write_file(directory / "paralyn-ir.txt", paralyn::dump_ir(generated));
    write_file(directory / "host.cpp", frontend.rewritten_host);
    write_file(directory / "handwritten.metal", native_source);
    std::vector<Sample> samples;
    for (int n : {1024, 65536, 1048576, 16777216}) {
      const std::size_t bytes = std::size_t(n) * sizeof(float);
      std::vector<float> a(n), b(n), output(n), expected(n);
      for (int i = 0; i < n; ++i) {
        a[i] = float((i % 127) - 63) * 0.5f;
        b[i] = float((i % 61) - 30) * 0.25f;
        expected[i] = a[i] + b[i];
      }
      float *da = nullptr, *db = nullptr, *dc = nullptr;
      check(cudaMalloc(&da, bytes));
      check(cudaMalloc(&db, bytes));
      check(cudaMalloc(&dc, bytes));
      const std::vector<paralyn::Argument> arguments{
          paralyn::Argument::from_buffer(da), paralyn::Argument::from_buffer(db),
          paralyn::Argument::from_buffer(dc), paralyn::Argument::from_i32(n)};
      for (bool measured : {false, true}) {
        for (int iteration = 0; iteration < (measured ? 100 : 10); ++iteration) {
          for (int order = 0; order < 2; ++order) {
            const bool generated_first = iteration % 2 == 0;
            const bool use_generated = order == 0 ? generated_first : !generated_first;
            std::fill(output.begin(), output.end(), -9999.0f);
            check(cudaMemcpy(dc, output.data(), bytes, cudaMemcpyHostToDevice));
            const auto before = paralyn::runtime_statistics();
            const auto start = Clock::now();
            check(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
            check(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
            const double h2d_api = seconds(start);
            {
              QuietRuntime quiet;
              if (use_generated)
                paralyn::launch(generated, {unsigned((n + 255) / 256), 1, 1}, {256, 1, 1},
                                arguments);
              else
                paralyn::launch_msl_for_test(native, native_source,
                                             {unsigned((n + 255) / 256), 1, 1}, {256, 1, 1},
                                             arguments);
              check(cudaDeviceSynchronize());
            }
            const auto copy_start = Clock::now();
            check(cudaMemcpy(output.data(), dc, bytes, cudaMemcpyDeviceToHost));
            const double d2h_api = seconds(copy_start);
            const double total = seconds(start);
            const auto after = paralyn::runtime_statistics();
            if (after.completed_launches != before.completed_launches + 1 ||
                !(after.last_gpu_seconds > 0))
              throw std::runtime_error("Missing physical GPU completion/timing evidence");
            for (int i = 0; i < n; ++i)
              if (output[i] != expected[i])
                throw std::runtime_error("CPU mismatch at size " + std::to_string(n) + " index " +
                                         std::to_string(i));
            samples.push_back({n, measured ? "measured" : "warmup", iteration, order,
                               use_generated ? "generated" : "handwritten",
                               after.pipeline_compile_seconds - before.pipeline_compile_seconds,
                               h2d_api, d2h_api,
                               after.host_to_device_seconds - before.host_to_device_seconds,
                               after.device_to_host_seconds - before.device_to_host_seconds,
                               after.last_gpu_seconds, total, after.peak_buffer_bytes});
          }
        }
      }
      check(cudaFree(da));
      check(cudaFree(db));
      check(cudaFree(dc));
      std::cout << "Verified " << n
                << " elements: 10 warmups + 100 measured iterations per variant\n"
                << std::flush;
    }
    setenv("PARALYN_ARTIFACT_DIR", directory.c_str(), 1);
    paralyn::shutdown();
    const auto statistics = paralyn::runtime_statistics();
    if (statistics.completed_launches != 880 || statistics.current_buffer_bytes != 0)
      throw std::runtime_error("Benchmark did not complete all launches or release all buffers");
    write_samples(directory, samples, frontend_seconds, statistics);
    write_file(directory / "verification.txt",
               "Verification: PASS\n880 real GPU launches; every output independently compared "
               "with CPU reference.\n");
    std::cout << "Benchmark verification: PASS; all 880 samples saved to " << directory << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
