#pragma once
// TEST DOUBLE ONLY. An in-process imitation of the CUDA Driver API / NVRTC
// function tables for exercising Paralyn's host-side CUDA backend logic
// (context push/pop, argument marshalling, error mapping, cleanup) on machines
// without NVIDIA hardware. It never executes kernels, never produces results
// and is linked only into tests/cuda/backend_tests. It is not a CPU fallback.
#include "api.hpp"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace fake_cuda {
using namespace paralyn::backend::cuda;
struct Launch {
  std::string function;
  unsigned grid[3], block[3];
  std::vector<std::vector<unsigned char>> parameters;
};
struct State {
  int device_count = 1, cc_major = 8, cc_minor = 6;
  std::vector<int> nvrtc_archs = {50, 60, 70, 75, 80, 86, 90};
  CUcontext primary = reinterpret_cast<CUcontext>(std::uintptr_t(0xC0DE0000));
  int primary_retained = 0, primary_retains = 0, primary_releases = 0;
  std::map<CUdeviceptr, std::vector<unsigned char>> memory; // host bytes standing in for allocations
  CUdeviceptr next_pointer = 0x100000;
  int streams = 0, events = 0, modules = 0, programs = 0;
  std::vector<std::string> nvrtc_options, sources, loaded_images;
  std::string nvrtc_log = "fake NVRTC log (test double)";
  std::vector<Launch> launches;
  std::vector<std::size_t> parameter_sizes; // set by the test for the next launch
  float elapsed_ms = 0.25f;
  std::map<std::string, int> fail_next;     // function name -> injected result
  std::map<std::string, int> fail_after;    // function name -> successful calls before fail_next applies
  std::vector<std::string> context_violations; // driver calls made without the primary current
};
State &state();
void reset();
DriverApi driver_table();
NvrtcApi nvrtc_table();
// Thread-local current-context stack exposed for assertions.
std::vector<CUcontext> &context_stack();
} // namespace fake_cuda
