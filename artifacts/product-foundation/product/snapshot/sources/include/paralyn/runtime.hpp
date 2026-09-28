#pragma once
#include "paralyn/ir.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace paralyn {
struct Dim3 {
  std::uint32_t x = 1, y = 1, z = 1;
};
struct Argument {
  bool buffer = false;
  void *token = nullptr;
  ScalarType type = ScalarType::I32;
  std::array<unsigned char, 4> bytes{};
  static Argument from_buffer(const void *value);
  static Argument from_i32(std::int32_t value);
  static Argument from_u32(std::uint32_t value);
  static Argument from_f32(float value);
};
// Cumulative counters for the active context. GPU counters include completed work only.
// Buffer bytes count requested MTLBuffer lengths, excluding driver/pipeline allocations.
struct RuntimeStatistics {
  double pipeline_compile_seconds = 0;
  double host_to_device_seconds = 0;
  double device_to_host_seconds = 0;
  double gpu_seconds = 0;
  double last_gpu_seconds = 0;
  std::uint64_t pipeline_compilations = 0;
  std::uint64_t completed_launches = 0;
  std::uint64_t host_to_device_bytes = 0;
  std::uint64_t device_to_host_bytes = 0;
  std::uint64_t current_buffer_bytes = 0;
  std::uint64_t peak_buffer_bytes = 0;
};
RuntimeStatistics runtime_statistics();
void select_device(const std::string &selector);
std::string devices_text();
void launch(const Kernel &, Dim3 grid, Dim3 block, const std::vector<Argument> &);
void launch_checked(const Kernel &, Dim3 grid, Dim3 block, const std::vector<Argument> &) noexcept;
bool validate_launch_configuration(std::size_t shared, const void *stream) noexcept;
void synchronize();
void shutdown();
// Test-only escape hatch: exercises the actual backend with independently written MSL.
void launch_msl_for_test(const Kernel &, const std::string &source, Dim3 grid, Dim3 block,
                         const std::vector<Argument> &);
} // namespace paralyn
