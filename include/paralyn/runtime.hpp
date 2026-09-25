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
