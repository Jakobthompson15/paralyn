#pragma once
#include "paralyn/ir.hpp"
#include <cstddef>
#include <vector>

namespace paralyn {
// Trusted-local compiled IR, not a native shader or arbitrary-code sandbox.
// Version 1 encodes fixed little-endian integers and the current precise-f32
// numerical policy. Decoding always verifies the complete IR before returning.
inline constexpr std::size_t artifact_max_bytes = 16 * 1024 * 1024;
std::vector<unsigned char> serialize_module(const std::vector<Kernel> &kernels);
std::vector<Kernel> deserialize_module(const void *data, std::size_t size);
} // namespace paralyn
