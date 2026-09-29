#pragma once
#include "paralyn/ir.hpp"
#include <string>
#include <vector>

namespace paralyn {
// CUDA C++ device source generated exclusively from verified scalar IR, for
// runtime compilation with NVRTC by the dynamically loaded CUDA backend.
//
// Semantics: one kernel parameter per IR parameter (no binding slots; CUDA
// pointers may alias, so same-allocation arguments need no special lowering).
// Buffer views are passed as base device address plus byte offset by the host.
// FP32 addition/multiplication are emitted as __fadd_rn/__fmul_rn, which CUDA
// documents are never contracted into FMA, in addition to --fmad=false.
std::string cuda_entrypoint(const Kernel &kernel);
std::string emit_cuda(const Kernel &kernel);
// Numerical options that accompany every generated module (numerical policy 1:
// round-to-nearest FP32 add/mul, no contraction, no flush-to-zero, IEEE div/sqrt).
// The caller adds the explicit --gpu-architecture option. Fast math is never used.
std::vector<std::string> cuda_numerical_options();
} // namespace paralyn
