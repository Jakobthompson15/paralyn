# Implementation audit — native-runtime checkpoint

The session began with local HEAD and remote `main` both at `0e7041f046ee5d5ff8f2560233ed5b1ba9c812cd`. There were no newer local implementations or uncommitted changes. That snapshot contained the CUDA/Metal Gates A/B implementation and evidence, six CTest targets, and design-only native/Python rows. The user's published-snapshot assessment was correct.

The audit uses source files, CMake targets, tests and physical-device captures. A roadmap or ledger change does not qualify a row.

| Input/interface | Implementation files and build targets | Tests and hardware evidence | Remaining scope |
|---|---|---|---|
| CUDA C++ | `compiler/frontend/frontend.cpp`, `compiler/ir/ir.cpp`, `compiler/codegen/metal.cpp`, CUDA adapter `backends/metal/runtime.mm`; `paralyn`, `paralyn_frontend`, `paralyn_runtime` | Existing `frontend`, `ir_and_codegen`, `metal_runtime`, `unobserved_runtime_failure`, `gate_a`, `gate_b_correctness`; immutable `artifacts/gate-b/` | Narrow scalar subset only; all cross-vendor cells unqualified |
| Native C/C++ | **New** `include/paralyn/native.h`, `native.hpp`, `runtime/native.cpp`; `paralyn_native` shared library; `native_c`, `native_cpp`, `native_tests` executables | **New** `native_c`, `native_cpp`, `native_tests` CTests; actual vector-add/affine, offsets, aliases, lifetimes, failures; `scripts/qualify_native.py` | First byte-buffer/launch contract only; further operators and backends required |
| Native Python | **New** `bindings/python/paralyn/__init__.py`; loads the exact same `paralyn_native` library | **New** `python_tests`, `python_example` CTests; actual GPU results and ownership/error tests | No arbitrary Python compiler, arrays, NumPy/Array API/DLPack integration |
| Shared module boundary | **New** `compiler/ir/artifact.cpp`, `include/paralyn/artifact.hpp`, `paralyn compile`; `paralyn_ir`, generated `native-kernels.prk` | **New** `kernel_artifacts`, `native_compile` tests; bounded format/verifier/compile-without-host-execution | Internal versioned verified scalar IR; not another qualified language frontend |
| Metal backend | **Refactored** `backends/metal/engine.mm`, `include/paralyn/detail/backend.hpp`; shared by CUDA and native API | Existing CUDA Gates A/B plus new native tests; real Apple M5 command timestamps and independently compared output | No universal driver-fault or cross-vendor guarantee |
| NVIDIA CUDA and AMD HIP/ROCm backends | No implementation files or build targets | No hardware capture; neither device is available in the observed local Metal inventory | Both required |

The remaining **14 input families** have no implemented import/lowering/execution build target or qualifying hardware test: HIP C++, Triton, OpenCL C, SYCL C++, OpenMP target offload, Slang, HLSL compute, GLSL compute, WGSL, SPIR-V, MLIR, public Metal source, PTX and SASS. All remain required. Internal handwritten MSL tests do not implement the public Metal-source frontend. PTX and SASS remain distinct tracks.

Numba CUDA, CuPy and CUDA Python integrations are also unimplemented and required. Framework integrations and FP32 matmul providers have no implementation targets/evidence. No native tensor/array surface, persistent disk cache, distributed system, or specialized accelerator is claimed.

`otool -L build/libparalyn_native.dylib` shows Metal, Foundation, libc++, libSystem and libobjc; the native runtime does not depend on LLVM/Clang. The compiler CLI still depends on its explicitly selected LLVM installation. The Python binding adds no third-party package dependency. Code remains independent; no upstream implementation was copied.

Permanent clean-capture results from `76217b7708d7b503449bf42e556e4c148bda88e6` are recorded in `artifacts/stage-b/` and `docs/status.md`: 14 CTests, 21 native GPU events, 75 CUDA correctness launches and 880 benchmark launches passed. They include actual compiled source/module/runtime identities, every captured shader and command, CPU verifier output, and the post-refactor CUDA regression/benchmark. Actual device-loss/timeout and allocation-exhaustion fault injection are explicitly unqualified. The exact resumable next task is in `docs/handoff.md`.
