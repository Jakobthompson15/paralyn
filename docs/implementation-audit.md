# Native-product batch audit — 2026-09-28

Baseline local and remote main were both `f3c6c9955daf5a81d764fd0e634eb786a8489bcd`, clean, with 14 passing CTests reproduced. The implemented native-product portion of the first batch passed clean qualification at `87978b1f99dab214565c232a406635f4222a1a65`: **23 of 23 CTests**, 53 product GPU events, a repeated 21-event native qualification, CUDA Gates A/B and the full 880-event benchmark. [Build validation](../artifacts/product-foundation/build-validation.json), [product qualification](../artifacts/product-foundation/product/qualification.json) and [installation validation](../artifacts/product-foundation/installation-validation.json) retain the actual commands, identities and evidence. The SPIR-V fixture and Windows inventory remain outstanding. The implementation is in these files/targets:

| Area | Implementation files | Build/tests |
|---|---|---|
| Runtime queries/executables | `include/paralyn/native.h`, `include/paralyn/detail/backend.hpp`, `runtime/native.cpp`, `backends/metal/engine.mm` | native_query_tests, native_queries, original native suites |
| Runtime event channel | `runtime/telemetry.cpp`, `backends/metal/runtime.mm` | telemetry_tests/runtime_telemetry, throwing-log/shutdown regressions |
| Native MSL input | `include/paralyn/executable.hpp`, `compiler/ir/executable.cpp`, `examples/metal/kernels.*` | executable_tests, msl_tests/public_msl,13structured negatives |
| Arrays/operators | `include/paralyn/array.hpp`, `bindings/python/paralyn/array.py`, `bindings/python/operators.cu` | array_operators, array_tests/arrays_cpp, arrays_python |
| Packaging | `bindings/python/build_wheel.py`, `bindings/python/test_install.py`, conditional/install CMake targets | deterministic wheel, offline isolated Python installs and relocated C++ |
| Terminal/process | `cli/main.cpp`, `cli/process_*`, `tests/native/cli_arrays.py`, pinned third_party CLI11/json | terminal_contract (60 command cases), process_execution, native_compile, native_cli_arrays, preserved Gates A/B |
| Inventory | `scripts/windows-inventory.ps1` | read-only script only; Windows not yet qualified |
| Qualification | `scripts/qualify_product.py`, `tests/product_audit_tests.py` | 50 source-linked main records plus three retained-output events, independent references, relocated readback audit and corruption rejection |

Versioned additions preserve ABI1 struct layouts and old artifact bytes. Native MSL pipeline objects and their resource contract are production code distinct from the old test hook. Runtime-only builds have noLLVM dependency. A no-backend build is explicitly unavailable execution, not CPU fallback or a vendor backend.

The 53-event product capture comprises C++ arrays (19 source-linked plus two parent-lifetime events), Python arrays (19 plus one post-context-close event), public MSL (ten events and 13 rejection checks), and two doctor probes. The MSL workloads are four vector-add executions including a read-only alias, three 64-lane reductions and three tiled transposes. The backend's retained dispatched MSL includes its explicit contraction-policy prelude; the audit verifies that exact transformation against the input source. Four additional native CLI array events are captured separately. Application verifier output remains distinct from the CLI's own verification status.

Installation evidence retains reproducible wheel hashes, offline Python 3.9.6/3.14.5 installs with two GPU events each, a relocated installed C++/Python CLI run with four events, an installed doctor event, and a compiler-free runtime doctor event. The two directly invoked doctors retain unknown/null revision fields in their original execution JSON; [installation build provenance](../artifacts/product-foundation/installation-builds.json) establishes their revision and binary identities instead. The no-backend configuration passes host tests and rejects doctor execution as expected. All execution was on the same Apple M5/macOS 26.5.1 host; these are scoped packaging and portability checks, not a multi-platform release qualification.

The other 13 input tracks, all three clients and NVIDIA/AMD backends are still unimplemented. Full arrays/tensor/function/offload/framework/library/application/terminal/release obligations remain in the adopted program. New source files and passing real GPU tests constitute the progress here; updated ledgers do not.

The following prior audit is retained as history.

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
