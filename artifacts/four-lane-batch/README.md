# Four-lane batch qualification — 2026-09-29

This capture covers the merged four-lane batch: tensors/matmul/MLP, projects and kernel cases, SPIR-V import, and the host-only CUDA backend. It also repeats the earlier product, native, Gate A/B and benchmark captures on the same revision. It is **not completion of the all-frontends platform** and is not a release tag. Software remains 0.0.1.

Every capture here used clean revision `cccb17788046dad66ab5c08e0c0c3f145d6865ca` (`main`, already pushed). An evidence-only commit archives these bytes afterward. The capture ran in an isolated git worktree on branch `lane/capture-cccb177`. `git status --porcelain` was empty before and after each step. Build directories (`build/`) and run directories (`artifacts/runs/`) are git-ignored. `gate-a/`, `paralyn-gate-a/`, `gate-b/`, `stage-b/`, `product-foundation/` and `benchmark-log-isolation/` are unchanged.

**Hardware and toolchain:** physical Apple M5 (10-core GPU, Metal 4, 16 GiB unified memory), macOS 26.5.1 (25F80), SDK 26.2. Compilers were Apple Clang 17.0.0 (clang-1700.6.3.2) and Homebrew LLVM/Clang 21.1.8 (`llvm@21`), with CMake 4.4.3 and Ninja 1.13.2. The pinned SPIR-V toolchain was SPIRV-Tools/Cross/Headers `vulkan-sdk-1.4.363.0`, built fresh inside the build directory from the vendored archives. Python was 3.14.5 (Homebrew) for scripts, with Python 3.9.6 (`/usr/bin/python3`) as a second install target. Dependencies were already installed. [environment.json](environment.json) has the full record.

## Recorded results

| Capture | Result | Evidence |
|---|---|---|
| Fresh Debug build, `-DPARALYN_ENABLE_SPIRV=ON` | configure + build passed (build directory created fresh) | [build-validation.json](build-validation.json), [configure](configure.log), [build](build.log) |
| Full regression suite, `ctest -j1` | **36/36 passed**, no skips or not-run tests | [ctest.log](ctest.log) |
| Product qualification | **53 GPU events** (50 source-linked + 3 retained-output); 60 CLI command cases | [qualification](product/qualification.json), [audit](product-audit.log), [corruption tests](audit-corruption.log) |
| Preserved ABI1 native qualification | **21 GPU events** (C 2, C++ 1, native tests 8, Python tests 9, Python example 1) | [qualification](native/qualification.json) |
| Native applications through build CLI | **4 GPU events** | [verification](native-cli/verification.json) |
| CUDA Gate A | 1 launch, 1,024 elements independently compared | [gate-a](gate-a/) |
| CUDA Gate B correctness | **75 launches**: 66 positive (46 + 3 + 9 + 5 + 3), plus 9 in the intentional exit-37 case | [summary](cuda/summary.json) |
| Full benchmark | **880 launches**: 4 sizes × 2 variants × (10 warmups + 100 measured); every output compared | [capture/audit](benchmark/capture.json), [samples](benchmark/samples.csv) |
| Tensors, C++ (`tensor_tests`) | **47 source-linked GPU commands** (38 tiled matmul); the driver separately reports 4 high-level events | [execution](tensors/cpp/execution.json), [log](tensors-cpp-evidence.log) |
| Tensors, Python (`test_tensors.py`) | **37 GPU commands** | [execution](tensors/python/execution.json) |
| MLP, C++ and Python | **4 + 4 GPU commands**; identical output SHA-256 `3ecb2b1d…` and provider SHA-256 `2b69286f…` | [C++](tensors/mlp-cpp/mlp-report.json), [Python](tensors/mlp-python/mlp-report.json) |
| SPIR-V on Metal (`run_spirv.py --mode gpu --keep`) | **17 C++ GPU events** (retained) + **6 Python GPU events** (stdout only); 1,467,859 + 12,511 values compared; 14 + 2 rejections | [evidence](spirv/metal/evidence/execution.json), [log](spirv-metal-evidence.log) |
| SPIR-V CLI (`--mode cli --keep`) | 4 fixture imports, 16 stable diagnostics, no GPU dispatch | [modules](spirv/cli/), [log](spirv-cli-evidence.log) |
| SPIR-V importer (host) | 174 checks, 58 stable rejections, no GPU work | [log](spirv-importer.log) |
| `kernel_cases` suite (CTest runner, verbose rerun) | **17 GPU events**, 95 negative checks; its evidence directory is temporary and was not retained | [verbose log](ctest-new-suites-verbose.log) |
| Shipped kernel cases, `paralyn verify` with retained evidence | **6 GPU events**, one each for 5 project cases and `ir_affine` on `native-kernels.prk`; 3,287 values compared | [cases/](cases/) |
| Direct doctor, build CLI | 1 GPU event; report carries `source_revision` cccb177, `build_dirty: false` | [report](doctor-build/report.json) |
| Offline wheel | two builds byte-identical (`d4cb114c…`); contains `paralyn/tensor.py`, and the bundled dylib exports `pr_tensor_operators_load/artifact` | [reproducibility](package/reproducibility.json), [wheel](package/) |
| Wheel install, Python 3.14.5 / 3.9.6 (fresh venv outside checkout) | **2 + 2 GPU events** | [3.14](installations/python314/installation.json), [3.9](installations/python39/installation.json) |
| Installed MLP example outside checkout (3.14.5 / 3.9.6) | **4 + 4 GPU events**; same output hash as the in-tree runs | [3.14](installations/mlp-python314/), [3.9](installations/mlp-python39/), [discovery](installed-mlp-discovery-2.log) |
| Relocated `cmake --install` CLI | **4 GPU events** for native applications, plus a **1-event doctor** carrying cccb177 / clean | [apps](installations/relocated-cli/verification.json), [doctor](installations/relocated-doctor/report.json) |
| Runtime-only build (`-DPARALYN_BUILD_COMPILER=OFF`) | doctor **1 GPU event** (compiler unavailable, provenance cccb177 / clean); kernel cases `vector-add` (MSL) and `ir-affine` (`.prk`) **2 GPU events**; no LLVM linkage | [doctor](runtime-only-doctor/report.json), [cases](runtime-only-cases/), [linkage](runtime-only-linkage.log) |

Counts are separate runs. The CTest executions, including the 9/9 verbose rerun of `tensors_*`, `kernel_cases` and `spirv_*`, repeat work covered by the direct captures, so do not add them to the totals. Outside CTest, the 42 retained `execution.json` records audit as completed Metal commands with valid GPU durations, `cpu_fallback: false`, revision `cccb177` and `dirty: false`.

**Build provenance is now embedded.** Every retained record carries revision `cccb177` and `dirty: false` from the embedded build information. This includes the relocated-installation doctor, the runtime-only doctor and the installed-wheel MLP. The product-foundation limitation (`paralyn_commit: unknown` in direct doctor reports) no longer applies. [installation-builds.json](installation-builds.json) also hashes the exact installed and runtime-only binaries.

## Benchmark conditions

GPU conditions: no other intentional GPU workload was running. Thermal state was 0, and low-power mode was off. Host conditions: the host was **not idle**. Load average was about 6.8 before the run. Interactive applications were active, including a Loom helper at about 90% CPU, WindowServer at about 25% and Chrome helpers. Snapshots are in `environment.json`. Descriptive medians of measured GPU duration were 2.52 µs at 1,024 elements, 4.62 µs at 65,536, 119/118 µs at 1,048,576 and 1.75 ms at 16,777,216 (generated/handwritten). No speedup or performance threshold is claimed. Use these samples as protocol and correctness evidence, not as a controlled performance baseline.

## Failures and anomalies

- `installed-mlp-discovery` (exit 1) failed because of a bug in the harness's own probe script, not in the product. `paralyn.tensor` is re-exported as a function, so `paralyn.tensor.__file__` does not exist. The corrected probe (`installed-mlp-discovery-2`) passed. Both records are retained.
- No product, qualification or benchmark step failed.

## Not covered by this capture

The no-backend build was not repeated here; the earlier `product-foundation/` record stands. Python SPIR-V events and `kernel_cases` negative-suite evidence exist only as stdout, because their runners use temporary directories. No NVIDIA kernel has executed. The CUDA backend is still host-tested only. Linux/Windows execution, the other 13 input frontends, the three interoperability clients, and framework/operator/release scope all remain unqualified.

## Exact commands

Run from the worktree root. `B=build/capture-cccb177`, `R=artifacts/runs/capture-cccb177`, `PY=/opt/homebrew/bin/python3.14`. Each step's exact argv, cwd, exit code and wall time are in `build-validation.json` and `installation-validation.json`.

```sh
cmake -S . -B $B -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPARALYN_ENABLE_SPIRV=ON
cmake --build $B -j8
ctest --test-dir $B --output-on-failure -j1
$PY scripts/qualify_product.py --build $B --output $R/product --require-clean
$PY scripts/qualify_product.py --output $R/product --audit-only
$PY tests/product_audit_tests.py --capture $R/product
$PY scripts/qualify_native.py --build $B --output $R/native --require-clean
$PY scripts/verify_gate_a.py --paralyn $B/paralyn --source examples/vector_add.cu --artifacts $R/gate-a --require-clean
$PY scripts/qualify_gate_b.py --paralyn $B/paralyn --artifacts $R/cuda --require-clean
$PY tests/native/cli_arrays.py --paralyn $B/paralyn --artifacts $R/native-cli
$PY scripts/verify_benchmark.py --benchmark $B/gate_b_benchmark --paralyn $B/paralyn \
  --source examples/vector_add.cu --artifacts $R/benchmark --require-clean
ctest --test-dir $B -V -j1 -R "^(tensors_|kernel_cases$|spirv_)"
# new suites with retained evidence (PARALYN_LIBRARY=$B/libparalyn_native.dylib, PYTHONPATH=bindings/python)
$B/tensor_tests $R/tensors/cpp
$PY tests/native/test_tensors.py --artifacts $R/tensors/python
$B/native_mlp --artifacts $R/tensors/mlp-cpp
$PY examples/native/mlp.py --artifacts $R/tensors/mlp-python
$PY tests/spirv/run_spirv.py --paralyn $B/paralyn --examples examples/spirv --negative tests/spirv/negative \
  --assembled $B/spirv-fixtures --mode gpu --driver $B/spirv_metal_tests --library $B/libparalyn_native.dylib \
  --python-path bindings/python --keep $R/spirv/metal
$PY tests/spirv/run_spirv.py ... --mode cli --keep $R/spirv/cli
$B/spirv_importer_tests examples/spirv tests/spirv/negative $B/spirv-fixtures
$B/paralyn verify examples/cases/paralyn.toml --case <vector-add|vector-add-inline|block-reduce|transpose|transpose-builtin> \
  --device metal:0 --artifacts $R/cases/<case>
$B/paralyn verify $B/native-kernels.prk --case examples/cases/ir_affine.toml --device metal:0 --artifacts $R/cases/ir-affine
$B/paralyn doctor --device metal:0 --json --artifacts $R/doctor-build
# packaging / installation (installation outputs outside the checkout, then copied here without venvs)
$PY bindings/python/build_wheel.py --compiler $B/paralyn --library $B/libparalyn_native.dylib --module $B/operators.prk --output $R/package
$PY bindings/python/test_install.py --wheel $R/package/paralyn-0.0.1-py3-none-macosx_26_0_arm64.whl --output <outside>/python314
/usr/bin/python3 bindings/python/test_install.py --wheel ... --output <outside>/python39
<outside>/python314/venv/bin/python -I <outside>/mlp-app/mlp.py --artifacts <outside>/mlp-python314   # PYTHONPATH/PARALYN_LIBRARY unset
cmake -S . -B build/runtime-only-cccb177 -G Ninja -DCMAKE_BUILD_TYPE=Debug -DPARALYN_BUILD_COMPILER=OFF -DBUILD_TESTING=OFF
cmake --build build/runtime-only-cccb177 -j8
build/runtime-only-cccb177/paralyn doctor --device metal:0 --json --artifacts $R/runtime-only-doctor
build/runtime-only-cccb177/paralyn verify examples/cases/paralyn.toml --case vector-add --device metal:0 --artifacts $R/runtime-only-cases/vector-add
build/runtime-only-cccb177/paralyn verify $B/native-kernels.prk --case examples/cases/ir_affine.toml --device metal:0 --artifacts $R/runtime-only-cases/ir-affine
cmake --install $B --prefix <outside>/prefix-original   # then copied to <outside>/prefix-relocated
<outside>/prefix-relocated/bin/paralyn doctor --device metal:0 --json --artifacts $R/installations/relocated-doctor
$PY tests/native/cli_arrays.py --paralyn <outside>/prefix-relocated/bin/paralyn --artifacts <outside>/installed-cli
```

`<outside>` was the session scratch directory `/private/tmp/claude-501/.../scratchpad/installed-cccb177`. Embedded absolute paths show where each run happened. They were not rewritten.

## Read-only reproduction audit

```sh
python3 scripts/qualify_product.py --output artifacts/four-lane-batch/product --audit-only
python3 tests/product_audit_tests.py --capture artifacts/four-lane-batch/product
```

Both passed against this archived copy ([archive-audit.json](archive-audit.json)). [SHA256SUMS.json](SHA256SUMS.json) covers every file in this archive except itself.
