# Paralyn

An independent experimental GPU runtime with a small CUDA source subset and native C/C++ and Python interfaces, beginning with Metal on Apple Silicon. It is not a general CUDA replacement. No CPU kernel fallback exists.

**Paralyn's Gate A passed on the physical Apple M5.** The complete ordinary CUDA vector-add program was parsed, its host launch rewritten, its kernel lowered through verified typed IR to MSL, and all 1,024 GPU results independently checked by its CPU verifier. No CPU kernel fallback was used. **Gate B also passed on Apple M5:** all six tests, 75 correctness launches and 880 benchmark launches have [clean-revision evidence](artifacts/gate-b/README.md). No tagged release has been published.

The fresh [execution record](artifacts/paralyn-gate-a/execution.json), [generated Metal](artifacts/paralyn-gate-a/generated.metal), and [verification transcript](artifacts/paralyn-gate-a/verification.txt) preserve a run from a clean Paralyn revision:

```text
$ build/paralyn run examples/vector_add.cu
Paralyn v0.0.1

Device: Apple M5
Backend: Metal
Selection: auto; first device in stable registry-ID order
Kernel: vector_add
Grid: 4 × 1 × 1
Block: 256 × 1 × 1
Compiling kernel...
Executing on GPU...

Verification: PASS (1024 independently checked elements)
```

Earlier evidence remains unchanged in [artifacts/gate-a](artifacts/gate-a/). See [status](docs/status.md) for both revisions and their provenance.

## Build and inspect

The selected environment is Apple M5, macOS 26.5.1, Xcode SDK 26.2, Apple Clang 17, and LLVM/Clang 21.1.8. The installed LLVM bottle requires macOS 26. Matching alternative LibTooling installations must pass the build and parser tests. The Metal runtime uses public runtime shader compilation; the standalone Metal compiler is unnecessary.

```sh
HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1 HOMEBREW_NO_AUTOREMOVE=1 brew install llvm@21 cmake ninja
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_DIR="$(brew --prefix llvm@21)/lib/cmake/llvm" \
  -DClang_DIR="$(brew --prefix llvm@21)/lib/cmake/clang"
cmake --build build -j 4
ctest --test-dir build --output-on-failure
build/paralyn devices
build/paralyn inspect examples/vector_add.cu
build/paralyn run examples/vector_add.cu
```

`run` compiles and executes trusted local source as a native program, with the same host access as launching that program yourself. It does not sandbox host C++.

Specify `--device auto` (default) or a listed index. Pass host arguments after `--`. Each run saves source, `paralyn-ir.txt`, generated Metal, execution metadata, and the real program transcript under `artifacts/runs/`. Use `--artifacts DIR` to select an empty directory; existing evidence is never overwritten by the CLI. Transformed native host code is preserved as `host.cpp` in each evidence directory, together with every dispatched `source-N.metal` referenced by the launch record.

## Native C/C++ and Python

The native ABI now exposes devices, contexts, owned buffers, offset views, verified kernel modules, typed launches, ordered queues, events and structured errors. C++ adds move-only ownership wrappers; Python binds the same shared library with standard-library `ctypes`. Both use the shared Metal engine. [Clean evidence](artifacts/stage-b/README.md) records 21 native GPU events and all 14 current CTests passing, with CUDA correctness and the 880-launch benchmark preserved. See the executable examples in `examples/native/` and the [native API contract](docs/native-api.md).

```sh
build/native_c build/native-kernels.prk artifacts/runs/native-c-new
build/native_cpp build/native-kernels.prk artifacts/runs/native-cpp-new
PYTHONPATH=bindings/python PARALYN_LIBRARY="$PWD/build/libparalyn_native.dylib" \
  python3 examples/native/vector_add.py --module build/native-kernels.prk \
  --artifacts artifacts/runs/native-python-new
python3 scripts/qualify_native.py --build build --output artifacts/runs/native-suite-new
```

The examples independently verify GPU vector addition and/or an affine transform. `paralyn compile SOURCE.cu --output NEW.prk` produces a reusable verified kernel module without executing host code. Native clients need the runtime library, not LLVM at execution time. This is an explicit byte-buffer interface; `paralyn.asarray`, tensor operators and arbitrary Python kernel compilation are not implemented.

## Current boundaries

- Single-file ordinary C++17 host programs using the documented CUDA subset.
- Supported kernel expression/statement nodes are deliberately narrow; unknown behavior fails explicitly.
- CUDA compatibility pointers are opaque base tokens. Native views separately support checked allocation offsets and retained ownership.
- Metal only. No NVIDIA/AMD backend, binary/PTX compatibility, framework integration, or distributed execution.
- No persistent cache or cross-vendor/performance claim. See the qualification and benchmark evidence for the tested Metal subset.

Read [CUDA compatibility](docs/cuda-compatibility.md), [architecture](docs/architecture.md), [prior art](docs/prior-art.md), and [roadmap](ROADMAP.md). CuMetal and other projects already overlap these goals; novelty is unproven. Original code is Apache-2.0; dependency and platform terms are listed in [third-party licenses](THIRD_PARTY_LICENSES.md).

## Qualification

`ctest` runs compiler/runtime negatives, artifact/compile tests, native C/C++/Python tests, Gate A, and the Gate B correctness suite on actual Metal hardware. Hardware absence fails qualification. Run the benchmark separately with an otherwise idle GPU:

```sh
python3 scripts/qualify_gate_b.py --paralyn build/paralyn --artifacts artifacts/runs/correctness-new
python3 scripts/verify_benchmark.py --benchmark build/gate_b_benchmark --paralyn build/paralyn \
  --source examples/vector_add.cu --artifacts artifacts/runs/benchmark-new
```

Use a fresh artifact directory each time. Add `--require-clean` for permanent milestone captures, saving under ignored `artifacts/runs/` or outside the checkout before archiving the results. The [benchmark protocol](benchmarks/README.md) explains the measured regions and limits; [numerics](docs/numerics.md) defines the FP32 policy.

The deprecated `build/unicuda` command and `unicuda/*.hpp` namespace aliases forward to Paralyn during the naming transition. New code should use Paralyn. The [v1.1 portfolio](docs/frontend-matrix.md) tracks required future work; unimplemented rows are not advertised as supported.
