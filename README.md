# UniCUDA

An independent experimental runtime for a small, explicit CUDA source subset, beginning with Metal on Apple Silicon. It is not a general CUDA replacement. No CPU kernel fallback exists.

The current objective is Gate A: compile a complete ordinary CUDA vector-add program, preserve its host computation, lower its kernel through verified typed IR, execute on a physical Apple GPU, and independently check the output. See [status](docs/status.md) for actual results.

## Build and inspect

The selected environment is Apple M5, macOS 26.5.1, Xcode SDK 26.2, Apple Clang 17, and LLVM/Clang 21.1.8. The installed LLVM bottle requires macOS 26. Matching alternative LibTooling installations must pass the build and parser tests. The Metal runtime uses public runtime shader compilation; the standalone Metal compiler is unnecessary.

```sh
HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1 HOMEBREW_NO_AUTOREMOVE=1 brew install llvm@21 cmake ninja
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_DIR="$(brew --prefix llvm@21)/lib/cmake/llvm" \
  -DClang_DIR="$(brew --prefix llvm@21)/lib/cmake/clang"
cmake --build build -j 4
ctest --test-dir build --output-on-failure
build/unicuda devices
build/unicuda inspect examples/vector_add.cu
build/unicuda run examples/vector_add.cu
```

`run` compiles and executes trusted local source as a native program, with the same host access as launching that program yourself. It does not sandbox host C++.

Specify `--device auto` (default) or a listed index. Pass host arguments after `--`. Each run saves source, IR, generated Metal, execution metadata, and the real program transcript under `artifacts/runs/`. Use `--artifacts DIR` to select an empty directory; existing evidence is never overwritten by the CLI. Intermediate native host code is retained under `build/runs/` for inspection.

## Current boundaries

- Single-file ordinary C++17 host programs using the documented CUDA subset.
- Supported kernel expression/statement nodes are deliberately narrow; unknown behavior fails explicitly.
- Device pointers are opaque base tokens. Host pointer arithmetic or dereference is unsupported.
- Metal only. No NVIDIA/AMD backend, binary/PTX compatibility, framework integration, or distributed execution.
- No persistent cache, benchmark claims, or public release qualification yet.

Read [CUDA compatibility](docs/cuda-compatibility.md), [architecture](docs/architecture.md), [prior art](docs/prior-art.md), and [roadmap](ROADMAP.md). CuMetal and other projects already overlap these goals; novelty is unproven. Original code is Apache-2.0; dependency and platform terms are listed in [third-party licenses](THIRD_PARTY_LICENSES.md).
