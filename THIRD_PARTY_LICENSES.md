# Third-party licenses and dependency inventory

Recorded 2026-09-25. UniCUDA's original implementation is Apache-2.0; see `LICENSE` and `NOTICE`. External documentation references are not imported implementation code. No NVIDIA CUDA toolkit, CuMetal, or MetaXuda dependency is used.

## Selected local development dependencies

The versions below were obtained from the installed tools. LLVM 21.1.8 frontend integration tests have passed locally; this does not itself establish the full source-to-GPU Gate A milestone. Acceptance results are maintained in `docs/status.md` and the Gate A evidence.

| Component | Observed version / identity | Use | License and source |
|---|---|---|---|
| LLVM/Clang/LibTooling | Homebrew LLVM 21.1.8, arm64; `/opt/homebrew/opt/llvm@21` | CUDA AST parsing and compiler libraries; not a runtime dependency | [Apache-2.0 with LLVM exceptions and component notices](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/LICENSE.TXT) |
| CMake / CTest | 4.4.3 | Build configuration and tests | [BSD-3-Clause; additional component notices](https://cmake.org/licensing/) |
| Ninja | 1.13.2 | Build executor | [Apache-2.0](https://github.com/ninja-build/ninja/blob/v1.13.2/COPYING) |
| Python | 3.14.5 selected by CMake | Gate A evidence test; standard library only, not a runtime dependency | [Python Software Foundation license](https://docs.python.org/3/license.html) |
| Apple Clang | 17.0.0, clang-1700.6.3.2 | System C++ / Objective-C++ toolchain as selected by build | Apple toolchain notices and [LLVM licensing](https://llvm.org/docs/DeveloperPolicy.html#copyright-license-and-patents) |
| macOS SDK | 26.2, selected by Xcode | Metal/Foundation headers and public system APIs | [Apple developer agreements](https://developer.apple.com/support/terms/); SDK not redistributed |
| Metal and Foundation frameworks | Provided by macOS 26.5.1, build 25F80 | Device discovery, buffers, runtime shader compilation, GPU submission | Apple system components; not bundled or relicensed |
| C++ standard library / system runtime | Supplied by selected compiler and macOS | Native host/runtime execution | Toolchain/system notices; [LLVM libc++ license](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/libcxx/LICENSE.TXT) where applicable |

The selected LLVM bottle declares minimum macOS 26.0; the current CLI and generated host builds use deployment target 26.0. Metal safe/precise API availability from macOS 15 is not a claim that this CLI supports macOS 15. See `docs/engineering-notes.md`.

The selected LLVM version is a tested-environment choice, not a requirement that all users obtain exactly 21.1.8. Matching LibTooling headers/libraries and a passing parser/build check are required for any alternate version.

Homebrew and the OS may install additional transitive components. This source repository does not vendor their binaries. Before redistributing a compiled compiler or package, inspect its actual linked and bundled dependencies, include all required license texts/notices, and update this inventory for that exact artifact. Do not treat the top-level LLVM license as a substitute for bundled third-party notices.

## Prior art, not dependencies

| Project | Observed upstream license status | UniCUDA use |
|---|---|---|
| [CuMetal](https://github.com/Lulzx/cuda-metal/blob/main/LICENSE) | Apache-2.0 | Documentation comparison only; no implementation imported |
| [ZLUDA](https://github.com/vosen/ZLUDA) | Apache-2.0 or MIT | Documentation comparison only |
| [MetaXuda](https://github.com/Perinban/MetaXuda/blob/main/LICENSE) | Custom restrictions on commercial use and modified redistribution | Research reference only; no code, native libraries, or dependency |
| Other frameworks, APIs, and papers | See `docs/prior-art.md`; each implementation has its own terms | Research reference only |

Review exact licenses before adding dependencies or copying examples. Public documentation and a familiar API name do not grant permission to copy proprietary implementations. If licensed third-party code is later adopted, retain its copyright, provenance, applicable license text, and notices in the same change.
