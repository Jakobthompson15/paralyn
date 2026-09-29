# Vendored dependencies

## CLI

These copies are build inputs; no network fetch occurs during configuration or runtime.

- CLI11 **2.4.2**, single header from <https://github.com/CLIUtils/CLI11/releases/download/v2.4.2/CLI11.hpp>, BSD-3-Clause, license in `cli11/LICENSE`.
- nlohmann/json **3.11.3**, single header from <https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp>, MIT, license in `nlohmann/LICENSE.MIT`.
- toml++ **3.4.0**, single header from <https://raw.githubusercontent.com/marzer/tomlplusplus/v3.4.0/toml.hpp> (tag commit `30172438cee64926dc41fdd9c11fb3ba5b2ba9de`), MIT, license in `tomlplusplus/LICENSE`. Git blob hashes of both files were checked against the upstream tag.

`SHA256SUMS.json` pins the actual checked-in bytes. All three are used only by the CLI. The native runtime and Python binding do not require any of them. Future FTXUI components are not yet imported.

## Optional SPIR-V import toolchain (`-DPARALYN_ENABLE_SPIRV=ON`)

`spirv/` holds the exact upstream source archives of one coherent Khronos release set, **vulkan-sdk-1.4.363.0**, each fetched by immutable commit (never a floating branch) from `https://codeload.github.com/KhronosGroup/<repo>/tar.gz/<commit>`:

| Project | Upstream commit | Archive SHA-256 | License |
|---|---|---|---|
| SPIRV-Headers | `496543121ce6419f23d6fa5d7194ba66c36212d2` | `a9bb9c48713245eacf97cc539b6f1d45405a92d8813f8b82e635f0a085ee9898` | MIT-style Khronos license (`spirv/licenses/SPIRV-Headers/`) |
| SPIRV-Tools (v2026.4) | `ef96ed763b43b59b33b31b362f09a02b729fa1c9` | `82c62146083fd558735a3171cf97cfc47903ca7d368482e87f94bd44883c0f00` | Apache-2.0 (`spirv/licenses/SPIRV-Tools/`) |
| SPIRV-Cross | `f11ba9f0b21ba8fc15153d50a2a1ae31ab1cf8f7` | `92b0458889ed77eac9892b3be5a41e3cf716849fd90f39632e159e555abf3d03` | Apache-2.0 (core sources are dual Apache-2.0 OR MIT; `spirv/licenses/SPIRV-Cross/`) |

`cmake/ParalynSpirv.cmake` checks these SHA-256 values, extracts the archives into the build tree and builds them as isolated `ExternalProject`s (Release; SPIRV-Tools `SPIRV-Tools-static`, `spirv-as`, `spirv-val`, `spirv-dis`; SPIRV-Cross `core`, `glsl`, `msl`). No network access occurs during configuration or build. SPIRV-Tools receives `FORCED_BUILD_VERSION_DESCRIPTION` so its embedded version names the pinned upstream commit rather than the enclosing Paralyn checkout. `scripts/fetch_spirv_sources.py [--download]` re-verifies the vendored bytes and, optionally, compares a fresh download of each pinned commit. The runtime-only and default builds never read these archives; generated modules execute without any SPIR-V library.

## Optional GLSL/HLSL compiler workers (`-DPARALYN_ENABLE_GLSL=ON`, `-DPARALYN_ENABLE_HLSL=ON`)

Both require `-DPARALYN_ENABLE_SPIRV=ON`. Each compiler is built by `cmake/ParalynShaderCompilers.cmake` as an isolated Release `ExternalProject` and runs only as a separate worker process; neither is linked into a Paralyn library or executable. Every archive below is fetched by immutable commit from `https://codeload.github.com/<owner>/<repo>/tar.gz/<commit>` and checked against its SHA-256 before extraction; configuration and build never download.

| Project | Upstream commit | Archive SHA-256 | Location | License |
|---|---|---|---|---|
| glslang (tag `vulkan-sdk-1.4.363.0`, 16.6.0) | `e1b562a8bed273a02f30b59b66a5d499793cede5` | `907174a24713c6202c146f164bf81783f1fbc79c8cb821a30f18f159eb980312` | vendored: `glslang/` | BSD-3-Clause, BSD-2-Clause, MIT, Apache-2.0, a Khronos MIT variant, and GPL-3.0-or-later **with the Bison 2.2 exception** for the generated parser (`glslang/licenses/LICENSE.txt`) |
| DirectXShaderCompiler (tag `v1.9.2607`) | `0d3ee6b551b8fa768fbf825300ebab81047ef6a8` | `36d9383cfcb1a189efbadf181c81719ab329bd940b2f417cc5ba2d39a2ab5aea` | fetched: `dxc/` | University of Illinois/NCSA (LLVM 3.7 base) plus the notices in `dxc/licenses/DirectXShaderCompiler/ThirdPartyNotices.txt` |
| SPIRV-Headers (DXC `external/` submodule at that tag) | `29981f65241605e08b0ede4cfeb999fe3b723c6a` | `232899f1ad4104fb5bc377b94596c7621575eee62ad9a9e8f929b63a7dd8a7ad` | fetched: `dxc/` | MIT-style Khronos license (`dxc/licenses/SPIRV-Headers/`) |
| SPIRV-Tools (DXC `external/` submodule at that tag) | `b707790a898e44038547df54580022fc1cf89c3d` | `05d8af89737bde57571c48dbd36714c9f520a69623e14de72c3be6b600e277d6` | fetched: `dxc/` | Apache-2.0 (`dxc/licenses/SPIRV-Tools/`) |
| DirectX-Headers (DXC `external/` submodule at that tag) | `980971e835876dc0cde415e8f9bc646e64667bf7` | `b5a4b6d8806ff7f29f19879f83d015dbe8740676d4ca0b48647a789cc7773c4e` | fetched: `dxc/` | MIT (`dxc/licenses/DirectX-Headers/`) |

- glslang's `known_good.json` at that tag names exactly the SPIRV-Tools/SPIRV-Headers commits pinned for the SPIR-V importer. glslang is built with `ENABLE_OPT=OFF` (no spirv-opt) and `ENABLE_HLSL=OFF` (its own HLSL frontend is not used; HLSL goes through DXC only); only the `glslang` standalone executable is built.
- The DXC set (about 29 MB) is **not committed**. `python3 scripts/fetch_shader_compilers.py --restore` downloads the four DXC archives into `dxc/` (git-ignored) and keeps each only when its SHA-256 matches; `--download` re-downloads everything and compares without writing. On 2026-09-29 fresh downloads of all five archives matched the pinned hashes. The SHA-256 values are also listed in `SHA256SUMS.json`; the DXC `.tar.gz` entries there describe fetched, not checked-in, bytes.
- DXC is configured with its documented `cmake/caches/PredefinedParams.cmake` plus tests off, `LLVM_APPEND_VC_REV=OFF`, `HLSL_ENABLE_FIXED_VER=ON` (its in-tree `utils/version/version.inc`, 1.9.2607.0) and `HLSL_SUPPORT_QUERY_GIT_COMMIT_INFO=OFF`; its bundled SPIRV-Tools receives `FORCED_BUILD_VERSION_DESCRIPTION`. Without these, the first build embedded the enclosing Paralyn checkout's commit (`dxc --version` printed `1.9(60-cccb1778)`); it now prints `1.9(1.9.2607.0)`.
- The license files under `glslang/licenses/` and `dxc/licenses/` are unmodified upstream copies taken from the pinned archives.
