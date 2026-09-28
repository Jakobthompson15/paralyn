# Vendored dependencies

## CLI

These copies are build inputs; no network fetch occurs during configuration or runtime.

- CLI11 **2.4.2**, single header from <https://github.com/CLIUtils/CLI11/releases/download/v2.4.2/CLI11.hpp>, BSD-3-Clause, license in `cli11/LICENSE`.
- nlohmann/json **3.11.3**, single header from <https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp>, MIT, license in `nlohmann/LICENSE.MIT`.

`SHA256SUMS.json` pins the actual checked-in bytes. Both are used only by the CLI. The native runtime and Python binding do not require either library. Future FTXUI and TOML components are not yet imported.

## Optional SPIR-V import toolchain (`-DPARALYN_ENABLE_SPIRV=ON`)

`spirv/` holds the exact upstream source archives of one coherent Khronos release set, **vulkan-sdk-1.4.363.0**, each fetched by immutable commit (never a floating branch) from `https://codeload.github.com/KhronosGroup/<repo>/tar.gz/<commit>`:

| Project | Upstream commit | Archive SHA-256 | License |
|---|---|---|---|
| SPIRV-Headers | `496543121ce6419f23d6fa5d7194ba66c36212d2` | `a9bb9c48713245eacf97cc539b6f1d45405a92d8813f8b82e635f0a085ee9898` | MIT-style Khronos license (`spirv/licenses/SPIRV-Headers/`) |
| SPIRV-Tools (v2026.4) | `ef96ed763b43b59b33b31b362f09a02b729fa1c9` | `82c62146083fd558735a3171cf97cfc47903ca7d368482e87f94bd44883c0f00` | Apache-2.0 (`spirv/licenses/SPIRV-Tools/`) |
| SPIRV-Cross | `f11ba9f0b21ba8fc15153d50a2a1ae31ab1cf8f7` | `92b0458889ed77eac9892b3be5a41e3cf716849fd90f39632e159e555abf3d03` | Apache-2.0 (core sources are dual Apache-2.0 OR MIT; `spirv/licenses/SPIRV-Cross/`) |

`cmake/ParalynSpirv.cmake` checks these SHA-256 values, extracts the archives into the build tree and builds them as isolated `ExternalProject`s (Release; SPIRV-Tools `SPIRV-Tools-static`, `spirv-as`, `spirv-val`, `spirv-dis`; SPIRV-Cross `core`, `glsl`, `msl`). No network access occurs during configuration or build. SPIRV-Tools receives `FORCED_BUILD_VERSION_DESCRIPTION` so its embedded version names the pinned upstream commit rather than the enclosing Paralyn checkout. `scripts/fetch_spirv_sources.py [--download]` re-verifies the vendored bytes and, optionally, compares a fresh download of each pinned commit. The runtime-only and default builds never read these archives; generated modules execute without any SPIR-V library.
