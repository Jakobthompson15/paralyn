# GLSL and HLSL compute frontends (via the pinned SPIR-V importer → Metal)

Lane: compilers/frontends (handoff task 3, first bullet). Branch `lane/hlsl-glsl-frontends`, based on `cccb177`. Software remains 0.0.1. This document describes an implemented, tested **partial** GLSL and HLSL compute input profile. It does not complete either portfolio row, provides **no GLSL/HLSL → CUDA/HIP path**, and is not a release qualification. `docs/status.md`, `docs/handoff.md`, `docs/portfolio-ledger.json`, `docs/frontend-matrix.md`, `ROADMAP.md` and `README.md` were not edited on this branch; the proposed reconciliation is at the end of this file.

## What exists

```text
foo.comp / foo.glsl ──► glslang worker (vulkan-sdk-1.4.363.0) ──┐   separate processes,
foo.hlsl ──► DXC worker (v1.9.2607, -spirv) ──► StorageBuffer  ─┤   never linked into Paralyn
               --entry / --profile cs_6_x      normalization    │
                                                                ▼
                     SPIR-V 1.3 (Vulkan 1.1, GLCompute) words
                                                                │
         existing importer (docs/spirv-import.md): profile scan ─► spirv-val (Vulkan 1.1)
         ─► SPIRV-Cross reflection/access analysis ─► MSL 3.1 ─► PARALYNX v2 (.prx)
                                                                │
                C ABI / C++ / Python pr_module_load ─► Metal engine ─► physical Apple GPU
```

- `-DPARALYN_ENABLE_GLSL=ON` and `-DPARALYN_ENABLE_HLSL=ON` (default **OFF**; both require `-DPARALYN_ENABLE_SPIRV=ON`, configuration fails otherwise) build the pinned compilers and the `paralyn_shader` library (`compiler/shader/frontends.cpp`, public header `include/paralyn/shader.hpp`), which the `paralyn` CLI links. Default, compiler-disabled and runtime-only builds never read the archives or build either compiler.
- **Isolated compiler workers.** glslang's standalone `glslang` executable and DXC's `dxc` (+ `libdxcompiler.dylib`) are built as isolated Release `ExternalProject`s in `build/_shader/` and are only ever run as child processes: `posix_spawn` with an argument vector (no shell), an **empty environment**, stdin from `/dev/null`, bounded stdout/stderr capture (1 MiB each), a 120 s timeout and a private `mkdtemp` directory holding exactly the source bytes. DXC contains its own fork of LLVM/Clang 3.7; it never shares an address space or ABI with the Clang 21 CUDA frontend. The worker paths are build-tree paths embedded at build time; the workers are **not installed or packaged**.
- The resulting SPIR-V goes through the **unchanged** SPIR-V importer. Everything the importer accepts or rejects (profile, validation, access analysis, generated-MSL profile, runtime re-derivation of access and workgroup from the retained SPIR-V) applies unchanged. Modules produce ordinary `ExecutableFormat::SpirvMsl` (`.prx` v2) containers and load through the unchanged C ABI 1, C++ and Python APIs, on the same Metal engine, queue, retention, timing and evidence path.
- CLI: `paralyn compile|check|inspect|explain FILE.comp|.glsl|.hlsl [--entry NAME] [--profile cs_6_x]`, and `paralyn run|verify|check FILE --entry NAME [--profile …] --case CASE.toml`. `run`/`verify` without a case stay `P-KERNEL-CASE-REQUIRED`. Projects (`paralyn.toml` or `--project`) accept `.comp/.glsl/.hlsl` modules with `entry` (GLSL optional, HLSL required) and `profile` (HLSL only, required). `paralyn support` lists **GLSL compute** and **HLSL compute** as `partial`, `full_profile_qualified: false`, with "no CUDA/HIP lowering" in the scope.

## Pinned compilers

| Compiler | Identity | Source | Build |
|---|---|---|---|
| glslang | tag `vulkan-sdk-1.4.363.0` (16.6.0), commit `e1b562a8bed273a02f30b59b66a5d499793cede5`, archive SHA-256 `907174a2…eb980312` | vendored `third_party/glslang/` (4.6 MB) | `ENABLE_OPT=OFF`, `ENABLE_HLSL=OFF`, `BUILD_EXTERNAL=OFF`; target `glslang-standalone` |
| DXC | tag `v1.9.2607`, commit `0d3ee6b551b8fa768fbf825300ebab81047ef6a8`, archive SHA-256 `36d9383c…2ab5aea`; submodules SPIRV-Headers `29981f65…`, SPIRV-Tools `b707790a…`, DirectX-Headers `980971e8…` (exact commits recorded in the tag) | fetched into `third_party/dxc/` by `scripts/fetch_shader_compilers.py --restore` (29 MB, git-ignored) | DXC's `cmake/caches/PredefinedParams.cmake` + tests off, `LLVM_APPEND_VC_REV=OFF`, `HLSL_ENABLE_FIXED_VER=ON`, `HLSL_SUPPORT_QUERY_GIT_COMMIT_INFO=OFF`; target `dxc` |

- glslang is the same Khronos SDK release as the importer's SPIRV-Tools/Headers (its `known_good.json` names exactly those commits). glslang's own HLSL frontend is disabled; HLSL goes only through DXC.
- All archives are SHA-256-checked by CMake before extraction; no network access during configure/build. `scripts/fetch_shader_compilers.py --download` re-downloads every archive and compares; on 2026-09-29 all five matched. Hashes are also in `third_party/SHA256SUMS.json`; licenses in `third_party/{glslang,dxc}/licenses/`, `third_party/README.md` and `THIRD_PARTY_LICENSES.md` (glslang: BSD/MIT/Apache-2.0 plus GPL-3.0-or-later WITH Bison-exception-2.2 for the generated parser; DXC: NCSA plus `ThirdPartyNotices.txt`).
- **Observed and fixed contamination:** the first DXC build embedded the enclosing Paralyn checkout's commit (`dxc --version` printed `libdxcompiler.dylib: 1.9(60-cccb1778)`; DXC's bundled SPIRV-Tools reported `v2026.3 cccb1778…`), because DXC's `utils/GetCommitInfo.py` and SPIRV-Tools' version script query the surrounding repository. With the options above it prints `libdxcompiler.dylib: 1.9(1.9.2607.0)`, the SPIRV-Tools identity names the pinned DXC submodule, and neither binary contains the Paralyn revision. glslang contains none.
- Build cost on the M5 with `-j 4`: glslang about 1 minute, DXC about 7 minutes (1,326 steps), both once per build tree.
- Every compiled container's `toolchain` field records the SPIR-V toolchain, the frontend's pinned identity, the SHA-256 of the worker executable (and of `libdxcompiler.dylib`), the source file name and SHA-256, the SHA-256 of the SPIR-V exactly as the worker wrote it, any normalization, and the exact worker arguments (source path shown as the file name). The CLI report adds `shader.{language, compiler, toolchain, worker_arguments, entry, profile, target, compiler_spirv_sha256, normalization, warnings}`.

## Accepted profile

Both languages are limited by the SPIR-V profile in [spirv-import.md](spirv-import.md): one GLCompute entry, fixed workgroup, set-0 storage buffers each holding one 32-bit `float`/`int`/`uint` runtime or fixed array, 32-bit scalar members of Uniform (`cbuffer`/`uniform` block) and push-constant blocks, the six compute builtins, structured control flow, workgroup (`shared`/`groupshared`) memory and barriers.

| | GLSL | HLSL |
|---|---|---|
| Worker invocation | `glslang --target-env vulkan1.1 -S comp --error-column --quiet [-e NAME --source-entrypoint main] -o OUT SOURCE` | `dxc -spirv -fspv-target-env=vulkan1.1 -T cs_6_x -E NAME -HV 2021 -Fo OUT SOURCE` |
| Stage | always compute (`-S comp`), any `.comp`/`.glsl` file | compute profiles `cs_6_0` … `cs_6_8` only (`cs_6_0`, `cs_6_6`, `cs_6_8` tested) |
| Entry | the GLSL function is always `main`; `--entry NAME` renames the SPIR-V/MSL entry point (without it SPIRV-Cross names it `main0`) | `--entry NAME` required; the MSL name is the HLSL name unless SPIRV-Cross must rename it (e.g. a function called `transpose` becomes `_transpose`; the example therefore uses `transpose_tiled`) |
| Buffers | `layout(set = 0, binding = N) [readonly\|writeonly] buffer B { float data[]; } name;` (named instance required; `std430`/`std140` for 32-bit scalar arrays are identical) | `[[vk::binding(N, 0)]] StructuredBuffer<float>` / `RWStructuredBuffer<float>` (also `int`/`uint`) |
| Scalars | `layout(push_constant) uniform P { uint n; … } p;` or a `uniform` block at set 0 | `[[vk::push_constant]] struct P { … } p;` or `[[vk::binding(N, 0)]] cbuffer` |
| Parameter names | buffer instance names and block member names | variable names and `cbuffer`/push-constant member names |

**HLSL StorageBuffer normalization.** DXC emits structured buffers in the Vulkan 1.1 / SPIR-V ≤ 1.3 legacy form `Uniform` + `BufferBlock` (its `RemoveBufferBlockVisitor` only switches to `StorageBuffer` at SPIR-V ≥ 1.4, which the importer profile does not accept). The importer rejects that form (`spirv.resource`). `paralyn::shader::normalize_storage_buffers` rewrites it exactly: variables whose pointee struct is decorated `BufferBlock` move to the `StorageBuffer` class with `Block`, and every pointer derived from them through `OpAccessChain`/`OpInBoundsAccessChain`/`OpPtrAccessChain`/`OpCopyObject` gets a `StorageBuffer` pointer type (reused or newly declared right after the Uniform pointer type it replaces). `cbuffer` data (`Uniform` + `Block`) is untouched. It **fails closed** (`hlsl.legalization`) for any other use of such a pointer — function-call arguments, `OpSelect`/`OpPhi`, stored pointer values, pointers used as indices, a `BufferBlock` struct also backing a Uniform variable — and reports atomics with the importer's `spirv.instruction`. The rewritten module is then validated by the pinned `spirv-val` and the full importer; the container retains the normalized SPIR-V and records the DXC output's SHA-256 separately. GLSL output is never rewritten (the retained SPIR-V hash equals glslang's output hash; tested).

**Not accepted** (tested, stable ids): `#include` (the exact hashed source bytes are compiled alone, so includes are rejected before compilation — conservatively, also inside comments), images/textures/samplers, 64-bit float and integer types, 16/8-bit types, subgroup/wave operations, atomics, non-compute HLSL profiles or shader-model 5.x/7.x/`cs_6_9`, and everything else the SPIR-V profile rejects. Specialization constants, descriptor arrays, non-zero descriptor sets and `OpArrayLength` (HLSL `GetDimensions`, GLSL `.length()` on runtime arrays) are rejected by the importer as before.

## Diagnostics

Frontend errors are `paralyn::shader::CompileError` with a stable `code`; `what()` is `ParalynError: <GLSL|HLSL> frontend [<code>]: <detail>`. The CLI maps `glsl.X`/`hlsl.X` to `P-GLSL-X`/`P-HLSL-X` and adds `diagnostic.source_diagnostics[]` = `{severity, file, line, column, message}` located in the user's file (the worker's temporary path is replaced by the source file name). Importer rejections keep their `P-SPIRV-*` ids with stage `import` and name the frontend and file. No output file is written on any failure.

| Id | Stage | Meaning |
|---|---|---|
| `P-GLSL-COMPILE`, `P-HLSL-COMPILE` | compilation | the pinned compiler rejected the source; located (glslang `file:line:column`, DXC `file:line:column`) when the compiler reports a location, e.g. an undeclared identifier (GLSL line 9 col 23; HLSL line 7 col 24), a vertex entry compiled with `cs_6_0` (DXC: "compute entry point must have a valid numthreads attribute", line 3 col 8), a missing entry (unlocated) |
| `P-GLSL-INCLUDE`, `P-HLSL-INCLUDE` | input | `#include` directive (located by line) |
| `P-HLSL-PROFILE` | input | missing profile, or not `cs_6_0`…`cs_6_8` (e.g. `ps_6_0`, `vs_6_0`, `cs_5_0`, `cs_6_9`, `lib_6_3`, `CS_6_0`) |
| `P-GLSL-PROFILE` | input | library-level: a profile given for GLSL (the CLI reports `P-PROFILE-NOT-APPLICABLE` first) |
| `P-PROFILE-NOT-APPLICABLE` | input | `--profile` on anything but `.hlsl` (GLSL, `.prx`, `.metal`, `.cu`, …) |
| `P-HLSL-ENTRY`, `P-GLSL-ENTRY` | input | HLSL without `--entry`; an entry that is not an identifier of ≤ 64 characters |
| `P-GLSL-INPUT`, `P-HLSL-INPUT` | input | empty, > 1 MiB, NUL-containing or invalid/overlong UTF-8 source |
| `P-GLSL-WORKER`, `P-HLSL-WORKER` | compilation | worker could not start, crashed, timed out, or wrote no/oversized SPIR-V |
| `P-HLSL-LEGALIZATION` | compilation | DXC output the StorageBuffer normalization cannot retype exactly (fail closed) |
| `P-GLSL-UNAVAILABLE`, `P-HLSL-UNAVAILABLE` | input | the build does not include that frontend |
| `P-SPIRV-*` | import | unchanged importer rejections, e.g. `P-SPIRV-RESOURCE` (images/textures), `P-SPIRV-CAPABILITY` (`Float64`, `Int64`, `GroupNonUniform*`), `P-SPIRV-INSTRUCTION` (atomics) |

## Examples and kernel cases

- `examples/glsl/vector_add.comp`; `examples/glsl/blur_rows.comp` + `blur_columns.comp`: a separable (2·radius+1)-tap blur of a row-major W×H FP32 image with a weight buffer, clamp-to-edge, 16×16 workgroups, taps accumulated `k = -r..r` with separate multiply and add.
- `examples/hlsl/vector_add.hlsl`; `examples/hlsl/transpose.hlsl` (`transpose_tiled`, 16×17 `groupshared` tile, `cbuffer` shape at binding 2); `examples/hlsl/reduce_sum.hlsl` (256-thread `groupshared` tree reduction, push-constant `n`/`scale`).
- Kernel cases in `examples/cases/`: `glsl_vector_add.toml`, `glsl_blur_rows.toml`, `glsl_blur_columns.toml` (pass 2 consumes the pass-1 reference, so the two cases chain the pipeline), `hlsl_vector_add.toml`, `hlsl_transpose.toml` (`transpose-f32` builtin), `hlsl_reduce_sum.toml` (`block-sum-f32` builtin, `group_size = 256`), and the project `shaders.toml`. The blur data and references (`data/blur_*.f32`) are produced independently by `examples/cases/generate_data.py` with FP32 rounding after every multiply and add in the shader's tap order; the pre-existing data files are unchanged (`generate_data.py --check` passes).

```sh
paralyn verify --project examples/cases/shaders.toml --case glsl-blur-rows
paralyn verify examples/hlsl/transpose.hlsl --entry transpose_tiled --profile cs_6_0 \
  --case examples/cases/hlsl_transpose.toml
```

## Numerics

Numerical policy 1 as for the SPIR-V profile: Metal 3.1 safe math, precise functions, contraction off (enforced by the engine and the generated-MSL profile). This is stricter than GLSL/HLSL/Vulkan, which permit contraction and relaxed precision. With it, FP32 multiply and add are correctly rounded, so the blur is **bit-identical** to an independent FP32 CPU reference that uses the shader's operation order — including a normalized Gaussian whose weights are not exactly representable and three chained blur iterations. Against a double-precision reference the observed maximum absolute error over all blur shapes was **1.30 × 10⁻⁷** (values in [0, 1), required ≤ 10⁻⁵); single-iteration binomial blurs are exact against double. Vector add, transpose and the reductions use exactly representable data and are compared exactly. Denormals, transcendental functions, division precision, signed-integer overflow, `precise`/`RelaxedPrecision` semantics and HLSL `min16float` are **not** qualified.

## Evidence (Apple M5, macOS 26.5.1 build 25F80, Apple clang 17.0.0, CMake 4.4.3, Ninja 1.13.2, 2026-09-29, this branch)

Debug build with `-DPARALYN_ENABLE_SPIRV=ON -DPARALYN_ENABLE_GLSL=ON -DPARALYN_ENABLE_HLSL=ON`, LLVM 21.1.8. Development captures in temporary directories, not archived clean-revision qualification evidence. No benchmark or timing qualification was run (the GPU was shared).

- `shader_frontends` (CPU): 9 example compilations imported (GLSL with and without `--entry`; HLSL at `cs_6_0`, `cs_6_6`, `cs_6_8`) with the expected entry names, workgroups, parameter names and access; pinned toolchain/worker/source hashes in every container; worker output reproducible byte for byte; normalization only for DXC and idempotent; 37 stable rejections: a hand-written DXC-style Uniform+BufferBlock module rejected as-is (`spirv.resource`) and accepted after normalization with the `cbuffer` left in place; fail-closed normalization for a buffer pointer passed to a function and for `OpSelect` of buffer pointers; an atomic on a structured buffer → `spirv.instruction`; located GLSL/HLSL compile errors; `#include`; nine rejected profiles; entry and input errors; nine unsupported-feature fixtures. 212 checks.
- `shader_cli`: 6 example compilations with provenance checks; `check` on sources and on the produced `.prx` passes Metal compilation/reflection with `gpu_work_submitted=false`; `inspect` does not compile for a device; 41 stable CLI diagnostics (15 negative fixtures × `compile`/`check` with no output written and located lines where applicable, profile/entry errors, `P-PROFILE-NOT-APPLICABLE` for GLSL and `.prx`, `run`/`verify` without a case); `support` rows.
- `shader_metal` (physical GPU): C ABI driver — **GLSL 33 GPU events** (vector add n = 1, 63, 64, 65, 1000, 262147, 1000003 with canaries; separable blur pipeline on 1×1 (r 3), 2×3 (r 3, 2 iterations), 17×5, 64×64 (3 iterations), 333×257 (r 3, 2 iterations), 1920×1080, 1023×769 (r 4, 3 iterations): 26 dispatches queued per shape without host synchronization, ping-pong buffers) and **HLSL 20 GPU events** (vector add at the same seven sizes; tiled transpose 1×1, 16×16, 17×33, 1000×3, 3×1000, 513×1025, 2048×1024 with canaries; reduction n = 1, 255, 256, 257, 100000, 1000003, scales 1, 0.5, −2), **8,111,154 values** compared with independent CPU references, 6 launch-contract rejections (workgroup, scalar type, access, in-place writable alias, argument count) with no event returned and inputs unchanged. Python binding: 5 GPU events (two-dispatch 97×61 blur, HLSL vector add n = 1, 65, 10007), 15,990 values compared, 1 rejection. All events completed with valid Metal system-clock timestamps; `execution.json` exported.
- `shader_cases` (physical GPU): all six cases pass `paralyn verify` through `shaders.toml` and four directly with `--entry/--profile` (10 verified GPU executions plus one `run`), `check --case` binds without dispatch, 7 refusals (entry mismatch, conflicting `--profile`, GLSL case without `--entry` exposing `main0`, HLSL case without profile, and three project-schema errors).
- Full `ctest -j1` of this SPIR-V + GLSL + HLSL Debug build: **40/40 passed** (36 pre-existing including Gate A, Gate B correctness, `kernel_cases`, `spirv_*`, `tensors_*`, plus the 4 `shader_*` tests), no skips. A separate default build (no SPIR-V/GLSL/HLSL) passed its **33/33** tests and reports `P-GLSL-UNAVAILABLE` for a `.comp` input and `not_implemented` support rows. A runtime-only build (`-DPARALYN_BUILD_COMPILER=OFF`, no LLVM, SPIR-V, glslang or DXC linked) Metal-checked a GLSL-derived and an HLSL-derived `.prx` (`backend_compilation: passed`, no dispatch).

## Explicitly not provided

- **No GLSL/HLSL → CUDA/HIP lowering.** The only lowering is SPIRV-Cross → MSL on Metal. A shader-to-CUDA/HIP bridge remains a separate required design task (handoff task 3) and needs NVIDIA/AMD hardware, which is blocked by the platform decision.
- No vertex/fragment/mesh/ray-tracing stages, no graphics pipeline, no images/textures/samplers, no 16/64-bit types, no subgroup/wave operations, no atomics, no `#include`, no preprocessor defines from the command line, no multiple entry points per module, no specialization constants, no HLSL root signatures, no DXIL output, no glslang HLSL mode.
- Workers are built from source per build tree and are not installed, packaged, or included in the wheel; an installed CLI would reference the build tree. No worker sandbox beyond process isolation (trusted local input only, like the rest of the CLI).
- Only the Apple M5 on macOS 26.5.1 was exercised. The glslang/DXC builds were not tried on Linux (the Linux container build remains the no-backend build); DXC on Linux is upstream-supported but unverified here. No other Apple GPU, no performance claim, no clean-revision archived capture.
- `cs_6_9` and later shader models are refused although DXC 1.9 may accept some of them; only `cs_6_0`/`cs_6_6`/`cs_6_8` were exercised.
- Include support, multi-file shaders, and mapping importer-level rejections back to source lines (e.g. via `OpLine` debug info) are not implemented.

## Proposed shared-document edits (for the integrator)

**docs/status.md** — add a section at the top:

> ### GLSL/HLSL compute frontends — 2026-09-29 (development evidence, branch `lane/hlsl-glsl-frontends`)
> Optional `-DPARALYN_ENABLE_GLSL=ON` / `-DPARALYN_ENABLE_HLSL=ON` (require SPIR-V). Pinned glslang vulkan-sdk-1.4.363.0 (vendored) and DXC v1.9.2607 (fetched, SHA-256-pinned) run as isolated worker processes; their SPIR-V goes through the unchanged SPIR-V importer to Metal. DXC's Uniform+BufferBlock structured buffers are normalized to StorageBuffer (fail closed). GLSL vector add and a multi-iteration separable blur pipeline, HLSL vector add, groupshared tiled transpose and tree reduction run on the physical M5: 58 GPU events in the C ABI/Python tests plus 11 kernel-case executions, 8.1 M values compared with independent CPU references (blur bit-identical to an ordered FP32 reference; max |error| vs double 1.3e-7). Source-located compile diagnostics and stable ids; images, 64-bit, subgroup/wave, atomics, `#include` and non-`cs_6_x` profiles are rejected. No GLSL/HLSL → CUDA/HIP path. Debug development run, not an archived capture; benchmark not rerun.

**docs/handoff.md** — in "Compilers/frontends" replace "Put HLSL (DXC) and GLSL (glslang) on the pinned SPIR-V importer." with:

> - GLSL (glslang) and HLSL (DXC) compute are on the pinned SPIR-V importer (`docs/hlsl-glsl-frontends.md`). Next: capture them in the clean qualification run (fetch DXC first: `python3 scripts/fetch_shader_compilers.py --restore`); build the workers in the Linux container; map importer rejections to source lines via `OpLine`; decide whether to widen the SPIR-V profile (e.g. `OpArrayLength`, vectors in buffers) for common shader idioms.
> - Design the shader-to-CUDA/HIP bridge (still open; GLSL/HLSL now feed the same SPIR-V entry point).

and in "Qualification" add `shader_*` to the suites to capture.

**docs/portfolio-ledger.json** — for the "GLSL compute" and "HLSL compute" rows: status `partial` (was not implemented), backend `metal`, evidence `docs/hlsl-glsl-frontends.md` + tests `shader_frontends`, `shader_cli`, `shader_metal`, `shader_cases`; `full_profile_qualified: false`; notes "via pinned glslang vulkan-sdk-1.4.363.0 / DXC v1.9.2607 worker → SPIR-V 1.3 → SPIRV-Cross MSL; no CUDA/HIP; buffers/scalars only".

**docs/frontend-matrix.md** — GLSL compute and HLSL compute: "partial (Metal via SPIR-V import); CUDA/HIP not provided".

**README.md / ROADMAP.md** — mention GLSL/HLSL compute as partial optional frontends in the input list; ROADMAP stage for shader frontends: GLSL/HLSL done at the development-evidence level, bridge to CUDA/HIP remaining.
