# SPIR-V import profile (Vulkan 1.1 GLCompute → Metal)

Lane: compilers/frontends, handoff task 2. Branch `lane/spirv-metal-import`, based on `4f8fb22`. Software remains 0.0.1. This document describes an implemented, tested **partial** SPIR-V input profile. It does not complete the SPIR-V portfolio row, does not provide any NVIDIA/AMD path and is not a release qualification.

## What exists

```text
foo.spvasm ──(pinned SPIRV-Tools assembler, Vulkan 1.1 env)──┐
foo.spv ─────────────────────────────────────────────────────┤
                                                             ▼
   Paralyn profile scan ─► spirv-val (Vulkan 1.1) ─► SPIRV-Cross reflection
   (Kernel model, version,     (pinned)                + static access analysis
    capabilities, modes)                                        │
                                                                ▼
             SPIRV-Cross MSL 3.1 ─► PARALYNX container v2 (.prx): generated MSL,
                                     retained SPIR-V + SHA-256, toolchain identity,
                                     reflected descriptor
                                                                │
       C ABI / C++ / Python pr_module_load ─► verify ─► existing Metal engine
       (no SPIR-V libraries needed at runtime)          compile + reflection check
                                                        ─► physical Apple GPU
```

- `-DPARALYN_ENABLE_SPIRV=ON` (default **OFF**) builds the importer (`compiler/spirv/importer.cpp`, public header `include/paralyn/spirv.hpp`) and links it into the `paralyn` CLI. Default, compiler-disabled and runtime-only builds neither extract nor build nor link SPIRV-Tools/SPIRV-Cross.
- `paralyn compile foo.spvasm|foo.spv --output x.prx`, `paralyn check …` (imports, then compiles on Metal and validates reflection; no dispatch, no host code) and `paralyn inspect …` (import and reflection only). `.prx` files produced this way also work with `check`/`inspect`. `paralyn run` refuses kernel modules (`P-KERNEL-CASE-REQUIRED`), as for MSL modules.
- Modules load through the unchanged C ABI 1 (`pr_module_load[_file]`), the C++ wrappers and the Python binding, and execute on the same Metal engine, queue, retention, timing and evidence path as native IR and public MSL. A runtime-only build (no LLVM, no SPIR-V libraries) was observed to load and Metal-check a SPIR-V-derived `.prx`.
- `pr_device_capabilities_v1.artifact_formats` gains the additive bit `PR_ARTIFACT_SPIRV_MSL (1 << 2)`.

## Pinned toolchain

One coherent Khronos release set, **vulkan-sdk-1.4.363.0**, vendored as the exact upstream commit archives in `third_party/spirv/` (hashes in `third_party/SHA256SUMS.json`, details in `third_party/README.md`, licenses in `THIRD_PARTY_LICENSES.md`):

| Project | Commit | Archive SHA-256 |
|---|---|---|
| SPIRV-Headers | `496543121ce6419f23d6fa5d7194ba66c36212d2` | `a9bb9c48…ee9898` |
| SPIRV-Tools v2026.4 | `ef96ed763b43b59b33b31b362f09a02b729fa1c9` | `82c62146…3c0f00` |
| SPIRV-Cross | `f11ba9f0b21ba8fc15153d50a2a1ae31ab1cf8f7` | `92b04588…3abf3d03` |

`cmake/ParalynSpirv.cmake` verifies the archive hashes, extracts them into `build/_spirv/src`, and builds two isolated Release `ExternalProject`s (no network; Paralyn warning flags do not leak into them; sub-build parallelism `PARALYN_SPIRV_BUILD_JOBS`, default 4). SPIRV-Tools is built with `FORCED_BUILD_VERSION_DESCRIPTION` so `spirv-as --version` reports `SPIRV-Tools v2026.4 vulkan-sdk-1.4.363.0 ef96ed76…` rather than the enclosing Paralyn checkout's git revision (an observed contamination the first build had). `scripts/fetch_spirv_sources.py --download` re-verifies the vendored bytes and compares fresh downloads of each pinned commit; on 2026-09-28 all three matched.

```sh
cmake -S . -B build -G Ninja -DPARALYN_ENABLE_SPIRV=ON [-DPARALYN_SPIRV_BUILD_JOBS=3] …
cmake --build build -j 3
ctest --test-dir build -j1 -L spirv --output-on-failure
```

## Accepted profile

| Area | Accepted | Rejected (stable code) |
|---|---|---|
| Binary | little-endian SPIR-V 1.0–1.3, ≤ 4 MiB, schema 0 | big-endian, truncated, misaligned (`spirv.invalid-binary`); SPIR-V ≥ 1.4 (`spirv.version`) |
| Execution model | exactly one `GLCompute` entry point | **OpenCL `Kernel` model, `Kernel` capability, `OpenCL` memory model or `OpenCL.std` import → `spirv.kernel-model`, checked first and never reinterpreted**; other models (`spirv.execution-model`); zero or several entry points (`spirv.entry-point`) |
| Capabilities | `Shader` (and implicit `Matrix`) | everything else, e.g. `Float64`, `Int64`, `Float16`, `Int8/16`, subgroup, `VariablePointers`, `PhysicalStorageBufferAddresses`, `VulkanMemoryModel`, `Addresses` (`spirv.capability`) |
| Extensions / imports | `SPV_KHR_storage_buffer_storage_class`; `GLSL.std.450` | others (`spirv.extension`) |
| Addressing / memory model | `Logical` + `GLSL450` | physical addressing, forward pointers (`spirv.addressing`) |
| Workgroup | literal `LocalSize` (product ≤ 1024); `WorkgroupSize` builtin on a non-spec constant | missing `LocalSize`, `LocalSizeId`, `OpExecutionModeId` (`spirv.workgroup-size`); other modes (`spirv.execution-mode`); any `OpSpecConstant*` (`spirv.specialization`) |
| Storage buffers | set 0, binding 0–29, `StorageBuffer` class `Block` with exactly one member: a 1-D runtime or fixed array of 32-bit `f32`/`i32`/`u32`, offset 0, `ArrayStride 4`; `OpName` required | other sets, bindings ≥ 30, duplicates (`spirv.binding`); legacy `Uniform`+`BufferBlock`, descriptor arrays, structs/vectors, `Volatile`/`Coherent`, declared-but-unused, unnamed (`spirv.resource`) |
| Scalars | 32-bit scalar members of one `Uniform` `Block` per binding (set 0, binding 0–29) and of the one push-constant block; each member needs an `OpMemberName` | vectors/arrays/structs as members, non-4-aligned offsets, name collisions (`spirv.resource`) |
| Builtins | `GlobalInvocationId`, `LocalInvocationId`, `LocalInvocationIndex`, `WorkgroupId`, `NumWorkgroups`, `WorkgroupSize` | others, block-member builtins (`spirv.builtin`) |
| Instructions | whatever spirv-val accepts under the above, including structured loops/selection, `OpControlBarrier`, `OpMemoryBarrier`, Workgroup variables | `OpArrayLength` and any lowering that needs a Metal side-channel buffer, atomics (`spirv.instruction`); passing storage-buffer pointers to function calls (`spirv.resource`) |
| Validation | pinned `spirv-val`, `SPV_ENV_VULKAN_1_1`, default options | any validator error (`spirv.validation`, validator text included) |
| Access | storage-buffer access = static use (loads/stores through `OpAccessChain` roots) | `NonWritable` buffer that is stored to / `NonReadable` buffer that is loaded (`spirv.access`) — spirv-val 2026.4 was observed to accept such stores |

Other codes: `spirv.assembly` (assembler errors with line/column), `spirv.cross` (SPIRV-Cross exception), `spirv.numerics` (generated MSL containing `fast::`/`FP_CONTRACT`/`_Pragma`), `spirv.descriptor` (produced descriptor fails the container verifier). Every error's `what()` is `ParalynError: SPIR-V import [<code>]: <detail>`; the CLI maps `spirv.<name>` to `P-SPIRV-<NAME>` with stage `import` and writes no output file.

## Descriptor mapping (PARALYNX container version 2)

- Parameter order: storage buffers and Uniform blocks sorted by binding (each Uniform block expands to its members in member order), then push-constant members in member order. For the fixtures: `vector_add(a, b, c, n)`, `reduce_sum(input, partial, n, scale)`.
- Metal slots: set-0 binding *N* → `[[buffer(N)]]`; the push-constant block → `[[buffer(30)]]` (`paralyn::spirv_push_constant_slot`). Scalars of one block share a slot and carry `block_offset`; at launch the engine copies each typed argument to its offset in a zeroed block of the Metal-reflected size and binds it once with `setBytes`.
- Entry: MSL entry name (SPIRV-Cross cleanses reserved names, e.g. `main` → `main0`), `required_block` = LocalSize (launch block must match exactly), `builtins` bitmask of statically used builtins (plus `WorkgroupSize` when declared).
- Wire format: version-1 containers are byte-for-byte unchanged and now reject SPIR-V fields. Version 2 = version-1 layout with payload kind 2 (`SpirvMsl`), plus per entry `builtins`, per parameter `block_offset, origin, set, binding`, and trailing `toolchain`, `spirv_sha256`, `spirv` bytes. A v1 header with kind 2, or v2 with kind 1, is rejected.
- Runtime verification (no SPIR-V library): hashes of MSL and retained SPIR-V, slot/binding/origin consistency, and **re-derivation of storage-buffer access from the retained SPIR-V** (the descriptor may not understate it). Metal reflection then checks each active slot's block layout, element type, array stride, scalar member offsets/types, read-only scalar blocks, that every descriptor slot is active, workgroup memory and the fixed workgroup against pipeline limits. SPIRV-Cross declares storage buffers as non-`const device` (per its source comment on descriptor aliasing), so Metal's qualifier-based access is not used for them; the SPIR-V-derived access is.
- Aliasing follows the MSL profile: repeated allocations are allowed only when every binding involved is read-only.

## Numerics

Numerical policy 1 as for public MSL: Metal 3.1 safe math, precise functions, `#pragma STDC FP_CONTRACT OFF` prepended by the engine, and the importer rejects generated source that could override it. This is stricter than Vulkan's permission to contract; the tests use exactly representable inputs (sums of small integers < 2^24 and values with ≤ 2 fractional bits) so GPU results are compared **exactly**. Signed-integer overflow, transcendental GLSL.std.450 functions, division precision and denormal behavior are not qualified by these tests.

## Evidence (Apple M5, macOS 26, 2026-09-28, this branch)

- `spirv_importer` (CPU only): 119 checks, 38 stable rejections — reflected descriptors for both fixtures; build-time `spirv-as` words equal in-process assembly and `.spv`/`.spvasm` imports produce identical MSL/SPIR-V; 18 container rejections (exact v2 round trips, truncation, version/kind confusion, tampered access/hash/slot/workgroup/push-slot/format); the four named negative fixtures; 16 further boundaries (12 single mutations of the valid fixture, 4 binary-framing cases).
- `spirv_cli`: 4 fixture imports (`.spvasm` and build-assembled `.spv`, identical SPIR-V/MSL hashes); `check` on sources and `.prx` passes Metal compilation/reflection with `gpu_work_submitted=false`; 8 stable CLI diagnostics (`P-SPIRV-KERNEL-MODEL`, `-CAPABILITY`, `-VALIDATION`, `-BINDING` for compile and check) with no output written; `run` refused; `support` lists SPIR-V as partial.
- `spirv_metal` (physical GPU): C ABI driver **14 GPU events** (vector add n = 1, 63, 64, 65, 1000, 262147, 1000003 with canaries; read-only alias; reduction n/groups/scale = 1/1/1, 64/1/0.5, 4097/3/0.5, 100000/16/1, 1000003/128/0.5, 777/40/−2), 1,267,651 values compared exactly with independent CPU references, 12 rejection checks (writable alias, argument count/type/kind, workgroup mismatch, view access, tampered element type, uniform member offset and type caught by Metal reflection, omitted storage buffer, unknown container version); Python binding **6 GPU events**, 12,511 values compared, 2 rejections. All events have completed status and valid Metal system-clock timestamps; `execution.json` and the dispatched MSL are exported by the driver.
- Full `ctest -j1` in the SPIR-V-enabled build: 26/26 passed (23 pre-existing + 3 new), no skips. A separate runtime-only build (`-DPARALYN_BUILD_COMPILER=OFF`, SPIR-V off) passed its 9 tests, reported `P-SPIRV-UNAVAILABLE` for SPIR-V input and Metal-checked a SPIR-V-derived `.prx` without SPIR-V or LLVM libraries.

These are development-test captures in temporary/build directories, not archived clean-revision qualification evidence; no benchmark or timing qualification was run.

## Explicitly not provided

- **No shader-SPIR-V → CUDA/HIP lowering.** SPIRV-Cross targets MSL here; it supplies no NVIDIA/AMD backend. A CUDA/HIP bridge needs owned resource/address-space/builtin/control/memory lowering and real NVIDIA/AMD hardware; it remains a separate required track. SPIR-V modules execute on Metal only.
- **No OpenCL/Kernel SPIR-V.** Kernel-model modules are a distinct required profile (OpenCL C 1.2/Clspv route); this importer only rejects them.
- HLSL, GLSL and WGSL frontends are not part of this lane; nothing here compiles those languages (glslang/DXC/Naga are not vendored).
- Not supported: atomics, subgroup operations, images/samplers/texel buffers, 8/16/64-bit types, vectors or structs in buffers, specialization constants, multiple entry points, descriptor arrays, non-zero descriptor sets, `OpArrayLength`, variable/physical pointers, pointer arguments to functions, the Vulkan memory model, `LocalSizeId`.
- The runtime re-derives access from the retained SPIR-V, but cannot re-prove that the stored MSL was generated from that SPIR-V without SPIRV-Cross; like all `.prx` payloads this is a trusted-local interface, not a sandbox. Declared minimum sizes do not bound dynamic shader indexing; callers supply correctly sized views.
- Only the Apple M5 was exercised. No Windows/Linux, no other Apple GPU, no performance claim, no clean-revision archived capture, no packaging of the importer into the wheel or installers.
