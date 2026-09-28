# Adopted complete-platform implementation program

Adopted 2026-09-28 from the user's “Complete GPU Portability Platform and Terminal Experience” plan. This extends master mandate v1.1 and preserves every existing qualification requirement. The software remains 0.0.1. This is an obligation record, not implementation evidence.

> Bring your computation. Choose your GPU. Understand and verify the result.

The north star is a dependable GPU computing platform for existing supported GPU programs, accelerated Python/C++, and familiar clients, using physical Apple, NVIDIA and AMD GPUs through one understandable terminal experience. Compilation, resources, synchronization, placement, transformations, numerical policy, errors, verification and provenance must be inspectable. Completion is a substantial pinned compatibility matrix with working applications, installers, offline reproduction and hardware evidence—not arbitrary future language/driver coverage.

## Required inventory

All 17 input/interface families remain required: CUDA C++, native C/C++, native Python, HIP C++, Triton, OpenCL C, SYCL C++, OpenMP target (including a separate Flang/Fortran route), Slang, HLSL compute, GLSL compute, WGSL, SPIR-V, MLIR, native Metal source, PTX and SASS. Native MSL is explicitly Metal-specific. Every designated portable profile must pass through actual Metal, CUDA and HIP/ROCm backends on physical reference hardware.

Required clients are Numba CUDA, CuPy and CUDA Python, with both genuine NVIDIA interoperability and the selected non-NVIDIA compilation/API profiles. CUDA Array Interface, DLPack and external streams require real allocation/device/ownership/ordering contracts. Opaque handles are never vendor pointers.

New explicit framework requirements are PyTorch, JAX and ONNX. PyTorch custom operators, a device backend, compiled graphs, GPU inference, gradients and training are separate achievements. StableHLO import is not a JAX device; graph partitions must expose all placement. Required library-compatibility deliverables are cuBLAS, cuFFT, cuRAND, cuSPARSE and cuDNN; native mathematical operators do not fulfill those ABIs.

Native interfaces must progress from arrays/asarray/to_host/add/affine through shapes/views/strides, reductions, FP32 matmul, transpose, gather/scatter, scans, normalization, softmax, convolution, attention, FFT, RNG, sparse, backward/training and finally qualified collectives. Provider identity, accumulation/precision, workspace, layouts, aliases, ordering and numerical policy remain explicit.

Python and C++ compiled computational functions and automatic region offload are three separate deliverables from native array operations. Pin Numba typed compilation and Clang respectively; do not translate Python text into CUDA strings. Automatic offload requires effect/alias analysis, shape/type guards, supported control/memory semantics and visible placement. A failed guard selects another qualified GPU variant or fails explicitly. It does not silently execute the kernel on CPU.

Required application gates: FP32 two-layer MLP inference; transformer block; a gradient-checked training step; scientific Jacobi/stencil; multi-stage image processing; and a pinned external AI source application such as llm.c when prerequisites exist. Report unchanged source, configuration changes, adapters and unchanged binary compatibility separately.

## Shared engineering contracts

Preserve C ABI1 and existing .prk version1 readers. Add versioned descriptors and queries. Preserve owner/context validation, offsets, alignment, alias semantics, retained resources and errors. Introduce backend-qualified stable identities, capabilities, compiled executable objects, multiple queues, asynchronous transfers/dependencies, external resources and structured execution events. Timing includes validity and clock domains; duration-only CUDA/HIP events never acquire fabricated absolute timestamps.

Use multiple compiler representations rather than forcing every program through the original scalar IR. Maintain a versioned executable container with artifact/payload versions, producer/toolchain identity, hashes/dependencies, entrypoints/resources, target/capabilities, numerical policy, workgroup/shared-memory requirements and provenance. Run optional pinned compiler toolchains in isolated workers to avoid LLVM ABI collisions. Existing qualified frontend stays pinned to LLVM/Clang21.1.8 initially.

NVIDIA backend: CUDA Driver API, primary-context interoperability and calling-thread context restoration; verified IR to generated device source to NVRTC/PTX. AMD: HIP resource/module/stream/event APIs and HIPRTC; Linux and native Windows configurations qualify separately. Metal remains public Metal. Runtime-only installs do not need LLVM; Metal-only installs do not need other vendors' SDKs. Native Windows needs its own process, path, DLL, interruption and error implementation and hardware evidence; WSL is distinct.

Compiler routes: HIP/Clang; Triton pinned upstream vendor routes plus owned Apple lowering; OpenCL C1.2/Clspv Vulkan compute; SYCL/AdaptiveCpp with integrated USM ownership/events then accessors; OpenMP/Clang libomptarget plugin plus device compilation; Slang compilation/reflection; HLSL/DXC; GLSL/glslang; WGSL/Naga; validated SPIR-V1.3/Vulkan1.1 compute with logical addressing; named MLIR arith/scf/memref/gpu then tensor/linalg/StableHLO profiles. Shader SPIR-V to CUDA/HIP requires owned resource/address-space/builtin/control/memory lowering; SPIRV-Cross does not supply those backends. Kernel/OpenCL SPIR-V and Shader/Vulkan SPIR-V stay distinct.

PTX and SASS remain separate required binary programs. Before translation, establish permitted provenance/terms, exact artifact/ISA and parameter/resource ABI, tools and actual NVIDIA reference hardware. Unresolved permissions or unavailable reference devices keep the track blocked and required while independent source work proceeds. SASS generation selection follows real reference hardware. Unchanged host binaries additionally need host ABI, driver/runtime/library behavior and OS compatibility.

## Terminal and distribution obligations

Use C++/CLI11, typed commands, versioned JSON and declarative strict TOML projects. Optional FTXUI explorer must reuse CLI implementations and remain optional. All commands remain required: doctor, devices, support, init, check, inspect, compile, run, verify, bench, explain, report, cache, toolchain and explore. Existing spellings, unicuda and numeric selectors remain compatible; qualified selectors include metal:0, cuda:0 and rocm:0.

Distinguish whole programs, declared kernel cases and native apps. Static checking/compilation never runs main. Kernel-only execution requires typed inputs, resources and launch geometry; invent neither inputs nor references. Single-file programs need no manifest. Project paths are relative to the declarative manifest; reject unknown/ambiguous fields.

Keep application stdout/stderr separate, progress on stderr, runtime events separate, and JSON stdout valid. `run --json` preserves named application sidecars and emits one result; `--report-json` preserves live streams. Add NDJSON automation, stable diagnostics/remedies, non-TTY/width/color policy, shell completions and correct interruption/exit propagation. Never claim submitted GPU work was cancelled without a backend guarantee. `run` is execution; `verify` requires a declared independent comparison. Printed application “PASS” is not runtime proof.

Ship tested modular runtime/compiler/Python/toolchain/provider packages; macOS, native Windows and qualified Linux installation; offline docs/tutorials/inputs/references; runtime-only deployment; permitted offline dependencies/export bundles; checksums, dependency and license inventory. No hosted Paralyn service should be necessary for the documented release matrix.

## Milestones and resumable acceptance

- M0: reproduce baseline, hardware inventory, pin profiles, begin binary provenance prerequisites.
- M1: native FP32 arrays/operators, relocatable modules, packaging foundation, doctor/devices/support and clean output; independently verified installed examples outside checkout.
- M2: native Windows/platform work and physical Apple/NVIDIA/AMD execution with versioned timing/capabilities.
- M3: loops/functions/cooperative kernels; public MSL and shader/SPIR-V paths with HLSL/GLSL/OpenCL/WGSL useful workloads.
- M4: HIP programs, SYCL, OpenMP, Triton, MLIR and Slang with real semantics/dependencies.
- M5: providers, matmul/reductions/normalization/softmax, PyTorch, MLP and transformer inference.
- M6: compiled Python/C++, automatic offload and all required clients.
- M7: language/runtime/library expansion, training/scientific/image and pinned external applications.
- M8: permitted PTX/SASS non-NVIDIA translation and reference comparison; prerequisites run concurrently from M0.
- M9: full required profile matrix, terminal, installers/offline release, reproducibility/security/performance review; no required placeholder row.

Four coordinated lanes own runtime/backends, compilers/frontends, operators/integrations, and CLI/qualification. Merge shared contracts before dependent implementations. Serialize hardware qualification and isolate performance captures. High-risk proofs must be executable and yield minimized failing boundaries and exact tasks if they fail; they are not indefinite research licenses.

Each checkpoint records starting/ending commits, actual files/targets, tests/hardware/evidence, newly qualified subsets, all remaining required profiles, blockers and one exact ready task per lane. Existing four-size/10-warmup/100-measurement benchmark remains required; short probes are not its replacement. Every advertised cell needs real input/import/resources/physical execution/completion/independent comparison/reproducible evidence, plus useful second workload and negative tests. Original historical evidence is immutable.

Known hardware: Apple M5 available; Windows GPU models and access are unresolved. Existing authorized hardware is preferred. Acquisition, rental and enrollment require a concrete separate decision. Distributed execution remains required later through explicit independent jobs after local portability; it does not replace unfinished local work. No end-date forecast until hardware and highest-risk proofs are established.
