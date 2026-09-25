# Prior art and compatibility boundaries

Research date: **2026-09-25**. This is a primary-source desk review, not a reproduced benchmark or certification of another project. Links below were consulted on that date; moving documentation can change. Capability statements describe documented scope. No third-party implementation was executed as part of this review. UniCUDA's own results are recorded separately in `status.md`.

Every entry answers the same ten questions: **problem**, **hardware**, **model**, **unchanged CUDA source**, **source transformation**, **binary compatibility**, **heterogeneous hardware**, **distributed hardware**, **limitations**, and **UniCUDA difference**. “No” for CUDA binary compatibility means the layer does not itself promise NVIDIA CUDA application/driver ABI compatibility; it does not mean that it cannot load any binary. “Not intrinsic” means an enclosing system can add that capability. Proposed UniCUDA differences are design intentions, not implemented or novel capabilities.

## CUDA ecosystem and Apple execution

### CUDA programming/runtime model

Source: [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html). NVIDIA documentation/toolkit terms apply; no proprietary implementation is imported.

| Question | Finding |
|---|---|
| 1. Problem | Express and execute parallel work on NVIDIA GPUs. |
| 2. Hardware | Supported NVIDIA CUDA devices, with architecture-specific capabilities. |
| 3. Model | Host/device C++, kernels, grids, thread blocks, memory spaces, streams. |
| 4. Unchanged CUDA source | Yes, within the selected compiler/toolkit/device contract. |
| 5. Source transformation | Normal compilation; no user port required for native CUDA. |
| 6. Binary compatibility | Native CUDA ABI and documented version/device compatibility, not universal machine-code portability. |
| 7. Heterogeneous | CPU plus NVIDIA devices; not a universal vendor backend. |
| 8. Distributed | Multi-GPU facilities exist; multi-node composition requires additional communication/software. |
| 9. Limitations | Device features, numerical behavior, ABI, and memory rules are explicit constraints. |
| 10. UniCUDA difference | Independently implement a tested source subset on non-NVIDIA devices. |

### CUDA Runtime API

Source: [Runtime API reference](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html), especially synchronization and driver/runtime differences. Same NVIDIA terms; independent declarations only.

| Question | Finding |
|---|---|
| 1. Problem | Manage device work, memory, streams, events, and errors from host applications. |
| 2. Hardware | NVIDIA CUDA devices. |
| 3. Model | `cuda*` API with compiler-supported kernel launches and runtime-managed context behavior. |
| 4. Unchanged CUDA source | Yes in the native stack. |
| 5. Source transformation | Compiler emits host launch support; applications use the public API. |
| 6. Binary compatibility | CUDA runtime ABI subject to NVIDIA's version rules. |
| 7. Heterogeneous | Host and one or more NVIDIA GPUs. |
| 8. Distributed | Not a cluster scheduler or network data plane. |
| 9. Limitations | Synchronization and error behavior matter; “async” cannot be inferred solely from function names. |
| 10. UniCUDA difference | Small source-facing compatibility API with checked tokens and explicit missing features. |

### CUDA Driver API

Source: [Driver API reference](https://docs.nvidia.com/cuda/cuda-driver-api/index.html). NVIDIA terms; no driver shim in Gate A.

| Question | Finding |
|---|---|
| 1. Problem | Explicit contexts, modules, functions, device memory, and execution control. |
| 2. Hardware | NVIDIA CUDA devices. |
| 3. Model | Lower-level `cu*` host API with loaded device modules. |
| 4. Unchanged CUDA source | Device source is compiled separately; arbitrary Runtime API applications are not this interface. |
| 5. Source transformation | Compilation or generated modules required, not necessarily source rewriting. |
| 6. Binary compatibility | Native driver interface and supported module formats/version rules. |
| 7. Heterogeneous | Host/NVIDIA devices, not an independent multi-vendor abstraction. |
| 8. Distributed | Not intrinsic. |
| 9. Limitations | Exposes NVIDIA module, context, and capability semantics. |
| 10. UniCUDA difference | Gate A compiles source and links its own runtime; no drop-in driver ABI promise. |

### PTX

Source: [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html). Publicly documented NVIDIA virtual ISA; documentation access does not relicense NVIDIA implementations.

| Question | Finding |
|---|---|
| 1. Problem | Describe parallel device code for later target-machine translation. |
| 2. Hardware | Designed around NVIDIA GPU execution and capabilities. |
| 3. Model | Typed virtual registers, instructions, address spaces, special registers. |
| 4. Unchanged CUDA source | Not an input language; a CUDA compiler can produce PTX. |
| 5. Source transformation | CUDA-to-PTX compilation and PTX-to-machine lowering. |
| 6. Binary compatibility | PTX is a virtual-code distribution format, not arbitrary cubin/SASS compatibility. |
| 7. Heterogeneous | Other targets require translators with semantic coverage. |
| 8. Distributed | Not intrinsic. |
| 9. Limitations | NVIDIA-specific semantics and versioned instructions complicate cross-vendor lowering. |
| 10. UniCUDA difference | Start above PTX with source AST and a restricted portable IR. |

### Metal

Source: [Apple Metal documentation](https://developer.apple.com/documentation/metal). Public API, proprietary Apple frameworks/SDK terms; no framework binaries are bundled.

| Question | Finding |
|---|---|
| 1. Problem | Explicit GPU graphics and compute execution on Apple platforms. |
| 2. Hardware | Metal-capable devices supported by the OS; initial UniCUDA target is Apple Silicon. |
| 3. Model | Devices, resources, pipelines, command buffers, command encoders, queues. |
| 4. Unchanged CUDA source | No native CUDA frontend. |
| 5. Source transformation | CUDA needs a frontend/lowering layer producing supported shaders and host calls. |
| 6. Binary compatibility | Metal libraries, not CUDA application ABI. |
| 7. Heterogeneous | CPU/GPU and multiple Metal devices where present; not a cross-OS universal runtime. |
| 8. Distributed | Not intrinsic. |
| 9. Limitations | Feature sets and numerical/address-space rules differ from CUDA. |
| 10. UniCUDA difference | Supply the CUDA-facing frontend/runtime above public Metal. |

### Metal Shading Language (MSL)

Source: [MSL specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf). Apple specification terms; use documented behavior, not private AIR interfaces.

| Question | Finding |
|---|---|
| 1. Problem | Express shaders and compute kernels accepted by Metal. |
| 2. Hardware | Metal device families with version-dependent features. |
| 3. Model | C++-derived language with address spaces, entrypoint attributes, and builtins. |
| 4. Unchanged CUDA source | No. |
| 5. Source transformation | CUDA thread/index/memory constructs need semantic lowering. |
| 6. Binary compatibility | No CUDA ABI; compiled libraries are Metal artifacts. |
| 7. Heterogeneous | Portable within supported Metal targets and feature contracts. |
| 8. Distributed | Not intrinsic. |
| 9. Limitations | Entry-buffer aliasing, FP64 availability, SIMD and memory semantics need explicit treatment. |
| 10. UniCUDA difference | Verified IR emits MSL with one binding per unique allocation; no textual CUDA substitution. |

### Metal compute pipelines

Source: [Performing calculations on a GPU](https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu). Apple public runtime API.

| Question | Finding |
|---|---|
| 1. Problem | Compile, encode, submit, and synchronize GPU calculations. |
| 2. Hardware | Available Metal compute devices. |
| 3. Model | Library/function → compute pipeline → encoded dispatch → command completion. |
| 4. Unchanged CUDA source | No. |
| 5. Source transformation | Device code and host submission must be adapted. |
| 6. Binary compatibility | Metal pipeline/library contract only. |
| 7. Heterogeneous | Integrates CPU submission with GPU execution. |
| 8. Distributed | No cluster execution contract. |
| 9. Limitations | Dispatch geometry, pipeline limits, resource lifetimes, and asynchronous failures are application responsibilities. |
| 10. UniCUDA difference | Runtime validates CUDA geometry and records actual GPU completion evidence. |

### Apple unified memory

Source: [Choosing a resource storage mode for Apple GPUs](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus). Hardware/API property, not an independent open-source project.

| Question | Finding |
|---|---|
| 1. Problem | Let CPU and GPU access a common physical memory system with suitable resource modes. |
| 2. Hardware | Apple unified-memory GPUs; other Metal hardware has different modes. |
| 3. Model | Shared/private resource storage and explicit synchronization rules. |
| 4. Unchanged CUDA source | No CUDA allocation or unified-memory implementation by itself. |
| 5. Source transformation | Requires host runtime adaptation. |
| 6. Binary compatibility | None with CUDA solely from shared physical memory. |
| 7. Heterogeneous | CPU/GPU physical-memory sharing. |
| 8. Distributed | No; memory in separate machines remains separately placed. |
| 9. Limitations | Shared storage does not remove races, command ordering, resource limits, or ownership requirements. |
| 10. UniCUDA difference | Explicit copies and opaque tokens initially; do not advertise CUDA managed-memory semantics. |

## Portable APIs and compiler infrastructure

### ROCm

Sources: [ROCm programming guide](https://rocm-handbook.amd.com/), [AMD ROCm repository overview](https://github.com/ROCm/legacy-rocm-build). Component licenses vary; evaluate exact imported versions separately.

| Question | Finding |
|---|---|
| 1. Problem | Supply an AMD GPU compute software stack. |
| 2. Hardware | AMD GPUs in the selected ROCm release's OS/device support matrix. |
| 3. Model | HIP, compilers, runtimes, numerical/ML libraries, and tools. |
| 4. Unchanged CUDA source | Not generally; migration or an additional compatibility layer is needed. |
| 5. Source transformation | Commonly CUDA-to-HIP plus library replacements. |
| 6. Binary compatibility | No blanket NVIDIA CUDA binary contract. |
| 7. Heterogeneous | CPU/AMD GPU computing and component-specific portability. |
| 8. Distributed | Ecosystem communication/framework components provide multi-node use. |
| 9. Limitations | Hardware/OS support and library/API coverage are version-specific. |
| 10. UniCUDA difference | Later ROCm backend beneath the same tested CUDA-source subset. |

### HIP

Sources: [HIP project](https://github.com/ROCm/HIP), [Clang HIP support](https://clang.llvm.org/docs/HIPSupport.html), [AMD portability example](https://rocm.blogs.amd.com/software-tools-optimization/hipify/README.html). Consult HIP component license at the chosen revision; not imported here.

| Question | Finding |
|---|---|
| 1. Problem | Port GPU C++ across supported implementation backends. |
| 2. Hardware | AMD is current HIP project focus; NVIDIA paths are documented in Clang/AMD portability material and must be version-verified. |
| 3. Model | CUDA-like C++ kernels and HIP runtime APIs. |
| 4. Unchanged CUDA source | Generally requires API/include migration. |
| 5. Source transformation | HIPIFY and manual changes for missing features/libraries. |
| 6. Binary compatibility | Source portability, not a universal CUDA binary replacement. |
| 7. Heterogeneous | CPU/GPU; backend coverage depends on implementation. |
| 8. Distributed | Through libraries and enclosing runtimes. |
| 9. Limitations | API/architecture differences and vendor-specific tuning remain. |
| 10. UniCUDA difference | Preserve a declared CUDA source interface and begin on Metal. |

### HIPIFY

Source: [HIPIFY documentation](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/). Open-source migration tools; verify tool license/version before importing code.

| Question | Finding |
|---|---|
| 1. Problem | Automate CUDA-to-HIP migration. |
| 2. Hardware | Translation tool runs on the host; generated HIP targets supported GPUs. |
| 3. Model | Clang-based semantic conversion or simpler Perl replacement. |
| 4. Unchanged CUDA source | Input may be ordinary CUDA, but output is HIP. |
| 5. Source transformation | Yes, its primary function. |
| 6. Binary compatibility | No. |
| 7. Heterogeneous | Inherited from HIP, not executed by HIPIFY itself. |
| 8. Distributed | Not intrinsic. |
| 9. Limitations | Unsupported CUDA libraries/features need manual work and tests. |
| 10. UniCUDA difference | Internal lowering/runtime compatibility rather than requiring a maintained HIP port. |

### OpenCL

Source: [Khronos OpenCL specification](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html). Specification and implementation licensing differ; no implementation dependency in v0.

| Question | Finding |
|---|---|
| 1. Problem | Standardize compute across heterogeneous devices. |
| 2. Hardware | CPU, GPU, and accelerators supported by installed implementations. |
| 3. Model | Platforms/devices, contexts, queues, memory objects, ND-range kernels. |
| 4. Unchanged CUDA source | No. |
| 5. Source transformation | Host API and device-language adaptation needed. |
| 6. Binary compatibility | OpenCL program binaries/IL, not CUDA ABI. |
| 7. Heterogeneous | Yes, a central design goal. |
| 8. Distributed | Not guaranteed by the base standard. |
| 9. Limitations | Optional capabilities and implementation availability affect portability. |
| 10. UniCUDA difference | CUDA-source compatibility layered over backends rather than a new application language/API. |

### SYCL

Source: [Khronos SYCL](https://www.khronos.org/sycl/). Standard specification terms; implementation licenses vary.

| Question | Finding |
|---|---|
| 1. Problem | Express heterogeneous compute in single-source modern C++. |
| 2. Hardware | Implementation-dependent CPUs, GPUs, FPGAs, and accelerators. |
| 3. Model | C++ queues, kernels, memory abstractions, templates, and lambdas. |
| 4. Unchanged CUDA source | No general guarantee. |
| 5. Source transformation | CUDA migration usually changes kernels and host interfaces. |
| 6. Binary compatibility | No CUDA ABI contract. |
| 7. Heterogeneous | Yes, including multiple device kinds in an application. |
| 8. Distributed | Requires additional libraries/runtime integration. |
| 9. Limitations | Feature and performance portability are different; target tuning remains necessary. |
| 10. UniCUDA difference | Retain supported CUDA syntax instead of asking users to adopt SYCL. |

### oneAPI

Source: [oneAPI specifications](https://oneapi.io/spec/). Umbrella of specifications and implementations; check each component's license.

| Question | Finding |
|---|---|
| 1. Problem | Provide cross-architecture programming interfaces and libraries. |
| 2. Hardware | CPUs, GPUs, and accelerators supported by implementations. |
| 3. Model | SYCL-based programming plus standardized domain libraries. |
| 4. Unchanged CUDA source | Not a general promise. |
| 5. Source transformation | CUDA-to-SYCL migration and library adaptation commonly required. |
| 6. Binary compatibility | Not a CUDA application ABI replacement. |
| 7. Heterogeneous | Yes, at the programming/library level. |
| 8. Distributed | Communication components may support it; not automatic memory unification. |
| 9. Limitations | Specification presence does not guarantee a backend or complete feature implementation. |
| 10. UniCUDA difference | Narrow CUDA compatibility contract with explicit execution evidence. |

### Vulkan Compute

Source: [Khronos Vulkan Guide](https://docs.vulkan.org/guide/latest/). Khronos specification terms; loaders, tools, and drivers have separate licenses.

| Question | Finding |
|---|---|
| 1. Problem | Explicit cross-platform GPU graphics and compute submission. |
| 2. Hardware | Conformant Vulkan devices and supported extensions. |
| 3. Model | SPIR-V shaders, pipelines, descriptor/resources, command buffers, synchronization. |
| 4. Unchanged CUDA source | No. |
| 5. Source transformation | Device lowering and host resource/submission adaptation. |
| 6. Binary compatibility | SPIR-V/pipeline formats, not CUDA ABI. |
| 7. Heterogeneous | Multiple implementations/device kinds, with explicit selection. |
| 8. Distributed | Not intrinsic. |
| 9. Limitations | Addressing, subgroup, synchronization, and feature limits need explicit mapping. |
| 10. UniCUDA difference | Potential later backend; CUDA host semantics remain a separate responsibility. |

### LLVM

Sources: [LLVM documentation](https://llvm.org/docs/), [LLVM license](https://github.com/llvm/llvm-project/blob/main/LICENSE.TXT). Apache-2.0 with LLVM exceptions, plus component notices.

| Question | Finding |
|---|---|
| 1. Problem | Reusable compilation, optimization, and target code generation. |
| 2. Hardware | Many CPU/GPU targets through target-specific backends. |
| 3. Model | Typed SSA IR, passes, target machines, and supporting libraries. |
| 4. Unchanged CUDA source | Requires a frontend such as Clang. |
| 5. Source transformation | Usually AST-to-IR lowering, not source rewriting alone. |
| 6. Binary compatibility | Object/code generation; no CUDA runtime emulation by itself. |
| 7. Heterogeneous | Supports component toolchains, not a universal scheduling runtime. |
| 8. Distributed | Not intrinsic. |
| 9. Limitations | Data layout, address spaces, target intrinsics, and runtime semantics remain target-specific. |
| 10. UniCUDA difference | Reuse Clang parsing; preserve a small higher-level IR for initial MSL output. |

### Clang CUDA support and LibTooling

Sources: [Compiling CUDA with Clang](https://llvm.org/docs/CompileCudaWithLLVM.html), [LibTooling](https://clang.llvm.org/docs/LibTooling.html), [offloading design](https://clang.llvm.org/docs/OffloadingDesign.html). LLVM license above.

| Question | Finding |
|---|---|
| 1. Problem | Parse/type-check CUDA and integrate host/device compilation; expose AST tools. |
| 2. Hardware | Host targets plus supported GPU offload toolchains. |
| 3. Model | Full C++/CUDA AST, compiler diagnostics, host/device compilation passes. |
| 4. Unchanged CUDA source | Many programs, subject to documented NVCC dialect/toolkit differences. |
| 5. Source transformation | Not inherently required for native compilation; tools can rewrite AST-owned ranges. |
| 6. Binary compatibility | Produces supported target code, not a CUDA-to-Metal runtime. |
| 7. Heterogeneous | Compiler infrastructure supports offloading models. |
| 8. Distributed | Not intrinsic. |
| 9. Limitations | Standard native CUDA flow expects CUDA SDK support; LibTooling API/version coupling matters. |
| 10. UniCUDA difference | Independent declarations plus restricted AST extraction and host launch rewrites. |

### MLIR

Sources: [MLIR](https://mlir.llvm.org/), [design rationale](https://mlir.llvm.org/docs/Rationale/Rationale/). LLVM license.

| Question | Finding |
|---|---|
| 1. Problem | Preserve and progressively lower domain-specific compilation abstractions. |
| 2. Hardware | Targets supplied by dialects and downstream lowering stacks. |
| 3. Model | Extensible typed operations, regions, dialects, verification, conversions. |
| 4. Unchanged CUDA source | Needs a CUDA frontend and runtime. |
| 5. Source transformation | Usually multi-level IR transformation. |
| 6. Binary compatibility | Not a CUDA binary loader. |
| 7. Heterogeneous | Enables heterogeneous compiler stacks. |
| 8. Distributed | Can model such programs; no cluster runtime by itself. |
| 9. Limitations | Dialect integration and semantic lowering remain engineering work. |
| 10. UniCUDA difference | Defer integration cost for Gate A; reconsider as operations and backends grow. |

### SPIR-V

Source: [Khronos SPIR-V](https://www.khronos.org/spirv/). Specification and tooling licenses are distinct.

| Question | Finding |
|---|---|
| 1. Problem | Standard intermediate representation for graphics/compute environments. |
| 2. Hardware | Targets exposed by compatible Vulkan/OpenCL and other consumers. |
| 3. Model | Binary typed instructions constrained by execution environment/capabilities. |
| 4. Unchanged CUDA source | Needs a frontend and semantic translator. |
| 5. Source transformation | Compilation to SPIR-V, with further lowering by a consumer. |
| 6. Binary compatibility | SPIR-V modules, not CUDA application/driver ABI. |
| 7. Heterogeneous | Portable across compatible implementations, not all capability sets. |
| 8. Distributed | Not intrinsic. |
| 9. Limitations | Environment rules differ; host API and CUDA semantics remain unimplemented. |
| 10. UniCUDA difference | Keep kernel IR above backend formats; potential later Vulkan lowering. |

## Direct compatibility projects

### ZLUDA

Sources: [upstream repository](https://github.com/vosen/ZLUDA), [project updates](https://vosen.github.io/ZLUDA/). Repository is dual Apache-2.0/MIT; not imported.

| Question | Finding |
|---|---|
| 1. Problem | Execute existing CUDA applications on non-NVIDIA GPUs. |
| 2. Hardware | AMD/ROCm paths are documented; consult current release notes for exact device/OS support. |
| 3. Model | CUDA compatibility interception plus device-code translation. |
| 4. Unchanged CUDA source | Existing compiled applications are the intended input, subject to coverage. |
| 5. Source transformation | User source rewriting is not the main model. |
| 6. Binary compatibility | Yes, an explicit project objective; coverage is not universal. |
| 7. Heterogeneous | Non-NVIDIA execution is the goal, not proof of every vendor combination. |
| 8. Distributed | No cluster scheduling guarantee established by this review. |
| 9. Limitations | Driver/library API coverage, PTX semantics, application behavior, and backend support. |
| 10. UniCUDA difference | Source-first restricted frontend on Metal rather than beginning with binary interception. |

### CuMetal / cuda-metal

Sources: [upstream project](https://github.com/Lulzx/cuda-metal), [license](https://github.com/Lulzx/cuda-metal/blob/main/LICENSE). Apache-2.0. No code is imported.

| Question | Finding |
|---|---|
| 1. Problem | Run a CUDA subset on Apple Silicon. |
| 2. Hardware | Apple M-series GPUs with supported macOS/toolchain. |
| 3. Model | Compiler, CUDA-facing runtime, compatibility libraries, typed IR and Metal lowering. |
| 4. Unchanged CUDA source | Upstream documents ordinary `.cu` compilation; coverage is explicitly incomplete. |
| 5. Source transformation | Internal compilation/lowering; emitted stages are inspectable. |
| 6. Binary compatibility | Optional `libcuda` shim exists; default source-first distribution excludes it. |
| 7. Heterogeneous | Apple Metal target; not demonstrated cross-vendor coverage here. |
| 8. Distributed | Not established in reviewed documentation. |
| 9. Limitations | Experimental; documented compiler gaps and differing source/PTX path coverage. |
| 10. UniCUDA difference | No unique Gate A feature established. A future common multi-vendor conformance contract is only a potential contribution. |

This is direct prior art, including upstream GPU-provenance requirements. Do not describe source-to-Metal execution, inspectable IR, or no-fallback validation as inventions of UniCUDA. Upstream reported results have not been independently reproduced here.

### MetaXuda

Sources: [README](https://github.com/Perinban/MetaXuda), [license](https://github.com/Perinban/MetaXuda/blob/main/LICENSE). Custom terms restrict commercial use and modified redistribution; **not an Apache-compatible dependency adopted by this project**. No code/binaries are imported.

| Question | Finding |
|---|---|
| 1. Problem | Run selected Numba CUDA workflows through Apple Metal. |
| 2. Hardware | Apple Silicon with documented Metal/macOS requirements. |
| 3. Model | Python initialization and native CUDA-library shims. |
| 4. Unchanged CUDA source | Claims unchanged Numba kernel bodies, not general ordinary CUDA C++. |
| 5. Source transformation | User adds initialization; runtime supplies translation/dispatch behavior. |
| 6. Binary compatibility | Claims core CUDA-library replacement; this review does not validate breadth. |
| 7. Heterogeneous | Single Apple GPU is the documented initial target. |
| 8. Distributed | Not established. |
| 9. Limitations | Upstream explicitly documents ignored scalar arithmetic in kernel arguments and incomplete APIs. |
| 10. UniCUDA difference | Independent C++ AST/IR lowering with explicit rejection rather than silently discarded operations. |

The documented scalar issue is a concrete reason to vary semantics and inputs during qualification. It is an upstream limitation statement, not a result reproduced by UniCUDA.

## Scheduling, frameworks, and compiler research

### Distributed GPU runtimes: Ray

Sources: [Ray accelerator scheduling](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html), [license](https://github.com/ray-project/ray/blob/master/LICENSE). Apache-2.0 with additional component notices.

| Question | Finding |
|---|---|
| 1. Problem | Schedule distributed tasks/actors with accelerator resource requirements. |
| 2. Hardware | Supported accelerator types advertised by configured nodes. |
| 3. Model | Distributed tasks/actors and logical resources; tasks use their compute framework. |
| 4. Unchanged CUDA source | Does not compile CUDA; tasks may invoke existing CUDA software. |
| 5. Source transformation | Application scheduling integration, not kernel translation. |
| 6. Binary compatibility | No CUDA ABI compatibility layer. |
| 7. Heterogeneous | Yes at resource scheduling level. |
| 8. Distributed | Yes, a central function. |
| 9. Limitations | Resource assignment does not make a kernel compatible or guarantee memory isolation/capacity. |
| 10. UniCUDA difference | Establish device execution compatibility first; borrow explicit placement lessons later. |

### Heterogeneous compute schedulers: StarPU

Sources: [StarPU](https://starpu.gitlabpages.inria.fr/), [features](https://starpu.gitlabpages.inria.fr/features.html), [research report](https://inria.hal.science/inria-00523937v1). Consult the chosen StarPU distribution's license before reuse; no dependency is adopted.

| Question | Finding |
|---|---|
| 1. Problem | Schedule dependent tasks and data transfers on heterogeneous machines. |
| 2. Hardware | CPUs and supported CUDA/HIP/OpenCL accelerators. |
| 3. Model | Codelets with per-architecture implementations, data handles, dependency graphs. |
| 4. Unchanged CUDA source | Existing kernels can be wrapped; whole applications require task integration. |
| 5. Source transformation | Explicit application/codelet adaptation rather than universal translation. |
| 6. Binary compatibility | No CUDA ABI emulation. |
| 7. Heterogeneous | Yes, a defining capability. |
| 8. Distributed | Cluster communication and scheduling are documented. |
| 9. Limitations | Requires suitable implementations; task granularity and transfers affect efficiency. |
| 10. UniCUDA difference | Compile a CUDA subset to device variants; future scheduling must acknowledge this prior art. |

### PyTorch device/backend architecture

Sources: [device abstractions](https://docs.pytorch.org/cppdocs/api/c10/device.html), [PrivateUse1 integration](https://docs.pytorch.org/tutorials/advanced/privateuseone.html), [license](https://github.com/pytorch/pytorch/blob/main/LICENSE). BSD-style project license and component notices; not linked in v0.

| Question | Finding |
|---|---|
| 1. Problem | Route tensor operations, storage, autograd, and device state to backends. |
| 2. Hardware | CPU/CUDA/MPS/XPU and other built-in or external device integrations. |
| 3. Model | Tensor operators and dispatcher keys, device guards, registered implementations. |
| 4. Unchanged CUDA source | No general translation of arbitrary CUDA extensions to another backend. |
| 5. Source transformation | Operators need implementations/integration for the target. |
| 6. Binary compatibility | PyTorch extension ABI constraints, not CUDA ABI emulation. |
| 7. Heterogeneous | Multiple device types; coverage depends on operator/backend. |
| 8. Distributed | Framework facilities exist; backend communication support is separate work. |
| 9. Limitations | Registering a device does not implement its operators or libraries. |
| 10. UniCUDA difference | Native CUDA-source runtime first; no transparent PyTorch claim. |

### JAX

Sources: [JAX documentation](https://docs.jax.dev/en/latest/), [license](https://github.com/jax-ml/jax/blob/main/LICENSE). Apache-2.0; dependency notices separate.

| Question | Finding |
|---|---|
| 1. Problem | Transform and compile array computations with differentiation and parallelism. |
| 2. Hardware | CPU/GPU/TPU and supported backend plugins. |
| 3. Model | Python arrays, tracing, transformations, staged compilation, sharding. |
| 4. Unchanged CUDA source | No, though custom calls/FFI can integrate code. |
| 5. Source transformation | Python computation staging; CUDA kernels need separate integration. |
| 6. Binary compatibility | No general CUDA application ABI replacement. |
| 7. Heterogeneous | Multiple backend ecosystems; mixing arbitrary devices is not automatic. |
| 8. Distributed | Explicit multi-host, distributed arrays, and sharding facilities. |
| 9. Limitations | Staging, backend availability, shapes, and numerical semantics constrain programs. |
| 10. UniCUDA difference | CUDA C++ source is the initial interface; interoperability is deferred. |

### XLA / OpenXLA

Sources: [XLA overview](https://openxla.org/xla), [license](https://github.com/openxla/xla/blob/main/LICENSE). Apache-2.0 plus dependencies.

| Question | Finding |
|---|---|
| 1. Problem | Compile high-level ML computations efficiently for varied hardware. |
| 2. Hardware | Supported CPUs, GPUs, and ML accelerators. |
| 3. Model | HLO/StableHLO-related graphs, optimization, target lowering, PJRT execution. |
| 4. Unchanged CUDA source | No general CUDA C++ frontend/host preservation. |
| 5. Source transformation | Framework graphs lower into compiler representations. |
| 6. Binary compatibility | No CUDA application/driver ABI promise. |
| 7. Heterogeneous | Multiple targets and extensible backend infrastructure. |
| 8. Distributed | Partitioning and parallel computation are supported in the stack. |
| 9. Limitations | Graph semantics and backend coverage differ from arbitrary imperative CUDA. |
| 10. UniCUDA difference | Preserve ordinary host C++ around a source-level CUDA subset. |

### Triton

Sources: [Triton documentation](https://triton-lang.org/main/index.html), [upstream repository](https://github.com/triton-lang/triton), [license](https://github.com/triton-lang/triton/blob/main/LICENSE). MIT project license; LLVM and other dependencies have separate terms.

| Question | Finding |
|---|---|
| 1. Problem | Productively express optimized GPU compute kernels. |
| 2. Hardware | Supported GPU backends; exact architecture coverage is release-dependent. |
| 3. Model | Python-based kernel language with block/tile operations and compiler lowering. |
| 4. Unchanged CUDA source | No. |
| 5. Source transformation | Rewrite kernels to Triton's model or generate them. |
| 6. Binary compatibility | Not CUDA ABI emulation. |
| 7. Heterogeneous | Multiple GPU backends, not arbitrary backend equivalence. |
| 8. Distributed | Integrates with larger distributed systems; not a general cluster scheduler. |
| 9. Limitations | DSL semantics, supported operations, layouts, and target performance constraints. |
| 10. UniCUDA difference | CUDA C++ entrypoint and compatibility, rather than a replacement kernel language. |

### IREE

Sources: [deployment configurations](https://iree.dev/guides/deployment-configurations/), [project](https://github.com/iree-org/iree), [license](https://github.com/iree-org/iree/blob/main/LICENSE). Apache-2.0 with LLVM exceptions; dependency notices separate.

| Question | Finding |
|---|---|
| 1. Problem | Compile and deploy ML programs through portable runtime/hardware abstractions. |
| 2. Hardware | Documented CPU, Vulkan, ROCm, CUDA, and Metal configurations. |
| 3. Model | MLIR-based compilation, runtime modules, HAL devices and drivers. |
| 4. Unchanged CUDA source | Not the advertised frontend contract. |
| 5. Source transformation | Framework/model lowering, then backend code generation. |
| 6. Binary compatibility | IREE modules/artifacts, not CUDA application ABI. |
| 7. Heterogeneous | Yes; explicit compiler target/runtime driver separation. |
| 8. Distributed | Do not infer cluster CUDA compatibility from multi-backend support. |
| 9. Limitations | Frontend/model and backend feature coverage determine supported programs. |
| 10. UniCUDA difference | CUDA source/host semantics; multi-backend IR/runtime architecture already exists. |

### Compiler research: MLIR paper (2021)

Source: [Lattner et al., “MLIR: Scaling Compiler Infrastructure for Domain Specific Computation”](https://research.google/pubs/mlir-scaling-compiler-infrastructure-for-domain-specific-computation/). Publication is research evidence, not a vendored code dependency; implementation licensing follows LLVM.

| Question | Finding |
|---|---|
| 1. Problem | Avoid losing domain information while sharing compiler infrastructure. |
| 2. Hardware | Extensible compiler design rather than a fixed hardware matrix. |
| 3. Model | Multiple abstraction levels and extensible IR dialects. |
| 4. Unchanged CUDA source | Not claimed by the paper. |
| 5. Source transformation | Progressive IR lowering is central. |
| 6. Binary compatibility | Not its objective. |
| 7. Heterogeneous | Supports designing compilers for varied targets. |
| 8. Distributed | Not a distributed runtime deliverable. |
| 9. Limitations | Infrastructure does not supply every frontend or semantics-preserving lowering. |
| 10. UniCUDA difference | Use a tiny initial IR while retaining an explicit future migration criterion. |

### Compiler research: Triton paper (2019)

Source: [Tillet, Kung, and Cox, “Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations”](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf). Research publication; no paper text/code imported.

| Question | Finding |
|---|---|
| 1. Problem | Generate efficient GPU kernels from tile-oriented computation. |
| 2. Hardware | GPU experiments in the paper; not a universal current hardware claim. |
| 3. Model | Tile programs and specialized compiler transformations. |
| 4. Unchanged CUDA source | No. |
| 5. Source transformation | Programs are expressed in the proposed intermediate language. |
| 6. Binary compatibility | Not a CUDA ABI project. |
| 7. Heterogeneous | Compiler retargetability differs from demonstrated universal portability. |
| 8. Distributed | Outside the paper's primary objective. |
| 9. Limitations | Domain-specific model and evaluated workloads bound conclusions. |
| 10. UniCUDA difference | Source compatibility first; postpone performance transformations until correctness. |

### Compiler research: TinyIREE (2022)

Source: [Liu et al., “TinyIREE: An ML Execution Environment for Embedded Systems from Compilation to Deployment”](https://arxiv.org/abs/2205.14479). Research publication; IREE's implementation license is separate.

| Question | Finding |
|---|---|
| 1. Problem | Scale a unified compiler/runtime down to embedded and bare-metal deployments. |
| 2. Hardware | Embedded targets and ISA/ABI choices evaluated by the work. |
| 3. Model | ML models lowered through MLIR/LLVM and deployment-specific runtime choices. |
| 4. Unchanged CUDA source | No general CUDA-source contract. |
| 5. Source transformation | Compiler lowering from ML program representations. |
| 6. Binary compatibility | Target artifacts, not CUDA application ABI. |
| 7. Heterogeneous | Designed for heterogeneous accelerators and deployment targets. |
| 8. Distributed | Scaling deployment targets is not evidence of transparent cluster execution. |
| 9. Limitations | Embedded deployment conclusions do not establish CUDA compatibility. |
| 10. UniCUDA difference | Study clean compiler/runtime boundaries while targeting a different source interface. |

## Consequences for UniCUDA

CUDA-source translation, typed portable IR, cross-vendor APIs, backend registries, device placement, and heterogeneous scheduling all have substantial prior art. CuMetal directly overlaps the initial Apple milestone. Gate A establishes that this independent implementation works; it does not establish novelty.

The defensible experiment is a narrow, inspectable CUDA compatibility contract with demonstrated execution and error behavior. Future cross-vendor claims require the same source and conformance inputs on each physical backend. Future scheduling must distinguish compatibility from resource availability and aggregate memory from a single allocation space.

There are deliberate gaps in this desk review: no rival performance reproduction, no exhaustive ABI audit, and no legal clearance of unadopted dependencies. Architecture choices should use these boundaries rather than claim the gaps are solved.
