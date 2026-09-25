# Paralyn - Master Engineering Mandate v1.1

## All-frontends scope amendment

This edition makes every named frontend/input family in Section 4 a required implementation, delivered in phases. It supersedes the earlier instruction to choose only selected frontends. It does not claim that those implementations exist, change the software release version, or waive hardware, correctness, licensing, and safety gates. Sections 3, 15, and 16 are aligned with this scope.

## 0. This is an upgrade, not a restart

You are continuing the existing UniCUDA/Paralyn repository as its systems engineer, compiler engineer, and developer-experience designer.

This is version 1.1 of the engineering mandate, NOT software release 1.1. Do not change the software's version or claim new capabilities because you received this document.

Extend the project's mission from CUDA-source compatibility to an open, general-purpose accelerated-computing platform. Preserve the working implementation, current compatibility contract, tests, architectural decisions, and unfinished acceptance gates.

Read AGENTS.md, the existing master mandate, revised v0.0.1 plan, README, architecture, compatibility documentation, roadmap, tests, and current Git state. Record what actually exists and reproduce the current baseline. Do not assume that Gate A passed, that no implementation exists, or that earlier example output represents a real execution.

Work in the existing repository. Do not create a replacement project or discard user changes. Keep UniCUDA command/import compatibility during any deliberate naming transition. Treat Paralyn as the working name, not as a legally cleared brand. Separate renaming commits from functional changes.

Where this mandate expands future scope, it supersedes the old long-term roadmap. Where the revised v0.0.1 plan specifies current correctness and qualification requirements, preserve those requirements. Resolve genuine conflicts explicitly in an architecture decision record rather than silently weakening guarantees.

## 1. Mission and the first useful product

Build an open-source platform that makes supported accelerated computations portable across hardware vendors through an understandable programming interface and an inspectable execution system.

CUDA compatibility is the first entry point, not the definition of the platform.

The long-term aspiration is:

> One computation. Multiple hardware choices. No hidden compromises.

The immediate product remains:

> Run a documented subset of complete CUDA C++ programs on Apple GPUs through Metal, with verified results and no CPU kernel fallback.

Three eventual developer journeys should share the same foundation: bring existing GPU source, write new applications with the native Paralyn API, or connect an existing framework through a tested adapter.

Do not claim arbitrary programs run everywhere. Compatibility, numerical behavior, performance, and distributed execution are separate properties that require separate evidence.

Do not claim the broad idea is unprecedented. Research IREE, SYCL, HIP/HIPIFY, OpenCL, Vulkan compute, LLVM/MLIR, Triton, Slang, OpenXLA, StableHLO, TVM, PyTorch, JAX, ONNX Runtime, MLX, wgpu, Kokkos, RAJA, Alpaka, ZLUDA, and CUDA-to-Metal projects. Extend existing prior-art documents rather than starting an endless survey. Verify current sources and distinguish upstream claims from reproduced results.

For each major new subsystem, explain what Paralyn adds, what it reuses, and why contributing to or integrating an existing project is insufficient on its own. Independent architecture does not require independently reinventing every compiler and runtime.

## 2. Preserve the current CUDA foundation

Retain the revised plan's complete-program path:

```text
ordinary vector_add.cu
    |-- host AST --> transformed native C++ --> runtime
    `-- kernel AST --> verified typed IR --> MSL --> Apple GPU
                                                        |
                                           independent verification
```

Preserve ordinary host C++ outside explicitly supported CUDA transformations. Retain AST-directed rewriting, source-located diagnostics, exactly-once expression evaluation, launch-configuration ordering, parameter conversions, unsigned index semantics, and program-exit propagation.

Retain the documented subset of dim3, cudaMalloc, cudaFree, cudaMemcpy, cudaDeviceSynchronize, cudaGetLastError, and cudaGetErrorString. Do not claim full CUDA runtime compatibility.

Preserve the current allocation-token contract, including unsupported host pointer arithmetic and interior pointers. Native Paralyn buffers must not inherit these CUDA-specific restrictions unnecessarily, but extending either interface requires its own design and tests. Do not manufacture unsafe integer-encoded pointers or bypass the live-allocation registry.

Preserve the alias-group design: supported same-allocation, same-pointee-type parameters share a binding without assuming independence. Keep mixed-pointee aliasing unsupported until proven. Retain alias-layout-aware in-process pipeline caching and negative alias tests.

Retain verified i32/u32/f32 kernel IR, runtime MSL compilation, checked copies, logical-grid preservation, ordered default-queue behavior, asynchronous resource retention, and command-failure propagation. Unsupported features must fail explicitly.

Preserve the working compiler while adding the required frontend portfolio. Refactor only when an evidenced integration requirement justifies the change; avoid a speculative rewrite.

## 3. Architecture: multiple representations, one execution contract

Do not force every workload through CUDA source or the original scalar kernel IR.

Use this conceptual architecture:

```text
SOURCE / PROGRAMMING MODELS       FRAMEWORKS / GRAPHS       NATIVE API
CUDA, HIP, Triton, etc.           PyTorch, JAX, ONNX        C/C++, Python
          |                              |                       |
    frontend adapters              framework adapters           |
          |                              |                       |
          |                    tensor/operator representation <--+
          |                              |
          |                  planning, fusion, provider selection
          |                         /                \
          v                        v                  v
    kernel representations / lowering        library/graph providers
          |                                           |
          +-------------- executable plans -----------+
                                 |
              shared buffers, queues, events, capabilities,
                   execution policy and diagnostics
                                 |
                Metal | CUDA | HIP/ROCm | future backends
```

A frontend adapter imports a language or representation. A framework adapter implements a framework integration boundary. An operator provider supplies specified math operations. An execution backend manages device resources and executable submission. These are different extension points.

The current kernel IR remains appropriate for its tested subset. Later tensor/operator representations should preserve operations such as matmul, reductions, convolution, and FFT until provider selection or lowering requires expansion. Preserve shapes, layouts, element and accumulation types, effects, aliasing, and numerical policy.

Investigate existing MLIR dialects, StableHLO, and compiler bridges before inventing a new graph format. Existing MLIR and StableHLO designs are relevant precedents, not evidence that importing arbitrary dialects will work. [R3, R4]

Allow a mature compiler bridge to produce validated backend artifacts without round-tripping through Paralyn's own kernel IR. Such artifacts must still obey the runtime's memory, argument, capability, synchronization, and diagnostic contracts.

Section 4 defines the required frontend portfolio and portable-subset backend targets. Implementation obligations are not current support claims: every advertised frontend-by-backend path requires explicit evidence. Backend-specific escape hatches are allowed, but must be labeled nonportable.

## 4. Frontend portfolio: every listed family is a required implementation

### 4.1 Scope is mandatory; delivery is phased

Build every frontend, native interface, and input family in the implementation inventory below. These are required engineering deliverables, not a menu from which to select one or two.

Sequencing determines WHEN a deliverable is built, not WHETHER it will be built. Do not remove a track, leave it permanently as research, or reduce the approved scope without explicit user approval. Preserve the existing Gate A/Gate B sequence and working implementation while extending the system.

This is an implementation mandate, not a claim that every integration has been proven feasible. Research must lead to working, tested support or an evidenced blocker with a concrete next investigation. A blocker remains incomplete required work; it does not become a completed feature. Never fabricate support to satisfy the mandate.

"All frontends built" means a real, documented, tested execution path for each specified compatibility subset. It does not mean every historical version, every construct, every library, or every input program is supported. Specify each subset before implementation and expand it through tests. Do not shrink an agreed subset merely to mark a track complete.

Independent architecture does not require a new compiler for every language. Reuse mature compilers, validators, reflection libraries, and runtime interfaces when their licenses and technical contracts permit. An upstream compiler bridge counts as implementation only when Paralyn actually integrates compilation, arguments/resources, execution, synchronization, errors, and verification. A command wrapper or placeholder registry entry alone does not qualify.

### 4.2 Required implementation inventory

Split combined rows into independent tracked deliverables. Successful HLSL support does not complete GLSL; successful SPIR-V support does not complete MLIR; PTX does not complete SASS.

| Track | Required deliverable | Qualification boundary |
|---|---|---|
| CUDA C++ | Preserve and extend the complete-program source compatibility path. | Tested host transformations, runtime subset, generated GPU execution, and ordinary source without Paralyn-specific annotations. |
| Native C/C++ | Build the native runtime interface and ergonomic C++ wrappers. | A native program allocates buffers, launches a supported kernel, handles completion/errors, and verifies results without CUDA allocation APIs. |
| Native Python | Build real bindings to the shared runtime, followed by the agreed small array/operator surface. | A Python application uses real device memory/execution; no separate fake execution engine and no claim to compile arbitrary Python. |
| HIP C++ | Build a compiler-assisted source/runtime adapter for an explicit HIP subset. | A HIP program runs through Paralyn with tested host/kernel semantics and clear unsupported-feature errors. |
| Triton | Integrate a pinned compiler/backend boundary and implement the required lowering/runtime bridge. | An actual Triton kernel executes through Paralyn; preserve the supported tile, mask, layout, and dtype contract rather than translating Python text. |
| OpenCL C | Implement kernel import, argument reflection, lowering, and launch integration. | GPU execution of supported OpenCL C kernels; full OpenCL host-runtime compatibility remains a separately specified expansion. |
| SYCL C++ | Integrate an existing compiler/runtime boundary and implement a bounded SYCL execution path. | A scoped SYCL application performs real GPU work; translating a pre-extracted kernel alone is not full SYCL application support. |
| OpenMP target offload | Integrate an appropriate compiler/offload boundary with Paralyn's execution contract. | A supported target region executes on the GPU with tested mapping, completion, and error behavior; host fallback cannot count. |
| Slang | Integrate compilation, reflection, resources, and runtime execution. | Real supported Slang compute programs, with documented compiler versions and target limitations. |
| HLSL compute | Build compute-source import through a qualified existing compiler bridge. | Supported HLSL compute entrypoints execute with verified resource bindings; no implied graphics support. |
| GLSL compute | Build its own compute-source import and execution tests. | Supported GLSL compute entrypoints execute correctly; HLSL success is not evidence for this track. |
| WGSL | Implement validation/import, supported resource mapping, and GPU execution. | Real WGSL compute with an explicit feature/environment contract. |
| SPIR-V inputs | Implement validated import for named versions, capabilities, and execution environments. | Actual supported modules execute; accepting arbitrary SPIR-V or graphics modules is not implied. |
| MLIR inputs | Implement import/lowering for named, versioned dialects and operations. | A supported module produces verified GPU results; arbitrary MLIR dialects are not presumed executable. |
| Metal source | Implement native MSL module compilation, arguments, and runtime execution. | A real Metal-native module path. Mark this path backend-specific unless cross-vendor translation is separately demonstrated. |
| PTX | Implement a versioned PTX compatibility subset with a real non-NVIDIA lowering path. | Supported PTX executes on Apple through Paralyn and is compared against an appropriate NVIDIA reference. Native PTX loading alone is not portable PTX support. |
| SASS | Build a separately gated binary-translation path for one explicitly named NVIDIA instruction-set generation and supported instruction subset, then expand. | Real supported machine instructions are decoded/lowered and execute on non-NVIDIA hardware with reference validation. Parsing, disassembly, or native NVIDIA loading alone does not complete this track. |

For PTX and SASS, do not assume architecture coverage, compiler-generated artifact rights, or runtime/host-binary compatibility. First establish a documented, authorized provenance and tooling path, a precise artifact/ABI contract, available reference hardware, and a feasibility prototype. Obtain any necessary legal review before taking a route whose terms are uncertain. Do not bypass applicable restrictions. If a required prerequisite cannot be established, report that track as BLOCKED rather than pretending it is shipped or quietly deleting it. Source frontend progress does not need to halt while a separate binary track is blocked.

### 4.3 Required named interoperability clients

Numba CUDA, CuPy, and CUDA Python were previously described as possible adapters. Treat each as a required, separately tracked integration for a bounded workload and pinned upstream version. They are clients/integrations, not new kernel languages.

For each, select and approve an explicit integration surface, implement it, run a real workload, and publish limitations. Do not turn a custom-operation demonstration into a claim of complete framework replacement. Do not falsify device identity or monkey-patch capability checks to claim compatibility.

The generic phrase "C++ portability libraries" is not an unlimited, enumerable requirement. Before adding such a library to the mandatory inventory, name the library, integration surface, and workload explicitly in a scope decision. The already named frontend families above cannot be made optional by this rule. This amendment does not silently make every research project in Section 1 a required implementation.

### 4.4 Qualification contract for every track

Create a durable record for each track containing:

- Exact input language/API/artifact versions and a meaningful supported subset.
- A named useful workload in addition to a simple reference example.
- Compiler/provider dependencies, licenses, provenance, and reproduction instructions.
- The actual import, lowering, executable, argument, memory, and synchronization path.
- Positive, negative, and integration tests; numerical and error guarantees.
- Tested hardware/toolchain identity and execution evidence.
- Target backends, currently verified cells, known gaps, and the next concrete implementation task.

A track is not implemented just because it appears in the CLI, emits an IR dump, or has a folder and a mock test. It must accept actual input from that family, compile or import it through the stated path, run GPU computation, and independently verify output.

At minimum, qualify varied inputs and dimensions, repeat execution, verify errors for unsupported input, and prove that computation is not hardcoded. Kernel-capable tracks should include bounds-checked vector addition and a second operation that exercises a distinct supported feature. API/client tracks require an equivalent end-to-end application. Binary tracks require executable reference artifacts, not fabricated golden outputs.

Hardware absence is UNAVAILABLE, not PASS. An implemented adapter with only one tested backend is useful partial progress, not evidence of cross-vendor support.

### 4.5 Cross-vendor target

For each frontend designated portable, define a nontrivial common subset and require qualification of that subset on Metal/Apple, CUDA/NVIDIA, and HIP/ROCm/AMD as those backends are implemented. Native/backend-specific modules must remain clearly labeled; native MSL execution does not automatically promise Metal-to-CUDA translation.

Where the upstream compiler does not already support a required target, that is an implementation gap to solve through a documented bridge or translation path, not permission to assume a target exists. Record progress per frontend-by-backend cell. Do not overstate support based on a vendor name alone.

Do not advertise universal cross-vendor coverage until all required portable cells actually pass. Individual experimental releases may ship a truthful partial matrix while the mandatory overall portfolio remains incomplete.

### 4.6 Build campaign and completion ledger

Maintain `docs/frontend-matrix.md` and a machine-readable portfolio ledger. Separate obligation from evidence:

```text
required: true
implementation: not_started | implementing | implemented | blocked
qualification: unavailable | failing | verified
portability: explicit per-backend results
```

A track can remain REQUIRED while hardware qualification is UNAVAILABLE. Those are different facts.

Use dependencies to choose the next ready implementation. Choose the first non-CUDA frontend to validate extension boundaries, then continue through the remaining inventory; it is not the only non-CUDA frontend to build.

Research, architecture, and scaffolding are intermediate work, not portfolio completion. Do not stop the entire project at a registry or one representative language. Preserve the complete implementation inventory in handoffs so later agent sessions continue rather than repeatedly replanning it.

An individual invocation can end at a useful checkpoint, a real blocker, or an execution limit. That does not authorize the agent to reduce the project's required scope. Report precisely what is implemented, what is verified, what is blocked, and what comes next.

## 5. Native Paralyn API: make new code feel natural

The native API must not require CUDA names, NVIDIA types, or a fictitious CUDA device.

First expose the real runtime: context, device, owned buffer, buffer view, compiled module, kernel, typed arguments, queue, event, and structured errors. Prefer a small C-compatible boundary with ergonomic C++ wrappers if the implementation justifies it. Do not promise a stable external ABI before it is tested.

Then bind that same runtime into Python. Do not create a separate Python memory manager or execution engine. Leave room for Rust, Swift, and other language bindings through the same runtime boundary; a language binding does not need to become another kernel compiler.

This is a PROPOSED future Python experience, not an already supported API:

```python
import paralyn as p

device = p.device("auto")

a = p.asarray([1.0, 2.0, 3.0], dtype=p.float32, device=device)
b = p.asarray([4.0, 5.0, 6.0], dtype=p.float32, device=device)

c = p.add(a, b)

print(c.device)
print(c.to_numpy())  # Explicit host materialization and synchronization.
```

Initially, auto selects one eligible local GPU or returns an actionable error. It must not secretly choose a CPU, rent cloud hardware, or join a cluster.

Separate owned buffers from borrowed views. Specify lifetime, mutation, synchronization, zero-length allocation behavior, shape/stride handling, dtype conversions, and copy semantics before expanding the array surface.

Begin Python support with buffers and launching already-supported kernels. Add a deliberately small array/operator subset afterward. Do not accidentally rebuild NumPy, autograd, or a complete deep-learning framework.

A future @paralyn.kernel syntax is a design candidate, not an instruction to write a Python compiler now. Evaluate reuse of Triton, Slang, or another suitable frontend before creating a new kernel language. Clearly document any restrictions on Python accepted for compilation.

Every introductory API example must eventually become an executable documentation test. Before implementation, label it DESIGN TARGET.

## 6. Framework integrations: use real extension boundaries

PyTorch integration must distinguish custom operators, a device backend, exported graphs, and a torch.compile backend. These are separate achievements. Evaluate documented accelerator/PrivateUse1 and compiler extension paths against a pinned PyTorch version. Never monkey-patch torch.cuda.is_available to pretend an Apple GPU is NVIDIA hardware. Never claim training support from an inference-only demonstration. [R5, R6]

JAX integration should investigate PJRT and the relevant graph/compiler boundary. Accepting StableHLO does not itself implement a working JAX device. Buffer ownership, compilation, execution, synchronization, error propagation, and version compatibility still require implementation. [R4, R7]

For ONNX, choose a tested importer or ONNX Runtime execution-provider path. Document operator sets, dynamic-shape limits, numerical behavior, and partition boundaries. Audit any framework fallback: a provider running one operator on GPU does not prove the entire model ran there. [R8]

TensorFlow support should begin with an explicitly supported export or plugin route, not a blanket compatibility promise. MLX, NumPy, and Array API interoperability should begin with tested data exchange and explicitly supported operators, not claims of complete semantic equivalence.

Evaluate DLPack where both sides genuinely support the memory and synchronization contract. It is not automatic zero-copy across different devices, runtimes, machines, or Metal integrations. Unsupported borrowing must fail or request an explicit documented copy. Never relabel a native device allocation to trick another framework into accepting it. [R9]

Choose ONE first framework integration after the native/runtime foundation is ready. Compare a small PyTorch operator, a bounded compiled graph, and a small ONNX model using actual implementation cost and user value.

## 7. Libraries: preserve operations instead of translating every instruction

Add a vendor-neutral operator layer separate from source compatibility.

Candidate families are dense linear algebra, reductions/scans, FFTs, random generation, sparse algebra, sorting/gather/scatter, neural-network primitives, and eventually collectives.

The first substantial library target should be a precisely specified FP32 matrix multiplication. Candidate native providers include cuBLAS on NVIDIA, rocBLAS on AMD, and suitable MPS functionality on Apple. Their existence does not establish interchangeable APIs or full equivalent coverage. Verify the operation contract on each selected provider. [R10, R11, R12]

Conceptual dispatch:

```text
native matmul / supported library compatibility call / graph matmul
                              |
                   common operator specification
                              |
            qualified native provider OR generated GPU kernel
                              |
                   shared runtime execution contract
```

Specify layout, transpose behavior, strides, batching, accumulation precision, workspace, aliasing, queue ordering, and failure behavior. Report the selected provider.

Native Paralyn math APIs and compatibility shims for cuBLAS/cuFFT/cuRAND/cuSPARSE/cuDNN are separate deliverables. One matmul implementation is not a cuBLAS implementation. Never expose a success-returning stub for an unimplemented library function.

For random generation, define reproducibility and stream-partitioning guarantees. For FFTs, define normalization and complex layout. For neural-network primitives, separate forward, backward, and mixed-precision coverage. For collectives, specify topology and collective participation before claiming compatibility.

Use existing optimized providers when they satisfy the contract. A portable generated GPU kernel may be an explicit alternative. CPU implementations remain test references unless an explicitly selected CPU backend is introduced later.

## 8. Backends and honest capability discovery

Metal remains first. Add NVIDIA CUDA and AMD HIP/ROCm through independent tested milestones. Evaluate Vulkan/WebGPU and Intel-specific execution only against named hardware, supported operating systems, and compiler paths.

Capability records must describe more than vendor names: supported artifact formats, dtypes, arithmetic modes, workgroup limits, subgroup behavior, shared memory, atomic scopes, synchronization, allocation limits, provider availability, and relevant driver/toolchain versions.

Validate a workload against capabilities before launch. Do not silently reduce precision, invent missing operations, or assume identical subgroup/warp behavior across architectures.

Model graph-only accelerators separately from general kernel backends. A possible Core ML/Neural Engine provider should accept compatible model/tensor work, not advertise arbitrary CUDA kernel execution. Core ML's documented compute-unit settings allow OS-managed combinations of CPU, GPU, and Neural Engine; report the execution policy and measured placement rather than promising guaranteed ANE-only execution. [R13]

A graph provider that permits CPU execution must require explicit policy consent. It cannot satisfy a GPU-only or ANE-only qualification gate when actual placement is unverified.

Likewise, future TPU/NPU/DSP/FPGA work requires a supported public compiler/runtime path and its own contract. Do not enumerate hypothetical devices as usable.

## 9. Memory, execution, and numerical contracts

Keep frontend-specific CUDA tokens outside the core. Core buffer views carry allocation identity, device/context ownership, byte offset, size, alignment, access mode, and lifetime information.

All adapters, providers, and backends must agree on argument representation and completion behavior. Define host/device scalar placement, mutable outputs, dependencies, resource retention, cancellation limits, and asynchronous error delivery.

Physical shared memory is not permission to skip synchronization. Aggregate cluster memory is not one address space. Report physical memory, allocation limits, current owned allocations, and working-set recommendations as distinct measurements.

Define portable semantics, optional capabilities, and explicitly backend-specific features. Preserve the current v0 restrictions until each expansion has tests.

Define numerical policy per operation. Distinguish dtype, accumulation dtype, FMA/contraction, reduced-precision modes, overflow, NaNs, signed zero, subnormals, rounding, and deterministic execution. Do not promise cross-vendor bitwise identity when the contract only supports a documented tolerance.

The default compatibility path has no CPU kernel fallback. A future CPU backend must be explicitly selected and reported. Ordinary host code and CPU-reference verification are not GPU kernel fallback; keep them visibly separate.

## 10. Developer experience: simple defaults, inspectable decisions

Make the simple case simple and the advanced case inspectable. For a supported local workload, aim for one obvious command and no mandatory YAML. Reveal compilation, placement, transfers, and precision controls progressively without hiding consequential behavior.

The intended command family is:

```bash
paralyn doctor
paralyn devices --json
paralyn inspect examples/vector_add.cu
paralyn run examples/vector_add.cu --device auto --explain
paralyn run examples/vector_add.cu --device metal:0 -- --program-argument
```

These are DESIGN TARGETS until implemented. Preserve existing working commands and numeric device selection through a documented transition.

Doctor should report detected hardware, usable backends, installed versus missing dependencies, and exact corrective steps. Do not claim a backend works merely because its library loads.

Inspect must reuse the real compilation/verification path, show supported kernels and capabilities, and distinguish static information from runtime expressions. It must not execute the user's host program to invent a supposedly static execution plan.

Explain should state the selected device/provider, why alternatives were rejected, known transfers, compilation behavior, and applicable numerical policy. Unknown costs remain unknown rather than fabricated estimates.

Errors should identify the user's source or operation, explain the unsupported capability, and recommend an actual available alternative. Do not recommend unimplemented flags or dependencies unrelated to the detected failure.

Default output should be readable; structured JSON and detailed IR/backend diagnostics should be available for automation. Avoid leaking sensitive source paths or tensors into public logs.

Keep installation modular. A Metal-only developer should not need every vendor SDK. Separate compiler/development dependencies from execution-only dependencies where practical. Never overwrite system compilers or perform privileged installations without authorization.

## 11. Execution planning and scalability

Start with deterministic single-device selection. Choose devices that satisfy the full supported capability and memory contract, then use a documented tie-breaker. Preserve data locality and explicit placement. Do not claim optimal scheduling without measurements.

Later, use measured compilation, launch, transfer, provider, and execution costs to compare plans. Explain estimates and uncertainty. Revalidate availability and allocation success at execution time.

Plan for multiple devices through explicit buffers and task dependencies. Do not distribute arbitrary CUDA kernels across machines or treat network barriers as equivalent to device-local synchronization.

The first distributed target should be independent jobs or explicitly partitioned tasks. Model/tensor partitioning comes later and requires a supported graph, not speculative analysis of arbitrary host C++.

Keep control-plane metadata separate from data-plane transfers. Model locality, bandwidth, latency, bounded queues, backpressure, memory budgets, quotas, cancellation, node failures, and version negotiation.

Define retry semantics. Never blindly retry work with externally visible side effects. Benchmark useful throughput and scaling efficiency, not just node count or summed memory.

Distributed code, a dashboard, cloud provisioning, and a compute marketplace remain outside current implementation scope. No cloud costs, external uploads, or network enrollment without explicit authorization.

## 12. Security and open-source collaboration

Keep project-owned code under the intended Apache-2.0 license, subject to dependency and ownership review. Preserve third-party notices, provenance, and applicable license exceptions. Do not copy proprietary implementations or claim that an independent implementation automatically resolves every IP issue.

Use documented public interfaces. Review applicable terms before binary translation, SDK-derived artifact translation, or redistribution. Keep optional proprietary drivers/SDKs distinct from the open-source core.

Maintain CONTRIBUTING, SECURITY, a code of conduct, dependency inventory, and an accurate compatibility matrix. Choose a contributor-signoff policy deliberately. Define contribution areas around real modules and scoped tests, not imaginary completed backends.

Source compilation and GPU execution are not inherently safe for untrusted submissions. Initially support trusted local code. Define a threat model before remote execution, plugin installation, or public multi-tenant operation.

Hardware CI must not run unreviewed pull-request code with host secrets or privileged access. GPU drivers and native libraries are not a sandbox. Use controlled test promotion, minimal credentials, and explicit runner ownership.

Do not publish the repository, packages, releases, binaries, or machine/network access as a side effect of implementing this mandate. Publication is a separately authorized action after the applicable qualification gate.

## 13. Qualification and evidence

Keep Gate A and Gate B from the revised plan intact.

Gate A proves the complete source-to-Apple-GPU path. Its canonical demonstration uses 1,024 elements and block size 256 with distinct allocations and exactly representable FP32 inputs. No compiler/runtime special-casing of that example is allowed.

Record source, verified IR, generated MSL, transformed host source, device/toolchain identity, command-buffer status, GPU timing evidence, and independently checked output. Only the actual comparison may emit Verification: PASS. Host wall-clock duration alone is not proof of GPU execution.

Gate B remains required before tagging or public release. Preserve varied lengths and launch shapes, randomized data, changed arithmetic, scalar arguments, repeated launches, sentinels/canaries, alias cases, invalid inputs, host-side semantics, explicit FP32 policies, and failure propagation.

Retain the benchmark protocol: four documented sizes, ten warmups, one hundred measured iterations, alternating generated/native order, all samples saved, and separate compilation, transfer, GPU duration, total latency, and runtime-owned memory reporting. No fabricated performance targets.

For later features, maintain a machine-readable frontend x operation x backend x version matrix. Suggested statuses are PROPOSED, EXPERIMENTAL, VERIFIED, UNSUPPORTED, and UNAVAILABLE. Mock or CPU-reference success cannot produce VERIFIED hardware status.

Add differential tests, randomized/property tests, negative capability tests, and compiler/IR fuzzing where useful. Cross-vendor results require execution on each claimed vendor. Failure on one path must not be hidden by quietly choosing another.

Measure API usability through a fresh setup and small end-to-end tasks, not only unit tests. Track actual time-to-first-result and dependency/setup failures without making universal installation-time promises.

## 14. Repository evolution and documentation

Extend existing directories rather than reorganizing everything at once. Conceptual boundaries are frontends, compiler representations/passes, runtime, backends, operator providers, framework adapters, language bindings, conformance tests, benchmarks, and documentation.

Create directories when they contain meaningful implementation or a necessary specification. Do not manufacture a forest of empty plugins.

Maintain or extend:

```text
AGENTS.md
ROADMAP.md
CHANGELOG.md
docs/status.md
docs/architecture.md
docs/prior-art.md
docs/novelty.md
docs/frontend-matrix.md
docs/native-api.md
docs/numerics.md
docs/execution-policy.md
docs/decisions/
```

Consolidate documents where that improves clarity. Generate status tables from evidence when practical. README examples must match the current release, and future examples must be clearly labeled.

Record tested versions and capability schemas. Revisit persistent caches only after the current path is correct. Any later cache must account for source/dependencies, compiler/IR/runtime versions, backend target, options, numerical policy, alias layout when relevant, and corrupted or incompatible entries.

## 15. Evidence-gated roadmap for the complete portfolio

The scope in Section 4 is required. Stages sequence implementation; they do not convert later tracks into optional ideas. Do not map these stages to promised dates or inflated software versions. Early releases may truthfully ship only the completed portion.

**Stage A: Finish the existing CUDA-to-Metal vertical slice.** Reproduce and complete Gate A, then Gate B. No new frontend or framework implementation blocks this milestone.

**Stage B: Build the native C/C++ and Python runtime interfaces.** Reuse the same runtime. Demonstrate allocation, supported kernel launch, explicit synchronization, errors, and verified results without CUDA allocation APIs. Complete their agreed initial qualification contracts and then add the documented small operator surface.

**Stage C: Prove a second hardware backend.** Reuse source/conformance workloads on available NVIDIA or AMD hardware. Keep native-API and source-compatibility results separate. Continue the unimplemented vendor backend in later stages; verifying one does not cancel the other.

**Stage D: Deliver a useful operator/provider path.** Add a defined FP32 matmul and test it through the native API with provider and numerical reporting. Qualify additional hardware paths as devices become available.

**Stage E: Start and execute the full source/IR frontend campaign.** Validate the extension boundary with one non-CUDA frontend, then continue until HIP, Triton, OpenCL C, SYCL, OpenMP target offload, Slang, HLSL compute, GLSL compute, WGSL, SPIR-V, MLIR, and native Metal source each have their required implemented and qualified paths. Order compiler bridges to maximize reuse, but do not delete tracks to simplify the roadmap. Complete work in bounded submilestones with evidence.

**Stage F: Deliver named interoperability clients and an external framework workload.** Implement the scoped Numba CUDA, CuPy, and CUDA Python integrations from Section 4. Separately deliver the first scoped PyTorch, JAX, or ONNX integration from Section 6 and a useful real workload. These client/framework claims must remain distinct from source compiler coverage.

**Stage G: Complete portable-subset cross-vendor qualification.** Finish NVIDIA and AMD backend coverage and the required common-subset frontend matrix. Expand operators and language constructs through conformance tests. Document any backend-specific exceptions rather than implying full portability.

**Stage H: Implement the separately gated PTX and SASS tracks.** Establish authorized artifact/tooling provenance, versioned subsets, ABI contracts, reference hardware, and genuine non-NVIDIA execution. Keep translation, native execution, and host-program compatibility separate. Research alone does not complete either track. An unresolved prerequisite remains a visible blocker in the full portfolio.

**Stage I: Distributed execution and specialized accelerators.** Retain the separate independent-job, graph-partitioning, and restricted-accelerator contracts. Do not divert the current implementation into a speculative cloud platform while established frontend milestones are unfinished. Independent investigation may proceed when it does not undermine the required core work.

A narrowly scoped external CUDA project may enter as soon as the supported subset permits. Stages can interleave when dependencies and tests justify it, especially when a track is blocked on unavailable hardware. Record the reason and preserve the required inventory. The project must never be described as having completed the all-frontends mandate while required implementations or qualification cells remain unverified.

## 16. Agent workflow and the next action

Work as:

```text
inspect -> identify next acceptance gap -> bounded research/design
        -> implement -> compile -> run -> verify -> document -> commit
```

Never report a successful build or test unless it actually ran. Separate code inspection, compilation, mock tests, CPU-reference checks, and physical-GPU execution in reports.

When hardware, permissions, dependencies, or tools are unavailable, record the exact blocker and commands needed in an appropriate environment. Continue useful independent work without marking blocked execution complete. Do not invent timings, device identities, or command output.

Keep changes small. Preserve unrelated user work. Use local commits only when appropriate; do not push, publish, change remotes, or acquire paid resources without authorization.

Start now by auditing the existing repository and reproducing its baseline. Write a short v1.1 all-frontends delta describing retained code, current proven behavior, architectural changes needed now, and the full required implementation ledger. Do not treat the new mandate as permission to restart the project.

If Gate A is incomplete, finish Gate A. If Gate A passed but Gate B is incomplete, finish qualification. If both passed, implement the smallest native-runtime API slice that advances its agreed contract. Then select the next dependency-ready milestone in the required portfolio.

Work through bounded, testable milestones. A completed milestone is a checkpoint, not proof that the all-frontends project is finished. When the user has authorized the ongoing build and the environment permits, continue to the next ready milestone instead of stopping permanently after the first non-CUDA integration. This does not authorize background execution, additional costs, publication, unsafe actions, or access beyond the available session.

At checkpoints or session end, record changed files, commands executed, tests passed/failed/unavailable, hardware evidence, remaining required tracks, blockers, and the exact next task. Preserve a resumable handoff. Never claim the whole project is complete because the current invocation reached its limit; never claim a session will continue running after it has stopped.

The standard is not how many integrations the diagram contains. It is how much useful computation a developer can run correctly, understand, and reproduce.

---

## Research anchors for this mandate

Research anchors retained from the original v1.1 mandate. This all-frontends amendment changes implementation requirements; it does not newly verify upstream support, integration feasibility, or artifact rights. Recheck these primary sources and pin exact upstream versions before implementation. Links are source references, not automatic download or execution instructions.

```text
R1  IREE deployment configurations
    https://iree.dev/guides/deployment-configurations/
R2  Khronos SYCL
    https://www.khronos.org/sycl/
R3  MLIR rationale
    https://mlir.llvm.org/docs/Rationale/Rationale/
R4  StableHLO specification
    https://openxla.org/stablehlo/spec
R5  PyTorch accelerator integration
    https://docs.pytorch.org/docs/main/accelerator/index.html
R6  PyTorch custom compiler backends
    https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_custom_backends.html
R7  PJRT plugin integration
    https://openxla.org/xla/pjrt/pjrt_integration
R8  ONNX Runtime architecture
    https://onnxruntime.ai/docs/reference/high-level-design.html
R9  DLPack Python specification
    https://dmlc.github.io/dlpack/latest/python_spec.html
R10 NVIDIA cuBLAS
    https://docs.nvidia.com/cuda/cublas/index.html
R11 AMD rocBLAS
    https://rocm.docs.amd.com/projects/rocBLAS/en/latest/index.html
R12 Apple MPSMatrixMultiplication
    https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication
R13 Apple Core ML compute units
    https://developer.apple.com/documentation/coreml/mlcomputeunits
R14 AMD HIPIFY
    https://rocm.docs.amd.com/projects/HIPIFY/en/latest/index.html
R15 Triton project
    https://github.com/triton-lang/triton/blob/main/README.md
R16 Slang documentation
    https://shader-slang.org/docs/
```
