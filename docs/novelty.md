# Novelty assessment

Assessed 2026-09-25. **Paralyn's novelty is unproven.** A successful Gate A is a demonstrated independent implementation, not evidence that its architecture or functionality is new. See `prior-art.md` for the dated, ten-question comparison.

## Existing solutions and overlap

[CuMetal](https://github.com/Lulzx/cuda-metal) is direct prior art for CUDA-source compilation, a CUDA-facing runtime, typed IR, inspectable compilation stages, and physical Apple GPU execution. Its published architecture and validation approach overlap the proposed Gate A substantially. We have not reproduced its claims or compared its performance.

[ZLUDA](https://github.com/vosen/ZLUDA) pursues existing CUDA application execution on non-NVIDIA GPUs. [HIPIFY](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/) migrates CUDA source into HIP. [SYCL](https://www.khronos.org/sycl/) and [OpenCL](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html) already expose heterogeneous compute programming interfaces.

[IREE](https://iree.dev/guides/deployment-configurations/) already separates portable compilation from runtime drivers across Metal, CUDA, ROCm, Vulkan, and CPU configurations. [StarPU](https://starpu.gitlabpages.inria.fr/features.html) already handles heterogeneous task placement, data transfers, and cluster communication. [JAX](https://docs.jax.dev/en/latest/) and [XLA](https://openxla.org/xla) already provide portable compiled array computations and distributed execution facilities.

[MetaXuda](https://github.com/Perinban/MetaXuda) explores Numba CUDA compatibility on Metal. Its custom restricted license and documented correctness limits rule it out as a dependency for this independent Apache-2.0 implementation.

## Unresolved engineering questions

This review does not identify a proven globally unsolved research problem. It identifies work Paralyn must still demonstrate:

- A precise CUDA-source subset whose host and device semantics are tested on each supported vendor.
- Consistent feature discovery, unsupported-feature diagnostics, pointer ownership, aliasing, and asynchronous error behavior across unlike backends.
- A common numerical/conformance policy backed by actual hardware evidence.
- A maintainable path from a tiny example to useful external programs without special-cased kernels.
- Eventually, explicit device/data placement and explainable scheduling without pretending that networked memory is physically unified.

Existing projects may already address portions of these questions. Comparative claims require a versioned implementation study and reproduced tests.

## Potential contribution

A small independent codebase may be useful as an auditable reference implementation and conformance harness. The longer-term potential contribution is a consistent CUDA-source compatibility contract across Apple, NVIDIA, and AMD, accompanied by evidence and limits for each backend. Gate A demonstrates only the Apple path and cannot establish that longer-term claim.

Selecting a custom IR, refusing CPU fallback, retaining compilation artifacts, or passing vector addition are sound engineering choices, not novelty claims. No benchmark superiority or feature superiority is asserted.

## Claims we must not make

Do not claim to be the first CUDA-on-Metal compiler, the first portable GPU runtime, a drop-in replacement for the whole CUDA toolkit, CUDA binary compatible, transparent to PyTorch, production-ready, universally numerically equivalent, or capable of combining cluster memory into one GPU allocation. Do not label a planned backend as supported or a successful CPU reference as successful GPU execution.

## Defensible mission

Independently implement and test a documented CUDA-source subset on supported GPU backends, preserving ordinary host C++ semantics and exposing clear limitations. Begin with a complete vector-add program on a physical Apple GPU; expand only when the previous compatibility gate is verified.
