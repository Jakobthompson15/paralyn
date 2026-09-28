# Checkpoint: native-product implementation batch

Starting local HEAD and remote main were both `f3c6c9955daf5a81d764fd0e634eb786a8489bcd`. This extends that working repository and preserves ABI1, `.prk`v1, CUDA semantics and immutable historical evidence. The user explicitly adopted the [complete-platform M0–M9 program](complete-platform-program.md). Software remains0.0.1; this checkpoint does not complete the program or imply a release tag.

## Actual implementation

- Runtime: conditional Metal/Objective-C++ and compiler targets; runtime-only/no-backend builds; backend-qualified identities and versioned capability/timing queries; production compiled MSL modules; retained resources; separate runtime progress and durable NDJSON events.
- Artifact/compiler: bounded hashed PARALYNX1 native MSL source/resource container, UTF-8 and tokenized explicit-signature validation, actual Metal reflection; C ABI module auto-detection. Existing `.prk` stays unchanged. Actual vector-add/read-alias, threadgroup reduction and tiled transpose inputs execute on M5.
- Native product: owned contiguous one-dimensional FP32 Python/C++ arrays, copied upload/readback, add/affine through shared GPU runtime; same-input aliases; empty operations; ownership/error tests; relocatable module discovery and deterministic offline macOS arm64 wheel.
- Terminal: C++/CLI11, devices/doctor/support/check/explain/inspect/compile/run/builtin verify/report, qualified Metal selectors, versioned JSON, separate application streams, runtime event sidecars, program exits, POSIX spawn/interrupt handling and diagnostics. Full terminal family/project/case configuration remains incomplete.

Exact files/targets and final clean evidence will be recorded in `docs/implementation-audit.md`, `docs/status.md` and the new qualification archive. Development testing includes all preserved CUDA/native paths, new arrays/MSL/terminal/query/process tests, installed Python3.9.6/3.14.5 and relocated C++ execution. An initial regression caught pipeline-reuse text moving out of the legacy verification transcript; the compatibility transcript now includes runtime logs while application streams remain separate. Gate B then passed unchanged. A full native-app CLI test also caught an evidence-directory collision; native apps now export to a separate new subdirectory and use the CLI's matching default native library/operator artifact. `native_cli_arrays` verifies both complete GPU applications and their saved evidence.

## Exact next task per workstream

1. **Runtime/backends:** collect existing authorized Windows inventories using `scripts/windows-inventory.ps1` and record GPU model/VRAM/driver/OS. No remote access or models have been supplied. Independently implement a dynamically loaded CUDA Driver/NVRTC backend behind the current Context/CompiledExecutable/Buffer/Event boundary, preserving primary-context interoperability/restoration and duration-only timing. Start with actual generated scalar-IR add/affine; qualification requires real NVIDIA hardware. HIP/HIPRTC follows as a separate required backend. Do not call the no-backend portability build an implemented vendor backend.
2. **Compilers/frontends:** pin exact SPIRV-Tools and SPIRV-Cross versions/hashes/build recipes. Implement Vulkan1.1/SPIR-V1.3 GLCompute buffer/scalar reflection into executable descriptors, including layout/builtin/workgroup/capability validation. Run real assembled vector addition and structured-loop/workgroup reduction on Metal. A shader-to-CUDA/HIP bridge remains separately required; do not reinterpret OpenCL Kernel modules. Current machine has no SPIR-V compiler tools installed.
3. **Operators/integrations:** introduce a versioned tensor descriptor with explicit FP32 shape/layout and contiguous2D matmul. Implement a real Metal provider or generated MSL kernel through the public executable/runtime contract; preserve1D arrays and independent reference tests, then build a two-layer FP32 MLP inference application. Record provider/accumulation/order/alias semantics. This does not complete cuBLAS compatibility, PyTorch integration or cross-vendor matmul.
4. **CLI/qualification:** add strict declarative TOML project and typed kernel-case schema, preserving single-file programs. Implement kernel-only execution for public MSL/IR with declared inputs/outputs/geometry and independent verification contracts. Add malformed/missingcase tests; never invent data or reference. Before large-output production use, stream application captures into bounded sidecar storage and retain partial reports on failures. Continue packaging/completion/offline and diagnostics work through shared commands.

M8 prerequisite work can proceed alongside these tasks: establish permitted PTX/SASS fixtures, applicable terms, exact ISA/ABI and available NVIDIA reference generation. Permission or hardware absence remains an incomplete prerequisite; it does not remove either track. No hardware acquisition, rental, enrollment or automatic remote execution is authorized by this checkpoint.

## Remaining full mandate

All17 input/interface families and three clients remain required. CUDA, native C/C++, native Python and native Metal have partial implemented profiles; HIP, Triton, OpenCL, SYCL, OpenMP/Fortran, Slang, HLSL, GLSL, WGSL, SPIR-V, MLIR, PTX and SASS still lack executable import/translation paths. Native MSL remains Metal-specific. NVIDIA/AMD backends, Numba CUDA/CuPy/CUDA Python, Python/C++ function compilation, automatic region offload, full tensor/operator/library compatibility, PyTorch/JAX/ONNX, AI inference/training, scientific/image/external applications, full terminal UX and multi-platform/offline release qualification remain unfinished. The ledger and complete-platform program preserve their individual obligations.

## Reproduction and clean capture

```sh
cmake -S . -B build -G Ninja
cmake --build build -j4
ctest --test-dir build --output-on-failure -j1
python3 scripts/qualify_product.py --build build --output artifacts/runs/product-new --require-clean
python3 scripts/qualify_native.py --build build --output artifacts/runs/native-new --require-clean
python3 scripts/qualify_gate_b.py --paralyn build/paralyn --artifacts artifacts/runs/cuda-new --require-clean
python3 scripts/verify_benchmark.py --benchmark build/gate_b_benchmark --paralyn build/paralyn \
  --source examples/vector_add.cu --artifacts artifacts/runs/benchmark-new --require-clean
```

Use fresh directories and serialize physical GPU captures. Performance captures require an otherwise idle GPU. Build/qualification scripts use actual source/library/module/binary hashes, exact versions and clean revision. Archive evidence unchanged in a subsequent evidence commit, then reconcile status. Do not convert development tests or a brief doctor probe into the full benchmark qualification. Actual hardware-fault injection, Windows/Linux execution and unavailable vendor tests remain unqualified.
