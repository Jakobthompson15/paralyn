# Changelog

## Unreleased — v0.0.1 (2026-09-25)

- Renamed the project from its original working name UniCUDA to Paralyn; current executable, namespace, build targets, and newly generated IR artifacts use the new name. All four automated tests pass after the rename.
- Independent CUDA AST frontend, typed verified IR, native host rewrite, and inspect command.
- Metal runtime with explicit memory copies, opaque allocation tokens, error propagation, and GPU execution evidence.
- Ordinary CUDA vector-add example and focused compiler/runtime tests.
- Prior-art and architecture documentation; no tagged release has been published.
- First complete CUDA-source execution on physical Apple M5: 1,024 results independently matched the CPU reference. Preserved five unchanged historical Gate A artifacts, including `unicuda-ir.txt`, from clean original-name implementation revision `3f3c960f0eb712869cbc99b81e3fdd84394e8efe`.
- Repeated the complete GPU proof under Paralyn from clean revision `c823dfcdc4c3d37d8ed1648b4b0d93825cbdb6b1`; saved fresh evidence in `artifacts/paralyn-gate-a/`.

### Gate B implementation

- Adopted mandate v1.1 without changing the software version; tracked all 17 required frontend families and three named clients with separate backend evidence.
- Added physical-GPU correctness qualification for edge/odd lengths, seeded inputs, changed arithmetic, scalar values, aliases, x/y/z indexing, integer conversions, FP32 numerical policy, host semantics and explicit failures.
- Added a four-size generated/handwritten-Metal benchmark with all 880 samples and independent evidence/provenance auditing.
- Fixed entrypoint pipeline-cache identity, inactive IR target serialization and decimal integer emission; disabled implicit FP32 contraction in generated Metal.
- Preserved transformed host code and per-launch shader source; added runtime timing and buffer accounting.
- Restored deprecated unicuda command, CMake library aliases and C++ compatibility includes/namespace while retaining Paralyn as the project name.

- Qualified Gate B on Apple M5 from clean revision `8af773c9b8925a27041fcca1ab4cc58c82c56edb`: all six CTest targets, 75 correctness launches and 880 benchmark launches passed. Preserved full raw evidence under `artifacts/gate-b/`; native APIs and the wider frontend/backend portfolio remain incomplete.
