# Changelog

## Unreleased — v0.0.1 (2026-09-25)

- Renamed the project from its original working name UniCUDA to Paralyn; current executable, namespace, build targets, and newly generated IR artifacts use the new name. Fresh renamed-build verification and GitHub publication are pending.
- Independent CUDA AST frontend, typed verified IR, native host rewrite, and inspect command.
- Metal runtime with explicit memory copies, opaque allocation tokens, error propagation, and GPU execution evidence.
- Ordinary CUDA vector-add example and focused compiler/runtime tests.
- Prior-art and architecture documentation; public release qualification remains deferred.
- First complete CUDA-source execution on physical Apple M5: 1,024 results independently matched the CPU reference. Preserved five unchanged historical Gate A artifacts, including `unicuda-ir.txt`, from clean original-name implementation revision `3f3c960f0eb712869cbc99b81e3fdd84394e8efe`. Fresh Paralyn evidence is reserved for `artifacts/paralyn-gate-a/`.
