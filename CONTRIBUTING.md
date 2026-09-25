# Contributing to UniCUDA

Start with `README.md`, `docs/status.md`, `docs/cuda-compatibility.md`, and `docs/architecture.md`. Planned features in `ROADMAP.md` are not supported features. Complete the active acceptance gate before expanding scope.

Use one change to solve one concrete compatibility or correctness problem. Describe the source program or failure that motivates it, the semantic behavior before and after, and how it was checked. For architecture changes, record the observed evidence and update the design.

## Engineering rules

- Preserve ordinary host C++ outside CUDA constructs owned by the frontend.
- Lower device code through verified typed IR; do not match example names, input sizes, or expected outputs.
- Keep backend-native objects outside portable compiler/runtime interfaces.
- Reject unsupported behavior clearly with source locations where available.
- Never mask compiler/runtime errors, fake a passing result, or execute kernels on the CPU as a silent fallback.
- Do not call a feature supported until the relevant tests and physical backend execution have passed.

Use C++17 for portable components and Objective-C++ only at the Metal boundary. Follow nearby code style. Match LLVM/Clang headers and libraries from one selected installation. Follow the current build commands in `README.md`; run CTest with failure output and run affected GPU tests on real hardware. Record unavailable hardware as unavailable, not passing.

Every enabled feature needs compiler, runtime, backend, and applicable conformance coverage. Prefer small tests that detect semantic failures. Preserve command errors and raw evidence needed to reproduce them. Benchmarks follow correctness and must retain all samples and environment details.

## Contributions and provenance

Original contributions are submitted under this repository's Apache-2.0 license. Only submit material you have the right to contribute. Record the source, exact version, copyright, and license of any imported third-party material in the change and `THIRD_PARTY_LICENSES.md`. Do not import proprietary implementation code, CuMetal code, or MetaXuda binaries as a shortcut to this independent implementation.

Before proposing a public release, finish Gate B, verify the dependency notices for any distributed binaries, and make `README.md` and `docs/status.md` accurately describe demonstrated behavior. Keep local commits small and descriptive. Publishing, pushing, and tagging are separate actions from completing a local development milestone.

Treat contributors with respect; see `CODE_OF_CONDUCT.md`. Handle potential vulnerabilities through `SECURITY.md`.
