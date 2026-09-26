# Native C/C++ and Python runtime checkpoint

Captured from clean implementation revision `76217b7708d7b503449bf42e556e4c148bda88e6` on the physical Apple M5, macOS 26.5.1 (25F80), SDK 26.2, LLVM/Clang 21.1.8, Apple Clang 17.0.0 and CPython 3.14.5. Software remains 0.0.1, native C ABI is 1, and no release tag is implied.

`build-validation.json` records a fresh configure/build, all **14 passing CTests** without skips, every qualification command/log/duration, exact toolchain and binary hashes. Dependencies were already installed. `native/qualification.json` records **21 actual GPU events** across C (2), C++ (1), rigorous C API tests (8), Python tests (9) and the Python example (1). Programs independently compared vector-add/affine output; the C API test additionally passed 35 structured negative checks.

Twenty native commands have complete source-linked execution records. The ninth Python test intentionally releases its original context handle before launching with retained children; its actual `pr_event_wait` timestamps and independent CPU result are captured in `native/python-tests/evidence/python-qualification.json`. The eight earlier events match native command timestamps exactly. The final event is explicitly distinguished, not claimed to have a ninth context-exported command record.

Every native command has actual device identity, completed status, positive GPU timestamps, no error and no CPU fallback. `native/kernels.cu`, `kernels.prk`, inspection output, source-N.metal files, logs, module/library/binary/source hashes and numerical policy are retained. `sources/` snapshots every source/header in the native provenance manifest, including ordinary C/C++/Python clients and the compiler/runtime/tests. All snapshots match the recorded hashes. Native clients link Metal/Foundation and system C/C++ libraries, without LLVM.

`gate-a/` repeats complete ordinary CUDA vector addition. `cuda-correctness/` repeats the current Gate B suite (**75 GPU launches**, including the expected host-exit-37 case). `benchmark/` captures all **880 launches**, four prescribed sizes, ten warmups and one hundred measured iterations per variant/size, alternating generated/native order; every output is checked and all samples retained. These are post-refactor regressions of the shared Metal engine. Earlier `artifacts/gate-a`, `paralyn-gate-a` and `gate-b` captures are unchanged.

Structured API validation, malformed artifacts, host-stream submission failure and unhandled Python finalization have actual tests. The finalizer tests use real C ABI invalid-handle errors and record zero GPU launches. **Actual device-loss/timeout and allocator-exhaustion fault injection remain unqualified.** Positive timestamps establish physical execution, not universal performance or numerical equivalence.

The first native byte-buffer/launch slice is implemented. Native arrays/operators, NVIDIA/AMD backends, fourteen other input families, three interoperability clients and framework/provider work remain incomplete and required. Resume the exact task in `docs/handoff.md`; see the actual-file audit in `docs/implementation-audit.md`.

This directory was copied without modifying captured files from ignored `artifacts/runs/stage-b-clean-76217b7`. Absolute paths in logs identify the original capture location. `SHA256SUMS.json` hashes all archived files except itself.
