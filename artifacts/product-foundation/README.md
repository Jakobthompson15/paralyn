# Product-foundation qualification — 2026-09-28

This immutable capture extends the original CUDA/native evidence with working arrays, public native MSL artifacts, structured CLI execution and relocatable packages. It is **not completion of the all-frontends platform** and is not a release tag.

Starting local HEAD and remote main were `f3c6c9955daf5a81d764fd0e634eb786a8489bcd`. All captures here used clean implementation revision `87978b1f99dab214565c232a406635f4222a1a65`. An evidence/documentation-only commit archives these bytes afterward. Original `gate-a/`, `paralyn-gate-a/`, `gate-b/` and `stage-b/` archives are unchanged.

Hardware: physical Apple M5; macOS 26.5.1 (25F80), SDK 26.2, LLVM/Clang 21.1.8 and Apple Clang 17. Dependencies were already installed. This measures neither a fresh operating-system setup nor dependency download time.

## Recorded results

| Capture | Result | Evidence |
|---|---|---|
| Fresh configured build and regression suite | **23/23 CTests passed**, no skips | [build-validation.json](build-validation.json), [ctest.log](ctest.log) |
| Product qualification | **53 GPU events**: C++ arrays 21, Python arrays 20, public MSL 10, CLI doctor 1, standalone doctor 1 | [qualification](product/qualification.json), [audit](product-audit.log) |
| Complete native applications through build CLI | **4 GPU events**; C++ and Python each compare add/affine with their own independent CPU reference | [verification](native-cli/verification.json) |
| Preserved ABI1 native qualification | **21 GPU events** | [qualification](native/qualification.json), [transcript](native/verification.txt) |
| CUDA Gate A | 1,024 results independently compared; completed GPU command | [execution](gate-a/execution.json), [verification](gate-a/verification.txt) |
| CUDA Gate B | **75 launches** including nine in the intentional exit 37 case | [summary](cuda/summary.json) |
| Full benchmark | **880 launches**, all four sizes, ten warmups and 100 measured iterations per variant/size; every output compared | [capture/audit](benchmark/capture.json), [all samples](benchmark/samples.csv) |
| Offline wheel, Python 3.14.5 | **2 GPU events**, isolated installation outside checkout | [installation](installations/python314/installation.json), [GPU evidence](installations/python314/gpu-evidence/execution.json) |
| Offline wheel, Python 3.9.6 | **2 GPU events**, isolated installation outside checkout | [installation](installations/python39/installation.json), [GPU evidence](installations/python39/gpu-evidence/execution.json) |
| Relocated installed CLI | **4 GPU events**, copied standalone C++/Python applications outside checkout | [verification](installations/relocated-cli/verification.json) |
| Relocated CLI doctor | **1 GPU event**, independent bundled reference | [report](installations/relocated-doctor/report.json) |
| Runtime-only build without LLVM | **1 GPU event**, independent doctor reference | [report](runtime-only-doctor/report.json), [linkage](runtime-only-linkage.log) |
| No-backend build | **5/5 CPU tests**, zero devices, explicit doctor failure | [tests](no-backend-ctest.log), [devices](no-backend-devices.log), [doctor](no-backend-doctor.log) |
| Deterministic wheel | Two builds produced identical bytes; offline pip used no index or dependencies | [hashes](package/reproducibility.json), [installation steps](installation-validation.json) |

Counts are separate runs; the CTest executions are repetitions and must not be added again when describing these captures. Product 53 contains 50 source-linked command records plus three separately recorded lifetime events after original owners close/die. The latter are two C++ and one Python events with actual completion/timing and CPU checks. Public MSL includes four additions/read-alias cases, three workgroup reductions and three tiled transposes, with 13 structured rejection checks. The CLI suite records 60 command cases including its doctor GPU probe.

GPU durations and timestamp validity come from observed Metal completion. Host event observation clocks are labeled separately. The MSL backend deliberately prepends exactly `#pragma STDC FP_CONTRACT OFF\n` to packaged source; the auditor verifies this exact policy transformation, not a loosely normalized source comparison. Corruption tests reject changed/removed/extra shader bytes and mismatched binaries. [Audit tests](audit-corruption.log) passed.

The [independent readback audit](independent-audit.json) reconciles 1,040 source-linked commands plus four retained-lifetime events across these separately named captures, excluding CTest repetitions. All 880 benchmark samples and 1,028 saved doctor outputs were independently checked.

A later baseline comparison identified a measurement-boundary regression: the benchmark quieted stdout, but new runtime progress uses stderr, so this capture includes per-launch progress I/O in host total latency. Numerical/completion evidence and raw samples remain valid; this capture must not establish a host-performance regression or speedup claim. A follow-up isolates both progress streams before repeating the full unchanged size/warmup/measurement protocol.

## Scope and provenance

Arrays are owned contiguous one-dimensional FP32 values with copied upload/readback, add and affine operations through the existing native runtime. Empty arrays, aliases, varying lengths, retained ownership and failures are exercised. Multidimensional tensors, matmul, arbitrary function compilation and automatic offload are not implemented by this capture.

Public MSL uses the declared PARALYNX1 resource/signature profile in [the contract](../../docs/executable-artifacts.md), actual Metal compilation/reflection, and the same native ownership/execution boundary. Writable aliasing across separate native MSL pointer parameters is explicitly rejected; existing scalar-IR/CUDA same-type alias behavior is preserved. No other shader frontend or portable backend is implied.

The CLI separates application streams from runtime events and JSON. Generic `run` reports `verification: not_requested` even if an application prints PASS. These named application fixtures are qualified because their retained source implements the independent CPU comparison and the harness checks real runtime evidence. General declared project/reference verification and kernel case execution remain required.

A local wheel is retained in `package/`; no package registry or release tag was published. The wheel contains the exact hashed native library/operator module and license notices. Its supported package profile is macOS 26+arm64, Python>=3.9; only Python 3.9.6/3.14.5 were executed here. The relocated compiler-enabled CLI still needs its selected LLVM installation for source compilation. The runtime-only CLI and installed wheel do not link LLVM. Optional precompiled modules are deployed separately with compiler-disabled CMake installs.

The relocated and runtime-only doctor inner execution records contain `paralyn_commit: unknown` and null dirty metadata: direct doctor does not populate revision fields without the capture environment. Their actual build/source identity is recorded externally in [installation steps](installation-validation.json) and [hashed executable/build snapshots](installation-builds.json). The limitation is preserved, not rewritten into the original records. Making direct doctor reports self-contained is a terminal follow-up.

All original capture trees were copied byte-for-byte. Embedded absolute paths identify where a run originally occurred; they were not rewritten to suggest another execution. `product/snapshot/` retains relevant implementation sources, four runtime/test binaries and build metadata with hashes. Other records identify sources, modules, dispatched shaders, references, completion, exact compiler/source revision and every benchmark sample. [SHA256SUMS.json](SHA256SUMS.json) covers all files in this archive except itself.

No CPU kernel fallback, fabricated vendor identity, mock execution or speedup claim is used. Actual driver-loss/timeout/exhaustion injection, Windows/Linux hardware execution, NVIDIA/AMD, 13 remaining input frontends, all three interoperability clients and full framework/operator/application/terminal/release scope remain unqualified. The [handoff](../../docs/handoff.md) preserves exact next work and the complete mandate.

## Read-only reproduction audit

```sh
python3 scripts/qualify_product.py --output artifacts/product-foundation/product --audit-only
python3 tests/product_audit_tests.py --capture artifacts/product-foundation/product
```

These commands audit saved evidence without dispatching GPU work. For fresh physical execution use the commands in [handoff](../../docs/handoff.md), a clean checkout and new capture directories. Run performance measurements serially without other intentional GPU qualification workloads.
