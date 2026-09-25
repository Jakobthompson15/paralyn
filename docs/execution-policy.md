# Execution policy

Current policy at mandate adoption, 2026-09-25. Future design targets are identified separately; this document does not claim that the all-frontends portfolio or Gate B is complete.

## Current CUDA/Metal path

The only implemented execution backend is local Metal. `paralyn run` parses the supported complete CUDA source, verifies typed IR, compiles rewritten native host C++, and submits generated MSL through a real Metal device. `auto` chooses the first enumerated device in stable registry-ID order; a numeric index selects explicitly. There is no resource-cost scheduler, external-provider choice, remote enrollment, or cloud allocation.

Device reporting distinguishes unified-memory status, the device's maximum buffer length, and its recommended working-set size. A recommendation is not measured free memory. Allocation still may fail at runtime. No aggregate cluster-memory figure is presented as one address space.

Launches validate argument kinds/types, live allocation-base tokens, supported same-type alias groups, Metal argument limits, positive geometry, and device/pipeline limits. These checks do not prove every dynamic kernel index safe. The user's bounds checks remain in generated code. Unsupported capabilities fail; the runtime does not shrink the block, silently reduce precision, or choose another backend.

One ordered queue serves the context. Scalar values are copied at submission; pending commands retain allocations. Blocking host/device copies, frees, synchronization, and normal shutdown inspect pending failures. Completion errors retain backend details and produce failure when unresolved. Abnormal termination cannot promise completion or artifact flushing. Host thread/error aggregation beyond the documented subset is not a qualified multi-threaded CUDA runtime.

## No CPU kernel fallback

A missing GPU, failed shader compilation, invalid launch, or failed command is not permission to run a CPU kernel. Ordinary host C++ and independent CPU-reference comparisons execute on the host by design and are reported separately. A future CPU backend would require explicit selection and visible reporting; it cannot satisfy GPU qualification.

The same rule applies to future framework/operator providers. Framework-level partitioning or hidden host execution cannot be reported as an entirely GPU-executed workload. A graph provider that may select CPU work requires explicit policy consent and actual placement evidence. Native MSL execution, native NVIDIA binary loading, and portable translation are distinct results.

## Compilation and artifacts

The current pipeline cache is in-process and scoped to one device/context, keyed by entrypoint plus emitted source with fixed numerical options and alias layout. Including the entrypoint prevents reusing a different kernel from the same source module. Persistent caching is not implemented. See [numerics](numerics.md) for what safe/precise compilation currently demonstrates.

`inspect` reuses parsing/verification but does not run the host program. Launch displays are source expressions, not invented runtime dimensions. Current working commands are `devices`, `inspect`, and `run`; `doctor`, `devices --json`, `--explain`, and backend-prefixed device selectors in the mandate remain design targets until code and tests establish them. The legacy `build/unicuda` executable and `unicuda/*` include/namespace aliases have been restored; development checks verified the version command and a C++ include/namespace compile-link path. They resolve to the same implementation and types, not a separate legacy runtime. Final baseline qualification remains recorded separately.

New runs use unique artifact directories, with explicit `--artifacts DIR` permitted only for an empty directory. Preserve original source, typed IR, generated backend source, actual execution metadata, and the independent-verification transcript. New captures now include transformed `host.cpp`, as required by mandate v1.1, and each recorded launch identifies its `source-N.metal` artifact; `generated.metal` remains the latest shader for compatibility. Preserve old five-file directories `artifacts/gate-a/` and `artifacts/paralyn-gate-a/` unchanged; do not retroactively add generated files or rewrite their provenance.

Evidence must report the actual source revision and whether it is dirty; unknown Git state cannot be reported as clean. Device identity, successful command status, and positive GPU timing evidence accompany verified output. Timestamps are execution evidence, not a benchmark. Only the actual numerical comparison may print `Verification: PASS`.

## Trust, extension, and publication boundaries

Inputs are trusted local native programs. Running host C++ has the caller's process/file/network access; it is not a sandbox. Driver/toolchain access does not make untrusted plugins or GPU submissions safe. Future remote execution and multi-tenant hardware CI require separate threat models, credential isolation, ownership, and authorization.

Future frontend adapters import a language/artifact; framework adapters integrate a framework boundary; operator providers implement operations; backends own device submission. They share explicit memory/argument/completion/capability and numerical contracts. Do not claim support because a dependency loads, a registry has an entry, or an IR is printable.

The all-frontends mandate authorizes phased implementation, not side-effect publication, paid resources, new machine enrollment, or remote access. Repository/package/release publication must have separate applicable authorization and the required qualification. Historical publication does not waive gates for future releases.
