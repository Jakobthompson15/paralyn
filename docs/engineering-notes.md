# Engineering evidence and design revisions

Recorded 2026-09-25 during Gate A implementation. These notes capture integration/review evidence, not a claim that Gate A has passed. Acceptance belongs in `status.md` and retained execution artifacts.

## LLVM package checks require enabling C in CMake

**Problem.** A C++/Objective-C++-only project configuration was insufficient for the selected LLVM package's dependency checks.

**Root cause.** Imported LLVM CMake modules perform compiler checks in addition to configuring C++ targets. Enabling only the languages used by UniCUDA source files did not satisfy the package's C-language checks.

**Evidence.** The local configuration failure was resolved by `project(UniCUDA ... LANGUAGES C CXX OBJCXX)`. The selected LLVM installation contains C compilation checks in its CMake modules (including `FindFFI.cmake` and `HandleLLVMOptions.cmake`). The resulting CMake cache has a configured C compiler, and LLVM 21.1.8 frontend integration tests pass. This does not require adding C implementation files.

**Potential solutions.** Enable C normally; patch or bypass LLVM's package checks; or maintain a custom dependency-discovery layer.

**Selected solution.** Enable C alongside C++ and Objective-C++. Keep upstream dependency checks intact. This is a build integration requirement, not a change to the C++17 core design.

## Toolchain deployment minimum exceeds the Metal API minimum

**Problem.** macOS 15 availability of safe/precise Metal APIs was incorrectly sufficient as an initial whole-program deployment assumption.

**Root cause.** The CLI links the installed Homebrew `libclang-cpp.dylib`, whose binary deployment requirement is newer than the Metal calls used by the runtime.

**Evidence.** `otool -l /opt/homebrew/opt/llvm@21/lib/libclang-cpp.dylib` reports `LC_BUILD_VERSION`, `minos 26.0`, and SDK `26.2`. The configured build cache records deployment target `26.0`; the host OS is macOS 26.5.1 build 25F80. CMake now defaults to 26.0 and passes the configured value to CLI-generated host compilation. The runtime's `@available(macOS 15.0, *)` check addresses API availability only.

**Potential solutions.** Use the installed bottle and target macOS 26; build/select matching LLVM libraries with an older deployment minimum and test them; or separate the compiler delivery from a separately qualified runtime package.

**Selected solution.** Use the verified local LLVM 21.1.8 installation and macOS 26.0 deployment target for this milestone. Do not claim CLI support for macOS 15. A future older-OS configuration must verify every linked dependency and run its own compatibility tests; changing a compiler flag alone is insufficient.

## Opaque token identity and asynchronous resource lifetime

**Problem.** Returning a raw Metal mapping as a CUDA-visible pointer would expose unsupported host access and make ownership validation harder. Freeing/reusing token objects could also make a stale pointer accidentally identify a new live allocation. Asynchronous submission requires resources to outlive the caller's arguments.

**Root cause.** A CUDA-looking C++ pointer does not need to be a device virtual address for the restricted Gate A API. Pointer spelling alone provides neither ownership nor asynchronous lifetime.

**Evidence.** The reviewed runtime implements `Token` objects retained by the context, a separate map from live token addresses to `Allocation`, and pending commands holding `shared_ptr<Allocation>`. `cudaFree` synchronizes and removes the live entry while retaining the token object. `Argument` copies scalar bytes, and `setBytes` submits those values without depending on the caller's stack lifetime. API copies validate byte counts and reject token identities on their host side. The handwritten smoke test includes oversized-copy and repeated-free rejection checks; this review is not a complete alias or adversarial-memory qualification.

**Potential solutions.** Expose mapped host addresses; encode numeric tags as pointers; or allocate retained token objects and validate them against a registry.

**Selected solution.** Retained token objects with base-token-only semantics. No host pointer arithmetic, interior views, or GPU-address promise. The context and small token tombstones persist until process exit; token memory therefore grows with allocation count. This is a documented initial ownership tradeoff, not an unbounded service-ready allocator. Device allocations are released from the live map on successful free after synchronization.

## Normal-exit synchronization and error-detail lifetime

**Problem.** Normal shutdown must inspect pending GPU failures, including errors ignored by host code. Exit callbacks cannot safely access a thread-local `std::string` after its destructor has run.

**Root cause.** C++ thread-local destruction and `atexit` callbacks have lifetime ordering that can invalidate error-detail storage before final GPU checks.

**Evidence.** Source review found an installed `atexit` shutdown handler, a process-lifetime context, and a thread-local pointer to retained string storage, with an explicit comment explaining the teardown-order reason. The handler synchronizes pending work and changes process termination to failure when an unresolved error remains. Pending commands are all inspected even if an earlier command failed. This describes the implemented path; it does not establish complete multi-threaded error aggregation.

**Potential solutions.** Own a larger process-wide lifetime manager; explicitly wrap every possible host termination path; or keep the small context/error state alive through process termination.

**Selected solution.** Retain the initial context/error storage for normal-exit checks and keep explicit synchronization/shutdown entrypoints. This does not cover abnormal termination such as process kill, `_Exit`, or a crash. Multi-threaded error aggregation and comprehensive teardown qualification remain outside Gate A.

## Source preservation requires rejecting context-sensitive host constructs initially

**Problem.** Retaining host C++ text while adding generated declarations/lambdas can still change `__LINE__`, `__FILE__`, function-context expressions, and preprocessing conditioned on CUDA compilation mode.

**Root cause.** The CUDA AST pass and native transformed compilation are distinct compilations with different file/context/preprocessor conditions. Source-range rewriting alone does not preserve all observable metadata.

**Evidence.** The frontend scans user files, including inactive branches, for CUDA-conditioned preprocessing and source-location/context tokens. It rejects those cases with named, source-located unsupported-feature diagnostics. Frontend tests include architecture conditionals in a user header, `__CUDACC__`, and source-line expressions.

**Potential solutions.** Implement complete original-file line maps and context-preserving rewrites; silently accept different behavior; or reject affected constructs until they have preservation tests.

**Selected solution.** Explicit rejection in Gate A. Preserve ordinary supported host C++ using native compilation and narrow CUDA edits. Add original-source line mapping or restore individual context-sensitive constructs only with tests demonstrating their semantics. This does not expand UniCUDA into a whole-C++ transpiler.

## Defer abstractions that have no implemented consumer

**Problem.** Initial architecture language suggested a backend registry, buffer-offset views, and inspect-time dimension evaluation already existed.

**Root cause.** Future design seams were being described too similarly to actual Gate A functionality.

**Evidence.** The runtime has one concrete Metal context; `Argument` contains a base token with no offset; `LaunchInfo` contains source-expression strings; and `inspect` labels launch values as unevaluated expressions.

**Potential solutions.** Implement unused registry/offset/analysis machinery now, or clearly distinguish current code from future extension points.

**Selected solution.** Document the current interfaces exactly. Preserve portable types and backend-private native objects. Add a registry when another backend exists, offset views when pointer support is tested, and constant launch analysis when it serves a concrete need. None belongs on Gate A's critical path.
