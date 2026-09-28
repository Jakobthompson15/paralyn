# CUDA Driver/NVRTC backend and native Windows port (lane: runtime/backends)

Written 2026-09-28 on branch `lane/cuda-backend-windows` (base `4f8fb22`). This is handoff task 1.

**Status: implemented but unqualified.** No NVIDIA GPU, NVIDIA driver, CUDA Toolkit, AMD GPU or
Windows machine was available, and remote access was not authorized. No CUDA kernel has run. The
Windows code has not been compiled. On machines without the libraries, the CUDA backend reports
**unavailable** and never passes a GPU gate. Everything below separates what was tested from what
still needs hardware.

## What exists

| Piece | Files | Tested here |
|---|---|---|
| CUDA C++ codegen from verified scalar IR | `compiler/codegen/cuda.cpp`, `include/paralyn/cuda_codegen.hpp` | Output compared with reviewed goldens (`tests/cuda/golden/*.cu`), for hand-built IR and for the real frontend's `native-kernels.prk`/`operators.prk`. Structure and rejection tests too. Not compiled by NVRTC. |
| Dynamic loader (`dlopen`/`LoadLibraryExW`) | `backends/cuda/api.hpp`, `backends/cuda/loader.cpp` | Missing libraries, a library without the symbols, and forced absence were tested. A real `libcuda`/NVRTC was never loaded. |
| Engine behind `Context`/`Buffer`/`Event` | `backends/cuda/engine.cpp`, `backends/cuda/engine.hpp` | Host logic was tested only against a labelled in-process **test double** of the driver/NVRTC tables (`tests/cuda/fake_driver.*`). The test double never executes kernels. It is not GPU evidence. |
| Selector registry (`cuda:N`) | `runtime/registry.cpp`, `include/paralyn/detail/cuda_backend.hpp` | Parsing, routing, and the HIP rejection. Metal selectors are unchanged (the existing `native_queries`, `gate_a` and `gate_b` tests still pass). |
| Native ABI 1, additive | `pr_backend_status_get` + `pr_backend_status_v1` in `include/paralyn/native.h` | Called through ctypes in `tests/cuda/cli_unavailable.py`. |
| CLI reporting | `cli/main.cpp` (`devices`, `doctor`, `support`, `run`) | Tested in `tests/cuda/cli_unavailable.py` with the libraries forced absent. |
| Qualification harness | `scripts/qualify_cuda.py` | The unavailable path exits 2 and prints no pass. `--harness-self-test-metal` ran on the M5 (details below). The CUDA path itself has not run. |
| Windows port | `cli/platform.hpp`, `cli/process_windows.cpp`, `cli/main.cpp`, `CMakeLists.txt`, Python `_NATIVE_NAME` | **Unbuilt and untested.** Correct by inspection only. |
| Windows inventory | `scripts/windows-inventory.ps1` (schema v2) | **Not run** (no Windows or PowerShell here). |

HIP/HIPRTC is **not implemented and has no stub**. `hip:N` and `rocm:N` report "not implemented".
`pr_backend_status_get("hip")` returns `implemented = 0`.

## Backend design

**Loading.** The build has no NVIDIA headers or libraries. `api.hpp` holds independently written
declarations of the documented Driver API/NVRTC subset. Library candidates are tried in this order:

- Linux: `libcuda.so.1`, `libcuda.so`; NVRTC: `libnvrtc.so.13`, `.so.12`, `.so.11.2`, `libnvrtc.so`, then `$CUDA_HOME/lib64|lib/libnvrtc.so`.
- Windows: `nvcuda.dll` (System32 only); NVRTC: `%CUDA_PATH%\bin\x64\` and `%CUDA_PATH%\bin\` with `nvrtc64_130_0.dll`, `nvrtc64_120_0.dll`, `nvrtc64_112_0.dll`, then the bare names through the standard search order.
- macOS: `libcuda.dylib`. The attempt is recorded, although NVIDIA has shipped no macOS driver since CUDA 10.2.

`PARALYN_CUDA_DRIVER_LIBRARY` and `PARALYN_NVRTC_LIBRARY` set exact paths, and an explicit path
never falls back to a default. Versioned symbols are bound by their `_v2` names. Optional symbols
are `cuDeviceGetUuid(_v2)` and `nvrtcGetSupportedArchs`. `cuEventElapsedTime_v2` is preferred over
`cuEventElapsedTime`. The loader runs once per process. The backend is **available** only if both
libraries and all required symbols load, `cuInit` succeeds, and at least one device exists.
Otherwise the exact reason and every failed candidate are reported.

**Contexts.** Each Paralyn context retains the device's **primary context**, which is the context
the CUDA runtime API uses, so a caller's `cudart` allocations interoperate with it. Every driver
operation runs inside `cuCtxPushCurrent`/`cuCtxPopCurrent`. On success, the pop is checked, and the
popped context must be the primary context. So the caller's current context, or no context, is
restored after each call, including on error paths. Buffers keep the retained primary context alive
past their `Context`. Release happens exactly once. The test double checks that every driver call
runs with the primary context current and that the caller's stack is restored.

**Memory and copies.** Memory comes from `cuMemAlloc` (zero bytes uses no allocation). Views pass
the base plus a byte offset. Copies are synchronous `cuMemcpyHtoD`/`cuMemcpyDtoH`, run after
draining this context's queue, matching Metal's ordering. Same-allocation aliases need no special
lowering on CUDA. Mixed pointee types on one allocation are rejected, matching Metal's portable
contract.

**Compilation.** `emit_cuda` produces `extern "C" __global__ void uc_kernel_<name>(...)`. It
renames all identifiers to `uc_arg_N` and `uc_local_N`. It encodes FP32 literals bit-exactly as
`__uint_as_float(bits)` and writes explicit `static_cast` conversions. FP32 add and multiply become
`__fadd_rn` and `__fmul_rn`, which CUDA documents are never contracted into FMA. NVRTC receives
`--gpu-architecture=compute_XY --fmad=false --ftz=false --prec-div=true --prec-sqrt=true`. Fast
math is never used. `compute_XY` is the device's compute capability if NVRTC supports it.
Otherwise it is the highest supported virtual architecture below the device, and it is recorded.
PTX is loaded with `cuModuleLoadDataEx` with JIT error/info logs attached. The module cache is
keyed by generated source and scoped to one context. There is no persistent cache.

**Launch, completion and timing.** Each context has one ordered stream. Every launch records a
start event and an end event. Waiting uses `cuEventSynchronize` on earlier events in queue order.
Duration comes from `cuEventElapsedTime`. **The clock domain is duration-only** (`clock_domain = 1`,
`timestamps_valid = false`, absolute start/end reported as 0 or `null`). Absolute timestamps are
never invented. Legacy `pr_event_wait` therefore returns zero start/end on CUDA. Use
`pr_event_timing` instead. Geometry is validated before any launch: positive dimensions, device
block and grid limits, the function's maximum threads, and 32-bit logical dimensions.

**Error mapping** (`map_result`):

| CUresult | Error code |
|---|---|
| `INVALID_VALUE`, `LAUNCH_OUT_OF_RESOURCES` | `invalid_value` |
| `OUT_OF_MEMORY` | `out_of_memory` |
| Driver/device absence and mismatch (3, 4, 34, 35, 100, 101, 802–804) | `invalid_device` |
| Image, PTX, JIT and symbol errors (200, 209, 218, 221, 222, 300, 500) | `compilation` (with JIT log) |
| `INVALID_HANDLE`, `INVALID_CONTEXT` | `internal` |
| `NOT_PERMITTED`, `NOT_SUPPORTED` | `unsupported` |
| Other codes | `execution` |

Sticky faults (214, 700, 702, 714–719, 999) and any completion failure make the context
permanently failed, matching Metal's terminal-failure contract. `UNSUPPORTED_PTX_VERSION` explains
the mismatch between NVRTC and the driver.

**Selectors and enumeration.** Parsing is strict: `cuda:N` takes a decimal ordinal up to
2^31−1. `metal:N`, legacy numeric indices, and all Metal messages are unchanged. `auto` picks the
first Metal device. Only when no Metal device exists does it pick `cuda:0`, and only if CUDA is
available. The native ABI lists Metal devices first, then available CUDA devices. For CUDA,
`pr_device_info.registry_id` holds the CUDA ordinal. `max_buffer_bytes` is total device memory, an
upper bound rather than free memory. `stable_id` is `cuda:uuid:…`, or `cuda:pci:…` when no UUID is
available. The capability record advertises only the verified-IR format for CUDA, because MSL
modules are Metal-specific and are rejected with `unsupported`.

**Evidence.** `write_evidence` writes `execution.json` with backend, device, ordinal, stable id,
compute capability, NVRTC architecture and options, driver and NVRTC versions, library paths,
`test_double`, `cpu_fallback: false`, and per-launch duration fields with `gpu_start_seconds` and
`gpu_end_seconds` set to `null`. It also writes `source-N.cu`, `source-N.ptx` and any NVRTC/JIT
`source-N.log`.

**Scope limits.**
- CUDA-source programs (`paralyn run x.cu`) still execute through the Metal compatibility runtime
  (`backends/metal/runtime.mm`). `run --device cuda:N` on a `.cu` program is rejected with
  `P-BACKEND-PROFILE`.
- The CUDA backend serves native C/C++/Python modules: `pr_*`, `paralyn::native`, `paralyn`
  Python, and FP32 arrays.
- Kernels are limited to the portable 31-parameter limit.
- There is no async copy, no multiple streams, no cross-context events, and no cancellation.
- NVRTC-generated PTX is **not** the mandate's PTX input track. That track (and SASS) needs
  genuine non-NVIDIA lowering and remains separate.

## Tests added (CPU-only; none executes GPU work)

- `cuda_codegen`: goldens and structure for 4 hand-built IR kernels (i32 `vector_add`/`affine`,
  u32 `array_add`/`array_affine`), literal encoding, keyword renaming, rejection of invalid IR, and
  the numerical options.
- `cuda_codegen_frontend`: the kernels in `build/native-kernels.prk` and `build/operators.prk`,
  produced by the real Clang frontend, generate byte-identical CUDA to the goldens (4 kernels).
- `cuda_backend_host`: 178 checks. The loader was tested with missing libraries, with Paralyn's own
  library (which lacks `cuInit` and `nvrtcVersion`), and with forced production absence. Also:
  selector parsing, the full error map and sticky set, and architecture selection. The engine
  was run against the test double. Covered: caller-context restoration after every operation and
  failure, NVRTC source and options, argument marshalling (base+offset, copied scalars), and
  duration-only timing. The test double must leave the output sentinel untouched, proving nothing
  computes on the CPU. Also covered: geometry, alias and MSL rejection before the driver;
  OOM, launch-resource, invalid-PTX, NVRTC and push failures; a sticky illegal address that stays
  failed; evidence marked `test_double: true`; and exact release of memory, modules, streams,
  events and primary-context retains.
- `cuda_unavailable_cli`: with the libraries forced absent, `support`, `devices`, and
  `doctor --device cuda:0` report the backend honestly. `doctor` fails with `P-BACKEND-UNAVAILABLE`
  and creates no evidence directory. `hip:0` gives `P-BACKEND-UNIMPLEMENTED`. The native ABI
  returns `PR_DEVICE_UNAVAILABLE` for `cuda:0`, `cuda:x` and `hip:0`, and rejects unknown query
  versions.

Full `ctest -j1` on the Apple M5: **27/27 passed** (23 existing plus 4 new). Metal Gates A/B,
native, array, MSL and terminal tests are unchanged.

`scripts/qualify_cuda.py --harness-self-test-metal` was run twice on the M5, once with
`operators.prk` and once with `native-kernels.prk`. Each run made 28 add/affine launches with
4,267,882 values independently compared, plus one doctor probe. These runs test the harness
itself. They are Metal events, **not** CUDA qualification, and their summaries are marked
`cuda_qualification: false`.

## Exact qualification steps (for a person with an NVIDIA machine)

Prerequisites: an x86-64 or aarch64 Linux or Windows machine with an NVIDIA GPU, an NVIDIA driver,
and a CUDA Toolkit that provides NVRTC (12.x or 13.x). The driver must be at least as new as NVRTC,
or loading fails with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. Also needed: CMake ≥ 3.24, Ninja,
Python ≥ 3.9, and optionally LLVM/Clang 21.1.8 for the CUDA frontend. No remote or automatic
execution is authorized by this document. A person runs these steps locally.

1. **Inventory.** On Windows, run
   `powershell -ExecutionPolicy Bypass -File scripts\windows-inventory.ps1 -OutputPath inv.json`
   and review `nvidia.gpus`, `cuda.driver_library` and `cuda.nvrtc_libraries`. On Linux, record
   `nvidia-smi --query-gpu=name,uuid,memory.total,driver_version,compute_cap --format=csv` and
   `uname -a`.
2. **Build** a fresh tree from a clean commit.
   On Linux, run `cmake -S . -B build -G Ninja -DPARALYN_BUILD_COMPILER=ON -DLLVM_DIR=… -DClang_DIR=…`,
   then `cmake --build build`.
   Without LLVM, use `-DPARALYN_BUILD_COMPILER=OFF` and copy `native-kernels.prk` and
   `operators.prk` from a compiler build. The `.prk` v1 format is platform-independent.
   On Windows, use a Developer Prompt with `cl` or `clang-cl`. **The Windows build has never been
   compiled.** Record and fix build errors as the first Windows result.
3. **CPU tests.** Run `ctest --test-dir build --output-on-failure -j1`. The CUDA tests force absence
   and should still pass. Metal-labelled tests are not built off macOS.
4. **Availability.** Run `build/paralyn devices --json`. Expected: a `cuda:0` device with backend
   `CUDA`, a `cuda:uuid:` stable id, and `backends.cuda.available: true` with driver and NVRTC
   versions. If it reports unavailable, record the reason. That result means unavailable, not
   failed.
5. **Doctor probe.** Run `build/paralyn doctor --device cuda:0 --json --artifacts runs/cuda-doctor`.
   This runs one verified-IR vector add through NVRTC and the driver, and compares 257 FP32 values
   exactly on the host. Check `runs/cuda-doctor/execution.json`: `backend: CUDA`,
   `test_double: false`, `cpu_fallback: false`, `gpu_duration_valid: true`,
   `gpu_timestamps_valid: false`, and the `source-0.cu` and `source-0.ptx` files.
6. **Compile-only check.** If the frontend is built, run
   `build/paralyn check examples/native/kernels.cu --device cuda:0 --json`. Expect
   `backend_compilation: passed`.
7. **Qualification harness.** Run
   `python3 scripts/qualify_cuda.py --paralyn build/paralyn --library build/libparalyn_native.so --module build/operators.prk --output runs/cuda-q-ops`.
   Repeat with `--module build/native-kernels.prk` and a new output directory. On Windows, use
   `paralyn_native.dll`. Each run makes 28 launches: sizes 1, 127, 128, 129, 1000, 65537 and
   1000003, at element offsets 0 and 3, for add and affine. Every value, including guard
   sentinels, is compared bit-exactly with an independent FP32 reference. Only a complete match
   prints `Verification: PASS`. Exit code 2 means unavailable.
8. **Negative checks.** Run with `PARALYN_NVRTC_LIBRARY=/nonexistent` and confirm that
   `doctor --device cuda:0` reports `P-BACKEND-UNAVAILABLE`. Run `--device cuda:99` and confirm
   an out-of-range error. Load `examples/metal/kernels.metal` through the native API on `cuda:0`
   and confirm `PR_UNSUPPORTED`.
9. **Not yet automated, still required.**
   - Primary-context interoperability with a caller's own CUDA runtime or driver context. This
     needs a small `cudart` host program that sets its own context, calls Paralyn, and checks
     `cuCtxGetCurrent` afterward.
   - Real device-fault injection.
   - Multi-GPU ordinals.
   - The existing native examples and tests (`examples/native/*`, `tests/native/*`) require
     absolute Metal timestamps and `backend == "Metal"`, so they intentionally fail on CUDA.
     Duration-based CUDA variants are follow-up work.
10. **Archive** the build log, inventory, `devices.json`, doctor and harness directories, the exact
    driver, NVRTC, GPU and OS versions, and the clean revision in a new `artifacts/` directory.
    Then update status and ledger. Qualification applies only to the exact hardware and versions
    recorded. Performance was not measured, and the Gate B benchmark protocol has not been applied
    to CUDA.

## Native Windows port (unbuilt, untested)

- **Toolchain.**
  - With MSVC or clang-cl, CMake adds `/utf-8 /fp:precise`, `NOMINMAX` and
    `_CRT_SECURE_NO_WARNINGS`.
  - `cli/platform.hpp` builds host-program command lines for two compiler families. MSVC-style
    (`cl` or `clang-cl`, detected by `CMAKE_CXX_COMPILER_FRONTEND_VARIANT`) gets
    `/std:c++17|c11 /Od /Zi /fp:precise /EHsc /utf-8 /I … /Fo<dir>\ /Fe<exe> /link <import libs>`.
  - GNU-style keeps the existing macOS command order unchanged (`-fno-fast-math -ffp-contract=off`,
    rpath only on ELF and Mach-O).
- **Import libraries.** Native host programs link `$<TARGET_LINKER_FILE:paralyn_native>`
  (`paralyn_native.lib`), passed to the CLI as `PARALYN_NATIVE_LINK_LIBRARY`. Static archives are
  `paralyn_runtime.lib` and `paralyn_ir.lib`. Installed lookup checks `lib\`, then `bin\`, where
  DLLs are installed. Output is `program.exe`.
- **Library discovery.** Windows has no rpath, so the child `PATH` gets the DLL directory prepended.
  `PARALYN_LIBRARY` points at `paralyn_native.dll`. The Python bindings pick `paralyn_native.dll`,
  `libparalyn_native.so` or `libparalyn_native.dylib` by platform. The default Python on Windows
  is `python`, and `PARALYN_PYTHON` overrides it.
- **PYTHONPATH.** Uses `;` on Windows and `:` elsewhere.
- **`cli/process_windows.cpp`.**
  - A `ChildGuard` ensures that unwinding leaves no running child. It terminates the child's job
    object (or the process as a fallback), joins both reader threads, and unregisters the console
    handler.
  - Reader threads cannot leak exceptions.
  - The child inherits only its three standard handles (`PROC_THREAD_ATTRIBUTE_HANDLE_LIST`), with
    duplicated stdin or `NUL`.
  - The environment block is case-insensitive (so `Path` and `PATH` merge) and sorted as Windows
    requires.
  - The 32767-character command-line limit is checked.
  - The child starts suspended until it is in the job.
- **Not done on Windows.**
  - No compile has been attempted.
  - `tests/process_tests.cpp` is POSIX-only; a Windows process test is still needed.
  - The Metal-only native tests are not registered on Windows.
  - Wheel packaging is macOS-only.
  - Ctrl-C forwarding is unverified.
  - The CUDA frontend (`PARALYN_BUILD_COMPILER`) on Windows needs LLVM 21 and a path for its
    `find_package` hints.

## Windows inventory (`scripts/windows-inventory.ps1`, schema v2)

The script is read-only and never overwrites its output. It records:

- **OS:** caption, version, build, UBR and display version.
- **Machine:** model and RAM.
- **Each display adapter:** name, vendor, PnP ID, driver version and date, and **64-bit VRAM** from
  the WDDM registry value `HardwareInformation.qwMemorySize`, alongside the truncating `AdapterRAM`.
- **NVIDIA (`nvidia-smi`):** per GPU, name, UUID, VRAM in MiB, driver, PCI bus and compute
  capability when the driver supports it, plus the driver's maximum CUDA version.
- **CUDA:** `nvcuda.dll` version, `CUDA_PATH`, NVRTC DLLs, `nvcc --version`, and whether Paralyn's
  loader prerequisites are present.
- **HIP:** `HIP_PATH`, `amdhip64*.dll`, HIPRTC DLLs, `hipcc` and `hipInfo`, with
  `paralyn_backend: not_implemented`.
- **Optionally:** `paralyn devices --json`.

Native tool stderr no longer aborts under Windows PowerShell 5.1. Tool failures are recorded as data.

## Next tasks

1. On NVIDIA hardware, run the qualification steps above and archive the evidence.
2. Compile the Windows port, add a Windows process test, and fix what the compiler finds.
3. Add duration-based CUDA variants of the native C, C++ and Python examples and tests. Expose
   `pr_event_timing` and `pr_backend_status_get` in the Python bindings.
4. Write the primary-context interoperability test with a `cudart` caller.
5. Route the CUDA-source compatibility runtime (`paralyn run x.cu --device cuda:N`) through this
   backend. That needs a host-token adapter like `backends/metal/runtime.mm`.
6. Build HIP/HIPRTC as a separate backend, reusing the loader pattern and not the CUDA table.
