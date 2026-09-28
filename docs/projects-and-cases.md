# Projects, kernel cases and build provenance

Lane: CLI/qualification (handoff task 4). Written 2026-09-28 on branch `lane/project-kernel-cases`, starting from `4f8fb22`. This file records what is implemented and tested. It does not qualify a release, and the software is still 0.0.1. `docs/status.md`, `docs/handoff.md` and the ledger were not edited on this branch. The integrator applies the proposed reconciliation at the end of this file.

## What exists

- **Kernel-only execution.** `paralyn run|verify|check MODULE [--manifest M.json] [--entry NAME] --case CASE.toml [--device metal:0] [--artifacts DIR]` works for:
  - a public MSL module given as `.metal` with its JSON manifest, or as a packaged `.prx`;
  - an existing verified-IR `.prk` v1 module, for example `build/native-kernels.prk` compiled from `examples/native/kernels.cu`.

  The GPU executes the kernel through the ordinary native runtime (C ABI 1). Host code only uploads, downloads and compares. There is no CPU fallback.
- **Strict case schema** (`paralyn.kernel-case`, version 1), parsed with vendored toml++ 3.4.0.
- **Strict project schema** (`paralyn.project`, version 1) in `paralyn.toml`. It declares modules, cases and single-file programs.
- **Verification contracts.** A contract names an independent reference and an explicit tolerance. `verify` requires a check for every declared output (`P-CASE-VERIFY-UNCOVERED` otherwise, before any GPU work) and prints `Verification: PASS` only after it has compared every element of every declared output.
- **Build revision and dirty state** are captured at build time, not only at configure time. They are embedded in the CLI and the runtime and reported by `doctor`, by `run`/`verify`/`check`/`inspect`/`explain`/`compile` reports for single-file and kernel-case targets, and by every `execution.json`.
- **Stretch goal: bounded streaming capture of application stdout/stderr, plus partial reports.**

Single-file programs still need no manifest. `paralyn run examples/vector_add.cu` and the native C/C++/Python programs behave as before. Gate A, Gate B and the product/native terminal tests pass unchanged; the counts are under "Test evidence" below.

## Case file (`CASE.toml`)

```toml
schema = "paralyn.kernel-case"
schema_version = 1

[case]
name = "vector-add-1003"
entry = "vector_add"          # optional if --entry is given; both must agree
description = "optional text"

[launch]
grid = [16, 1, 1]             # threadgroups (CUDA gridDim), three positive u32
block = [64, 1, 1]            # threads per group (CUDA blockDim)

[scalars.n]                   # one table per scalar parameter, keyed by parameter name
type = "u32"                  # f32 | i32 | u32, must equal the reflected parameter type
value = 1003

[buffers.a]                   # one table per buffer parameter, keyed by parameter name
dtype = "f32"
length = 1003                 # elements, 1 .. 67,108,864 (256 MiB of 32-bit data)
file = "data/vector_add_a.f32"          # raw little-endian 32-bit elements
sha256 = "daed10ac…"                    # required with file; checked before use

[buffers.out]
dtype = "f32"
length = 1003
fill = -65536.0               # explicit initial contents (canary/sentinel value)
output = true                 # read back and saved; at least one output is required

[[verify]]                    # optional for run; verify needs one per output buffer
buffer = "out"                # must be a declared output; exactly one check per buffer
reference = { file = "data/vector_add_expected.f32", sha256 = "3246…" }
tolerance = { kind = "exact" }
```

Rules. Each rule is enforced and has a negative test in `tests/kernel_cases.py`.

- **Unknown fields fail.** This applies at every level, including inline tables. Duplicate keys and malformed TOML also fail. Errors carry `file:line:column`.
- **Every buffer declares exactly one data source:** `values` (inline, exact length), `fill`, or `file` + `sha256`. Nothing is zero-filled or generated implicitly.
- **Numeric conversion is exact or rejected.**
  - f32 literals must be exactly representable. For example, `0.1` is rejected and must come from a data file.
  - Integers are range-checked for i32/u32. Floats are rejected where integers are required.
- **Arguments bind by parameter name to the module's reflected parameters.** A missing, extra, wrongly typed or duplicated argument fails. An output declared on a read-only parameter fails. The case schema has no aliasing. Each buffer parameter gets its own allocation, bound with the parameter's declared access.
- **The entry is never guessed.** `--entry` and `case.entry` must agree when both are given. Neither is taken from the module, even when it has only one entry.
- **Paths resolve relative to the case file.**
- **`sha256` values are lowercase hex.** A digest mismatch is `P-CASE-DATA-HASH`.

### References and tolerances

A reference is exactly one of:

- `{ file = "…", sha256 = "…" }`: the same raw format as the output.
- `{ values = [ … ] }`: inline values with the output's full length.
- `{ builtin = NAME, role = "argument", … }`: a registered host CPU reference computed from the declared case inputs. It is never computed from GPU results. Registered builtins:
  - `vector-add-f32`, with roles `lhs`, `rhs` (f32 buffers) and `count` (u32/i32 scalar).
  - `block-sum-f32`, with roles `input` and `count`, plus the constant `group_size`. It sums sequentially in double and then rounds to f32, which is not the GPU's tree order. Declare a tolerance that fits the data; the shipped case uses small integers, so `exact` is justified.
  - `transpose-f32`, with roles `input`, `width` and `height`.

  An unknown builtin is `P-REFERENCE-BUILTIN-UNKNOWN`. Output elements that a builtin does not define must keep their declared initial contents, so trailing sentinels are checked too.

  A builtin's shape contract is static, because the case declares every buffer length and scalar value. `count`, `width*height` or the number of `block-sum-f32` groups exceeding a declared buffer, or a negative i32 role, is rejected with `P-REFERENCE-SHAPE` when the case is loaded. This applies to `run`, `check` and `verify` alike, before any module load or GPU dispatch, and leaves no evidence directory.

The tolerance is required and has no default:

- `{ kind = "exact" }` compares identical bit patterns: `-0` ≠ `+0`, and a NaN must match bit for bit.
- `{ kind = "ulp", ulp = N }` uses ordered-integer distance with `+0 == -0`. A NaN matches only a NaN.
- `{ kind = "absolute_relative", absolute = A, relative = R }` passes when `|a-e| <= A + R*|e|`.

Integer outputs accept only `exact`. The report records per-check compared and mismatch counts, the first 16 mismatches, the maximum absolute error and the maximum ULP distance.

`run` executes and saves outputs but never verifies. Its report has `verification.status = "not_requested"` and `contract_declared` true or false; a contract that covers only some outputs is accepted by `run`. `verify` fails with `P-REFERENCE-REQUIRED` when the case has no `[[verify]]`, and with `P-CASE-VERIFY-UNCOVERED` (listing the buffers) when any `output = true` buffer has no check. Both are raised before any GPU work. A mismatch exits 1, prints `Verification: FAIL`, and records `P-VERIFY-MISMATCH` with `failure_origin = "verification"`. `check MODULE --case` compiles the module, reflects it and binds the case without submitting GPU work.

## Project file (`paralyn.toml`)

```toml
schema = "paralyn.project"
schema_version = 1

[project]
name = "metal-kernel-examples"
default = "vector-add"        # optional; must name a declared case or program

[modules.kernels]             # .metal (manifest required) | .prx | .prk (manifest rejected)
source = "../metal/kernels.metal"
manifest = "../metal/kernels.json"

[cases.vector-add]
module = "kernels"
file = "vector_add.toml"

[programs.cuda-vector-add]    # an ordinary single-file program
source = "../vector_add.cu"
arguments = []                # optional; conflicts with arguments after `--`
```

### Selecting a target

A project is used only in these cases:

- `--project FILE` is given.
- TARGET's file name is exactly `paralyn.toml`.
- There is no TARGET and `./paralyn.toml` exists.

Otherwise TARGET is the single file and nothing else changes.

Within a project, a target is selected as follows:

- `--case NAME` or `--program NAME` selects a target. Giving both is `P-PROJECT-SELECTION-AMBIGUOUS`.
- With no selector, `project.default` is used.
- With no selector and no default, the only declared target is used. With several targets it is `P-PROJECT-SELECTION-AMBIGUOUS`, and the message lists the targets.

Other ambiguous combinations fail with `P-TARGET-AMBIGUOUS`: TARGET together with `--project`, and `--manifest` in project mode. Flags that do not apply to the selected target are rejected rather than ignored: `--entry` without a kernel case (complete programs, project programs, or `check MODULE --entry` without `--case`) is `P-CASE-NOT-APPLICABLE`, and `--manifest` on anything but a `.metal` target (`.cu`, `.py`, `.c/.cpp`, `.prx`, `.prk`) is `P-MANIFEST-NOT-APPLICABLE`. `run`/`verify` on a kernel module with `--entry` but no `--case` stays `P-KERNEL-CASE-REQUIRED`. `verify` accepts cases only; selecting a program is `P-REFERENCE-REQUIRED`. Names must be unique across cases and programs. All paths resolve relative to the project file. Reports include `project {path, sha256, name, case|program, module}`.

## Stable error identifiers

| Id | Meaning |
|---|---|
| `P-KERNEL-CASE-REQUIRED` | `run`/`verify` on a kernel module without `--case`. The message gives the exact invocation. |
| `P-CASE-NOT-APPLICABLE` | `--case` on a complete program; `--entry` without a kernel case. |
| `P-MANIFEST-NOT-APPLICABLE` | `--manifest` on a target that is not `.metal`. |
| `P-CASE-FILE`, `P-CASE-SYNTAX`, `P-CASE-SCHEMA`, `P-CASE-VERSION` | Unreadable file, invalid TOML, wrong schema name, unsupported version. |
| `P-CASE-UNKNOWN-FIELD`, `P-CASE-MISSING-FIELD`, `P-CASE-TYPE`, `P-CASE-VALUE` | Strict field validation. |
| `P-CASE-DATA-SOURCE`, `P-CASE-DATA-SIZE`, `P-CASE-DATA-FILE`, `P-CASE-DATA-HASH` | Declared data problems. |
| `P-CASE-OUTPUT-REQUIRED`, `P-CASE-DUPLICATE-ARGUMENT`, `P-CASE-VERIFY-BUFFER`, `P-CASE-VERIFY-ROLE` | Case structure. |
| `P-CASE-VERIFY-UNCOVERED` | `verify` on a case whose output buffers do not all have a `[[verify]]` check. |
| `P-CASE-ENTRY-REQUIRED`, `P-CASE-ENTRY-MISMATCH`, `P-CASE-ENTRY-UNKNOWN` | Entry selection. |
| `P-CASE-ARGUMENT-MISSING`, `P-CASE-ARGUMENT-UNKNOWN`, `P-CASE-ARGUMENT-TYPE`, `P-CASE-OUTPUT-ACCESS`, `P-CASE-ARGUMENTS` | Binding against reflected parameters; program arguments given to a kernel case. |
| `P-REFERENCE-REQUIRED`, `P-REFERENCE-BUILTIN-UNKNOWN`, `P-REFERENCE-SHAPE`, `P-VERIFY-MISMATCH` | Verification. |
| `P-PROJECT-*` (`SYNTAX`, `UNKNOWN-FIELD`, `MISSING-FIELD`, `TYPE`, `VALUE`, `VERSION`, `SCHEMA`, `REFERENCE`, `AMBIGUOUS-NAME`, `EMPTY`, `SELECTION-AMBIGUOUS`, `SELECTION-UNKNOWN`, `ARGUMENTS-AMBIGUOUS`, `REQUIRED`) | Project schema and selection. |
| `P-TARGET-AMBIGUOUS`, `P-TARGET-REQUIRED` | Target selection. |
| `P-CAPTURE-LIMIT` | Invalid `PARALYN_CAPTURE_LIMIT_BYTES` for `run PROGRAM`; validated before the evidence directory is created or host code is compiled. |

## Evidence layout for kernel cases

A new or empty `--artifacts DIR`, or `.paralyn/runs/<id>/` by default, contains:

- `case.toml`: the exact bytes that were parsed; their SHA256 is `case.sha256` in the report (the file is not re-read).
- `module.metal.prx`, `module.prx` or `module.prk`: the exact module bytes that were loaded.
- `outputs/<buffer>.bin`, with its SHA256 recorded in the report.
- `runtime.log` and `runtime-events.ndjson`, now routed here for the CLI's in-process runtime.
- `runtime/execution.json`, `runtime/generated.metal` and `runtime/source-N.metal`.
- `report.json`.

Input data files are identified by the case SHA256 values and by `initial_sha256` per argument; they are not copied.

Failures are handled in two ways:

- If binding is rejected before submission, the directory is removed, together with only those parent directories it created, so rejected input leaves no evidence directory.
- If a failure happens after submission, a `partial: true` report is kept beside whatever evidence exists.

## Build revision and dirty state

`cmake/build_info.cmake` runs as the always-built `paralyn_build_info` target. It writes `build/generated/paralyn_build_info.h` only when `git rev-parse HEAD`, `git status --porcelain` or the SHA256 of those changes differ. The changes hash covers the tracked diff plus the untracked-file list, not the contents of untracked files. Without git, the values are `unknown` and dirty is `null`.

The header is compiled into:

- **The CLI.** Reports gain `build {revision, dirty, changes_sha256, captured: "build"}`. `source_revision` and `build_dirty` now come from this header rather than the configure-time values. `.prx` `producer_version` uses it too, in the same format the qualification scripts check.
- **The runtime (`backends/metal/engine.mm`).** `execution.json` keeps `paralyn_commit` and `paralyn_dirty`, and a caller's `PARALYN_COMMIT` / `PARALYN_SOURCE_DIRTY` environment still wins. They now fall back to the embedded identity instead of `"unknown"`/`null`. `execution.json` also adds:
  - `paralyn_revision_source`: `environment` or `embedded_build`.
  - `paralyn_runtime_build`.

`doctor` now records `build`, `source_revision` and `build_dirty` in its report, and sets `PARALYN_LLVM_VERSION` for its own `execution.json`. A direct doctor capture therefore no longer needs the outer build capture for revision provenance. The installed and runtime-only probes still need to be recaptured to demonstrate this; see "Not done" below. `--version` output is unchanged, because three qualification scripts parse it exactly.

## Bounded capture and partial reports (stretch)

`paralyn run PROGRAM` now appends application stdout and stderr to the `application.stdout` and `application.stderr` sidecars while the child runs:

- The files are created with `O_EXCL`, so they are never overwritten.
- Each stream is limited to 1 GiB, or to the value of `PARALYN_CAPTURE_LIMIT_BYTES`. An invalid value fails with `P-CAPTURE-LIMIT` before any evidence directory exists.
- Excess bytes are counted and reported as truncated, in `application.capture` in the report.
- Only the first 16 MiB per stream stays in memory for the `verification.txt` compatibility transcript, which records any truncation.

A `report.json` with `status: running, partial: true` is written before any application code runs, and the final report replaces it. The original `execute()` overload is unchanged; `Capture` is an additive overload. The Windows runner writes the sidecars after the child exits (`streamed: false`) and remains unqualified.

## Test evidence (this branch, Apple M5, macOS 26.5.1, LLVM 21.1.8, Debug build)

`tests/kernel_cases.py` is registered as ctest `kernel_cases` with labels `gpu;cli;cases`. It reports **95 negative checks and 17 GPU events**. Every negative check asserts the exact stable id and that no evidence was created.

Review fixes (follow-up on this branch) added: `P-CASE-VERIFY-UNCOVERED` with a two-output Metal kernel written by the test; `P-REFERENCE-SHAPE` for `vector-add-f32` (count 8 over 5-element inputs, under `run`, `verify` and `check`) and for `transpose-f32` (width+1); `P-CASE-NOT-APPLICABLE` for `--entry` on a `.cu`, a `.py`, a project program and `check MODULE`; `P-MANIFEST-NOT-APPLICABLE` for `.cu`, `.py`, `inspect .cu`, `.prx` and `.prk`; `P-CAPTURE-LIMIT` asserted by id for a `.py` and a `.cu` program with the `--artifacts` directory absent afterwards. Positive checks added: the evidence `case.toml` hashes to `case.sha256` for every GPU run, and `inspect`/`check` of a `.prx` report `build`.

The GPU events are:

- **Five public MSL verifications:**
  - vector_add from `.metal` + manifest against a file reference;
  - vector_add from `.prx` against a builtin reference with sentinels;
  - block_reduce against the builtin reference;
  - tiled_transpose against a file reference;
  - tiled_transpose against a builtin reference, selected through the project.
- **One block_reduce run through a `paralyn.toml` TARGET.**
- **One `.prk` IR affine run** against a file reference at 0 ULP.
- **Two project-selection runs:** one discovered `./paralyn.toml` with a single target, and one declared default.
- **One `run` without verification.**
- **Two runs that prove mismatch detection:**
  - the wrong reference fails with exactly one mismatch at index 6;
  - the same one-ULP difference passes only when `ulp = 1` is declared.
- **One CUDA program through a project**, whose own comparison prints PASS; the report still says `not_requested`.
- **One `doctor`** with revision provenance in both the report and `execution.json`.
- **Three two-output kernel events** (test-local `two_outputs.metal`): `run` with a contract covering only `x` (allowed, not verified); `verify` with both outputs covered and a wrong `y` reference (fails with `P-VERIFY-MISMATCH`, no PASS); `verify` with both outputs covered and correct references (PASS, 8 values across 2 checks, re-audited in Python).

Every verified output is re-read by the test and compared in Python against the shipped reference bytes and an independent recomputation. The test also requires that `execution.json` records exactly one completed launch with the case's grid/block, and that the NDJSON events contain exactly one completed command.

`examples/cases/generate_data.py --check` proves that the shipped data and reference files reproduce byte for byte. `process_execution` adds bounded-capture checks: exact sidecar prefixes, byte counts, memory bounds, refusal to overwrite, and no leaked descriptors.

The full serialized `ctest -j1` passes **24 of 24** tests with no skips: the 23 existing targets plus `kernel_cases`. Gate A, `gate_b_correctness`, `public_msl`, `terminal_contract`, the native suites and `native_cli_arrays` are unchanged and passing. Timing qualification and `verify_benchmark.py` were deliberately **not** run in this lane, because the GPU is shared.

## Not done / unqualified

- **Case schema scope:**
  - Only 32-bit f32/i32/u32 contiguous buffers and scalars.
  - No buffer aliasing or offset views.
  - No multi-launch sequences or dependent launches.
  - No generated inputs, such as ranges or seeded RNG. They were deliberately left out because every input must be declared.
  - No per-range comparisons. There are three builtins.
  - A case always compares the whole output buffer.
- **Kernel cases only for Metal-backed modules:** public MSL and scalar IR on Metal. No other backend exists.
- **Clean capture not run.** No clean-revision permanent evidence capture was made. These results come from a dirty development worktree; the reports honestly record `dirty: true`. Archiving a clean capture, like the existing `qualify_product.py` flow, is still to do, and `qualify_product.py` does not yet run kernel cases.
- **Installed and runtime-only builds not recaptured.** Their doctor reports should now carry revision provenance, but that was not re-demonstrated here.
- **Streaming capture only on POSIX.** The Windows capture overload is untested and was not compiled in this lane. Python-level logs of the native runtime are not bounded by this change.
- **Terminal features still missing:** `init`, shell completions, named `bench`, `cache` and `toolchain` commands, and NDJSON command output.
- **`docs/terminal.md` has stale sentences.** It still says "project TOML" and "kernel-case manifests" are unimplemented; see the proposed edits below.

## Proposed reconciliation (for the integrator; not applied on this branch)

- **`docs/status.md` / `docs/handoff.md`:** mark handoff task 4 as partly done:
  - done: declarative TOML projects and typed kernel cases; kernel-only execution for public MSL and IR with declared independent verification; missing/malformed case tests; embedded build revision/dirty state in doctor and execution evidence; bounded streaming capture with partial reports on POSIX.
  - Record the evidence above (95 negative checks, 17 GPU events, 24/24 ctest) as development evidence from a dirty worktree, not as a clean permanent capture.
  - Next task 4 step: a clean `qualify_product.py`-style capture that runs the shipped cases; recapture installed/runtime-only doctor provenance; then init, completions, NDJSON output and multi-launch cases.
- **`docs/terminal.md`:** replace "project TOML … remain required future work" and "kernel-case manifests are not yet implemented" with a pointer to this file. State that in-memory capture was replaced by bounded streaming sidecars on POSIX.
- **`docs/portfolio-ledger.json`:** in the terminal/CLI row, add project/case configuration, kernel-only execution and declared-reference verify as implemented partial capabilities, and keep full terminal UX incomplete. In the Metal-source and native rows, note CLI kernel-case execution as an additional entry point; this is not new frontend coverage.
- **`README.md`:** optionally add a two-line example of `paralyn verify examples/cases/paralyn.toml --case vector-add`.
