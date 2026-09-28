# Terminal contract at the native-product checkpoint

The command parser is C++/CLI11 2.4.2. JSON reports use schema `paralyn.report`, version 1. Human-facing command output is currently plain text; `NO_COLOR`, non-TTY and `--color` never introduce escape codes. Animated/guided output, project TOML, shell completions and the explorer remain required future work.

Implemented commands:

```sh
paralyn devices --json
paralyn devices --requires fp32
paralyn doctor --device metal:0
paralyn support
paralyn inspect examples/vector_add.cu
paralyn check examples/vector_add.cu --device metal:0
paralyn explain examples/vector_add.cu --device metal:0 --json
paralyn compile examples/native/kernels.cu --output kernels.prk
paralyn compile examples/metal/kernels.metal --manifest examples/metal/kernels.json --output kernels.prx
paralyn check kernels.prx --device metal:0
paralyn run examples/vector_add.cu --device metal:0 --report-json result.json
paralyn run examples/native/arrays.py --device metal:0 --json
paralyn run examples/native/arrays.cpp --device metal:0
paralyn verify builtin:vector-add --device metal:0
paralyn report .paralyn/runs/RUN_ID/report.json --json
```

`auto` and the old numeric selector `0` remain supported. `unicuda` remains a command alias. `metal:0` names an enumeration slot; the capability record separately contains the backend-qualified registry identity. NVIDIA and ROCm selectors fail as unimplemented backends, not missing SDKs. No SDK installation alone enables them.

`inspect` validates the source/container contract and reports static launch expressions without evaluating application host code. `check` and `explain` additionally load/compile/reflection-check the module on the selected backend without dispatching work or running `main`. A Metal `compile` packages a bounded, hashed source/resource contract; Metal compilation/reflection happens at `check` or module load. CUDA `compile` writes verified scalar IR. Existing artifacts are never overwritten.

`doctor` runs a bundled 257-element FP32 vector-add kernel on the physical GPU and independently computes and compares its host reference. It preserves actual inputs, output, module, generated Metal, completion and timing. `verify builtin:vector-add` uses that same declared conformance fixture. Arbitrary source verification protocols and kernel-case manifests are not yet implemented; these requests receive an explicit error. The standalone Gate A/B/native/product qualification scripts remain the larger verification and benchmark interfaces.

`run` executes a complete CUDA program or a native Python/C/C++ application. Native applications use the existing native API; this command does **not** compile arbitrary Python functions or offload ordinary C++ automatically. `PARALYN_PYTHON` selects the Python executable; otherwise it uses `python3`. Python 3.9.6 and 3.14.5 are tested. `PARALYN_DEVICE` supplies the default native context selection; an application may explicitly override it, so actual runtime evidence must be consulted for placement.

Application arguments after `--` are passed unchanged as an argument vector, including spaces, Unicode, empty arguments and shell metacharacters. There is no shell command evaluation. C sources use the configured C compiler in C11 mode; C++ uses C++17. CUDA host rewriting preserves the prior source contract.

## Streams, events, failure and evidence

Application stdout and stderr retain separate streams. Paralyn progress/diagnostics go to stderr. Runtime progress goes into a separate `runtime.log` during CLI runs; `--verbose` shows that captured log. Direct native clients default to runtime progress on stderr. `PARALYN_RUNTIME_LOG` overrides the text-log destination; `PARALYN_EVENT_LOG` enables a separate durable NDJSON runtime event channel.

Runtime event schema `paralyn.runtime.event` version 1 includes a process-local sequence, category/status/detail, context/operation identifiers, byte counts and a **host** monotonic observation time. Host observation time is never reported as GPU timing. Context creation, pipeline compilation, transfers, prepared submission, observed command completion and API errors are distinguished. A prepared submission is not proof that a command completed. Completion records follow observed Metal command status; errors in event/log storage propagate.

By default a run writes `.paralyn/runs/<run-id>/`, or a new/empty `--artifacts` directory. Preserved files include application stdout/stderr, runtime log/events, transformed CUDA host source, IR, generated shaders and execution records when supplied by the runtime. Native apps receive a separate new `native/` evidence directory through `PARALYN_ARTIFACT_DIR`; this preserves the native exporter's non-overwrite contract while the CLI owns its sibling streams/reports. The CLI supplies its matching default library/operator paths unless explicitly overridden. Native apps without exported execution records still have observed runtime events; reports do not manufacture shader/timing evidence for unobserved work. Compiler diagnostics are separate sidecars. `verification.txt` remains a compatibility transcript for existing qualifiers; arbitrary text in it is not an authenticated verification result.

`--json` emits one report object on stdout and captures application streams in named files. Without `--json`, `--report-json PATH` keeps the application streams live and additionally writes the report. An existing report is rejected before application execution. Application exit codes are retained. `failure_origin` distinguishes application, runtime and Paralyn failures where the event evidence establishes the distinction. Parser failures also produce JSON when requested.

An application printing “PASS” cannot change the report's `verification.status = not_requested`. `run` never upgrades that text into a reference comparison. Reports retain source SHA256, build revision/dirty flag, toolchain identity, actual device/event evidence, program arguments, exit status and limitations. Arbitrary host file I/O inputs are not yet traced. `report` checks the saved report's top-level schema name and version; it does not authenticate its contents, revalidate its evidence or rerun it.

On POSIX, process launch uses `posix_spawnp` with parent-built environment/argument vectors and separate pipes. Ctrl-C/termination is forwarded to the child's process group. Reports record interruption without claiming that submitted Metal commands were cancelled. Child cleanup and original signal handlers are restored on exceptional paths. A native Windows process implementation exists but remains unqualified until tested on an authorized Windows machine.

The full plan additionally requires init, project/case configuration, general verify, named bench, cache, toolchain, explore, NDJSON command output, rich diagnostics/remedies, completions and installer/offline UX. None is implied by this checkpoint. Process output is currently captured in memory before final sidecar persistence; streaming bounded storage and partial crash reports remain an explicit hardening task.
