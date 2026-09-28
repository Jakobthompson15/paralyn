# Benchmark progress isolation correction

Clean production code qualified at `87978b1f99dab214565c232a406635f4222a1a65`; the measurement-only fix was captured at `e465e74`. Compiler, runtime, backend, native bindings and CLI implementation files did not change between those revisions. The product's 23 CTests and named GPU qualifications remain in [product-foundation](../product-foundation/README.md).

The terminal work moved runtime progress to stderr. The old benchmark's `QuietRuntime` only suppressed stdout, so the initial product capture wrote 100,223 transcript bytes during timed launches instead of the historical 493 bytes. Its output verification and GPU completion are valid, but its host totals include this measurement-boundary regression.

The fix restores both stream buffers with RAII and disables inherited `PARALYN_RUNTIME_LOG`/`PARALYN_EVENT_LOG` file output before measured work. Errors still propagate; records remain in memory and export after measurement. Sizes, ten warmups, 100 measured iterations, alternating variants, independent CPU comparisons and all 880 samples remain unchanged. The capture deliberately supplied log-file destinations and confirmed neither was created. [Build and execution commands](build-validation.json) and [source/binary snapshots](snapshot-sha256.json) identify the exact tested implementation.

The [corrected full benchmark](corrected/capture.json) passed with 499 transcript bytes and 880 completed, independently checked GPU launches. Two full baseline runs were also performed from an isolated clean `f3c6c99` checkout: [first](baseline-investigation/baseline/capture.json) and [repeat](baseline-repeat/capture.json). All raw samples, source artifacts and timings are preserved, without replacing any earlier archive.

[Comparison](comparison.json) computes medians over all 100 measured samples per size/variant for each capture. Corrected tiny-kernel timings fall within the range observed across baseline runs. Larger cases and baseline repetitions also vary substantially, so these sequential, non-isolated desktop measurements support neither a universal speedup nor an absence-of-regression claim. They establish the repaired measurement boundary and full numerical/protocol qualification. Controlled performance testing remains an explicit release requirement.

Hardware is the same physical Apple M5 and recorded macOS/toolchain. No other qualification workload was intentionally run concurrently; other system work, thermal stabilization and device power scheduling were not controlled. This archive adds 2,640 GPU launches (three full protocols) separately from the initial product archive's counts.

Every file except the checksum manifest itself is covered by [SHA256SUMS.json](SHA256SUMS.json). Embedded original paths and provenance were preserved. No original Gate A/B, stage-B or product-foundation evidence was edited.
