# Gate B measurement

Build `gate_b_benchmark`, then run from the repository root:

```sh
python3 scripts/verify_benchmark.py \
  --benchmark build/gate_b_benchmark --paralyn build/paralyn \
  --source examples/vector_add.cu --artifacts artifacts/benchmark-new \
  --require-clean
```

The wrapper derives the Git revision/dirty state and the selected binary's LLVM version, runs the benchmark, and independently audits the complete evidence. Omit `--require-clean` for development runs, which record their actual dirty state. The output directory must not exist. Use an otherwise idle GPU; no process isolation or thermal stabilization is enforced. `execution.json` records the actual device, OS, memory model, thermal state and low-power mode at capture. Do not describe a run as isolated merely because it completed.

The generated variant comes from the real CUDA frontend and verified typed IR. The handwritten variant is an independently written Metal vector addition. Both use the same allocation, copy, dispatch, synchronization and safe/precise compilation path. Both disable floating-point contraction. This comparison isolates kernel generation; it is not a comparison with a separate native Metal application's runtime overhead.

For each of 1,024, 65,536, 1,048,576 and 16,777,216 elements, each variant receives ten warmups and one hundred measured iterations. Generated runs first on even iterations; handwritten runs first on odd iterations. Each run initializes a distinct output sentinel, transfers both inputs, launches, synchronizes, reads the result and compares every element with a CPU reference. The comparison and sentinel initialization are outside the reported total latency.

`samples.csv` retains all 880 warmup and measured samples. Columns separate Metal library/pipeline compilation, H2D and D2H API duration, actual shared-memory copies, command-buffer GPU duration, and total latency. Pipeline compilation occurs on the first warmup of each variant; all later launches reuse an in-process pipeline. No attempt is made to evict Apple's driver caches. Frontend compilation is timed once and reported separately in `benchmark.json`. Per-launch IR verification and generated MSL emission remain in total latency; they are not included in the pipeline-compilation column.

The runtime-owned memory counter measures requested lengths of live Metal buffers. It excludes host vectors, source/IR storage, command metadata, pipeline objects and driver allocations. At the largest size, three FP32 buffers occupy 201,326,592 counted bytes; all are freed before final capture.

Measured launches discard progress through in-memory stdout and stderr sinks. The benchmark disables `PARALYN_RUNTIME_LOG` and `PARALYN_EVENT_LOG` before execution so inherited settings cannot add file writes or durable flushes to measured work. Actual command records remain in memory and are exported after measurement. This preserves the original quiet-progress measurement boundary after runtime progress moved to stderr. The earlier product-foundation capture retains valid numerical/completion evidence, but its host totals include unsuppressed stderr progress and should not be used as a before/after runtime performance comparison.

Evidence also includes original CUDA, transformed host source, verified IR, handwritten Metal, every distinct dispatched source, per-launch command status and GPU timestamps. No timing target determines success. Success requires the complete protocol and all CPU comparisons to pass on a physical GPU.
