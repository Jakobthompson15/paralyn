# Gate B qualification — Apple M5

Captured from clean source revision `8af773c9b8925a27041fcca1ab4cc58c82c56edb` on 2026-09-25. This qualifies the documented CUDA/Metal subset, not the entire v1.1 portfolio or another GPU vendor. No CPU kernel fallback exists.

- `build-validation.json` and logs: fresh configure/build with already-installed dependencies; all six CTest tests passed. First independently verified GPU result took 6.698 seconds in this local run. This is not fresh-OS installation timing or a general setup-time promise.
- `gate-a/`: repeated complete ordinary CUDA vector addition with current implementation; includes transformed host source. Original Gate A captures elsewhere remain unchanged.
- `correctness/`: 66 positive GPU launches plus nine launches in the deliberately nonzero host-exit case; 75 completed Metal commands total. Edge/odd sizes through 1,000,003, seeded arithmetic, dependent launches, canaries, xyz indices, signed/unsigned boundaries, alias layout reuse, const/mutable aliases, FP32 exceptional cases, and host semantics were independently checked. Negative source diagnostics and inspection without host execution are recorded too.
- `benchmark/`: four sizes, ten warmups and one hundred measured iterations per variant/size, alternating order. Every output from all 880 launches was compared with an independent CPU reference. All samples, compile/copy/command/total times and memory counters are retained. See [protocol](../../benchmarks/README.md).

The benchmark compares generated and handwritten kernels through the same runtime. It does not establish a universal speedup. The maximum counted live Metal-buffer storage was 201,326,592 bytes; final counted storage was zero. Driver/pipeline/host storage is excluded.

Numerical observations on this machine: normal add/multiply samples had maximum 0 ULP distance under the 1-ULP policy; four subnormal cases differed by allowed flushing; the non-contracted multiply/add probe and signed-zero/exceptional-value checks passed. See [numerical policy](../../docs/numerics.md) for limits.

Capture commands and original absolute paths remain in raw logs. Runs were first saved beneath ignored `artifacts/runs/` so generated evidence did not dirty the source checkout; this archive is a byte-for-byte copy of those records. `manifest.json` hashes the archived records (excluding itself). No timing, source or status record was rewritten when archiving.
