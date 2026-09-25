# Checkpoint after CUDA/Metal Gate B

Implementation revision: `8af773c9b8925a27041fcca1ab4cc58c82c56edb`. Evidence is archived under `artifacts/gate-b/`; software remains 0.0.1 and no release tag was created. Historical first-execution files remain unchanged.

## Completed

Gate B correctness, numerical policy, negative paths, host semantics and benchmark are implemented and verified on the physical Apple M5. The fresh build passed all six CTest targets without skips; permanent qualification includes 75 correctness GPU launches and 880 benchmark launches. See `docs/status.md` for exact scope and `artifacts/gate-b/build-validation.json` for every command/toolchain/time/binary hash.

## Resume

Read `AGENTS.md`, `docs/master-mandate-v1.1.md`, `docs/portfolio-ledger.json`, and the native API design. Implement Stage B: a small C-compatible context/device/owned-buffer/view/module/kernel/queue/event/error boundary that reuses the actual Metal runtime and keeps CUDA-specific tokens in its compatibility adapter; then bind that same runtime from Python. Choose one concrete real-GPU native workload and implement its ownership/error tests before expanding the operator surface. Current native API documentation is a design, not an implemented API.

All other required frontend families and clients remain incomplete; finishing Gate B does not shrink the mandate. NVIDIA/AMD hardware was not available in this local session, so no CUDA/ROCm or cross-vendor claim exists. Keep PTX and SASS distinct and honor their provenance/reference-hardware gates. No unreviewed dependency installation, cloud spending, or background execution is implied by this checkpoint.

## Reproduce

```sh
cmake --build build -j 4
ctest --test-dir build --output-on-failure
python3 scripts/qualify_gate_b.py --paralyn build/paralyn --artifacts artifacts/runs/new-correctness --require-clean
python3 scripts/verify_benchmark.py --benchmark build/gate_b_benchmark --paralyn build/paralyn \
  --source examples/vector_add.cu --artifacts artifacts/runs/new-benchmark --require-clean
```

Use new output directories. For permanent captures, finish/commit source changes, run from a clean checkout into ignored or external paths, then archive the unchanged evidence in a later commit. Run the benchmark without other intentionally concurrent GPU workloads; the protocol does not claim complete system isolation.
