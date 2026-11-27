# NOW — gpu-backtester

_Last updated 2026-11-27._

## State

- 41 tests green locally (RTX 4050, NVIDIA backend). CI = CPU only (no GPU on hosted runners).
- **Divergence fix landed 2026-11-27:** variants are sorted before launch (bias, window, gap,
  stop outer; targets inner). `sim_shared` 352 -> 247 ms on default, 1,957 -> 1,372 ms on large
  (1.43x both), checksums unchanged. End to end: default sweep ~0.48 s, 35x the 16-thread CPU.
- Workflow: `scripts/gate.sh` (build, tests, bit-check, timed sweep, perf log) +
  `scripts/ab-sort.sh`. Rules in `CLAUDE.md`.

## Where the time goes now (default grid, sorted)

| span | ms | note |
|---|---|---|
| `sim_shared` | ~247 | still divergence-bound: targets still vary inside a warp, and different days end trades on different bars |
| GPU context + upload | ~200 | fixed per process; dominates small runs |
| `metrics` | ~8 | 89% of measured peak bandwidth: done |

## Candidates (Aurik ranks)

1. **Run on AMD Developer Cloud** (MI300X): wave64 numbers + rocprofv3 to check my profiler.
   Costs money; price not checked yet.
2. **Small ROCm upstream PR** (rocPRIM/hipCUB docs or tests): open-source credit for AMD.
3. **Next kernel step:** data-driven order (group variants by the entry bar they actually
   trade on a sample of days), or split entry search and trade management into two passes.
   Measure the expected gain from warp-efficiency first (needs Nsight Compute, works outside
   WSL's PMU limits).
4. **Cut the fixed ~0.2 s** (persistent context / reuse buffers across runs) for the tuner and
   small sweeps.
