# NOW — gpu-backtester

_Last updated 2026-11-27._

## State

- 41 tests green locally (RTX 4050, NVIDIA backend). CI = CPU only (no GPU on hosted runners).
- **Divergence fix landed 2026-11-27:** variants are sorted before launch (bias, window, gap,
  stop outer; targets inner). `sim_shared` 352 -> 247 ms on default, 1,957 -> 1,372 ms on large
  (1.43x both), checksums unchanged. End to end: default sweep ~0.48 s, 35x the 16-thread CPU.
- **Single-loop kernel tried 2026-11-27: 0.81x, kept off** (`--fused`, `results/ab-fused.txt`).
  Bit-identical, same registers; nested loops already reconverge after the sort.
- 43 tests (34 in the CPU-only build).
- Workflow: `scripts/gate.sh` (build, tests, bit-check, timed sweep, perf log) +
  `scripts/ab-sort.sh`. Rules in `CLAUDE.md`.

## Where the time goes now (default grid, sorted)

| span | ms | note |
|---|---|---|
| `sim_shared` | ~247-270 | still divergence-bound: targets still vary inside a warp, and different days end trades on different bars |
| GPU context creation | 183-282 | fixed per process (own span since 2026-11-27); dominates small runs |
| sort + unsort (host) | 15-27 | ~a quarter of what the sort saves; cheaper key possible |
| alloc + free buffers | 21-28 | |
| `metrics` | ~8 | 89% of measured peak bandwidth: done |

**Aurik, human-only:** Nsight Compute works under WSL2 but the counters are blocked
(`ERR_NVGPUCTRPERM`). NVIDIA Control Panel (Windows) -> Desktop menu -> Enable Developer
Settings -> Developer -> Manage GPU Performance Counters -> "Allow access to the GPU
performance counters to all users". Needs admin. Then `ncu` gives real warp efficiency.

## Candidates (Aurik ranks)

1. **Run on AMD Developer Cloud** (MI300X): wave64 numbers + rocprofv3 to check my profiler.
   Costs money; price not checked yet.
2. **Small ROCm upstream PR** (rocPRIM/hipCUB docs or tests): open-source credit for AMD.
3. **Next kernel step:** measure before guessing again. The single-loop guess was wrong, so
   get warp-efficiency numbers first (Nsight Compute; check it runs under WSL2). Then the
   candidates are a data-driven order (group variants by the entry bar they actually trade
   on a sample of days) or splitting entry search and trade management into two kernels.
4. **The fixed 0.18-0.28 s is GPU context creation**, once per process: only a long-lived process
   (one `gbt` run doing several sweeps) avoids it. Buffer reuse would save the 21-28 ms.
