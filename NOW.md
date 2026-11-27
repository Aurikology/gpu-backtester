# NOW — gpu-backtester

_Last updated 2026-11-27 (improve loop, unattended hour)._

## State

- 46 tests (37 in the CPU-only build CI runs). RTX 4050 via WSL2 `Ubuntu-24.04`.
- **Variant sort** (default since 2026-11-27): bias, window, gap, tp2, stop, tp1. `sim_shared`
  352 -> ~250 ms on default, 1,957 -> ~1,390 ms on large; checksums unchanged
  (`a6bb0fa189bff7ee`, `a72cfddb9ccc5ff1`). End to end ~0.5 s, ~35x the 16-thread CPU.
  - sort itself vs grid order: 1.43x (`results/ab-sort.txt`)
  - tp2 ahead of the stop keys: a further 1.0-1.3% (`results/ab-tp2-order.txt`)
- **Single-loop kernel: 0.81x, kept off** (`--fused`, `results/ab-fused.txt`).
- **Divergence model** `gbt divergence [--explore]`: CPU replay of loop counts per warp. Predicted
  the sort at 1.35-1.55x vs 1.43x measured; predicted tp2-outer at 3.4% vs ~1.2% measured. Ranks
  orders correctly, overstates small gains ~2.5x. Shipped order is ~80% warp-efficient, so even
  perfect packing is worth at most ~1.25x more.
- **Host spans** in the sweep report: GPU context, sort/unsort, alloc/free (`results/sweep-wall-breakdown.txt`).
- Workflow: `scripts/verify.sh` (double check) -> commit -> push both -> ls-remote. Rules in `CLAUDE.md`.

## Where the time goes now (default grid)

| span | ms | note |
|---|---|---|
| `sim_shared` | ~250 | ~80% warp-efficient by the model; rest is targets/days ending trades on different bars |
| GPU context creation | 183-282 | fixed per process; only a long-lived process avoids it |
| sort + unsort (host) | 15-27 | a cheaper packed key could cut this |
| alloc + free buffers | 21-28 | |
| upload, `metrics`, download | ~17-24 | `metrics` at ~89% of measured peak: done |

**Aurik, human-only:** Nsight Compute works under WSL2 but the counters are blocked
(`ERR_NVGPUCTRPERM`). NVIDIA Control Panel (Windows) -> Desktop menu -> Enable Developer
Settings -> Developer -> Manage GPU Performance Counters -> "Allow access to the GPU
performance counters to all users". Needs admin. Then `ncu` gives real warp efficiency, and
the software model can be checked against it.

## Candidates (Aurik ranks)

1. **Run on AMD Developer Cloud** (MI300X): wave64 numbers + rocprofv3. Costs money; price unchecked.
2. **Small ROCm upstream PR** (rocPRIM/hipCUB docs or tests): free, but he writes and sends it.
3. **Kernel: share the entry scan (measured ceiling ~2x).** 62% of useful iterations are the
   entry scan, and the scan never reads tp1/tp2, so each of the 13,824 distinct scans
   (default grid) is repeated by 11 variants. Sharing it removes at most 56.7% of iterations
   (`results/scan-share.txt`; large grid 56.4%). Packing tweaks are capped at ~1.25x, so this is
   the lever. Design: kernel 1, one thread per (session, scan group) -> entry bar, direction,
   entry price, stop, risk; kernel 2, one thread per (session, variant) manages from that entry.
   Bit-identity is checkable (same expressions); host maps variant -> group. The model has
   overstated small gains ~2.5x, so expect less than 2x; measure.
4. **Host overhead:** packed sort key (15-27 ms -> a few), buffer reuse (21-28 ms).
