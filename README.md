# gpu-backtester

[![ci](https://github.com/Aurikology/gpu-backtester/actions/workflows/ci.yml/badge.svg)](https://github.com/Aurikology/gpu-backtester/actions/workflows/ci.yml)

A GPU backtesting engine for intraday Nasdaq-100 futures (NQ) strategies, written in
C++20 and **HIP**, AMD's GPU programming model. The same source compiles for AMD GPUs
(ROCm) and NVIDIA GPUs (HIP's CUDA backend); the numbers below are from an NVIDIA
RTX 4050 laptop GPU.

**Headline:** 152,064 strategy variants x 2,013 trading days (306 million session
simulations) in **0.34-0.45 s**, about **40-50x a 16-thread CPU** (16.9 s), with
**bit-identical results** on every backend. The larger grid, 855,360 variants (1.72 billion
simulations), runs in 1.2 s. About half of each default run is now fixed GPU start-up.

It answers two questions for a discretionary trader:
1. *Which version of my rules actually works?* A parameter sweep, ranked on 2019-2024 and
   judged on held-out 2024-2026 data.
2. *At what position size am I most likely to pass a prop-firm evaluation?* A GPU Monte
   Carlo over the trader's own results.

Most of this README is a performance investigation: what each version was limited by,
what changed, and which number moved.

---

## Results at a glance

| | |
|---|---|
| Data | 3.44M one-minute bars, 2019-01-02 to 2026-11-12, cut into 2,013 New York sessions |
| Engines | 1 CPU thread, 16-thread pool, naive GPU kernel, shared-memory GPU kernel |
| Correctness | 47 tests; every GPU configuration must match the CPU checksum to the bit |
| Engine speed-up | 2.6M sims/s (1 thread), 22M (16 threads), 86M (naive GPU), **580M-820M (tuned GPU)** |
| Primitives | hand-written single-pass scan at **0.92-1.02x hipCUB** and ~100% of measured peak bandwidth |
| Profiler | Perfetto timeline, per-kernel GB/s vs a *measured* 174 GB/s peak, Linux `perf_event` counters |

Reproduce everything: `scripts/setup-wsl.sh`, `scripts/build.sh`, then the commands in
[Reproducing](#reproducing).

---

## 1. The problem shape

One backtest is sequential: the equity curve is a running sum. But a **parameter sweep**
is `variants x sessions` independent simulations, because the strategy is flat at every
16:00 close, so no state crosses a day. That is hundreds of millions of independent tasks,
which is what a GPU is for. The only sequential part, each variant's equity curve and
drawdown, runs afterwards as one thread per variant.

## 2. One strategy function, two compilers

`include/gbt/strategy.hpp` holds the trading rules as a single
`__host__ __device__` function, `simulate_session()`. The CPU reference and both GPU
kernels call exactly this code.

That makes a strong correctness contract possible: **the GPU must produce the same
floating-point bits as the CPU.** Fused multiply-add is turned off on both compilers
(`-ffp-contract=off`, `--fmad=false`) so the arithmetic is identical, and `gbt bench`
compares an FNV-1a hash over every metric of every variant. A GPU bug shows up as a test
failure instead of as a suspiciously profitable backtest.

**The rules** are a parameterised "inverted fair value gap" (IFVG) day-trading setup:
- A fair value gap is three bars where bar 1 and bar 3 don't overlap. When price later
  *closes through* the gap, that's the entry signal.
- Stop behind the recent swing; **half off at TP1, stop to break-even, runner to TP2**.
- Optional bias filter from the 09:30 fifteen-minute candle. Entries before it closes are
  forbidden: no look-ahead.
- **Pessimistic fills:** if one bar touches both stop and target, the stop wins; stops pay
  a tick of slippage.

The tests check hand-built sessions with known answers. They also check a **mirror
property**: flip every price, longs become shorts, and the R must be identical.

## 3. Four engines, each a hypothesis

`gbt bench`, 8,192 variants x 2,013 sessions (16.5M simulations):

| backend | seconds | sims/s | vs 1 thread | checksum |
|---|---|---|---|---|
| `cpu1`, one thread | 6.32 | 2.6M | 1x | `18ec259c230f2286` |
| `cpu`, 16 threads | 0.77 | 21.6M | 8.3x | `18ec259c230f2286` |
| `gpu-naive` | 0.19 | 85.7M | 33x | `18ec259c230f2286` |
| `gpu` (shared memory) | 0.029 | 578M | **222x** | `18ec259c230f2286` |

**16 threads gave 8.3x, not 16x.** The i5-14450HX has 6 performance cores and 4
efficiency cores. Its 16 hardware threads include hyper-threads that share one core's
execution units. A static partition means the slowest core sets the finish time.

**Naive GPU kernel (`sim_naive`).** One thread per (variant, session). Its two problems:
- Neighbouring threads in a warp test the same variant on different days. Their bar reads
  are 1.5 KB apart, so each warp load becomes up to 32 separate memory transactions.
- Every day's bars are fetched once per variant: 152,064 times each.

**Shared-memory kernel (`sim_shared`).** One block per (day, 128 variants):
- The block copies that day's 390 bars into on-chip shared memory once (6 KB).
- Every thread then tests a *different variant* on the *same* bars. All 32 lanes read the
  same bar at the same moment, which shared memory serves as a single broadcast.
- Results are written session-major (`out[s * V + v]`), so a warp's 32 writes land in one
  contiguous 128-byte line, and the metrics pass reads them coalesced too.

Result: **6.7x over the naive kernel.**

## 4. What limits it now

From the profiler (`results/sweep-default-gpu.txt`):

| kernel | time share | GB/s | % of measured peak |
|---|---|---|---|
| `sim_shared` | ~95% | 3.3 | 2% |
| `metrics` | 2% | 156 | **90%** |

- **`metrics` is memory-bound and already at 90% of the ceiling.** There is nothing left
  to gain there without moving fewer bytes.
- **`sim_shared` uses 2% of bandwidth, so memory isn't its limit.** It is limited by
  control flow. Threads in one warp hold different parameters, so they take different
  branches and close their trades on different bars, and the warp waits for its slowest
  lane. That is **branch divergence**.

**The fix: sort variants so each warp holds similar ones.** `make_grid()` builds the grid
with the bias filter as its *innermost* axis, so every warp mixed all three filters, and the
filters disagree on almost everything: whether the day trades at all, the first allowed
entry bar, even the trade direction. The window was already an outer axis, so it was never
the problem. Before launch the engine now sorts variants by the parameters that decide
*whether and where* a trade opens (bias, window, gap size, stop). Only the targets, which
decide *when* it closes, still vary inside a warp. Results are scattered back to the
caller's order, so nothing downstream changes.

| grid | `sim_shared`, grid order | sorted | speed-up | full-grid checksum |
|---|---|---|---|---|
| default, 152,064 variants | 352 ms | 247 ms | **1.43x** | `a6bb0fa189bff7ee` both |
| large, 855,360 variants | 1,957 ms | 1,372 ms | **1.43x** | `a72cfddb9ccc5ff1` both |

End to end, a `gbt sweep` gains less (0.57 to 0.48 s), because the kernel is only part of
the wall time. The profiler now times the host side of a sweep too
(default grid, range over four runs; two are in `results/sweep-wall-breakdown.txt`):

| span | ms | |
|---|---|---|
| GPU context creation | 183-282 | fixed per process, as large as the kernel itself, and the noisiest part |
| `sim_shared` | 254-274 | the kernel |
| sort + unsort variants (host) | 15-27 | the cost of the divergence fix: 15-25% of the ~105 ms it saves |
| alloc + free device buffers | 21-28 | |
| upload, `metrics`, download | ~17-24 | |

`--no-sort` keeps the old order, and `scripts/ab-sort.sh` alternates the two on one binary
(`results/ab-sort.txt`).

**A divergence model, because the hardware counters are locked.** Under WSL2, Nsight
Compute refuses to read the GPU's counters without a driver setting, so `gbt divergence`
estimates warp efficiency on the CPU instead. It replays every (variant, sampled session)
with an instrumented copy of the strategy that counts its loop iterations. Each warp of 32
consecutive variants then costs either its slowest lane (*lockstep*) or its longest scan
plus the longest trade per distinct entry bar (*serialised*, lanes that enter later wait for
the group before them). Its first job was to predict a number already measured:

| order (default grid, every 4th session) | warp efficiency | predicted vs grid order | measured |
|---|---|---|---|
| grid order | 49-57% | 1.00x | 1.00x |
| sorted | 77% | 1.35x (lockstep) to 1.55x (serialised) | **1.43x** |

The measurement falls inside the model's bracket, so it can rank candidate orders in two
seconds of CPU time before anything runs on the GPU (`--explore`,
`results/divergence-model.txt`). Of eight orders it tried, the best moved the runner target
`tp2` ahead of the stop parameters, since it decides how long a trade stays open. The model
said 3.4% fewer warp iterations; on the GPU it was **1.0-1.3% faster** (8 of 8 alternating
runs, same checksums, `results/ab-tp2-order.txt`). The model ranks orders correctly but
overstates small effects about 2.5x, so its predictions are a shortlist, not a result. That
order is now the default. It also accounts for the failed experiment below: once
sorted, the two models differ by 1%, so a single loop could save at most ~1% of iterations
while costing more per iteration.

**The model's biggest finding: most of the work was repeated.** It splits the useful
iterations by loop: **62% are the entry scan**, and the scan never reads the targets. On
the default grid that means 13,824 distinct scans each run by 11 variants
(`results/scan-share.txt`). Sharing the scan removes up to 57% of the iterations, a
ceiling near 2.3x, against ~1.25x left for any reordering. So the kernel now does it
(`sim_grouped`): the strategy is split into `find_entry()` and `manage_trade()` (a test
proves together they equal `simulate_session()` to the bit). One block takes one session
and up to 128 scan groups. Phase 1: each thread runs one group's scan and leaves the entry
in shared memory. Phase 2: the block's threads manage every variant of those groups from
it.

| grid | one scan per variant | shared scan | speed-up | checksum |
|---|---|---|---|---|
| default, kernel | 252 ms | 131 ms | **1.92x** | `a6bb0fa189bff7ee` both |
| large, kernel | 1,400 ms | 727 ms | **1.93x** | `a72cfddb9ccc5ff1` both |
| default, `gbt sweep` wall | 0.47-0.53 s | 0.34-0.45 s | | |
| large, `gbt sweep` wall | 1.84-1.94 s | 1.20 s | | |

The first version was *slower*, 0.71x. It declared a fixed 1,024-entry shared array, 20 KB
per block against the 6 KB before, which cut how many blocks fit on each SM. Sizing it to
the block (2.5 KB at 128 threads) turned 0.71x into 1.92x. Block 256 runs 6% faster than
128 here (`results/ab-shared-scan.txt`), so the tuner's old winner is due a re-run.
`--no-share` restores the per-variant kernel.

**A follow-up that failed: one loop instead of two.** The strategy nests trade management
inside the entry scan, so I expected lanes that entered on different bars to run their trades
one after another. `simulate_session_fused()` rewrites the rules as a single loop over bars
with a scan/manage state, so every lane advances one bar per iteration. It is bit-identical
(same checksums on both grids, and a CPU test compares the two value for value) and uses the
same registers (39 vs 40), but it is **slower: 0.81x** on both grids (247 to 307 ms, 1,375
to 1,705 ms; `results/ab-fused.txt`). After the sort, a warp's lanes mostly enter on the same
bar, so the nested loops were already reconverging, and the fused loop pays its state check
and both halves' code every bar. That reading is a hypothesis: confirming it needs Nsight
Compute's warp-efficiency counters. `--fused` keeps the experiment reproducible; the default is
the nested loop.

## 5. Auto-tuning with a correctness oracle

`scripts/tune.py` runs every kernel at block sizes 32 to 1024. It **rejects any candidate
whose checksum differs from the CPU reference**, then keeps the fastest survivor.

| kernel | best block | sims/s |
|---|---|---|
| `gpu-naive` | 64 | 93M |
| `gpu` | **128** | **821M** (45x the 16-thread CPU on this subset) |

The default was 256; on 16,384 variants the tuner found 128 is ~25% faster, so 128 is
now the default (`results/tune.json`). The winner depends on problem size: on a 4,096-variant
run 256 edged ahead, because with fewer blocks in flight, larger blocks hide latency better.
Re-run the tuner for your workload and GPU. The design idea is the important part: anything, a script or an AI
agent, can propose a change, and the test oracle decides whether it counts.

## 6. GPU primitives from scratch: reduce and scan

`include/gbt/prims.cuh` builds reduce and inclusive scan in layers:
- **Warp level:** shuffles.
- **Block level:** shared memory.
- **Device level:** across blocks.

These are the building blocks of AMD's rocPRIM and NVIDIA's CUB. No code hard-codes a warp
width of 32, because AMD Instinct wavefronts are 64 lanes wide.

The benchmark (`gbt bench-prims`, `results/bench-prims.txt`) compares them with hipCUB. On
AMD hardware hipCUB is a wrapper around rocPRIM; on NVIDIA it wraps CUB.

| op, 67M uint32 | mine GB/s | hipCUB GB/s | mine / hipCUB | exact |
|---|---|---|---|---|
| reduce | 185.0 | 186.6 | 0.99 | yes |
| scan, reduce-then-scan (3 kernels) | 115.4 | 172.4 | **0.67** | yes |
| scan, single-pass decoupled look-back | 175.4 | 172.4 | **1.02** | yes |

**The 0.67 was predicted before it was measured.**
- Reduce-then-scan reads the input twice and writes it once, about 3n bytes.
- A single-pass scan moves 2n bytes.
- A memory-bound kernel that moves 1.5x the bytes should run at 2/3 the speed, and it did.

**The single-pass version** implements decoupled look-back (Merrill and Garland, 2016):
- Each tile publishes its sum as one 64-bit `[flag | value]` word, so no reader can see a
  flag next to a stale value.
- It then walks back through its predecessors' published values instead of waiting for a
  separate kernel.
- Tiles take their number from an atomic counter, not from `blockIdx`. A tile can only wait
  on tiles that are already running, which rules out deadlock.

It matches hipCUB at 16M-67M elements (0.92-1.02x across runs).

**Two quirks in the full table:**
- At 1M elements, rates exceed "100% of peak" because 4-8 MB fits in the GPU's 24 MB L2
  cache, while the peak was measured on main memory.
- At 65K elements every kernel is launch-bound.

## 7. Profiler

`include/gbt/profiler.hpp`, about 250 lines:
- **Timeline:** CPU spans plus GPU spans timed with device events, so they record real
  kernel time rather than launch time. `--trace` writes a Chrome trace; open it at
  [ui.perfetto.dev](https://ui.perfetto.dev).
- **Bandwidth against a measured peak.** `gbt info` times a device-to-device copy (174 GB/s
  here, against 192 GB/s on the spec sheet). Every span that moves bytes is reported as a
  percentage of that measured figure.
- **Hardware counters** through `perf_event_open(2)`: cycles, instructions, cache misses,
  branch misses. WSL2 does not expose the CPU's performance-monitoring unit, so on this
  machine the counters report `unavailable` with the reason instead of inventing numbers.
  On native Linux they fill in the bench table.

## 8. Prop-firm pass odds (`gbt lucid`)

**Method:**
- Each simulated attempt draws trades at random from a list of R-multiples, one trade per
  day, at a fixed dollar risk.
- An attempt passes at the profit target, fails at the drawdown limit, or ends unresolved
  when it runs out of days.
- The drawdown limit can trail the high-water mark (end of day) or stay static. The rules
  are inputs (`--target`, `--max-dd`, `--static`, ...); firms change them, so check yours.

**Randomness is counter-based** (SplitMix64 of seed, path and day). The GPU and CPU
therefore produce the same paths, and a test checks that they match path for path.

**Validation:** against the gambler's-ruin formula. With a fair +/-1R coin, a $500 target
and a $1,000 static floor, the pass rate must be exactly 2/3.

| R sample (target $3,000, trailing DD $2,000, 120 days, 1M paths) | mean R | best risk | pass rate |
|---|---|---|---|
| discretionary journal, 312 bar-replay trades | +0.42 | $250 | ~100% |
| best mechanical variant, 663 backtest trades | +0.14 | $200 | **56%** |

- With the mechanical edge, sizing matters: at $200 the pass rate is 56%; at $400 it drops
  to 45%, because failures climb faster than passes.
- The journal sample is the same 312 trades the companion Python project scores: one
  session that practised a different strategy (top/bottom-tick entries) is left out by what
  it traded, not how it did. At ~100% the simulator can no longer tell sizes apart between
  $200 and $650 (`results/lucid-journal.txt`).
- That figure assumes live trades will look like bar-replay trades, and they usually don't.
- 15 million paths take 0.2-0.3 s.

## 9. What it says about the trading

| | in-sample R (2019 to Jul 2024) | deflated Sharpe | out-of-sample R (Jul 2024 to Nov 2026) |
|---|---|---|---|
| rank 1 of 152,064 | +89.9 | 0.61 | +1.2 |
| best in-sample total R in the top 15 | **+184.1** | **0.00** | **-36.0** |
| top 15 that stayed profitable | | 0 of 15 reach 0.95 | 13 of 15, mostly +1R to +25R |

**Reading it honestly:**
- After 152,064 tries, the best in-sample row is the most overfit one by construction.
- Out-of-sample returns are a small fraction of in-sample.
- **Deflated Sharpe ratio** (Bailey and Lopez de Prado, `include/gbt/stats.hpp`): every
  one of the 152,064 variants counts as a trial. With no edge, the best per-trade Sharpe
  among them is expected to reach SR0 = 0.143; DSR is the probability a row's true Sharpe
  beats that, after skew and fat tails. The Sharpes come from sums the GPU already
  streams, so the correction adds no kernel work; only the printed rows replay their trades.
- **It flags the row that failed.** The +184R variant has the highest in-sample R but a
  Sharpe of 0.075, well under SR0, so DSR 0.00, and it lost 36R out of sample. No row
  clears 0.95. The variants are correlated, so 152,064 overstates the independent trials
  and the test leans conservative.
- The port is tested against the Python implementation's numbers on the same inputs.

The tool's most useful output was measuring how much of an apparent edge is search noise.

---

## Reproducing

```bash
# Ubuntu 24.04 (native or WSL2), NVIDIA GPU:
sudo bash scripts/setup-wsl.sh            # gcc, CMake, GoogleTest, CUDA 12.8, HIP headers
bash scripts/build.sh NVIDIA build        # builds and runs the 47 tests
# AMD GPU with ROCm installed:  bash scripts/build.sh AMD build
# No GPU:                       bash scripts/build.sh OFF build

./build/gbt data bars.csv --out bars.gbt  # any 1-minute CSV: time,open,high,low,close
./build/gbt bench --data bars.gbt --variants 8192
./build/gbt sweep --data bars.gbt --grid default --trace trace.json
./build/gbt bench-prims
./build/gbt divergence --data bars.gbt --every 4 --explore   # CPU-only warp-efficiency model
./build/gbt lucid --r trades.txt --target 3000 --max-dd 2000 --risk 100:800:50
python3 scripts/tune.py --gbt build/gbt --data bars.gbt
```

Without `--data`, `sweep` and `bench` run on synthetic sessions. Market data is not
included in the repo. The results above use Dukascopy's free NAS100 one-minute history.

## Layout

```
include/gbt/  strategy.hpp   rules (host+device), metrics, parameter grids
              data.hpp       bars, sessions, New York time and DST
              engine.hpp     the four backends
              prims.cuh      warp/block/device reduce and scan
              montecarlo.hpp evaluation simulator
              stats.hpp      probabilistic and deflated Sharpe ratio
              profiler.hpp   timeline, Chrome trace, perf_event counters
src/          gpu_engine.cu  sweep + Monte Carlo kernels, peak-bandwidth probe
              prims.cu       device scans and the hipCUB benchmark
tests/        hand-built sessions, mirror property, CPU==GPU, gambler's ruin, scan sizes,
              deflated Sharpe parity with Python
scripts/      setup, build, gate (build + tests + bit-check + timed sweep -> results/perf-log.tsv),
              divergence A/B, auto-tuner
docs/         TALKING-POINTS.md
```

## Limitations

- **NVIDIA hardware only so far.** The AMD build path (`GBT_GPU=AMD`) is written but has not
  run on AMD hardware. Wave64 timing on an Instinct GPU is the obvious next measurement.
- **Data:** Dukascopy NAS100 is a CFD on the index, not the CME futures contract. The prices
  track closely but are not identical.
- **One trade per session, no commissions in the sweep.** Commissions are an input to the
  Monte Carlo (`--cost`).
- **CI runs without a GPU.** Hosted runners have none, so GPU tests skip there and run
  locally.

MIT licensed.
