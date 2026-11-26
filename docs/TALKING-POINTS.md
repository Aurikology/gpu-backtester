# Talking points

How to explain this project out loud, from 30 seconds to a full technical interview.
Every number here comes from `results/` and can be regenerated with the commands in the
README.

## The 30-second version

> "I trade Nasdaq futures on paper and wanted to know which version of my rules actually
> works. Testing one rule change over seven years of minute data took seconds; testing
> every combination took hours. So I wrote a backtester in C++ and HIP, AMD's GPU
> language, that runs **152,064 strategy variants over 2,013 trading days in
> 0.6 seconds**, 30x faster than my 16-thread CPU version, and
> the GPU agrees with the CPU to the last bit. Then I built a Monte Carlo simulator on top
> that tells me my odds of passing a prop-firm evaluation at each position size."

## The 2-minute walkthrough (one sentence per stage)

1. **Data.** 3.4 million one-minute bars become 2,013 New York trading sessions. I wrote the
   daylight-saving conversion by hand and tested its exact switch-over minutes, because a
   one-hour shift silently breaks every intraday rule.
2. **One strategy function, two compilers.** The trading rules are one `__host__ __device__`
   function. The CPU and the GPU run literally the same code, so I can demand bit-identical
   results and catch GPU bugs as test failures instead of as suspicious profits.
3. **Four engines, each measured.** One CPU thread, a 16-thread pool, a naive GPU kernel,
   and a shared-memory GPU kernel. Each step is a hypothesis about where the time goes,
   and the benchmark either confirms it or does not.
4. **The key optimisation.** In the naive kernel every thread fetches its own day's bars
   from global memory: the same bars get re-read once per variant, and neighbouring threads
   read addresses 1.5 KB apart. The shared-memory kernel gives each block one trading day:
   the block copies that day's 390 bars on-chip once and every thread tests a different
   variant against them. Result: 6.7x.
5. **Primitives.** I wrote reduce and scan from warp shuffles up, the same building blocks
   as AMD's rocPRIM, and benchmarked them against hipCUB. My first scan ran at 0.67x
   hipCUB, and I'd predicted that before measuring: reduce-then-scan moves about 3n bytes,
   a single-pass scan moves 2n, so a memory-bound kernel should run at two thirds. Then I
   built the single-pass version (decoupled look-back) and it matched hipCUB, 0.92-1.02x,
   at about 100% of measured bandwidth.
6. **Profiler.** Device-event timing per kernel, exported as a Perfetto timeline, with
   bandwidth judged against a peak I measured (174 GB/s), not the spec sheet. CPU
   hardware counters through `perf_event_open`, which degrade honestly to "unavailable"
   under WSL2 where the PMU is hidden.
7. **Auto-tuner.** A search loop that tries launch configurations and *rejects any whose
   checksum differs from the CPU reference*. Wrong is never fast.
8. **What it told me about trading.** One variant made +184R from 2019 to mid-2024, then
   *lost* 36R on the held-out 2024 to 2026 data. Most top variants stayed positive, but at
   a fraction of their in-sample returns. And the Monte Carlo showed my mechanical rules
   give a 56% pass rate at best, against ~100% for my 312 hand-traded journal trades: the discretion is
   where the edge is, if it survives live trading.

## Questions you will get, and answers

**Why a GPU for a backtest? Isn't it sequential?**
One backtest is sequential in time, yes: the equity curve is a running sum. But a parameter
sweep is (variants x days) independent simulations, and each day is independent once
positions are flat at the close. That's millions of independent tasks. The sequential
part (the equity curve per variant) runs as one thread per variant afterwards.

**Why HIP and not CUDA?**
HIP is AMD's API, and it is nearly 1:1 with CUDA. On an AMD GPU it compiles with ROCm; on
my NVIDIA laptop GPU, HIP's headers map every call to CUDA at compile time. Same source,
both vendors. The one thing I had to make portable by hand is warp width: NVIDIA warps are
32 lanes, AMD Instinct wavefronts are 64, so no code hard-codes 32.

**How do you know the GPU results are right?**
Three layers. Hand-built sessions with known answers (a stop that's hit on the same bar as
the target must lose). A mirror-image property test: flip every price, longs become shorts,
the R must be identical. And the CPU reference: every GPU configuration must produce the
same checksum over every metric's exact bits. I turned off fused multiply-add on both
compilers so that's achievable.

**What limits the GPU kernel now?**
Two kernels, two different answers. The metrics kernel runs at 90% of measured peak
bandwidth, so it's memory-bound and nearly done. The simulation kernel uses 2% of
bandwidth, so memory isn't the problem: it's branch divergence. Lanes of one warp hold
different parameters and close their trades on different bars, and the warp waits for the
slowest lane. Next step: sort variants so each warp holds similar ones.

**Walk me through decoupled look-back.**
Each tile of 2,048 elements computes its own sum and publishes it right away as one 64-bit
word, flag plus value, so nobody can read a flag with a stale value. Then it walks
backwards over its predecessors: it adds their published sums until it reaches one that
has published a full prefix, publishes its own full prefix, and writes its output. The trap
is deadlock: if tile 5 spins on tile 4 and tile 4 was never scheduled, it waits forever.
So tiles take their number from an atomic counter when they start, not from blockIdx. A
tile's predecessors are then always already running.

**Why did 16 CPU threads give only 8.3x?**
The laptop CPU has 6 performance cores and 4 efficiency cores. Of its 16 threads, 6 are
hyper-threads sharing a core. And with static partitioning, the slowest core decides when
the job finishes.

**How did you avoid fooling yourself about the trading results?**
The last 30% of days are held out and never used for ranking. Fills are pessimistic: if one
bar touches both the stop and the target, the stop wins, and stops pay a tick of slippage.
The top in-sample variants mostly lost money out of sample, and I report that rather than
hiding it.

**What would you do with an AMD GPU?**
Run the same code on an MI300X through AMD's Developer Cloud, compare wave64 vs my wave32
numbers, and profile with rocprofv3 to check my own profiler's kernel times against AMD's.

**What did you use AI tools for?**
I built it with Claude Code as a pair programmer and used the test suite and the CPU
checksum as the oracle: nothing an agent proposed counted unless it matched the reference.
The auto-tuner is the same idea automated.

## Resume bullets

- Built a C++20/HIP GPU backtesting engine that evaluates 152,064 parameter variants over
  7 years of 1-minute Nasdaq futures data in 0.6 s, 30x a 16-thread
  CPU baseline, with bit-identical CPU/GPU results enforced by tests.
- Raised kernel throughput 6.7x by staging each trading day in shared memory
  and switching to a session-major layout for coalesced writes; profiled with a custom
  Perfetto timeline and measured-peak bandwidth reporting.
- Implemented warp-shuffle reduce and a single-pass decoupled look-back scan matching hipCUB
  (0.92-1.02x, ~100% of measured bandwidth); built a GPU Monte Carlo simulator estimating
  prop-firm evaluation pass rates across position sizes.
