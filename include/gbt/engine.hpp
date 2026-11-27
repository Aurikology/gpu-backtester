#pragma once
// Sweep engines: run every parameter variant over every session and summarise each
// variant. Four backends, slowest to fastest, so each speed-up can be measured:
//
//   Cpu1      one CPU thread, the reference everything else is checked against
//   CpuN      std::thread pool, one contiguous block of variants per thread
//   GpuNaive  one GPU thread per (variant, session); every thread reads its bars
//             from global memory
//   GpuShared one GPU block per (session, 256 variants); the block stages that
//             session's bars in shared memory once and every thread reuses them

#include <string>
#include <vector>

#include "gbt/data.hpp"
#include "gbt/strategy.hpp"

namespace gbt {

class Timeline;  // profiler.hpp

enum class Backend { Cpu1, CpuN, GpuNaive, GpuShared };
Backend parse_backend(const std::string& name);
const char* backend_name(Backend b);
bool gpu_available();  // false in a CPU-only build or with no device

struct VariantResult {
  Metrics in_sample;      // sessions before the split date
  Metrics out_of_sample;  // sessions after it, never used for ranking
};

struct SweepOptions {
  Backend backend = Backend::GpuShared;
  double oos_fraction = 0.3;  // last 30% of sessions are held out
  int threads = 0;            // CpuN: 0 = hardware concurrency
  int block_size = 128;       // GPU threads per block; 128 won scripts/tune.py on an RTX 4050
  size_t max_batch_bytes = size_t(1) << 30;  // GPU result buffer per batch
  // GpuShared: run variants in an order where neighbouring lanes take the same branches
  // (bias filter outermost, targets innermost), then return results in the caller's order.
  bool sort_variants = true;
  // GpuShared: simulate with the single-loop simulate_session_fused() instead of the
  // nested-loop simulate_session(). Same bits; the CPU always runs the nested one.
  // Off: measured 0.81x on both grids (results/ab-fused.txt). Kept as an ablation.
  bool fused_loop = false;
  Timeline* timeline = nullptr;               // optional profiling
};

// Index of the first out-of-sample session.
size_t oos_split(size_t num_sessions, double oos_fraction);

std::vector<VariantResult> run_sweep(const BarSeries& bars, const std::vector<Params>& grid,
                                     const SweepOptions& opt);

// GPU entry point (gpu_engine.cu). Not available in CPU-only builds.
std::vector<VariantResult> run_sweep_gpu(const BarSeries& bars, const std::vector<Params>& grid,
                                         const SweepOptions& opt);

// Per-trade R list of one variant, date order (feeds the Lucid simulator).
std::vector<float> trades_of(const BarSeries& bars, const Params& p, size_t first_session = 0,
                             size_t last_session = SIZE_MAX);

// Order-independent fingerprint of a sweep, used to prove two backends agree.
uint64_t checksum(const std::vector<VariantResult>& results);

}  // namespace gbt
