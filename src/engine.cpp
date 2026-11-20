#include "gbt/engine.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "gbt/profiler.hpp"

namespace gbt {

Backend parse_backend(const std::string& name) {
  if (name == "cpu1") return Backend::Cpu1;
  if (name == "cpu") return Backend::CpuN;
  if (name == "gpu-naive") return Backend::GpuNaive;
  if (name == "gpu" || name == "gpu-shared") return Backend::GpuShared;
  throw std::invalid_argument("backend must be cpu1, cpu, gpu-naive or gpu");
}

const char* backend_name(Backend b) {
  switch (b) {
    case Backend::Cpu1: return "cpu1";
    case Backend::CpuN: return "cpu";
    case Backend::GpuNaive: return "gpu-naive";
    case Backend::GpuShared: return "gpu";
  }
  return "?";
}

size_t oos_split(size_t num_sessions, double oos_fraction) {
  const double f = std::clamp(oos_fraction, 0.0, 1.0);
  return num_sessions - size_t(double(num_sessions) * f);
}

// One variant over all sessions, in date order. This loop is the CPU reference.
static VariantResult run_variant(const BarSeries& b, const Params& p, size_t split) {
  VariantResult res;
  float eq_is = 0, pk_is = 0, eq_oos = 0, pk_oos = 0;
  const float *o = b.open.data(), *h = b.high.data(), *l = b.low.data(), *c = b.close.data();
  for (size_t s = 0; s < b.sessions.size(); ++s) {
    const size_t i = size_t(b.sessions[s].begin);
    const float r = simulate_session(o + i, h + i, l + i, c + i, p);
    if (s < split) metrics_add(res.in_sample, eq_is, pk_is, r);
    else metrics_add(res.out_of_sample, eq_oos, pk_oos, r);
  }
  return res;
}

static std::vector<VariantResult> run_cpu(const BarSeries& bars, const std::vector<Params>& grid,
                                          const SweepOptions& opt, int threads) {
  const size_t split = oos_split(bars.num_sessions(), opt.oos_fraction);
  std::vector<VariantResult> out(grid.size());
  const double items = double(grid.size()) * double(bars.num_sessions());
  CpuSpan span(opt.timeline, threads == 1 ? "cpu1 sweep" : "cpu sweep", 0, items);
  if (threads <= 1) {
    for (size_t v = 0; v < grid.size(); ++v) out[v] = run_variant(bars, grid[v], split);
    return out;
  }
  // Static partitioning: variants cost roughly the same, so equal contiguous chunks
  // balance well and each thread writes its own slice of `out` (no sharing, no locks).
  std::vector<std::thread> pool;
  const size_t n = grid.size();
  for (int t = 0; t < threads; ++t) {
    const size_t lo = n * size_t(t) / size_t(threads), hi = n * size_t(t + 1) / size_t(threads);
    pool.emplace_back([&, lo, hi] {
      for (size_t v = lo; v < hi; ++v) out[v] = run_variant(bars, grid[v], split);
    });
  }
  for (auto& th : pool) th.join();
  return out;
}

std::vector<VariantResult> run_sweep(const BarSeries& bars, const std::vector<Params>& grid,
                                     const SweepOptions& opt) {
  switch (opt.backend) {
    case Backend::Cpu1:
      return run_cpu(bars, grid, opt, 1);
    case Backend::CpuN: {
      int t = opt.threads > 0 ? opt.threads : int(std::thread::hardware_concurrency());
      return run_cpu(bars, grid, opt, std::max(t, 1));
    }
    case Backend::GpuNaive:
    case Backend::GpuShared:
      if (!gpu_available()) throw std::runtime_error("no GPU backend in this build or no device");
      return run_sweep_gpu(bars, grid, opt);
  }
  throw std::logic_error("bad backend");
}

std::vector<float> trades_of(const BarSeries& b, const Params& p, size_t first, size_t last) {
  std::vector<float> r;
  last = std::min(last, b.num_sessions());
  for (size_t s = first; s < last; ++s) {
    const size_t i = size_t(b.sessions[s].begin);
    const float x = simulate_session(&b.open[i], &b.high[i], &b.low[i], &b.close[i], p);
    if (is_trade(x)) r.push_back(x);
  }
  return r;
}

uint64_t checksum(const std::vector<VariantResult>& results) {
  // FNV-1a over the exact bits of every metric: equal only if the backends agree to
  // the last bit, which they should, because both run the same IEEE float operations
  // in the same order (FMA contraction is disabled on both compilers).
  uint64_t h = 1469598103934665603ull;
  auto mix = [&](const void* p, size_t n) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  };
  for (const auto& r : results)
    for (const Metrics* m : {&r.in_sample, &r.out_of_sample}) {
      mix(&m->total_r, 4);
      mix(&m->max_dd, 4);
      mix(&m->sum_sq, 4);
      mix(&m->trades, 4);
      mix(&m->wins, 4);
    }
  return h;
}

}  // namespace gbt
