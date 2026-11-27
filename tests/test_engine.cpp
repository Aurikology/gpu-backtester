// Every backend must agree with the single-thread reference to the last bit.
#include <gtest/gtest.h>

#include "gbt/engine.hpp"
#include "gbt/profiler.hpp"

using namespace gbt;

namespace {
const BarSeries& bars() {
  static BarSeries b = make_synthetic(300, 2024);
  return b;
}
std::vector<VariantResult> run(Backend be, int block = 256) {
  SweepOptions opt;
  opt.backend = be;
  opt.block_size = block;
  return run_sweep(bars(), make_grid(GridSize::Small), opt);
}
}  // namespace

TEST(Engine, ThreadPoolMatchesSingleThread) {
  EXPECT_EQ(checksum(run(Backend::Cpu1)), checksum(run(Backend::CpuN)));
}

TEST(Engine, GpuKernelsMatchCpuBitForBit) {
  if (!gpu_available()) GTEST_SKIP() << "no GPU";
  const auto ref = run(Backend::Cpu1);
  for (Backend be : {Backend::GpuNaive, Backend::GpuShared})
    for (int block : {64, 128, 256, 512}) {
      const auto got = run(be, block);
      ASSERT_EQ(got.size(), ref.size());
      for (size_t v = 0; v < ref.size(); ++v) {
        ASSERT_EQ(got[v].in_sample.trades, ref[v].in_sample.trades) << "variant " << v;
        ASSERT_EQ(got[v].in_sample.total_r, ref[v].in_sample.total_r) << "variant " << v;
        ASSERT_EQ(got[v].out_of_sample.max_dd, ref[v].out_of_sample.max_dd) << "variant " << v;
      }
      EXPECT_EQ(checksum(got), checksum(ref)) << backend_name(be) << " block " << block;
    }
}

TEST(Engine, GpuBatchingDoesNotChangeResults) {
  if (!gpu_available()) GTEST_SKIP() << "no GPU";
  SweepOptions opt;
  opt.max_batch_bytes = 300 * sizeof(float) * 7;  // 7 variants per batch: many batches
  const auto grid = make_grid(GridSize::Small);
  EXPECT_EQ(checksum(run_sweep(bars(), grid, opt)), checksum(run(Backend::Cpu1)));
}

// The divergence fix reorders variants on the device and must hand every result back to
// its own variant: compare field by field, not just the order-independent checksum.
TEST(Engine, GpuVariantSortKeepsCallerOrder) {
  if (!gpu_available()) GTEST_SKIP() << "no GPU";
  const auto grid = make_grid(GridSize::Small);
  ASSERT_NE(grid[0].bias, grid[1].bias) << "grid order already sorted: test proves nothing";
  SweepOptions opt;
  opt.max_batch_bytes = 300 * sizeof(float) * 7;  // sorting and batching together
  opt.sort_variants = false;
  const auto plain = run_sweep(bars(), grid, opt);
  opt.sort_variants = true;
  const auto sorted = run_sweep(bars(), grid, opt);
  ASSERT_EQ(plain.size(), sorted.size());
  for (size_t v = 0; v < plain.size(); ++v) {
    ASSERT_EQ(sorted[v].in_sample.trades, plain[v].in_sample.trades) << "variant " << v;
    ASSERT_EQ(sorted[v].in_sample.total_r, plain[v].in_sample.total_r) << "variant " << v;
    ASSERT_EQ(sorted[v].out_of_sample.total_r, plain[v].out_of_sample.total_r) << "variant " << v;
    ASSERT_EQ(sorted[v].out_of_sample.max_dd, plain[v].out_of_sample.max_dd) << "variant " << v;
  }
}

TEST(Engine, GpuFusedLoopMatchesCpu) {
  if (!gpu_available()) GTEST_SKIP() << "no GPU";
  SweepOptions opt;
  opt.fused_loop = true;
  EXPECT_EQ(checksum(run_sweep(bars(), make_grid(GridSize::Small), opt)),
            checksum(run(Backend::Cpu1)));
}

// A one-lane warp cannot diverge: every model must equal the useful work exactly.
TEST(DivergenceModel, OneLaneWarpsHaveNoWaste) {
  const auto grid = make_grid(GridSize::Small);
  const auto m = model_divergence(bars(), grid, divergence_order(grid), 1, 1, 2);
  EXPECT_GT(m.useful, 0.0);
  EXPECT_EQ(m.lockstep, m.useful);
  EXPECT_EQ(m.serialised, m.useful);
}

// Waste can only grow with warp width, serialised >= lockstep >= useful, and the
// sorted order must beat the grid order (that is the whole point of the sort).
TEST(DivergenceModel, BoundsAndSortedBeatsGridOrder) {
  const auto grid = make_grid(GridSize::Small);
  std::vector<uint32_t> plain(grid.size());
  for (size_t i = 0; i < plain.size(); ++i) plain[i] = uint32_t(i);
  const auto g = model_divergence(bars(), grid, plain, 32, 1, 2);
  const auto s = model_divergence(bars(), grid, divergence_order(grid), 32, 1, 2);
  for (const auto& m : {g, s}) {
    EXPECT_GE(m.serialised, m.lockstep);
    EXPECT_GE(m.lockstep, m.useful);
  }
  EXPECT_DOUBLE_EQ(g.useful, s.useful);  // same work, different packing
  EXPECT_LT(s.lockstep, g.lockstep);
}

// Counting must not change the answer.
TEST(DivergenceModel, CountingLeavesResultsUnchanged) {
  const auto grid = make_grid(GridSize::Small);
  const BarSeries& b = bars();
  for (size_t s = 0; s < 40; ++s) {
    const size_t i = size_t(b.sessions[s].begin);
    for (const Params& p : grid) {
      StepCount st;
      const float x = simulate_session(&b.open[i], &b.high[i], &b.low[i], &b.close[i], p);
      const float y = simulate_session<true>(&b.open[i], &b.high[i], &b.low[i], &b.close[i], p, &st);
      ASSERT_TRUE((x == y) || (x != x && y != y));
      ASSERT_EQ(st.entry >= 0, is_trade(x));  // an entry always produces a result
      if (st.entry >= 0) {
        ASSERT_GE(st.manage, 1);
      }
    }
  }
}

TEST(Engine, MetricsAgreeWithTheTradeList) {
  const auto grid = make_grid(GridSize::Small);
  const auto res = run(Backend::Cpu1);
  const size_t split = oos_split(bars().num_sessions(), 0.3);
  for (size_t v = 0; v < grid.size(); v += 17) {
    const auto is = trades_of(bars(), grid[v], 0, split);
    const auto oos = trades_of(bars(), grid[v], split);
    ASSERT_EQ(int(is.size()), res[v].in_sample.trades);
    ASSERT_EQ(int(oos.size()), res[v].out_of_sample.trades);
    float sum = 0, eq = 0, peak = 0, dd = 0;
    for (float r : is) {
      sum += r;
      eq += r;
      peak = std::max(peak, eq);
      dd = std::max(dd, peak - eq);
    }
    EXPECT_EQ(sum, res[v].in_sample.total_r);
    EXPECT_EQ(dd, res[v].in_sample.max_dd);
  }
}

TEST(Engine, OutOfSampleSplitIsByDateOrder) {
  EXPECT_EQ(oos_split(1000, 0.3), 700u);
  EXPECT_EQ(oos_split(1000, 0.0), 1000u);
}

TEST(Profiler, ChromeTraceIsWritten) {
  Timeline tl;
  { CpuSpan s(&tl, "work", 1e6, 10); }
  ASSERT_EQ(tl.spans().size(), 1u);
  EXPECT_NE(tl.report(100).find("work"), std::string::npos);
}
