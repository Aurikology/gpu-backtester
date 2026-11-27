// Hand-built sessions with known answers, plus a mirror-symmetry property.
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

#include "gbt/data.hpp"
#include "gbt/strategy.hpp"

using namespace gbt;

namespace {

struct Sess {
  std::vector<float> o, h, l, c;
  Sess() : o(kSessionMinutes, 100), h(kSessionMinutes, 100), l(kSessionMinutes, 100), c(kSessionMinutes, 100) {}
  void bar(int k, float open, float high, float low, float close) {
    o[size_t(k)] = open; h[size_t(k)] = high; l[size_t(k)] = low; c[size_t(k)] = close;
  }
  void flat(int from, int to, float p) {
    for (int k = from; k < to; ++k) bar(k, p, p, p, p);
  }
  float run(const Params& p) const { return simulate_session(o.data(), h.data(), l.data(), c.data(), p); }
};

// Bars 0-10 flat at 100. Bars 11-12 fall and leave a bearish gap between bar 10's low
// (100) and bar 12's high (95). Bar 20 closes at 101, above the gap: a long IFVG entry.
// Swing low of bars 18-20 is 92, so risk is 9 points, tp1 = 110, tp2 = 119.
Sess long_setup() {
  Sess s;
  s.bar(11, 100, 100, 90, 90);
  s.bar(12, 90, 95, 88, 89);
  s.flat(13, 20, 92);
  s.bar(20, 92, 101, 92, 101);
  s.flat(21, kSessionMinutes, 101);
  return s;
}

Params base() { return Params{0, 390, 0.f, 3, 0.f, 1.f, 2.f, 0}; }

}  // namespace

TEST(Strategy, TakesHalfAtTp1ThenRunnerStopsAtBreakEven) {
  Sess s = long_setup();
  s.bar(21, 101, 110.5f, 100, 110);  // tp1 (110) hit, stop never touched
  s.bar(22, 110, 111, 101, 101);     // back to entry: break-even stop, minus one tick
  const float r = s.run(base());
  EXPECT_NEAR(r, 0.5f + 0.5f * (-0.25f / 9.f), 1e-6);
}

TEST(Strategy, StopWinsWhenOneBarTouchesBoth) {
  Sess s = long_setup();
  s.bar(21, 101, 111, 91, 100);  // touches tp1 AND the stop: assume the worst
  EXPECT_NEAR(s.run(base()), (91.75f - 101.f) / 9.f, 1e-6);
}

TEST(Strategy, RunnerReachesTp2) {
  Sess s = long_setup();
  s.bar(21, 101, 110, 100, 109);
  s.bar(22, 109, 119, 105, 118);
  EXPECT_NEAR(s.run(base()), 0.5f * 1.f + 0.5f * 2.f, 1e-6);
}

TEST(Strategy, OpenTradeExitsAtTheClose) {
  Sess s = long_setup();
  s.flat(21, kSessionMinutes, 105);
  EXPECT_NEAR(s.run(base()), 4.f / 9.f, 1e-6);
}

TEST(Strategy, GapSmallerThanMinimumIsIgnored) {
  Params p = base();
  p.min_gap = 6;  // the gap is 5 points
  EXPECT_FALSE(is_trade(long_setup().run(p)));
}

TEST(Strategy, InversionBeforeTheWindowUsesUpTheGap) {
  Params p = base();
  p.win_start = 21;  // the inversion happens at bar 20
  EXPECT_FALSE(is_trade(long_setup().run(p)));
}

TEST(Strategy, StopsTighterThanMinimumAreSkipped) {
  Sess s = long_setup();
  s.bar(20, 99, 101, 99, 101);  // swing low of the last bar only = 99, risk 2 < 3
  Params p = base();
  p.stop_lookback = 1;
  EXPECT_FALSE(is_trade(s.run(p)));
}

TEST(Strategy, BiasFilterFollowsThe0930FifteenMinuteCandle) {
  // The first 15 bars close at 92 after opening at 100: a bearish 15m candle.
  Sess s = long_setup();
  Params p = base();
  p.bias = 1;  // with the candle: shorts only, so this long is rejected
  EXPECT_FALSE(is_trade(s.run(p)));
  p.bias = 2;  // against the candle: the long is allowed
  EXPECT_TRUE(is_trade(s.run(p)));
}

TEST(Strategy, BiasFilterNeverEntersBeforeTheCandleCloses) {
  // Same pattern but the inversion lands at bar 10, inside the first 15 minutes.
  Sess s;
  s.bar(5, 100, 100, 90, 90);
  s.bar(6, 90, 95, 88, 89);
  s.flat(7, 10, 92);
  s.bar(10, 92, 101, 92, 101);
  s.flat(11, 15, 99);  // 15m candle: 100 -> 99, bearish, so bias=2 allows longs
  s.flat(15, kSessionMinutes, 99);
  Params p = base();
  p.bias = 2;
  EXPECT_FALSE(is_trade(s.run(p))) << "an entry at bar 10 would use a candle that closes at bar 14";
}

// Flip every price around a level: highs become lows, longs become shorts. A correct
// rule set must give exactly the same R on the mirror image.
TEST(Strategy, MirrorImageGivesTheSameResult) {
  BarSeries b = make_synthetic(60, 99);
  const auto grid = make_grid(GridSize::Small);
  int traded = 0;
  for (const auto& sess : b.sessions) {
    const size_t i = size_t(sess.begin);
    std::vector<float> o(kSessionMinutes), h(kSessionMinutes), l(kSessionMinutes), c(kSessionMinutes);
    for (int k = 0; k < kSessionMinutes; ++k) {
      o[size_t(k)] = 32768.f - b.open[i + size_t(k)];
      c[size_t(k)] = 32768.f - b.close[i + size_t(k)];
      h[size_t(k)] = 32768.f - b.low[i + size_t(k)];
      l[size_t(k)] = 32768.f - b.high[i + size_t(k)];
    }
    for (const auto& p : grid) {
      const float a = simulate_session(&b.open[i], &b.high[i], &b.low[i], &b.close[i], p);
      const float m = simulate_session(o.data(), h.data(), l.data(), c.data(), p);
      ASSERT_EQ(is_trade(a), is_trade(m)) << describe(p);
      if (is_trade(a)) {
        ASSERT_NEAR(a, m, 1e-4) << describe(p);
        ++traded;
      }
    }
  }
  EXPECT_GT(traded, 100) << "the property needs trades to mean anything";
}

TEST(Strategy, GridsHaveTheAdvertisedSizes) {
  EXPECT_EQ(make_grid(GridSize::Small).size(), 288u);
  EXPECT_EQ(make_grid(GridSize::Default).size(), 152064u);
  EXPECT_EQ(make_grid(GridSize::Large).size(), 855360u);
  for (const auto& p : make_grid(GridSize::Default)) ASSERT_GT(p.tp2_r, p.tp1_r);
}

// simulate_session_fused() (GPU, one loop) must reproduce simulate_session() (CPU,
// nested loops) to the bit, NaN for NaN, on every variant of the small grid and a slice
// of the default grid across synthetic sessions.
TEST(Strategy, FusedLoopIsBitIdenticalToNested) {
  const BarSeries b = make_synthetic(120, 7);
  std::vector<Params> grid = make_grid(GridSize::Small);
  const auto def = make_grid(GridSize::Default);
  for (size_t i = 0; i < def.size(); i += 97) grid.push_back(def[i]);
  size_t trades = 0;
  for (const Session& s : b.sessions) {
    const size_t i = size_t(s.begin);
    const float *o = &b.open[i], *h = &b.high[i], *l = &b.low[i], *c = &b.close[i];
    for (const Params& p : grid) {
      const float want = simulate_session(o, h, l, c, p);
      const float got = simulate_session_fused(o, h, l, c, p);
      uint32_t wb, gb;
      std::memcpy(&wb, &want, 4);
      std::memcpy(&gb, &got, 4);
      ASSERT_EQ(gb, wb) << describe(p) << " session " << s.date;
      trades += is_trade(want) ? 1 : 0;
    }
  }
  EXPECT_GT(trades, size_t(1000)) << "too few trades: the comparison proves little";
}

// The shared-scan kernel computes find_entry() once per group and manage_trade() per
// variant. Together they must equal simulate_session() to the bit, NaN for NaN.
TEST(Strategy, SplitEntryAndManageIsBitIdentical) {
  const BarSeries b = make_synthetic(120, 11);
  std::vector<Params> grid = make_grid(GridSize::Small);
  const auto def = make_grid(GridSize::Default);
  for (size_t i = 0; i < def.size(); i += 89) grid.push_back(def[i]);
  size_t trades = 0;
  for (const Session& s : b.sessions) {
    const size_t i = size_t(s.begin);
    const float *o = &b.open[i], *h = &b.high[i], *l = &b.low[i], *c = &b.close[i];
    for (const Params& p : grid) {
      const float want = simulate_session(o, h, l, c, p);
      const float got = manage_trade(h, l, c, p, find_entry(o, h, l, c, p));
      uint32_t wb, gb;
      std::memcpy(&wb, &want, 4);
      std::memcpy(&gb, &got, 4);
      ASSERT_EQ(gb, wb) << describe(p) << " session " << s.date;
      trades += is_trade(want) ? 1 : 0;
    }
  }
  EXPECT_GT(trades, size_t(1000));
}
