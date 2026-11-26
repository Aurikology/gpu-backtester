#include <gtest/gtest.h>

#include <cmath>

#include "gbt/stats.hpp"

using namespace gbt;

// Reference values come from the companion Python project (nq-quant, quant/stats.py) on the
// same inputs, so the two implementations cannot drift apart silently.
static const std::vector<float> kR = {1.5f, -1, -1, 4, 0, 0, -1, 2, 0.5f, -1,
                                      1.5f, 0,  -1, 3, -1, 0, 2, -1, 0,   1};

TEST(Stats, NormInvInvertsCdf) {
  for (double p : {1e-9, 1e-4, 0.01, 0.3, 0.5, 0.77, 0.99, 1 - 1e-6})
    EXPECT_NEAR(norm_cdf(norm_inv(p)), p, 1e-12 * std::max(1.0, p / (1 - p)));
  EXPECT_NEAR(norm_inv(0.975), 1.959963984540054, 1e-12);
}

TEST(Stats, SharpeAndPsrMatchPython) {
  EXPECT_NEAR(sharpe(kR), 0.28538513481008876, 1e-9);
  EXPECT_NEAR(psr(kR, 0.0), 0.9189395763383084, 1e-9);
  EXPECT_NEAR(psr(kR, 0.1), 0.8180922792830516, 1e-9);
}

TEST(Stats, ExpectedMaxSharpeMatchesPython) {
  EXPECT_NEAR(expected_max_sharpe(1000, 0.01), 0.32551215136527234, 1e-9);
  EXPECT_NEAR(expected_max_sharpe(152064, 0.0025), 0.22405732683915996, 1e-9);
  EXPECT_EQ(expected_max_sharpe(1, 0.01), 0.0);
}

TEST(Stats, DeflatedSharpeMatchesPython) {
  const auto d = deflated_sharpe(kR, {0.1, 0.3, -0.2, 0.05, 0.25, 0.0, NAN});
  EXPECT_EQ(d.trials, 6u);  // NaN dropped
  EXPECT_NEAR(d.sr0, 0.235, 5e-4);
  EXPECT_NEAR(d.psr0, 0.9189, 5e-5);
  EXPECT_NEAR(d.dsr, 0.5975, 5e-5);
}

TEST(Stats, StreamedSharpeEqualsListSharpe) {
  Metrics m;
  float eq = 0, peak = 0;
  for (float x : kR) metrics_add(m, eq, peak, x);
  metrics_add(m, eq, peak, no_trade());  // a session without a trade must not count
  EXPECT_EQ(m.trades, int(kR.size()));
  EXPECT_NEAR(sharpe_of(m), sharpe(kR), 1e-6);
}
