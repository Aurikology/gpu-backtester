#include <gtest/gtest.h>

#include <cmath>

#include "gbt/engine.hpp"
#include "gbt/montecarlo.hpp"

using namespace gbt;

// A fair coin, +1R or -1R at $100 risk, static floor $1,000 down, target $500 up.
// Gambler's ruin: P(reach +5 steps before -10) = 10 / (5 + 10) = 2/3 exactly.
TEST(MonteCarlo, MatchesGamblersRuin) {
  EvalRules rules;
  rules.profit_target = 500;
  rules.max_drawdown = 1000;
  rules.trailing = false;
  rules.max_days = 100000;
  McOptions opt;
  opt.risks = {100};
  opt.paths = 200000;
  opt.use_gpu = false;
  const auto res = run_montecarlo({1.f, -1.f}, rules, opt);
  const double se = std::sqrt((2.0 / 3) * (1.0 / 3) / opt.paths);
  EXPECT_NEAR(res[0].pass, 2.0 / 3.0, 4 * se);
  EXPECT_NEAR(res[0].unresolved, 0.0, 1e-4);
}

TEST(MonteCarlo, AlwaysWinPassesAlwaysLoseFails) {
  EvalRules rules;
  McOptions opt;
  opt.risks = {500};
  opt.paths = 1000;
  opt.use_gpu = false;
  EXPECT_DOUBLE_EQ(run_montecarlo({2.f}, rules, opt)[0].pass, 1.0);
  EXPECT_DOUBLE_EQ(run_montecarlo({2.f}, rules, opt)[0].mean_days_to_pass, 3.0);  // 1000/day
  EXPECT_DOUBLE_EQ(run_montecarlo({-1.f}, rules, opt)[0].fail, 1.0);
}

TEST(MonteCarlo, TrailingDrawdownIsStricterThanStatic) {
  EvalRules rules;
  McOptions opt;
  opt.risks = {250};
  opt.paths = 50000;
  opt.use_gpu = false;
  const std::vector<float> r = {2.f, -1.f, -1.f, 1.5f, -1.f};
  const double trailing = run_montecarlo(r, rules, opt)[0].pass;
  rules.trailing = false;
  const double fixed = run_montecarlo(r, rules, opt)[0].pass;
  EXPECT_LT(trailing, fixed);
}

TEST(MonteCarlo, GpuMatchesCpuPathForPath) {
  if (!gpu_available()) GTEST_SKIP() << "no GPU";
  EvalRules rules;
  McOptions opt;
  opt.risks = {100, 300, 700};
  opt.paths = 20000;
  const std::vector<float> r = {2.f, -1.f, -1.f, 3.f, 0.f, -1.f, 0.5f};
  opt.use_gpu = false;
  const auto cpu = run_montecarlo(r, rules, opt);
  const auto gpu = run_montecarlo_gpu(r, rules, opt);
  for (size_t i = 0; i < cpu.size(); ++i) {
    EXPECT_EQ(cpu[i].pass, gpu[i].pass);
    EXPECT_EQ(cpu[i].fail, gpu[i].fail);
    EXPECT_EQ(cpu[i].mean_days_to_pass, gpu[i].mean_days_to_pass);
  }
}
