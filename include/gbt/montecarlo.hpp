#pragma once
// Prop-firm evaluation odds: "if I trade like this, how likely am I to pass?"
//
// Input is a list of per-trade results in R (from a trading journal or a backtest).
// Each simulated attempt draws trades at random from that list (bootstrap with
// replacement), one trade per day, at a fixed dollar risk per trade, and stops when it
// either reaches the profit target (pass), breaches the drawdown limit (fail), or runs
// out of days (unresolved). Sweeping the dollar risk shows the size that maximises the
// pass rate: too small never reaches the target in time, too large hits the drawdown.
//
// The evaluation rules are inputs, not constants: firms change them, so check the
// current rules for your account size before trusting a number.

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "gbt/common.hpp"

namespace gbt {

struct EvalRules {
  float profit_target = 3000.f;  // dollars
  float max_drawdown = 2000.f;   // dollars below the high-water mark
  bool trailing = true;          // false = static floor at -max_drawdown
  bool lock_at_start = true;     // trailing floor stops rising once it reaches the start balance
  float daily_loss_limit = 0.f;  // dollars, 0 = none (with one trade a day this caps a single loss)
  float cost_per_trade = 0.f;    // commissions + fees, dollars
  int max_days = 60;             // attempts unresolved after this many trading days
};

struct RiskOutcome {
  float risk = 0;          // dollars risked per trade (1R)
  double pass = 0, fail = 0, unresolved = 0;  // fractions of paths
  double mean_days_to_pass = 0;
};

// Counter-based random numbers: the n-th draw of path p depends only on (seed, p, n),
// so the CPU and GPU produce identical paths and any single path can be replayed.
GBT_HD inline uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// Outcome of one path: +days if passed on that day, -days if failed, 0 if unresolved.
GBT_HD inline int simulate_path(const float* r, int nr, float risk, const EvalRules& rules,
                                uint64_t seed, uint64_t path) {
  float balance = 0.f, high = 0.f;
  const uint64_t base = splitmix64(seed ^ (path * 0xD1B54A32D192ED03ull));
  for (int day = 1; day <= rules.max_days; ++day) {
    const uint64_t u = splitmix64(base + uint64_t(day));
    const int idx = int((u >> 32) * uint64_t(nr) >> 32);  // unbiased enough: 32-bit multiply-shift
    float pnl = r[idx] * risk - rules.cost_per_trade;
    if (rules.daily_loss_limit > 0.f && pnl < -rules.daily_loss_limit) pnl = -rules.daily_loss_limit;
    balance += pnl;
    float floor_ = rules.trailing ? high - rules.max_drawdown : -rules.max_drawdown;
    if (rules.trailing && rules.lock_at_start && floor_ > 0.f) floor_ = 0.f;
    if (balance <= floor_) return -day;
    if (balance >= rules.profit_target) return day;
    high = fmaxf(high, balance);  // end-of-day high-water mark
  }
  return 0;
}

struct McOptions {
  std::vector<float> risks;   // dollar risk levels to test
  int paths = 100000;         // attempts per risk level
  uint64_t seed = 42;
  bool use_gpu = true;
};

std::vector<RiskOutcome> run_montecarlo(const std::vector<float>& r_samples, const EvalRules& rules,
                                        const McOptions& opt);
std::vector<RiskOutcome> run_montecarlo_gpu(const std::vector<float>& r_samples,
                                            const EvalRules& rules, const McOptions& opt);

// Reads R multiples: a plain list (one number per line) or an FX Replay trade export
// (uses the avgRiskReward column of closed trades).
std::vector<float> load_r_samples(const std::string& path);

}  // namespace gbt
