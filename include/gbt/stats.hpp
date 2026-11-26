#pragma once
// Multiple-testing-aware Sharpe statistics, ported from the companion Python project
// (nq-quant, quant/stats.py) and tested against its numbers.
//
// Bailey and Lopez de Prado, "The Deflated Sharpe Ratio" (2014):
// - PSR(SR*) = P(true Sharpe > SR*) given the observed per-trade Sharpe, the sample length,
//   and the skew and kurtosis of the returns (fat tails widen a Sharpe estimate's error).
// - Under the null (no edge), the best of N Sharpe estimates with cross-trial variance V is
//   expected to reach SR0 = sqrt(V) * ((1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e))).
// - DSR = PSR(SR0): the chance the chosen variant beats what luck alone gives after N tries.
//
// Sharpe here is per trade (mean R / std R), not annualised.

#include <cstddef>
#include <vector>

#include "gbt/strategy.hpp"

namespace gbt {

double norm_cdf(double x);
double norm_inv(double p);  // inverse of norm_cdf, p in (0, 1)

double sharpe(const std::vector<float>& r);  // sample std (n - 1); NaN below 2 trades or zero std
double sharpe_of(const Metrics& m);           // the same, from streamed sums (no per-trade list)
double psr(const std::vector<float>& r, double sr_star);
double expected_max_sharpe(size_t n_trials, double trial_var);

struct Deflated {
  double sharpe = 0, sr0 = 0, psr0 = 0, dsr = 0;
  size_t trials = 0;
};
// DSR of the chosen variant's trades `r`, deflated for every Sharpe in `trial_sharpes`
// (non-finite values are dropped).
Deflated deflated_sharpe(const std::vector<float>& r, const std::vector<double>& trial_sharpes);
// SR0 alone, for when many rows share one set of trials.
double sr0_of(const std::vector<double>& trial_sharpes, size_t* trials = nullptr);

}  // namespace gbt
