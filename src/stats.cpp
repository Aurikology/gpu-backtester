#include "gbt/stats.hpp"

#include <cmath>
#include <limits>

namespace gbt {
namespace {

constexpr double kEulerGamma = 0.5772156649015329;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.141592653589793;
constexpr double kE = 2.718281828459045;

double mean_of(const std::vector<float>& r) {
  double s = 0;
  for (float x : r) s += x;
  return s / double(r.size());
}

}  // namespace

double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

// Acklam's rational approximation (relative error 1.15e-9), then one Halley step against
// erfc, which brings it to double precision.
double norm_inv(double p) {
  if (!(p > 0 && p < 1)) return kNaN;
  static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                             1.383577518672690e+02,  -3.066479806614716e+01, 2.506628277459239e+00};
  static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                             6.680131188771972e+01,  -1.328068155288572e+01};
  static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                             -2.549732539343734e+00, 4.374664141464968e+00,  2.938163982698783e+00};
  static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                             3.754408661907416e+00};
  const double lo = 0.02425;
  double x;
  if (p < lo) {
    const double q = std::sqrt(-2 * std::log(p));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p <= 1 - lo) {
    const double q = p - 0.5, r = q * q;
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  } else {
    const double q = std::sqrt(-2 * std::log(1 - p));
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const double e = norm_cdf(x) - p;
  const double u = e * std::sqrt(2 * kPi) * std::exp(x * x / 2);
  return x - u / (1 + x * u / 2);
}

double sharpe(const std::vector<float>& r) {
  if (r.size() < 2) return kNaN;
  const double m = mean_of(r);
  double ss = 0;
  for (float x : r) ss += (x - m) * (x - m);
  const double sd = std::sqrt(ss / double(r.size() - 1));
  return sd > 0 ? m / sd : kNaN;
}

double sharpe_of(const Metrics& m) {
  if (m.trades < 2) return kNaN;
  const double n = m.trades, mean = double(m.total_r) / n;
  const double var = (double(m.sum_sq) - n * mean * mean) / (n - 1);
  return var > 0 ? mean / std::sqrt(var) : kNaN;
}

double psr(const std::vector<float>& r, double sr_star) {
  const double sr = sharpe(r);
  if (!std::isfinite(sr)) return kNaN;
  const double m = mean_of(r);
  double m2 = 0, m3 = 0, m4 = 0;
  for (float x : r) {
    const double z = x - m;
    m2 += z * z;
    m3 += z * z * z;
    m4 += z * z * z * z;
  }
  const double n = double(r.size());
  m2 /= n, m3 /= n, m4 /= n;  // population moments, as in the Python version
  const double skew = m3 / std::pow(m2, 1.5), kurt = m4 / (m2 * m2);
  const double denom = 1 - skew * sr + (kurt - 1) / 4 * sr * sr;
  if (denom <= 0) return kNaN;
  return norm_cdf((sr - sr_star) * std::sqrt(n - 1) / std::sqrt(denom));
}

double expected_max_sharpe(size_t n, double var) {
  if (n < 2 || var <= 0) return 0.0;
  const double N = double(n);
  return std::sqrt(var) * ((1 - kEulerGamma) * norm_inv(1 - 1 / N) +
                           kEulerGamma * norm_inv(1 - 1 / (N * kE)));
}

double sr0_of(const std::vector<double>& t, size_t* trials) {
  double s = 0, ss = 0;
  size_t n = 0;
  for (double x : t)
    if (std::isfinite(x)) s += x, ++n;
  const double m = n ? s / double(n) : 0;
  for (double x : t)
    if (std::isfinite(x)) ss += (x - m) * (x - m);
  if (trials) *trials = n;
  return expected_max_sharpe(n, n > 1 ? ss / double(n - 1) : 0.0);
}

Deflated deflated_sharpe(const std::vector<float>& r, const std::vector<double>& trial_sharpes) {
  Deflated d;
  d.sr0 = sr0_of(trial_sharpes, &d.trials);
  d.sharpe = sharpe(r);
  d.psr0 = psr(r, 0.0);
  d.dsr = psr(r, d.sr0);
  return d;
}

}  // namespace gbt
