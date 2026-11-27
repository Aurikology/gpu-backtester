#pragma once
// The strategy: a parameterised version of an NQ "inverted fair value gap" (IFVG)
// day-trading rule set. One function, simulate_session(), is compiled for both the
// CPU and the GPU, so the GPU result can be checked against the CPU bar for bar.
//
// Vocabulary
//   FVG (fair value gap): three bars where bar 1 and bar 3 do not overlap.
//     bearish FVG: high[j] < low[j-2]   (price fell so fast it left a hole)
//     bullish FVG: low[j]  > high[j-2]
//   IFVG (inversion): price later CLOSES through the far side of the gap, which
//     flips its meaning. Closing above a bearish gap is a long signal; closing below
//     a bullish gap is a short signal.
//
// Trade management (matches the live rules this was written for):
//   - entry at the close of the inversion bar, one trade per session
//   - stop beyond the swing of the last `stop_lookback` bars, plus `stop_buf` points
//   - minimum stop of kMinRisk points, else the setup is skipped
//   - half off at tp1 (in R), stop moves to break-even, runner targets tp2
//   - pessimistic fills: if one bar touches the stop and a target, the stop wins;
//     stops pay kStopSlip points of slippage; anything open at 16:00 exits at the close
//
// Optional bias filter from the 09:30 15-minute candle (bars 0..14): trade only with
// its direction (bias=1) or only against it (bias=2). Entries before bar 15 are not
// allowed when the filter is on, because that candle has not closed yet (no look-ahead).

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "gbt/common.hpp"

namespace gbt {

constexpr float kMinRisk = 3.0f;    // points; smaller stops are noise on NQ
constexpr float kStopSlip = 0.25f;  // one NQ tick of slippage on every stop

struct Params {
  int32_t win_start;      // minutes after 09:30 when entries open
  int32_t win_len;        // how many minutes entries stay open
  float min_gap;          // smallest gap that counts, in points
  int32_t stop_lookback;  // bars of swing the stop sits behind
  float stop_buf;         // extra points beyond that swing
  float tp1_r;            // first target in R: half off, stop to break-even
  float tp2_r;            // runner target in R
  int32_t bias;           // 0 = any direction, 1 = with the 09:30 15m candle, 2 = against it
};
static_assert(sizeof(Params) == 32, "Params is copied to the GPU as raw bytes");

// A session with no trade returns NaN (not 0.0, because a break-even trade is 0.0).
GBT_HD inline float no_trade() { return NAN; }
GBT_HD inline bool is_trade(float r) { return r == r; }  // false only for NaN

// Simulates one session for one parameter set. Pointers point at the 09:30 bar;
// each array holds kSessionMinutes bars. Returns the day's result in R (risk units
// of the full position), or NaN if no trade was taken.
GBT_HD inline float simulate_session(const float* o, const float* h, const float* l,
                                     const float* c, const Params& p) {
  const int n = kSessionMinutes;

  int dir15 = 0;
  if (p.bias != 0) {
    const float d = c[14] - o[0];
    dir15 = d > 0.f ? 1 : (d < 0.f ? -1 : 0);
    if (dir15 == 0) return no_trade();
  }
  int start = p.win_start;
  if (p.bias != 0 && start < 15) start = 15;
  int end = p.win_start + p.win_len;  // exclusive
  if (end > n - 1) end = n - 1;       // leave at least one bar to manage the trade

  // Most recent untouched gap on each side.
  bool bear_on = false, bull_on = false;
  float bear_top = 0.f, bull_bot = 0.f;

  for (int j = 2; j < end; ++j) {
    if (h[j] < l[j - 2] && l[j - 2] - h[j] >= p.min_gap) { bear_on = true; bear_top = l[j - 2]; }
    if (l[j] > h[j - 2] && l[j] - h[j - 2] >= p.min_gap) { bull_on = true; bull_bot = h[j - 2]; }

    int dir = 0;
    if (bear_on && c[j] > bear_top) { dir = 1; bear_on = false; }
    else if (bull_on && c[j] < bull_bot) { dir = -1; bull_on = false; }
    if (dir == 0 || j < start) continue;  // an inversion before the window just uses up the gap
    if (p.bias == 1 && dir != dir15) continue;
    if (p.bias == 2 && dir != -dir15) continue;

    // Swing stop.
    const float entry = c[j];
    int first = j - p.stop_lookback + 1;
    if (first < 0) first = 0;
    float swing = dir > 0 ? l[j] : h[j];
    for (int k = first; k < j; ++k) swing = dir > 0 ? fminf(swing, l[k]) : fmaxf(swing, h[k]);
    float stop = dir > 0 ? swing - p.stop_buf : swing + p.stop_buf;
    const float risk = dir > 0 ? entry - stop : stop - entry;
    if (risk < kMinRisk) continue;

    const float tp1 = entry + dir * p.tp1_r * risk;
    const float tp2 = entry + dir * p.tp2_r * risk;
    bool full = true;  // both halves still open
    float r = 0.f;
    for (int k = j + 1; k < n; ++k) {
      const bool stop_hit = dir > 0 ? l[k] <= stop : h[k] >= stop;
      if (stop_hit) {
        const float exit = stop - dir * kStopSlip;
        const float rr = dir * (exit - entry) / risk;
        return r + (full ? rr : 0.5f * rr);
      }
      if (full && (dir > 0 ? h[k] >= tp1 : l[k] <= tp1)) {
        r += 0.5f * p.tp1_r;
        full = false;
        stop = entry;  // break-even for the runner
      }
      if (!full && (dir > 0 ? h[k] >= tp2 : l[k] <= tp2)) return r + 0.5f * p.tp2_r;
    }
    const float rr = dir * (c[n - 1] - entry) / risk;
    return r + (full ? rr : 0.5f * rr);
  }
  return no_trade();
}

// The same rules as simulate_session(), restructured as ONE loop over bars with a
// two-state machine (scanning for an entry / managing the open trade), for the GPU.
//
// Why: in simulate_session() trade management is a loop nested inside the scan loop.
// When lanes of one warp enter on different bars, the warp runs one lane group's whole
// management loop, then resumes the others' scan: the trades are serialised. Here
// every lane advances exactly one bar per iteration, so the warp reconverges each bar
// and costs ~(bars until its last lane finishes) instead of a sum of trade lengths.
//
// The arithmetic is written expression for expression like simulate_session(), so the
// result is bit-identical; tests and the CPU==GPU checksum hold the two to that.
//
// Measured result: SLOWER, 0.81x on both grids (results/ab-fused.txt), with the same
// register count. After the variant sort a warp's lanes mostly enter on the same bar,
// so the nested loops already reconverge, and the fused body pays its state check and
// both halves' code every bar. Off by default; gbt sweep --fused reproduces it.
GBT_HD inline float simulate_session_fused(const float* o, const float* h, const float* l,
                                           const float* c, const Params& p) {
  const int n = kSessionMinutes;

  int dir15 = 0;
  if (p.bias != 0) {
    const float d = c[14] - o[0];
    dir15 = d > 0.f ? 1 : (d < 0.f ? -1 : 0);
    if (dir15 == 0) return no_trade();
  }
  int start = p.win_start;
  if (p.bias != 0 && start < 15) start = 15;
  int end = p.win_start + p.win_len;  // exclusive
  if (end > n - 1) end = n - 1;

  bool bear_on = false, bull_on = false;
  float bear_top = 0.f, bull_bot = 0.f;

  bool in_trade = false;
  int dir = 0;
  float entry = 0.f, stop = 0.f, risk = 0.f, tp1 = 0.f, tp2 = 0.f, r = 0.f;
  bool full = true;

  for (int i = 2; i < n; ++i) {
    if (!in_trade) {
      if (i >= end) return no_trade();
      const int j = i;
      if (h[j] < l[j - 2] && l[j - 2] - h[j] >= p.min_gap) { bear_on = true; bear_top = l[j - 2]; }
      if (l[j] > h[j - 2] && l[j] - h[j - 2] >= p.min_gap) { bull_on = true; bull_bot = h[j - 2]; }
      int d = 0;
      if (bear_on && c[j] > bear_top) { d = 1; bear_on = false; }
      else if (bull_on && c[j] < bull_bot) { d = -1; bull_on = false; }
      if (d == 0 || j < start) continue;
      if (p.bias == 1 && d != dir15) continue;
      if (p.bias == 2 && d != -dir15) continue;
      const float e = c[j];
      int first = j - p.stop_lookback + 1;
      if (first < 0) first = 0;
      float swing = d > 0 ? l[j] : h[j];
      for (int k = first; k < j; ++k) swing = d > 0 ? fminf(swing, l[k]) : fmaxf(swing, h[k]);
      const float st = d > 0 ? swing - p.stop_buf : swing + p.stop_buf;
      const float rk = d > 0 ? e - st : st - e;
      if (rk < kMinRisk) continue;
      in_trade = true;
      dir = d;
      entry = e;
      stop = st;
      risk = rk;
      tp1 = entry + dir * p.tp1_r * risk;
      tp2 = entry + dir * p.tp2_r * risk;
      full = true;
      r = 0.f;
    } else {
      const int k = i;
      const bool stop_hit = dir > 0 ? l[k] <= stop : h[k] >= stop;
      if (stop_hit) {
        const float exit = stop - dir * kStopSlip;
        const float rr = dir * (exit - entry) / risk;
        return r + (full ? rr : 0.5f * rr);
      }
      if (full && (dir > 0 ? h[k] >= tp1 : l[k] <= tp1)) {
        r += 0.5f * p.tp1_r;
        full = false;
        stop = entry;
      }
      if (!full && (dir > 0 ? h[k] >= tp2 : l[k] <= tp2)) return r + 0.5f * p.tp2_r;
    }
  }
  if (!in_trade) return no_trade();
  const float rr = dir * (c[n - 1] - entry) / risk;
  return r + (full ? rr : 0.5f * rr);
}

// ---- per-variant statistics over the sessions, in date order ----
struct Metrics {
  float total_r = 0.f;
  float max_dd = 0.f;  // deepest peak-to-trough fall of the cumulative-R curve
  float sum_sq = 0.f;  // sum of squared daily R, for the volatility of the curve
  int32_t trades = 0;
  int32_t wins = 0;
};

// Streaming update: feed sessions in date order. `peak` and `equity` are running state.
GBT_HD inline void metrics_add(Metrics& m, float& equity, float& peak, float r) {
  if (!is_trade(r)) return;
  m.trades += 1;
  m.wins += r > 0.f ? 1 : 0;
  m.total_r += r;
  m.sum_sq += r * r;
  equity += r;
  peak = fmaxf(peak, equity);
  m.max_dd = fmaxf(m.max_dd, peak - equity);
}

// ---- parameter grids ----
enum class GridSize { Small, Default, Large };
std::vector<Params> make_grid(GridSize size);
GridSize parse_grid(const std::string& name);
std::string describe(const Params& p);

}  // namespace gbt
