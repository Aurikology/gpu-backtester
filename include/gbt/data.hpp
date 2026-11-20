#pragma once
// Market data: one-minute OHLC bars cut into New York regular-hours sessions.
//
// Layout is structure-of-arrays (one array per field) rather than an array of bar
// structs. On a GPU, 32 neighbouring threads reading close[i..i+31] is one coalesced
// memory transaction; reading bars[i].close from a struct array is a strided gather.

#include <cstdint>
#include <string>
#include <vector>

#include "gbt/common.hpp"

namespace gbt {

struct Session {
  int32_t begin;  // index of the 09:30 bar in the bar arrays
  int32_t date;   // yyyymmdd, New York date
};

struct BarSeries {
  // Every session holds exactly kSessionMinutes bars, so bar k of session s lives at
  // sessions[s].begin + k. Missing minutes are forward-filled at load time.
  std::vector<float> open, high, low, close;
  std::vector<Session> sessions;

  size_t num_sessions() const { return sessions.size(); }
  size_t num_bars() const { return close.size(); }
};

struct LoadStats {
  size_t rows_read = 0;
  size_t rth_rows = 0;
  size_t sessions_kept = 0;
  size_t sessions_dropped = 0;  // half days, holidays, flat (no-trade) days
  size_t minutes_filled = 0;
};

// Parses a CSV with header time,open,high,low,close[,volume]. The time column may be
// Unix seconds, Unix milliseconds, or ISO-8601 UTC ("2026-09-15T02:47:00Z").
BarSeries load_csv(const std::string& path, LoadStats* stats = nullptr);

// Binary cache: loading 3.4M CSV rows takes seconds, the cache takes milliseconds.
void save_binary(const BarSeries& bars, const std::string& path);
BarSeries load_binary(const std::string& path);

// Loads .gbt as binary, anything else as CSV.
BarSeries load_any(const std::string& path, LoadStats* stats = nullptr);

// Synthetic random-walk sessions for tests, demos and CI (no market data in the repo).
BarSeries make_synthetic(int num_sessions, uint64_t seed);

// ---- time helpers (exposed for tests) ----
// Days since 1970-01-01 -> civil date, and back (Howard Hinnant's algorithms).
void civil_from_days(int64_t days, int* y, int* m, int* d);
int64_t days_from_civil(int y, int m, int d);
// UTC offset of New York, in hours (-4 in daylight time, -5 otherwise), for a UTC instant.
int ny_utc_offset_hours(int64_t unix_seconds);

}  // namespace gbt
