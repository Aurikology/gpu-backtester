#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>

#include "gbt/data.hpp"

using namespace gbt;

static int64_t utc(int y, int m, int d, int hh, int mm) {
  return days_from_civil(y, m, d) * 86400 + hh * 3600 + mm * 60;
}

TEST(Calendar, CivilRoundTrip) {
  for (int64_t z = -800000; z < 800000; z += 37) {
    int y, m, d;
    civil_from_days(z, &y, &m, &d);
    ASSERT_EQ(days_from_civil(y, m, d), z);
  }
  EXPECT_EQ(days_from_civil(1970, 1, 1), 0);
  EXPECT_EQ(days_from_civil(2024, 2, 29), 19782);
}

TEST(Calendar, NewYorkDaylightTimeBoundaries) {
  // 2024: DST began Sun Mar 10 at 07:00 UTC and ended Sun Nov 3 at 06:00 UTC.
  EXPECT_EQ(ny_utc_offset_hours(utc(2024, 3, 10, 6, 59)), -5);
  EXPECT_EQ(ny_utc_offset_hours(utc(2024, 3, 10, 7, 0)), -4);
  EXPECT_EQ(ny_utc_offset_hours(utc(2024, 11, 3, 5, 59)), -4);
  EXPECT_EQ(ny_utc_offset_hours(utc(2024, 11, 3, 6, 0)), -5);
  EXPECT_EQ(ny_utc_offset_hours(utc(2024, 7, 1, 12, 0)), -4);
  EXPECT_EQ(ny_utc_offset_hours(utc(2024, 1, 15, 12, 0)), -5);
}

namespace {
std::string tmp_path(const char* name) {
  return (std::filesystem::temp_directory_path() / name).string();
}

// One full regular session: 09:30 New York on a summer Monday is 13:30 UTC.
void write_session(std::ofstream& f, int y, int m, int d, int utc_hour, const char* format,
                   int skip_minute = -1, int drop_after = 390) {
  const int64_t t0 = utc(y, m, d, utc_hour, 30);
  for (int k = 0; k < drop_after; ++k) {
    if (k == skip_minute) continue;
    const int64_t t = t0 + k * 60;
    const float p = 100.f + float(k % 7);
    if (std::string(format) == "ms") f << t * 1000;
    else f << t;
    f << "," << p << "," << p + 1 << "," << p - 1 << "," << p + 0.5f << "\n";
  }
}
}  // namespace

TEST(Csv, BuildsSessionsInNewYorkTime) {
  const std::string path = tmp_path("gbt_test_bars.csv");
  {
    std::ofstream f(path);
    f << "time,open,high,low,close\n";
    write_session(f, 2024, 7, 1, 13, "s", /*skip_minute=*/100);  // EDT: 09:30 = 13:30 UTC
    write_session(f, 2024, 1, 8, 14, "ms");                       // EST: 09:30 = 14:30 UTC
    write_session(f, 2024, 7, 3, 13, "s", -1, 210);               // half day, ends 13:00
    // An overnight bar that must be ignored (20:00 New York).
    f << utc(2024, 7, 2, 0, 0) << ",1,1,1,1\n";
  }
  LoadStats st;
  BarSeries b = load_csv(path, &st);
  ASSERT_EQ(b.num_sessions(), 2u);
  EXPECT_EQ(b.sessions[0].date, 20240108);  // sorted by date
  EXPECT_EQ(b.sessions[1].date, 20240701);
  EXPECT_EQ(st.sessions_dropped, 1u);
  EXPECT_EQ(st.minutes_filled, 1u);
  EXPECT_EQ(b.num_bars(), 2u * kSessionMinutes);
  // The skipped minute (k=100) is a flat bar at the previous close.
  const size_t i = size_t(b.sessions[1].begin) + 100;
  EXPECT_EQ(b.open[i], b.close[i - 1]);
  EXPECT_EQ(b.high[i], b.low[i]);
  // First bar of the session is the 09:30 bar (k=0 -> p=100).
  EXPECT_FLOAT_EQ(b.open[size_t(b.sessions[0].begin)], 100.f);
  std::filesystem::remove(path);
}

TEST(Csv, BinaryCacheRoundTrips) {
  BarSeries a = make_synthetic(20, 5);
  const std::string path = tmp_path("gbt_test.gbt");
  save_binary(a, path);
  BarSeries b = load_binary(path);
  EXPECT_EQ(a.close, b.close);
  EXPECT_EQ(a.high, b.high);
  ASSERT_EQ(a.sessions.size(), b.sessions.size());
  EXPECT_EQ(a.sessions.back().date, b.sessions.back().date);
  std::filesystem::remove(path);
}

TEST(Synthetic, BarsAreConsistent) {
  BarSeries b = make_synthetic(50, 3);
  for (size_t i = 0; i < b.num_bars(); ++i) {
    ASSERT_LE(b.low[i], std::min(b.open[i], b.close[i]));
    ASSERT_GE(b.high[i], std::max(b.open[i], b.close[i]));
  }
}
