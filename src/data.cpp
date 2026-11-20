#include "gbt/data.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string_view>

namespace gbt {

// ---------- calendar ----------

// Howard Hinnant, "chrono-Compatible Low-Level Date Algorithms".
int64_t days_from_civil(int y, int m, int d) {
  y -= m <= 2;
  const int64_t era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = unsigned(y - era * 400);
  const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + int64_t(doe) - 719468;
}

void civil_from_days(int64_t z, int* y, int* m, int* d) {
  z += 719468;
  const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
  const unsigned doe = unsigned(z - era * 146097);
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const int64_t yy = int64_t(yoe) + era * 400;
  const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const unsigned mp = (5 * doy + 2) / 153;
  *d = int(doy - (153 * mp + 2) / 5 + 1);
  *m = int(mp < 10 ? mp + 3 : mp - 9);
  *y = int(yy + (*m <= 2));
}

static int weekday(int64_t days) {  // 0 = Sunday
  return int((days % 7 + 11) % 7);  // 1970-01-01 was a Thursday (4)
}

// n-th Sunday (1-based) of a month, as days since epoch.
static int64_t nth_sunday(int y, int m, int n) {
  const int64_t first = days_from_civil(y, m, 1);
  const int64_t first_sun = first + (7 - weekday(first)) % 7;
  return first_sun + 7 * (n - 1);
}

// US daylight time since 2007: second Sunday of March 02:00 local (07:00 UTC) to
// first Sunday of November 02:00 local (06:00 UTC).
int ny_utc_offset_hours(int64_t t) {
  const int64_t days = t >= 0 ? t / 86400 : (t - 86399) / 86400;
  int y, m, d;
  civil_from_days(days, &y, &m, &d);
  const int64_t start = nth_sunday(y, 3, 2) * 86400 + 7 * 3600;
  const int64_t end = nth_sunday(y, 11, 1) * 86400 + 6 * 3600;
  return (t >= start && t < end) ? -4 : -5;
}

// ---------- CSV parsing ----------

static bool parse_float(std::string_view s, float* out) {
  // std::from_chars for floats needs GCC 11+, which Ubuntu 24.04 has.
  auto res = std::from_chars(s.data(), s.data() + s.size(), *out);
  return res.ec == std::errc();
}

static bool parse_time(std::string_view s, int64_t* unix_seconds) {
  if (s.size() >= 19 && s[4] == '-' && s[10] == 'T') {  // 2026-09-15T02:47:00Z
    auto num = [&](size_t pos, size_t len) {
      int v = 0;
      std::from_chars(s.data() + pos, s.data() + pos + len, v);
      return v;
    };
    const int64_t days = days_from_civil(num(0, 4), num(5, 2), num(8, 2));
    *unix_seconds = days * 86400 + num(11, 2) * 3600 + num(14, 2) * 60 + num(17, 2);
    return true;  // assumes UTC ("Z"); offsets are not used by the supported sources
  }
  int64_t v = 0;
  auto res = std::from_chars(s.data(), s.data() + s.size(), v);
  if (res.ec != std::errc()) return false;
  *unix_seconds = v > 100000000000LL ? v / 1000 : v;  // milliseconds vs seconds
  return true;
}

namespace {
struct Row {
  float o, h, l, c;
};
}  // namespace

BarSeries load_csv(const std::string& path, LoadStats* stats) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::string line;
  std::getline(in, line);  // header

  // date (yyyymmdd) -> minute-of-session -> bar
  std::map<int32_t, std::vector<Row>> days;
  std::map<int32_t, std::vector<uint8_t>> seen;
  LoadStats st;

  std::string_view f[6];
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    ++st.rows_read;
    int nf = 0;
    size_t pos = 0;
    while (nf < 6) {
      size_t comma = line.find(',', pos);
      f[nf++] = std::string_view(line).substr(pos, comma == std::string::npos ? std::string::npos
                                                                              : comma - pos);
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
    if (nf < 5) continue;
    int64_t t;
    Row r;
    if (!parse_time(f[0], &t) || !parse_float(f[1], &r.o) || !parse_float(f[2], &r.h) ||
        !parse_float(f[3], &r.l) || !parse_float(f[4], &r.c))
      continue;

    const int64_t local = t + int64_t(ny_utc_offset_hours(t)) * 3600;
    const int64_t days_local = local >= 0 ? local / 86400 : (local - 86399) / 86400;
    const int minute_of_day = int((local - days_local * 86400) / 60);
    const int k = minute_of_day - (9 * 60 + 30);
    if (k < 0 || k >= kSessionMinutes) continue;
    const int wd = weekday(days_local);
    if (wd == 0 || wd == 6) continue;
    ++st.rth_rows;

    int y, m, d;
    civil_from_days(days_local, &y, &m, &d);
    const int32_t date = y * 10000 + m * 100 + d;
    auto& bars = days[date];
    auto& mask = seen[date];
    if (bars.empty()) {
      bars.resize(kSessionMinutes);
      mask.assign(kSessionMinutes, 0);
    }
    bars[k] = r;
    mask[k] = 1;
  }

  BarSeries out;
  for (auto& [date, bars] : days) {
    auto& mask = seen[date];
    int present = 0;
    for (uint8_t b : mask) present += b;
    // Keep full days only: holidays and 13:00 half days would skew a 16:00 exit.
    // A flat day (no range at all) is a data-vendor placeholder, not a market.
    float hi = -INFINITY, lo = INFINITY;
    for (int k = 0; k < kSessionMinutes; ++k)
      if (mask[k]) { hi = std::max(hi, bars[k].h); lo = std::min(lo, bars[k].l); }
    if (present < 370 || !mask[0] || !(hi > lo)) {
      ++st.sessions_dropped;
      continue;
    }
    // Forward-fill the few missing minutes with a flat bar at the previous close.
    for (int k = 1; k < kSessionMinutes; ++k) {
      if (!mask[k]) {
        const float c = bars[k - 1].c;
        bars[k] = {c, c, c, c};
        ++st.minutes_filled;
      }
    }
    out.sessions.push_back({int32_t(out.close.size()), date});
    for (const Row& r : bars) {
      out.open.push_back(r.o);
      out.high.push_back(r.h);
      out.low.push_back(r.l);
      out.close.push_back(r.c);
    }
    ++st.sessions_kept;
  }
  if (stats) *stats = st;
  return out;
}

// ---------- binary cache ----------

static constexpr char kMagic[8] = {'G', 'B', 'T', 'B', 'A', 'R', 'S', '1'};

void save_binary(const BarSeries& b, const std::string& path) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + path);
  const uint64_t ns = b.sessions.size(), nb = b.close.size();
  out.write(kMagic, 8);
  out.write(reinterpret_cast<const char*>(&ns), 8);
  out.write(reinterpret_cast<const char*>(&nb), 8);
  out.write(reinterpret_cast<const char*>(b.sessions.data()), ns * sizeof(Session));
  for (const auto* v : {&b.open, &b.high, &b.low, &b.close})
    out.write(reinterpret_cast<const char*>(v->data()), nb * sizeof(float));
}

BarSeries load_binary(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  char magic[8];
  uint64_t ns = 0, nb = 0;
  in.read(magic, 8);
  if (std::memcmp(magic, kMagic, 8) != 0) throw std::runtime_error(path + " is not a .gbt file");
  in.read(reinterpret_cast<char*>(&ns), 8);
  in.read(reinterpret_cast<char*>(&nb), 8);
  if (nb != ns * kSessionMinutes) throw std::runtime_error(path + ": corrupt header");
  BarSeries b;
  b.sessions.resize(ns);
  in.read(reinterpret_cast<char*>(b.sessions.data()), ns * sizeof(Session));
  for (auto* v : {&b.open, &b.high, &b.low, &b.close}) {
    v->resize(nb);
    in.read(reinterpret_cast<char*>(v->data()), nb * sizeof(float));
  }
  if (!in) throw std::runtime_error(path + ": truncated");
  return b;
}

BarSeries load_any(const std::string& path, LoadStats* stats) {
  if (path.size() > 4 && path.compare(path.size() - 4, 4, ".gbt") == 0) return load_binary(path);
  return load_csv(path, stats);
}

// ---------- synthetic data ----------

BarSeries make_synthetic(int num_sessions, uint64_t seed) {
  BarSeries b;
  uint64_t s = seed ? seed : 1;
  auto next = [&]() {  // xorshift64*
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 2685821657736338717ull;
  };
  auto gauss = [&]() {  // sum of uniforms: cheap, roughly normal
    double acc = 0;
    for (int i = 0; i < 4; ++i) acc += double(next() >> 11) * (1.0 / 9007199254740992.0);
    return (acc - 2.0) * 1.7320508;
  };
  float price = 15000.f;
  int64_t day = days_from_civil(2020, 1, 6);  // a Monday
  for (int sidx = 0; sidx < num_sessions; ++sidx) {
    while (weekday(day) == 0 || weekday(day) == 6) ++day;
    int y, m, d;
    civil_from_days(day, &y, &m, &d);
    b.sessions.push_back({int32_t(b.close.size()), y * 10000 + m * 100 + d});
    const double vol = 4.0 + 3.0 * double(next() >> 11) * (1.0 / 9007199254740992.0);
    for (int k = 0; k < kSessionMinutes; ++k) {
      // Round to the 0.25 tick so prices look like NQ.
      const float o = price;
      float c = float(std::round((o + vol * gauss()) * 4.0) / 4.0);
      const float wick = float(std::round(std::fabs(vol * gauss()) * 2.0) / 4.0);
      b.open.push_back(o);
      b.close.push_back(c);
      b.high.push_back(std::max(o, c) + wick);
      b.low.push_back(std::min(o, c) - wick);
      price = c;
    }
    ++day;
  }
  return b;
}

}  // namespace gbt
