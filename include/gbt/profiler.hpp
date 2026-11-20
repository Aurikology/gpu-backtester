#pragma once
// A small profiler, built rather than borrowed, because measuring is the point:
//
//   Timeline      named spans on a CPU track and a GPU track, exported as a Chrome
//                 trace (open it at https://ui.perfetto.dev). GPU spans come from
//                 device events, so they show real kernel time, not launch time.
//   bytes/GB/s    each span can carry the bytes it moved; the report divides by a
//                 *measured* peak bandwidth instead of the number on the box
//   PerfCounters  Linux hardware counters (cycles, instructions, cache misses,
//                 branch misses) through perf_event_open(2). Many VMs, including
//                 WSL2, hide the PMU; then the counters report "unavailable"
//                 rather than inventing numbers.

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace gbt {

struct Span {
  std::string name;
  std::string track;  // "cpu" or "gpu"
  double start_us = 0;
  double dur_us = 0;
  double bytes = 0;    // memory traffic, 0 if not meaningful
  double items = 0;    // work items (e.g. session simulations), 0 if not meaningful
};

class Timeline {
 public:
  Timeline();
  double now_us() const;  // microseconds since construction
  void add(Span s) { spans_.push_back(std::move(s)); }
  const std::vector<Span>& spans() const { return spans_; }
  bool write_chrome_trace(const std::string& path) const;
  // Text table: name, count, total ms, GB/s, % of peak (if peak_gbs > 0), items/s.
  std::string report(double peak_gbs) const;

 private:
  std::chrono::steady_clock::time_point t0_;
  std::vector<Span> spans_;
};

// RAII CPU span.
class CpuSpan {
 public:
  CpuSpan(Timeline* tl, std::string name, double bytes = 0, double items = 0);
  ~CpuSpan();

 private:
  Timeline* tl_;
  Span s_;
};

struct CounterValues {
  bool available = false;
  std::string why_unavailable;
  uint64_t cycles = 0, instructions = 0, cache_misses = 0, branch_misses = 0;
  double ipc() const { return cycles ? double(instructions) / double(cycles) : 0.0; }
};

class PerfCounters {
 public:
  PerfCounters();  // opens the counter group for this process (all threads it spawns)
  ~PerfCounters();
  void start();
  CounterValues stop();

 private:
  int fds_[4] = {-1, -1, -1, -1};
  bool ok_ = false;
  std::string why_;
};

}  // namespace gbt
