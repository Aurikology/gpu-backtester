#include "gbt/profiler.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace gbt {

Timeline::Timeline() : t0_(std::chrono::steady_clock::now()) {}

double Timeline::now_us() const {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0_).count();
}

static std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    o += c;
  }
  return o;
}

bool Timeline::write_chrome_trace(const std::string& path) const {
  std::ofstream out(path);
  if (!out) return false;
  out << "{\"traceEvents\":[\n";
  out << R"({"name":"thread_name","ph":"M","pid":1,"tid":1,"args":{"name":"CPU"}},)" << "\n";
  out << R"({"name":"thread_name","ph":"M","pid":1,"tid":2,"args":{"name":"GPU stream"}})";
  for (const Span& s : spans_) {
    out << ",\n{\"name\":\"" << json_escape(s.name) << "\",\"ph\":\"X\",\"pid\":1,\"tid\":"
        << (s.track == "gpu" ? 2 : 1) << ",\"ts\":" << s.start_us << ",\"dur\":" << s.dur_us
        << ",\"args\":{\"bytes\":" << s.bytes << ",\"items\":" << s.items;
    if (s.dur_us > 0 && s.bytes > 0) out << ",\"GB/s\":" << s.bytes / (s.dur_us * 1e3);
    out << "}}";
  }
  out << "\n]}\n";
  return bool(out);
}

std::string Timeline::report(double peak_gbs) const {
  struct Agg {
    int n = 0;
    double us = 0, bytes = 0, items = 0;
    std::string track;
  };
  std::map<std::string, Agg> by;
  std::vector<std::string> order;
  for (const Span& s : spans_) {
    if (!by.count(s.name)) order.push_back(s.name);
    Agg& a = by[s.name];
    a.n++;
    a.us += s.dur_us;
    a.bytes += s.bytes;
    a.items += s.items;
    a.track = s.track;
  }
  std::ostringstream o;
  char line[256];
  std::snprintf(line, sizeof line, "%-28s %4s %5s %10s %9s %7s %12s\n", "span", "trk", "count",
                "total ms", "GB/s", "%peak", "items/s");
  o << line;
  for (const auto& name : order) {
    const Agg& a = by[name];
    const double gbs = a.us > 0 && a.bytes > 0 ? a.bytes / (a.us * 1e3) : 0;
    const double ips = a.us > 0 && a.items > 0 ? a.items / (a.us * 1e-6) : 0;
    char g[16] = "-", pk[16] = "-", it[24] = "-";
    if (gbs > 0) std::snprintf(g, sizeof g, "%.1f", gbs);
    if (gbs > 0 && peak_gbs > 0) std::snprintf(pk, sizeof pk, "%.0f%%", 100 * gbs / peak_gbs);
    if (ips > 0) std::snprintf(it, sizeof it, "%.3g", ips);
    std::snprintf(line, sizeof line, "%-28s %4s %5d %10.2f %9s %7s %12s\n", name.c_str(),
                  a.track.c_str(), a.n, a.us / 1e3, g, pk, it);
    o << line;
  }
  return o.str();
}

CpuSpan::CpuSpan(Timeline* tl, std::string name, double bytes, double items) : tl_(tl) {
  if (!tl_) return;
  s_.name = std::move(name);
  s_.track = "cpu";
  s_.bytes = bytes;
  s_.items = items;
  s_.start_us = tl_->now_us();
}

CpuSpan::~CpuSpan() {
  if (!tl_) return;
  s_.dur_us = tl_->now_us() - s_.start_us;
  tl_->add(s_);
}

// ---------- hardware counters ----------

PerfCounters::PerfCounters() {
#if defined(__linux__)
  const uint64_t configs[4] = {PERF_COUNT_HW_CPU_CYCLES, PERF_COUNT_HW_INSTRUCTIONS,
                               PERF_COUNT_HW_CACHE_MISSES, PERF_COUNT_HW_BRANCH_MISSES};
  for (int i = 0; i < 4; ++i) {
    perf_event_attr attr;
    std::memset(&attr, 0, sizeof attr);
    attr.size = sizeof attr;
    attr.type = PERF_TYPE_HARDWARE;
    attr.config = configs[i];
    attr.disabled = i == 0 ? 1 : 0;  // the group leader starts disabled
    attr.inherit = 1;                // count threads spawned after this point too
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    attr.read_format = 0;
    const int group = i == 0 ? -1 : fds_[0];
    // inherit is not allowed with PERF_FORMAT_GROUP reads, so each counter is read alone.
    fds_[i] = int(syscall(SYS_perf_event_open, &attr, 0, -1, group, 0));
    if (fds_[i] < 0) {
      why_ = std::string("perf_event_open: ") + std::strerror(errno) +
             (errno == ENOENT || errno == EOPNOTSUPP
                  ? " (no hardware PMU exposed, common in VMs and WSL2)"
                  : errno == EACCES ? " (raise access: sysctl kernel.perf_event_paranoid=1)" : "");
      for (int j = 0; j <= i; ++j)
        if (fds_[j] >= 0) close(fds_[j]), fds_[j] = -1;
      return;
    }
  }
  ok_ = true;
#else
  why_ = "hardware counters need Linux";
#endif
}

PerfCounters::~PerfCounters() {
#if defined(__linux__)
  for (int fd : fds_)
    if (fd >= 0) close(fd);
#endif
}

void PerfCounters::start() {
#if defined(__linux__)
  if (!ok_) return;
  ioctl(fds_[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
  ioctl(fds_[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
#endif
}

CounterValues PerfCounters::stop() {
  CounterValues v;
#if defined(__linux__)
  if (!ok_) {
    v.why_unavailable = why_;
    return v;
  }
  ioctl(fds_[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
  uint64_t vals[4] = {0, 0, 0, 0};
  for (int i = 0; i < 4; ++i)
    if (read(fds_[i], &vals[i], sizeof(uint64_t)) != sizeof(uint64_t)) vals[i] = 0;
  v.available = true;
  v.cycles = vals[0];
  v.instructions = vals[1];
  v.cache_misses = vals[2];
  v.branch_misses = vals[3];
#else
  v.why_unavailable = why_;
#endif
  return v;
}

}  // namespace gbt
