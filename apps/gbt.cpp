// gbt: command-line front end.
//
//   gbt info                         GPU, measured bandwidth, CPU threads
//   gbt data  <csv> [--out f.gbt]    parse bars into sessions, cache as binary
//   gbt sweep --data f.gbt ...       run a parameter grid, rank in-sample, show out-of-sample
//   gbt bench --data f.gbt ...       same work on every backend: speed + exact agreement
//   gbt bench-prims                  hand-written reduce/scan vs hipCUB
//   gbt lucid --r trades.csv ...     prop-firm evaluation pass odds by dollar risk
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "gbt/data.hpp"
#include "gbt/engine.hpp"
#include "gbt/gpu_info.hpp"
#include "gbt/montecarlo.hpp"
#include "gbt/profiler.hpp"
#include "gbt/strategy.hpp"

using namespace gbt;

namespace {

struct Args {
  std::string cmd;
  std::vector<std::string> pos;
  std::map<std::string, std::string> kv;
  bool has(const std::string& k) const { return kv.count(k) > 0; }
  std::string str(const std::string& k, const std::string& d) const {
    auto it = kv.find(k);
    return it == kv.end() ? d : it->second;
  }
  double num(const std::string& k, double d) const {
    auto it = kv.find(k);
    return it == kv.end() ? d : std::stod(it->second);
  }
};

Args parse(int argc, char** argv) {
  Args a;
  if (argc > 1) a.cmd = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string s = argv[i];
    if (s.rfind("--", 0) == 0) {
      const std::string key = s.substr(2);
      if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) a.kv[key] = argv[++i];
      else a.kv[key] = "1";
    } else {
      a.pos.push_back(s);
    }
  }
  return a;
}

double seconds_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

BarSeries load_data(const Args& a) {
  const std::string path = a.str("data", "");
  if (path.empty() || path == "synthetic") {
    const int n = int(a.num("sessions", 1900));
    std::fprintf(stderr, "using %d synthetic sessions (pass --data for real bars)\n", n);
    return make_synthetic(n, 12345);
  }
  return load_any(path);
}

std::string date_str(int32_t d) {
  char b[16];
  std::snprintf(b, sizeof b, "%04d-%02d-%02d", d / 10000, d / 100 % 100, d % 100);
  return b;
}

// ---------------------------------------------------------------------------

int cmd_info(const Args&) {
  std::printf("GPU:          %s\n", gpu_device_name().c_str());
  if (gpu_available()) std::printf("GPU peak:     %.1f GB/s (measured device copy)\n", gpu_measured_peak_gbs());
  std::printf("CPU threads:  %u\n", std::thread::hardware_concurrency());
  PerfCounters pc;
  pc.start();
  volatile double x = 0;
  for (int i = 0; i < 1000000; ++i) x += i * 0.5;
  CounterValues cv = pc.stop();
  if (cv.available) std::printf("HW counters:  available (IPC %.2f on a test loop)\n", cv.ipc());
  else std::printf("HW counters:  unavailable: %s\n", cv.why_unavailable.c_str());
  return 0;
}

int cmd_data(const Args& a) {
  if (a.pos.empty()) throw std::invalid_argument("usage: gbt data <bars.csv> [--out bars.gbt]");
  const auto t = std::chrono::steady_clock::now();
  LoadStats st;
  BarSeries b = load_csv(a.pos[0], &st);
  std::printf("read %zu rows in %.1fs: %zu in regular hours\n", st.rows_read, seconds_since(t), st.rth_rows);
  std::printf("sessions kept %zu, dropped %zu (holidays, half days, flat placeholder days)\n",
              st.sessions_kept, st.sessions_dropped);
  std::printf("missing minutes forward-filled: %zu (%.3f%% of kept bars)\n", st.minutes_filled,
              b.num_bars() ? 100.0 * double(st.minutes_filled) / double(b.num_bars()) : 0.0);
  if (!b.sessions.empty())
    std::printf("range %s to %s\n", date_str(b.sessions.front().date).c_str(),
                date_str(b.sessions.back().date).c_str());
  if (a.has("out")) {
    save_binary(b, a.str("out", ""));
    std::printf("wrote %s\n", a.str("out", "").c_str());
  }
  return 0;
}

// Recovery factor: in-sample R earned per R of worst drawdown. Ranking on total R alone
// rewards variants that make money by surviving one huge hole.
double score(const Metrics& m, int min_trades) {
  if (m.trades < min_trades) return -1e30;
  return double(m.total_r) / std::max(1.0, double(m.max_dd));
}

int cmd_sweep(const Args& a) {
  BarSeries bars = load_data(a);
  const auto grid = make_grid(parse_grid(a.str("grid", "default")));
  SweepOptions opt;
  opt.backend = parse_backend(a.str("backend", gpu_available() ? "gpu" : "cpu"));
  opt.oos_fraction = a.num("oos", 0.3);
  opt.block_size = int(a.num("block", 128));
  Timeline tl;
  opt.timeline = &tl;
  const size_t split = oos_split(bars.num_sessions(), opt.oos_fraction);

  std::printf("%zu sessions (%s to %s), %zu variants, backend %s\n", bars.num_sessions(),
              date_str(bars.sessions.front().date).c_str(),
              date_str(bars.sessions.back().date).c_str(), grid.size(), backend_name(opt.backend));
  std::printf("in-sample: first %zu sessions (to %s); out-of-sample: last %zu, never used to rank\n",
              split, date_str(bars.sessions[split - 1].date).c_str(), bars.num_sessions() - split);

  const auto t = std::chrono::steady_clock::now();
  auto res = run_sweep(bars, grid, opt);
  const double secs = seconds_since(t);
  const double sims = double(grid.size()) * double(bars.num_sessions());
  std::printf("%.3g session simulations in %.2fs = %.3g per second\n\n", sims, secs, sims / secs);

  const int min_trades = int(a.num("min-trades", 50));
  std::vector<size_t> idx(res.size());
  std::iota(idx.begin(), idx.end(), size_t(0));
  std::sort(idx.begin(), idx.end(), [&](size_t x, size_t y) {
    return score(res[x].in_sample, min_trades) > score(res[y].in_sample, min_trades);
  });

  const int top = int(a.num("top", 15));
  std::printf("%-4s %-62s | %6s %6s %5s %5s | %6s %6s %5s\n", "rank", "variant", "IS R", "IS DD",
              "trd", "win%", "OOS R", "OOS DD", "trd");
  int shown = 0, oos_positive = 0;
  for (size_t k = 0; k < idx.size() && shown < top; ++k) {
    const auto& r = res[idx[k]];
    if (r.in_sample.trades < min_trades) break;
    const auto& is = r.in_sample;
    const auto& os = r.out_of_sample;
    std::printf("%-4d %-62s | %6.1f %6.1f %5d %4.0f%% | %6.1f %6.1f %5d\n", shown + 1,
                describe(grid[idx[k]]).c_str(), is.total_r, is.max_dd, is.trades,
                100.0 * is.wins / std::max(1, is.trades), os.total_r, os.max_dd, os.trades);
    oos_positive += os.total_r > 0 ? 1 : 0;
    ++shown;
  }
  if (shown == 0) std::printf("(no variant reached %d in-sample trades)\n", min_trades);
  else
    std::printf("\n%d of the top %d stayed profitable out of sample. The best in-sample row is\n"
                "the most overfit one by construction: judge a rule by its OOS column.\n",
                oos_positive, shown);

  if (a.has("save-best-r") && shown > 0) {
    const auto r = trades_of(bars, grid[idx[0]]);
    std::ofstream f(a.str("save-best-r", ""));
    for (float x : r) f << x << "\n";
    std::printf("wrote %zu per-trade R values of rank 1 to %s (feed it to `gbt lucid --r`)\n",
                r.size(), a.str("save-best-r", "").c_str());
  }
  if (a.has("trace")) {
    tl.write_chrome_trace(a.str("trace", ""));
    std::printf("trace: %s (open in https://ui.perfetto.dev)\n", a.str("trace", "").c_str());
  }
  std::printf("\n%s", tl.report(gpu_available() ? gpu_measured_peak_gbs() : 0).c_str());
  return 0;
}

// Every backend runs the same subset of the grid; checksums must match to the bit.
int cmd_bench(const Args& a) {
  BarSeries bars = load_data(a);
  auto full = make_grid(parse_grid(a.str("grid", "default")));
  const size_t want = size_t(a.num("variants", 4096));
  std::vector<Params> grid;
  const size_t step = std::max<size_t>(1, full.size() / want);  // spread across the grid
  for (size_t i = 0; i < full.size() && grid.size() < want; i += step) grid.push_back(full[i]);

  std::vector<std::string> names = {"cpu1", "cpu", "gpu-naive", "gpu"};
  if (a.has("backends")) {
    names.clear();
    std::string s = a.str("backends", "");
    size_t p = 0;
    while (p <= s.size()) {
      size_t q = s.find(',', p);
      names.push_back(s.substr(p, q == std::string::npos ? std::string::npos : q - p));
      if (q == std::string::npos) break;
      p = q + 1;
    }
  }
  const bool json = a.has("json");
  const double sims = double(grid.size()) * double(bars.num_sessions());
  if (!json)
    std::printf("%zu variants x %zu sessions = %.3g session simulations per backend\n\n", grid.size(),
                bars.num_sessions(), sims);

  double base = 0;
  uint64_t ref_sum = 0;
  bool all_match = true;
  if (!json)
    std::printf("%-10s %10s %14s %9s %18s  %s\n", "backend", "seconds", "sims/second", "speed-up",
                "checksum", "CPU counters");
  for (const auto& name : names) {
    SweepOptions opt;
    opt.backend = parse_backend(name);
    opt.block_size = int(a.num("block", 128));
    if ((opt.backend == Backend::GpuNaive || opt.backend == Backend::GpuShared) && !gpu_available()) {
      if (!json) std::printf("%-10s (no GPU)\n", name.c_str());
      continue;
    }
    PerfCounters pc;
    if (opt.backend == Backend::GpuShared || opt.backend == Backend::GpuNaive)
      run_sweep(bars, grid, opt);  // warm-up: first launch pays context + module load
    pc.start();
    const auto t = std::chrono::steady_clock::now();
    auto res = run_sweep(bars, grid, opt);
    const double secs = seconds_since(t);
    CounterValues cv = pc.stop();
    const uint64_t sum = checksum(res);
    if (base == 0) { base = secs; ref_sum = sum; }
    all_match = all_match && sum == ref_sum;
    if (json) {
      std::printf("{\"backend\":\"%s\",\"block\":%d,\"seconds\":%.6f,\"sims_per_s\":%.6g,"
                  "\"checksum\":\"%016llx\"}\n",
                  name.c_str(), opt.block_size, secs, sims / secs, (unsigned long long)sum);
      continue;
    }
    char counters[96] = "-";
    if (opt.backend == Backend::Cpu1 || opt.backend == Backend::CpuN) {
      if (cv.available)
        std::snprintf(counters, sizeof counters, "IPC %.2f, %.1f%% branch-miss/instr",
                      cv.ipc(), 100.0 * double(cv.branch_misses) / double(std::max<uint64_t>(1, cv.instructions)));
      else
        std::snprintf(counters, sizeof counters, "unavailable");
    }
    std::printf("%-10s %10.3f %14.3g %8.1fx %18llx  %s\n", name.c_str(), secs, sims / secs,
                base / secs, (unsigned long long)sum, counters);
  }
  if (!json)
    std::printf("\nall backends agree to the bit: %s\n", all_match ? "yes" : "NO (bug)");
  return all_match ? 0 : 2;
}

int cmd_bench_prims(const Args& a) {
  std::vector<size_t> sizes = {size_t(1) << 16, size_t(1) << 20, size_t(1) << 24, size_t(1) << 26};
  std::printf("GPU: %s\n%s", gpu_device_name().c_str(), bench_prims(sizes, int(a.num("reps", 20))).c_str());
  return 0;
}

std::vector<float> parse_range(const std::string& s) {  // "lo:hi:step" or "a,b,c"
  std::vector<float> v;
  if (s.find(':') != std::string::npos) {
    float lo, hi, step;
    if (std::sscanf(s.c_str(), "%f:%f:%f", &lo, &hi, &step) != 3 || step <= 0)
      throw std::invalid_argument("--risk lo:hi:step");
    for (float x = lo; x <= hi + 1e-3f; x += step) v.push_back(x);
  } else {
    size_t p = 0;
    while (p < s.size()) {
      size_t q = s.find(',', p);
      v.push_back(std::stof(s.substr(p, q == std::string::npos ? std::string::npos : q - p)));
      if (q == std::string::npos) break;
      p = q + 1;
    }
  }
  return v;
}

int cmd_lucid(const Args& a) {
  if (!a.has("r")) throw std::invalid_argument("usage: gbt lucid --r <trades.csv | r-list.txt> [options]");
  const auto r = load_r_samples(a.str("r", ""));
  if (r.empty()) throw std::runtime_error("no R values found");
  EvalRules rules;
  rules.profit_target = float(a.num("target", 3000));
  rules.max_drawdown = float(a.num("max-dd", 2000));
  rules.trailing = !a.has("static");
  rules.lock_at_start = !a.has("no-lock");
  rules.daily_loss_limit = float(a.num("daily-limit", 0));
  rules.cost_per_trade = float(a.num("cost", 0));
  rules.max_days = int(a.num("days", 60));
  McOptions opt;
  opt.risks = parse_range(a.str("risk", "50:1000:50"));
  opt.paths = int(a.num("paths", 100000));
  opt.seed = uint64_t(a.num("seed", 42));
  opt.use_gpu = !a.has("cpu");

  double mean = 0, wins = 0;
  for (float x : r) { mean += x; wins += x > 0; }
  mean /= double(r.size());
  std::printf("%zu trades in the sample: mean %+.2fR, win rate %.0f%%\n", r.size(), mean,
              100.0 * wins / double(r.size()));
  std::printf("rules: target $%.0f, %s drawdown $%.0f%s, %d days max, cost $%.2f/trade%s\n",
              rules.profit_target, rules.trailing ? "trailing" : "static", rules.max_drawdown,
              rules.trailing && rules.lock_at_start ? " (locks at start balance)" : "", rules.max_days,
              rules.cost_per_trade, rules.daily_loss_limit > 0 ? ", daily loss limit on" : "");
  std::printf("%d simulated attempts per risk level on the %s, one trade a day\n\n", opt.paths,
              opt.use_gpu && gpu_available() ? "GPU" : "CPU");

  const auto t = std::chrono::steady_clock::now();
  auto res = run_montecarlo(r, rules, opt);
  const double secs = seconds_since(t);

  size_t best = 0;
  for (size_t i = 0; i < res.size(); ++i)
    if (res[i].pass > res[best].pass) best = i;
  std::printf("%10s %8s %8s %11s %14s\n", "risk/trade", "pass", "fail", "unresolved", "days to pass");
  for (size_t i = 0; i < res.size(); ++i) {
    const auto& o = res[i];
    std::printf("%9.0f$ %7.1f%% %7.1f%% %10.1f%% %14.1f%s\n", o.risk, 100 * o.pass, 100 * o.fail,
                100 * o.unresolved, o.mean_days_to_pass, i == best ? "   <- best" : "");
  }
  // 95% binomial interval on the best pass rate, so the "best" is not read as exact.
  const double p = res[best].pass, se = std::sqrt(p * (1 - p) / opt.paths);
  std::printf("\nbest: risk $%.0f per trade -> %.1f%% pass (+/- %.1f%% simulation error)\n",
              res[best].risk, 100 * p, 196 * se);
  std::printf("caveat: assumes future trades look like this sample. %zu trades is %s.\n", r.size(),
              r.size() < 100 ? "a small sample: treat the curve's shape as the result, not the digits"
                             : "a usable sample, but regimes change");
  std::printf("%.3g paths in %.2fs\n", double(opt.paths) * double(res.size()), secs);
  return 0;
}

void usage() {
  std::puts(
      "gbt - GPU backtester for NQ intraday strategies\n\n"
      "  gbt info\n"
      "  gbt data <bars.csv> [--out bars.gbt]\n"
      "  gbt sweep [--data bars.gbt] [--grid small|default|large] [--backend cpu1|cpu|gpu-naive|gpu]\n"
      "            [--oos 0.3] [--top 15] [--min-trades 50] [--trace trace.json] [--save-best-r r.txt]\n"
      "  gbt bench [--data bars.gbt] [--variants 4096] [--backends cpu1,cpu,gpu-naive,gpu] [--block 128] [--json]\n"
      "  gbt bench-prims [--reps 20]\n"
      "  gbt lucid --r <trades.csv|r.txt> [--target 3000] [--max-dd 2000] [--static] [--no-lock]\n"
      "            [--daily-limit 0] [--cost 0] [--days 60] [--risk 50:1000:50] [--paths 100000] [--cpu]\n\n"
      "Without --data, sweep and bench use synthetic sessions.");
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse(argc, argv);
  try {
    if (a.cmd == "info") return cmd_info(a);
    if (a.cmd == "data") return cmd_data(a);
    if (a.cmd == "sweep") return cmd_sweep(a);
    if (a.cmd == "bench") return cmd_bench(a);
    if (a.cmd == "bench-prims") return cmd_bench_prims(a);
    if (a.cmd == "lucid") return cmd_lucid(a);
    usage();
    return a.cmd.empty() || a.cmd == "help" || a.cmd == "--help" ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
