#include "gbt/montecarlo.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "gbt/engine.hpp"

namespace gbt {

static RiskOutcome summarise(float risk, const int* outcomes, int paths) {
  RiskOutcome o;
  o.risk = risk;
  long long pass = 0, fail = 0, days = 0;
  for (int i = 0; i < paths; ++i) {
    if (outcomes[i] > 0) { ++pass; days += outcomes[i]; }
    else if (outcomes[i] < 0) ++fail;
  }
  o.pass = double(pass) / paths;
  o.fail = double(fail) / paths;
  o.unresolved = 1.0 - o.pass - o.fail;
  o.mean_days_to_pass = pass ? double(days) / double(pass) : 0.0;
  return o;
}

std::vector<RiskOutcome> run_montecarlo(const std::vector<float>& r, const EvalRules& rules,
                                        const McOptions& opt) {
  if (r.empty()) throw std::invalid_argument("no R samples");
  if (opt.use_gpu && gpu_available()) return run_montecarlo_gpu(r, rules, opt);

  std::vector<RiskOutcome> out;
  std::vector<int> outcomes(size_t(opt.paths));
  const int nt = std::max(1, int(std::thread::hardware_concurrency()));
  for (size_t ri = 0; ri < opt.risks.size(); ++ri) {
    const float risk = opt.risks[ri];
    // Path ids are global across risk levels so a GPU run with the same seed matches.
    const uint64_t base = uint64_t(ri) * uint64_t(opt.paths);
    std::vector<std::thread> pool;
    for (int t = 0; t < nt; ++t) {
      pool.emplace_back([&, t] {
        for (int p = t; p < opt.paths; p += nt)
          outcomes[size_t(p)] = simulate_path(r.data(), int(r.size()), risk, rules, opt.seed,
                                              base + uint64_t(p));
      });
    }
    for (auto& th : pool) th.join();
    out.push_back(summarise(risk, outcomes.data(), opt.paths));
  }
  return out;
}

// Exposed for the GPU file, which reduces on the host the same way.
RiskOutcome summarise_outcomes(float risk, const int* outcomes, int paths) {
  return summarise(risk, outcomes, paths);
}

static std::vector<std::string> split_csv(const std::string& line) {
  // Minimal CSV: FX Replay exports have no quoted commas in the columns we read.
  std::vector<std::string> f;
  std::string cur;
  for (char ch : line) {
    if (ch == ',') { f.push_back(cur); cur.clear(); }
    else if (ch != '\r') cur += ch;
  }
  f.push_back(cur);
  return f;
}

std::vector<float> load_r_samples(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::string line;
  std::vector<float> r;
  if (!std::getline(in, line)) return r;
  auto header = split_csv(line);
  int col = -1, status = -1;
  for (size_t i = 0; i < header.size(); ++i) {
    if (header[i] == "avgRiskReward") col = int(i);
    if (header[i] == "status") status = int(i);
  }
  if (col < 0) {  // plain list; the first line is data too (or a header we skip)
    auto take = [&](const std::string& s) {
      try { r.push_back(std::stof(s)); } catch (...) {}
    };
    take(line);
    while (std::getline(in, line)) take(line);
    return r;
  }
  while (std::getline(in, line)) {
    auto f = split_csv(line);
    if (int(f.size()) <= col) continue;
    if (status >= 0 && int(f.size()) > status && f[size_t(status)] != "closed") continue;
    try { r.push_back(std::stof(f[size_t(col)])); } catch (...) {}
  }
  return r;
}

}  // namespace gbt
