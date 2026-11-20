#include "gbt/strategy.hpp"

#include <cstdio>
#include <stdexcept>

namespace gbt {

namespace {
struct Axes {
  std::vector<int> win_start, win_len, stop_lookback, bias;
  std::vector<float> min_gap, stop_buf, tp1_r, tp2_r;
};

Axes axes_for(GridSize size) {
  switch (size) {
    case GridSize::Small:  // 2*2*2*3*2*2 * 3 valid target pairs = 288
      return {{0, 15}, {60, 180}, {3, 5}, {0, 1, 2}, {2, 6}, {0, 2}, {1, 2}, {2, 4}};
    case GridSize::Default:  // 152,064 variants
      return {{0, 5, 10, 15, 20, 30, 45, 60},
              {30, 60, 90, 120, 180, 240},
              {2, 3, 5, 8},
              {0, 1, 2},
              {0, 2, 4, 6, 8, 12},
              {0, 1, 2, 4},
              {1, 1.5f, 2},
              {2, 3, 4, 6}};
    case GridSize::Large:  // 855,360 variants
      return {{0, 5, 10, 15, 20, 25, 30, 45, 60, 90},
              {15, 30, 45, 60, 90, 120, 180, 240, 300},
              {2, 3, 4, 5, 8, 12},
              {0, 1, 2},
              {0, 1, 2, 3, 4, 6, 8, 12},
              {0, 0.5f, 1, 2, 3, 4},
              {1, 1.5f, 2},
              {2, 3, 4, 6}};
  }
  throw std::logic_error("bad grid");
}
}  // namespace

std::vector<Params> make_grid(GridSize size) {
  const Axes a = axes_for(size);
  std::vector<Params> g;
  for (int ws : a.win_start)
    for (int wl : a.win_len)
      for (float mg : a.min_gap)
        for (int lb : a.stop_lookback)
          for (float sb : a.stop_buf)
            for (float t1 : a.tp1_r)
              for (float t2 : a.tp2_r)
                for (int bi : a.bias)
                  if (t2 > t1) g.push_back(Params{ws, wl, mg, lb, sb, t1, t2, bi});
  return g;
}

GridSize parse_grid(const std::string& name) {
  if (name == "small") return GridSize::Small;
  if (name == "default") return GridSize::Default;
  if (name == "large") return GridSize::Large;
  throw std::invalid_argument("grid must be small, default or large");
}

std::string describe(const Params& p) {
  static const char* bias[] = {"any", "with-15m", "against-15m"};
  char buf[160];
  std::snprintf(buf, sizeof buf, "win %d+%dm gap>=%.0f stop %db+%.0f tp %.1fR/%.1fR bias %s",
                p.win_start, p.win_len, p.min_gap, p.stop_lookback, p.stop_buf, p.tp1_r, p.tp2_r,
                bias[p.bias]);
  return buf;
}

}  // namespace gbt
