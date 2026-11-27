// GPU sweep and Monte Carlo kernels.
#include <algorithm>
#include <cstring>
#include <optional>
#include <tuple>

#include "gbt/engine.hpp"
#include "gbt/gpu_info.hpp"
#include "gbt/montecarlo.hpp"
#include "gpu_runtime.cuh"

namespace gbt {

RiskOutcome summarise_outcomes(float risk, const int* outcomes, int paths);  // montecarlo.cpp

bool gpu_available() {
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}

std::string gpu_device_name() {
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) return "none";
  return prop.name;
}

// ---------------------------------------------------------------------------
// Sweep kernels
// ---------------------------------------------------------------------------

// v1: one thread per (variant, session). Neighbouring threads in a warp share a
// variant but read different sessions, so their bar reads are 1.5 KB apart: every
// load is a separate memory transaction, and each session's bars are fetched again by
// every variant (V times in total). Output is variant-major: out[v * S + s].
__global__ void sim_naive(const float* __restrict__ o, const float* __restrict__ h,
                          const float* __restrict__ l, const float* __restrict__ c,
                          const Session* __restrict__ sess, int S,
                          const Params* __restrict__ grid, int V, float* __restrict__ out) {
  const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size_t(V) * size_t(S)) return;
  const int v = int(idx / size_t(S));
  const int s = int(idx % size_t(S));
  const size_t b = size_t(sess[s].begin);
  out[idx] = simulate_session(o + b, h + b, l + b, c + b, grid[v]);
}

// v2: one block per (session, chunk of variants). The block copies its session's 390
// bars into on-chip shared memory once (6 KB), then every thread simulates a different
// variant against that copy. All lanes of a warp read the same bar at the same time,
// which shared memory serves as a broadcast. Output is session-major:
// out[s * V + v], so a warp's 32 writes land in one contiguous 128-byte line.
// Fused=true uses simulate_session_fused() (one loop over bars, lanes reconverge every
// bar); false uses the nested-loop simulate_session() that the CPU reference runs.
template <bool Fused>
__global__ void sim_shared(const float* __restrict__ o, const float* __restrict__ h,
                           const float* __restrict__ l, const float* __restrict__ c,
                           const Session* __restrict__ sess, const Params* __restrict__ grid,
                           int V, float* __restrict__ out) {
  __shared__ float so[kSessionMinutes], sh[kSessionMinutes], sl[kSessionMinutes],
      sc[kSessionMinutes];
  const int s = blockIdx.y;
  const size_t b = size_t(sess[s].begin);
  for (int k = threadIdx.x; k < kSessionMinutes; k += blockDim.x) {
    so[k] = o[b + k];
    sh[k] = h[b + k];
    sl[k] = l[b + k];
    sc[k] = c[b + k];
  }
  __syncthreads();
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= V) return;
  out[size_t(s) * size_t(V) + size_t(v)] = Fused ? simulate_session_fused(so, sh, sl, sc, grid[v])
                                                 : simulate_session(so, sh, sl, sc, grid[v]);
}

// v3, shared scan. Variants are sorted so each scan group (same bias, window, gap, stop;
// only the targets differ) is contiguous; gbegin[g] is group g's first variant. One block
// per (session, up to blockDim groups):
//   phase 1  thread t runs group g0+t's entry scan once and leaves the Entry in shared memory
//   phase 2  the block's threads stride over all of those groups' variants and manage each
//            trade from its group's Entry
// The scan, 62% of the useful work, now runs once per group instead of once per variant.
constexpr int kMaxBlock = 1024;
__global__ void sim_grouped(const float* __restrict__ o, const float* __restrict__ h,
                            const float* __restrict__ l, const float* __restrict__ c,
                            const Session* __restrict__ sess, const Params* __restrict__ grid,
                            const int* __restrict__ gbegin, const int* __restrict__ vgroup,
                            int g_lo, int g_hi, int v_lo, int VB, float* __restrict__ out) {
  __shared__ float so[kSessionMinutes], sh[kSessionMinutes], sl[kSessionMinutes],
      sc[kSessionMinutes];
  // Sized at launch to blockDim entries: a fixed kMaxBlock array (20 KB) cut occupancy.
  extern __shared__ Entry se[];
  const int s = blockIdx.y;
  const size_t b = size_t(sess[s].begin);
  for (int k = threadIdx.x; k < kSessionMinutes; k += blockDim.x) {
    so[k] = o[b + k];
    sh[k] = h[b + k];
    sl[k] = l[b + k];
    sc[k] = c[b + k];
  }
  __syncthreads();
  const int g_first = g_lo + int(blockIdx.x) * int(blockDim.x);
  const int g = g_first + int(threadIdx.x);
  if (g < g_hi) se[threadIdx.x] = find_entry(so, sh, sl, sc, grid[gbegin[g]]);
  __syncthreads();
  const int g_last = min(g_first + int(blockDim.x), g_hi);
  if (g_first >= g_last) return;
  for (int v = gbegin[g_first] + int(threadIdx.x); v < gbegin[g_last]; v += int(blockDim.x))
    out[size_t(s) * size_t(VB) + size_t(v - v_lo)] =
        manage_trade(sh, sl, sc, grid[v], se[vgroup[v] - g_first]);
}

// One thread per variant walks its sessions in date order (the equity curve is a
// running sum, so order matters). stride_v/stride_s describe the result layout:
// session-major (1, V) makes a warp's reads contiguous; variant-major (S, 1) does not.
__global__ void metrics_kernel(const float* __restrict__ r, int V, int S, size_t stride_v,
                               size_t stride_s, int split, VariantResult* __restrict__ out) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= V) return;
  VariantResult res;
  float eq = 0.f, pk = 0.f;
  for (int s = 0; s < split; ++s) metrics_add(res.in_sample, eq, pk, r[v * stride_v + s * stride_s]);
  eq = 0.f;
  pk = 0.f;
  for (int s = split; s < S; ++s)
    metrics_add(res.out_of_sample, eq, pk, r[v * stride_v + s * stride_s]);
  out[v] = res;
}

std::vector<VariantResult> run_sweep_gpu_ordered(const BarSeries& bars,
                                                 const std::vector<Params>& grid,
                                                 const SweepOptions& opt,
                                                 const std::vector<int>* gbegin = nullptr);

namespace {
auto scan_key(const Params& p) {
  return std::make_tuple(p.bias, p.win_start, p.win_len, p.min_gap, p.stop_lookback, p.stop_buf);
}
// Group-major order for the shared scan: scan groups contiguous, tp2 then tp1 inside.
std::vector<uint32_t> scan_group_order(const std::vector<Params>& grid) {
  std::vector<uint32_t> idx(grid.size());
  for (size_t i = 0; i < idx.size(); ++i) idx[i] = uint32_t(i);
  std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
    const Params &x = grid[a], &y = grid[b];
    return std::tuple_cat(scan_key(x), std::make_tuple(x.tp2_r, x.tp1_r)) <
           std::tuple_cat(scan_key(y), std::make_tuple(y.tp2_r, y.tp1_r));
  });
  return idx;
}
}  // namespace

std::vector<VariantResult> run_sweep_gpu(const BarSeries& bars, const std::vector<Params>& grid,
                                         const SweepOptions& opt) {
  if (!opt.sort_variants || opt.backend != Backend::GpuShared)
    return run_sweep_gpu_ordered(bars, grid, opt);
  std::vector<uint32_t> order;
  std::vector<Params> sorted(grid.size());
  std::vector<int> gbegin;
  {
    CpuSpan s(opt.timeline, "sort variants (host)");
    order = opt.share_scan ? scan_group_order(grid) : divergence_order(grid);
    for (size_t i = 0; i < order.size(); ++i) sorted[i] = grid[order[i]];
    if (opt.share_scan) {
      for (size_t i = 0; i < sorted.size(); ++i)
        if (i == 0 || scan_key(sorted[i]) != scan_key(sorted[i - 1])) gbegin.push_back(int(i));
      gbegin.push_back(int(sorted.size()));
    }
  }
  const auto res_sorted =
      run_sweep_gpu_ordered(bars, sorted, opt, opt.share_scan ? &gbegin : nullptr);
  CpuSpan s(opt.timeline, "unsort results (host)");
  std::vector<VariantResult> res(grid.size());
  for (size_t i = 0; i < order.size(); ++i) res[order[i]] = res_sorted[i];
  return res;
}

std::vector<VariantResult> run_sweep_gpu_ordered(const BarSeries& bars,
                                                 const std::vector<Params>& grid,
                                                 const SweepOptions& opt,
                                                 const std::vector<int>* gbegin) {
  const int S = int(bars.num_sessions());
  const int V = int(grid.size());
  const int split = int(oos_split(size_t(S), opt.oos_fraction));
  const bool naive = opt.backend == Backend::GpuNaive;
  const int block = opt.block_size;

  {
    // The runtime creates the GPU context lazily on the first real call; doing it here
    // gives that fixed cost its own span instead of hiding it inside the first upload.
    CpuSpan s(opt.timeline, "gpu context (host)");
    GBT_CHECK(hipFree(nullptr));
  }
  GpuTimer timer(opt.timeline);
  int t_up = timer.begin("upload bars+grid",
                         double(bars.num_bars()) * 16 + double(S) * sizeof(Session) +
                             double(V) * sizeof(Params));
  DeviceBuffer<float> d_o(bars.open), d_h(bars.high), d_l(bars.low), d_c(bars.close);
  DeviceBuffer<Session> d_sess(bars.sessions);
  DeviceBuffer<Params> d_grid(grid);
  timer.end(t_up);

  // Result matrix for one batch of variants: S floats per variant. Batching keeps it
  // inside GPU memory (1.7M variants x 1,900 sessions would be 13 GB).
  const size_t per_variant = size_t(S) * sizeof(float);
  const int batch = int(std::max<size_t>(1, std::min<size_t>(size_t(V), opt.max_batch_bytes / per_variant)));
  std::optional<CpuSpan> alloc_span(std::in_place, opt.timeline, "alloc result buffers (host)");
  DeviceBuffer<float> d_r(size_t(batch) * size_t(S));
  DeviceBuffer<VariantResult> d_res{size_t(V)};
  alloc_span.reset();

  if (gbegin && !naive) {
    // Shared scan. Batches hold whole groups so a group's variants never straddle two.
    const auto& gb = *gbegin;
    const int G = int(gb.size()) - 1;
    std::vector<int> vgroup(static_cast<size_t>(V));
    for (int g = 0; g < G; ++g)
      for (int v = gb[size_t(g)]; v < gb[size_t(g) + 1]; ++v) vgroup[size_t(v)] = g;
    DeviceBuffer<int> d_gbegin(gb), d_vgroup(vgroup);
    if (block > kMaxBlock) throw std::invalid_argument("block size above 1024");
    for (int g0 = 0; g0 < G;) {
      int g1 = g0 + 1;
      while (g1 < G && gb[size_t(g1) + 1] - gb[size_t(g0)] <= batch) ++g1;
      const int v0 = gb[size_t(g0)], vb = gb[size_t(g1)] - v0;
      if (vb > batch) throw std::runtime_error("one scan group exceeds the result batch");
      const double items = double(vb) * double(S);
      dim3 grid_dim(unsigned((g1 - g0 + block - 1) / block), unsigned(S));
      int t = timer.begin("sim_grouped", items * 4, items);
      sim_grouped<<<grid_dim, block, size_t(block) * sizeof(Entry)>>>(
          d_o.ptr, d_h.ptr, d_l.ptr, d_c.ptr, d_sess.ptr, d_grid.ptr, d_gbegin.ptr, d_vgroup.ptr,
          g0, g1, v0, vb, d_r.ptr);
      GBT_CHECK(hipGetLastError());
      timer.end(t);
      t = timer.begin("metrics (session-major)", items * 4 + vb * sizeof(VariantResult));
      metrics_kernel<<<(vb + block - 1) / block, block>>>(d_r.ptr, vb, S, 1, size_t(vb), split,
                                                         d_res.ptr + v0);
      GBT_CHECK(hipGetLastError());
      timer.end(t);
      g0 = g1;
    }
  }
  for (int v0 = 0; v0 < V && !(gbegin && !naive); v0 += batch) {
    const int vb = std::min(batch, V - v0);
    const double items = double(vb) * double(S);
    if (naive) {
      const size_t total = size_t(vb) * size_t(S);
      const unsigned blocks = unsigned((total + size_t(block) - 1) / size_t(block));
      int t = timer.begin("sim_naive", items * 4, items);
      sim_naive<<<blocks, block>>>(d_o.ptr, d_h.ptr, d_l.ptr, d_c.ptr, d_sess.ptr, S,
                                   d_grid.ptr + v0, vb, d_r.ptr);
      GBT_CHECK(hipGetLastError());
      timer.end(t);
      t = timer.begin("metrics (variant-major)", items * 4 + vb * sizeof(VariantResult));
      metrics_kernel<<<(vb + block - 1) / block, block>>>(d_r.ptr, vb, S, size_t(S), 1, split,
                                                         d_res.ptr + v0);
      GBT_CHECK(hipGetLastError());
      timer.end(t);
    } else {
      dim3 g(unsigned((vb + block - 1) / block), unsigned(S));
      int t = timer.begin("sim_shared", items * 4, items);
      if (opt.fused_loop)
        sim_shared<true><<<g, block>>>(d_o.ptr, d_h.ptr, d_l.ptr, d_c.ptr, d_sess.ptr,
                                       d_grid.ptr + v0, vb, d_r.ptr);
      else
        sim_shared<false><<<g, block>>>(d_o.ptr, d_h.ptr, d_l.ptr, d_c.ptr, d_sess.ptr,
                                        d_grid.ptr + v0, vb, d_r.ptr);
      GBT_CHECK(hipGetLastError());
      timer.end(t);
      t = timer.begin("metrics (session-major)", items * 4 + vb * sizeof(VariantResult));
      metrics_kernel<<<(vb + block - 1) / block, block>>>(d_r.ptr, vb, S, 1, size_t(vb), split,
                                                         d_res.ptr + v0);
      GBT_CHECK(hipGetLastError());
      timer.end(t);
    }
  }
  int t_down = timer.begin("download results", double(V) * sizeof(VariantResult));
  auto out = d_res.to_host();
  timer.end(t_down);
  GBT_CHECK(hipDeviceSynchronize());
  timer.flush();
  {
    CpuSpan s(opt.timeline, "free device buffers (host)");
    for (auto* b : {&d_o, &d_h, &d_l, &d_c, &d_r}) b->release();
    d_sess.release();
    d_grid.release();
    d_res.release();
  }
  return out;
}

// ---------------------------------------------------------------------------
// Peak bandwidth, measured rather than quoted
// ---------------------------------------------------------------------------

__global__ void copy_kernel(const float4* __restrict__ in, float4* __restrict__ out, size_t n) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    out[i] = in[i];
}

double gpu_measured_peak_gbs() {
  static double cached = 0;
  if (cached > 0) return cached;
  const size_t bytes = size_t(256) << 20;
  const size_t n = bytes / sizeof(float4);
  DeviceBuffer<float4> a(n), b(n);
  GBT_CHECK(hipMemset(a.ptr, 0, bytes));
  hipEvent_t e0, e1;
  GBT_CHECK(hipEventCreate(&e0));
  GBT_CHECK(hipEventCreate(&e1));
  hipDeviceProp_t prop;
  GBT_CHECK(hipGetDeviceProperties(&prop, 0));
  const unsigned blocks = unsigned(prop.multiProcessorCount * 8);
  copy_kernel<<<blocks, 256>>>(a.ptr, b.ptr, n);  // warm-up
  float best = 1e30f;
  for (int i = 0; i < 10; ++i) {
    GBT_CHECK(hipEventRecord(e0));
    copy_kernel<<<blocks, 256>>>(a.ptr, b.ptr, n);
    GBT_CHECK(hipEventRecord(e1));
    GBT_CHECK(hipEventSynchronize(e1));
    float ms = 0;
    GBT_CHECK(hipEventElapsedTime(&ms, e0, e1));
    best = std::min(best, ms);
  }
  (void)hipEventDestroy(e0);
  (void)hipEventDestroy(e1);
  cached = 2.0 * double(bytes) / (double(best) * 1e6);  // read + write
  return cached;
}

// ---------------------------------------------------------------------------
// Monte Carlo
// ---------------------------------------------------------------------------

__global__ void mc_kernel(const float* __restrict__ r, int nr, const float* __restrict__ risks,
                          int nrisk, int paths, EvalRules rules, uint64_t seed,
                          int* __restrict__ out) {
  const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size_t(nrisk) * size_t(paths)) return;
  const int ri = int(idx / size_t(paths));
  // Same global path id as the CPU version: risk index * paths + path.
  out[idx] = simulate_path(r, nr, risks[ri], rules, seed, uint64_t(idx));
}

std::vector<RiskOutcome> run_montecarlo_gpu(const std::vector<float>& r, const EvalRules& rules,
                                            const McOptions& opt) {
  DeviceBuffer<float> d_r(r), d_risk(opt.risks);
  const size_t total = opt.risks.size() * size_t(opt.paths);
  DeviceBuffer<int> d_out(total);
  const int block = 256;
  mc_kernel<<<unsigned((total + block - 1) / block), block>>>(
      d_r.ptr, int(r.size()), d_risk.ptr, int(opt.risks.size()), opt.paths, rules, opt.seed,
      d_out.ptr);
  GBT_CHECK(hipGetLastError());
  auto h = d_out.to_host();
  std::vector<RiskOutcome> res;
  for (size_t i = 0; i < opt.risks.size(); ++i)
    res.push_back(summarise_outcomes(opt.risks[i], h.data() + i * size_t(opt.paths), opt.paths));
  return res;
}

}  // namespace gbt
