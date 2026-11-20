// Device-wide reduce and scan, hand-written, plus a benchmark against hipCUB.
//
// Scan uses "reduce-then-scan" in three kernels:
//   1. every block sums its 2,048-element tile               reads n
//   2. one block scans the per-tile sums                      (tiny)
//   3. every block scans its tile again, adding its offset   reads n, writes n
// That is ~3n of memory traffic. rocPRIM and CUB use a single-pass "decoupled
// look-back" scan (Merrill & Garland, 2016) that moves ~2n, so the best this design
// can reach is about two thirds of their speed. The benchmark shows how close it gets.
// Only the two algorithms used: the umbrella hipcub.hpp in ROCm 6.4 includes a CUB
// header that CUDA 12.8 removed.
#include <hipcub/device/device_reduce.hpp>
#include <hipcub/device/device_scan.hpp>

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <sstream>

#include "gbt/gpu_info.hpp"
#include "gbt/prims.cuh"
#include "gpu_runtime.cuh"

namespace gbt::prims {

static size_t tiles(size_t n) { return (n + size_t(kBlock * kItems) - 1) / size_t(kBlock * kItems); }

// Sum of one 2,048-element tile per block.
__global__ void tile_reduce(const uint32_t* __restrict__ in, size_t n, uint32_t* __restrict__ sums) {
  __shared__ uint32_t scratch[kBlock / kWarp];
  const size_t base = size_t(blockIdx.x) * kBlock * kItems;
  uint32_t acc = 0;
  // Striped reads here: consecutive threads read consecutive elements (coalesced).
  for (int k = 0; k < kItems; ++k) {
    const size_t i = base + size_t(k) * kBlock + threadIdx.x;
    if (i < n) acc += in[i];
  }
  acc = block_reduce(acc, Sum{}, scratch);
  if (threadIdx.x == 0) sums[blockIdx.x] = acc;
}

// Single block: exclusive scan of the tile sums, 2,048 at a time with a running carry.
__global__ void scan_tile_sums(uint32_t* sums, size_t m) {
  __shared__ uint32_t scratch[kBlock / kWarp];
  __shared__ uint32_t carry;
  if (threadIdx.x == 0) carry = 0;
  __syncthreads();
  for (size_t base = 0; base < m; base += kBlock) {
    const size_t i = base + threadIdx.x;
    const uint32_t v = i < m ? sums[i] : 0;
    uint32_t total;
    const uint32_t incl = block_inclusive_scan(v, Sum{}, scratch, &total);
    if (i < m) sums[i] = carry + incl - v;  // exclusive
    __syncthreads();
    if (threadIdx.x == 0) carry += total;
    __syncthreads();
  }
}

// Each thread scans its kItems in registers, the block scans the thread totals, then
// every element gets its thread offset plus the tile offset.
__global__ void tile_scan(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, size_t n,
                          const uint32_t* __restrict__ offsets) {
  __shared__ uint32_t tile[kBlock * kItems];
  __shared__ uint32_t scratch[kBlock / kWarp];
  const size_t base = size_t(blockIdx.x) * kBlock * kItems;
  // Coalesced (striped) load into shared memory, then each thread reads its own
  // contiguous run. Shared memory does the transpose that registers cannot.
  for (int k = 0; k < kItems; ++k) {
    const size_t i = base + size_t(k) * kBlock + threadIdx.x;
    tile[k * kBlock + threadIdx.x] = i < n ? in[i] : 0;
  }
  __syncthreads();
  uint32_t v[kItems];
  uint32_t run = 0;
  for (int k = 0; k < kItems; ++k) {
    run += tile[threadIdx.x * kItems + k];
    v[k] = run;
  }
  const uint32_t incl = block_inclusive_scan(run, Sum{}, scratch);
  const uint32_t off = offsets[blockIdx.x] + incl - run;
  for (int k = 0; k < kItems; ++k) tile[threadIdx.x * kItems + k] = v[k] + off;
  __syncthreads();
  for (int k = 0; k < kItems; ++k) {
    const size_t i = base + size_t(k) * kBlock + threadIdx.x;
    if (i < n) out[i] = tile[k * kBlock + threadIdx.x];
  }
}

__global__ void final_reduce(const uint32_t* __restrict__ sums, size_t m, uint32_t* out) {
  __shared__ uint32_t scratch[kBlock / kWarp];
  uint32_t acc = 0;
  for (size_t i = threadIdx.x; i < m; i += kBlock) acc += sums[i];
  acc = block_reduce(acc, Sum{}, scratch);
  if (threadIdx.x == 0) *out = acc;
}

// ---------------------------------------------------------------------------
// Single-pass scan with decoupled look-back (Merrill & Garland, 2016)
//
// Every tile owns a 64-bit status word: [flag:32 | value:32], written in one store so
// a reader never sees a flag with a stale value.
//   flag 0  not ready
//   flag 1  AGGREGATE: value = this tile's own sum
//   flag 2  PREFIX:    value = sum of everything up to and including this tile
// A tile publishes its aggregate as soon as it has it, then walks backwards adding
// aggregates until it meets a PREFIX, publishes its own PREFIX, and finishes. The
// walk is usually one or two steps, because predecessors finish first.
//
// Deadlock trap: if tile 5 spins waiting on tile 4, but the hardware scheduled tile 5
// and never tile 4, it waits forever. So tiles are numbered by an atomic counter at
// block start (not by blockIdx): a tile's predecessors are always already running.
// ---------------------------------------------------------------------------

constexpr unsigned long long kFlagAggregate = 1ull << 32;
constexpr unsigned long long kFlagPrefix = 2ull << 32;

__global__ void scan_lookback(const uint32_t* __restrict__ in, uint32_t* __restrict__ out,
                              size_t n, unsigned long long* status, unsigned int* counter) {
  __shared__ uint32_t tile[kBlock * kItems];
  __shared__ uint32_t scratch[kBlock / kWarp];
  __shared__ unsigned int s_tile;
  __shared__ uint32_t s_prefix;
  if (threadIdx.x == 0) s_tile = atomicAdd(counter, 1u);
  __syncthreads();
  const unsigned int t = s_tile;
  const size_t base = size_t(t) * kBlock * kItems;

  for (int k = 0; k < kItems; ++k) {
    const size_t i = base + size_t(k) * kBlock + threadIdx.x;
    tile[k * kBlock + threadIdx.x] = i < n ? in[i] : 0;
  }
  __syncthreads();
  uint32_t v[kItems];
  uint32_t run = 0;
  for (int k = 0; k < kItems; ++k) {
    run += tile[threadIdx.x * kItems + k];
    v[k] = run;
  }
  uint32_t total;
  const uint32_t incl = block_inclusive_scan(run, Sum{}, scratch, &total);

  if (threadIdx.x == 0) {
    uint32_t excl = 0;
    if (t == 0) {
      atomicExch(&status[0], kFlagPrefix | total);
    } else {
      atomicExch(&status[t], kFlagAggregate | total);
      long long j = (long long)t - 1;
      for (;;) {
        const unsigned long long w = *(volatile unsigned long long*)&status[j];
        const unsigned long long flag = w & ~0xFFFFFFFFull;
        if (flag == 0) continue;  // predecessor still working: spin
        excl += uint32_t(w);
        if (flag == kFlagPrefix) break;
        --j;
      }
      atomicExch(&status[t], kFlagPrefix | uint32_t(excl + total));
    }
    s_prefix = excl;
  }
  __syncthreads();

  const uint32_t off = s_prefix + incl - run;
  for (int k = 0; k < kItems; ++k) tile[threadIdx.x * kItems + k] = v[k] + off;
  __syncthreads();
  for (int k = 0; k < kItems; ++k) {
    const size_t i = base + size_t(k) * kBlock + threadIdx.x;
    if (i < n) out[i] = tile[k * kBlock + threadIdx.x];
  }
}

size_t lookback_tmp_bytes(size_t n) { return 8 + tiles(n) * sizeof(unsigned long long); }

void device_inclusive_scan_lookback(const uint32_t* in, uint32_t* out, size_t n, void* tmp,
                                    hipStream_t s) {
  const size_t m = tiles(n);
  auto* counter = static_cast<unsigned int*>(tmp);
  auto* status = reinterpret_cast<unsigned long long*>(static_cast<char*>(tmp) + 8);
  (void)hipMemsetAsync(tmp, 0, lookback_tmp_bytes(n), s);
  scan_lookback<<<unsigned(m), kBlock, 0, s>>>(in, out, n, status, counter);
}

size_t reduce_tmp_bytes(size_t n) { return tiles(n) * sizeof(uint32_t); }
size_t scan_tmp_bytes(size_t n) { return tiles(n) * sizeof(uint32_t); }

void device_reduce_sum(const uint32_t* in, uint32_t* out, size_t n, void* tmp, hipStream_t s) {
  const size_t m = tiles(n);
  auto* sums = static_cast<uint32_t*>(tmp);
  tile_reduce<<<unsigned(m), kBlock, 0, s>>>(in, n, sums);
  final_reduce<<<1, kBlock, 0, s>>>(sums, m, out);
}

void device_inclusive_scan(const uint32_t* in, uint32_t* out, size_t n, void* tmp, hipStream_t s) {
  const size_t m = tiles(n);
  auto* sums = static_cast<uint32_t*>(tmp);
  tile_reduce<<<unsigned(m), kBlock, 0, s>>>(in, n, sums);
  scan_tile_sums<<<1, kBlock, 0, s>>>(sums, m);
  tile_scan<<<unsigned(m), kBlock, 0, s>>>(in, out, n, sums);
}

}  // namespace gbt::prims

namespace gbt {

namespace {
template <class F>
float time_ms(F&& f, int reps) {
  hipEvent_t a, b;
  GBT_CHECK(hipEventCreate(&a));
  GBT_CHECK(hipEventCreate(&b));
  f();  // warm-up
  GBT_CHECK(hipDeviceSynchronize());
  float best = 1e30f;
  for (int i = 0; i < reps; ++i) {
    GBT_CHECK(hipEventRecord(a));
    f();
    GBT_CHECK(hipEventRecord(b));
    GBT_CHECK(hipEventSynchronize(b));
    float ms = 0;
    GBT_CHECK(hipEventElapsedTime(&ms, a, b));
    best = std::min(best, ms);
  }
  (void)hipEventDestroy(a);
  (void)hipEventDestroy(b);
  return best;
}
}  // namespace

std::string bench_prims(const std::vector<size_t>& sizes, int reps) {
  const double peak = gpu_measured_peak_gbs();
  std::ostringstream o;
  char line[256];
  std::snprintf(line, sizeof line,
                "measured peak (device copy): %.1f GB/s\n"
                "%-10s %8s %10s %8s %10s %8s %8s  %s\n",
                peak, "op", "elements", "mine GB/s", "%peak", "hipCUB", "%peak", "mine/cub",
                "exact");
  o << line;
  std::mt19937 rng(7);
  for (size_t n : sizes) {
    std::vector<uint32_t> h(n);
    for (auto& x : h) x = rng() & 0xFF;
    DeviceBuffer<uint32_t> in(h), out(n), one(1);
    DeviceBuffer<uint8_t> tmp(std::max(prims::scan_tmp_bytes(n), prims::reduce_tmp_bytes(n)));

    // hipCUB temp storage (query, then allocate the larger of the two).
    size_t cub_scan_bytes = 0, cub_red_bytes = 0;
    GBT_CHECK(hipcub::DeviceScan::InclusiveSum(nullptr, cub_scan_bytes, in.ptr, out.ptr, int(n)));
    GBT_CHECK(hipcub::DeviceReduce::Sum(nullptr, cub_red_bytes, in.ptr, one.ptr, int(n)));
    DeviceBuffer<uint8_t> cub_tmp(std::max(cub_scan_bytes, cub_red_bytes));

    // Reference on the CPU.
    std::vector<uint32_t> ref(n);
    std::inclusive_scan(h.begin(), h.end(), ref.begin());
    const uint32_t ref_sum = n ? ref.back() : 0;

    // --- reduce ---
    float mine = time_ms([&] { prims::device_reduce_sum(in.ptr, one.ptr, n, tmp.ptr); }, reps);
    const bool ok_red = one.to_host()[0] == ref_sum;
    float cub = time_ms(
        [&] { (void)hipcub::DeviceReduce::Sum(cub_tmp.ptr, cub_red_bytes, in.ptr, one.ptr, int(n)); },
        reps);
    const bool ok_red_cub = one.to_host()[0] == ref_sum;
    const double bytes_red = double(n) * 4;  // minimum traffic: read once
    std::snprintf(line, sizeof line, "%-10s %8zu %10.1f %7.0f%% %10.1f %7.0f%% %8.2f  %s\n",
                  "reduce", n, bytes_red / (mine * 1e6), 100 * bytes_red / (mine * 1e6) / peak,
                  bytes_red / (cub * 1e6), 100 * bytes_red / (cub * 1e6) / peak, cub / mine,
                  ok_red && ok_red_cub ? "yes" : "NO");
    o << line;

    // --- scan, both designs ---
    cub = time_ms(
        [&] { (void)hipcub::DeviceScan::InclusiveSum(cub_tmp.ptr, cub_scan_bytes, in.ptr, out.ptr, int(n)); },
        reps);
    const bool ok_scan_cub = out.to_host() == ref;
    const double bytes_scan = double(n) * 8;  // minimum traffic: read once, write once
    struct Variant {
      const char* name;
      void (*fn)(const uint32_t*, uint32_t*, size_t, void*, hipStream_t);
      size_t tmp_bytes;
    };
    for (const Variant& sv :
         {Variant{"scan-3pass", prims::device_inclusive_scan, prims::scan_tmp_bytes(n)},
          Variant{"scan-1pass", prims::device_inclusive_scan_lookback, prims::lookback_tmp_bytes(n)}}) {
      DeviceBuffer<uint8_t> stmp(sv.tmp_bytes);
      GBT_CHECK(hipMemset(out.ptr, 0, n * 4));
      mine = time_ms([&] { sv.fn(in.ptr, out.ptr, n, stmp.ptr, 0); }, reps);
      const bool ok_scan = out.to_host() == ref;
      std::snprintf(line, sizeof line, "%-10s %8zu %10.1f %7.0f%% %10.1f %7.0f%% %8.2f  %s\n",
                    sv.name, n, bytes_scan / (mine * 1e6), 100 * bytes_scan / (mine * 1e6) / peak,
                    bytes_scan / (cub * 1e6), 100 * bytes_scan / (cub * 1e6) / peak, cub / mine,
                    ok_scan && ok_scan_cub ? "yes" : "NO");
      o << line;
    }
  }
  o << "GB/s counts the minimum traffic (reduce: n reads; scan: n reads + n writes).\n"
       "mine/cub > 1.00 would mean the hand-written kernel is faster.\n";
  return o.str();
}

}  // namespace gbt
