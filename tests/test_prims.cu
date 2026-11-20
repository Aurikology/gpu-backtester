// Hand-written device primitives against std:: on awkward sizes (tile edges, 1 element).
#include <gtest/gtest.h>

#include <numeric>
#include <random>

#include "../src/gpu_runtime.cuh"
#include "gbt/prims.cuh"

using namespace gbt;

class Prims : public ::testing::TestWithParam<size_t> {};

TEST_P(Prims, ScanAndReduceAreExact) {
  const size_t n = GetParam();
  std::mt19937 rng{unsigned(n)};
  std::vector<uint32_t> h(n);
  for (auto& x : h) x = rng();  // full 32-bit values: wrap-around must match too
  std::vector<uint32_t> ref(n);
  std::inclusive_scan(h.begin(), h.end(), ref.begin());

  DeviceBuffer<uint32_t> in(h), out(n), sum(1);
  DeviceBuffer<uint8_t> tmp(std::max(prims::scan_tmp_bytes(n), prims::reduce_tmp_bytes(n)));
  prims::device_inclusive_scan(in.ptr, out.ptr, n, tmp.ptr);
  prims::device_reduce_sum(in.ptr, sum.ptr, n, tmp.ptr);
  EXPECT_EQ(out.to_host(), ref);
  EXPECT_EQ(sum.to_host()[0], ref.back());

  DeviceBuffer<uint32_t> out2(n);
  DeviceBuffer<uint8_t> tmp2(prims::lookback_tmp_bytes(n));
  for (int rep = 0; rep < 3; ++rep) {  // races show up as intermittent failures: repeat
    prims::device_inclusive_scan_lookback(in.ptr, out2.ptr, n, tmp2.ptr);
    ASSERT_EQ(out2.to_host(), ref) << "look-back scan, repetition " << rep;
  }
}

INSTANTIATE_TEST_SUITE_P(Sizes, Prims,
                         ::testing::Values(size_t(1), size_t(31), size_t(255), size_t(2047),
                                           size_t(2048), size_t(2049), size_t(600000),
                                           size_t(1) << 20, (size_t(1) << 22) + 3));
