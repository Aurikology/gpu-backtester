#pragma once
// Shared by the CPU build (g++) and the GPU build (nvcc via HIP, or hipcc on AMD).
// Anything marked GBT_HD compiles for both the host and the device, which is how the
// CPU reference and the GPU kernel run literally the same strategy code.

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GBT_HD __host__ __device__
#define GBT_GPU_COMPILE 1
#else
#define GBT_HD
#endif

#include <cstdint>

namespace gbt {

// Regular trading hours, New York time: 09:30 to 16:00 is 390 one-minute bars.
constexpr int kSessionMinutes = 390;

}  // namespace gbt
