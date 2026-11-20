#pragma once
// Hand-written GPU primitives: reduce and inclusive scan at warp, block and device level.
// These are the same building blocks AMD's rocPRIM (and NVIDIA's CUB) provide; writing
// them is how you learn where the time goes. bench/ compares them against hipCUB.
//
// Device code only: include from .cu files.

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gbt::prims {

// Warp ("wavefront" on AMD) width. NVIDIA warps are 32 lanes; AMD Instinct (CDNA)
// wavefronts are 64, and RDNA can run 32. Code below never hard-codes 32.
#if defined(__HIP_PLATFORM_AMD__)
#if defined(__AMDGCN_WAVEFRONT_SIZE)
constexpr int kWarp = __AMDGCN_WAVEFRONT_SIZE;
#else
constexpr int kWarp = 64;
#endif
#else
constexpr int kWarp = 32;
#endif

template <class T>
__device__ inline T shfl_up(T v, int delta) {
#if defined(__HIP_PLATFORM_AMD__)
  return __shfl_up(v, delta, kWarp);
#else
  return __shfl_up_sync(0xffffffffu, v, delta, kWarp);
#endif
}

template <class T>
__device__ inline T shfl_xor(T v, int mask) {
#if defined(__HIP_PLATFORM_AMD__)
  return __shfl_xor(v, mask, kWarp);
#else
  return __shfl_xor_sync(0xffffffffu, v, mask, kWarp);
#endif
}

template <class T>
__device__ inline T shfl(T v, int lane) {
#if defined(__HIP_PLATFORM_AMD__)
  return __shfl(v, lane, kWarp);
#else
  return __shfl_sync(0xffffffffu, v, lane, kWarp);
#endif
}

struct Sum {
  template <class T>
  __device__ T operator()(T a, T b) const { return a + b; }
};
struct Max {
  template <class T>
  __device__ T operator()(T a, T b) const { return a > b ? a : b; }
};

// Butterfly reduction: log2(kWarp) shuffle steps, every lane ends with the total.
template <class T, class Op>
__device__ inline T warp_reduce(T v, Op op) {
  for (int m = kWarp / 2; m > 0; m >>= 1) v = op(v, shfl_xor(v, m));
  return v;
}

// Kogge-Stone inclusive scan inside one warp: log2(kWarp) shuffle steps.
template <class T, class Op>
__device__ inline T warp_inclusive_scan(T v, Op op) {
  const int lane = threadIdx.x % kWarp;
  for (int d = 1; d < kWarp; d <<= 1) {
    const T u = shfl_up(v, d);
    if (lane >= d) v = op(u, v);
  }
  return v;
}

// Block-wide inclusive scan of one value per thread. `scratch` must hold
// blockDim.x / kWarp elements of shared memory. Returns this thread's prefix; the
// block total is written to *total if non-null (valid for every thread).
template <class T, class Op>
__device__ inline T block_inclusive_scan(T v, Op op, T* scratch, T* total = nullptr) {
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  const int nwarps = blockDim.x / kWarp;
  v = warp_inclusive_scan(v, op);
  if (lane == kWarp - 1) scratch[warp] = v;
  __syncthreads();
  if (warp == 0) {
    T w = lane < nwarps ? scratch[lane] : scratch[0];
    T s = warp_inclusive_scan(w, op);
    if (lane < nwarps) scratch[lane] = s;
  }
  __syncthreads();
  if (warp > 0) v = op(scratch[warp - 1], v);
  if (total) *total = scratch[nwarps - 1];
  __syncthreads();  // scratch may be reused by the caller
  return v;
}

template <class T, class Op>
__device__ inline T block_reduce(T v, Op op, T* scratch) {
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  const int nwarps = blockDim.x / kWarp;
  v = warp_reduce(v, op);
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  T r = scratch[0];
  for (int w = 1; w < nwarps; ++w) r = op(r, scratch[w]);
  __syncthreads();
  return r;
}

// ---- device-wide (host-callable, implemented in prims.cu) ----
constexpr int kBlock = 256;
constexpr int kItems = 8;  // items per thread: a block covers 2048 elements

// Sum of n uint32 values. `tmp` needs tmp_bytes(n) bytes of device memory.
size_t reduce_tmp_bytes(size_t n);
void device_reduce_sum(const uint32_t* in, uint32_t* out, size_t n, void* tmp, hipStream_t s = 0);

// Inclusive prefix sum, reduce-then-scan (3 passes, ~3n memory traffic).
size_t scan_tmp_bytes(size_t n);
void device_inclusive_scan(const uint32_t* in, uint32_t* out, size_t n, void* tmp,
                           hipStream_t s = 0);

// Inclusive prefix sum, single pass with decoupled look-back (~2n memory traffic).
// The algorithm rocPRIM and CUB use: each tile publishes its sum, then waits only on
// its predecessors' published values instead of on a separate kernel.
size_t lookback_tmp_bytes(size_t n);
void device_inclusive_scan_lookback(const uint32_t* in, uint32_t* out, size_t n, void* tmp,
                                    hipStream_t s = 0);

}  // namespace gbt::prims
