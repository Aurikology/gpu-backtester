#pragma once
// HIP runtime helpers. The source is written against HIP, AMD's GPU API. On an AMD
// GPU it compiles with hipcc/ROCm; on an NVIDIA GPU, HIP's header maps every hip*
// call onto CUDA at compile time (that is what HIP_PLATFORM=nvidia means), so the
// same file runs on both vendors.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "gbt/profiler.hpp"

#define GBT_CHECK(expr)                                                                    \
  do {                                                                                     \
    hipError_t err_ = (expr);                                                              \
    if (err_ != hipSuccess)                                                                \
      throw std::runtime_error(std::string(#expr) + ": " + hipGetErrorString(err_) + " (" + \
                               __FILE__ + ":" + std::to_string(__LINE__) + ")");           \
  } while (0)

namespace gbt {

template <class T>
struct DeviceBuffer {
  T* ptr = nullptr;
  size_t n = 0;
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t count) : n(count) {
    if (n) GBT_CHECK(hipMalloc(&ptr, n * sizeof(T)));
  }
  DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size()) {
    if (n) GBT_CHECK(hipMemcpy(ptr, host.data(), n * sizeof(T), hipMemcpyHostToDevice));
  }
  ~DeviceBuffer() {
    if (ptr) (void)hipFree(ptr);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  std::vector<T> to_host() const {
    std::vector<T> h(n);
    if (n) GBT_CHECK(hipMemcpy(h.data(), ptr, n * sizeof(T), hipMemcpyDeviceToHost));
    return h;
  }
};

// Times GPU work with device events and records it on the timeline's GPU track.
// Events are resolved lazily in flush(), so timing never forces an extra sync.
class GpuTimer {
 public:
  explicit GpuTimer(Timeline* tl) : tl_(tl) {
    if (!tl_) return;
    GBT_CHECK(hipEventCreate(&ref_));
    GBT_CHECK(hipEventRecord(ref_));
    GBT_CHECK(hipEventSynchronize(ref_));
    ref_host_us_ = tl_->now_us();
  }
  ~GpuTimer() {
    try { flush(); } catch (...) {}
    if (ref_) (void)hipEventDestroy(ref_);
  }
  // Call begin() before launching, end() after; both just enqueue events.
  int begin(const std::string& name, double bytes = 0, double items = 0) {
    if (!tl_) return -1;
    Pending p;
    p.span.name = name;
    p.span.track = "gpu";
    p.span.bytes = bytes;
    p.span.items = items;
    GBT_CHECK(hipEventCreate(&p.a));
    GBT_CHECK(hipEventCreate(&p.b));
    GBT_CHECK(hipEventRecord(p.a));
    pending_.push_back(p);
    return int(pending_.size()) - 1;
  }
  void end(int id) {
    if (id < 0) return;
    GBT_CHECK(hipEventRecord(pending_[size_t(id)].b));
  }
  void flush() {
    if (!tl_) return;
    for (auto& p : pending_) {
      GBT_CHECK(hipEventSynchronize(p.b));
      float t_start = 0, t_dur = 0;
      GBT_CHECK(hipEventElapsedTime(&t_start, ref_, p.a));
      GBT_CHECK(hipEventElapsedTime(&t_dur, p.a, p.b));
      p.span.start_us = ref_host_us_ + double(t_start) * 1e3;
      p.span.dur_us = double(t_dur) * 1e3;
      tl_->add(p.span);
      (void)hipEventDestroy(p.a);
      (void)hipEventDestroy(p.b);
    }
    pending_.clear();
  }

 private:
  struct Pending {
    Span span;
    hipEvent_t a = nullptr, b = nullptr;
  };
  Timeline* tl_;
  hipEvent_t ref_ = nullptr;
  double ref_host_us_ = 0;
  std::vector<Pending> pending_;
};

}  // namespace gbt
