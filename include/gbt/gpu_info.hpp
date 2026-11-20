#pragma once
// Host-callable GPU utilities (stubbed out in CPU-only builds).
#include <cstddef>
#include <string>
#include <vector>

namespace gbt {

std::string gpu_device_name();
// Device-to-device copy bandwidth (read + write), the realistic ceiling for any
// memory-bound kernel on this GPU. Measured once and cached.
double gpu_measured_peak_gbs();

// Benchmarks the hand-written reduce/scan against hipCUB for each size (elements),
// checks every result for exact equality, and returns a text table.
std::string bench_prims(const std::vector<size_t>& sizes, int reps);

}  // namespace gbt
