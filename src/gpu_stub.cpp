// Compiled only in CPU-only builds (GBT_GPU=OFF), e.g. CI runners without a GPU.
#include <stdexcept>

#include "gbt/engine.hpp"
#include "gbt/gpu_info.hpp"
#include "gbt/montecarlo.hpp"

namespace gbt {

bool gpu_available() { return false; }
std::string gpu_device_name() { return "none (CPU-only build)"; }
double gpu_measured_peak_gbs() { return 0.0; }
std::string bench_prims(const std::vector<size_t>&, int) { return "built without GPU support\n"; }

std::vector<VariantResult> run_sweep_gpu(const BarSeries&, const std::vector<Params>&,
                                         const SweepOptions&) {
  throw std::runtime_error("built without GPU support");
}

std::vector<RiskOutcome> run_montecarlo_gpu(const std::vector<float>&, const EvalRules&,
                                            const McOptions&) {
  throw std::runtime_error("built without GPU support");
}

}  // namespace gbt
