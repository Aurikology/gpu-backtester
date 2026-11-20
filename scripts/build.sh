#!/usr/bin/env bash
# Build and test. Usage: scripts/build.sh [NVIDIA|AMD|OFF] [build-dir]
set -euo pipefail
GPU=${1:-NVIDIA}
BUILD=${2:-$HOME/gbt-build-${GPU,,}}
SRC=$(cd "$(dirname "$0")/.." && pwd)
export PATH=/usr/local/cuda/bin:/opt/rocm/bin:$PATH
EXTRA=()
if [ "$GPU" = "NVIDIA" ] && [ -z "${CUDA_ARCH:-}" ]; then
  # Ask the driver for the card's compute capability (e.g. 8.9 -> 89).
  CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
fi
[ -n "${CUDA_ARCH:-}" ] && EXTRA+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
cmake -S "$SRC" -B "$BUILD" -G Ninja -DGBT_GPU="$GPU" "${EXTRA[@]}" >/dev/null
cmake --build "$BUILD"
if [ "${SKIP_TESTS:-0}" != "1" ]; then
  "$BUILD/gbt_tests" --gtest_brief=1
fi
echo "built: $BUILD/gbt"
