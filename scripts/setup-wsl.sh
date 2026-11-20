#!/usr/bin/env bash
# One-time toolchain setup for Ubuntu 24.04 (native or WSL2). Run as root.
#   - build tools: gcc, CMake, Ninja, GoogleTest
#   - CUDA 12.8 toolkit (the NVIDIA backend HIP compiles through)
#   - HIP from AMD's ROCm repo, configured for NVIDIA GPUs (HIP_PLATFORM=nvidia)
#   - hipCUB headers (on AMD this wraps rocPRIM, on NVIDIA it wraps CUB)
# On a machine with an AMD GPU, install ROCm instead and skip the CUDA step.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y build-essential cmake ninja-build git wget gnupg ca-certificates \
  python3 python3-venv libgtest-dev linux-tools-common

# CUDA toolkit. On WSL the driver lives on Windows, so use NVIDIA's wsl-ubuntu repo
# (it ships the toolkit without a Linux driver).
if [ -d /usr/lib/wsl ]; then CUDA_REPO=wsl-ubuntu; else CUDA_REPO=ubuntu2404; fi
wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit-12-8

# HIP from ROCm, NVIDIA flavour.
ROCM_VER=6.4
mkdir -p /etc/apt/keyrings
wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/keyrings/rocm.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VER} noble main" \
  > /etc/apt/sources.list.d/rocm.list
printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' > /etc/apt/preferences.d/rocm-pin-600
apt-get update
# Not `apt-get install hip-runtime-nvidia`: it depends on the "cuda" metapackage and
# pulls a second, newer full CUDA. The headers are all HIP-on-NVIDIA needs.
bash "$(dirname "$0")/install-hip-headers.sh"

cat > /etc/profile.d/gpu-backtester.sh <<'EOF'
export PATH=/usr/local/cuda/bin:/opt/rocm/bin:$PATH
export HIP_PLATFORM=nvidia
export CUDA_PATH=/usr/local/cuda
EOF
echo "setup done"
