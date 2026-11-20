#!/usr/bin/env bash
# HIP on NVIDIA is header-only on top of CUDA. AMD's hip-runtime-nvidia package depends
# on the generic "cuda" metapackage, which drags in a second, newer full CUDA install;
# unpacking the header packages directly avoids that.
set -euo pipefail
dir=$(mktemp -d)
cd "$dir"
apt-get download hip-dev hipcub-dev rocm-core hipcc-nvidia hipcc
for d in *.deb; do dpkg -x "$d" /; done
ln -sfn "$(ls -d /opt/rocm-* | sort | tail -1)" /opt/rocm
echo "HIP headers in $(readlink -f /opt/rocm)"
