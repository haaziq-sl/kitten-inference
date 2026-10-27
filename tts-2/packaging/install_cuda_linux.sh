#!/bin/sh
# Install the CUDA 12.8 compiler, cuobjdump and nvdisasm (for the GPU code check) and the headers/static libraries the engine builds
# with, in a manylinux_2_28
# (AlmaLinux 8) build container. Pinned to the exact versions of the nvidia/cuda:12.8.1-devel image the default build
# is checked with (nvcc 12.8.93), so the GPU code is the same. The CUDA runtime is linked statically; the NVIDIA driver
# is loaded at run time.
set -eu
if [ -x /usr/local/cuda-12.8/bin/nvcc ] && [ -x /usr/local/cuda-12.8/bin/cuobjdump ] && [ -x /usr/local/cuda-12.8/bin/nvdisasm ]; then
  /usr/local/cuda-12.8/bin/nvcc --version | tail -2; exit 0
fi
dnf install -y --disablerepo='epel*' dnf-plugins-core
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf install -y --disablerepo='epel*' \
  cuda-nvcc-12-8-12.8.93-1 cuda-crt-12-8-12.8.93-1 cuda-nvvm-12-8-12.8.93-1 \
  cuda-cudart-devel-12-8-12.8.90-1 cuda-driver-devel-12-8-12.8.90-1 cuda-cccl-12-8-12.8.90-1 \
  libcurand-devel-12-8-10.3.9.90-1 cuda-cuobjdump-12-8-12.8.90-1 cuda-nvdisasm-12-8-12.8.90-1
/usr/local/cuda-12.8/bin/nvcc --version | tail -2
