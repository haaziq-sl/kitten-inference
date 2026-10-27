#!/bin/bash
# Build the Linux wheel in the manylinux_2_28 container with Docker (what the CI job runs):
#   tts-2/packaging/build_linux_docker.sh [OUT_DIR]   (default tts-2/dist)
# The source is mounted at /root/src/port, the path of the build that packaging/sass-reference.txt was made from: nvcc
# puts the source file's path into the names of GPU functions in anonymous namespaces, so the same path gives the
# same names.
set -euo pipefail
SRC=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(mkdir -p "${1:-$SRC/dist}" && cd "${1:-$SRC/dist}" && pwd)
IMAGE=${MANYLINUX_IMAGE:-quay.io/pypa/manylinux_2_28_x86_64:2026.10.24-1}
docker run --rm -e JOBS="${JOBS:-$(nproc)}" -e WHEEL_VERSION="${WHEEL_VERSION:-}" -e KT2N_SASS_CHECK="${KT2N_SASS_CHECK:-1}" \
  -v "$SRC":/root/src/port -v "$OUT":/out -w /root/src/port "$IMAGE" \
  bash packaging/build_wheel_linux.sh /out
