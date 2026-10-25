#!/bin/bash
# Build the Linux kitten-inference-tts2-cu12 wheel (py3-none-manylinux_2_28_x86_64). Runs inside
# quay.io/pypa/manylinux_2_28_x86_64 (the CI job, or tts-2/packaging/build_linux_docker.sh):
#   tts-2/packaging/build_wheel_linux.sh [OUT_DIR]        (default tts-2/dist)
# Steps: CUDA 12.8 (install_cuda_linux.sh), the text dependencies (tools/build_text_deps.py), `cmake --preset wheel`,
# the GPU code check against the default build (sass_manifest.py), install into a stage folder, make_wheel.py, then
# `auditwheel show` (the wheel must need nothing outside the manylinux_2_28 policy: the CUDA runtime is static and the
# driver is loaded at run time). Environment:
#   JOBS (default: the CPUs available), WHEEL_VERSION (default 0.2.0), KT2N_TEXT_DEPS (reuse a text-deps prefix),
#   KT2N_SASS_CHECK=0 (skip the GPU code check: never for a release).
set -euo pipefail
SRC=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-$SRC/dist}
JOBS=${JOBS:-$(nproc)}
PY=${PYTHON:-/opt/python/cp312-cp312/bin/python}
cd "$SRC"
t0=$(date +%s)
sh packaging/install_cuda_linux.sh
export PATH=/usr/local/cuda-12.8/bin:$PATH CUDACXX=/usr/local/cuda-12.8/bin/nvcc
gcc --version | head -1; cmake --version | head -1
if [ -z "${KT2N_TEXT_DEPS:-}" ]; then
  "$PY" tools/build_text_deps.py --prefix "$SRC/build-deps/text" --jobs "$JOBS"
  export KT2N_TEXT_DEPS="$SRC/build-deps/text"
fi
t1=$(date +%s)
cmake --preset wheel
cmake --build --preset wheel --parallel "$JOBS"
# The GPU code must be the default build's, function for function (packaging/sass_manifest.py): files whose code came
# out differently are compiled again. KT2N_SASS_CHECK=0 skips this (development builds with other architectures only).
if [ "${KT2N_SASS_CHECK:-1}" != 0 ]; then
  "$PY" packaging/sass_manifest.py check build/wheel/lib/libkt2n_engine.so --reference packaging/sass-reference.txt \
    --build-dir build/wheel --attempts 6 -- cmake --build --preset wheel --parallel "$JOBS"
else
  echo "WARNING: KT2N_SASS_CHECK=0: the GPU code was not compared with the default build's; do not release this wheel"
fi
t2=$(date +%s)
rm -rf build/wheel-stage
cmake --install build/wheel --prefix build/wheel-stage > /dev/null
mkdir -p "$OUT"
"$PY" packaging/make_wheel.py --stage build/wheel-stage --plat manylinux_2_28_x86_64 --out "$OUT" \
  --version "${WHEEL_VERSION:-0.2.0}"
WHL=$(ls -t "$OUT"/kitten_inference_tts2_cu12-*-py3-none-manylinux_2_28_x86_64.whl | head -1)
"$PY" -m pip install -q "auditwheel>=6,<7"
"$PY" -m auditwheel show "$WHL" | tee "$OUT/auditwheel-show.txt"
# auditwheel names the oldest policy the files satisfy; manylinux_2_28 must be at least as new as that.
"$PY" packaging/auditwheel_tag.py "$OUT/auditwheel-show.txt" manylinux_2_28_x86_64
echo "timings: deps+toolchain $((t1 - t0)) s, engine build $((t2 - t1)) s, total $(( $(date +%s) - t0 )) s"
ls -l "$WHL" build/wheel-stage/lib/libkt2n_engine.so
