#!/bin/bash
# Build the Linux kitten-inference-tts2-cu12 wheel (py3-none-manylinux_2_28_x86_64). Runs inside
# quay.io/pypa/manylinux_2_28_x86_64 (the CI job, or tts-2/packaging/build_linux_docker.sh):
#   tts-2/packaging/build_wheel_linux.sh [OUT_DIR]        (default tts-2/dist)
# Steps: CUDA 12.8 (install_cuda_linux.sh), the text dependencies (tools/build_text_deps.py), `cmake --preset wheel`,
# install into a stage folder, make_wheel.py, then `auditwheel show` (the wheel must need nothing outside the
# manylinux_2_28 policy: the CUDA runtime is static and the driver is loaded at run time). Environment:
#   JOBS (default: the CPUs available), WHEEL_VERSION (default 0.2.0), KT2N_TEXT_DEPS (reuse a text-deps prefix).
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
"$PY" - "$OUT/auditwheel-show.txt" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r'consistent with the following platform tag: "manylinux_2_(\d+)_x86_64"', text)
if not m or int(m.group(1)) > 28:
    sys.exit("auditwheel: the wheel does not satisfy manylinux_2_28_x86_64")
print(f"auditwheel: OK (satisfies manylinux_2_{m.group(1)}_x86_64, tagged manylinux_2_28_x86_64)")
PY
echo "timings: deps+toolchain $((t1 - t0)) s, engine build $((t2 - t1)) s, total $(( $(date +%s) - t0 )) s"
ls -l "$WHL" build/wheel-stage/lib/libkt2n_engine.so
