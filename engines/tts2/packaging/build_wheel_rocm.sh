#!/bin/bash
# Builds the kitten-inference-tts2-rocm wheel: the HIP (ROCm) engine library, the native CLI and the text-data helper
# as the package kitten_inference/_tts2_rocm, tagged py3-none-manylinux_2_28_x86_64.
#
# The build runs in the manylinux_2_28 container (AlmaLinux 8, glibc 2.28) with ROCm 7.1 from AMD's RHEL 8 repository.
# On a machine with Docker, run it from anywhere in the repository; it starts the container itself:
#
#   engines/tts2/packaging/build_wheel_rocm.sh [OUT_DIR]            (default OUT_DIR: engines/tts2/dist)
#
# Inside the container (or any manylinux_2_28 x86_64 environment) it builds directly: KT2N_IN_CONTAINER=1.
#
# Settings (environment):
#   KT2N_HIP_ARCHS       GPU targets (default: the hip preset's gfx1100;gfx1101;gfx1102;gfx1200;gfx1201)
#   JOBS                 parallel compile jobs (default: the number of CPUs)
#   KT2N_ROCM_REPO       ROCm package repository (default: AMD's RHEL 8 repository for ROCm 7.1)
#   MANYLINUX_IMAGE      build container (default: the pinned manylinux_2_28 image below)
#   SOURCE_DATE_EPOCH    file times inside the wheel (default: the last commit's time)
#   WHEEL_VERSION        the distribution's version (default: the engine's, from CMakeLists.txt); a packaging fix of
#                        the same engine is that version plus .postN, e.g. WHEEL_VERSION=0.2.0.post1
#
# The wheel links the system ROCm runtime (libamdhip64.so.7), which is excluded from auditwheel's repair: users need
# ROCm 7.x installed (engines/tts2/docs/amd-radeon.md). The ROCm build has no voice cloning.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=$(cd "$HERE/.." && pwd)                         # engines/tts2/
REPO=$(cd "$SRC/.." && pwd)
OUT=${1:-$SRC/dist}
MANYLINUX_IMAGE=${MANYLINUX_IMAGE:-quay.io/pypa/manylinux_2_28_x86_64:2026.10.10-1@sha256:4131ddd7acb9c6c966af02fe7e71edc51e078c2d41f475966697676430bfa2ae}
KT2N_HIP_ARCHS=${KT2N_HIP_ARCHS:-gfx1100;gfx1101;gfx1102;gfx1200;gfx1201}
KT2N_ROCM_REPO=${KT2N_ROCM_REPO:-https://repo.radeon.com/rocm/rhel8/7.1/main}
DIST=kitten-inference-tts2-rocm
PKG=kitten_inference/_tts2_rocm

log() { echo "[build_wheel_rocm $(date -u +%H:%M:%S)] $*"; }

# The engine's version (CMake's project VERSION, numbers only) and the distribution's: the same, or a post release of it.
ENGINE_VERSION=$(sed -n 's/^  VERSION \([0-9.]*\)$/\1/p' "$SRC/CMakeLists.txt" | head -n 1)
[ -n "$ENGINE_VERSION" ] || { echo "no project VERSION found in $SRC/CMakeLists.txt" >&2; exit 2; }
VERSION=${WHEEL_VERSION:-$ENGINE_VERSION}
if ! [[ "$VERSION" =~ ^${ENGINE_VERSION//./\\.}(\.post[0-9]+)?$ ]]; then
  echo "WHEEL_VERSION=$VERSION: must be the engine's version $ENGINE_VERSION or $ENGINE_VERSION.postN" >&2
  exit 2
fi

# ---- outside the container: start it ---------------------------------------------------------------------------
if [ "${KT2N_IN_CONTAINER:-0}" != 1 ]; then
  command -v docker > /dev/null || { echo "docker not found: run this inside a manylinux_2_28 container with KT2N_IN_CONTAINER=1" >&2; exit 2; }
  mkdir -p "$OUT"
  OUT=$(cd "$OUT" && pwd)
  if [ -z "${SOURCE_DATE_EPOCH:-}" ]; then
    SOURCE_DATE_EPOCH=$(git -C "$REPO" log -1 --format=%ct 2> /dev/null || date +%s)
  fi
  log "container $MANYLINUX_IMAGE; source $SRC; output $OUT"
  exec docker run --rm -v "$REPO:/io" -v "$OUT:/out" -w /io \
    -e KT2N_IN_CONTAINER=1 -e "KT2N_HIP_ARCHS=$KT2N_HIP_ARCHS" -e "KT2N_ROCM_REPO=$KT2N_ROCM_REPO" \
    -e "JOBS=${JOBS:-}" -e "SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH" -e "WHEEL_VERSION=$VERSION" \
    -e "HOST_UID=$(id -u)" -e "HOST_GID=$(id -g)" \
    "$MANYLINUX_IMAGE" bash "/io/${SRC#"$REPO"/}/packaging/build_wheel_rocm.sh" /out
fi

# ---- inside the container --------------------------------------------------------------------------------------
JOBS=${JOBS:-$(nproc)}
[ -n "$JOBS" ] || JOBS=$(nproc)
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
WORK=${KT2N_WHEEL_WORK:-/tmp/kt2n-rocm-wheel}
mkdir -p "$WORK"
PY=/opt/python/cp312-cp312/bin/python3
[ -x "$PY" ] || PY=$(command -v python3)
export PATH="$(dirname "$PY"):$PATH"
export SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-$(git -C "$REPO" log -1 --format=%ct 2> /dev/null || date +%s)}
grep -q 'release 8' /etc/redhat-release 2> /dev/null || log "WARNING: not an AlmaLinux / RHEL 8 (manylinux_2_28) system"

# 1. ROCm 7.1: the HIP compiler and headers, the HIP runtime to link against, rocRAND's device headers.
if [ ! -x /opt/rocm/lib/llvm/bin/clang++ ] && [ ! -x /opt/rocm/llvm/bin/clang++ ]; then
  log "installing ROCm from $KT2N_ROCM_REPO"
  cat > /etc/yum.repos.d/rocm.repo << EOF
[rocm]
name=ROCm
baseurl=$KT2N_ROCM_REPO
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF
  # EPEL is not needed and its mirrors can fail mid-sync.
  dnf install -y -q --disablerepo='epel*' hip-devel hip-runtime-amd hipcc rocm-llvm rocm-device-libs comgr \
    hsa-rocr-devel rocrand-devel rocm-core
fi
ROCM=$(readlink -f /opt/rocm)
log "ROCm $(cat "$ROCM/.info/version" 2> /dev/null || echo '?') at $ROCM"
HIP_CLANG=$ROCM/llvm/bin/clang++
[ -x "$HIP_CLANG" ] || HIP_CLANG=$ROCM/lib/llvm/bin/clang++
"$HIP_CLANG" --version | head -n 1
cmake --version | head -n 1
gcc --version | head -n 1

# 2. Text dependencies (OpenFst, PCRE2), built with the container's compiler.
DEPS=$WORK/deps/text
if [ ! -f "$DEPS/lib/libfst.a" ] || [ ! -f "$DEPS/lib/libpcre2-8.a" ]; then
  log "building the text dependencies"
  "$PY" "$SRC/tools/build_text_deps.py" --prefix "$DEPS" --jobs "$JOBS" > "$WORK/textdeps.log" 2>&1 \
    || { tail -n 30 "$WORK/textdeps.log"; exit 1; }
fi

# 3. Engine: the hip preset without voice cloning, tests or examples; libraries in lib/ (not lib64/).
B=$WORK/build
STAGE=$WORK/stage
rm -rf "$STAGE"
[ "${KT2N_WHEEL_KEEP_BUILD:-0}" = 1 ] || rm -rf "$B"   # a clean build unless asked otherwise
# ROCm's clang would pick the container's base GCC 8 (no C++20 library, no libstdc++ to link); point it at the GCC 14
# toolset that the manylinux image uses for C and C++ (its libstdc++ keeps the manylinux_2_28 symbol versions).
HIP_FLAGS=""
GCC_DIR=$(dirname "$(gcc -print-libgcc-file-name)")
case "$GCC_DIR" in /opt/rh/gcc-toolset-*) HIP_FLAGS="--gcc-install-dir=$GCC_DIR";; esac
log "configuring (archs $KT2N_HIP_ARCHS; HIP flags '$HIP_FLAGS')"
cmake --preset hip -S "$SRC" -B "$B" \
  -DCMAKE_HIP_COMPILER="$HIP_CLANG" -DCMAKE_PREFIX_PATH="$ROCM" -DCMAKE_HIP_FLAGS="$HIP_FLAGS" \
  -DKT2N_TEXT_DEPS="$DEPS" -DKT2N_HIP_ARCHS="$KT2N_HIP_ARCHS" \
  -DKT2N_BUILD_CLONING=OFF -DKT2N_VOICE_AV=OFF -DKT2N_AV=OFF -DKT2N_BUILD_TESTS=OFF -DKT2N_BUILD_EXAMPLES=OFF \
  -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_PREFIX="$STAGE" > "$WORK/configure.log" 2>&1 \
  || { tail -n 40 "$WORK/configure.log"; exit 1; }
grep -E 'HIP compiler identification|kt2n: HIP ON' "$WORK/configure.log" || true
log "building with $JOBS jobs"
cmake --build "$B" --parallel "$JOBS" > "$WORK/build.log" 2>&1 || { grep -E -m 20 'error|Error' "$WORK/build.log"; tail -n 20 "$WORK/build.log"; exit 1; }
cmake --install "$B" > "$WORK/install.log" 2>&1 || { tail -n 20 "$WORK/install.log"; exit 1; }

# 4. Package tree. One copy of the library, named libkt2n_engine.so (a wheel cannot hold symbolic links); the CLI
#    finds it through its RPATH ($ORIGIN/../lib).
TREE=$WORK/tree
rm -rf "$TREE"
mkdir -p "$TREE/$PKG/lib" "$TREE/$PKG/bin" "$TREE/$PKG/licenses"
REAL=$(readlink -f "$STAGE/lib/libkt2n_engine.so")
SONAME=$(patchelf --print-soname "$REAL")
cp "$REAL" "$TREE/$PKG/lib/libkt2n_engine.so"
patchelf --set-soname libkt2n_engine.so "$TREE/$PKG/lib/libkt2n_engine.so"
cp "$STAGE/bin/kitten-tts-native" "$STAGE/bin/kt2n_prepare_text_data" "$TREE/$PKG/bin/"
patchelf --replace-needed "$SONAME" libkt2n_engine.so "$TREE/$PKG/bin/kitten-tts-native"
cp "$SRC/LICENSE" "$SRC/NOTICE" "$TREE/$PKG/licenses/"
cp "$SRC/licenses/"*.txt "$TREE/$PKG/licenses/"
# The source of the LGPL-2.1-or-later code compiled into the library (NOTICE, "Audio processing")
mkdir -p "$TREE/$PKG/licenses/lgpl-source"
cp "$SRC/src/post/atempo.cc" "$SRC/include/kt2n/post/atempo.h" "$TREE/$PKG/licenses/lgpl-source/"
# ROCm code compiled into the library (NOTICE, "AMD ROCm"): the licence texts in licenses/rocm/ must be the ones the
# installed ROCm packages ship, so that a ROCm update with a changed licence stops the build here.
mkdir -p "$TREE/$PKG/licenses/rocm"
for pair in rocRAND.txt:rocrand/LICENSE.md HIP.txt:hip/LICENSE.md ROCm-Device-Libs.txt:ROCm-Device-Libs/LICENSE.TXT; do
  ours=$SRC/licenses/rocm/${pair%%:*}
  theirs=$ROCM/share/doc/${pair#*:}
  cmp -s "$ours" "$theirs" || { echo "$ours differs from the installed ROCm's $theirs (or one is missing): update it and NOTICE" >&2; exit 1; }
  cp "$ours" "$TREE/$PKG/licenses/rocm/"
done

# The engine's kt2n_build_info() text and ABI version, read from the built library (loading it needs only the ROCm
# runtime library, not a GPU).
read -r ABI BUILD_INFO < <("$PY" - "$TREE/$PKG/lib/libkt2n_engine.so" << 'PY'
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
lib.kt2n_build_info.restype = ctypes.c_char_p
lib.kt2n_abi_version.restype = ctypes.c_uint32
print(lib.kt2n_abi_version(), lib.kt2n_build_info().decode())
PY
)
log "engine $ENGINE_VERSION, wheel $VERSION, ABI $ABI, build info: $BUILD_INFO"
[ "$ABI" = 2 ] || { echo "unexpected engine ABI $ABI (the package contract is ABI 2)" >&2; exit 1; }
case "$BUILD_INFO" in *cap:voice.cloning*) echo "this build includes voice cloning; the ROCm wheel must not" >&2; exit 1;; esac
"$PY" - "$TREE/$PKG/__init__.py" "$VERSION" "$ABI" "$BUILD_INFO" "$KT2N_HIP_ARCHS" << 'PY'
import sys
path, version, abi, build_info, archs = sys.argv[1:]
text = f'''"""Kitten TTS 2 native engine for AMD Radeon GPUs (ROCm / HIP): the engine library, the native command line tool
and the text-data helper. The Python API is kitten_inference.tts2 (package kitten-inference-tts2).

Needs the system ROCm 7.x runtime (libamdhip64.so.7) and an AMD Radeon RX 7000 or RX 9000 GPU. No voice cloning.
"""
from pathlib import Path

BACKEND = "rocm"
ABI = {int(abi)}
VERSION = {version!r}
_HERE = Path(__file__).resolve().parent
LIB = _HERE / "lib" / "libkt2n_engine.so"
CLI = _HERE / "bin" / "kitten-tts-native"
TEXT_DATA_TOOL = _HERE / "bin" / "kt2n_prepare_text_data"
BUILD_INFO = {build_info!r}
CAPS = ()
HIP_ARCHS = {tuple(archs.split(";"))!r}
'''
open(path, "w", encoding="utf-8", newline="\n").write(text)
PY

# 5. Wheel: metadata, RECORD, deterministic zip; then auditwheel checks the glibc / libstdc++ floor and tags it.
INFO=$TREE/kitten_inference_tts2_rocm-$VERSION.dist-info
mkdir -p "$INFO"
cat > "$INFO/METADATA" << EOF
Metadata-Version: 2.1
Name: $DIST
Version: $VERSION
Summary: Kitten TTS 2 native engine for AMD Radeon GPUs (ROCm), used by kitten-inference-tts2
Home-page: https://github.com/KittenML/kitten-inference
License: Apache-2.0 for KittenML's code. The library also contains third-party code under other licenses (LGPL-2.1-or-later, BSD-3-Clause, MIT, NCSA and more, including AMD ROCm code), listed in kitten_inference/_tts2_rocm/licenses/NOTICE; the whole is not Apache-2.0-only
Classifier: Operating System :: POSIX :: Linux
Classifier: Environment :: GPU
Requires-Python: >=3.9
Description-Content-Type: text/markdown

# kitten-inference-tts2-rocm

The Kitten TTS 2 native engine for AMD Radeon RX 7000 (RDNA3) and RX 9000 (RDNA4) GPUs, for Linux x86_64 and WSL2.
Install it with \`pip install "kitten-inference[tts-2-rocm]"\`; the Python API is \`kitten_inference.tts2\`.

Requires the system ROCm 7.x runtime (\`libamdhip64.so.7\`, see the AMD Radeon guide in the repository). The ROCm
build has no voice cloning. Built for $KT2N_HIP_ARCHS.
EOF
cat > "$INFO/WHEEL" << EOF
Wheel-Version: 1.0
Generator: kitten-inference build_wheel_rocm.sh
Root-Is-Purelib: false
Tag: py3-none-linux_x86_64
EOF
RAW=$WORK/raw
rm -rf "$RAW" && mkdir -p "$RAW"
"$PY" - "$TREE" "$INFO" "$RAW/kitten_inference_tts2_rocm-$VERSION-py3-none-linux_x86_64.whl" << 'PY'
import base64, hashlib, os, sys, time, zipfile
tree, info, out = sys.argv[1:]
stamp = time.gmtime(max(int(os.environ["SOURCE_DATE_EPOCH"]), 315532800))[:6]
files = sorted(os.path.relpath(os.path.join(d, f), tree) for d, _, fs in os.walk(tree) for f in fs)
record = os.path.relpath(os.path.join(info, "RECORD"), tree)
files = [f for f in files if f != record]
rows = []
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for rel in files:
        data = open(os.path.join(tree, rel), "rb").read()
        zi = zipfile.ZipInfo(rel, stamp)
        zi.external_attr = ((0o755 if os.access(os.path.join(tree, rel), os.X_OK) else 0o644) | 0o100000) << 16
        zi.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(zi, data)
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append(f"{rel},sha256={digest},{len(data)}")
    rows.append(f"{record},,")
    zi = zipfile.ZipInfo(record, stamp)
    zi.external_attr = (0o644 | 0o100000) << 16
    z.writestr(zi, "\n".join(rows) + "\n", zipfile.ZIP_DEFLATED)
PY
log "auditwheel repair (libamdhip64.so.7 excluded: the system ROCm provides it)"
auditwheel repair --plat manylinux_2_28_x86_64 --only-plat --exclude libamdhip64.so.7 -w "$WORK/repaired" "$RAW"/*.whl > "$WORK/repair.log" 2>&1 \
  || { cat "$WORK/repair.log"; exit 1; }
WHL=$(ls "$WORK/repaired"/*.whl)
# For the record only: auditwheel show follows libamdhip64.so.7 into the system ROCm (show has no --exclude), so it
# reports linux_x86_64. The glibc / libstdc++ floor is enforced by repair --plat above and by the symbols check below.
auditwheel show "$WHL" > "$WORK/auditwheel-show.txt" 2>&1 || true
CHECK=$SRC/packaging/check_wheel_rocm.py
"$PY" "$CHECK" tag "$WHL" py3-none-manylinux_2_28_x86_64 | tee "$WORK/checks.txt"
if "$PY" "$CHECK" tag "$RAW"/*.whl py3-none-manylinux_2_28_x86_64 > /dev/null 2>&1; then
  echo "the tag check accepted the untagged wheel (it cannot fail)" >&2; exit 1
fi
# Nothing may be grafted into the wheel: it carries exactly the three binaries.
if "$PY" -m zipfile -l "$WHL" | grep -q '\.libs/'; then echo "auditwheel grafted libraries into the wheel" >&2; exit 1; fi

# 6. Check: a fresh virtual environment installs it, imports it, and both tools start.
rm -rf "$WORK/venv"
"$PY" -m venv "$WORK/venv"
"$WORK/venv/bin/python" -m pip install -q --no-index --no-deps "$WHL"
"$WORK/venv/bin/python" - "$VERSION" << 'PY'
import sys
import kitten_inference._tts2_rocm as m
for p in (m.LIB, m.CLI, m.TEXT_DATA_TOOL):
    assert p.is_file(), p
assert m.BACKEND == "rocm" and m.ABI == 2 and m.CAPS == (), (m.BACKEND, m.ABI, m.CAPS)
assert m.VERSION == sys.argv[1], (m.VERSION, sys.argv[1])
print("import OK:", m.VERSION, m.BUILD_INFO)
PY
CLI=$("$WORK/venv/bin/python" -c 'import kitten_inference._tts2_rocm as m; print(m.CLI)')
"$PY" "$CHECK" symbols "$CLI" "$(dirname "$CLI")/kt2n_prepare_text_data" "$(dirname "$CLI")/../lib/libkt2n_engine.so" \
  | tee -a "$WORK/checks.txt"
"$CLI" --help > "$WORK/cli-help.txt" 2>&1 || { tail -n 5 "$WORK/cli-help.txt"; echo "kitten-tts-native --help failed" >&2; exit 1; }
ldd "$CLI" | sed 's/ (0x[0-9a-f]*)//' > "$WORK/ldd-cli.txt"
ldd "$(dirname "$CLI")/../lib/libkt2n_engine.so" | sed 's/ (0x[0-9a-f]*)//' > "$WORK/ldd-lib.txt"
grep -q 'libkt2n_engine.so => .*_tts2_rocm/bin/../lib/libkt2n_engine.so' "$WORK/ldd-cli.txt" || { cat "$WORK/ldd-cli.txt"; exit 1; }

cp "$WHL" "$OUT/"
{
  echo "wheel $(basename "$WHL")"
  echo "size_bytes $(stat -c %s "$WHL")"
  echo "sha256 $(sha256sum "$WHL" | cut -d' ' -f1)"
  echo "rocm $(cat "$ROCM/.info/version" 2> /dev/null)"
  echo "hip_clang $("$HIP_CLANG" --version | head -n 1)"
  echo "archs $KT2N_HIP_ARCHS"
  echo "build_info $BUILD_INFO"
  echo "lib_sha256 $(sha256sum "$TREE/$PKG/lib/libkt2n_engine.so" | cut -d' ' -f1)"
  echo "lib_size_bytes $(stat -c %s "$TREE/$PKG/lib/libkt2n_engine.so")"
  echo "--- checks"; cat "$WORK/checks.txt"
  echo "--- auditwheel show (reports linux_x86_64 because of the system ROCm; see the script)"; cat "$WORK/auditwheel-show.txt"
  echo "--- ldd lib"; cat "$WORK/ldd-lib.txt"
  echo "--- contents"; "$PY" -m zipfile -l "$WHL"
} > "$OUT/build-rocm-wheel.txt"
[ -n "${HOST_UID:-}" ] && chown "$HOST_UID:${HOST_GID:-$HOST_UID}" "$OUT"/* 2> /dev/null || true
log "done: $OUT/$(basename "$WHL") ($(stat -c %s "$WHL") bytes)"
