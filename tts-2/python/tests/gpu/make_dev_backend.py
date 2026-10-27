#!/usr/bin/env python3
"""Packs an engine build directory into a development wheel of the engine package (``kitten_inference._tts2_cuda``
or ``_tts2_rocm``), with the layout of the release wheels, for the GPU tests of the Python API.

    python tests/gpu/make_dev_backend.py --build ../build/default --out dist/ [--backend cuda]

It takes lib*/libkt2n_engine.so (or kt2n_engine.dll), bin/kitten-tts-native and bin/kt2n_prepare_text_data from
the build directory. The wheel is a local distribution, kitten-inference-dev-cuda (or -dev-rocm) 0.0.0+dev, which
is never uploaded. Install it in an environment without the kitten-inference release wheel, which has its own engine
files at the same paths. From tts-2/python, for example:

    python -m venv dev && . dev/bin/activate
    pip install ../../umbrella . dist/kitten_inference_dev_cuda-0.0.0+dev-*.whl pytest
    python -m pytest tests/gpu

The release wheels are made by tts-2/packaging and .github/scripts/assemble_wheels.py; this is only for testing a
local build.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

EXE = ".exe" if os.name == "nt" else ""
LIB = "kt2n_engine.dll" if os.name == "nt" else "libkt2n_engine.so"
DIST = {"cuda": "kitten_inference_dev_cuda", "rocm": "kitten_inference_dev_rocm"}
DIST_VERSION = "0.0.0+dev"  # a local version: never the release's, never on an index

INIT = '''"""Kitten TTS 2 engine ({backend}): development build packed by tests/gpu/make_dev_backend.py."""
from pathlib import Path

_here = Path(__file__).resolve().parent
BACKEND = "{backend}"
ABI = 2
VERSION = "{version}"
LIB = _here / "lib" / "{lib}"
CLI = _here / "bin" / "kitten-tts-native{exe}"
TEXT_DATA_TOOL = _here / "bin" / "kt2n_prepare_text_data{exe}"
BUILD_INFO = {build_info!r}
CAPS = {caps!r}
'''


def find(build: Path, name: str) -> Path:
    for candidate in [build / "bin" / name, build / "lib" / name, build / name, *build.rglob(name)]:
        if candidate.is_file():
            return candidate
    raise SystemExit(f"{name} not found under {build}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backend", choices=sorted(DIST), default="cuda")
    p.add_argument("--version", help="the engine's VERSION (default: the one in its build info)")
    a = p.parse_args()
    lib = find(a.build, LIB)
    cli = find(a.build, "kitten-tts-native" + EXE)
    tool = find(a.build, "kt2n_prepare_text_data" + EXE)
    version_out = subprocess.run([str(cli), "--version"], capture_output=True, text=True, check=True).stdout
    build_info = version_out.split("\n", 1)[1].strip() if "\n" in version_out else version_out.strip()
    caps = ("voice_cloning",) if "cap:voice.cloning" in build_info.split() else ()
    words = build_info.split()  # "Kitten TTS 2 native 0.2.0 cap:..."
    version = a.version or (words[4] if words[:4] == ["Kitten", "TTS", "2", "native"] and len(words) > 4
                            else "0.0.0")
    pkg = f"kitten_inference/_tts2_{a.backend}"
    files = {
        f"{pkg}/__init__.py": INIT.format(backend=a.backend, version=version, lib=LIB, exe=EXE,
                                          build_info=build_info, caps=caps).encode(),
        f"{pkg}/lib/{LIB}": lib.read_bytes(),
        f"{pkg}/bin/kitten-tts-native{EXE}": cli.read_bytes(),
        f"{pkg}/bin/kt2n_prepare_text_data{EXE}": tool.read_bytes(),
    }
    if os.name != "nt":  # a versioned build: the tool needs the library under its SONAME too (bin/../lib)
        needed = subprocess.run(["readelf", "-d", str(cli)], capture_output=True, text=True).stdout
        for name in sorted(set(re.findall(r"\[(libkt2n_engine\.so[.0-9]*)\]", needed)) - {LIB}):
            files[f"{pkg}/lib/{name}"] = lib.read_bytes()
    dist = DIST[a.backend]
    name = dist.replace("_", "-")
    info = f"{dist}-{DIST_VERSION}.dist-info"
    tag = "py3-none-" + ("win_amd64" if os.name == "nt" else "linux_x86_64")
    files[f"{info}/METADATA"] = (f"Metadata-Version: 2.1\nName: {name}\nVersion: {DIST_VERSION}\n"
                                 "Summary: development build of the Kitten TTS 2 engine\n").encode()
    files[f"{info}/WHEEL"] = f"Wheel-Version: 1.0\nGenerator: make_dev_backend\nRoot-Is-Purelib: false\nTag: {tag}\n".encode()
    record = []
    for path, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        record.append(f"{path},sha256={digest},{len(data)}")
    record.append(f"{info}/RECORD,,")
    files[f"{info}/RECORD"] = ("\n".join(record) + "\n").encode()
    a.out.mkdir(parents=True, exist_ok=True)
    wheel = a.out / f"{dist}-{DIST_VERSION}-{tag}.whl"
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as z:
        for path, data in files.items():
            zi = zipfile.ZipInfo(path, date_time=(2026, 1, 1, 0, 0, 0))
            zi.external_attr = ((0o755 if "/bin/" in path else 0o644) | 0o100000) << 16  # S_IFREG: pip keeps +x
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, data)
    print(wheel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
