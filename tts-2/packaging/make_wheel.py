#!/usr/bin/env python3
"""Assemble the kitten-inference-tts2-cu12 wheel from an installed wheel-preset build (standard library only).

  make_wheel.py --stage STAGE --plat manylinux_2_28_x86_64|win_amd64 --out DIR [--version 0.2.0]

STAGE is `cmake --install build/<wheel preset> --prefix STAGE`: bin/kitten-tts-native(.exe),
bin/kt2n_prepare_text_data(.exe) and the engine library (lib/libkt2n_engine.so, or bin/kt2n_engine.dll on Windows,
which goes to lib/ in the wheel). The wheel holds exactly the package layout of the kitten-inference contract:

  kitten_inference/_tts2_cuda/__init__.py   BACKEND, ABI, VERSION, LIB, CLI, TEXT_DATA_TOOL, BUILD_INFO, CAPS
  kitten_inference/_tts2_cuda/lib/          the engine library
  kitten_inference/_tts2_cuda/bin/          the command-line tool and the text-data helper

plus the licences in the .dist-info folder. BUILD_INFO is the library's kt2n_build_info() text, read here with ctypes
(it needs no GPU or driver). The archive is deterministic: sorted entries, fixed timestamps (SOURCE_DATE_EPOCH, else
the time of the source commit given with --epoch, else 2026-10-25).
"""
from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import os
import sys
import zipfile
from pathlib import Path

NAME = "kitten-inference-tts2-cu12"
PKG = "kitten_inference/_tts2_cuda"
SOURCE = Path(__file__).resolve().parents[1]  # tts-2/

INIT = '''"""Kitten TTS 2 native engine for NVIDIA GPUs (CUDA 12.8, NVIDIA driver R570 or newer): the engine library, its
command-line tool and the text-data helper. Use it through kitten_inference.tts2 (pip install "kitten-inference[tts-2]").
"""
import sys
from pathlib import Path

BACKEND = "cuda"
ABI = 2
VERSION = {version!r}
_HERE = Path(__file__).parent
_WINDOWS = sys.platform == "win32"
LIB = _HERE / "lib" / ("kt2n_engine.dll" if _WINDOWS else "libkt2n_engine.so")
CLI = _HERE / "bin" / ("kitten-tts-native.exe" if _WINDOWS else "kitten-tts-native")
TEXT_DATA_TOOL = _HERE / "bin" / ("kt2n_prepare_text_data.exe" if _WINDOWS else "kt2n_prepare_text_data")
BUILD_INFO = {build_info!r}
CAPS = {caps!r}
'''

README = """# kitten-inference-tts2-cu12

The Kitten TTS 2 native engine for NVIDIA GPUs, with its command-line tool. On an RTX 4090 with the current model,
this package streams its first audio after **19 ms** (median of 5 runs). Complete files take 6 to 10 times less time
than with Kitten TTS 2 on vLLM or llama.cpp (one request at a time on an RTX PRO 6000, earlier 47-voice model).

Install it with the Python API, which uses it:

```sh
python -m pip install -U pip                 # older pip versions can skip an extra with only a warning
pip install "kitten-inference[tts-2]"
```

| NVIDIA GPU | Status |
|---|---|
| Compute capability 8.0 or newer (GeForce RTX 30 series and newer) | supported |
| GeForce GTX 10, GTX 16 and RTX 20 series (compute capability 6.1 and 7.5) | supported, not optimal: no bf16 tensor cores, so part of the math runs in software. Real-time factor 0.22 to 0.76 with `--stream false`, against 0.03 to 0.17 for RTX 30 series and newer. The audio is close to, not bit for bit the same as, a newer GPU's |

It needs NVIDIA driver R570 or newer; the CUDA 12.8 runtime is built in. Voice cloning reads the reference audio with
the `ffmpeg` program (`sudo apt-get install -y ffmpeg`, or on Windows `winget install --id Gyan.FFmpeg -e`).
"""


def record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def build_info(lib: Path) -> str:
    if os.name == "nt":
        os.add_dll_directory(str(lib.parent))
    engine = ctypes.CDLL(str(lib))
    engine.kt2n_build_info.restype = ctypes.c_char_p
    engine.kt2n_abi_version.restype = ctypes.c_uint32
    abi = engine.kt2n_abi_version()
    if abi != 2:
        sys.exit(f"{lib}: ABI {abi}, expected 2")
    return engine.kt2n_build_info().decode()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--plat", required=True, choices=["manylinux_2_28_x86_64", "win_amd64"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--version", default="0.2.0")
    p.add_argument("--epoch", type=int, help="timestamp for the archive entries (default SOURCE_DATE_EPOCH)")
    a = p.parse_args()

    windows = a.plat == "win_amd64"
    exe = ".exe" if windows else ""
    stage = a.stage.resolve()
    lib_src = stage / ("bin/kt2n_engine.dll" if windows else "lib/libkt2n_engine.so")
    files = {  # path in the wheel -> source file
        f"{PKG}/lib/{lib_src.name}": lib_src,
        f"{PKG}/bin/kitten-tts-native{exe}": stage / f"bin/kitten-tts-native{exe}",
        f"{PKG}/bin/kt2n_prepare_text_data{exe}": stage / f"bin/kt2n_prepare_text_data{exe}",
    }
    for dest, src in files.items():
        if not src.is_file() or src.is_symlink():
            sys.exit(f"missing (or a symlink): {src}")
    info = build_info(lib_src)
    caps = ("voice_cloning",) if "cap:voice.cloning" in info.split() else ()
    if not caps:
        sys.exit(f"the library has no voice cloning: {info}")

    dist = NAME.replace("-", "_")
    distinfo = f"{dist}-{a.version}.dist-info"
    tag = f"py3-none-{a.plat}"
    entries: dict[str, tuple[bytes, bool]] = {}  # wheel path -> (data, executable)
    entries[f"{PKG}/__init__.py"] = (INIT.format(version=a.version, build_info=info, caps=caps).encode(), False)
    for dest, src in files.items():
        entries[dest] = (src.read_bytes(), "/bin/" in dest and not windows)
    metadata = "\n".join([
        "Metadata-Version: 2.1",
        f"Name: {NAME}",
        f"Version: {a.version}",
        "Summary: Kitten TTS 2 native engine for NVIDIA GPUs (CUDA 12.8): library and command-line tool",
        "Home-page: https://github.com/KittenML/kitten-inference",
        "Author: KittenML",
        "License: Apache-2.0 (KittenML code); bundled third-party components keep their licenses (see NOTICE)",
        "Classifier: Environment :: GPU :: NVIDIA CUDA :: 12",
        "Classifier: Programming Language :: Python :: 3",
        "Classifier: Operating System :: " + ("Microsoft :: Windows" if windows else "POSIX :: Linux"),
        "Classifier: Topic :: Multimedia :: Sound/Audio :: Speech",
        "Requires-Python: >=3.9",
        "Description-Content-Type: text/markdown",
        "", README])
    entries[f"{distinfo}/METADATA"] = (metadata.encode(), False)
    entries[f"{distinfo}/WHEEL"] = ("\n".join(["Wheel-Version: 1.0", "Generator: kitten-inference make_wheel.py",
                                               "Root-Is-Purelib: false", f"Tag: {tag}", ""]).encode(), False)
    entries[f"{distinfo}/top_level.txt"] = (b"kitten_inference\n", False)
    for lic in [SOURCE / "LICENSE", SOURCE / "NOTICE", *sorted((SOURCE / "licenses").glob("*.txt"))]:
        rel = lic.name if lic.parent == SOURCE else f"licenses/{lic.name}"
        entries[f"{distinfo}/licenses/{rel}"] = (lic.read_bytes(), False)
    record = [f"{path},{record_hash(data)},{len(data)}" for path, (data, _) in sorted(entries.items())]
    record.append(f"{distinfo}/RECORD,,")
    entries[f"{distinfo}/RECORD"] = (("\n".join(record) + "\n").encode(), False)

    epoch = a.epoch or int(os.environ.get("SOURCE_DATE_EPOCH", "1792886400"))  # 2026-10-25T00:00:00Z
    import time
    stamp = time.gmtime(max(epoch, 315532800))[:6]
    a.out.mkdir(parents=True, exist_ok=True)
    wheel = a.out / f"{dist}-{a.version}-{tag}.whl"
    order = sorted(p for p in entries if not p.startswith(distinfo)) + sorted(p for p in entries if p.startswith(distinfo)
                                                                              and not p.endswith("RECORD"))
    order.append(f"{distinfo}/RECORD")
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in order:
            data, executable = entries[path]
            zi = zipfile.ZipInfo(path, date_time=stamp)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = ((0o755 if executable else 0o644) | 0o100000) << 16
            z.writestr(zi, data, compresslevel=9)
    size = wheel.stat().st_size
    print(f"{wheel} {size} bytes ({size / 1e6:.1f} MB); library {lib_src.stat().st_size} bytes; {info}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
