#!/usr/bin/env python3
"""Smoke test of a Kitten TTS 2 engine in the installed kitten-inference (run with the venv's python).

  smoke_wheel.py [--backend cuda|rocm]  layout, version, build info, the programs start (no GPU needed)
  smoke_wheel.py [--backend cuda|rocm] --assets DIR --out hello.wav
                                        also generates speech with the engine's command-line tool (needs that GPU with
                                        its driver or runtime, and a prepared model directory)

--backend picks the package: kitten_inference._tts2_cuda (NVIDIA, the default) or kitten_inference._tts2_rocm (AMD,
Linux only). It is the same in a kitten-inference wheel and in the payload wheel an engine build makes (README.md).
VERSION and the version in BUILD_INFO must be the engine's version, project VERSION in tts-2/CMakeLists.txt.

The CUDA library loads without a GPU or driver. The ROCm library needs the ROCm 7 runtime (libamdhip64.so.7): on a
machine without it, such as the CI runners, the check is that the library and the command-line tool fail naming that
file, as Kitten TTS 2 then tells users; with it, the library gets the same checks as the CUDA one.

When the other engine is installed too (Linux x86-64 wheels), both text-data tools must write the same bytes from the
same input, because one prepared model folder serves both engines.
"""
import argparse
import ctypes
import importlib
import importlib.util
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--backend", choices=("cuda", "rocm"), default="cuda")
p.add_argument("--assets")
p.add_argument("--out", default="hello.wav")
a = p.parse_args()
if a.backend == "rocm" and not sys.platform.startswith("linux"):
    sys.exit("--backend rocm: the AMD (ROCm) engine is in the Linux x86-64 wheels only")

# The engine's version, read as make_wheel.py reads it. A copy of this script outside the repository (a GPU test host)
# can only compare VERSION with the library's build info.
cmake = Path(__file__).resolve().parents[1] / "CMakeLists.txt"
engine_version = None
if cmake.is_file():
    found = re.search(r"^project\(kt2n\s+VERSION\s+([0-9.]+)", cmake.read_text(encoding="utf-8"), re.M)
    assert found, f"no project VERSION in {cmake}"
    engine_version = found.group(1)
else:
    print(f"{cmake} not found: VERSION is compared with the build info only")

m = importlib.import_module(f"kitten_inference._tts2_{a.backend}")
print(f"checking {Path(m.__file__).parent}")
assert m.BACKEND == a.backend and m.ABI == 2, (m.BACKEND, m.ABI)
info = m.BUILD_INFO.split()
assert info[:4] == ["Kitten", "TTS", "2", "native"] and len(info) > 4, m.BUILD_INFO
engine_version = engine_version or info[4]
assert m.VERSION == info[4] == engine_version, f"VERSION {m.VERSION}, build info {info[4]}, engine {engine_version}"
for path in (m.LIB, m.CLI, m.TEXT_DATA_TOOL):
    assert path.is_file(), path
if os.name != "nt":
    for path in (m.CLI, m.TEXT_DATA_TOOL):
        assert os.access(path, os.X_OK), f"not executable: {path}"
if a.backend == "cuda":
    assert m.CAPS == ("voice_cloning",), m.CAPS
    assert "cap:voice.cloning" in info and "cap:voice.decoded_pcm_only" in info, m.BUILD_INFO
else:
    assert m.CAPS == () and "cap:voice.cloning" not in info, (m.CAPS, m.BUILD_INFO)
    print(f"GPU targets {' '.join(m.HIP_ARCHS)}")
try:
    lib = ctypes.CDLL(str(m.LIB))
except OSError as e:
    # Only the ROCm library may fail to load, and only for want of the ROCm runtime.
    if a.backend != "rocm" or "libamdhip64.so.7" not in str(e):
        raise
    if a.assets:
        sys.exit(f"--assets needs the ROCm 7 runtime: {e}")
    lib = None
    print(f"no ROCm runtime here: {e}")
cli = subprocess.run([str(m.CLI), "--version"], capture_output=True, text=True)
if lib is not None:
    lib.kt2n_build_info.restype = ctypes.c_char_p
    lib.kt2n_abi_version.restype = ctypes.c_uint32
    assert lib.kt2n_abi_version() == 2
    assert lib.kt2n_build_info().decode() == m.BUILD_INFO
    assert cli.returncode == 0 and m.BUILD_INFO in cli.stdout, (cli.returncode, cli.stdout, cli.stderr[-300:])
    version = cli.stdout.splitlines()[0]
else:
    out = cli.stdout + cli.stderr
    assert cli.returncode != 0 and "libamdhip64.so.7" in out, (cli.returncode, out[-300:])
    version = f"{m.CLI.name} names libamdhip64.so.7"
tool = subprocess.run([str(m.TEXT_DATA_TOOL)], capture_output=True, text=True)
assert "usage: kt2n_prepare_text_data" in tool.stderr, (tool.returncode, tool.stderr[-300:])
# The text-data tool reads and writes paths outside the ANSI code page (on Windows through its UTF-8 code page
# manifest): a one-state graph converted from and into a folder whose name is not ASCII.
if os.name == "nt":
    for exe in (m.CLI, m.TEXT_DATA_TOOL):
        assert b"activeCodePage" in exe.read_bytes(), f"no UTF-8 code page manifest in {exe.name}"
    print(f"system ANSI code page {ctypes.windll.kernel32.GetACP()}; both programs carry the UTF-8 manifest")
folder = Path(tempfile.mkdtemp()) / "Łukasz 田中"
folder.mkdir()
(folder / "in.fst").write_bytes(b"KITTEN1\0" + struct.pack("<IIIIIf", 0, 1, 0, 0, 0, 0.0))
tool = subprocess.run([str(m.TEXT_DATA_TOOL), str(folder / "in.fst"), str(folder / "out.fst")], capture_output=True,
                      text=True, encoding="utf-8", errors="replace")
assert tool.returncode == 0, (tool.returncode, tool.stderr[-300:])
assert (folder / "out.fst").stat().st_size > 0
print("text-data tool OK with a non-ASCII folder name")
other = "rocm" if a.backend == "cuda" else "cuda"
if importlib.util.find_spec(f"kitten_inference._tts2_{other}") is not None:
    o = importlib.import_module(f"kitten_inference._tts2_{other}")
    tool = subprocess.run([str(o.TEXT_DATA_TOOL), str(folder / "in.fst"), str(folder / "other.fst")],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert tool.returncode == 0, (tool.returncode, tool.stderr[-300:])
    assert (folder / "other.fst").read_bytes() == (folder / "out.fst").read_bytes(), \
        f"the {other} engine's text-data tool writes other bytes"
    print(f"the {other} engine's text-data tool writes the same bytes")
shutil.rmtree(folder.parent)
print(f"layout OK: {m.LIB.name} {m.LIB.stat().st_size} bytes; {version}; CAPS {m.CAPS}")

if a.assets:
    r = subprocess.run([str(m.CLI), "--assets", a.assets, "--text", "Hello from the Kitten TTS 2 wheel.", "--voice",
                        "Bruno", "--output", a.out], capture_output=True, text=True)
    sys.stderr.write(r.stderr[-2000:])
    assert r.returncode == 0, r.returncode
    with wave.open(a.out) as w:
        assert w.getframerate() == 24000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        seconds = w.getnframes() / 24000
    assert seconds > 0.5, seconds
    print(f"generated {a.out}: {seconds:.2f} s of audio")
print(f"SMOKE OK ({a.backend})")
