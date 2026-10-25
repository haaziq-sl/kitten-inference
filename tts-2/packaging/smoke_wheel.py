#!/usr/bin/env python3
"""Smoke test of an installed kitten-inference-tts2-cu12 wheel (run with the venv's python).

  smoke_wheel.py                      layout, build info, the programs start (no GPU needed)
  smoke_wheel.py --assets DIR --out hello.wav
                                      also generates speech with the bundled command-line tool (needs an NVIDIA GPU
                                      and a prepared model directory)
"""
import argparse
import ctypes
import os
import subprocess
import sys
import wave

p = argparse.ArgumentParser()
p.add_argument("--assets")
p.add_argument("--out", default="hello.wav")
a = p.parse_args()

import kitten_inference._tts2_cuda as m  # noqa: E402

assert m.BACKEND == "cuda" and m.ABI == 2, (m.BACKEND, m.ABI)
for path in (m.LIB, m.CLI, m.TEXT_DATA_TOOL):
    assert path.is_file(), path
if os.name != "nt":
    for path in (m.CLI, m.TEXT_DATA_TOOL):
        assert os.access(path, os.X_OK), f"not executable: {path}"
assert m.CAPS == ("voice_cloning",), m.CAPS
assert "cap:voice.cloning" in m.BUILD_INFO.split() and "cap:voice.decoded_pcm_only" in m.BUILD_INFO.split(), m.BUILD_INFO
lib = ctypes.CDLL(str(m.LIB))
lib.kt2n_build_info.restype = ctypes.c_char_p
lib.kt2n_abi_version.restype = ctypes.c_uint32
assert lib.kt2n_abi_version() == 2
assert lib.kt2n_build_info().decode() == m.BUILD_INFO
version = subprocess.run([str(m.CLI), "--version"], capture_output=True, text=True, check=True).stdout
assert m.BUILD_INFO in version, version
tool = subprocess.run([str(m.TEXT_DATA_TOOL)], capture_output=True, text=True)
assert "usage: kt2n_prepare_text_data" in tool.stderr, (tool.returncode, tool.stderr[-300:])
print(f"layout OK: {m.LIB.name} {m.LIB.stat().st_size} bytes; {version.splitlines()[0]}; CAPS {m.CAPS}")

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
print("SMOKE OK")
