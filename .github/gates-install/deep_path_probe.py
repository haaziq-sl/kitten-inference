#!/usr/bin/env python3
"""Does TTS 0.8 still speak when its venv sits in a deep folder? espeak-ng (from espeakng_loader) is told its data
folder through ESPEAK_DATA_PATH; this measures, per length of that folder's path, whether (a) phonemizer + espeak-ng
alone and (b) KittenTTS(...).generate() work, each in its own process.

    deep_path_probe.py WHEELS OUT.json PYTHON [--kittenml]
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

WIN = sys.platform == "win32"
wheels, out, py = sys.argv[1:4]
with_kittenml = "--kittenml" in sys.argv
ver = subprocess.run([py, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], stdout=subprocess.PIPE,
                     check=True).stdout.decode().strip()
tail = os.path.join("Lib", "site-packages") if WIN else os.path.join("lib", f"python{ver}", "site-packages")
tail = os.path.join(tail, "espeakng_loader", "espeak-ng-data")

ESPEAK_ONLY = r'''
import os, espeakng_loader
from phonemizer.backend.espeak.wrapper import EspeakWrapper
EspeakWrapper.set_library(espeakng_loader.get_library_path())
os.environ["ESPEAK_DATA_PATH"] = espeakng_loader.get_data_path()
import phonemizer
print(len(espeakng_loader.get_data_path()), phonemizer.phonemize("hello world", language="en-us", backend="espeak"))
'''
OURS = r'''
from kitten_inference.tts08 import KittenTTS
a = KittenTTS("KittenML/kitten-tts-nano-0.8").generate("Hello from a deep folder.", voice="Jasper")
print("audio_s", round(len(a) / 24000, 3))
'''
KITTENML = r'''
from kittenml import KittenTTS
a = KittenTTS("KittenML/kitten-tts-nano-0.8").generate("Hello from a deep folder.", voice="Jasper")
print("audio_s", round(len(a) / 24000, 3))
'''


def run(cmd, timeout=900):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    lines = p.stdout.decode("utf-8", "replace").strip().splitlines()
    return {"rc": p.returncode, "tail": [ln for ln in lines if "unauthenticated requests" not in ln][-2:]}


base = Path(tempfile.mkdtemp(prefix="dp-"))
rows = []
for target in (120, 140, 145, 148, 150, 152, 154, 156, 158, 160, 170, 200, 230):
    pad = target - len(str(base)) - len(os.sep) - len(os.sep) - len(tail) - 1  # base/v<pad x>/tail
    if pad < 0:
        continue
    venv = base / ("v" + "x" * pad)
    row = {"target": target, "data_path_len": len(os.path.join(str(venv), tail))}
    try:
        r = subprocess.run([py, "-m", "venv", str(venv)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        vpy = str(venv / ("Scripts" if WIN else "bin") / ("python.exe" if WIN else "python"))
        inst = run([vpy, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--find-links", wheels,
                    "kitten-inference[tts-0.8]"])
        if with_kittenml and inst["rc"] == 0:
            # the ONNX package without its torch dependency (its TTS 0.8 path does not import torch)
            inst = run([vpy, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-deps", "kittenml==0.9.6"])
            if inst["rc"] == 0:
                inst = run([vpy, "-m", "pip", "install", "-q", "--disable-pip-version-check", "onnxruntime"])
        row["install"] = inst
        if inst["rc"] == 0:
            row["espeak_only"] = run([vpy, "-c", ESPEAK_ONLY])
            row["kitten_inference_tts08"] = run([vpy, "-c", OURS])
            if with_kittenml:
                row["kittenml_0.9.6"] = run([vpy, "-c", KITTENML])
    except Exception as e:  # e.g. a path Windows cannot create
        row["error"] = f"{type(e).__name__}: {e}"[:300]
    rows.append(row)
    print(json.dumps(row)[:600], flush=True)
    shutil.rmtree(venv, ignore_errors=True)
Path(out).write_text(json.dumps({"base": str(base), "python": ver, "rows": rows}, indent=1))
