#!/usr/bin/env python3
"""Size check of built wheels (a CI step): the engine payload wheels, and the kitten-inference wheels made from them.

  size_check.py WHEEL... [--max-mb 100]

Prints each wheel's size and its largest members (uncompressed and compressed), and fails when a wheel is larger than
--max-mb: by default PyPI's file limit of 100 MB, above which the upload needs a larger limit from PyPI. The limit
matters for the kitten-inference wheels, the only ones uploaded. The payloads of 0.2.0 are 31.3 MB (CUDA, Linux),
31.1 MB (CUDA, Windows) and 20.7 MB (ROCm); the kitten-inference wheels with engines are about 52.9 MB (Linux x86-64,
both engines) and 32.4 MB (Windows x64), estimated on 27 Oct 2026 from those payloads and the TTS 0.8 wheels.
"""
import argparse
import os
import sys
import zipfile

p = argparse.ArgumentParser()
p.add_argument("wheels", nargs="+")
p.add_argument("--max-mb", type=float, default=100.0)
a = p.parse_args()
ok = True
summary = []
for w in a.wheels:
    size = os.path.getsize(w)
    with zipfile.ZipFile(w) as z:
        members = sorted(z.infolist(), key=lambda i: -i.file_size)
    print(f"{os.path.basename(w)}: {size} bytes ({size / 1e6:.1f} MB)")
    for i in members[:4]:
        print(f"  {i.filename}: {i.file_size / 1e6:.1f} MB, {i.compress_size / 1e6:.1f} MB compressed")
    line = f"{os.path.basename(w)} {size / 1e6:.1f} MB"
    if size > 100e6:
        line += " (above PyPI's default 100 MB limit)"
    if size > a.max_mb * 1e6:
        print(f"::error::{os.path.basename(w)} is {size / 1e6:.1f} MB, above {a.max_mb:.0f} MB")
        ok = False
    summary.append(line)
step = os.environ.get("GITHUB_STEP_SUMMARY")
if step:
    with open(step, "a") as f:
        f.write("### Wheel sizes\n" + "".join(f"- {s}\n" for s in summary))
sys.exit(0 if ok else 1)
