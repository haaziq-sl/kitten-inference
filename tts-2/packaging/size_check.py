#!/usr/bin/env python3
"""Size check of built kitten-inference-tts2-cu12 wheels (a CI step).

  size_check.py WHEEL... [--max-mb 300]

Prints each wheel's size and its largest members (uncompressed and compressed). PyPI's default file limit is 100 MB:
a wheel above it is reported as a warning (the project then needs a larger limit from PyPI). Above --max-mb the
check fails: the wheel has grown far beyond what the size work measured (tts-2/packaging/README.md).
"""
import argparse
import os
import sys
import zipfile

p = argparse.ArgumentParser()
p.add_argument("wheels", nargs="+")
p.add_argument("--max-mb", type=float, default=300.0)
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
        print(f"::warning::{os.path.basename(w)} is {size / 1e6:.1f} MB, above PyPI's default 100 MB file limit")
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
