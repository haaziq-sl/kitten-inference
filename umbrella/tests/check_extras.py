#!/usr/bin/env python3
"""Installs the umbrella with each extra into a fresh virtual environment, from local folders only, and checks
which kitten-inference distributions pip chose.

    python umbrella/tests/check_extras.py --find-links dist --find-links stubs --find-links deps [--version 0.2.0]

dist: the built umbrella and kitten-inference-tts2 wheels; stubs: make_stub_wheels.py output; deps: a wheelhouse
of kitten-inference-tts2's own dependencies (pip download). Exit status 1 when any extra resolves wrongly.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED = {
    "": {"kitten-inference"},
    "[tts-0.8]": {"kitten-inference", "kitten-inference-tts08"},
    "[tts-0-8]": {"kitten-inference", "kitten-inference-tts08"},
    "[tts-2]": {"kitten-inference", "kitten-inference-tts2", "kitten-inference-tts2-cu12"},
    "[tts-2-rocm]": {"kitten-inference", "kitten-inference-tts2", "kitten-inference-tts2-rocm"},
    "[tts-0.8,tts-2]": {"kitten-inference", "kitten-inference-tts08", "kitten-inference-tts2",
                        "kitten-inference-tts2-cu12"},
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--find-links", action="append", type=Path, required=True)
    p.add_argument("--version", default="0.2.0")
    a = p.parse_args()
    links = [arg for d in a.find_links for arg in ("--find-links", str(d.resolve()))]
    failed = 0
    for extra, expected in EXPECTED.items():
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, "-m", "venv", tmp], check=True)
            py = str(Path(tmp) / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
            spec = f"kitten-inference{extra}=={a.version}"
            run = subprocess.run([py, "-m", "pip", "install", "--disable-pip-version-check", "--no-index", *links,
                                  spec], capture_output=True, text=True)
            listed = subprocess.run([py, "-m", "pip", "list", "--format", "json"], capture_output=True, text=True)
            names = {d["name"].lower().replace("_", "-") for d in json.loads(listed.stdout or "[]")}
            ours = {n for n in names if n.startswith("kitten-inference")}
            ok = run.returncode == 0 and ours == expected
            warnings = [l for l in run.stdout.splitlines() + run.stderr.splitlines() if "WARNING" in l]
            print(f"{'ok  ' if ok else 'FAIL'} pip install \"{spec}\" -> {sorted(ours)}"
                  + (f"  [{'; '.join(warnings)}]" if warnings else ""))
            if not ok:
                failed += 1
                print(run.stdout[-3000:], run.stderr[-3000:], sep="\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
