#!/usr/bin/env python3
"""Writes empty stand-in wheels for the engine distributions, with the platform tags of the real ones, so that the
umbrella's extras resolve from a local folder (pip --no-index --find-links) as they would on PyPI, without building
the engines: kitten-inference-tts2-cu12 only for Linux x86-64 and Windows x64, -rocm only for Linux x86-64.

    python umbrella/tests/make_stub_wheels.py OUT_DIR [--version 0.2.0] [--old OLD_DIR]

--old also writes stand-ins of the earlier kitten-inference 0.1.0 and 0.1.1 (the TTS 0.8 engine itself, no extras,
on PyPI for every platform), into their own folder: pip falls back to them when an extra cannot be installed.
"""
import argparse
import base64
import hashlib
import zipfile
from pathlib import Path

# distribution -> wheel tags of its release wheels (CONTRACT.md); tts08 has wheels for every CI platform.
ENGINES = {
    "kitten-inference-tts08": ["py3-none-any"],
    "kitten-inference-tts2-cu12": ["py3-none-manylinux_2_28_x86_64", "py3-none-win_amd64"],
    "kitten-inference-tts2-rocm": ["py3-none-manylinux_2_28_x86_64"],
}
OLD = {"kitten-inference": ["0.1.0", "0.1.1"]}


def stub(out: Path, name: str, version: str, tag: str = "py3-none-any", requires_python: str = "") -> Path:
    dist = name.replace("-", "_")
    info = f"{dist}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\nSummary: test stub\n"
    if requires_python:
        metadata += f"Requires-Python: {requires_python}\n"
    files = {
        f"{info}/METADATA": metadata,
        f"{info}/WHEEL": f"Wheel-Version: 1.0\nGenerator: make_stub_wheels\nRoot-Is-Purelib: "
                         f"{'true' if tag == 'py3-none-any' else 'false'}\nTag: {tag}\n",
    }
    record = []
    for path, text in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b"=").decode()
        record.append(f"{path},sha256={digest},{len(text.encode())}")
    files[f"{info}/RECORD"] = "\n".join(record + [f"{info}/RECORD,,"]) + "\n"
    wheel = out / f"{dist}-{version}-{tag}.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        for path, text in files.items():
            z.writestr(path, text)
    return wheel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--version", default="0.2.0")
    p.add_argument("--old", type=Path, help="also write the 0.1.x stand-ins of kitten-inference into this folder")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name, tags in ENGINES.items():
        for tag in tags:
            print(stub(a.out, name, a.version, tag))
    if a.old:
        a.old.mkdir(parents=True, exist_ok=True)
        for name, versions in OLD.items():
            for version in versions:
                print(stub(a.old, name, version, requires_python=">=3.8"))


if __name__ == "__main__":
    main()
