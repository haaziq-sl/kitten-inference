#!/usr/bin/env python3
"""Writes empty stand-in wheels for the engine distributions, so the umbrella's extras can be resolved from a local
folder (pip --no-index --find-links) without building the engines.

    python umbrella/tests/make_stub_wheels.py OUT_DIR [--version 0.2.0]
"""
import argparse
import base64
import hashlib
import zipfile
from pathlib import Path

ENGINES = ["kitten-inference-tts08", "kitten-inference-tts2-cu12", "kitten-inference-tts2-rocm"]


def stub(out: Path, name: str, version: str) -> Path:
    dist = name.replace("-", "_")
    info = f"{dist}-{version}.dist-info"
    files = {
        f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\nSummary: test stub\n",
        f"{info}/WHEEL": "Wheel-Version: 1.0\nGenerator: make_stub_wheels\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = []
    for path, text in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b"=").decode()
        record.append(f"{path},sha256={digest},{len(text.encode())}")
    files[f"{info}/RECORD"] = "\n".join(record + [f"{info}/RECORD,,"]) + "\n"
    wheel = out / f"{dist}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        for path, text in files.items():
            z.writestr(path, text)
    return wheel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--version", default="0.2.0")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name in ENGINES:
        print(stub(a.out, name, a.version))


if __name__ == "__main__":
    main()
