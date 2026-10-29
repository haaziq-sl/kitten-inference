#!/usr/bin/env python3
"""Writes stand-ins of the earlier kitten-inference 0.1.0 and 0.1.1 (on PyPI for every platform: the TTS 0.8 engine
itself, no extras), so that check_extras.py --old can serve them beside the new wheels, as PyPI does: pip falls back
to them on a system that has no wheel of the new version, unless they are yanked.

    python tests/packaging/make_stub_wheels.py [OUT] --old OLD_DIR

The stand-ins go into OLD_DIR (or into OUT when --old is not given). There are no engine stand-ins any more: the
engines are in the kitten-inference wheels themselves, so with --old nothing is written into OUT.
"""
import argparse
import base64
import hashlib
import zipfile
from pathlib import Path

OLD = {"kitten-inference": ["0.1.0", "0.1.1"]}


def stub(out: Path, name: str, version: str, requires_python: str = "") -> Path:
    dist = name.replace("-", "_")
    info = f"{dist}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\nSummary: test stub\n"
    if requires_python:
        metadata += f"Requires-Python: {requires_python}\n"
    files = {
        f"{info}/METADATA": metadata,
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
    p.add_argument("out", type=Path, nargs="?", help="written only when --old is not given")
    p.add_argument("--old", type=Path, help="write the 0.1.x stand-ins of kitten-inference into this folder")
    a = p.parse_args()
    target = a.old or a.out
    if target is None:
        p.error("give --old OLD_DIR (or OUT)")
    target.mkdir(parents=True, exist_ok=True)
    for name, versions in OLD.items():
        for version in versions:
            print(stub(target, name, version, requires_python=">=3.8"))


if __name__ == "__main__":
    main()
