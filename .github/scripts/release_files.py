#!/usr/bin/env python3
"""Release uploads: copies the kitten-inference wheels of a v* tag from the assembled artifacts to the upload folder.

    python .github/scripts/release_files.py SRC DEST [--tag v0.2.0]

SRC is searched recursively. Only kitten_inference-<version>-cp*.whl files are copied, and only when their version is
the tag's: a tag that names another version is an error, as is a SRC that holds anything a release must never upload
(a kitten-inference sdist or py3-none-any wheel, which would install without any engine, or any other distribution's
wheel or sdist, such as the internal TTS 0.8 and engine builds the wheels are assembled from), several versions, or
the same file twice. DEST must be empty. --tag defaults to $GITHUB_REF_NAME.
"""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

NAME = "kitten-inference"


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse(filename: str):
    """(distribution, version, python tag, platform tag) of a wheel file name, (distribution, version, None, None) of
    an sdist, None for any other file."""
    if filename.endswith(".whl"):
        parts = filename[:-4].split("-")
        if len(parts) not in (5, 6):
            return None
        return normalise(parts[0]), parts[1], parts[-3], parts[-1]
    if filename.endswith(".tar.gz"):
        name, sep, version = filename[:-7].rpartition("-")
        return (normalise(name), version, None, None) if sep else None
    return None


def select(src: Path, tag: str) -> list:
    """The wheels the tag releases, sorted by name; SystemExit for a tag or a set of files that must not be released."""
    if not tag.startswith("v") or len(tag) < 2:
        raise SystemExit(f"error: {tag or 'no tag'}: a release tag is v<version>, for example v0.2.0")
    version = tag[1:]
    found = {}
    for path in sorted(src.rglob("*")):
        info = parse(path.name) if path.is_file() else None
        if info is None:
            continue
        dist, _, py, _ = info
        if dist != NAME:
            raise SystemExit(f"error: {path}: a {dist} file; a release uploads only kitten-inference wheels")
        if py is None:
            raise SystemExit(f"error: {path}: a kitten-inference sdist would install without any engine")
        if not py.startswith("cp"):
            raise SystemExit(f"error: {path}: a {py} kitten-inference wheel would install without any engine")
        if path.name in found:
            raise SystemExit(f"error: {path.name} is in {src} twice ({found[path.name]}, {path})")
        found[path.name] = path
    if not found:
        raise SystemExit(f"error: no kitten-inference wheel in {src}")
    versions = sorted({parse(n)[1] for n in found})
    if len(versions) != 1:
        raise SystemExit(f"error: {src} holds several versions: {', '.join(versions)}")
    if versions[0] != version:
        raise SystemExit(f"error: the wheels are version {versions[0]}, but the tag {tag} names {version}")
    return [found[n] for n in sorted(found)]


def copy(files: list, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise SystemExit(f"error: {dest} is not empty")
    for f in files:
        shutil.copy2(f, dest / f.name)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("--tag", default=os.environ.get("GITHUB_REF_NAME", ""))
    a = p.parse_args()
    files = select(a.src, a.tag)
    copy(files, a.dest)
    for f in files:
        print(f"{f.name}  {f.stat().st_size} bytes")
    print(f"{len(files)} kitten-inference {a.tag[1:]} wheels in {a.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
