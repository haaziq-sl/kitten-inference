#!/usr/bin/env python3
"""Release uploads: picks one distribution's files from a workflow's artifacts and says whether this tag releases them.

    python .github/scripts/release_files.py ARTIFACT_DIR UPLOAD_DIR --dist kitten-inference-tts2-cu12 \
        [--platform manylinux_2_28_x86_64 --platform win_amd64] [--tag v0.2.0]

A tag v<version> releases a distribution only when the files built for it have exactly that version, so a tag such
as v0.2.0.post1 can carry the post release of one distribution while the others keep their version and are left out
(RELEASING.md). The files are copied to UPLOAD_DIR and `upload=true` or `upload=false` goes to $GITHUB_OUTPUT. It
fails when ARTIFACT_DIR has no files of the distribution, when they have different versions, or when a --platform has
no wheel. --tag defaults to $GITHUB_REF_NAME.
"""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse(filename: str):
    """(distribution, version) of a wheel or sdist file name, or None for any other file."""
    if filename.endswith(".whl"):
        parts = filename[:-4].split("-")
        return (normalise(parts[0]), parts[1]) if len(parts) >= 5 else None
    if filename.endswith(".tar.gz"):
        name, sep, version = filename[:-7].rpartition("-")
        return (normalise(name), version) if sep else None
    return None


def select(artifacts: Path, dist: str):
    """The files of `dist` under `artifacts` (searched recursively), sorted by name."""
    found = {}
    for path in sorted(artifacts.rglob("*")):
        info = parse(path.name) if path.is_file() else None
        if info and info[0] == normalise(dist):
            if path.name in found:
                raise SystemExit(f"error: {path.name} is in the artifacts twice ({found[path.name]}, {path})")
            found[path.name] = path
    return [found[name] for name in sorted(found)]


def decide(files, tag: str, platforms):
    """(version, upload?) for the selected files and the tag; raises SystemExit for a broken set of files."""
    if not files:
        raise SystemExit("error: no wheel or sdist of this distribution in the artifacts")
    versions = sorted({parse(f.name)[1] for f in files})
    if len(versions) != 1:
        raise SystemExit(f"error: the artifacts hold several versions: {', '.join(versions)}")
    absent = [p for p in platforms if not any(f.name.endswith(f"-{p}.whl") for f in files)]
    if absent:
        raise SystemExit(f"error: no wheel for {', '.join(absent)}")
    tag_version = tag[1:] if tag.startswith("v") else tag
    return versions[0], versions[0] == tag_version


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("artifacts", type=Path)
    p.add_argument("upload", type=Path)
    p.add_argument("--dist", required=True)
    p.add_argument("--platform", action="append", default=[])
    p.add_argument("--tag", default=os.environ.get("GITHUB_REF_NAME", ""))
    a = p.parse_args()
    files = select(a.artifacts, a.dist)
    version, upload = decide(files, a.tag, a.platform)
    for f in files:
        print(f"{f.name}  {f.stat().st_size} bytes")
    if upload:
        a.upload.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, a.upload / f.name)
        print(f"{a.dist} {version}: released by the tag {a.tag}")
    else:
        print(f"::notice::{a.dist} is version {version}; the tag {a.tag} names another version, so it is not uploaded")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as out:
            out.write(f"upload={'true' if upload else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
