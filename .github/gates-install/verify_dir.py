#!/usr/bin/env python3
"""Keep only the release files (.whl, .tar.gz) in DIR and check each one's sha256 against SHA256SUMS.
Exit 1 on any missing, extra or different file.

    verify_dir.py DIR SHA256SUMS
"""
import hashlib
import sys
from pathlib import Path

d, sums = Path(sys.argv[1]), Path(sys.argv[2])
expected = {}
for line in sums.read_text().splitlines():
    if line.strip():
        digest, name = line.split(None, 1)
        expected[name.strip()] = digest
for p in list(d.rglob("*")):
    if p.is_file() and not p.name.endswith((".whl", ".tar.gz")):
        p.unlink()  # build logs that travel in the same artifacts
for p in list(d.rglob("*")):
    if p.is_file() and p.parent != d:
        raise SystemExit(f"nested file {p}")
have = {p.name: p for p in d.iterdir() if p.is_file()}
bad = 0
for name, digest in sorted(expected.items()):
    if name not in have:
        print("MISSING", name)
        bad += 1
    elif hashlib.sha256(have[name].read_bytes()).hexdigest() != digest:
        print("DIFFERENT", name)
        bad += 1
for name in sorted(set(have) - set(expected)):
    print("EXTRA", name)
    bad += 1
print(f"files {len(have)} expected {len(expected)} problems {bad}")
sys.exit(1 if bad else 0)
