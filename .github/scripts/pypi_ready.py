#!/usr/bin/env python3
"""The wheel sets of a kitten-inference release, and whether PyPI has all of them after the upload.

    python .github/scripts/pypi_ready.py complete --dist kitten-inference assembled/*.whl
    python .github/scripts/pypi_ready.py complete --dist kitten-inference-tts08 dist/*.whl
    python .github/scripts/pypi_ready.py verify --version 0.2.0 [--timeout-minutes 30] upload/*.whl

complete: whether the wheels are exactly the set COMPLETE lists for that name, one wheel per (Python, platform): the
54 kitten-inference wheels a release uploads (packages.yml), or the 62 TTS 0.8 wheels tts-0.8.yml builds, which the
kitten-inference wheels are assembled from (all but the 8 manylinux2014 x86-64 ones). It keeps the build matrices,
the assembler and this list in step.

verify (the publish job, after the upload): every file is on PyPI under that version, not yanked, with PyPI's sha256
equal to the local file's, and PyPI's provenance for it (PEP 740, the attestation trusted publishing uploads with it)
names this repository and packages.yml as its publisher; PyPI has no other file for that version. PyPI has no upload
of several files at once, so a stop partway leaves part of the set there: verify reports how many are missing and
which, and tries again until --timeout-minutes have passed. A failed request (an error answer from PyPI, a timeout, a
dropped connection) counts as "not there yet". It also warns (GitHub annotation) while the earlier kitten-inference
0.1.0 / 0.1.1 are not yanked. Read-only: it only reads PyPI's JSON and integrity APIs.
"""
import argparse
import hashlib
import http.client
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

NAME = "kitten-inference"
# The workflow that publishes kitten-inference, as PyPI's provenance names it.
WORKFLOW = "packages.yml"
PUBLISHER = "KittenML/kitten-inference"

# One wheel per CPython and platform. tts-0.8.yml builds 62 TTS 0.8 wheels; each kitten-inference wheel is one of them
# plus the TTS 2 Python API (and on Linux x86-64 and Windows x64 the TTS 2 engines), with the same Python and platform
# tags, except the manylinux2014 x86-64 ones (manylinux_2_17_x86_64), which a release leaves out: every Linux x86-64
# wheel is the manylinux_2_28 build. A wheel listed here that is not built fails `complete`, and a release whose files
# are not all on PyPI fails `verify`.
_PYTHONS = ["cp38", "cp39", "cp310", "cp311", "cp312", "cp313", "cp314", "cp315"]
_TTS08_PLATFORMS = ["manylinux_2_17_x86_64", "manylinux_2_17_aarch64", "manylinux_2_28_x86_64",
                    "manylinux_2_28_aarch64", "macosx_11_0_x86_64", "macosx_11_0_arm64", "win_amd64"]
_OTHERS = ([(py, "win_arm64") for py in _PYTHONS if py not in ("cp38", "cp39", "cp310")]
           + [("cp313", "android_24_arm64_v8a")])
COMPLETE = {
    "kitten-inference-tts08": [(py, plat) for py in _PYTHONS for plat in _TTS08_PLATFORMS] + _OTHERS,
    "kitten-inference": [(py, plat) for py in _PYTHONS for plat in _TTS08_PLATFORMS
                         if plat != "manylinux_2_17_x86_64"] + _OTHERS,
}
EARLIER = (NAME, ["0.1.0", "0.1.1"])


def wheel_tags(filename: str) -> set:
    """The (python tag, platform tag) pairs a wheel file name covers (compressed tag sets such as
    manylinux2014_x86_64.manylinux_2_17_x86_64 give one pair per tag)."""
    parts = filename[:-len(".whl")].split("-") if filename.endswith(".whl") else []
    if len(parts) < 5:
        return set()
    return {(py, plat) for py in parts[-3].split(".") for plat in parts[-1].split(".")}


def file_version(filename: str) -> str:
    """The version in a wheel file name."""
    parts = filename.split("-")
    return parts[1] if len(parts) >= 5 else ""


def short(filename: str) -> str:
    """cp312-manylinux_2_28_x86_64 for kitten_inference-0.2.0-cp312-cp312-manylinux_2_28_x86_64.whl."""
    parts = filename[:-len(".whl")].split("-")
    return f"{parts[-3]}-{parts[-1]}" if len(parts) >= 5 else filename


def incomplete(name: str, wheels: list) -> list:
    """The expected (python, platform) wheels of ``name`` (COMPLETE) that ``wheels`` lacks, as "cp39-win_amd64"."""
    have = set().union(*(wheel_tags(w) for w in wheels)) if wheels else set()
    return [f"{py}-{plat}" for py, plat in COMPLETE.get(name, []) if (py, plat) not in have]


def check_built(name: str, files: list) -> list:
    """Problems with the wheels ``files`` of ``name`` against COMPLETE: expected wheels that are not there, wheels
    COMPLETE does not list, and two wheels for one (python, platform)."""
    wheels = [Path(f).name for f in files]
    expected = set(COMPLETE.get(name, []))
    problems = [f"not built: {w}" for w in incomplete(name, wheels)]
    problems += [f"built but not in pypi_ready.COMPLETE: {w}" for w in sorted(wheels)
                 if not wheel_tags(w) & expected]
    seen = {}
    for w in sorted(wheels):
        for pair in sorted(wheel_tags(w) & expected):
            if pair in seen:
                problems.append(f"two wheels for {pair[0]}-{pair[1]}: {seen[pair]} and {w}")
            seen[pair] = w
    return problems


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json(url: str):
    """The JSON at ``url``; None for a 404."""
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def fetch(index: str, name: str):
    """PyPI's JSON of a project (every release); None when the project does not exist."""
    return _json(f"{index}/pypi/{name}/json")


def fetch_release(index: str, name: str, version: str):
    """PyPI's JSON of one release (its files under "urls"); None when that version is not there."""
    return _json(f"{index}/pypi/{name}/{version}/json")


def fetch_provenance(index: str, name: str, version: str, filename: str):
    """PyPI's provenance of one uploaded file (None when the file has none)."""
    return _json(f"{index}/integrity/{name}/{version}/{filename}/provenance")


def published_by(provenance, repository: str, workflow: str) -> bool:
    """Whether a provenance names GitHub Actions in ``repository`` with ``workflow`` as the file's publisher."""
    for bundle in (provenance or {}).get("attestation_bundles") or []:
        pub = bundle.get("publisher") or {}
        if (pub.get("kind") == "GitHub" and str(pub.get("repository", "")).lower() == repository.lower()
                and str(pub.get("workflow", "")).rsplit("/", 1)[-1] == workflow):
            return True
    return False


def on_pypi(index: str, version: str, local: dict, get=fetch_release, provenance=fetch_provenance,
            publisher: str = PUBLISHER, checked=None) -> list:
    """What PyPI does not have yet, or has wrong, of the release ``local`` ({file name: sha256}): [] when every file is
    there, unyanked, byte for byte, published by this repository's packages.yml, and nothing else is.
    ``checked`` (a set) keeps the files already found good, so a retry does not ask for their provenance again."""
    checked = set() if checked is None else checked
    info = get(index, NAME, version)
    if info is None:
        return [f"{NAME} {version}: not on {index} yet (0 of {len(local)} wheels)"]
    files = {f.get("filename"): f for f in info.get("urls") or []}
    problems = []
    absent = [n for n in sorted(local) if n not in files]
    if absent:
        shown = ", ".join(short(n) for n in absent[:6]) + (f" and {len(absent) - 6} more" if len(absent) > 6 else "")
        problems.append(f"{NAME} {version}: {len(absent)} of {len(local)} wheels not on {index} yet ({shown})")
    for n in sorted(local):
        f = files.get(n)
        if f is None or n in checked:
            continue
        theirs = (f.get("digests") or {}).get("sha256")
        if f.get("yanked"):
            problems.append(f"{n}: yanked on {index}")
        elif theirs != local[n]:
            problems.append(f"{n}: sha256 on {index} is {theirs}, this file's is {local[n]}")
        elif provenance is not None and not published_by(provenance(index, NAME, version, n), publisher, WORKFLOW):
            problems.append(f"{n}: not published by {publisher} ({WORKFLOW}); is the upload someone else's?")
        else:
            checked.add(n)
    other = sorted(n for n in files if n not in local)
    if other:
        problems.append(f"{NAME} {version} on {index} has files this release does not: {', '.join(other)}")
    return problems


def earlier_not_yanked(index: str, get=fetch) -> list:
    name, versions = EARLIER
    info = get(index, name) or {}
    releases = info.get("releases") or {}
    return [v for v in versions if releases.get(v) and not all(f.get("yanked") for f in releases[v])]


# What a request to PyPI can fail with that is worth trying again (HTTPError is a URLError; socket timeouts and reset
# connections are OSErrors; a cut-off answer is not JSON).
TRANSIENT = (urllib.error.URLError, OSError, http.client.HTTPException, ValueError)


def verify(index: str, version: str, local: dict, timeout_s: float, every_s: float, get=fetch_release,
           provenance=fetch_provenance, sleep=time.sleep, clock=time.monotonic, publisher: str = PUBLISHER) -> list:
    """[] once PyPI has the whole release (on_pypi); else the problems of the last check, when the timeout has
    passed."""
    deadline = clock() + timeout_s
    checked = set()
    while True:
        try:
            problems = on_pypi(index, version, local, get=get, provenance=provenance, publisher=publisher,
                               checked=checked)
        except TRANSIENT as exc:
            problems = [f"PyPI request failed ({type(exc).__name__}: {exc}); trying again"]
        if not problems:
            return []
        print("not there yet:", "; ".join(problems), flush=True)
        if clock() > deadline:
            return problems
        sleep(every_s)


def main() -> int:
    mode = sys.argv[1:2]
    if mode == ["complete"]:
        c = argparse.ArgumentParser(prog="pypi_ready.py complete")
        c.add_argument("--dist", required=True)
        c.add_argument("wheels", nargs="*")
        a = c.parse_args(sys.argv[2:])
        name = re.sub(r"[-_.]+", "-", a.dist).lower()
        problems = check_built(name, a.wheels) if name in COMPLETE else [f"{name} has no COMPLETE entry"]
        for problem in problems:
            print(f"::error::{name}: {problem}")
        if not problems:
            print(f"{name}: the {len(a.wheels)} built wheels are the complete set (pypi_ready.COMPLETE)")
        return 1 if problems else 0
    if mode != ["verify"]:
        sys.exit("usage: pypi_ready.py complete --dist NAME WHEEL... | verify --version V WHEEL...")
    v = argparse.ArgumentParser(prog="pypi_ready.py verify")
    v.add_argument("--version", required=True)
    v.add_argument("wheels", nargs="+")
    v.add_argument("--index", default="https://pypi.org")
    v.add_argument("--timeout-minutes", type=float, default=30)
    v.add_argument("--every-seconds", type=float, default=60)
    v.add_argument("--publisher", default=PUBLISHER, help="the repository whose packages.yml must have published them")
    a = v.parse_args(sys.argv[2:])
    names = [Path(w).name for w in a.wheels]
    other = sorted(n for n in names if file_version(n) != a.version)
    problems = [f"{n} is not version {a.version}" for n in other] + check_built(NAME, a.wheels)
    if problems:
        for problem in problems:
            print(f"::error::{problem}")
        return 1
    local = {Path(w).name: sha256(w) for w in a.wheels}
    print(f"checking {NAME} {a.version} on {a.index}: {len(local)} wheels")
    problems = verify(a.index, a.version, local, a.timeout_minutes * 60, a.every_seconds, publisher=a.publisher)
    if problems:
        print(f"::error::{NAME} {a.version} is not complete on {a.index} (re-run the publish job: it skips the "
              "files already there): " + "; ".join(problems))
        return 1
    try:
        old = earlier_not_yanked(a.index)
    except TRANSIENT as exc:  # only a reminder: never a reason to fail the release
        print(f"::notice::could not check whether kitten-inference 0.1.x are yanked ({exc})")
        old = []
    if old:
        print(f"::warning::kitten-inference {', '.join(old)} not yanked: where no 0.2 wheel matches, pip installs that "
              "version instead (with only a warning). Yank them on PyPI (pinned installs still work).")
    print(f"{NAME} {a.version}: all {len(local)} wheels are on {a.index}, byte for byte, published by {a.publisher} "
          f"({WORKFLOW})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
