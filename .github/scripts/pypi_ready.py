#!/usr/bin/env python3
"""Before the umbrella goes to PyPI: is every distribution its extras name already there, with a release in the
required range and wheels for the platforms it ships? An umbrella published first would point its extras at
nothing (pip then fails, or falls back to the earlier kitten-inference 0.1.1).

    python .github/scripts/pypi_ready.py check  [--pyproject umbrella/pyproject.toml] [--index https://pypi.org]
    python .github/scripts/pypi_ready.py wait   [--timeout-minutes 120] [--every-seconds 60]
    python .github/scripts/pypi_ready.py wait   --only kitten-inference-tts2-cu12 --only kitten-inference-tts2-rocm
    python .github/scripts/pypi_ready.py complete --dist kitten-inference-tts08 dist/*.whl

complete: whether built wheels are exactly the set the check waits for (COMPLETE), so the two stay in step.
check: exit status 0 when ready, 1 when not (the missing items are printed). wait: check until ready or the timeout;
a failed request (an error answer from PyPI, a timeout, a dropped connection) counts as "not ready yet" and is tried
again at the next check.
A file counts only when PyPI's provenance for it (PEP 740, the attestation trusted publishing uploads with it) names
this repository and the distribution's own workflow as its publisher. So a release someone else uploaded under one of
the names (a squatted project name) never makes the umbrella publish; --publisher sets the repository.
--only: just these distributions of the extras (the release order: the engine wheels before kitten-inference-tts2
and kitten-inference-tts08, those before the umbrella). Without --only it also warns (GitHub annotation) while the
earlier kitten-inference 0.1.0 / 0.1.1 are not yanked.
Read-only: it only reads PyPI's JSON and integrity APIs.
"""
import argparse
import http.client
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python < 3.11
    tomllib = None

# The platforms each distribution must have a wheel for before it counts as ready, and whose provenance is checked
# (CONTRACT.md); any wheel will do for distributions not listed.
PLATFORMS = {
    "kitten-inference-tts2-cu12": ["manylinux_2_28_x86_64", "win_amd64"],
    "kitten-inference-tts2-rocm": ["manylinux_2_28_x86_64"],
}
# Distributions with one wheel per CPython and platform: every one of them must be there. tts-0.8 uploads its 62
# wheels in one twine run sorted by file name, so grouped by Python version (cp310 ... cp315, cp38, cp39), and every
# platform has a wheel within the first few files: only the whole set shows that an upload which stopped partway
# (a PyPI error or a dropped connection after file 20) is incomplete. Without it the umbrella would publish while
# `pip install "kitten-inference[tts-0.8]"` finds no wheel for most Python versions. This is tts-0.8.yml's build
# matrix: keep the two in step. A wheel listed here that the workflow does not build keeps the umbrella waiting (and
# failing at the timeout), never publishing early.
_TTS08_PYTHONS = ["cp38", "cp39", "cp310", "cp311", "cp312", "cp313", "cp314", "cp315"]
_TTS08_PLATFORMS = ["manylinux_2_17_x86_64", "manylinux_2_17_aarch64", "manylinux_2_28_x86_64",
                    "manylinux_2_28_aarch64", "macosx_11_0_x86_64", "macosx_11_0_arm64", "win_amd64"]
COMPLETE = {
    "kitten-inference-tts08": ([(py, plat) for py in _TTS08_PYTHONS for plat in _TTS08_PLATFORMS]
                               + [(py, "win_arm64") for py in _TTS08_PYTHONS if py not in ("cp38", "cp39", "cp310")]
                               + [("cp313", "android_24_arm64_v8a")]),
}
EARLIER = ("kitten-inference", ["0.1.0", "0.1.1"])
# The workflow that publishes each distribution (RELEASING.md), as PyPI's provenance names it.
WORKFLOWS = {
    "kitten-inference": "packages.yml",
    "kitten-inference-tts2": "packages.yml",
    "kitten-inference-tts08": "tts-0.8.yml",
    "kitten-inference-tts2-cu12": "tts-2-engine.yml",
    "kitten-inference-tts2-rocm": "tts-2-rocm.yml",
}
PUBLISHER = "KittenML/kitten-inference"


def requirements(pyproject: Path) -> dict:
    """distribution -> specifier, from the umbrella's optional-dependencies."""
    text = pyproject.read_text(encoding="utf-8")
    if tomllib is not None:
        extras = tomllib.loads(text)["project"]["optional-dependencies"]
        specs = [s for reqs in extras.values() for s in reqs]
    else:
        block = text.split("[project.optional-dependencies]", 1)[1].split("\n[", 1)[0]
        specs = re.findall(r'"([A-Za-z0-9_.\-]+[<>=!~][^"]*)"', block)
    out = {}
    for spec in specs:
        name = re.match(r"[A-Za-z0-9_.\-]+", spec).group(0)
        out[re.sub(r"[-_.]+", "-", name).lower()] = spec[len(name):].strip()
    return out


def _version(v: str) -> tuple:
    """A sortable key for the plain versions used here (0.2.0, 0.2.0.post1); pre-releases sort before."""
    m = re.match(r"(\d+(?:\.\d+)*)(?:\.post(\d+))?(.*)$", v)
    if not m:
        return (-1,)
    nums = tuple(int(x) for x in m.group(1).split("."))
    nums += (0,) * (4 - len(nums))
    return nums + (int(m.group(2) or 0), 0 if m.group(3) else 1)


def satisfies(version: str, spec: str) -> bool:
    """version against a specifier such as ">=0.2.0,<0.3" (only the operators the extras use)."""
    v = _version(version)
    if v == (-1,) or v[-1] == 0:  # unknown form or a pre-release: never
        return False
    for clause in filter(None, (c.strip() for c in spec.split(","))):
        op, ref = re.match(r"(>=|<=|==|<|>|!=)\s*(.+)", clause).groups()
        r = _version(ref)[:-1] + (1,)
        if not {">=": v >= r, "<=": v <= r, "==": v == r, "<": v < r, ">": v > r, "!=": v != r}[op]:
            return False
    return True


def fetch(index: str, name: str):
    try:
        with urllib.request.urlopen(f"{index}/pypi/{name}/json", timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def fetch_provenance(index: str, name: str, version: str, filename: str):
    """PyPI's provenance of one uploaded file (None when the file has none)."""
    try:
        with urllib.request.urlopen(f"{index}/integrity/{name}/{version}/{filename}/provenance", timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def published_by(provenance, repository: str, workflow: str) -> bool:
    """Whether a provenance names GitHub Actions in ``repository`` with ``workflow`` as the file's publisher."""
    for bundle in (provenance or {}).get("attestation_bundles") or []:
        pub = bundle.get("publisher") or {}
        if (pub.get("kind") == "GitHub" and str(pub.get("repository", "")).lower() == repository.lower()
                and str(pub.get("workflow", "")).rsplit("/", 1)[-1] == workflow):
            return True
    return False


def only(reqs: dict, names: list) -> dict:
    """The requirements of just `names` (normalised); a name the extras do not list is an error."""
    if not names:
        return reqs
    wanted = [re.sub(r"[-_.]+", "-", n).lower() for n in names]
    unknown = [n for n in wanted if n not in reqs]
    if unknown:
        raise SystemExit(f"error: the extras do not name {', '.join(unknown)}")
    return {n: reqs[n] for n in wanted}


def wheel_tags(filename: str) -> set:
    """The (python tag, platform tag) pairs a wheel file name covers (compressed tag sets such as
    manylinux2014_x86_64.manylinux_2_17_x86_64 give one pair per tag)."""
    parts = filename[:-len(".whl")].split("-") if filename.endswith(".whl") else []
    if len(parts) < 5:
        return set()
    return {(py, plat) for py in parts[-3].split(".") for plat in parts[-1].split(".")}


def incomplete(name: str, wheels: list) -> list:
    """The expected (python, platform) wheels of ``name`` (COMPLETE) that ``wheels`` lacks, as "cp39-win_amd64"."""
    have = set().union(*(wheel_tags(w) for w in wheels)) if wheels else set()
    return [f"{py}-{plat}" for py, plat in COMPLETE.get(name, []) if (py, plat) not in have]


def missing(index: str, reqs: dict, get=fetch, provenance=None, publisher: str = PUBLISHER) -> list:
    """What is not on the index yet: one line per distribution. With ``provenance`` (a fetch_provenance-like
    function), only files that this repository's own workflow published count."""
    problems = []
    for name, spec in sorted(reqs.items()):
        info = get(index, name)
        if info is None:
            problems.append(f"{name}: not on {index}")
            continue
        good = {}
        for version, files in (info.get("releases") or {}).items():
            wheels = [f["filename"] for f in files if f.get("packagetype") == "bdist_wheel" and not f.get("yanked")]
            if wheels and satisfies(version, spec):
                good[version] = wheels
        if not good:
            problems.append(f"{name}: no release {spec} with wheels")
            continue
        best = max(good, key=_version)
        absent = [p for p in PLATFORMS.get(name, []) if not any(w.endswith(f"-{p}.whl") for w in good[best])]
        if absent:
            problems.append(f"{name} {best}: no wheel for {', '.join(absent)}")
            continue
        lacking = incomplete(name, good[best])
        if lacking:
            shown = ", ".join(lacking[:6]) + (f" and {len(lacking) - 6} more" if len(lacking) > 6 else "")
            problems.append(f"{name} {best}: {len(lacking)} of {len(COMPLETE[name])} wheels not there yet ({shown})")
            continue
        if provenance is not None:
            # One wheel per required platform (or the first one): a squatted project has none from this repository.
            checks = [next(w for w in good[best] if w.endswith(f"-{p}.whl")) for p in PLATFORMS.get(name, [])]
            for wheel in checks or sorted(good[best])[:1]:
                if not published_by(provenance(index, name, best, wheel), publisher, WORKFLOWS.get(name, "")):
                    problems.append(f"{name} {best}: {wheel} was not published by {publisher} "
                                    f"({WORKFLOWS.get(name, '?')}); is the project name someone else's?")
                    break
    return problems


def earlier_not_yanked(index: str, get=fetch) -> list:
    name, versions = EARLIER
    info = get(index, name) or {}
    releases = info.get("releases") or {}
    return [v for v in versions if releases.get(v) and not all(f.get("yanked") for f in releases[v])]


# What a request to PyPI can fail with that is worth trying again (HTTPError is a URLError; socket timeouts and reset
# connections are OSErrors; a cut-off answer is not JSON).
TRANSIENT = (urllib.error.URLError, OSError, http.client.HTTPException, ValueError)


def wait(index: str, reqs: dict, mode: str, timeout_s: float, every_s: float, get=fetch, sleep=time.sleep,
         clock=time.monotonic, provenance=None, publisher: str = PUBLISHER) -> list:
    """[] once every distribution is there; else the problems of the last check (mode check: after one check;
    mode wait: when the timeout has passed)."""
    deadline = clock() + timeout_s
    while True:
        try:
            problems = missing(index, reqs, get=get, provenance=provenance, publisher=publisher)
        except TRANSIENT as exc:
            problems = [f"PyPI request failed ({type(exc).__name__}: {exc}); trying again"]
        if not problems:
            return []
        print("not ready:", "; ".join(problems), flush=True)
        if mode == "check" or clock() > deadline:
            return problems
        sleep(every_s)


def check_built(name: str, files: list) -> list:
    """Problems with the built wheels ``files`` of ``name`` against COMPLETE: expected wheels that were not built, and
    built ones the check would not wait for."""
    wheels = [Path(f).name for f in files]
    expected = set(COMPLETE.get(name, []))
    problems = [f"not built: {w}" for w in incomplete(name, wheels)]
    problems += [f"built but not in pypi_ready.COMPLETE: {w}" for w in sorted(wheels)
                 if not wheel_tags(w) & expected]
    return problems


def main() -> int:
    if sys.argv[1:2] == ["complete"]:
        c = argparse.ArgumentParser(prog="pypi_ready.py complete")
        c.add_argument("--dist", required=True)
        c.add_argument("wheels", nargs="*")
        a = c.parse_args(sys.argv[2:])
        name = re.sub(r"[-_.]+", "-", a.dist).lower()
        problems = check_built(name, a.wheels) if name in COMPLETE else [f"{name} has no COMPLETE entry"]
        for problem in problems:
            print(f"::error::{name}: {problem}")
        if not problems:
            print(f"{name}: the {len(a.wheels)} built wheels are the set the umbrella's release check waits for")
        return 1 if problems else 0
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["check", "wait"])
    p.add_argument("--pyproject", type=Path, default=Path("umbrella/pyproject.toml"))
    p.add_argument("--index", default="https://pypi.org")
    p.add_argument("--timeout-minutes", type=float, default=120)
    p.add_argument("--every-seconds", type=float, default=60)
    p.add_argument("--only", action="append", default=[], help="just this distribution (repeatable)")
    p.add_argument("--publisher", default=PUBLISHER, help="the repository whose workflows must have published the files")
    a = p.parse_args()
    reqs = only(requirements(a.pyproject), a.only)
    print("waiting for" if a.only else "the umbrella's extras need:",
          ", ".join(f"{n}{s}" for n, s in sorted(reqs.items())))
    problems = wait(a.index, reqs, a.mode, a.timeout_minutes * 60, a.every_seconds, provenance=fetch_provenance,
                    publisher=a.publisher)
    if problems:
        what = "this distribution" if a.only else "the umbrella"
        print(f"::error::{what} must not be published before these are on PyPI: " + "; ".join(problems))
        return 1
    try:
        old = [] if a.only else earlier_not_yanked(a.index)
    except TRANSIENT as exc:  # only a reminder: never a reason to fail the release
        print(f"::notice::could not check whether kitten-inference 0.1.x are yanked ({exc})")
        old = []
    if old:
        print(f"::warning::kitten-inference {', '.join(old)} not yanked: where an extra's engine has no wheel, pip "
              "installs that version instead (with only a warning). Yank them on PyPI (pinned installs still work).")
    print("ready:", "these distributions are" if a.only else "every distribution the extras name is", "on", a.index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
