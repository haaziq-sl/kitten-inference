#!/usr/bin/env python3
"""Before the umbrella goes to PyPI: is every distribution its extras name already there, with a release in the
required range and wheels for the platforms it ships? An umbrella published first would point its extras at
nothing (pip then fails, or falls back to the earlier kitten-inference 0.1.1).

    python .github/scripts/pypi_ready.py check  [--pyproject umbrella/pyproject.toml] [--index https://pypi.org]
    python .github/scripts/pypi_ready.py wait   [--timeout-minutes 120] [--every-seconds 60]

check: exit status 0 when ready, 1 when not (the missing items are printed). wait: check until ready or the timeout.
It also warns (GitHub annotation) while the earlier kitten-inference 0.1.0 / 0.1.1 are not yanked.
Read-only: it only reads PyPI's JSON API.
"""
import argparse
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

# The platforms each engine ships wheels for (CONTRACT.md); any wheel will do for the others.
PLATFORMS = {
    "kitten-inference-tts2-cu12": ["manylinux_2_28_x86_64", "win_amd64"],
    "kitten-inference-tts2-rocm": ["manylinux_2_28_x86_64"],
}
EARLIER = ("kitten-inference", ["0.1.0", "0.1.1"])


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


def missing(index: str, reqs: dict, get=fetch) -> list:
    """What is not on the index yet: one line per distribution."""
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
    return problems


def earlier_not_yanked(index: str, get=fetch) -> list:
    name, versions = EARLIER
    info = get(index, name) or {}
    releases = info.get("releases") or {}
    return [v for v in versions if releases.get(v) and not all(f.get("yanked") for f in releases[v])]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["check", "wait"])
    p.add_argument("--pyproject", type=Path, default=Path("umbrella/pyproject.toml"))
    p.add_argument("--index", default="https://pypi.org")
    p.add_argument("--timeout-minutes", type=float, default=120)
    p.add_argument("--every-seconds", type=float, default=60)
    a = p.parse_args()
    reqs = requirements(a.pyproject)
    print("the umbrella's extras need:", ", ".join(f"{n}{s}" for n, s in sorted(reqs.items())))
    deadline = time.monotonic() + a.timeout_minutes * 60
    while True:
        problems = missing(a.index, reqs)
        if not problems:
            break
        print("not ready:", "; ".join(problems), flush=True)
        if a.mode == "check" or time.monotonic() > deadline:
            print("::error::the umbrella must not be published before its engines: " + "; ".join(problems))
            return 1
        time.sleep(a.every_seconds)
    old = earlier_not_yanked(a.index)
    if old:
        print(f"::warning::kitten-inference {', '.join(old)} not yanked: where an extra's engine has no wheel, pip "
              "installs that version instead (with only a warning). Yank them on PyPI (pinned installs still work).")
    print("ready: every distribution the extras name is on", a.index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
