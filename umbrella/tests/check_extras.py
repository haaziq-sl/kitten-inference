#!/usr/bin/env python3
"""Installs the umbrella with each extra into a fresh virtual environment, from local folders only, and checks
which kitten-inference distributions pip chose on this platform.

    python umbrella/tests/check_extras.py --find-links dist --find-links stubs --find-links deps [--old old]

dist: the built umbrella and kitten-inference-tts2 wheels; stubs: make_stub_wheels.py output (real platform tags);
deps: a wheelhouse of kitten-inference-tts2's own dependencies. Pinned (==0.2.0) installs must give the extra's
distributions where its engine has wheels, and fail elsewhere.

--old OLD: make_stub_wheels.py --old output. Then the same folders are also served as a local package index with the
earlier kitten-inference 0.1.0 / 0.1.1 in it, as on PyPI, and unpinned installs are checked: where an engine has no
wheel, pip falls back to kitten-inference 0.1.1 (only a warning, exit status 0) unless 0.1.x are yanked; with them
yanked it fails with an error instead. Exit status 1 when anything differs from what is expected here.
"""
import argparse
import hashlib
import html
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BASE = {"kitten-inference"}
TTS08 = {"kitten-inference-tts08"}
TTS2 = {"kitten-inference-tts2"}
CU12 = {"kitten-inference-tts2-cu12"}
ROCM = {"kitten-inference-tts2-rocm"}


def engines_here() -> dict:
    """Which engine wheels exist for this platform (the release wheel tags in make_stub_wheels.ENGINES)."""
    machine = platform.machine().lower()
    linux_x64 = sys.platform.startswith("linux") and machine in ("x86_64", "amd64")
    windows_x64 = sys.platform == "win32" and machine in ("amd64", "x86_64")
    return {"cu12": linux_x64 or windows_x64, "rocm": linux_x64}


def expected() -> dict:
    """extra -> the distributions a pinned install gives, or None when it must fail on this platform."""
    here = engines_here()
    return {
        "": BASE,
        "[tts-0.8]": BASE | TTS08,
        "[tts-0-8]": BASE | TTS08,
        "[tts-2]": BASE | TTS2 | CU12 if here["cu12"] else None,
        "[tts-2-rocm]": BASE | TTS2 | ROCM if here["rocm"] else None,
        "[tts-0.8,tts-2]": BASE | TTS08 | TTS2 | CU12 if here["cu12"] else None,
    }


def venv(tmp: str, links: list, upgrade_pip: bool) -> str:
    subprocess.run([sys.executable, "-m", "venv", tmp], check=True)
    py = str(Path(tmp) / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    if upgrade_pip:  # the pip in the wheelhouse (current), not the one the venv was created with
        subprocess.run([py, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-index", *links,
                        "-U", "pip"], check=True)
    return py


def install(py: str, source: list, spec: str):
    run = subprocess.run([py, "-m", "pip", "install", "--disable-pip-version-check", *source, spec],
                         capture_output=True, text=True)
    listed = subprocess.run([py, "-m", "pip", "list", "--format", "json"], capture_output=True, text=True)
    names = {d["name"].lower().replace("_", "-"): d["version"] for d in json.loads(listed.stdout or "[]")}
    ours = {n: v for n, v in names.items() if n.startswith("kitten-inference")}
    warnings = [l.strip() for l in run.stdout.splitlines() + run.stderr.splitlines() if "WARNING" in l
                or "ERROR" in l]
    return run, ours, warnings


def wheel_project(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.split("-")[0]).lower()


def make_index(folders: list, root: Path, yank: set) -> str:
    """A PEP 503 simple index on disk (file:// URL) of every wheel in folders; (project, version) in yank are marked
    yanked (PEP 592), as PyPI does."""
    projects = {}
    for folder in folders:
        for wheel in sorted(Path(folder).glob("*.whl")):
            projects.setdefault(wheel_project(wheel.name), []).append(wheel)
    for project, wheels in projects.items():
        d = root / project
        d.mkdir(parents=True, exist_ok=True)
        links = []
        for wheel in wheels:
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            version = wheel.name.split("-")[1]
            yanked = ' data-yanked="superseded by 0.2.0"' if (project, version) in yank else ""
            links.append(f'<a href="{wheel.resolve().as_uri()}#sha256={digest}"{yanked}>{html.escape(wheel.name)}</a>')
        (d / "index.html").write_text("<!DOCTYPE html><html><body>\n" + "\n".join(links) + "\n</body></html>\n")
    return root.resolve().as_uri()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--find-links", action="append", type=Path, required=True)
    p.add_argument("--old", type=Path)
    p.add_argument("--version", default="0.2.0")
    p.add_argument("--upgrade-pip", action="store_true", help="first install the pip found in --find-links")
    a = p.parse_args()
    links = [arg for d in a.find_links for arg in ("--find-links", str(d.resolve()))]
    print(f"platform {sys.platform} {platform.machine()}: engine wheels here: {engines_here()}")
    failed = 0
    for extra, want in expected().items():
        with tempfile.TemporaryDirectory() as tmp:
            py = venv(tmp, links, a.upgrade_pip)
            pip_version = subprocess.run([py, "-m", "pip", "--version"], capture_output=True,
                                         text=True).stdout.split()[1]
            spec = f"kitten-inference{extra}=={a.version}"
            run, ours, warnings = install(py, ["--no-index", *links], spec)
            ok = (run.returncode == 0 and set(ours) == want) if want is not None else run.returncode != 0
            outcome = sorted(ours) if run.returncode == 0 else f"error (exit {run.returncode})"
            print(f"{'ok  ' if ok else 'FAIL'} pip {pip_version}: pip install \"{spec}\" -> {outcome}"
                  + ("" if want is not None else "  [expected: no engine wheel for this platform]")
                  + (f"  [{'; '.join(warnings[-2:])}]" if warnings else ""))
            if not ok:
                failed += 1
                print(run.stdout[-3000:], run.stderr[-3000:], sep="\n")
    if a.old:
        failed += check_fallback(a, links)
    return 1 if failed else 0


def check_fallback(a, links) -> int:
    """Unpinned installs from a local index that also has the earlier kitten-inference 0.1.x, as PyPI has."""
    folders = [Path(links[i + 1]) for i in range(0, len(links), 2)] + [a.old]
    here = engines_here()
    failed = 0
    for yanked in (False, True):
        with tempfile.TemporaryDirectory() as idx, tempfile.TemporaryDirectory() as tmp:
            yank = {("kitten-inference", "0.1.0"), ("kitten-inference", "0.1.1")} if yanked else set()
            url = make_index(folders, Path(idx) / "simple", yank)
            py = venv(tmp, ["--index-url", url], upgrade_pip=False)
            subprocess.run([py, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--index-url", url,
                            "-U", "pip"], check=True)
            for extra, engine in (("[tts-2]", "cu12"), ("[tts-2-rocm]", "rocm")):
                subprocess.run([py, "-m", "pip", "uninstall", "-y", "-q", "kitten-inference", "kitten-inference-tts2",
                                "kitten-inference-tts2-cu12", "kitten-inference-tts2-rocm"], capture_output=True)
                spec = f"kitten-inference{extra}"
                run, ours, warnings = install(py, ["--index-url", url], spec)
                if here[engine]:
                    want = f"{a.version} with the engine"
                    ok = run.returncode == 0 and ours.get("kitten-inference") == a.version and len(ours) == 3
                elif not yanked:
                    want = "falls back to kitten-inference 0.1.1 (warning only, exit 0)"
                    ok = run.returncode == 0 and ours == {"kitten-inference": "0.1.1"}
                else:
                    want = "an error (0.1.x yanked: no fallback)"
                    ok = run.returncode != 0 and not ours
                outcome = ours if run.returncode == 0 else f"error (exit {run.returncode})"
                print(f"{'ok  ' if ok else 'FAIL'} index with 0.1.x{' yanked' if yanked else ''}: pip install "
                      f"\"{spec}\" -> {outcome}  [expected: {want}]" + (f"  [{'; '.join(warnings[-2:])}]"
                                                                        if warnings else ""))
                if not ok:
                    failed += 1
                    print(run.stdout[-3000:], run.stderr[-3000:], sep="\n")
    return failed


if __name__ == "__main__":
    sys.exit(main())
