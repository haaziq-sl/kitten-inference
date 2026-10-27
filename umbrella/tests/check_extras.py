#!/usr/bin/env python3
"""Installs kitten-inference with each extra into a fresh virtual environment, from local folders only, and checks
what pip chose on this platform.

    python umbrella/tests/check_extras.py --find-links assembled --find-links deps [--old old] [--version V]
                                          [--upgrade-pip]

The folders hold the kitten-inference wheels and a wheelhouse of the extras' libraries (and pip, for --upgrade-pip).
They are served as a local package index, as PyPI serves them: build inputs found in them (the py3-none-any
kitten-inference wheel and every kitten_inference_* engine build) are left out, since they are never uploaded. Pinned
installs (==V) must give:
- one kitten distribution, kitten-inference V, whatever the extra;
- that extra's libraries: [tts-0.8] five; [tts-2] four where Kitten TTS 2 runs (Linux x86-64 and Windows x64, Python
  3.9 or newer) and none elsewhere;
- the subpackages of this platform's wheel: tts08 and tts2 everywhere, _tts2_cuda on Linux x86-64 and Windows x64,
  _tts2_rocm on Linux x86-64 (both from Python 3.9);
- for [tts-2-rocm] (the AMD spelling of earlier drafts of the docs; no release has it) the package alone, with pip's
  "does not provide the extra" warning; the same for [tts-0.8] with pip older than 23.3, which does not match the
  dotted name to the extra tts-0-8.

--old OLD: make_stub_wheels.py --old output. The index then also has the earlier kitten-inference 0.1.0 and 0.1.1, as
PyPI has, and unpinned installs are checked: this platform gets V (also with [tts-2-rocm]); a system with no V wheel
(a musllinux CPython, through pip download --platform) falls back to 0.1.1 unless 0.1.x are yanked, and gets an error
when they are. Exit status 1 when anything differs from what is expected here.
"""
import argparse
import functools
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

TTS08 = {"espeakng-loader", "phonemizer", "soundfile", "numpy", "huggingface-hub"}
TTS2 = {"numpy", "huggingface-hub", "tokenizers", "filelock"}
# Libraries that only one extra brings in (the others also come as dependencies of huggingface-hub).
EXCLUSIVE = {"espeakng-loader", "phonemizer", "soundfile", "tokenizers"}
SPECS = ["", "[tts-0.8]", "[tts-0-8]", "[tts-2]", "[tts-0.8,tts-2]", "[tts-2-rocm]"]
# No kitten-inference wheel has this tag: pip download --platform sees what such a system would get.
FOREIGN = ["--platform", "musllinux_1_2_x86_64", "--implementation", "cp", "--python-version", "3.12", "--abi", "cp312"]

# Run in the virtual environment: what is installed, and which kitten_inference subpackages are on disk.
PROBE = """
import json, os
from importlib import metadata, util
spec = util.find_spec("kitten_inference")
roots = list(spec.submodule_search_locations or []) if spec else []
subs = {n for r in roots for n in os.listdir(r) if n != "__pycache__" and os.path.isdir(os.path.join(r, n))}
dists = {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata["Name"]}
try:
    wheel = metadata.distribution("kitten-inference").read_text("WHEEL") or ""
except metadata.PackageNotFoundError:
    wheel = ""
print(json.dumps({"dists": dists, "subpackages": sorted(subs),
                  "tags": [l.split(":", 1)[1].strip() for l in wheel.splitlines() if l.startswith("Tag:")]}))
"""


def engines_here() -> dict:
    """The GPU engines a kitten-inference wheel for this interpreter holds. [tts-2] adds its libraries under the same
    condition (its markers): the CUDA engine's."""
    machine = platform.machine().lower()
    linux_x64 = sys.platform.startswith("linux") and machine in ("x86_64", "amd64")
    windows_x64 = sys.platform == "win32" and machine in ("amd64", "x86_64")
    py39 = sys.version_info >= (3, 9)
    return {"_tts2_cuda": py39 and (linux_x64 or windows_x64), "_tts2_rocm": py39 and linux_x64}


def subpackages() -> set:
    return {"tts08", "tts2"} | {name for name, here in engines_here().items() if here}


def expected(pip_version: tuple) -> dict:
    """extra -> (the libraries it adds, the extra pip must warn about or None)."""
    tts2 = TTS2 if engines_here()["_tts2_cuda"] else set()
    dotted = pip_version >= (23, 3)
    tts08, warn08 = (TTS08, None) if dotted else (set(), "tts-0.8")
    return {
        "": (set(), None),
        "[tts-0.8]": (tts08, warn08),
        "[tts-0-8]": (TTS08, None),
        "[tts-2]": (tts2, None),
        "[tts-0.8,tts-2]": (tts08 | tts2, warn08),
        "[tts-2-rocm]": (set(), "tts-2-rocm"),
    }


def project_of(filename: str) -> str:
    return re.sub(r"[-_.]+", "-", filename.split("-")[0]).lower()


def served(folders: list, version: str) -> tuple:
    """(the wheels in folders that PyPI would have, the names of the build inputs left out)."""
    wheels, left_out, seen = [], [], set()
    for folder in folders:
        for wheel in sorted(Path(folder).glob("*.whl")):
            project = project_of(wheel.name)
            if project.startswith("kitten-inference-") or wheel.name == f"kitten_inference-{version}-py3-none-any.whl":
                left_out.append(wheel.name)
            elif wheel.name not in seen:
                seen.add(wheel.name)
                wheels.append(wheel)
    return wheels, left_out


@functools.lru_cache(maxsize=None)
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_index(wheels: list, root: Path, yank: set) -> str:
    """A PEP 503 simple index on disk (file:// URL) of wheels; (project, version) in yank are marked yanked (PEP 592),
    as PyPI does."""
    projects = {}
    for wheel in wheels:
        projects.setdefault(project_of(wheel.name), []).append(wheel)
    for project, files in projects.items():
        d = root / project
        d.mkdir(parents=True, exist_ok=True)
        links = []
        for wheel in files:
            version = wheel.name.split("-")[1]
            yanked = ' data-yanked="superseded"' if (project, version) in yank else ""
            links.append(f'<a href="{wheel.resolve().as_uri()}#sha256={sha256(wheel)}"{yanked}>'
                         f'{html.escape(wheel.name)}</a>')
        (d / "index.html").write_text("<!DOCTYPE html><html><body>\n" + "\n".join(links) + "\n</body></html>\n")
    return root.resolve().as_uri()


def venv(tmp: str, index: str, upgrade_pip: bool) -> str:
    subprocess.run([sys.executable, "-m", "venv", tmp], check=True)
    py = str(Path(tmp) / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    if upgrade_pip:  # the pip in the folders (current), not the one the venv was created with
        subprocess.run([py, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--index-url", index,
                        "-U", "pip"], check=True)
    return py


def pip_version(py: str) -> tuple:
    out = subprocess.run([py, "-m", "pip", "--version"], capture_output=True, text=True).stdout.split()[1]
    return tuple(int(x) for x in re.findall(r"\d+", out)[:2])


def probe(py: str) -> dict:
    found = json.loads(subprocess.run([py, "-I", "-c", PROBE], capture_output=True, text=True, check=True).stdout)
    found["dists"] = {re.sub(r"[-_.]+", "-", n).lower(): v for n, v in found["dists"].items()}
    return found


def install(py: str, index: str, spec: str):
    run = subprocess.run([py, "-m", "pip", "install", "--disable-pip-version-check", "--index-url", index, spec],
                         capture_output=True, text=True)
    warnings = [l.strip() for l in run.stdout.splitlines() + run.stderr.splitlines() if "WARNING" in l
                or "ERROR" in l]
    return run, warnings


def warned_about(warnings: list) -> set:
    """The extras pip said the package does not provide."""
    return {m.group(1) for w in warnings for m in [re.search(r"does not provide the extra '([^']+)'", w)] if m}


def check_pinned(py: str, index: str, extra: str, version: str) -> bool:
    pv = pip_version(py)
    want, warn = expected(pv)[extra]
    before = probe(py)["dists"]
    spec = f"kitten-inference{extra}=={version}"
    run, warnings = install(py, index, spec)
    got = probe(py)
    new = {n: v for n, v in got["dists"].items() if n not in before}
    kitten = {n: v for n, v in new.items() if n == "kitten-inference" or n.startswith("kitten-inference-")}
    libraries = set(new) - set(kitten)
    problems = []
    if run.returncode != 0:
        problems.append(f"exit {run.returncode}")
    if kitten != {"kitten-inference": version}:
        problems.append(f"kitten distributions {kitten}, expected kitten-inference {version} only")
    if want - libraries:
        problems.append(f"missing {sorted(want - libraries)}")
    unwanted = (libraries & (EXCLUSIVE - want)) if want else libraries
    if unwanted:
        problems.append(f"not expected here: {sorted(unwanted)}")
    if set(got["subpackages"]) != subpackages():
        problems.append(f"subpackages {got['subpackages']}, expected {sorted(subpackages())}")
    if warned_about(warnings) != ({warn} if warn else set()):
        problems.append(f"pip warned about the extras {sorted(warned_about(warnings))}, expected "
                        f"{[warn] if warn else 'none'}")
    adds = ", ".join(sorted(want)) if want else "no libraries"
    print(f"{'FAIL' if problems else 'ok  '} pip {'.'.join(map(str, pv))}: pip install \"{spec}\" -> "
          f"{'/'.join(got['tags']) or 'nothing'} with {', '.join(got['subpackages']) or 'no subpackages'}; adds "
          f"{adds}{f' (+{len(libraries - want)} of their dependencies)' if libraries - want else ''}"
          + (f"  [warned about the extra {warn}]" if warn and not problems else "")
          + (f"  [{'; '.join(problems)}]" if problems else ""))
    if problems:
        print(run.stdout[-3000:], run.stderr[-3000:], sep="\n")
    return not problems


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--find-links", action="append", type=Path, required=True)
    p.add_argument("--old", type=Path)
    p.add_argument("--version", default="0.2.0")
    p.add_argument("--upgrade-pip", action="store_true", help="first install the pip found in --find-links")
    a = p.parse_args()
    wheels, left_out = served(a.find_links, a.version)
    print(f"platform {sys.platform} {platform.machine()}, Python {platform.python_version()}: kitten_inference "
          f"subpackages expected: {', '.join(sorted(subpackages()))}. {len(wheels)} wheels served"
          + (f"; build inputs left out: {', '.join(left_out)}" if left_out else ""))
    failed = 0
    with tempfile.TemporaryDirectory() as idx:
        index = make_index(wheels, Path(idx) / "simple", set())
        for extra in SPECS:
            with tempfile.TemporaryDirectory() as tmp:
                py = venv(tmp, index, a.upgrade_pip)
                failed += not check_pinned(py, index, extra, a.version)
    if a.old:
        failed += check_fallback(a, wheels)
    return 1 if failed else 0


def check_fallback(a, wheels: list) -> int:
    """Unpinned installs from a local index that also has the earlier kitten-inference 0.1.x, as PyPI has."""
    old = [w for w in sorted(a.old.glob("*.whl")) if w.name not in {x.name for x in wheels}]
    failed = 0
    for yanked in (False, True):
        with tempfile.TemporaryDirectory() as idx, tempfile.TemporaryDirectory() as tmp:
            yank = {("kitten-inference", "0.1.0"), ("kitten-inference", "0.1.1")} if yanked else set()
            index = make_index(wheels + old, Path(idx) / "simple", yank)
            where = f"index with 0.1.x{' yanked' if yanked else ''}"
            py = venv(tmp, index, upgrade_pip=True)
            for extra, warn in (("[tts-2]", set()), ("[tts-2-rocm]", {"tts-2-rocm"})):
                subprocess.run([py, "-m", "pip", "uninstall", "-y", "-q", "kitten-inference"], capture_output=True)
                spec = f"kitten-inference{extra}"
                run, warnings = install(py, index, spec)
                got = probe(py)["dists"].get("kitten-inference")
                ok = run.returncode == 0 and got == a.version and warned_about(warnings) == warn
                outcome = f"kitten-inference {got}" if run.returncode == 0 else f"error (exit {run.returncode})"
                print(f"{'ok  ' if ok else 'FAIL'} {where}: pip install \"{spec}\" -> {outcome}  [expected: "
                      f"{a.version}{', with the unknown-extra warning' if warn else ''}]")
                if not ok:
                    failed += 1
                    print(run.stdout[-3000:], run.stderr[-3000:], sep="\n")
            dest = Path(tmp) / "download"
            run = subprocess.run([py, "-m", "pip", "download", "--disable-pip-version-check", "--no-deps",
                                  "--only-binary=:all:", *FOREIGN, "--dest", str(dest), "--index-url", index,
                                  "kitten-inference"], capture_output=True, text=True)
            got = sorted(w.name for w in dest.glob("*.whl")) if dest.is_dir() else []
            if not yanked:
                want = "falls back to kitten-inference 0.1.1"
                ok = run.returncode == 0 and got == ["kitten_inference-0.1.1-py3-none-any.whl"]
            else:
                want = "an error (0.1.x yanked: no fallback)"
                ok = run.returncode != 0 and not got
            outcome = ", ".join(got) if run.returncode == 0 else f"error (exit {run.returncode})"
            print(f"{'ok  ' if ok else 'FAIL'} {where}: pip download for {FOREIGN[1]} CPython 3.12 (no {a.version} "
                  f"wheel) -> {outcome}  [expected: {want}]")
            if not ok:
                failed += 1
                print(run.stdout[-3000:], run.stderr[-3000:], sep="\n")
    return failed


if __name__ == "__main__":
    sys.exit(main())
