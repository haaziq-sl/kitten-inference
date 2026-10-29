#!/usr/bin/env python3
"""Checks plain installation and dependencies from an offline release wheelhouse."""
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
SPECS = [""]
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
    return {"": (TTS08 | (TTS2 if engines_here()["_tts2_cuda"] else set()), None)}


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
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
