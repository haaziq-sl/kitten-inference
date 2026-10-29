#!/usr/bin/env python3
"""Assembles the kitten-inference release wheels from the tested build outputs, without building anything.

    python .github/scripts/assemble_wheels.py verify-inputs --lock .github/release-inputs.json --inputs IN
    python .github/scripts/assemble_wheels.py build --lock .github/release-inputs.json --inputs IN --pure DIST \
        --source . --out OUT [--tag-run]

IN holds the wheels the lock names: the CI artifacts of the runs it records, downloaded into one folder (sub-folders
are searched too; other files are ignored). DIST holds kitten_inference-<V>-py3-none-any.whl and
kitten_inference_tts2-<base>-py3-none-any.whl, built by packages.yml from the same commit. --source is the checkout of
that commit: the overlay Python files and the licence files are read from its HEAD commit with git, so a wheel depends
on the committed tree only (never on uncommitted changes).

verify-inputs: every file the lock names is in IN, with the locked sha256 and size.
build: one kitten-inference wheel per TTS 0.8 input W, into OUT, plus OUT/assembly-report.json:
  - W's kitten_inference/tts08/** and kitten_inference_tts08.libs/**, except the lock's tts08_python_overlay files,
    which come from engines/tts08/ at HEAD; the umbrella's kitten_inference/__init__.py; the tts2 wheel's
    kitten_inference/tts2/**; and on CPython 3.9 or newer the engines: CUDA and ROCm for a manylinux_2_28_x86_64 W
    (whose wheel then carries only that tag), CUDA for win_amd64. A copied member keeps its compressed bytes, date,
    mode and RECORD row.
  - a new kitten_inference-<V>.dist-info: the umbrella's METADATA with License-File lines added, WHEEL, the umbrella's
    entry_points.txt, the licence files from the source, W's DELVEWHEEL (Windows), RECORD.
It refuses before writing any wheel when: an input differs from the lock or from its own RECORD; a member is outside
what its input may supply, or two inputs supply one path; versions or the engine ABI disagree; the extras'
requirements drifted from the engines' own (packaging, parsed); or the engine and TTS 0.8 sources at HEAD differ from
the ones the tested builds were made from (the lock's source_trees). --tag-run (a v* tag) also requires the lock's
tested_packaged_trees to match HEAD.

Never repair an assembled wheel (auditwheel, delvewheel or delocate would change the tested files):
check_assembled.py checks them instead. Needs git; build also needs the packaging module.
"""
from __future__ import annotations

import argparse
import ast
import base64
import collections
import configparser
import csv
import email.parser
import email.policy
import hashlib
import json
import os
import platform
import re
import struct
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

SCHEMA = 2
DIST = "kitten-inference"
# Every [tts-2] requirement carries this marker: TTS 2 runs on CPython 3.9 or newer, on Linux x86-64 and Windows x64.
TTS2_MARKER = ('python_version >= "3.9" and ((sys_platform == "linux" and platform_machine == "x86_64") '
               'or (sys_platform == "win32" and platform_machine == "AMD64"))')
TTS2_MIN_MINOR = 9  # cp39: the oldest CPython whose wheels carry the engines (cp38 wheels carry TTS 0.8 only)
PARTS = ("tts08", "cuda-linux", "cuda-windows", "rocm")
RUN_OF_PART = {"tts08": "tts08", "cuda-linux": "cuda", "cuda-windows": "cuda", "rocm": "rocm"}
ENGINE_PARTS = {"cuda-linux": ("kitten-inference-tts2-cu12", "cuda"),
                "cuda-windows": ("kitten-inference-tts2-cu12", "cuda"),
                "rocm": ("kitten-inference-tts2-rocm", "rocm")}  # part -> (build name, BACKEND)
# What an engine payload supplies; its licenses/ folder is left out (the dist-info licence set covers it).
ENGINE_FILE = re.compile(r"kitten_inference/_tts2_(cuda|rocm)/(__init__\.py|bin/[^/]+|lib/[^/]+)")
ENGINE_LICENCES = re.compile(r"kitten_inference/_tts2_(cuda|rocm)/licenses/.+")
TTS08_PREFIXES = ("kitten_inference/tts08/", "kitten_inference_tts08.libs/")
TTS2_PREFIX = "kitten_inference/tts2/"
UMBRELLA_FILE = "kitten_inference/__init__.py"
# The sources each tested build was made from, recorded by lock_inputs.py at the build's commit (lock source_trees):
# git tree or blob ids, None for a path that does not exist. TTS08_PYTHON is recorded file by file, without the
# overlay files, so that only those may differ at the release commit.
TTS2_SOURCES = ("engines/tts2/src", "engines/tts2/include", "engines/tts2/cli", "engines/tts2/CMakeLists.txt", "engines/tts2/cmake",
                "engines/tts2/CMakePresets.json")
ENGINE_SOURCES = {"tts08": ("engines/tts08/engine", "engines/tts08/CMakeLists.txt"), "cuda": TTS2_SOURCES, "rocm": TTS2_SOURCES}
TTS08_PYTHON = "src/kitten_inference/tts08"
# The packaged paths whose git trees the GPU tests and the test-PyPI run used (lock tested_packaged_trees).
TESTED_TREE_PATHS = ("pyproject.toml", "build_support.py", "src/kitten_inference", "umbrella", "engines/tts2/python", "engines/tts2/tools/kt2n_assets", "src/kitten_inference/tts08", "NOTICE",
                     "LICENSE", "engines/tts08/NOTICE", "engines/tts08/licenses", "engines/tts2/NOTICE", "engines/tts2/licenses")
# Licence files: dist-info/licenses/<path> from <path> in the source, so each NOTICE's "licenses/" stays true.
LICENCE_FILES = ("LICENSE", "NOTICE", "engines/tts08/NOTICE", "engines/tts2/NOTICE")
LICENCE_DIRS = ("engines/tts08/licenses", "engines/tts2/licenses")
LGPL_SOURCE = {"engines/tts2/licenses/lgpl-source/atempo.cc": "engines/tts2/src/post/atempo.cc",
               "engines/tts2/licenses/lgpl-source/atempo.h": "engines/tts2/include/kt2n/post/atempo.h"}
# Members written here get hatchling's fixed date, so building twice from one tree gives the same bytes.
DATE = (2020, 2, 2, 0, 0, 0)
GENERATOR = "kitten-inference assemble_wheels.py"
CHUNK = 1 << 20
WHEEL_NAME = re.compile(r"(?P<dist>[^-]+)-(?P<version>[^-]+)-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>[^-]+)\.whl")
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


def refuse(msg: str):
    raise SystemExit(f"error: {msg}")


def canonical(name: str) -> str:
    """PEP 503 name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def record_hash(digest: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def hex_of(row_hash: str) -> str:
    """'sha256=<urlsafe base64>' (a RECORD hash) as hex."""
    algo, _, b64 = row_hash.partition("=")
    if algo != "sha256":
        refuse(f"RECORD hash {row_hash!r} is not sha256")
    return base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4)).hex()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def member_digest(z: zipfile.ZipFile, info: zipfile.ZipInfo):
    """(sha256 digest, size) of a member, streamed (the ROCm library is 85 MB); zipfile checks the CRC too."""
    h, n = hashlib.sha256(), 0
    with z.open(info) as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
            n += len(block)
    return h.digest(), n


def expand_tags(py: str, abi: str, plat: str) -> set:
    """The (python, abi, platform) tags of a compressed tag set (cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64
    gives two)."""
    return {(p, a, t) for p in py.split(".") for a in abi.split(".") for t in plat.split(".")}


def headers(metadata: bytes):
    return email.parser.BytesParser(policy=email.policy.compat32).parsebytes(metadata, headersonly=True)


def constants(source: bytes) -> dict:
    """Module-level NAME = <literal> assignments of a Python file."""
    out = {}
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)):
            out[node.targets[0].id] = node.value.value
    return out


class Wheel:
    """An input wheel, read once. Refuses unless every member matches its RECORD row and the file name, WHEEL and
    METADATA agree."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.file = self.path.name
        m = WHEEL_NAME.fullmatch(self.file)
        if not m:
            refuse(f"{self.file}: not a wheel file name (name-version-python-abi-platform.whl)")
        self.plat_text = m["plat"]
        with zipfile.ZipFile(self.path) as z:
            infos = z.infolist()
            twice = sorted(n for n, c in collections.Counter(i.filename for i in infos).items() if c > 1)
            if twice:
                refuse(f"{self.file}: {twice[0]} is in the archive twice")
            tops = {i.filename.split("/", 1)[0] for i in infos if "/" in i.filename}
            dist_infos = sorted(t for t in tops if t.endswith(".dist-info"))
            if len(dist_infos) != 1:
                refuse(f"{self.file}: {len(dist_infos)} .dist-info folders, expected 1")
            self.dist_info = dist_infos[0]
            record = f"{self.dist_info}/RECORD"
            # path -> (the row as it is written, hash, size); CRLF or LF line ends are both found in the inputs.
            self.rows = {}
            for line in self._read(z, record).decode("utf-8").splitlines():
                if not line:
                    continue
                fields = next(csv.reader([line]))
                if len(fields) != 3:
                    refuse(f"{self.file}: RECORD row {line!r} does not have 3 fields")
                if fields[0] in self.rows:
                    refuse(f"{self.file}: RECORD lists {fields[0]} twice")
                self.rows[fields[0]] = (line, fields[1], fields[2])
            self.members = {}  # path -> ZipInfo, files only (directory entries are never copied)
            for info in infos:
                if info.is_dir():
                    continue
                if info.flag_bits & 0x1:
                    refuse(f"{self.file}: {info.filename} is encrypted")
                if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    refuse(f"{self.file}: {info.filename} uses compression method {info.compress_type}")
                if max(info.file_size, info.compress_size, info.header_offset) >= 0xFFFFFFFF:
                    refuse(f"{self.file}: {info.filename} needs zip64, which these wheels never use")
                if info.filename == record:
                    continue
                row = self.rows.get(info.filename)
                if row is None:
                    refuse(f"{self.file}: {info.filename} is not in its RECORD")
                digest, size = member_digest(z, info)
                if row[1] != record_hash(digest) or row[2] != str(size):
                    refuse(f"{self.file}: {info.filename} does not match its RECORD row")
                self.members[info.filename] = info
            stale = sorted(p for p in self.rows if p != record and p not in self.members)
            if stale:
                refuse(f"{self.file}: RECORD lists {stale[0]}, which the wheel lacks")
            self.metadata = self._read(z, f"{self.dist_info}/METADATA")
            wheel_text = self._read(z, f"{self.dist_info}/WHEEL").decode("utf-8")
        meta = headers(self.metadata)
        self.name = canonical(meta["Name"] or "")
        self.version = (meta["Version"] or "").strip()
        if canonical(m["dist"]) != self.name or m["version"] != self.version:
            refuse(f"{self.file}: the file name does not match METADATA ({meta['Name']} {self.version})")
        self.tags = [tuple(line.split(":", 1)[1].strip().split("-")) for line in wheel_text.splitlines()
                     if line.startswith("Tag:")]
        if set(self.tags) != expand_tags(m["py"], m["abi"], m["plat"]):
            refuse(f"{self.file}: WHEEL has the tags {['-'.join(t) for t in self.tags]}, the file name others")
        self.requires = meta.get_all("Requires-Dist") or []

    def _read(self, z: zipfile.ZipFile, path: str) -> bytes:
        try:
            return z.read(path)
        except KeyError:
            refuse(f"{self.file}: {path} is missing")

    def read(self, path: str) -> bytes:
        with zipfile.ZipFile(self.path) as z:
            return self._read(z, path)


class Source:
    """The release commit (HEAD, or another revision) of a checkout, read with git."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.commit = self._git("rev-parse", "--verify", "HEAD^{commit}").decode().strip()

    def _git(self, *args) -> bytes:
        try:
            r = subprocess.run(["git", "-C", str(self.root), *args], capture_output=True)
        except OSError as exc:
            refuse(f"cannot run git ({exc})")
        if r.returncode:
            refuse(f"git {' '.join(args)} in {self.root}: {r.stderr.decode('utf-8', 'replace').strip()}")
        return r.stdout

    def has_commit(self, rev: str) -> bool:
        return subprocess.run(["git", "-C", str(self.root), "cat-file", "-e", f"{rev}^{{commit}}"],
                              capture_output=True).returncode == 0

    def object_id(self, path: str, rev: str = "HEAD"):
        """The git tree or blob id of path at rev, or None when it does not exist there."""
        r = subprocess.run(["git", "-C", str(self.root), "rev-parse", "--verify", "--quiet", f"{rev}:./{path}"],
                           capture_output=True)
        return r.stdout.decode().strip() if r.returncode == 0 else None

    def tree(self, path: str, rev: str = "HEAD") -> dict:
        """path -> blob id of every file under path at rev."""
        out = {}
        for entry in self._git("ls-tree", "-r", "-z", rev, "--", path).decode("utf-8").split("\0"):
            if entry:
                info, name = entry.split("\t", 1)
                out[name] = info.split()[2]
        return out

    def read(self, path: str, rev: str = "HEAD") -> bytes:
        return self._git("cat-file", "blob", f"{rev}:./{path}")


def source_trees(src: Source, key: str, overlay, rev: str = "HEAD") -> dict:
    """What lock source_trees records for one build (key tts08, cuda or rocm), at rev."""
    out = {p: src.object_id(p, rev) for p in ENGINE_SOURCES[key]}
    if key == "tts08":
        skip = {f"src/{p}" for p in overlay}
        out.update({p: oid for p, oid in src.tree(TTS08_PYTHON, rev).items() if p not in skip})
    return out


def tested_trees(src: Source, rev: str = "HEAD") -> dict:
    return {p: src.object_id(p, rev) for p in TESTED_TREE_PATHS}


def check_overlay(paths):
    for p in paths:
        if (not p.startswith("kitten_inference/tts08/") or not p.endswith(".py") or p.endswith("/__init__.py")
                or "\\" in p or any(part in ("", ".", "..") for part in p.split("/"))):
            # tts08/__init__.py is never replaced: on Windows the tested one holds the delvewheel patch.
            refuse(f"tts08_python_overlay: {p!r} must be a .py file under kitten_inference/tts08/ other than "
                   "__init__.py")
    if len(set(paths)) != len(paths):
        refuse("tts08_python_overlay lists a file twice")


def load_lock(path: Path) -> dict:
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    if lock.get("schema") != SCHEMA:
        refuse(f"{path}: schema {lock.get('schema')!r}, this script reads schema {SCHEMA}")
    if not re.fullmatch(r"\d+(\.\d+)*", str(lock.get("version", ""))):
        refuse(f"{path}: version {lock.get('version')!r} is not a plain release version")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", str(lock.get("repository", ""))):
        refuse(f"{path}: repository {lock.get('repository')!r} is not OWNER/REPO")
    runs = lock.get("runs") or {}
    if set(runs) != set(RUN_OF_PART.values()):
        refuse(f"{path}: runs must be exactly tts08, cuda and rocm")
    for key, run in runs.items():
        if not (isinstance(run.get("id"), int) and isinstance(run.get("attempt"), int)
                and HEX40.fullmatch(str(run.get("commit", ""))) and str(run.get("workflow", "")).endswith(".yml")):
            refuse(f"{path}: runs.{key} needs id, attempt, a full commit sha and workflow")
    names = collections.Counter(r.get("file") for r in lock.get("inputs") or [])
    if not names or any(c > 1 for c in names.values()):
        refuse(f"{path}: inputs must list each file once")
    for row in lock["inputs"]:
        if (row.get("part") not in PARTS or row.get("run") != RUN_OF_PART[row["part"]]
                or not HEX64.fullmatch(str(row.get("sha256", ""))) or not isinstance(row.get("size"), int)
                or not row.get("artifact")):
            refuse(f"{path}: input row {row.get('file')!r} needs part, run, sha256, size and artifact")
    parts = collections.Counter(r["part"] for r in lock["inputs"])
    if any(parts[p] != 1 for p in ENGINE_PARTS) or not parts["tts08"]:
        refuse(f"{path}: inputs need TTS 0.8 wheels and exactly one wheel of each engine part")
    for row in lock.get("engine_files") or []:
        if (row.get("part") not in ENGINE_PARTS or not ENGINE_FILE.fullmatch(f"kitten_inference/{row.get('path')}")
                or not HEX64.fullmatch(str(row.get("sha256", "")))
                or not re.fullmatch(r"0[0-7]{3}", str(row.get("mode"))) or not isinstance(row.get("size"), int)):
            refuse(f"{path}: engine_files row {row.get('part')} {row.get('path')!r} needs part, path, size, mode "
                   "and sha256")
    if not lock.get("engine_files"):
        refuse(f"{path}: engine_files is empty")
    check_overlay(lock.get("tts08_python_overlay") or [])
    if not isinstance(lock.get("source_trees"), dict):
        refuse(f"{path}: source_trees is missing")
    if lock.get("tested_packaged_trees") is not None and not isinstance(lock["tested_packaged_trees"], dict):
        refuse(f"{path}: tested_packaged_trees must be null or a map")
    return lock


def find_wheels(folder: Path) -> dict:
    """file name -> path of every .whl under folder."""
    if not Path(folder).is_dir():
        refuse(f"{folder} is not a folder")
    found = {}
    for path in sorted(Path(folder).rglob("*.whl")):
        if path.is_file():
            if path.name in found:
                refuse(f"{path.name} is in {folder} twice ({found[path.name]}, {path})")
            found[path.name] = path
    return found


def verify_inputs(lock: dict, folder: Path) -> dict:
    """file name -> path of each lock input, after checking its size and sha256."""
    found = find_wheels(folder)
    problems = []
    for row in lock["inputs"]:
        path = found.get(row["file"])
        if path is None:
            problems.append(f"{row['file']} (artifact {row['artifact']} of run {lock['runs'][row['run']]['id']}) "
                            f"is not in {folder}")
        elif path.stat().st_size != row["size"]:
            problems.append(f"{row['file']}: {path.stat().st_size} bytes, the lock says {row['size']}")
        elif file_sha256(path) != row["sha256"]:
            problems.append(f"{row['file']}: its sha256 differs from the lock")
    if problems:
        refuse("the inputs do not match the lock:\n  " + "\n  ".join(problems))
    others = sorted(set(found) - {r["file"] for r in lock["inputs"]})
    if others:
        print(f"::notice::{len(others)} wheels in {folder} are not in the lock and are not used: "
              + ", ".join(others[:5]) + (" ..." if len(others) > 5 else ""))
    return {row["file"]: found[row["file"]] for row in lock["inputs"]}


def check_sources(lock: dict, src: Source):
    """Refuses when the engine or TTS 0.8 sources at HEAD differ from those of the tested builds."""
    changed = []
    for key in sorted(set(RUN_OF_PART.values())):
        was = lock["source_trees"].get(key)
        if not was:
            refuse(f"the lock records no source trees for the {key} build: run lock_inputs.py")
        now = source_trees(src, key, lock["tts08_python_overlay"])
        changed += [f"{p} ({key})" for p in sorted(set(was) | set(now)) if was.get(p, "-") != now.get(p, "-")]
    if changed:
        refuse(f"the sources at {src.commit[:12]} differ from those the tested builds were made from: "
               + ", ".join(changed[:10]) + (f" and {len(changed) - 10} more" if len(changed) > 10 else "")
               + ". The locked binaries would not match this commit: run a new build and GPU-test round, "
               "then lock_inputs.py.")


def check_tested_trees(lock: dict, src: Source, tag_run: bool):
    """C12: a tag releases only the packaged tree the GPU tests and the test-PyPI run used."""
    here = os.environ.get("GITHUB_REPOSITORY", "")
    if tag_run and here and here.lower() != lock["repository"].lower():  # GitHub names are not case-sensitive
        refuse(f"the lock's inputs are in {lock['repository']}, this tag run is in {here}")
    trees = lock.get("tested_packaged_trees")
    if trees is None:
        if tag_run:
            refuse("the lock's tested_packaged_trees is not filled in. Once the GPU tests and the test-PyPI run "
                   "passed on this tree: lock_inputs.py --set-tested-trees <commit>")
        print("::notice::the lock's tested_packaged_trees is not filled in yet (required for a tag run)")
        return
    now = tested_trees(src)
    differ = [p for p in TESTED_TREE_PATHS if trees.get(p, "-") != now[p]]
    if differ:
        msg = (f"the packaged files at {src.commit[:12]} differ from the tested tree (tested_packaged_trees): "
               + ", ".join(differ))
        if tag_run:
            refuse(msg)
        print(f"::warning::{msg}")


def environments():
    """A grid of target environments, for comparing markers by what they select."""
    systems = (("linux", "Linux", "posix"), ("win32", "Windows", "nt"), ("darwin", "Darwin", "posix"),
               ("android", "Android", "posix"))
    for minor in range(8, 16):
        for sys_platform, system, os_name in systems:
            for machine in ("x86_64", "AMD64", "aarch64", "arm64", "ARM64", "i686"):
                yield {"python_version": f"3.{minor}", "python_full_version": f"3.{minor}.0",
                       "sys_platform": sys_platform, "platform_system": system, "os_name": os_name,
                       "platform_machine": machine, "implementation_name": "cpython",
                       "platform_python_implementation": "CPython"}


ENVIRONMENTS = list(environments())


def selects(marker, extra: str) -> tuple:
    """Where a requirement's marker applies (None: everywhere), for one value of `extra`."""
    return tuple(True if marker is None else marker.evaluate(dict(env, extra=extra)) for env in ENVIRONMENTS)


def check_requirements(umbrella: Wheel, tts08_requires, tts2_requires):
    """Internal inputs and the public package use the same root dependency table."""
    from packaging.requirements import Requirement
    def normalized(lines):
        result = set()
        for line in lines:
            r = Requirement(line)
            result.add((canonical(r.name), r.specifier, frozenset(canonical(e) for e in r.extras), selects(r.marker, "")))
        return result
    if headers(umbrella.metadata).get_all("Provides-Extra"):
        refuse("model extras are no longer supported")
    for line in umbrella.requires:
        r = Requirement(line)
        if canonical(r.name).startswith("kitten-inference") or r.url:
            refuse(f"invalid package dependency: {line}")
    if normalized(umbrella.requires) != normalized(tts08_requires) or normalized(umbrella.requires) != normalized(tts2_requires):
        refuse("build inputs differ from the root dependency table; rebuild the Python inputs")


def check_entry_points(umbrella: bytes, tts2: bytes):
    parser = configparser.ConfigParser(delimiters=("=",), interpolation=None)
    parser.read_string(umbrella.decode())
    if dict(parser["console_scripts"]) != {"kitten-tts": "kitten_inference._cli:main"}:
        refuse("expected kitten-tts = kitten_inference._cli:main")


def licences(src: Source) -> dict:
    """dist-info/licenses/<path> -> bytes, from the source at HEAD."""
    out = {p: src.read(p) for p in LICENCE_FILES}
    for folder in LICENCE_DIRS:
        files = src.tree(folder)
        if not files:
            refuse(f"{folder} has no files at HEAD")
        out.update({p: src.read(p) for p in files})
    for target, path in LGPL_SOURCE.items():
        if target in out:
            refuse(f"{target} is in the source and would also be copied from {path}")
        out[target] = src.read(path)
    return {p.replace("engines/tts08/", "tts-0.8/").replace("engines/tts2/", "tts-2/"): value for p, value in out.items()}


def with_licence_files(metadata: bytes, names) -> bytes:
    """The umbrella's METADATA with a License-File line for every licence file, the new ones sorted after its own."""
    if b"\r" in metadata:
        refuse("the umbrella's METADATA has CR line ends")
    head, sep, body = metadata.partition(b"\n\n")
    lines = head.split(b"\n")
    key = b"License-File: "
    at = [i for i, line in enumerate(lines) if line.startswith(key)]
    have = [lines[i][len(key):].decode("utf-8") for i in at]
    if "LICENSE" not in have:
        refuse("the umbrella's METADATA has no License-File: LICENSE")
    unknown = sorted(set(have) - set(names))
    if unknown:
        refuse(f"the umbrella's METADATA names licence files the source does not have: {unknown}")
    lines[at[-1] + 1:at[-1] + 1] = [key + n.encode("utf-8") for n in sorted(set(names) - set(have))]
    return b"\n".join(lines) + sep + body


class Member:
    """One member of an assembled wheel: a raw copy of an input member (info) or new bytes (data)."""
    __slots__ = ("path", "row", "source", "info", "data", "origin")

    def __init__(self, path, row, origin, source=None, info=None, data=None):
        self.path, self.row, self.origin, self.source, self.info, self.data = path, row, origin, source, info, data


def copied(w: Wheel, path: str, new_path: str = None) -> Member:
    line, digest, size = w.rows[path]
    row = line if new_path is None else f"{new_path},{digest},{size}"
    return Member(new_path or path, row, w.file, source=w.path, info=w.members[path])


def written(path: str, data: bytes) -> Member:
    if any(c in path for c in ',"\r\n'):
        refuse(f"{path}: not a path this script writes into RECORD")
    return Member(path, f"{path},{record_hash(hashlib.sha256(data).digest())},{len(data)}", "new", data=data)


LOCAL = struct.Struct("<4sHHHHHIIIHH")
CENTRAL = struct.Struct("<4sBBHHHHHIIIHHHHHII")
END = struct.Struct("<4sHHHHIIH")


def dos_datetime(date_time):
    y, mo, d, h, mi, s = date_time
    return (h << 11) | (mi << 5) | (s // 2), ((max(y, 1980) - 1980) << 9) | (mo << 5) | d


def write_zip(path: Path, members):
    """Writes members in order. Raw copies keep their compressed bytes, CRC, date, attributes and creator; new
    members are deflated at level 9 with mode 0644 and DATE. No data descriptors and no extra fields, so the local
    and central headers agree (uv checks that). No zip64."""
    tmp = path.with_name(path.name + ".part")
    central, handles = [], {}
    try:
        with open(tmp, "wb") as out:
            for m in members:
                name = m.path.encode("utf-8")
                offset = out.tell()
                if m.info is not None:
                    i = m.info
                    flags = i.flag_bits & ~0x08
                    method, crc, csize, usize = i.compress_type, i.CRC, i.compress_size, i.file_size
                    stamp = dos_datetime(i.date_time)
                    creator, internal, external = (i.create_version, i.create_system), i.internal_attr, i.external_attr
                    src = handles.get(m.source) or handles.setdefault(m.source, open(m.source, "rb"))
                    src.seek(i.header_offset)
                    head = LOCAL.unpack(src.read(LOCAL.size))
                    if head[0] != b"PK\x03\x04":
                        refuse(f"{m.origin}: bad local header for {i.filename}")
                    src.seek(i.header_offset + LOCAL.size + head[9] + head[10])
                    payload = None
                else:
                    c = zlib.compressobj(9, zlib.DEFLATED, -15)
                    payload = c.compress(m.data) + c.flush()
                    flags, method = 0, zipfile.ZIP_DEFLATED
                    crc, csize, usize = zlib.crc32(m.data) & 0xFFFFFFFF, len(payload), len(m.data)
                    stamp = dos_datetime(DATE)
                    creator, internal, external = (20, 3), 0, 0o100644 << 16
                if not m.path.isascii():
                    flags |= 0x800
                out.write(LOCAL.pack(b"PK\x03\x04", 20, flags, method, stamp[0], stamp[1], crc, csize, usize,
                                     len(name), 0) + name)
                if payload is None:
                    left = csize
                    while left:
                        block = src.read(min(CHUNK, left))
                        if not block:
                            refuse(f"{m.origin}: {m.info.filename} is truncated")
                        out.write(block)
                        left -= len(block)
                else:
                    out.write(payload)
                central.append(CENTRAL.pack(b"PK\x01\x02", creator[0], creator[1], 20, flags, method, stamp[0],
                                            stamp[1], crc, csize, usize, len(name), 0, 0, 0, internal, external,
                                            offset) + name)
                if out.tell() >= 0xFFFFFFFF:
                    refuse(f"{path.name} would need zip64")
            start = out.tell()
            for entry in central:
                out.write(entry)
            if out.tell() >= 0xFFFFFFFF or len(central) >= 0xFFFF:
                refuse(f"{path.name} would need zip64")
            out.write(END.pack(b"PK\x05\x06", 0, 0, len(central), len(central), out.tell() - start, start, 0))
    except BaseException:
        for f in handles.values():
            f.close()
        handles = {}
        if tmp.exists():
            tmp.unlink()
        raise
    finally:
        for f in handles.values():
            f.close()
    os.replace(tmp, path)


def engines_for(w: Wheel):
    """(engine parts, output tags, platform text of the file name) for a TTS 0.8 input."""
    pys, abis = {t[0] for t in w.tags}, {t[1] for t in w.tags}
    m = re.fullmatch(r"cp3(\d+)", next(iter(pys))) if len(pys) == 1 else None
    if not m or len(abis) != 1:
        refuse(f"{w.file}: expected one CPython tag (cp3X) and one ABI tag")
    minor = int(m.group(1))
    x86 = [t for t in w.tags if t[2] == "manylinux_2_28_x86_64"]
    if minor >= TTS2_MIN_MINOR and x86:
        # The engine wheels were built and checked for manylinux_2_28 only: the 2_27 alias is dropped.
        return ["cuda-linux", "rocm"], x86, "manylinux_2_28_x86_64"
    if minor >= TTS2_MIN_MINOR and any(t[2] == "win_amd64" for t in w.tags):
        return ["cuda-windows"], list(w.tags), w.plat_text
    return [], list(w.tags), w.plat_text


def select_tts08(w: Wheel, overlay):
    """(members to copy, the dropped paths) of a TTS 0.8 input."""
    keep, dropped = [], []
    for path in sorted(w.members):
        if path.startswith(w.dist_info + "/"):
            rel = path[len(w.dist_info) + 1:]
            if rel == "DELVEWHEEL":
                keep.append((path, f"{{dist_info}}/{rel}"))
            elif rel in ("METADATA", "WHEEL") or rel.startswith(("licenses/", "sboms/")):
                # Replaced by the new dist-info; the auditwheel SBOM names the build-only distribution (C3).
                dropped.append(path)
            else:
                refuse(f"{w.file}: {path} is not a file the assembled wheel can carry")
        elif not path.startswith(TTS08_PREFIXES):
            refuse(f"{w.file}: {path} is outside kitten_inference/tts08/ and kitten_inference_tts08.libs/")
        elif path in overlay:
            dropped.append(path)  # replaced by the source's file
        else:
            keep.append((path, None))
    return keep, dropped


def select_payload(part: str, p: Wheel):
    prefix = "kitten_inference/_tts2_cuda/" if ENGINE_PARTS[part][1] == "cuda" else "kitten_inference/_tts2_rocm/"
    keep, dropped = [], []
    for path in sorted(p.members):
        if path.startswith(p.dist_info + "/"):
            continue
        if path.startswith(prefix) and ENGINE_FILE.fullmatch(path):
            keep.append(path)
        elif path.startswith(prefix) and ENGINE_LICENCES.fullmatch(path):
            dropped.append(path)  # an older copy of the licence texts the dist-info carries (C16)
        else:
            refuse(f"{p.file}: {path} is outside {prefix}{{__init__.py,bin/,lib/}}")
    return keep, dropped


def build(lock_path: Path, inputs: Path, pure: Path, source: Path, out: Path, tag_run: bool = False) -> dict:
    lock = load_lock(lock_path)
    base, overlay_paths = lock["version"], lock["tts08_python_overlay"]
    src = Source(source)
    check_sources(lock, src)
    check_tested_trees(lock, src, tag_run)
    files = verify_inputs(lock, inputs)

    rows = {r["file"]: r for r in lock["inputs"]}
    tts08 = [Wheel(files[r["file"]]) for r in lock["inputs"] if r["part"] == "tts08"]
    payloads = {r["part"]: Wheel(files[r["file"]]) for r in lock["inputs"] if r["part"] != "tts08"}
    found = find_wheels(pure)
    umbrellas = [p for n, p in found.items() if re.fullmatch(r"kitten_inference-[^-]+-py3-none-any\.whl", n)]
    tts2s = [p for n, p in found.items() if re.fullmatch(r"kitten_inference_tts2-[^-]+-py3-none-any\.whl", n)]
    if len(umbrellas) != 1 or len(tts2s) != 1:
        refuse(f"{pure} must hold one kitten_inference-*-py3-none-any.whl and one "
               f"kitten_inference_tts2-*-py3-none-any.whl (found {len(umbrellas)} and {len(tts2s)})")
    umbrella, tts2 = Wheel(umbrellas[0]), Wheel(tts2s[0])

    # Versions: one base version everywhere; the umbrella may be a post release of it (only its version changes).
    version = umbrella.version
    if umbrella.name != DIST:
        refuse(f"{umbrella.file}: Name is {umbrella.name}, expected {DIST}")
    if not re.fullmatch(re.escape(base) + r"(\.post\d+)?", version):
        refuse(f"kitten-inference {version} is neither {base} (the lock's version) nor a post release of it")
    if umbrella.dist_info != f"kitten_inference-{version}.dist-info":
        refuse(f"{umbrella.file}: its dist-info folder is {umbrella.dist_info}")
    for w in tts08 + list(payloads.values()) + [tts2]:
        if w.version != base:
            refuse(f"{w.file}: version {w.version}, the lock is for {base}")
    for w in tts08:
        if w.name != "kitten-inference-tts08":
            refuse(f"{w.file}: Name is {w.name}, expected kitten-inference-tts08")
    if tts2.name != "kitten-inference-tts2":
        refuse(f"{tts2.file}: Name is {tts2.name}, expected kitten-inference-tts2")

    # The members each input supplies.
    umbrella_files = [p for p in umbrella.members if not p.startswith(umbrella.dist_info + "/")]
    if UMBRELLA_FILE not in umbrella_files or any(not p.startswith("kitten_inference/") or p.count("/") != 1 or not p.endswith(".py") for p in umbrella_files):
        refuse(f"{umbrella.file}: expected only {UMBRELLA_FILE} besides its dist-info, found {sorted(umbrella_files)}")
    umbrella_info = {p[len(umbrella.dist_info) + 1:] for p in umbrella.members
                     if p.startswith(umbrella.dist_info + "/")}
    if not umbrella_info <= {"METADATA", "WHEEL", "entry_points.txt", "licenses/LICENSE"}:
        refuse(f"{umbrella.file}: unexpected dist-info files {sorted(umbrella_info)}")
    entry_points = f"{umbrella.dist_info}/entry_points.txt"
    if entry_points not in umbrella.members:
        refuse(f"{umbrella.file}: no entry_points.txt (the kitten-tts2 command)")
    tts2_files = sorted(p for p in tts2.members if not p.startswith(tts2.dist_info + "/"))
    outside = [p for p in tts2_files if not p.startswith(TTS2_PREFIX)]
    if outside:
        refuse(f"{tts2.file}: {outside[0]} is outside {TTS2_PREFIX}")
    tts2_entry = f"{tts2.dist_info}/entry_points.txt"
    check_entry_points(umbrella.read(entry_points), tts2.read(tts2_entry) if tts2_entry in tts2.members else b"")
    selected = {part: select_payload(part, p) for part, p in payloads.items()}
    tts08_selected = {w.file: select_tts08(w, overlay_paths) for w in tts08}

    # Engine ABI and versions, and the engine files against the lock (the files the GPU tests ran).
    abi = constants(tts2.read(f"{TTS2_PREFIX}_abi.py")).get("ABI_VERSION")
    for part, p in payloads.items():
        name, backend = ENGINE_PARTS[part]
        if p.name != name:
            refuse(f"{p.file}: Name is {p.name}, expected {name} for {part}")
        init = constants(p.read(f"kitten_inference/_tts2_{backend}/__init__.py"))
        if init.get("BACKEND") != backend or init.get("VERSION") != base:
            refuse(f"{p.file}: BACKEND {init.get('BACKEND')!r} VERSION {init.get('VERSION')!r}, expected "
                   f"{backend!r} {base!r}")
        if init.get("ABI") != abi:
            refuse(f"{p.file}: engine ABI {init.get('ABI')}, kitten_inference.tts2 needs ABI {abi}")
        want = {f"kitten_inference/{r['path']}": r for r in lock["engine_files"] if r["part"] == part}
        got = selected[part][0]
        for path in sorted(set(want) | set(got)):
            info = p.members.get(path)
            r = want.get(path)
            if (r is None or info is None or info.file_size != r["size"] or hex_of(p.rows[path][1]) != r["sha256"]
                    or "%04o" % (info.external_attr >> 16 & 0o7777) != r["mode"]):
                refuse(f"{p.file}: {path} differs from the lock's engine_files")

    # METADATA: the umbrella's, after the drift checks, with every licence file named.
    if len({tuple(sorted(w.requires)) for w in tts08}) != 1:
        refuse("the TTS 0.8 inputs do not all have the same Requires-Dist")
    check_requirements(umbrella, tts08[0].requires, tts2.requires)
    from packaging.specifiers import SpecifierSet
    requires_python = SpecifierSet(headers(umbrella.metadata)["Requires-Python"] or "")
    lic = licences(src)
    if umbrella.read(f"{umbrella.dist_info}/licenses/LICENSE") != lic["LICENSE"]:
        refuse("the umbrella wheel's LICENSE differs from LICENSE at HEAD")
    metadata = with_licence_files(umbrella.metadata, list(lic))
    overlay = {p: src.read(f"src/{p}") for p in overlay_paths}

    # Plan every wheel before writing any.
    dist_info = f"kitten_inference-{version}.dist-info"
    plans = {}
    for w in tts08:
        parts, tags, plat = engines_for(w)
        py, abi_tag = tags[0][0], tags[0][1]
        if not requires_python.contains(f"3.{py[3:]}"):
            refuse(f"{w.file}: Requires-Python {requires_python} excludes {py}")
        name = f"kitten_inference-{version}-{py}-{abi_tag}-{plat}.whl"
        if name in plans:
            refuse(f"two inputs give {name}")
        keep, dropped = tts08_selected[w.file]
        members = [copied(w, p, new.format(dist_info=dist_info) if new else None) for p, new in keep]
        members += [written(p, data) for p, data in sorted(overlay.items())]
        members += [copied(umbrella, p) for p in umbrella_files]
        members += [copied(tts2, p) for p in tts2_files]
        for part in parts:
            members += [copied(payloads[part], p) for p in selected[part][0]]
            dropped += selected[part][1]
        wheel = ("Wheel-Version: 1.0\nGenerator: " + GENERATOR + "\nRoot-Is-Purelib: false\n"
                 + "".join(f"Tag: {'-'.join(t)}\n" for t in tags)).encode()
        members += [written(f"{dist_info}/METADATA", metadata), written(f"{dist_info}/WHEEL", wheel),
                    copied(umbrella, entry_points)]
        members += [written(f"{dist_info}/licenses/{p}", data) for p, data in lic.items()]
        seen = {}
        for m in members:
            if m.path in seen:
                refuse(f"{name}: {m.path} would come from both {seen[m.path]} and {m.origin}")
            seen[m.path] = m.origin
        info_members = sorted((m for m in members if m.path.startswith(dist_info + "/")), key=lambda m: m.path)
        members = sorted((m for m in members if not m.path.startswith(dist_info + "/")), key=lambda m: m.path)
        members += info_members
        record = "\n".join([m.row for m in members] + [f"{dist_info}/RECORD,,"]) + "\n"
        members.append(Member(f"{dist_info}/RECORD", None, "new", data=record.encode("utf-8")))
        inputs_used = [w] + [payloads[p] for p in parts]
        plans[name] = (w, members, tags, dropped, inputs_used)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        refuse(f"{out} is not empty")
    pure_sha = {umbrella.file: file_sha256(umbrella.path), tts2.file: file_sha256(tts2.path)}
    report = {"version": version, "base_version": base, "commit": src.commit,
              "lock_sha256": file_sha256(Path(lock_path)), "python": platform.python_version(),
              "zlib": zlib.ZLIB_RUNTIME_VERSION, "inputs_checked": len(lock["inputs"]), "wheels": []}
    for name in sorted(plans):
        w, members, tags, dropped, inputs_used = plans[name]
        path = out / name
        write_zip(path, members)
        sha, size = file_sha256(path), path.stat().st_size
        new = [m.path for m in members if m.origin == "new"]
        report["wheels"].append({
            "file": name, "sha256": sha, "size": size, "tags": ["-".join(t) for t in tags],
            "inputs": [{"file": i.file, "part": rows[i.file]["part"], "sha256": rows[i.file]["sha256"]}
                       for i in inputs_used]
            + [{"file": f, "part": "pure", "sha256": s} for f, s in sorted(pure_sha.items())],
            "copied": len(members) - len(new), "overlay": sorted(overlay), "new": new, "dropped": sorted(dropped)})
        print(f"{name}  {size} bytes  sha256 {sha}  ({len(members) - len(new)} files copied unchanged, "
              f"{len(new)} written)")
    (out / "assembly-report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"{len(lock['inputs'])} inputs match the lock; {len(plans)} wheels in {out}; "
          f"{len(overlay)} Python files from {src.commit[:12]}")
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify-inputs", help="check the downloaded inputs against the lock")
    v.add_argument("--lock", type=Path, required=True)
    v.add_argument("--inputs", type=Path, required=True)
    b = sub.add_parser("build", help="write the kitten-inference wheels")
    b.add_argument("--lock", type=Path, required=True)
    b.add_argument("--inputs", type=Path, required=True)
    b.add_argument("--pure", type=Path, required=True, help="folder with the umbrella and tts2 wheels")
    b.add_argument("--source", type=Path, required=True, help="checkout of the release commit")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--tag-run", action="store_true", help="a v* tag run: the tested tree must match HEAD")
    a = p.parse_args(argv)
    if a.command == "verify-inputs":
        lock = load_lock(a.lock)
        verify_inputs(lock, a.inputs)
        parts = collections.Counter(r["part"] for r in lock["inputs"])
        print(f"{len(lock['inputs'])} inputs match the lock ({parts['tts08']} TTS 0.8 wheels, "
              f"{len(lock['inputs']) - parts['tts08']} engine wheels)")
    else:
        build(a.lock, a.inputs, a.pure, a.source, a.out, tag_run=a.tag_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
