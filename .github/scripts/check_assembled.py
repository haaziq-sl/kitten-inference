#!/usr/bin/env python3
"""Checks the assembled kitten-inference wheels, independently of assemble_wheels.py (it shares no code with it).

    python .github/scripts/check_assembled.py --lock .github/release-inputs.json --inputs IN --source . OUT/*.whl

IN is the folder of inputs the wheels were assembled from, --source the release commit's checkout. Checked:
  - inputs: every file the lock names is in IN with its sha256 and size;
  - the set: one wheel per TTS 0.8 input, named as the lock implies, and the (Python, platform) pairs of
    pypi_ready.COMPLETE["kitten-inference"];
  - each wheel: one kitten_inference-<V>.dist-info; WHEEL tags equal the file name's, Root-Is-Purelib false; RECORD
    lists every member once, with the right hash and size; no directory entries, __pycache__, .pyc or SBOM; every
    License-File is there and every licence file is listed; the kitten-tts2 command; at most 100 MB; no member path
    over 90 characters (the dist-info ones counted as for a .post99 version);
  - METADATA and entry_points.txt are the same bytes in every wheel; the extras' requirements, parsed, match the TTS
    0.8 build's and tts-2/python/pyproject.toml's, and every [tts-2] one has the Linux x86-64 / Windows x64 marker;
  - byte for byte: every TTS 0.8 member equals its input's RECORD row and zip attributes, except the overlay files,
    which equal the source; the TTS 0.8 Python and JSON files equal the source (CRLF and the delvewheel block aside);
    kitten_inference/__init__.py, tts2/ and the licence files equal the source; the engine files are exactly the
    lock's engine_files rows, in the wheels that carry engines (CPython 3.9 or newer: CUDA and ROCm in
    manylinux_2_28_x86_64, CUDA in win_amd64), with the Linux programs and the ROCm library 0755;
  - on manylinux_2_28_x86_64 wheels, every ELF file passes `check_wheel_rocm.py symbols` (extracted one at a time);
  - every library in kitten_inference_tts08.libs/ is named in the TTS 0.8 NOTICE;
  - the engine and TTS 0.8 sources at HEAD equal the lock's source_trees.
Prints one ::error:: line per problem and exits 1 when there is any; writes a table of the wheels to
$GITHUB_STEP_SUMMARY when it is set. Needs git, readelf (binutils) and the packaging module.
"""
from __future__ import annotations

import argparse
import base64
import configparser
import csv
import hashlib
import io
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from email.parser import BytesHeaderParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pypi_ready  # noqa: E402

MAX_BYTES, WARN_BYTES = 100 * 1024 * 1024, 90 * 1024 * 1024
MAX_PATH = 90  # Windows' 260-character limit with long paths off, less the venv's site-packages path
C2_MARKER = ('python_version >= "3.9" and ((sys_platform == "linux" and platform_machine == "x86_64") '
             'or (sys_platform == "win32" and platform_machine == "AMD64"))')
ENGINE_DIRS = {"cuda-linux": "_tts2_cuda", "cuda-windows": "_tts2_cuda", "rocm": "_tts2_rocm"}
DELVEWHEEL_BLOCK = re.compile(rb"\n\n# start delvewheel patch\n.*?\n# end delvewheel patch\n\n", re.S)
LIBRARY = re.compile(r"kitten_inference_tts08\.libs/[^/]+\.(dll|dylib|so)(\.\d+)*")
ALWAYS = ("kitten_inference/__init__.py", "kitten_inference/tts2/__init__.py",
          "kitten_inference/tts2/_assets/prepare.py")


def b64(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1 << 20)
            if not block:
                return h.hexdigest()
            h.update(block)


def name_tags(filename: str):
    """(distribution, version, python, abi, [platforms]) of a wheel file name, or None."""
    parts = filename[:-4].split("-") if filename.endswith(".whl") else []
    if len(parts) != 5:
        return None
    return parts[0], parts[1], parts[2], parts[3], parts[4].split(".")


def engines_expected(py: str, platforms) -> set:
    minor = int(py[3:]) if re.fullmatch(r"cp3\d+", py) else 0
    if minor < 9:
        return set()
    if "manylinux_2_28_x86_64" in platforms:
        return {"cuda-linux", "rocm"}
    return {"cuda-windows"} if "win_amd64" in platforms else set()


def record_rows(data: bytes) -> dict:
    rows = {}
    for fields in csv.reader(io.StringIO(data.decode("utf-8"))):
        if fields:
            rows.setdefault(fields[0], []).append(fields[1:])
    return rows


class Tree:
    """The source checkout: file bytes from disk, the HEAD commit's trees from git."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.commit = self.git("rev-parse", "HEAD").strip()

    def git(self, *args, check=True) -> str:
        r = subprocess.run(["git", "-C", str(self.root), *args], capture_output=True, encoding="utf-8")
        if check and r.returncode:
            raise SystemExit(f"error: git {' '.join(args)}: {r.stderr.strip()}")
        return r.stdout if r.returncode == 0 else ""

    def files(self, folder: str) -> list:
        """The files git tracks under folder (relative to the checkout), without __pycache__ and .pyc."""
        names = [n for n in self.git("ls-files", "-z", "--", folder).split("\0") if n]
        return [n for n in names if "__pycache__" not in n.split("/") and not n.endswith((".pyc", ".pyo"))]

    def read(self, path: str):
        p = self.root / path
        return p.read_bytes() if p.is_file() else None

    def head_id(self, path: str):
        return self.git("rev-parse", "--verify", "--quiet", f"HEAD:./{path}", check=False).strip() or None

    def head_files(self, folder: str) -> dict:
        out = {}
        for entry in self.git("ls-tree", "-r", "-z", "HEAD", "--", folder).split("\0"):
            if entry:
                info, name = entry.split("\t", 1)
                out[name] = info.split()[2]
        return out


def run_symbols(source: Path):
    """The default ELF check: tts-2/packaging/check_wheel_rocm.py symbols FILE (None when it passes)."""
    script = Path(source) / "tts-2" / "packaging" / "check_wheel_rocm.py"

    def check(path: Path):
        r = subprocess.run([sys.executable, str(script), "symbols", str(path)], capture_output=True, text=True)
        return None if r.returncode == 0 else (r.stderr or r.stdout).strip()
    return check


class Check:
    def __init__(self, lock: dict, inputs: Path, source: Path, expected_pairs=None, elf=None):
        self.lock, self.inputs = lock, Path(inputs)
        self.tree = Tree(source)
        self.expected_pairs = expected_pairs
        self.elf = elf if elf is not None else run_symbols(source)
        self.problems, self.warnings = [], []
        self.copied = self.elf_files = 0
        self.engine_rows_seen, self.elf_seen = set(), set()
        self.summary = []

    def fail(self, msg: str):
        self.problems.append(msg)

    # ---------------------------------------------------------------- inputs

    def load_inputs(self):
        found = {}
        for p in sorted(self.inputs.rglob("*.whl")):
            if p.name in found:
                self.fail(f"{p.name} is in {self.inputs} twice")
            found[p.name] = p
        self.input_records, self.input_attrs, self.input_meta = {}, {}, {}
        for row in self.lock["inputs"]:
            p = found.get(row["file"])
            if p is None or p.stat().st_size != row["size"] or sha256_file(p) != row["sha256"]:
                self.fail(f"input {row['file']} is missing from {self.inputs} or differs from the lock")
                continue
            with zipfile.ZipFile(p) as z:
                record = next(n for n in z.namelist() if re.fullmatch(r"[^/]+\.dist-info/RECORD", n))
                self.input_records[row["file"]] = {k: v[0] for k, v in record_rows(z.read(record)).items()}
                self.input_attrs[row["file"]] = {i.filename: (i.external_attr, i.create_system, i.date_time)
                                                 for i in z.infolist()}
                self.input_meta[row["file"]] = z.read(record[:-len("RECORD")] + "METADATA")
        # The lock's engine_files rows are the payload wheels' own RECORD rows.
        payload = {r["part"]: r["file"] for r in self.lock["inputs"] if r["part"] in ENGINE_DIRS}
        for r in self.lock["engine_files"]:
            row = self.input_records.get(payload.get(r["part"]), {}).get(f"kitten_inference/{r['path']}")
            want = base64.urlsafe_b64encode(bytes.fromhex(r["sha256"])).rstrip(b"=").decode()
            if row is None or row != [f"sha256={want}", str(r["size"])]:
                self.fail(f"engine_files {r['part']} {r['path']} is not its payload wheel's RECORD row")

    # ---------------------------------------------------------------- sources

    def check_sources(self):
        """C9: the engine and TTS 0.8 sources at HEAD are the ones the tested builds were made from."""
        overlay = {f"tts-0.8/{p}" for p in self.lock["tts08_python_overlay"]}
        for key, recorded in sorted(self.lock["source_trees"].items()):
            python = {p for p in recorded if p.startswith("tts-0.8/kitten_inference/tts08/")}
            for path, oid in sorted(recorded.items()):
                if path not in python and self.tree.head_id(path) != oid:
                    self.fail(f"{path} at HEAD differs from the {key} build's source")
            if key == "tts08":
                now = {p: o for p, o in self.tree.head_files("tts-0.8/kitten_inference/tts08").items()
                       if p not in overlay}
                for path in sorted(set(now) | python):
                    if now.get(path) != recorded.get(path):
                        self.fail(f"{path} at HEAD differs from the TTS 0.8 build's source (only the overlay "
                                  "files may differ)")

    def licence_sources(self) -> dict:
        """dist-info/licenses/<path> -> source path."""
        out = {p: p for p in ("LICENSE", "NOTICE", "tts-0.8/NOTICE", "tts-2/NOTICE")}
        for folder in ("tts-0.8/licenses", "tts-2/licenses"):
            out.update({p: p for p in self.tree.files(folder)})
        out["tts-2/licenses/lgpl-source/atempo.cc"] = "tts-2/src/post/atempo.cc"
        out["tts-2/licenses/lgpl-source/atempo.h"] = "tts-2/include/kt2n/post/atempo.h"
        return out

    def tts2_sources(self) -> dict:
        """kitten_inference/tts2/<path> -> source path (what tts-2/python/hatch_build.py copies)."""
        out = {}
        for p in self.tree.files("tts-2/python/kitten_inference/tts2"):
            out["kitten_inference/tts2/" + p[len("tts-2/python/kitten_inference/tts2/"):]] = p
        for p in self.tree.files("tts-2/tools/kt2n_assets"):
            out["kitten_inference/tts2/_assets/" + p[len("tts-2/tools/kt2n_assets/"):]] = p
        for p in self.tree.files("tts-2/licenses"):
            out["kitten_inference/tts2/licenses/" + p[len("tts-2/licenses/"):]] = p
        out["kitten_inference/tts2/NOTICE"] = "tts-2/NOTICE"
        return out

    # ---------------------------------------------------------------- METADATA

    def check_metadata(self, meta: bytes, version: str, tts08_meta: bytes):
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
        from packaging.markers import Marker

        h = BytesHeaderParser().parsebytes(meta)
        if canonicalize_name(h["Name"] or "") != "kitten-inference" or (h["Version"] or "").strip() != version:
            self.fail(f"METADATA says {h['Name']} {h['Version']}, the file names kitten-inference {version}")
        if {canonicalize_name(e) for e in h.get_all("Provides-Extra") or []} != {"tts-0-8", "tts-2"}:
            self.fail(f"METADATA provides the extras {h.get_all('Provides-Extra')}, expected tts-0-8 and tts-2")
        grid = [dict(python_version=f"3.{m}", python_full_version=f"3.{m}.0", sys_platform=s, platform_system=ps,
                     os_name=o, platform_machine=mach, implementation_name="cpython",
                     platform_python_implementation="CPython")
                for m, (s, ps, o), mach in itertools.product(
                    range(8, 16), [("linux", "Linux", "posix"), ("win32", "Windows", "nt"),
                                   ("darwin", "Darwin", "posix"), ("android", "Android", "posix")],
                    ["x86_64", "AMD64", "aarch64", "arm64", "ARM64", "i686"])]

        def where(marker, extra):
            return [marker is None or marker.evaluate(dict(env, extra=extra)) for env in grid]

        groups = {"tts-0-8": {}, "tts-2": {}}
        for line in h.get_all("Requires-Dist") or []:
            r = Requirement(line)
            owners = [e for e in groups if any(where(r.marker, e))]
            if canonicalize_name(r.name).startswith("kitten-inference") or any(where(r.marker, "")) or len(owners) != 1:
                self.fail(f"Requires-Dist {line!r}: each requirement must belong to one extra and name no "
                          "kitten-inference distribution")
                continue
            groups[owners[0]][canonicalize_name(r.name)] = r
        c2 = where(Marker(C2_MARKER), "")
        for name, r in sorted(groups["tts-2"].items()):
            if where(r.marker, "tts-2") != c2:
                self.fail(f"[tts-2] {r}: its marker must be {C2_MARKER} (C2)")

        def compare(extra, mine, lines, whose):
            theirs = {canonicalize_name(r.name): r for r in map(Requirement, lines)}
            if set(mine) != set(theirs):
                self.fail(f"[{extra}] names {sorted(mine)}, {whose} {sorted(theirs)}")
            for n in sorted(set(mine) & set(theirs)):
                if mine[n].specifier != theirs[n].specifier or set(mine[n].extras) != set(theirs[n].extras):
                    self.fail(f"[{extra}] has {mine[n]}, {whose} {theirs[n]}")

        compare("tts-0-8", groups["tts-0-8"], BytesHeaderParser().parsebytes(tts08_meta).get_all("Requires-Dist") or [],
                "the TTS 0.8 build")
        for n, r in groups["tts-0-8"].items():
            if not all(where(r.marker, "tts-0-8")):
                self.fail(f"[tts-0-8] {r} does not apply everywhere, as the TTS 0.8 build's requirement does")
        pyproject = self.tree.read("tts-2/python/pyproject.toml")
        try:
            import tomllib
        except ImportError:
            tomllib = None
        if pyproject is not None and tomllib is not None:
            deps = tomllib.loads(pyproject.decode("utf-8"))["project"].get("dependencies", [])
            compare("tts-2", groups["tts-2"], deps, "tts-2/python/pyproject.toml")
        else:
            self.warnings.append("[tts-2] was not compared with tts-2/python/pyproject.toml (needs Python 3.11+)")

    # ---------------------------------------------------------------- wheels

    def check_wheel(self, path: Path, w_file: str, version: str, base: str, licences: dict, tts2: dict):
        name = path.name
        _, _, py, abi, plats = name_tags(name)
        di = f"kitten_inference-{version}.dist-info"
        size = path.stat().st_size
        if size > MAX_BYTES:
            self.fail(f"{name}: {size} bytes, over PyPI's 100 MB limit")
        elif size > WARN_BYTES:
            self.warnings.append(f"{name}: {size} bytes, close to PyPI's 100 MB limit")
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            names = [i.filename for i in infos]
            data, attrs = {}, {}
            for i in infos:
                if i.filename.endswith("/"):
                    self.fail(f"{name}: directory entry {i.filename}")
                    continue
                data[i.filename] = z.read(i)  # reads and CRC-checks every member
                attrs[i.filename] = (i.external_attr, i.create_system, i.date_time)
        if len(set(names)) != len(names):
            self.fail(f"{name}: a member name is in the archive twice")
        tops = {n.split("/", 1)[0] for n in names if "/" in n}
        if {t for t in tops if t.endswith((".dist-info", ".data"))} != {di}:
            self.fail(f"{name}: expected exactly one dist-info folder, {di}")
        for n in names:
            parts = n.split("/")
            if (n.startswith("/") or "\\" in n or ".." in parts or "__pycache__" in parts
                    or n.endswith((".pyc", ".pyo"))):
                self.fail(f"{name}: member {n} must not be in a release wheel")
            length = len(n) if not n.startswith(di + "/") else len(n) - len(version) + len(base + ".post99")
            if length > MAX_PATH:
                self.fail(f"{name}: {n} is {length} characters (with a .post99 version), over {MAX_PATH}")
        if any(n.startswith(f"{di}/sboms/") for n in names):
            self.fail(f"{name}: carries an SBOM (it would name the TTS 0.8 build, not kitten-inference)")
        # RECORD
        record = data.get(f"{di}/RECORD")
        rows = record_rows(record) if record is not None else {}
        for n, content in data.items():
            got = rows.get(n)
            if n == f"{di}/RECORD":
                if got != [["", ""]]:
                    self.fail(f"{name}: RECORD's own row must be {n},,")
            elif got is None or len(got) != 1 or got[0] != [b64(content), str(len(content))]:
                self.fail(f"{name}: {n} does not match its RECORD row (or has none, or two)")
        for n in sorted(set(rows) - set(data)):
            self.fail(f"{name}: RECORD lists {n}, which the wheel lacks")
        # WHEEL
        wheel = data.get(f"{di}/WHEEL", b"").decode("utf-8").splitlines()
        tags = {line.split(":", 1)[1].strip() for line in wheel if line.startswith("Tag:")}
        if tags != {f"{py}-{abi}-{p}" for p in plats} or "Root-Is-Purelib: false" not in wheel:
            self.fail(f"{name}: WHEEL tags {sorted(tags)} differ from the file name, or Root-Is-Purelib is not false")
        # licences
        meta = data.get(f"{di}/METADATA", b"")
        listed = {line[len("License-File: "):] for line in meta.decode("utf-8").split("\n\n", 1)[0].splitlines()
                  if line.startswith("License-File: ")}
        present = {n[len(f"{di}/licenses/"):] for n in data if n.startswith(f"{di}/licenses/")}
        if listed != present:
            self.fail(f"{name}: License-File lines and licence files differ: {sorted(listed ^ present)[:5]}")
        if present != set(licences):
            self.fail(f"{name}: licence files differ from the source's: {sorted(present ^ set(licences))[:5]}")
        for rel, src in licences.items():
            if f"{di}/licenses/{rel}" in data and data[f"{di}/licenses/{rel}"] != self.tree.read(src):
                self.fail(f"{name}: licenses/{rel} differs from {src}")
        # the command
        parser = configparser.ConfigParser(delimiters=("=",), interpolation=None)
        parser.optionxform = str
        parser.read_string(data.get(f"{di}/entry_points.txt", b"").decode("utf-8"))
        if not parser.has_section("console_scripts") or \
                parser["console_scripts"].get("kitten-tts2") != "kitten_inference.tts2._cli:main":
            self.fail(f"{name}: no kitten-tts2 = kitten_inference.tts2._cli:main in entry_points.txt")
        # contents
        for n in ALWAYS:
            if n not in data:
                self.fail(f"{name}: {n} is missing")
        if not any(n.startswith("kitten_inference/tts08/_engine") for n in data):
            self.fail(f"{name}: no TTS 0.8 engine (kitten_inference/tts08/_engine*)")
        if data.get("kitten_inference/__init__.py") != self.tree.read("umbrella/kitten_inference/__init__.py"):
            self.fail(f"{name}: kitten_inference/__init__.py differs from umbrella/kitten_inference/__init__.py")
        have_tts2 = {n for n in data if n.startswith("kitten_inference/tts2/")}
        if have_tts2 != set(tts2):
            self.fail(f"{name}: kitten_inference/tts2/ differs from the source's files: "
                      f"{sorted(have_tts2 ^ set(tts2))[:5]}")
        for n in sorted(have_tts2 & set(tts2)):
            if data[n] != self.tree.read(tts2[n]):
                self.fail(f"{name}: {n} differs from {tts2[n]}")
        # engines
        parts = engines_expected(py, plats)
        if parts and len(plats) != 1:
            self.fail(f"{name}: a wheel with engines must have one platform tag, not {plats}")
        want = {f"kitten_inference/{r['path']}": r for r in self.lock["engine_files"] if r["part"] in parts}
        have = {n for n in data if n.startswith(("kitten_inference/_tts2_cuda/", "kitten_inference/_tts2_rocm/"))}
        if have != set(want):
            self.fail(f"{name}: engine files {sorted(have ^ set(want))[:4]} are missing or should not be there")
        for n in sorted(have & set(want)):
            r, content = want[n], data[n]
            mode = "%04o" % (attrs[n][0] >> 16 & 0o7777)
            if hashlib.sha256(content).hexdigest() != r["sha256"] or len(content) != r["size"] or mode != r["mode"]:
                self.fail(f"{name}: {n} differs from the lock's engine_files row ({r['part']})")
                continue
            executable = "/bin/" in n or n.startswith("kitten_inference/_tts2_rocm/lib/")
            if plats[0] != "win_amd64" and executable and mode != "0755":
                self.fail(f"{name}: {n} is {mode}, expected 0755")
            self.engine_rows_seen.add((r["part"], r["path"]))
            self.copied += 1
        # TTS 0.8: byte for byte against W, the overlay against the source
        overlay = set(self.lock["tts08_python_overlay"])
        w_rows, w_attrs = self.input_records.get(w_file, {}), self.input_attrs.get(w_file, {})
        w_dist = next((k.split("/", 1)[0] for k in w_rows if k.split("/", 1)[0].endswith(".dist-info")), "")
        want_tts08 = {k for k in w_rows if k.startswith(("kitten_inference/tts08/", "kitten_inference_tts08.libs/"))}
        have_tts08 = {n for n in data if n.startswith(("kitten_inference/tts08/", "kitten_inference_tts08.libs/"))}
        if have_tts08 != (want_tts08 | overlay):
            self.fail(f"{name}: TTS 0.8 files differ from {w_file} plus the overlay: "
                      f"{sorted(have_tts08 ^ (want_tts08 | overlay))[:4]}")
        for n in sorted(have_tts08):
            if n in overlay:
                if data[n] != self.tree.read(f"tts-0.8/{n}"):
                    self.fail(f"{name}: overlay file {n} differs from tts-0.8/{n}")
                continue
            if n not in w_rows or rows.get(n, [[]])[0] != w_rows[n] or attrs[n] != w_attrs.get(n):
                self.fail(f"{name}: {n} differs from {w_file} (RECORD row or zip attributes)")
                continue
            self.copied += 1
            if n.endswith((".py", ".json")):
                src = self.tree.read(f"tts-0.8/{n}")
                text = data[n].replace(b"\r\n", b"\n")
                if n.endswith("/tts08/__init__.py"):
                    text = DELVEWHEEL_BLOCK.sub(b"", text, count=1)
                if src is not None and text != src:
                    self.fail(f"{name}: {n} differs from tts-0.8/{n} (CRLF and the delvewheel block aside)")
        for p in self.tree.files("tts-0.8/kitten_inference/tts08"):
            if p[len("tts-0.8/"):] not in data:
                self.fail(f"{name}: {p[len('tts-0.8/'):]} (in the source) is missing")
        delve = f"{w_dist}/DELVEWHEEL"
        if delve in w_rows and (f"{di}/DELVEWHEEL" not in data or b64(data[f"{di}/DELVEWHEEL"]) != w_rows[delve][0]):
            self.fail(f"{name}: DELVEWHEEL differs from {w_file}'s")
        # NOTICE names every bundled TTS 0.8 library (the rule of tts-0.8.yml, at the new path)
        notice = data.get(f"{di}/licenses/tts-0.8/NOTICE", b"").decode("utf-8", "replace").lower()
        for n in data:
            if LIBRARY.fullmatch(n):
                lib = re.sub(r"-[0-9a-f]{8,}.*$", "", n.rsplit("/", 1)[-1])
                if lib.lower() not in notice:
                    self.fail(f"{name}: bundled {n} is not named in licenses/tts-0.8/NOTICE")
        # ELF files of manylinux_2_28_x86_64 wheels, one at a time
        if "manylinux_2_28_x86_64" in plats:
            for n in sorted(data):
                if data[n][:4] == b"\x7fELF":
                    digest = hashlib.sha256(data[n]).hexdigest()
                    if digest in self.elf_seen:
                        continue
                    self.elf_seen.add(digest)
                    with tempfile.TemporaryDirectory() as tmp:
                        f = Path(tmp) / n.rsplit("/", 1)[-1]
                        f.write_bytes(data[n])
                        problem = self.elf(f)
                    self.elf_files += 1
                    if problem:
                        self.fail(f"{name}: {n}: {problem}")
        self.summary.append((name, size, sha256_file(path)))
        return meta, data.get(f"{di}/entry_points.txt", b"")

    def run(self, wheels) -> list:
        self.load_inputs()
        self.check_sources()
        base = self.lock["version"]
        parsed = [(Path(w), name_tags(Path(w).name)) for w in wheels]
        bad = [w.name for w, t in parsed if t is None or t[0] != "kitten_inference"]
        if bad:
            self.fail(f"not kitten_inference wheel names: {bad}")
            return self.problems
        versions = {t[1] for _, t in parsed}
        if len(versions) != 1 or not re.fullmatch(re.escape(base) + r"(\.post\d+)?", next(iter(versions))):
            self.fail(f"the wheels have the versions {sorted(versions)}; expected one, {base} or {base}.postN")
            return self.problems
        version = versions.pop()
        # The expected file names, from the lock's TTS 0.8 inputs.
        expected = {}
        for row in self.lock["inputs"]:
            if row["part"] != "tts08":
                continue
            _, _, py, abi, plats = name_tags(row["file"])
            plat = "manylinux_2_28_x86_64" if engines_expected(py, plats) == {"cuda-linux", "rocm"} else ".".join(plats)
            expected[f"kitten_inference-{version}-{py}-{abi}-{plat}.whl"] = row["file"]
        given = {w.name for w, _ in parsed}
        for n in sorted(set(expected) - given):
            self.fail(f"missing wheel {n} (from {expected[n]})")
        for n in sorted(given - set(expected)):
            self.fail(f"unexpected wheel {n}")
        pairs = set(self.expected_pairs if self.expected_pairs is not None
                    else pypi_ready.COMPLETE["kitten-inference"])
        seen = {}
        for n in sorted(given):
            hit = pypi_ready.wheel_tags(n) & pairs
            if not hit:
                self.fail(f"{n} is not in pypi_ready.COMPLETE")
            for pair in sorted(hit):
                if pair in seen:
                    self.fail(f"two wheels for {pair[0]}-{pair[1]}: {seen[pair]} and {n}")
                seen[pair] = n
        missing = sorted(pairs - set(seen))
        if missing:
            self.fail(f"no wheel for {len(missing)} of the {len(pairs)} pairs in pypi_ready.COMPLETE: "
                      + ", ".join(f"{a}-{b}" for a, b in missing[:6]))
        tts08_requires = {tuple(sorted(BytesHeaderParser().parsebytes(m).get_all("Requires-Dist") or []))
                          for f, m in self.input_meta.items() if f in expected.values()}
        if len(tts08_requires) > 1:
            self.fail("the TTS 0.8 inputs do not all have the same Requires-Dist")
        licences, tts2 = self.licence_sources(), self.tts2_sources()
        metas, entries = set(), set()
        for w, _ in sorted(parsed):
            if w.name not in expected:
                continue
            meta, entry = self.check_wheel(w, expected[w.name], version, base, licences, tts2)
            metas.add(meta)
            entries.add(entry)
        if len(metas) > 1 or len(entries) > 1:
            self.fail("METADATA or entry_points.txt is not the same in every wheel")
        if metas:
            w_file = next(iter(expected.values()))
            if w_file in self.input_meta:
                self.check_metadata(next(iter(metas)), version, self.input_meta[w_file])
        rows = {(r["part"], r["path"]) for r in self.lock["engine_files"]}
        if self.engine_rows_seen != rows:
            self.fail(f"{len(self.engine_rows_seen & rows)}/{len(rows)} engine files of the lock are in the wheels")
        return self.problems


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("wheels", nargs="+", type=Path)
    a = p.parse_args(argv)
    lock = json.loads(a.lock.read_text(encoding="utf-8"))
    check = Check(lock, a.inputs, a.source)
    problems = check.run(a.wheels)
    for w in check.warnings:
        print(f"::warning::{w}")
    for problem in problems:
        print(f"::error::{problem}")
    rows = len(lock["engine_files"])
    print(f"{len(check.summary)} wheels: {check.copied} files copied unchanged, "
          f"{len(lock['tts08_python_overlay'])} Python files from {check.tree.commit[:12]}, "
          f"{len(check.engine_rows_seen)}/{rows} engine files match the lock, "
          f"{check.elf_files} ELF files within manylinux_2_28")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as out:
            out.write(f"### kitten-inference wheels ({len(check.summary)})\n\n"
                      "| wheel | MB | sha256 |\n|---|---:|---|\n")
            for name, size, sha in check.summary:
                out.write(f"| {name} | {size / 1e6:.1f} | `{sha}` |\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
