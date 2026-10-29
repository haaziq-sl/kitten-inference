"""Small synthetic inputs for test_assemble_wheels.py, with the layout of the real ones and no real engines: TTS 0.8
wheels for seven platforms (CRLF RECORD on Linux and macOS, the Windows attributes, CRLF files and delvewheel patch,
the auditwheel SBOM), the CUDA and ROCm payload wheels (the ROCm one with directory entries and a licenses/ folder),
the umbrella and tts2 wheels, a git repository standing in for the source (the tested commit, then the release
commit with the new overlay files), the CI artifact zips, and the lock that lock_inputs.make_lock writes from them.
No network.

    world = stub_inputs.World(tmp_path)
    world.build("out")                      # assemble_wheels.build into tmp_path/out
"""
import base64
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / ".github" / "scripts"
VERSION = "0.2.0"
DATE = (2026, 10, 26, 13, 38, 22)
C2 = ("python_version >= '3.9' and ((sys_platform == 'linux' and platform_machine == 'x86_64') or "
      "(sys_platform == 'win32' and platform_machine == 'AMD64'))")
try:
    import tomllib
except ImportError:
    import tomli as tomllib
ROOT_PROJECT = (SCRIPTS.parents[1] / 'pyproject.toml').read_text()
UMBRELLA_REQUIRES = tomllib.loads(ROOT_PROJECT)['project']['dependencies']
TTS08_REQUIRES = TTS2_REQUIRES = UMBRELLA_REQUIRES
OVERLAY = ["kitten_inference/tts08/_deps.py", "kitten_inference/tts08/get_model.py",
           "kitten_inference/tts08/cpp_model.py", "kitten_inference/tts08/_espeak_path.py"]
TESTED = {  # the tested build's overlay files (the release commit has new ones, and _deps.py)
    "kitten_inference/tts08/get_model.py": b"from huggingface_hub import hf_hub_download\n",
    "kitten_inference/tts08/cpp_model.py": b"MESSAGE = 'reinstall kitten-inference-tts08'\n",
    "kitten_inference/tts08/_espeak_path.py": b"WHO = 'kitten-inference-tts08'\n",
}
RELEASE = {
    "kitten_inference/tts08/_deps.py": b"REQUIRED = ('espeakng_loader', 'phonemizer')\n",
    "kitten_inference/tts08/get_model.py": b"from ._deps import require as _require; _require()\n",
    "kitten_inference/tts08/cpp_model.py": b"MESSAGE = 'reinstall Kitten TTS 0.8'\n",
    "kitten_inference/tts08/_espeak_path.py": b"WHO = 'Kitten TTS 0.8'\n",
}
SOURCE = {
    "pyproject.toml": ROOT_PROJECT.encode(),
    "LICENSE": b"Apache License 2.0 (stand-in)\n",
    "NOTICE": b"kitten-inference notices (stand-in)\n",
    "engines/tts08/NOTICE": b"TTS 0.8 bundles libgomp (GCC runtime), msvcp140 (MSVC) and libc++_shared (LLVM).\n",
    "engines/tts08/licenses/GPL-3.0.txt": b"GPL 3 (stand-in)\n",
    "engines/tts08/licenses/pybind11.txt": b"pybind11 (stand-in)\n",
    "engines/tts2/NOTICE": b"TTS 2 notices (stand-in)\n",
    "engines/tts2/licenses/Apache-2.0.txt": b"Apache 2.0 (stand-in)\n",
    "engines/tts2/licenses/rocm/HIP.txt": b"HIP (stand-in)\n",
    "engines/tts2/src/post/atempo.cc": b"// atempo (LGPL) stand-in\n",
    "engines/tts2/src/engine.cc": b"// engine\n",
    "engines/tts2/include/kt2n/post/atempo.h": b"// atempo header\n",
    "engines/tts2/cli/main.cc": b"// cli\n",
    "engines/tts2/CMakeLists.txt": b"project(kt2n)\n",
    "engines/tts2/cmake/flags.cmake": b"# flags\n",
    "engines/tts2/CMakePresets.json": b"{}\n",
    "engines/tts08/engine/engine.cpp": b"// tts08 engine\n",
    "src/kitten_inference/tts08/__init__.py": b'"""Kitten TTS 0.8."""\nfrom .preprocess import x\n',
    "src/kitten_inference/tts08/preprocess.py": b"x = 1\n",
    "src/kitten_inference/tts08/model_defs/a.json": b'{"a": 1}\n',
    "src/kitten_inference/__init__.py": b'__version__ = "0.2.0"\n',
    "src/kitten_inference/tts2/__init__.py": b'"""Kitten TTS 2."""\n',
    "src/kitten_inference/tts2/_abi.py": b"ABI_VERSION = 2\n",
    "src/kitten_inference/tts2/_cli.py": b"def main():\n    return 0\n",
    "engines/tts2/tools/kt2n_assets/__init__.py": b"",
    "engines/tts2/tools/kt2n_assets/prepare.py": b"# prepare\n",
    "engines/tts2/python/pyproject.toml": (b'[project]\nname = "kitten-inference-tts2"\nversion = "0.2.0"\ndependencies = '
                                    b'["numpy>=1.24", "huggingface-hub>=0.25,<2", "tokenizers>=0.22,<0.23", '
                                    b'"filelock>=3.12"]\n'),
}
DELVEWHEEL = (b"\n\n# start delvewheel patch\ndef _delvewheel_patch_1_13_2():\n    import os\n"
              b"_delvewheel_patch_1_13_2()\ndel _delvewheel_patch_1_13_2\n# end delvewheel patch\n\n")
# TTS 0.8 inputs: (python, platform text, artifact)
TTS08 = [
    ("cp38", "manylinux_2_27_x86_64.manylinux_2_28_x86_64", "tts08-wheels-linux-x86_64-manylinux-2-28"),
    ("cp312", "manylinux_2_27_x86_64.manylinux_2_28_x86_64", "tts08-wheels-linux-x86_64-manylinux-2-28"),
    ("cp312", "manylinux2014_aarch64.manylinux_2_17_aarch64", "tts08-wheels-linux-aarch64-manylinux2014"),
    ("cp312", "macosx_11_0_arm64", "tts08-wheels-macos-arm64"),
    ("cp38", "win_amd64", "tts08-wheels-windows-x86_64"),
    ("cp312", "win_amd64", "tts08-wheels-windows-x86_64"),
    ("cp313", "android_24_arm64_v8a", "tts08-wheels-android-arm64"),
]
PAYLOADS = {  # part -> (file, artifact)
    "cuda-linux": ("kitten_inference_tts2_cu12-0.2.0-py3-none-manylinux_2_28_x86_64.whl", "tts2-cu12-linux"),
    "cuda-windows": ("kitten_inference_tts2_cu12-0.2.0-py3-none-win_amd64.whl", "tts2-cu12-windows"),
    "rocm": ("kitten_inference_tts2_rocm-0.2.0-py3-none-manylinux_2_28_x86_64.whl",
             "kitten-inference-tts2-rocm-manylinux_2_28_x86_64"),
}
RUN_OF_ARTIFACT = {a: "tts08" for _, _, a in TTS08}
RUN_OF_ARTIFACT.update({"tts2-cu12-linux": "cuda", "tts2-cu12-windows": "cuda",
                        "kitten-inference-tts2-rocm-manylinux_2_28_x86_64": "rocm"})


class M:
    """A member to write: name, bytes (None for a directory entry), mode, creator system (3 Unix, 0 Windows), date."""

    def __init__(self, name, data, mode=0o644, system=3, date=DATE):
        self.name, self.data, self.mode, self.system, self.date = name, data, mode, system, date


def b64(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def write_wheel(path: Path, members, dist_info: str, crlf=False, record_members=None):
    """Writes members plus a RECORD (of record_members when given, to make a RECORD that does not match)."""
    rows = [f"{m.name},{b64(m.data)},{len(m.data)}" for m in (record_members or members) if m.data is not None]
    end = "\r\n" if crlf else "\n"
    record = end.join(rows + [f"{dist_info}/RECORD,,"]) + end
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        for m in members + [M(f"{dist_info}/RECORD", record.encode())]:
            zi = zipfile.ZipInfo(m.name, m.date)
            zi.create_system = m.system
            if m.data is None:
                zi.external_attr = (0o40755 << 16) | 0x10
                z.writestr(zi, b"")
            else:
                zi.external_attr = (m.mode if m.system == 0 else 0o100000 | m.mode) << 16
                zi.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(zi, m.data)
    return path


def git(repo: Path, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=stub", "-c", "user.email=stub@example.com",
                    "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false", *args],
                   check=True, capture_output=True)
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def scripts():
    """The three scripts as modules (lock_inputs and check_assembled import their neighbours by name)."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import assemble_wheels
    import check_assembled
    import lock_inputs
    return assemble_wheels, check_assembled, lock_inputs


class World:
    def __init__(self, root: Path, umbrella_version: str = VERSION, umbrella_requires=None, abi: int = 2,
                 extra_source=None):
        self.root = Path(root)
        self.inputs, self.pure, self.source = self.root / "in", self.root / "dist", self.root / "src"
        self.lock_path = self.root / "release-inputs.json"
        self.specs = {}  # file -> (folder, members, dist_info, crlf)
        self.source.mkdir(parents=True)
        git(self.source, "init", "-q")
        files = dict(SOURCE)
        files.update(extra_source or {})
        files.update({f"src/{p}": data for p, data in TESTED.items()})
        self.tested_commit = self.commit(files, "tested")
        self.release_commit = self.commit({f"src/{p}": data for p, data in RELEASE.items()}, "release")
        self.licences = {p: d for p, d in files.items() if p.startswith(("engines/tts08/licenses/", "engines/tts2/licenses/"))}
        for py, plat, _ in TTS08:
            self.add(self.inputs, *self.tts08(py, plat))
        for part in PAYLOADS:
            self.add(self.inputs, *self.payload(part, abi))
        self.add(self.pure, *self.umbrella(umbrella_version, umbrella_requires))
        self.add(self.pure, *self.tts2())
        self.relock()

    # ---------------------------------------------------------------- the source repository

    def commit(self, files: dict, message: str) -> str:
        for path, data in files.items():
            p = self.source / path
            if data is None:
                p.unlink()
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        git(self.source, "add", "-A")
        return git(self.source, "commit", "-q", "-m", message)

    # ---------------------------------------------------------------- wheels

    def add(self, folder: Path, file: str, members, dist_info: str, crlf: bool = False):
        self.specs[file] = (folder, members, dist_info, crlf)
        write_wheel(folder / file, members, dist_info, crlf)

    def rewrite(self, file: str, members, stale_record: bool = False, relock: bool = True):
        """Writes `file` again with other members; stale_record keeps the old RECORD."""
        folder, old, dist_info, crlf = self.specs[file]
        write_wheel(folder / file, members, dist_info, crlf, record_members=old if stale_record else None)
        if not stale_record:
            self.specs[file] = (folder, members, dist_info, crlf)
        if relock:
            self.relock()

    def members(self, file: str):
        return list(self.specs[file][1])

    def tts08(self, py: str, plat: str):
        windows, linux = plat.startswith("win"), "linux" in plat
        system = 0 if windows else 3
        crlf = (lambda b: b.replace(b"\n", b"\r\n")) if windows else (lambda b: b)
        info = f"kitten_inference_tts08-{VERSION}.dist-info"
        init = SOURCE["src/kitten_inference/tts08/__init__.py"]
        if windows:  # where delvewheel puts its patch: after the docstring's line
            first, rest = init.split(b"\n", 1)
            init = first + b"\n" + DELVEWHEEL + rest
        wmode = 0o100666 if windows else 0
        mode = (lambda m: wmode) if windows else (lambda m: m)
        ext = {"win_amd64": ".pyd", "macosx_11_0_arm64": ".so"}.get(plat, ".so")
        engine = (b"MZ" if windows else b"\xcf\xfa\xed\xfe" if "macosx" in plat else b"\x7fELF") + py.encode() * 8
        members = [] if windows or "android" in plat else [M("kitten_inference/", None),
                                                           M("kitten_inference/tts08/", None)]
        members += [
            M("kitten_inference/tts08/__init__.py", crlf(init), mode(0o644), system),
            M("kitten_inference/tts08/preprocess.py", crlf(SOURCE["src/kitten_inference/tts08/preprocess.py"]),
              mode(0o644), system),
            M("kitten_inference/tts08/model_defs/a.json",
              crlf(SOURCE["src/kitten_inference/tts08/model_defs/a.json"]), mode(0o644), system),
            M("kitten_inference/tts08/_cpu_requirements.txt", b"avx2 fma\n", mode(0o644), system),
            M(f"kitten_inference/tts08/_engine.{py}-{plat.split('.')[-1]}{ext}", engine,
              mode(0o644 if "android" in plat else 0o755), system),
        ]
        members += [M(p, crlf(data), mode(0o644), system) for p, data in sorted(TESTED.items())]
        if linux:
            members.append(M("kitten_inference_tts08.libs/libgomp-e985bcbb.so.1.0.0", b"\x7fELFgomp" + py.encode(),
                             0o755))
        if "android" in plat:
            members.append(M("kitten_inference_tts08.libs/libc++_shared-d523468d.so", b"\x7fELFc++", 0o644))
        if windows:
            members += [M("kitten_inference_tts08.libs/msvcp140-7c26614e1d733892c2deac7e245ce115.dll", b"MZmsvcp",
                          wmode, 0),
                        M("kitten_inference/tts08/_msvc/vcruntime140_1.dll", b"MZvcruntime", 0o600, 0,
                          (2026, 1, 1, 0, 0, 0))]
        tags = [f"{py}-{py}-{p}" for p in plat.split(".")]
        members += [
            M(f"{info}/METADATA", ("Metadata-Version: 2.4\nName: kitten-inference-tts08\nVersion: 0.2.0\n"
                                   + "".join(f"Requires-Dist: {r}\n" for r in TTS08_REQUIRES)
                                   + "\n# Kitten TTS 0.8\n").encode(), mode(0o664), system),
            M(f"{info}/WHEEL", ("Wheel-Version: 1.0\nGenerator: stub\nRoot-Is-Purelib: false\n"
                                + "".join(f"Tag: {t}\n" for t in tags)).encode(), mode(0o664), system),
            M(f"{info}/licenses/LICENSE", crlf(SOURCE["LICENSE"]), mode(0o664), system),
            M(f"{info}/licenses/NOTICE", crlf(SOURCE["engines/tts08/NOTICE"]), mode(0o664), system),
        ]
        if linux:
            members.append(M(f"{info}/sboms/auditwheel.cdx.json", b'{"bomFormat": "CycloneDX"}'))
        if windows:
            members.append(M(f"{info}/DELVEWHEEL", b"Version: 1.13.2\n", wmode, 0))
        file = f"kitten_inference_tts08-{VERSION}-{py}-{py}-{plat}.whl"
        return file, members, info, not windows and "android" not in plat

    def payload(self, part: str, abi: int):
        file, _ = PAYLOADS[part]
        windows, rocm = part == "cuda-windows", part == "rocm"
        backend = "rocm" if rocm else "cuda"
        d = f"kitten_inference/_tts2_{backend}"
        info = f"kitten_inference_tts2_{'rocm' if rocm else 'cu12'}-{VERSION}.dist-info"
        system = 0 if windows else 3
        init = f'BACKEND = "{backend}"\nABI = {abi}\nVERSION = \'0.2.0\'\n'.encode()
        exe, lib = (".exe", "kt2n_engine.dll") if windows else ("", "libkt2n_engine.so")
        magic = b"MZ" if windows else b"\x7fELF"
        members = [M("kitten_inference/", None), M(f"{d}/", None), M(f"{d}/bin/", None)] if rocm else []
        members += [
            M(f"{d}/__init__.py", init, 0o644, system),
            M(f"{d}/bin/kitten-tts-native{exe}", magic + b"cli" + backend.encode(), 0o644 if windows else 0o755,
              system),
            M(f"{d}/bin/kt2n_prepare_text_data{exe}", magic + b"tool" + backend.encode(),
              0o644 if windows else 0o755, system),
            M(f"{d}/lib/{lib}", magic + b"engine" + backend.encode() * 50, 0o755 if rocm else 0o644, system),
            M(f"{info}/METADATA", b"Metadata-Version: 2.1\nName: kitten-inference-tts2-"
              + (b"rocm" if rocm else b"cu12") + b"\nVersion: 0.2.0\n", 0o644, system),
            M(f"{info}/WHEEL", f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {file[:-4].split('-', 2)[2]}\n"
              .encode(), 0o644, system),
        ]
        if rocm:
            members += [M(f"{d}/licenses/", None), M(f"{d}/licenses/NOTICE", b"old NOTICE naming build names\n"),
                        M(f"{d}/licenses/rocm/HIP.txt", SOURCE["engines/tts2/licenses/rocm/HIP.txt"])]
        else:
            members.append(M(f"{info}/top_level.txt", b"kitten_inference\n", 0o644, system))
        return file, members, info, rocm

    def umbrella(self, version: str, requires=None):
        info = f"kitten_inference-{version}.dist-info"
        meta = ("Metadata-Version: 2.4\nName: kitten-inference\nVersion: " + version + "\nSummary: Kitten TTS\n"
                "License-Expression: Apache-2.0\nLicense-File: LICENSE\nRequires-Python: >=3.8\n"
                ""
                + "".join(f"Requires-Dist: {r}\n" for r in (UMBRELLA_REQUIRES if requires is None else requires))
                + "Description-Content-Type: text/markdown\n\n# kitten-inference\n")
        members = [
            M("kitten_inference/__init__.py", SOURCE["src/kitten_inference/__init__.py"]),
            M(f"{info}/METADATA", meta.encode()),
            M(f"{info}/WHEEL", b"Wheel-Version: 1.0\nGenerator: hatchling 1.27.0\nRoot-Is-Purelib: true\n"
              b"Tag: py3-none-any\n"),
            M(f"{info}/entry_points.txt", b"[console_scripts]\nkitten-tts = kitten_inference._cli:main\n"),
            M(f"{info}/licenses/LICENSE", SOURCE["LICENSE"]),
        ]
        return f"kitten_inference-{version}-py3-none-any.whl", members, info

    def tts2(self):
        info = f"kitten_inference_tts2-{VERSION}.dist-info"
        files = {}
        for path, data in SOURCE.items():
            if path.startswith("src/kitten_inference/tts2/"):
                files["kitten_inference/tts2/" + path[len("src/kitten_inference/tts2/"):]] = data
            elif path.startswith("engines/tts2/tools/kt2n_assets/"):
                files["kitten_inference/tts2/_assets/" + path[len("engines/tts2/tools/kt2n_assets/"):]] = data
        files.update({"kitten_inference/tts2/licenses/" + p[len("engines/tts2/licenses/"):]: d
                      for p, d in self.licences.items() if p.startswith("engines/tts2/licenses/")})
        files["kitten_inference/tts2/NOTICE"] = SOURCE["engines/tts2/NOTICE"]
        members = [M(p, d, date=(2020, 2, 2, 0, 0, 0)) for p, d in sorted(files.items())]
        members += [
            M(f"{info}/METADATA", ("Metadata-Version: 2.4\nName: kitten-inference-tts2\nVersion: 0.2.0\n"
                                   "Requires-Python: >=3.9\n" + "".join(f"Requires-Dist: {r}\n" for r in TTS2_REQUIRES)
                                   ).encode()),
            M(f"{info}/WHEEL", b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"),
            M(f"{info}/entry_points.txt", b"[console_scripts]\nkitten-tts = kitten_inference._cli:main\n"),
        ]
        return f"kitten_inference_tts2-{VERSION}-py3-none-any.whl", members, info

    # ---------------------------------------------------------------- artifacts and the lock

    def artifacts(self):
        """One zip per CI artifact (as GitHub stores it), with the digest GitHub records."""
        folder = self.root / "artifacts"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir()
        groups = {}
        for py, plat, artifact in TTS08:
            groups.setdefault(artifact, []).append(f"kitten_inference_tts08-{VERSION}-{py}-{py}-{plat}.whl")
        for file, artifact in PAYLOADS.values():
            groups[artifact] = [file]
        out = []
        for n, (artifact, files) in enumerate(sorted(groups.items()), 1):
            path = folder / f"{artifact}.zip"
            with zipfile.ZipFile(path, "w") as z:
                for f in files:
                    z.write(self.inputs / f, f)
                if artifact == "tts2-cu12-linux":
                    z.writestr("auditwheel-show.txt", "not a wheel\n")
            digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            out.append({"run": RUN_OF_ARTIFACT[artifact], "name": artifact, "id": 1000 + n, "digest": digest,
                        "zip": path})
        return out

    def runs(self):
        return {key: {"id": n, "attempt": 1, "commit": self.tested_commit, "workflow": workflow}
                for n, (key, workflow) in enumerate((("tts08", "tts-0.8.yml"), ("cuda", "tts-2-engine.yml"),
                                                     ("rocm", "tts-2-rocm.yml")), 1)}

    def relock(self):
        aw, _, li = scripts()
        lock = li.make_lock("example/kitten-inference", self.runs(), self.artifacts(), self.inputs,
                            aw.Source(self.source))
        self.save_lock(lock)
        return lock

    def lock(self) -> dict:
        return json.loads(self.lock_path.read_text(encoding="utf-8"))

    def save_lock(self, lock: dict):
        _, _, li = scripts()
        self.lock_path.write_text(li.dump(lock), encoding="utf-8")

    def pairs(self):
        """The (python, platform) pairs of the outputs, as pypi_ready.COMPLETE lists them for the real set."""
        out = set()
        for py, plat, _ in TTS08:
            out.add((py, plat.split(".")[-1]))
        return out

    def build(self, out: str = "out", tag_run: bool = False) -> Path:
        aw, _, _ = scripts()
        aw.build(self.lock_path, self.inputs, self.pure, self.source, self.root / out, tag_run=tag_run)
        return self.root / out

    def check(self, out: Path, elf=None):
        """check_assembled on the wheels in out: (problems, the Check)."""
        _, ca, _ = scripts()
        check = ca.Check(self.lock(), self.inputs, self.source, expected_pairs=self.pairs(),
                         elf=elf if elf is not None else (lambda path: None))
        return check.run(sorted(Path(out).glob("*.whl"))), check
