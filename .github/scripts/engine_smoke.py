#!/usr/bin/env python3
"""The engines of the installed kitten-inference through kitten_inference.tts2: discovery, the installed engine files
(with --lock, byte for byte against the tested ones in the lock's engine_files), library load and ABI check, the
bundled executables through the kitten-tts2 command, the GPU check's answer, the messages of a machine without a GPU
runtime, and (on Windows) how the engine and its text-data tool treat non-ASCII paths, which decides where the model
folder may be. It runs on CI runners (no GPU) and on GPU hosts.

    python .github/scripts/engine_smoke.py [--backend cuda|rocm] [--lock .github/release-inputs.json]

--backend: the engine to check, NVIDIA (cuda, the default) or AMD (rocm, Linux only). Run it with the Python of the
environment that has kitten-inference installed, from a folder outside the repository.
"""
import argparse
import ctypes
import hashlib
import importlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import kitten_inference
from kitten_inference.tts2 import _abi, _backend, _gpu, _prepare

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # paths below are not ASCII (Windows: cp1252)
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--backend", choices=("cuda", "rocm"), default="cuda")
p.add_argument("--lock", help="the release lock (.github/release-inputs.json): hash the engine files against it")
a = p.parse_args()
failed = []


def check(ok, what):
    print(f"{'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        failed.append(what)


def mode_bits(mode) -> int:
    text = str(mode)
    return int(mode) if isinstance(mode, int) else int(text[2:] if text.startswith("0o") else text, 8)


os.environ.pop(_backend.ENV, None)
try:
    backend = _backend.find(a.backend)
except ImportError as exc:
    sys.exit(f"engine_smoke: {exc}")
print(f"backend {backend.name}: {backend.label} {backend.version} abi {backend.abi} caps {backend.caps}")
runtime = _gpu.runtime(backend.name)  # loads it when found, as KittenTTS2 does before the engine library
print(f"GPU runtime: {runtime or 'not found'}")
package_dir = Path(importlib.import_module(backend.module).__file__).resolve().parents[1]  # .../kitten_inference

if a.lock:
    lock_path = Path(a.lock)
    if not lock_path.is_file() and not lock_path.is_absolute():
        lock_path = Path(__file__).resolve().parents[2] / a.lock
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    part = "rocm" if backend.name == "rocm" else ("cuda-windows" if sys.platform == "win32" else "cuda-linux")
    rows = [r for r in lock["engine_files"] if r["part"] == part]
    hashed = set()
    for row in rows:
        rel = row["path"][len("kitten_inference/"):] if row["path"].startswith("kitten_inference/") else row["path"]
        path = package_dir / rel
        data = path.read_bytes() if path.is_file() else None
        same = (data is not None and len(data) == int(row["size"])
                and hashlib.sha256(data).hexdigest() == row["sha256"])
        if same and os.name != "nt":  # pip keeps the executable bit of the wheel's zip entry
            same = bool(path.stat().st_mode & stat.S_IXUSR) == bool(mode_bits(row["mode"]) & stat.S_IXUSR)
        check(same, f"{rel} is the tested file ({row['sha256'][:12]}, {row['size']} B, mode {row['mode']})")
        hashed.add(path.resolve())
    check(bool(rows) and {backend.lib.resolve(), backend.cli.resolve(), backend.text_data_tool.resolve()} <= hashed,
          f"{len(rows)} {part} engine files in the lock, LIB, CLI and TEXT_DATA_TOOL among them")

# The CUDA library loads without a GPU or driver; the ROCm one needs the ROCm runtime (libamdhip64.so.7).
lib = None
if backend.name == "rocm" and runtime is None:
    try:
        _abi.load(backend.lib)
        check(False, "without ROCm, loading the AMD engine library fails naming libamdhip64.so.7")
    except _abi.LibraryError as exc:
        print("library load:", exc)
        check(_gpu.HIP_RUNTIME in str(exc),
              "without ROCm, loading the AMD engine library fails naming libamdhip64.so.7")
else:
    lib = _abi.load(backend.lib)
    info = _abi.text(lib.kt2n_build_info())
    print("kt2n_build_info:", info)
    check(int(lib.kt2n_abi_version()) == 2, "the library implements ABI 2 and every prototype resolves")
    check(info == backend.build_info, "BUILD_INFO captured at build time matches the library")
    check(("cap:voice.cloning" in info.split()) == ("voice_cloning" in backend.caps), "CAPS agrees with the library")

cli = shutil.which("kitten-tts2")
run = subprocess.run([cli, "--backend", backend.name, "--version"], capture_output=True, text=True)
print(run.stdout.strip(), run.stderr.strip()[-600:])
if backend.name == "rocm" and runtime is None:
    check(run.returncode == 1 and _gpu.HIP_RUNTIME in run.stderr and len(run.stderr.strip().splitlines()) == 1,
          "kitten-tts2 --backend rocm --version without ROCm: one line naming libamdhip64.so.7")
else:
    first = run.stdout.splitlines()[0] if run.stdout else ""
    check(run.returncode == 0 and backend.version in run.stdout,
          "kitten-tts2 --version runs the bundled kitten-tts-native")
    version = getattr(kitten_inference, "__version__", None)
    check(first == f"kitten-tts2 {version} ({backend.label} {backend.version})",
          "kitten-tts2 --version names the kitten-inference version and the engine")

try:
    count = _gpu.check(backend.name, 0)
    print("GPU check: passed, GPUs:", count)
    if runtime is None:  # only the AMD check gets here without its runtime: the engine library reports that
        check(backend.name == "rocm" and count == -1, "without ROCm the GPU check leaves the report to the engine")
    else:
        check(True, "the GPU check passes on this GPU host")
except _gpu.GPUError as exc:
    print("GPU check:", exc)
    if backend.name == "cuda" and runtime is None:
        check("no NVIDIA driver found" in str(exc), "the GPU check reports this runner's missing NVIDIA driver")
    else:
        check(False, f"the GPU check passes on this GPU host ({exc})")

# Linux x86-64 has both engines. On a runner without either GPU runtime, the NVIDIA one is picked, marked, and the
# message names both vendors.
both = all(_backend._import(k) is not None for k in _backend.MODULES)
if sys.platform == "linux" and both and _gpu.runtime("cuda") is None and _gpu.runtime("rocm") is None:
    picked = _backend.find()
    check((picked.name, picked.runtime_found) == ("cuda", False),
          "no GPU runtime: find() gives cuda, runtime_found=False")
    with tempfile.TemporaryDirectory() as tmp:
        run = subprocess.run([cli, "--assets", tmp, "--text", "x"], capture_output=True, text=True)
    print(run.stderr.strip()[-600:])
    check(run.returncode == 1 and len(run.stderr.strip().splitlines()) == 1 and "R570" in run.stderr
          and "ROCm 7" in run.stderr and "--backend rocm" in run.stderr,
          "no GPU runtime: kitten-tts2 says in one line what NVIDIA and AMD GPUs need")
    if backend.name == "cuda":  # the run with --backend rocm checks this above
        try:
            _abi.load(_backend.find("rocm").lib)
            text = ""
        except _abi.LibraryError as exc:
            text = str(exc)
        check(_gpu.HIP_RUNTIME in text, 'no ROCm: the library of find("rocm") fails naming libamdhip64.so.7')
        run = subprocess.run([cli, "--backend", "rocm", "--version"], capture_output=True, text=True)
        check(run.returncode == 1 and _gpu.HIP_RUNTIME in run.stderr and len(run.stderr.strip().splitlines()) == 1,
              "no ROCm: kitten-tts2 --backend rocm says in one line that libamdhip64.so.7 is missing")


def create(folder: Path):
    """kt2n_engine_create with every model path under folder (the files do not exist): its error message."""
    c = _abi.EngineConfig()
    c.struct_size = ctypes.sizeof(_abi.EngineConfig)
    c.abi_version = _abi.ABI_VERSION
    keep = [str(folder / n).encode("utf-8") for n in ("model.kt2n", "s3gen.kt2n", "voices.kt2n", "tokenizer.json",
                                                      "text")]
    c.lm_file, c.s3gen_file, c.voices_file, c.tokenizer_json, c.text_data_dir = keep
    c.flags = _abi.CFG_INFERENCE_ONLY
    handle, err = ctypes.c_void_p(), _abi.Error()
    status = lib.kt2n_engine_create(ctypes.byref(c), ctypes.byref(handle), ctypes.byref(err))
    if handle.value:
        lib.kt2n_engine_destroy(handle)
    return status, _abi.text(err.message)


if lib is not None:
    with tempfile.TemporaryDirectory() as tmp:
        plain = Path(tmp) / "models"
        other = Path(tmp) / "Łukasz José 田中"
        plain.mkdir()
        other.mkdir()
        s_plain, m_plain = create(plain)
        s_other, m_other = create(other)
        print(f"engine create, ASCII folder: status {s_plain}: {m_plain}")
        print(f"engine create, non-ASCII folder: status {s_other}: {m_other}")
        check(s_plain != 0 and "non-ASCII" not in m_plain, "an ASCII model path passes the engine's path check")
        windows_ansi = sys.platform == "win32" and not _prepare._narrow_paths_are_utf8()
        if windows_ansi:
            check("non-ASCII" in m_other, "Windows, ANSI code page: the engine refuses a non-ASCII model path")
            monkey = Path(tmp) / "home" / "Łukasz Wiśniewski"
            _prepare.DEFAULT_CACHE = monkey / ".cache" / "kitten-inference" / "tts2"
            os.environ["PROGRAMDATA"] = str(Path(tmp) / "ProgramData")
            root = _prepare.cache_root()
            root.mkdir(parents=True, exist_ok=True)
            s_root, m_root = create(root)
            print(f"engine create, the model folder cache_root() picks ({root}): status {s_root}: {m_root}")
            check("non-ASCII" not in m_root, "the folder cache_root() picks passes the engine's path check")
            tool = str(backend.text_data_tool)
            # The text-data tool carries a UTF-8 code page manifest (like kitten-tts-native), so it opens the file in
            # both folders and then rejects its content; "cannot open input graph" would mean the manifest is missing.
            want = "not a public Kitten compact graph"
            for folder in (other, root):
                graph = folder / "tagger.compact"
                graph.write_bytes(b"x" * 64)
                t = subprocess.run([tool, str(graph), str(folder / "tagger.fst")], capture_output=True, text=True)
                print(f"text-data tool on {graph}: exit {t.returncode}: {(t.stderr or t.stdout).strip()[-200:]}")
                check(t.returncode != 0 and want in (t.stderr + t.stdout),
                      f"text-data tool in {folder.name!r} opens the file: {want}")
        else:
            check("non-ASCII" not in m_other, "non-ASCII model paths are accepted here")

print("FAILED: " + "; ".join(failed) if failed else "ALL OK")
sys.exit(1 if failed else 0)
