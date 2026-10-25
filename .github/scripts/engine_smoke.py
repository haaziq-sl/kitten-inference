#!/usr/bin/env python3
"""The real engine wheel through kitten_inference.tts2 on a runner without a GPU: discovery, library load and ABI
check, the bundled executables through the kitten-tts2 command, the GPU check's answer, and (on Windows) how the
engine and its text-data tool treat non-ASCII paths, which decides where the model folder may be.

    python .github/scripts/engine_smoke.py      (in an environment with the umbrella, kitten-inference-tts2 and
                                                  kitten-inference-tts2-cu12 installed)
"""
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from kitten_inference.tts2 import _abi, _backend, _gpu, _prepare

failed = []


def check(ok, what):
    print(f"{'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        failed.append(what)


backend = _backend.find()
print(f"backend {backend.name} {backend.distribution} {backend.version} abi {backend.abi} caps {backend.caps}")
lib = _abi.load(backend.lib)
info = _abi.text(lib.kt2n_build_info())
print("kt2n_build_info:", info)
check(int(lib.kt2n_abi_version()) == 2, "the library implements ABI 2 and every prototype resolves")
check(info == backend.build_info, "BUILD_INFO captured at build time matches the library")
check(("cap:voice.cloning" in info.split()) == ("voice_cloning" in backend.caps), "CAPS agrees with the library")

cli = shutil.which("kitten-tts2")
run = subprocess.run([cli, "--version"], capture_output=True, text=True)
print(run.stdout.strip(), run.stderr.strip()[-300:])
check(run.returncode == 0 and backend.version in run.stdout, "kitten-tts2 --version runs the bundled kitten-tts-native")

try:
    _gpu.check(backend.name, 0)
    check(False, "the GPU check reports this runner's missing NVIDIA driver")
except _gpu.GPUError as exc:
    print("GPU check:", exc)
    check("no NVIDIA driver found" in str(exc), "the GPU check reports this runner's missing NVIDIA driver")


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
        for folder, want in ((other, "cannot open input graph"), (root, "not a public Kitten compact graph")):
            graph = folder / "tagger.compact"
            graph.write_bytes(b"x" * 64)
            t = subprocess.run([tool, str(graph), str(folder / "tagger.fst")], capture_output=True, text=True)
            print(f"text-data tool on {graph}: exit {t.returncode}: {(t.stderr or t.stdout).strip()[-200:]}")
            check(t.returncode != 0 and want in (t.stderr + t.stdout), f"text-data tool in {folder.name!r}: {want}")
    else:
        check("non-ASCII" not in m_other, "non-ASCII model paths are accepted here")

print("FAILED: " + "; ".join(failed) if failed else "ALL OK")
sys.exit(1 if failed else 0)
