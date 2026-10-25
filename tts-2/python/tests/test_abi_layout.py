"""The ctypes structs match include/kt2n/engine.h: every size, field offset and named constant.

A small C program that includes the header prints sizeof/offsetof of every field ctypes declares, and the values
of the constants; this test compares them with ctypes. It needs a C compiler (cc, gcc, clang, or cl on Windows) and
the repository's header ($KT2N_ENGINE_H, default ../include/kt2n/engine.h next to this package's source).
"""
import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kitten_inference.tts2 import _abi

HEADER = Path(os.environ.get("KT2N_ENGINE_H", Path(__file__).resolve().parents[2] / "include" / "kt2n" / "engine.h"))
CONSTANTS = {
    "KT2N_ABI_VERSION": "ABI_VERSION", "KT2N_INVALID_ID": "INVALID_ID",
    "KT2N_OK": "OK", "KT2N_ERR_INVALID_ARGUMENT": "ERR_INVALID_ARGUMENT",
    "KT2N_ERR_INVALID_HANDLE": "ERR_INVALID_HANDLE", "KT2N_ERR_ABI_MISMATCH": "ERR_ABI_MISMATCH",
    "KT2N_ERR_AGAIN": "ERR_AGAIN", "KT2N_ERR_CANCELLED": "ERR_CANCELLED", "KT2N_ERR_NOT_READY": "ERR_NOT_READY",
    "KT2N_ERR_UNSUPPORTED": "ERR_UNSUPPORTED", "KT2N_ERR_TOO_SMALL": "ERR_TOO_SMALL",
    "KT2N_ERR_INVALID_REQUEST": "ERR_INVALID_REQUEST", "KT2N_ERR_INVALID_INPUT": "ERR_INVALID_INPUT",
    "KT2N_ERR_INVALID_VOICE": "ERR_INVALID_VOICE", "KT2N_ERR_REQUEST_TOO_LARGE": "ERR_REQUEST_TOO_LARGE",
    "KT2N_ERR_VOICE_NOT_FOUND": "ERR_VOICE_NOT_FOUND", "KT2N_ERR_GPU_MEMORY_LIMIT": "ERR_GPU_MEMORY_LIMIT",
    "KT2N_ERR_INVALID_AUDIO": "ERR_INVALID_AUDIO", "KT2N_ERR_INVALID_AUDIO_DURATION": "ERR_INVALID_AUDIO_DURATION",
    "KT2N_ERR_INVALID_AUDIO_SIGNAL": "ERR_INVALID_AUDIO_SIGNAL",
    "KT2N_ERR_VOICE_ARTIFACT_VERSION_MISMATCH": "ERR_VOICE_ARTIFACT_VERSION_MISMATCH",
    "KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE": "ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE",
    "KT2N_ERR_INVALID_REFERENCE_TRANSCRIPT": "ERR_INVALID_REFERENCE_TRANSCRIPT",
    "KT2N_ERR_DEADLINE_EXCEEDED": "ERR_DEADLINE_EXCEEDED", "KT2N_ERR_INTERNAL": "ERR_INTERNAL",
    "KT2N_CFG_INFERENCE_ONLY": "CFG_INFERENCE_ONLY",
    "KT2N_VOICE_CACHE_ALL": "VOICE_CACHE_ALL", "KT2N_VOICE_CACHE_LRU": "VOICE_CACHE_LRU",
    "KT2N_VOICE_CACHE_NONE": "VOICE_CACHE_NONE",
    "KT2N_MEMORY_PROFILE_DEFAULT": "MEMORY_PROFILE_DEFAULT", "KT2N_MEMORY_PROFILE_LOW": "MEMORY_PROFILE_LOW",
    "KT2N_MEMORY_PROFILE_AUTO": "MEMORY_PROFILE_AUTO",
    "KT2N_VOICE_BUILTIN": "VOICE_BUILTIN", "KT2N_VOICE_INLINE": "VOICE_INLINE", "KT2N_VOICE_SAVED": "VOICE_SAVED",
    "KT2N_VB_INLINE": "VB_INLINE", "KT2N_VB_SAVED": "VB_SAVED", "KT2N_VB_DECODED_PCM": "VB_DECODED_PCM",
    "KT2N_FORMAT_PCM": "FORMAT_PCM", "KT2N_FORMAT_WAV": "FORMAT_WAV", "KT2N_FORMAT_F32": "FORMAT_F32",
    "KT2N_OUTPUT_STREAM": "OUTPUT_STREAM", "KT2N_OUTPUT_STREAM_FILE": "OUTPUT_STREAM_FILE",
    "KT2N_OUTPUT_COMPLETE": "OUTPUT_COMPLETE", "KT2N_SPEECH_MODE_DEFAULT": "SPEECH_MODE_DEFAULT",
    "KT2N_SAMPLER_UNSET": "SAMPLER_UNSET", "KT2N_SAMPLER_STABLE": "SAMPLER_STABLE",
    "KT2N_SAMPLER_EXPRESSIVE": "SAMPLER_EXPRESSIVE",
    "KT2N_SAMPLER_HAS_MODE": "SAMPLER_HAS_MODE", "KT2N_SAMPLER_HAS_TEMPERATURE": "SAMPLER_HAS_TEMPERATURE",
    "KT2N_SAMPLER_HAS_TOP_P": "SAMPLER_HAS_TOP_P", "KT2N_SAMPLER_HAS_TOP_K": "SAMPLER_HAS_TOP_K",
    "KT2N_SAMPLER_HAS_MIN_P": "SAMPLER_HAS_MIN_P", "KT2N_SAMPLER_HAS_MAX_NEW_TOKENS": "SAMPLER_HAS_MAX_NEW_TOKENS",
    "KT2N_REQ_SEED": "REQ_SEED", "KT2N_REQ_NORMALIZE_OFF": "REQ_NORMALIZE_OFF",
    "KT2N_CANCEL_REQUEST": "CANCEL_REQUEST",
    "KT2N_EVENT_AUDIO": "EVENT_AUDIO", "KT2N_EVENT_TIMELINE": "EVENT_TIMELINE", "KT2N_EVENT_METRICS": "EVENT_METRICS",
    "KT2N_EVENT_PHRASE_DONE": "EVENT_PHRASE_DONE", "KT2N_EVENT_DONE": "EVENT_DONE",
    "KT2N_EVENT_VOICE_BUILT": "EVENT_VOICE_BUILT", "KT2N_EVENT_HEADERS": "EVENT_HEADERS",
}


def _compiler():
    for name in (os.environ.get("CC"), "cc", "gcc", "clang", "cl"):
        if name and shutil.which(name):
            return name
    return None


def _program() -> str:
    lines = ["#include <stddef.h>", "#include <stdio.h>", '#include "kt2n/engine.h"', "int main(void) {"]
    for struct in _abi.STRUCTS:
        c = struct._c_name_
        lines.append(f'  printf("size {c} %zu\\n", sizeof({c}));')
        for field, *_ in struct._fields_:
            lines.append(f'  printf("offset {c}.{field} %zu\\n", offsetof({c}, {field}));')
    for name in CONSTANTS:
        lines.append(f'  printf("const {name} %lld\\n", (long long)({name}));')
    lines.append(f'  printf("string KT2N_CAP_VOICE_CLONING %s\\n", KT2N_CAP_VOICE_CLONING);')
    lines.append("  return 0;\n}")
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def layout(tmp_path_factory):
    if not HEADER.is_file():
        pytest.skip(f"engine.h not found at {HEADER} (set KT2N_ENGINE_H)")
    cc = _compiler()
    if cc is None:
        if os.environ.get("KT2N_REQUIRE_ABI_TEST") == "1":
            pytest.fail("no C compiler found")
        pytest.skip("no C compiler (cc, gcc, clang or cl)")
    work = tmp_path_factory.mktemp("abi")
    source = work / "abi_layout.c"
    source.write_text(_program())
    exe = work / ("abi_layout.exe" if sys.platform == "win32" else "abi_layout")
    include = str(HEADER.parent.parent)
    if Path(cc).name.lower().startswith("cl"):
        cmd = [cc, "/nologo", f"/I{include}", str(source), f"/Fe{exe}", f"/Fo{work}\\"]
    else:
        cmd = [cc, "-std=c11", f"-I{include}", str(source), "-o", str(exe)]
    subprocess.run(cmd, check=True, cwd=work)
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    table = {}
    for line in out.splitlines():
        kind, key, value = line.split(" ", 2)
        table[(kind, key)] = value
    return table


@pytest.mark.parametrize("struct", _abi.STRUCTS, ids=lambda s: s._c_name_)
def test_struct_layout(layout, struct):
    c = struct._c_name_
    assert ctypes.sizeof(struct) == int(layout[("size", c)]), f"sizeof({c})"
    for field, *_ in struct._fields_:
        assert getattr(struct, field).offset == int(layout[("offset", f"{c}.{field}")]), f"offsetof({c}, {field})"


def test_constants(layout):
    for c_name, py_name in CONSTANTS.items():
        assert getattr(_abi, py_name) == int(layout[("const", c_name)]), c_name
    assert _abi.CAP_VOICE_CLONING == layout[("string", "KT2N_CAP_VOICE_CLONING")]


def test_every_header_function_is_declared():
    if not HEADER.is_file():
        pytest.skip(f"engine.h not found at {HEADER}")
    import re
    declared = set(re.findall(r"KT2N_API\s+[^;(]*?\b(kt2n_\w+)\s*\(", HEADER.read_text()))
    assert declared, "no KT2N_API functions found in the header"
    assert declared == set(_abi.PROTOTYPES), (declared ^ set(_abi.PROTOTYPES))


def test_config_struct_covers_every_appended_field():
    # The engine reads the voice cache / memory block and custom_voice_max_prefix_ids only when struct_size covers
    # them, so the binding must declare the whole struct.
    names = [f for f, *_ in _abi.EngineConfig._fields_]
    assert names[-1] == "custom_voice_max_prefix_ids"
    assert {"voice_cache_mode", "memory_profile", "min_free_device_bytes"} <= set(names)
