"""Finds the engine to use: ``kitten_inference._tts2_cuda`` (NVIDIA) or ``kitten_inference._tts2_rocm`` (AMD).

The Linux x86-64 install has both, the Windows x64 one the NVIDIA engine only, other platforms none. This module
imports no third-party package, and it runs on Python 3.8 too, to say what that Python lacks.
"""
from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from . import _abi, _gpu

ENV = "KITTEN_TTS2_BACKEND"
MODULES = {"cuda": "kitten_inference._tts2_cuda", "rocm": "kitten_inference._tts2_rocm"}
LABELS = {"cuda": "NVIDIA (CUDA) engine", "rocm": "AMD (ROCm) engine"}
# The libraries the Python API needs, which the [tts-2] extra installs (kitten-tts2 needs them to prepare a model).
PYTHON_MODULES = ("numpy", "huggingface_hub", "tokenizers", "filelock")
INSTALL = 'pip install "kitten-inference"'
REINSTALL = 'pip install --force-reinstall "kitten-inference"'
SYSTEMS = {"darwin": "macOS", "linux": "Linux", "win32": "Windows", "android": "Android", "ios": "iOS"}


class EngineNotInstalled(ImportError):
    """This install has no Kitten TTS 2 engine (or not the one asked for), or this platform has none."""


class EngineMismatch(ImportError):
    """The installed engine does not match this kitten_inference.tts2 (another engine ABI)."""


@dataclass(frozen=True)
class Backend:
    name: str                # "cuda" or "rocm"
    module: str              # the import name of the engine package
    label: str               # "NVIDIA (CUDA) engine" or "AMD (ROCm) engine", for messages
    version: str
    abi: int
    lib: Path                # engine shared library
    cli: Path                # kitten-tts-native executable
    text_data_tool: Path     # kt2n_prepare_text_data executable (model preparation)
    build_info: str
    caps: Tuple[str, ...]
    runtime_found: Optional[bool] = None  # whether find() found its GPU runtime; None: asked for, nothing probed

    @property
    def can_clone(self) -> bool:
        return "voice_cloning" in self.caps


def _import(name: str):
    """The engine module, or None when it is not installed (an error inside an installed one propagates)."""
    module = MODULES[name]
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == module or module.startswith(exc.name + ".")):
            return None
        raise


def _engine_platform() -> bool:
    """Whether Kitten TTS 2 has an engine for this platform (the markers of the [tts-2] extra say the same)."""
    machine = platform.machine()
    return (sys.platform == "linux" and machine == "x86_64") or (sys.platform == "win32" and machine == "AMD64")


def python_message() -> Optional[str]:
    """What to do on a Python older than 3.9, or None."""
    version = tuple(sys.version_info[:2])
    if version >= (3, 9):
        return None
    return (f"Kitten TTS 2 needs Python 3.9 or newer (this is Python {version[0]}.{version[1]}). Kitten TTS 0.8 "
            "(kitten_inference.tts08) works on Python 3.8.")


def no_engine_message(name: Optional[str] = None) -> Optional[str]:
    """What to do when this install has no engine (or not the engine ``name``): one for platforms without an
    engine, one for an install that lost its engine files; None when it has one."""
    if any(_import(key) is not None for key in ([name] if name else MODULES)):
        return None
    if not _engine_platform():
        system = SYSTEMS.get(sys.platform, sys.platform)
        return ("Kitten TTS 2 runs on Linux x86-64 (NVIDIA or AMD Radeon GPU) and Windows x64 (NVIDIA GPU); this is "
                f"{system} {platform.machine() or 'unknown'}. Kitten TTS 0.8 (kitten_inference.tts08) runs here.")
    return f"This install is missing its Kitten TTS 2 engine files. Reinstall it: {REINSTALL}"


def missing_modules() -> List[str]:
    """The PYTHON_MODULES that are not installed (looked up without importing them)."""
    missing = []
    for name in PYTHON_MODULES:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    return missing


def modules_message() -> Optional[str]:
    """What to do when libraries of the [tts-2] extra are missing, or None."""
    missing = missing_modules()
    if not missing:
        return None
    return f"Kitten TTS 2 needs more Python packages (not installed: {', '.join(missing)}). Install them: {INSTALL}"


def require_python_api() -> None:
    """Raises ImportError, saying what to do, when KittenTTS2 cannot work in this install. Checked in this order: the
    Python version, an engine for this platform in this install, the libraries of the [tts-2] extra."""
    message = python_message()
    if message is not None:
        raise ImportError(message)
    message = no_engine_message()
    if message is not None:
        raise EngineNotInstalled(message)
    message = modules_message()
    if message is not None:
        raise ImportError(message)


def _driver_present(name: str) -> bool:
    """Whether this machine has the GPU runtime the engine loads at run time (libcuda / nvcuda.dll, libamdhip64)."""
    return _gpu.runtime(name) is not None


def _describe(name: str, module) -> Backend:
    label = LABELS[name]
    try:
        abi = int(module.ABI)
        backend = Backend(name=name, module=MODULES[name], label=label, version=str(module.VERSION), abi=abi,
                          lib=Path(module.LIB), cli=Path(module.CLI), text_data_tool=Path(module.TEXT_DATA_TOOL),
                          build_info=str(getattr(module, "BUILD_INFO", "")),
                          caps=tuple(getattr(module, "CAPS", ())))
    except AttributeError as exc:
        raise EngineMismatch(f"the {label} in this install is not one this kitten-inference understands ({exc}): "
                             f"reinstall it in a fresh environment: {REINSTALL}") from None
    if abi != _abi.ABI_VERSION:
        raise EngineMismatch(f"the {label} in this install has engine ABI {abi}, but this kitten-inference needs ABI "
                             f"{_abi.ABI_VERSION}: reinstall it in a fresh environment: {REINSTALL}")
    return backend


def find(name: Optional[str] = None) -> Backend:
    """The engine to use: ``name`` ("cuda" or "rocm"), else $KITTEN_TTS2_BACKEND, else the installed one whose GPU
    runtime this machine has (NVIDIA first). With neither runtime it is the NVIDIA engine, with
    ``runtime_found=False``; the messages then name both."""
    requested = name if name is not None else (os.environ.get(ENV) or "").strip() or None
    if requested is not None:
        key = requested.strip().lower()
        if key not in MODULES:
            source = "backend" if name is not None else ENV
            raise ValueError(f"{source}={requested!r}: use 'cuda' (NVIDIA) or 'rocm' (AMD)")
        if key == "rocm" and sys.platform == "win32":
            raise EngineNotInstalled("the AMD (ROCm) engine runs on Linux only; on Windows, run Kitten TTS 2 inside "
                                     f"WSL2: {_gpu.ROCM_DOCS}")
        module = _import(key)
        if module is None:
            raise EngineNotInstalled(no_engine_message(key))
        return _describe(key, module)
    found = [(key, module) for key, module in ((k, _import(k)) for k in MODULES) if module is not None]
    if not found:
        raise EngineNotInstalled(no_engine_message())
    # Request CUDA eager loading before probing runtimes. HIP's loading policy is set by the native engine after
    # dlopen; forcing it here can hang ROCm kernel registration (see _gpu.request_eager_loading).
    _gpu.request_eager_loading("rocm")
    for key, module in found:
        if _driver_present(key):
            return dataclasses.replace(_describe(key, module), runtime_found=True)
    return dataclasses.replace(_describe(*found[0]), runtime_found=False)


def fallback(backend: Backend) -> Optional[Backend]:
    """The AMD engine, when find() picked the NVIDIA one by itself and this machine also has the AMD engine and its
    ROCm runtime (an NVIDIA driver library can be left over on an AMD machine); else None."""
    if backend.name != "cuda" or backend.runtime_found is None:
        return None
    module = _import("rocm")
    if module is None or not _driver_present("rocm"):
        return None
    return dataclasses.replace(_describe("rocm", module), runtime_found=True)


def check_gpu(backend: Backend, device: int = 0, log: Optional[Callable[[str], None]] = None) -> Backend:
    """``_gpu.check`` for ``backend``; returns the engine to start. When the NVIDIA driver is missing or cannot start
    and fallback() has the AMD engine, that one is checked instead, once, and returned. Its messages give only what
    went wrong with the NVIDIA driver (``reason``), not that error's advice, which names the AMD route too."""
    try:
        _gpu.check(backend.name, device, log=log)
        return backend
    except _gpu.DriverStartError as exc:
        other = fallback(backend)
        if other is None:
            raise
        reason = exc.reason
    if log is not None:
        log(f"{reason}; using the {other.label} instead")
    try:
        _gpu.check(other.name, device, log=log)
    except _gpu.GPUError as exc:
        raise _gpu.GPUError(f"{reason}; the {other.label} cannot start either: {exc}") from None
    return other
