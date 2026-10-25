"""Finds the installed engine package: ``kitten_inference._tts2_cuda`` (NVIDIA) or ``kitten_inference._tts2_rocm`` (AMD)."""
from __future__ import annotations

import ctypes
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from . import _abi

ENV = "KITTEN_TTS2_BACKEND"
MODULES = {"cuda": "kitten_inference._tts2_cuda", "rocm": "kitten_inference._tts2_rocm"}
DISTRIBUTIONS = {"cuda": "kitten-inference-tts2-cu12", "rocm": "kitten-inference-tts2-rocm"}
INSTALL = {"cuda": 'pip install "kitten-inference[tts-2]"', "rocm": 'pip install "kitten-inference[tts-2-rocm]"'}
MISSING = ('no Kitten TTS 2 engine is installed. Install one with: pip install "kitten-inference[tts-2]" (NVIDIA) '
           'or "kitten-inference[tts-2-rocm]" (AMD)')


class EngineNotInstalled(ImportError):
    """No engine package is installed (or not the one that was asked for)."""


class EngineMismatch(ImportError):
    """The installed engine package does not match this kitten-inference-tts2 (another engine ABI)."""


@dataclass(frozen=True)
class Backend:
    name: str                # "cuda" or "rocm"
    module: str              # the import name of the engine package
    distribution: str        # its distribution name
    version: str
    abi: int
    lib: Path                # engine shared library
    cli: Path                # kitten-tts-native executable
    text_data_tool: Path     # kt2n_prepare_text_data executable (model preparation)
    build_info: str
    caps: Tuple[str, ...]

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


def _driver_present(name: str) -> bool:
    """Whether this machine has the GPU runtime the engine loads at run time (used only when both are installed)."""
    if name == "cuda":
        candidates = ["nvcuda.dll"] if sys.platform == "win32" else ["libcuda.so.1"]
    else:
        candidates = ["libamdhip64.so.7"]
    for lib in candidates:
        try:
            ctypes.CDLL(lib)
            return True
        except OSError:
            continue
    return False


def _describe(name: str, module) -> Backend:
    dist = DISTRIBUTIONS[name]
    try:
        abi = int(module.ABI)
        backend = Backend(name=name, module=MODULES[name], distribution=dist, version=str(module.VERSION), abi=abi,
                          lib=Path(module.LIB), cli=Path(module.CLI), text_data_tool=Path(module.TEXT_DATA_TOOL),
                          build_info=str(getattr(module, "BUILD_INFO", "")),
                          caps=tuple(getattr(module, "CAPS", ())))
    except AttributeError as exc:
        raise EngineMismatch(f"{dist} is not a Kitten TTS 2 engine package this version understands ({exc}); "
                             f"reinstall it: {INSTALL[name]}") from None
    if abi != _abi.ABI_VERSION:
        from . import __version__
        raise EngineMismatch(f"{dist} {backend.version} has engine ABI {abi}, but kitten-inference-tts2 "
                             f"{__version__} needs ABI {_abi.ABI_VERSION}. Install matching versions: "
                             f"pip install -U \"kitten-inference[{'tts-2' if name == 'cuda' else 'tts-2-rocm'}]\"")
    return backend


def find(name: Optional[str] = None) -> Backend:
    """The engine to use: ``name`` ("cuda" or "rocm"), else $KITTEN_TTS2_BACKEND, else the one installed.

    With both installed and no choice made, the one whose GPU runtime this machine has is used (CUDA first).
    """
    requested = name if name is not None else (os.environ.get(ENV) or "").strip() or None
    if requested is not None:
        key = requested.strip().lower()
        if key not in MODULES:
            source = "backend" if name is not None else ENV
            raise ValueError(f"{source}={requested!r}: use 'cuda' (NVIDIA) or 'rocm' (AMD)")
        module = _import(key)
        if module is None:
            raise EngineNotInstalled(f"the {key} engine ({DISTRIBUTIONS[key]}) is not installed. Install it with: "
                                     f"{INSTALL[key]}")
        return _describe(key, module)
    found = [(key, module) for key, module in ((k, _import(k)) for k in MODULES) if module is not None]
    if not found:
        raise EngineNotInstalled(MISSING)
    if len(found) > 1:
        for key, module in found:
            if _driver_present(key):
                return _describe(key, module)
    return _describe(*found[0])
