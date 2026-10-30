"""Kitten TTS 2 backend for :class:`kitten_inference.KittenTTS`."""

__version__ = "0.2.0"

import importlib as _importlib
import sys as _sys

# Importing this package needs no third-party module: KittenTTS2 and the names below come on first use, after a
# check that says what is missing (Python 3.9, an engine for this platform, and its dependencies).
_LAZY = {"KittenTTS2": "_api", "Voice": "_api", "SAMPLE_RATE": "_api", "AudioError": "_audio",
         "FFmpegNotFound": "_audio"}

if _sys.version_info >= (3, 9):
    from ._backend import EngineMismatch, EngineNotInstalled  # noqa: E402
    from ._bundle import BundleError  # noqa: E402
    from ._engine import EngineError  # noqa: E402
    from ._prepare import PrepareError  # noqa: E402

__all__ = ["KittenTTS2", "Voice", "SAMPLE_RATE", "EngineError", "EngineNotInstalled", "EngineMismatch",
           "PrepareError", "BundleError", "AudioError", "FFmpegNotFound", "__version__"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import _backend
    _backend.require_python_api()  # ImportError, never AttributeError: `from ... import KittenTTS2` shows its text
    value = getattr(_importlib.import_module(f"{__name__}.{_LAZY[name]}"), name)
    globals()[name] = value
    return value
