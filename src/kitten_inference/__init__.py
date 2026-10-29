"""Native Kitten speech synthesis: ``KittenTTS(model)`` selects a model explicitly."""
__version__ = "0.2.0"

# Internal release inputs may be installed separately during native-engine tests.
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
del extend_path

__all__ = ['KittenTTS', 'Voice', 'normalize_text', 'normalize_text_result',
           'NormalizedSpan', 'NormalizedTextResult', 'EngineError', 'EngineNotInstalled',
           'EngineMismatch', 'PrepareError', 'BundleError', 'AudioError', 'FFmpegNotFound']


def __getattr__(name):
    if name == 'KittenTTS':
        from ._api import KittenTTS
        return KittenTTS
    if name in ('normalize_text', 'normalize_text_result', 'NormalizedSpan', 'NormalizedTextResult'):
        from .tts08 import preprocess
        return getattr(preprocess, name)
    if name in __all__:
        from . import tts2
        return getattr(tts2, name)
    raise AttributeError(name)
