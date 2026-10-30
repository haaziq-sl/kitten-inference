"""Native Kitten speech synthesis: ``KittenTTS(model)`` selects a model explicitly."""
__version__ = "0.2.0"

# Internal release inputs may be installed separately during native-engine tests.
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
del extend_path

_EXPORTS = {
    'KittenTTS': '._api',
    'Voice': '.tts2._api',
    'normalize_text': '.tts08.preprocess',
    'normalize_text_result': '.tts08.preprocess',
    'NormalizedSpan': '.tts08.preprocess',
    'NormalizedTextResult': '.tts08.preprocess',
    'EngineError': '.tts2._engine',
    'EngineNotInstalled': '.tts2._backend',
    'EngineMismatch': '.tts2._backend',
    'PrepareError': '.tts2._prepare',
    'BundleError': '.tts2._bundle',
    'AudioError': '.tts2._audio',
    'FFmpegNotFound': '.tts2._audio',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    from importlib import import_module
    return getattr(import_module(_EXPORTS[name], __name__), name)
