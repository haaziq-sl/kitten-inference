"""Kitten TTS 0.8 (15M, 40M and 80M models) on the native C++ engine.

    from kitten_inference.tts08 import KittenTTS
    tts = KittenTTS("KittenML/kitten-tts-mini-0.8")
    audio = tts.generate("Hello", voice="Jasper")   # numpy float32, 24 kHz
"""
from .preprocess import NormalizedSpan, NormalizedTextResult, normalize_text, normalize_text_result

__version__ = "0.2.0"

__all__ = [
    "get_model",
    "KittenTTS",
    "normalize_text",
    "normalize_text_result",
    "NormalizedSpan",
    "NormalizedTextResult",
]


def __getattr__(name):
    # Lazy, so text normalisation works without loading espeak or the engine.
    if name in {"get_model", "KittenTTS"}:
        from .get_model import KittenTTS, get_model

        # Bind the names here: importing the submodule set `get_model` to the module.
        globals().update(KittenTTS=KittenTTS, get_model=get_model)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
