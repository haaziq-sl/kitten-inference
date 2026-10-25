"""Kitten TTS 2 on NVIDIA and AMD GPUs.

    from kitten_inference.tts2 import KittenTTS2
    tts = KittenTTS2()                      # first use downloads and prepares the model
    audio = tts.generate("Hello from Kitten TTS 2.", voice="Bruno")
    tts.save_wav("hello.wav", audio)

The engine comes from a separate package: pip install "kitten-inference[tts-2]" (NVIDIA) or
"kitten-inference[tts-2-rocm]" (AMD).
"""

__version__ = "0.2.0"

from ._api import SAMPLE_RATE, KittenTTS2, Voice  # noqa: E402
from ._audio import AudioError, FFmpegNotFound  # noqa: E402
from ._backend import EngineMismatch, EngineNotInstalled  # noqa: E402
from ._bundle import BundleError  # noqa: E402
from ._engine import EngineError  # noqa: E402
from ._prepare import PrepareError  # noqa: E402

__all__ = ["KittenTTS2", "Voice", "SAMPLE_RATE", "EngineError", "EngineNotInstalled", "EngineMismatch",
           "PrepareError", "BundleError", "AudioError", "FFmpegNotFound", "__version__"]
