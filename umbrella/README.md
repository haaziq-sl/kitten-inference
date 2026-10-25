# kitten-inference

Fast inference engines for Kitten TTS models. This package only holds the extras; each one installs an engine.

| Install | Engine |
|---|---|
| `pip install "kitten-inference[tts-0.8]"` | Kitten TTS 0.8 (15M, 40M, 80M) on the CPU: `kitten-inference-tts08` |
| `pip install "kitten-inference[tts-2]"` | Kitten TTS 2 on NVIDIA GPUs: `kitten-inference-tts2` + `kitten-inference-tts2-cu12` |
| `pip install "kitten-inference[tts-2-rocm]"` | Kitten TTS 2 on AMD Radeon GPUs (ROCm 7): `kitten-inference-tts2` + `kitten-inference-tts2-rocm` |

Extras combine: `"kitten-inference[tts-0.8,tts-2]"`. `[tts-0-8]` is the same extra as `[tts-0.8]`.
Python: `from kitten_inference.tts08 import KittenTTS`, `from kitten_inference.tts2 import KittenTTS2`.

Docs, speed numbers and examples: https://github.com/KittenML/kitten-inference

Version 0.1.x of this package was the TTS 0.8 engine itself (`import model_inference`); code that uses it pins `kitten-inference==0.1.1`.
