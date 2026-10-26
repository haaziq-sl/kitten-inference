# kitten-inference

Fast inference engines for Kitten TTS models: Kitten TTS 2 streams its first audio after **19 ms** on an RTX 4090, and Kitten TTS 0.8 nano speaks about **60 times faster than real time** on a 12-core desktop CPU.

```sh
python -m pip install -U pip                 # pip older than 23.3 skips "[tts-0.8]" with only a warning
pip install "kitten-inference[tts-0.8]"      # TTS 0.8 (15M, 40M, 80M) on the CPU: kitten-inference-tts08
pip install "kitten-inference[tts-2]"        # TTS 2 on NVIDIA GPUs: kitten-inference-tts2 + kitten-inference-tts2-cu12
pip install "kitten-inference[tts-2-rocm]"   # TTS 2 on AMD Radeon GPUs (ROCm 7): kitten-inference-tts2 + kitten-inference-tts2-rocm
```

Kitten TTS 2 needs Python 3.9 or newer on Linux x86-64 or Windows x64 (AMD: Linux x86-64); elsewhere pip installs version 0.1.1 of this package instead, without an error. Extras combine: `"kitten-inference[tts-0.8,tts-2]"`.
Python: `from kitten_inference.tts08 import KittenTTS`, `from kitten_inference.tts2 import KittenTTS2`. Docs, speed numbers and examples: https://github.com/KittenML/kitten-inference

Version 0.1.x of this package was the TTS 0.8 engine itself (`import model_inference`); code that uses it pins `kitten-inference==0.1.1`.

Licenses: the package code is Apache-2.0. The models each engine downloads keep their own licenses: the Kitten TTS 0.8 weights are Apache-2.0, while the Kitten TTS 2 model is under the [Stellon Labs Community License](https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md) (commercial use needs registration; the license ends above US$1M revenue or funding; no training on or distilling from outputs; redistribution carries the agreement, a notice file and "Powered by KittenML"). Read it before using TTS 2.
