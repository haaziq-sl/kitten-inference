# kitten-inference

Fast inference engines for the Kitten TTS text-to-speech models.

- **Kitten TTS 2 on NVIDIA and AMD GPUs:** first audio after **19 ms** when streaming on an RTX 4090, and complete
  files 6 to 10 times faster than Kitten TTS 2 on vLLM or llama.cpp.
- **Kitten TTS 0.8 on the CPU:** the nano model speaks about **60 times faster than real time** on a 12-core desktop
  CPU, and 1.1 to 2.3 times faster than kittenml (ONNX Runtime) with default settings.

## Install

```sh
python -m pip install -U pip                 # pip older than 23.3 skips "[tts-0.8]" with only a warning
pip install "kitten-inference[tts-0.8]"      # Kitten TTS 0.8 (15M, 40M, 80M), CPU
pip install "kitten-inference[tts-2]"        # Kitten TTS 2, NVIDIA GPUs
pip install "kitten-inference[tts-2-rocm]"   # Kitten TTS 2, AMD Radeon RX 7000 and RX 9000 (ROCm 7)
```

| Engine | Python | Systems |
|---|---|---|
| Kitten TTS 0.8 | 3.8 to 3.15 | Linux x86-64 and ARM64, Windows x64, macOS (Metal on Apple Silicon) |
| Kitten TTS 2, NVIDIA | 3.9 or newer | Linux x86-64 and Windows x64, NVIDIA driver R570 or newer |
| Kitten TTS 2, AMD | 3.9 or newer | Linux x86-64 with ROCm 7 |

```python
from kitten_inference.tts08 import KittenTTS
audio = KittenTTS("KittenML/kitten-tts-nano-0.8").generate("Hello from Kitten TTS.", voice="Jasper")

from kitten_inference.tts2 import KittenTTS2
audio = KittenTTS2().generate("Hello from Kitten TTS 2.", voice="Bruno")   # first use prepares the model
```

## Docs

- [tts-0.8](tts-0.8/README.md): models, voices, platforms and speed of Kitten TTS 0.8.
- [tts-2/python](tts-2/python/README.md): the Python package and the `kitten-tts2` command for Kitten TTS 2.
- [tts-2](tts-2/README.md): the Kitten TTS 2 engine itself: building from source, the command-line tool, the C API,
  [Windows](tts-2/docs/windows.md), [AMD Radeon](tts-2/docs/amd-radeon.md) and
  [GPU memory](tts-2/docs/memory.md); speed and audio on each GPU group in
  [Consumer GPUs](tts-2/docs/consumer-gpus.md).

## License

The code is licensed under Apache-2.0 ([LICENSE](LICENSE)); some components keep their own licences, listed in
[NOTICE](NOTICE). The models are downloaded on first use and keep their publishers' licences (see each engine's
README).
