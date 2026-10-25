# Kitten TTS 0.8

Small, fast text-to-speech that runs on a CPU. Three models from 15M to 80M parameters, eight voices,
24 kHz audio, on a native C++ engine.

<!-- HEADLINE NUMBERS PLACEHOLDER: replace this block with the benchmark results
     (speed and peak memory vs the ONNX path in kittenml, on Linux x86, Apple Silicon and Linux ARM). -->
> **Headline numbers: to be added from the benchmark.**

## Install

```bash
python -m pip install -U pip              # pip older than 23.3 skips "[tts-0.8]" with only a warning
pip install "kitten-inference[tts-0.8]"
```

## Use

```python
from kitten_inference.tts08 import KittenTTS

tts = KittenTTS("KittenML/kitten-tts-mini-0.8")
audio = tts.generate("This high-quality voice runs without a GPU.", voice="Jasper")  # float32, 24 kHz
tts.generate_to_file("Hello, world.", "hello.wav", voice="Luna", speed=1.2)
```

The first use of a model downloads it from Hugging Face; later uses load it from the local cache.

## Models and voices

| Model | Parameters | Download |
|---|---|---|
| `KittenML/kitten-tts-mini-0.8` | 80M | 293 MB |
| `KittenML/kitten-tts-micro-0.8` | 40M | 140 MB |
| `KittenML/kitten-tts-nano-0.8` | 15M | 56 MB |

Add `-int8` to a name, for example `KittenML/kitten-tts-nano-0.8-int8`, for the smaller int8 weights
(116, 71 and 19 MB). On Linux ARM64 and on the CPU of Apple Silicon Macs they are also faster than the
full-size weights; elsewhere they are slower: 1.3 to 2 times on x86_64 and about 8 times on Windows ARM64.
Apple Silicon runs them on the CPU by default, because on Metal they are about 5 times slower.

Voices: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki and Leo (`tts.available_voices`).

Options:
- `speed`: 1.0 is normal, higher is faster.
- `clean_text=True`: spells out numbers, currencies and units before speaking.
- `tts.generate_stream(text, voice=...)`: yields the audio one sentence at a time.
- `KittenTTS(..., backend="cpu")` or `"metal"`: by default Apple Silicon uses Metal for the full-size
  models and the CPU for the int8 models. On the CPU, models in different threads generate at the same
  time; one model shared by several threads runs their calls one at a time, and Metal runs one call at a time.

Audio can differ slightly between: Metal and the CPU (clip lengths within about 3.5%), and, for the int8
models, different CPU types (within about 2%).

## Platforms

| Platform | Python | |
|---|---|---|
| Linux x86_64 | 3.8–3.15 | tested |
| Linux ARM64 | 3.8–3.15 | tested |
| Windows x64 | 3.8–3.15 | tested |
| macOS Apple Silicon | 3.8–3.15 | tested |
| macOS Intel | 3.8–3.15 | built; passes an automated audio check, not tested by hand |
| Windows ARM64 | 3.11–3.15 | built; passes an automated audio check, not tested by hand |
| Android ARM64 (experimental) | 3.13 | built, not tested |

- Linux needs glibc 2.29 or newer: Ubuntu 20.04, Debian 11, RHEL 9 or later. macOS needs version 11 or newer.
- x86_64 needs a CPU with AVX2 (Intel Core since 2013, AMD since 2015).
- Linux ARM64 needs an ARMv8.2 CPU with dot-product instructions, such as a Raspberry Pi 5.
  A Raspberry Pi 4 is not supported.
- Windows ARM64 is supported, not optimal: that build has no ARM vector kernels, so it runs about
  9 times slower than Linux ARM64 on comparable test machines, and the int8 models about 8 times
  slower again. Use the full-size models there.
- The CPU engine on macOS is supported, not optimal: the macOS build has no OpenMP, so it uses one
  CPU core. On a 4-core Intel Mac, a build using all 4 cores was about 2 times faster. This affects
  macOS Intel, where the CPU is the only backend, and the CPU on Apple Silicon (int8 models by default).

## Upgrading from kitten-inference 0.1.x

Version 0.2 replaces the `model_inference` module with `kitten_inference.tts08`. Code that runs
`import model_inference` keeps working only with 0.1.1: pin `kitten-inference==0.1.1`.

## License

Apache-2.0. See `LICENSE` and `NOTICE` in this folder. The model weights are downloaded separately
and have their own terms.
