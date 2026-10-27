# Kitten TTS 0.8

Small, fast text-to-speech that runs on a CPU. Three models from 15M to 80M parameters, eight voices,
24 kHz audio, on a native C++ engine.

**The nano model speaks about 60 times faster than real time on a 12-core desktop CPU, and 1.1 to 2.3 times
faster than kittenml 0.9.6 (ONNX Runtime) on Linux (x86-64 and ARM64), Windows x64 and Apple Silicon Macs.** On
Linux x86-64 and Windows x64 it also uses 3 to 42% less memory than kittenml.

A 41-word text (15 to 17 s of speech) takes nano 0.29 s against 0.54 s on a Ryzen 9 9900X, and 0.37 s against
0.71 s on an Apple M4, where this engine uses the GPU (Metal) and ONNX Runtime the CPU. Both packages at default
settings on the same machine, median of 10 calls.

![Kitten TTS 0.8 engine against ONNX Runtime on the CPU, 80M int8 model: latency and peak memory on an Android phone (Cortex-A77), a Core i7-12650H laptop and an Apple M2 Pro](docs/assets/benchmarks/cpu_onnx_summary.svg)

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -U pip
pip install "kitten-inference[tts-0.8]"
```

- On Windows, run the commands in a Command Prompt, with `py -m venv .venv` and then `.venv\Scripts\activate.bat` in place of the first line.
- On Ubuntu and Debian, `python3 -m venv` needs `sudo apt-get update && sudo apt-get install -y python3-venv` first.
- Upgrade pip as shown: pip older than 23.3 skips the `[tts-0.8]` extra with only a warning, and Kitten TTS 0.8
  then reports the missing packages when you import it.
- With a pip you cannot upgrade, write `"kitten-inference[tts-0-8]"`: supported, not optimal, because it is a
  workaround for old pip, and on Linux x86-64 it still needs pip 20.3 or newer.

## Use

```python
from kitten_inference.tts08 import KittenTTS

tts = KittenTTS("KittenML/kitten-tts-mini-0.8")
audio = tts.generate("This high-quality voice runs without a GPU.", voice="Jasper")
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
(116, 71 and 19 MB). They are usually faster than kittenml on AMD CPUs and on Linux ARM64. Apple Silicon runs them
on the CPU by default; on Metal they are about 5 times slower.

- int8 on Intel CPUs: supported, not optimal, because the int8 kernels for x86 are not tuned yet (1.2 to 2.2 times
  the kittenml time).
- int8 on the Mac CPU: supported, not optimal, because the macOS build uses one CPU core (1.4 to 1.8 times the
  kittenml time on an Apple M4).

Voices: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki and Leo (`tts.available_voices`).

Options:
- `speed`: 1.0 is normal, higher is faster.
- `clean_text=True`: spells out numbers, currencies and units before speaking.
- `tts.generate_stream(text, voice=...)`: yields the audio one sentence at a time.
- `KittenTTS(..., backend="cpu")` or `"metal"`: by default Apple Silicon uses Metal for the full-size
  models and the CPU for the int8 models. On the CPU, models in different threads generate at the same
  time; one model shared by several threads runs their calls one at a time, and Metal runs one call at a time.
- On Metal, create each model once and reuse it. Creating it again is supported, not optimal, because the engine
  does not free its GPU copy of the weights yet: every new `KittenTTS` keeps about 290 MB (nano) to 470 MB (mini)
  until the process ends. The CPU backend frees it.

Audio can differ slightly between Metal and the CPU (clip lengths within about 3.5%), between CPU types, and with
the number of CPU threads (`OMP_NUM_THREADS`).

## Platforms

| Platform | Python | Status |
|---|---|---|
| Linux x86_64 and ARM64 | 3.8–3.14 | Supported |
| Windows x64 | 3.8–3.14 | Supported |
| macOS Apple Silicon | 3.8–3.14 | Supported |
| macOS Intel | 3.8–3.14 | Experimental |
| Windows ARM64 | 3.11–3.14 | Experimental; about 9 times slower than Linux ARM64 |
| Android ARM64 | 3.13 | Experimental |

- Linux needs glibc 2.29 or newer (Ubuntu 20.04, Debian 11, RHEL 9 or later); macOS needs version 11 or newer.
- The CPU needs AVX2 on x86-64, or dot-product instructions on ARM64 (a Raspberry Pi 5 works, a Raspberry Pi 4 does not).
- On Windows, keep the virtual environment's path under about 150 characters. If `KittenTTS()` reports "espeak not
  installed", install the Microsoft Visual C++ Redistributable: `winget install Microsoft.VCRedist.2015+.x64`.

## Upgrading from kitten-inference 0.1.x

Version 0.2 replaces the `model_inference` module with `kitten_inference.tts08`. Code that runs
`import model_inference` keeps working only with 0.1.1: pin `kitten-inference==0.1.1`.

## License

The code is licensed under Apache-2.0. The Kitten TTS 0.8 engine also contains third-party code under its own
licences (MIT, BSD-3-Clause, GPL-3.0-or-later with the GCC Runtime Library Exception, Apache-2.0 with LLVM
exceptions, and the Microsoft Visual C++ runtime terms): see
[NOTICE](https://github.com/KittenML/kitten-inference/blob/main/tts-0.8/NOTICE), which each wheel also
carries in its `.dist-info/licenses/tts-0.8/` folder. The same wheels also hold Kitten TTS 2, whose third-party
code is listed in [tts-2/NOTICE](https://github.com/KittenML/kitten-inference/blob/main/tts-2/NOTICE).

The models are the Kitten TTS 0.8 models, which KittenML publishes under Apache-2.0
([nano](https://huggingface.co/KittenML/kitten-tts-nano-0.8),
[micro](https://huggingface.co/KittenML/kitten-tts-micro-0.8),
[mini](https://huggingface.co/KittenML/kitten-tts-mini-0.8)); this package downloads them converted to its
engine's format from [KittenML/meownn-models](https://huggingface.co/KittenML/meownn-models).

## Work in progress

Kitten TTS 0.8 support is still being improved. Next: a faster GPU path and faster int8 models.
