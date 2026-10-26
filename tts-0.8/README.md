# Kitten TTS 0.8

Small, fast text-to-speech that runs on a CPU. Three models from 15M to 80M parameters, eight voices,
24 kHz audio, on a native C++ engine.

**The nano model speaks about 60 times faster than real time on a 12-core desktop CPU, and 1.1 to 2.3 times
faster than kittenml 0.9.6 (ONNX Runtime) on every machine we compared: macOS, Linux (x86-64 and Arm) and
Windows.** On Linux x86-64 and Windows it also uses 3 to 42% less memory than kittenml.

Default settings of both packages on the same machine, a 41-word text (15 to 17 s of speech), median of 10 calls.
On a Ryzen 9 9900X, nano speaks it in 0.29 s against 0.54 s (real-time factor 0.017). On Apple Silicon this engine
runs nano on the GPU (Metal) and ONNX Runtime on the CPU: 0.37 s against 0.71 s on an Apple M4.

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate   # Windows: py -m venv .venv, then .venv\Scripts\activate
python -m pip install -U pip                    # pip older than 23.3 skips "[tts-0.8]" with only a warning
pip install "kitten-inference[tts-0.8]"
```

- On Ubuntu and Debian, `python3 -m venv` needs `sudo apt-get install -y python3-venv` first.
- With a pip you cannot upgrade, write `"kitten-inference[tts-0-8]"`: supported, not optimal, because pip 22.0.2 to
  23.2.1 install no engine for `[tts-0.8]` and still report success.

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
(116, 71 and 19 MB). On the AMD and Linux ARM64 machines we tested they are faster than kittenml. Supported, not
optimal on Intel CPUs and on the Mac CPU: 1.6 to 2.4 times the kittenml time on the Intel CPUs we tested, because
the int8 kernels for x86 are not tuned yet, and 1.4 to 1.8 times on an Apple M4, because the macOS build uses one
CPU core. Apple Silicon runs them on the CPU by default, because on Metal they are about 5 times slower.

Voices: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki and Leo (`tts.available_voices`).

Options:
- `speed`: 1.0 is normal, higher is faster.
- `clean_text=True`: spells out numbers, currencies and units before speaking.
- `tts.generate_stream(text, voice=...)`: yields the audio one sentence at a time.
- `KittenTTS(..., backend="cpu")` or `"metal"`: by default Apple Silicon uses Metal for the full-size
  models and the CPU for the int8 models. On the CPU, models in different threads generate at the same
  time; one model shared by several threads runs their calls one at a time, and Metal runs one call at a time.

Audio can differ slightly between Metal and the CPU (clip lengths within about 3.5%), between CPU types, and with
the number of threads (the same clip lengths). On Metal, the same text can also give slightly different audio from
one run to the next.

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
- x86_64 needs a CPU with AVX2 (Intel Core since 2013, AMD since 2015), and Linux ARM64 an ARMv8.2
  CPU with dot-product instructions, such as a Raspberry Pi 5 (a Raspberry Pi 4 is not supported).
  On other CPUs, `KittenTTS()` stops with an error that names the missing instructions.
- Windows ARM64 is supported, not optimal: that build has no ARM vector kernels, so it runs about
  9 times slower than Linux ARM64 on comparable test machines, and the int8 models about 8 times
  slower again. Use the full-size models there.
- Intel CPUs with performance and efficiency cores: by default the engine runs on the performance
  cores' threads and lets the faster threads take more of its main matrix work, instead of splitting
  it evenly over every core (`OMP_NUM_THREADS` still sets the thread count). A short sentence took
  0.12 s instead of 0.26 s on a Core i9-12900, and 0.09 s instead of 1.99 s on a Core i9-14900K.
- The CPU engine on macOS is supported, not optimal: the macOS build has no OpenMP, so it uses one
  CPU core. On a 4-core Intel Mac, a build using all 4 cores was about 2 times faster. This affects
  macOS Intel, where the CPU is the only backend, and the CPU on Apple Silicon (int8 models by default).
- Windows: the wheel brings the Microsoft Visual C++ runtime that the text front end (espeak-ng)
  needs, and uses it only when Windows has none. If `KittenTTS()` still reports "espeak not installed",
  install the Microsoft Visual C++ Redistributable: `winget install Microsoft.VCRedist.2015+.x64`.

## Upgrading from kitten-inference 0.1.x

Version 0.2 replaces the `model_inference` module with `kitten_inference.tts08`. Code that runs
`import model_inference` keeps working only with 0.1.1: pin `kitten-inference==0.1.1`.

## License

The code is licensed under Apache-2.0. The wheels also contain third-party code under its own licences
(MIT, BSD-3-Clause, GPL-3.0-or-later with the GCC Runtime Library Exception, Apache-2.0 with LLVM
exceptions, and the Microsoft Visual C++ runtime terms): see
[NOTICE](https://github.com/KittenML/kitten-inference/blob/main/tts-0.8/NOTICE), which each wheel also
carries in its `.dist-info/licenses/` folder.

The models are downloaded on first use. They are the Kitten TTS 0.8 models, which KittenML publishes under
Apache-2.0 ([nano](https://huggingface.co/KittenML/kitten-tts-nano-0.8),
[micro](https://huggingface.co/KittenML/kitten-tts-micro-0.8),
[mini](https://huggingface.co/KittenML/kitten-tts-mini-0.8)); this package downloads them converted to its
engine's format from [KittenML/meownn-models](https://huggingface.co/KittenML/meownn-models).
