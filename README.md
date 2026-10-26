# kitten-inference

Fast native inference engines for the Kitten TTS text-to-speech models: Kitten TTS 2 on NVIDIA and AMD GPUs
([`tts-2/`](tts-2/)) and Kitten TTS 0.8 on the CPU ([`tts-0.8/`](tts-0.8/)). Install them with pip; no PyTorch needed.

|  | **Kitten TTS 2** on an NVIDIA GPU | **Kitten TTS 0.8** on the CPU |
|---|---|---|
| Headline | Complete files **5 to 7 times faster** than Kitten TTS 2 on vLLM or llama.cpp | The nano model **1.1 to 2.3 times faster** than kittenml 0.9.6 (ONNX Runtime), on every machine we tested |
| Speed | First streamed audio after **about 11 ms** | **About 60 times faster than real time** |
| Example | A short text (one chunk) in 82 ms, a long text (about three chunks) in 0.95 s | A 41-word text (15 to 17 s of speech) in 0.29 s, against 0.54 s with ONNX Runtime |
| Measured on | RTX PRO 6000, one request at a time | Ryzen 9 9900X (12 cores); the 1.1 to 2.3 times covers macOS, Linux (x86-64 and ARM64) and Windows |

- **Kitten TTS 2:** the pip package with its default settings and the current model, medians of 300 short and 12 long
  texts, the same texts, voice and seeds for every engine. On vLLM (kittenml 0.9.6) 453 ms / 4.9 s with 4-bit weights
  and 584 ms / 6.6 s with bf16, its default; on llama.cpp (GPU) 520 ms / 5.4 s. Word error rate 0.02 to 0.05% for all
  four on the short texts. vLLM and llama.cpp stream whole sentences, so on a short text their first audio comes with
  the complete file (481 to 643 ms).
- **Kitten TTS 0.8:** the default settings of both packages on the same machine, median of 10 calls. On Apple Silicon
  this engine runs nano on the GPU (Metal) and ONNX Runtime on the CPU. "Times faster than real time" is speech length
  divided by compute time over six test clips (real-time factor 0.017).

## Install

Each block starts a Python virtual environment and upgrades pip in it. On Ubuntu and Debian, `python3 -m venv` needs
`sudo apt-get install -y python3-venv` first.

### Kitten TTS 2: NVIDIA and AMD GPUs

```sh
python3 -m venv .venv && . .venv/bin/activate   # Windows: py -m venv .venv, then .venv\Scripts\activate
python -m pip install -U pip                    # pip older than 23.3 skips the "[tts-0.8]" extra with only a warning
pip install "kitten-inference[tts-2]"           # NVIDIA: Linux x86-64 or Windows x64, driver R570 or newer
kitten-tts2 --text "Hello from Kitten TTS 2." --voice Bruno --output hello.wav
```

- Python 3.9 or newer. First use downloads the model (about 4.6 GB) and prepares it once: about 1 to 3 minutes on
  our test machines (57 s with a 1 Gbit/s connection, 160 s on Windows). Later starts reuse the prepared model.
- **AMD Radeon RX 7000 and RX 9000** on Linux x86-64: install ROCm 7 first (step 1 of
  [AMD Radeon](tts-2/docs/amd-radeon.md)), then `pip install "kitten-inference[tts-2-rocm]"` in place of `[tts-2]`.
  Real-time factor about 0.06 on an RX 9070 XT (10 s of speech in about 0.6 s). No voice cloning. Tested inside WSL2
  on Windows; not tested on Linux without WSL2.
- **Voice cloning** (NVIDIA only) needs FFmpeg (`sudo apt-get install -y ffmpeg`; on Windows
  `winget install Gyan.FFmpeg`, where Windows 10 and 11 include winget and Windows Server 2022 needs it installed
  first) and about 11,300 MiB (11 GB) of free GPU memory while the voice is built: on a 12 GB card, other programs may
  use at most about 500 MiB.
- **GeForce GTX 10, GTX 16 and RTX 20 series:** supported, not optimal: they have no bf16 tensor cores, so part of the
  math runs in software (real-time factor 0.22 to 0.28 on an RTX 2060, GTX 1660 Ti and GTX 1080 Ti, against 0.05 to
  0.08 on RTX 40 and 50 series cards).
- **Without Python:** the engine is also a C library and the `kitten-tts-native` command. Build it with
  `git clone --branch v0.2.0 https://github.com/KittenML/kitten-inference.git`, then `cd kitten-inference/tts-2`, and
  follow [tts-2/README.md](tts-2/README.md).

The package picks its memory settings from the free GPU memory:

| GPU memory | Built-in voices | Voice cloning |
|---|---|---|
| 24 GB or more | default profile: about 13,500 MiB with one voice | yes: about 15,600 MiB while a voice is built |
| 16 GB | default profile when it fits, else the low-memory profile | yes, with the low-memory profile: about 11,300 MiB |
| 12 GB | low-memory profile, about 9,100 MiB: supported, not optimal, because requests at the same time share smaller buffers (one at a time runs as fast: real-time factor 0.050 against 0.051 on a 16 GB card) | only with about 11,300 MiB free |
| 4 to 8 GB | low-memory profile fitted to the card: supported, not optimal (one request at a time; complete files about 1.5% slower) | no: build and save the voice on a larger GPU, then load it |

### Kitten TTS 0.8: the CPU

```sh
python3 -m venv .venv && . .venv/bin/activate   # Windows: py -m venv .venv, then .venv\Scripts\activate
python -m pip install -U pip                    # pip older than 23.3 skips "[tts-0.8]" with only a warning
pip install "kitten-inference[tts-0.8]"
python -c "from kitten_inference.tts08 import KittenTTS; KittenTTS('KittenML/kitten-tts-nano-0.8').generate_to_file('Hello from Kitten TTS.', 'hello.wav', voice='Jasper')"
```

- Python 3.8 to 3.15 on Linux x86-64 and ARM64, Windows x64 and Apple Silicon Macs. Intel Macs and Windows ARM64:
  built; passes an automated audio check, not tested by hand. Android ARM64: built, not tested.
- On Windows, if `KittenTTS()` reports "espeak not installed", install the Microsoft Visual C++ Redistributable:
  `winget install Microsoft.VCRedist.2015+.x64`.
- With a pip you cannot upgrade, write `"kitten-inference[tts-0-8]"`: supported, not optimal, because pip 22.0.2 to
  23.2.1 install no engine for `[tts-0.8]` and still report success.
- **Supported, not optimal:**
  - int8 models on Intel CPUs: 1.6 to 2.4 times the ONNX Runtime time, because the int8 kernels for x86 are not
    tuned yet (on the AMD and Arm CPUs we tested they are faster than ONNX Runtime);
  - the Mac CPU (int8 models on Apple Silicon, every model on Intel Macs): the macOS build uses one CPU core, so int8
    on an Apple M4 takes 1.4 to 1.8 times the ONNX Runtime time;
  - Intel CPUs with performance and efficiency cores: the work is spread evenly over all cores, so a Core i9-12900
    was slower than ONNX Runtime for 5 of 6 models; on its performance cores only it took 0.64 to 0.75 times as long;
  - Windows ARM64: built without ARM vector kernels, so about 9 times slower than Linux ARM64.

Both engines in one environment: `pip install "kitten-inference[tts-0.8,tts-2]"` (Linux x86-64 and Windows x64,
Python 3.9 or newer).

## Same audio?

- **Kitten TTS 2:** seeded complete files are bit for bit the same on Linux and Windows with the same GPU model.
  Different GPU models give slightly different samples with the same words and voice. These can differ more: AMD
  Radeon, GeForce GTX 10, GTX 16 and RTX 20 (close to, not bit for bit the same as, a newer NVIDIA GPU's), and 4 GB
  cards, which cut long streamed text into shorter pieces (word error rate 1.43% against 0.58%). Streamed audio with
  the same seed can differ from run to run on any GPU.
- **Kitten TTS 0.8:** audio can differ slightly between Metal and the CPU, between CPU types and with the number of
  threads; on Metal the same text can also differ slightly from one run to the next.

## Use from Python

```python
from kitten_inference.tts2 import KittenTTS2

tts = KittenTTS2()                                    # first use downloads and prepares the model
tts.save_wav("hello.wav", tts.generate("Hello from Kitten TTS 2.", voice="Bruno"))   # float32, 24 kHz mono
for chunk in tts.stream("Streaming speech, chunk by chunk.", voice="Bruno"):
    ...                                               # play or send each chunk as it arrives
```

```python
from kitten_inference.tts08 import KittenTTS

KittenTTS("KittenML/kitten-tts-nano-0.8").generate_to_file("Hello from Kitten TTS.", "hello.wav", voice="Jasper")
```

## Docs

- [Kitten TTS 2 from Python and the `kitten-tts2` command](tts-2/python/README.md): voices, streaming, voice
  cloning, saved voices, caches.
- [Kitten TTS 2 engine](tts-2/README.md): speed tables, [supported GPUs](tts-2/README.md#supported-gpus), building
  from source, the command-line tool and the C API. Also: [Windows](tts-2/docs/windows.md),
  [AMD Radeon](tts-2/docs/amd-radeon.md), [GPU memory](tts-2/docs/memory.md),
  [speed and audio by GPU group](tts-2/docs/consumer-gpus.md).
- [Kitten TTS 0.8](tts-0.8/README.md): models and voices, options, platforms, and upgrading from 0.1.x
  (`import model_inference` needs `kitten-inference==0.1.1`).

## License

KittenML's code is Apache-2.0 ([LICENSE](LICENSE)); the packages also contain third-party code under other licences
([NOTICE](NOTICE), [tts-2/NOTICE](tts-2/NOTICE), [tts-0.8/NOTICE](tts-0.8/NOTICE)). The models download on first use and keep their publishers' licences. The Kitten TTS 2 model is
under the [Stellon Labs Community License](https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md), not
Apache or MIT: it limits commercial use, training on the outputs and redistribution, so read it before you build on
Kitten TTS 2. The Kitten TTS 0.8 model licences are in [tts-0.8/README.md](tts-0.8/README.md#license).
