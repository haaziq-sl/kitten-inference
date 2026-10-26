# kitten-inference-tts2

Kitten TTS 2 text-to-speech on NVIDIA and AMD GPUs, from Python or the `kitten-tts2` command. On an RTX PRO 6000 it
streams its first audio after about **11 ms**, and finishes complete files **5 to 7 times faster** than Kitten TTS 2
on vLLM or llama.cpp (82 ms against 453 to 584 ms for a short text; one request at a time). On smaller GPUs the lead
is smaller: first audio after 22 ms and 4.5 to 6.9 times faster on an RTX 4090, 58 ms and 2.4 to 4.3 times on a 12 GB
RTX 3060.

```sh
python3 -m venv .venv && . .venv/bin/activate   # Windows, in a Command Prompt: py -m venv .venv, then .venv\Scripts\activate.bat
python -m pip install -U pip                    # pip older than 23.3 skips the "[tts-0.8]" extra with only a warning
pip install "kitten-inference[tts-2]"           # NVIDIA: Linux x86-64 or Windows x64, driver R570 or newer
```

- On Ubuntu and Debian, `python3 -m venv` needs `sudo apt-get update && sudo apt-get install -y python3-venv` first.
- AMD Radeon RX 7000 and RX 9000, inside WSL2 on Windows 11 (tested) or on Linux x86-64 (not tested): install
  ROCm 7.x first (WSL2: step 1 of [AMD Radeon](https://github.com/KittenML/kitten-inference/blob/main/tts-2/docs/amd-radeon.md); Linux
  without WSL2: [AMD's ROCm guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) for your distribution),
  then `pip install "kitten-inference[tts-2-rocm]"` in place of the last line. That package has no voice cloning.
- Python 3.9 to 3.14 (Python 3.15 works only where a C compiler is installed, until PyYAML (a dependency) ships ready-made 3.15 packages). Other systems have no engine: pip then stops with "No matching distribution", or installs the
  old version 0.1.1 of `kitten-inference` instead (`pip show kitten-inference` says which).
- GeForce GTX 10, GTX 16 and RTX 20: supported, not optimal: these GPUs have no bf16 tensor cores, so part of the
  math runs in software (real-time factor 0.22 to 0.28, against 0.05 to 0.08 on RTX 40 and 50 cards; measured with
  an earlier model revision).
- Cards with 4 to 8 GB: supported, not optimal: one request at a time, complete files about 1.5% slower (measured
  with the command-line tool and an earlier model revision), and no voice cloning (load a voice saved on a larger GPU).

```python
from kitten_inference.tts2 import KittenTTS2

tts = KittenTTS2()                     # first use downloads and prepares the model
audio = tts.generate("Hello from Kitten TTS 2.", voice="Bruno")   # float32, 24 kHz, mono
tts.save_wav("hello.wav", audio)
for chunk in tts.stream("Streaming speech, chunk by chunk.", voice="Bruno"):
    ...                                # play or send each chunk as it arrives
```

```sh
kitten-tts2 --text "Hello from Kitten TTS 2." --voice Bruno --output hello.wav
kitten-tts2 --list-voices
```

First use checks the GPU driver, then downloads about 4.6 GB (the latest model revision on Hugging Face unless
`revision=` / `--revision` pins one) and prepares it once per revision: about 1 to 3 minutes in all on our test
machines (55 to 168 s, download included); later starts reuse the prepared model. No PyTorch and no Hugging Face account are needed. First use also fetches a few
small pinned files from PyPI (`files.pythonhosted.org`) and GitHub (`raw.githubusercontent.com`); these are checked
first, so a network that blocks either host fails in seconds rather than after the large download. The downloads stay in the
Hugging Face cache (`~/.cache/huggingface`); the prepared model (2 GB) is in `~/.cache/kitten-inference/tts2/`.
`cache_dir=` / `--cache-dir` keeps both in one folder instead: a new folder, or one that belongs to you and that no
other user can change (new model folders are made private). A frozen app (PyInstaller, cx_Freeze) cannot prepare the
model itself: prepare it once with `kitten-tts2 --prepare-only` and pass the same `revision=` and `cache_dir=`. When a newer revision cannot be prepared (for example for lack of disk space), the one prepared before is
used, with a warning. Each start without a pinned revision asks Hugging Face for the latest one (at most 10 s);
without a connection the prepared model is used, and `HF_HUB_OFFLINE=1` skips the question. On Windows, a user folder path with
non-ASCII characters can move the model to a folder of yours in `C:\ProgramData\kitten-inference`: supported, not
optimal (with the usual Windows code page the engine opens only ASCII paths; the command says where the model went).

Voice cloning (NVIDIA only) needs about 11,300 MiB (11 GB) of free GPU memory while the voice is built, and the
system `ffmpeg`: `sudo apt-get update && sudo apt-get install -y ffmpeg` on Ubuntu, `winget install --id Gyan.FFmpeg -e` on Windows (then open
a new Command Prompt and activate the venv again; Windows 10 and 11 include winget, Windows Server 2022 needs it installed
first). On a 12 GB card other programs may then use at most about 500 MiB of it: cloning worked with
11,285 MiB free and failed with 11,157 MiB free. The first clone adds 1.1 GB to the prepared model.

```python
voice = tts.clone_voice("me.wav", "The exact words spoken in the recording.")
tts.save_voice(voice, "me.voice")      # later: voice = tts.load_voice("me.voice")
audio = tts.generate("New speech in my voice.", voice=voice)
```

A saved voice works only with the model revision it was made with. When Hugging Face has a newer model, keep using
the voice with `KittenTTS2(revision="...")` or `kitten-tts2 --revision ...`: `tts.revision` is the one in use,
`kitten-tts2 --save-voice` prints it, and `load_voice()` and `kitten-tts2 --load-voice` name the one a voice needs.

The Python API gives the same audio and the same first-audio time as the command (11.4 ms both on an RTX PRO 6000).
After `clone_voice()`, each request's first audio comes about 15 ms later (measured on an RTX 4090): supported, not
optimal. The engine then keeps no voice ready between requests, so that building a voice still fits a 12 GB GPU.

GPU memory: `memory_profile="auto"` (the default) uses the default profile (about 13,500 MiB with one voice) when it
fits and the low-memory profile (about 9,100 MiB) otherwise, as on most 12 GB cards: supported, not optimal, because
requests at the same time then share smaller buffers; one request at a time runs about as fast (real-time factor 0.050
against 0.051 on a 16 GB card, measured on an earlier build and model revision). Sizes for each card class:
[GPU memory](https://github.com/KittenML/kitten-inference/blob/main/tts-2/docs/memory.md).

Same audio: a seeded complete file (`generate(..., seed=N)`) is bit for bit the same on Linux and Windows with the
same GPU model. Other GPU models give slightly different samples with the same words and voice. AMD Radeon, GeForce
GTX 10, GTX 16 and RTX 20, and 4 GB cards (long streamed text) can differ more
([details](https://github.com/KittenML/kitten-inference/blob/main/tts-2/docs/consumer-gpus.md#output-differences)).
Streamed audio with the same seed can differ from run to run on any GPU.

## Model license

The package code is Apache-2.0 AND MIT: KittenML's own code is Apache-2.0, and the model preparation includes ported
MIT-licensed code (see [NOTICE](https://github.com/KittenML/kitten-inference/blob/main/tts-2/NOTICE)). The Kitten TTS 2
model this downloads on first use is under the
[Stellon Labs Community License](https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md), not Apache or
MIT: commercial use needs registration, the license ends once your revenue or funding passes US$1M, training on or
distilling from the outputs is not allowed, and redistribution must include the agreement, a notice file and
"Powered by KittenML". Read it before you build on this.

Docs and numbers: https://github.com/KittenML/kitten-inference/tree/main/tts-2
