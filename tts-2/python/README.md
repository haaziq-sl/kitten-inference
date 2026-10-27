# Kitten TTS 2 from Python

Kitten TTS 2 text-to-speech on NVIDIA and AMD GPUs, from Python or the `kitten-tts2` command. On an RTX PRO 6000 it
streams its first audio after about **11 ms**, and finishes complete files **5 to 7 times faster** than Kitten TTS 2
on vLLM or llama.cpp (82 ms against 453 to 584 ms for a short text; one request at a time). On smaller GPUs the lead
is smaller: first audio after 22 ms and 4.5 to 6.9 times faster on an RTX 4090, 58 ms and 2.4 to 4.3 times on a 12 GB
RTX 3060.

## Install

```sh
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -U pip
pip install "kitten-inference[tts-2]"
```

- NVIDIA GPU on Linux x86-64 or Windows x64, with driver R570
  or newer.
- On Windows, run the commands in a Command Prompt, with `py -m venv .venv` and then `.venv\Scripts\activate.bat` in place of the first line.
- On Ubuntu and Debian, `python3 -m venv` needs `sudo apt-get update && sudo apt-get install -y python3-venv` first.
- AMD Radeon RX 7000 and RX 9000: supported inside WSL2 on Windows 11 (not the RX 7600 or RX 7600 XT, which AMD's
  WSL2 support leaves out), experimental on Linux x86-64 without WSL2 and on the RX 7700 XT and 8 GB Radeons.
  Install ROCm 7.x first (WSL2: step 1 of [AMD Radeon](https://github.com/KittenML/kitten-inference/blob/main/tts-2/docs/amd-radeon.md); Linux
  without WSL2: [AMD's ROCm guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) for your distribution),
  then run the same commands: the package finds the Radeon. No voice cloning on Radeon. With both an NVIDIA and an
  AMD GPU the NVIDIA one is used; `KittenTTS2(backend="rocm")`, `kitten-tts2 --backend rocm` or
  `KITTEN_TTS2_BACKEND=rocm` picks the AMD one.
- Python 3.9 to 3.14. Python 3.15 also needs a C compiler, until PyYAML (a dependency) ships ready-made 3.15
  packages. Kitten TTS 2 runs on Linux x86-64 and Windows x64; elsewhere the Python API and `kitten-tts2` stop with a
  message that says so.
- GeForce GTX 10, GTX 16 and RTX 20: supported, not optimal: no bf16 tensor cores, so part of the math runs in
  software (real-time factor 0.22 to 0.37 on desktop cards with 6 GB or more, against 0.05 to 0.08 on RTX 40 and 50
  cards).
- Cards with 4 to 8 GB: supported, not optimal: one request at a time, complete files about 1.5% slower, and no voice
  cloning (load a voice saved on a larger GPU).

## Use

```python
from kitten_inference.tts2 import KittenTTS2

tts = KittenTTS2()
audio = tts.generate("Hello from Kitten TTS 2.", voice="Bruno")
tts.save_wav("hello.wav", audio)
for chunk in tts.stream("Streaming speech, chunk by chunk.", voice="Bruno"):
    ...
```

```sh
kitten-tts2 --text "Hello from Kitten TTS 2." --voice Bruno --output hello.wav
kitten-tts2 --list-voices
```

## First use and caches

First use checks the GPU driver, then downloads about 4.6 GB (the latest model revision on Hugging Face unless
`revision=` / `--revision` pins one) and prepares it once per revision, in about 1 to 3 minutes; later starts reuse
the prepared model. No PyTorch and no Hugging Face account are needed. First use also fetches a few small files from
PyPI (`files.pythonhosted.org`) and GitHub (`raw.githubusercontent.com`); a network that blocks either host fails in
seconds, before the large download.

- The downloads stay in the Hugging Face cache (`~/.cache/huggingface`); the prepared model (2 GB) is in
  `~/.cache/kitten-inference/tts2/`.
- `cache_dir=` / `--cache-dir` keeps both in one folder instead: a new folder, or one that belongs to you and that no
  other user can change.
- A frozen app (PyInstaller, cx_Freeze) cannot prepare the model itself: prepare it once with
  `kitten-tts2 --prepare-only` and pass the same `revision=` and `cache_dir=`.
- Each start without a pinned revision asks Hugging Face for the latest one (at most 10 s); without a connection the
  prepared model is used, and `HF_HUB_OFFLINE=1` skips the question. When a newer revision cannot be prepared (for
  example for lack of disk space), the one prepared before is used, with a warning.
- On Windows, a user folder path with non-ASCII characters can move the model to a folder of yours in
  `C:\ProgramData\kitten-inference`: supported, not optimal (with the usual Windows code page the engine opens only
  ASCII paths; the command says where the model went).

## Voice cloning

Voice cloning (NVIDIA only) needs about 11,300 MiB (11 GB) of free GPU memory while the voice is built (on a 12 GB
card, other programs may use at most about 500 MiB), and the system `ffmpeg`:
`sudo apt-get update && sudo apt-get install -y ffmpeg` on Ubuntu, `winget install --id Gyan.FFmpeg -e` on Windows
(then open a new Command Prompt and activate the venv again; Windows 10 and 11 include winget, Windows Server 2022
needs it installed first). The first clone adds 1.1 GB to the prepared model.

```python
voice = tts.clone_voice("me.wav", "The exact words spoken in the recording.")
tts.save_voice(voice, "me.voice")
audio = tts.generate("New speech in my voice.", voice=voice)
```

A saved voice works only with the model revision it was made with. When Hugging Face has a newer model, keep using
the voice with `KittenTTS2(revision="...")` or `kitten-tts2 --revision ...`: `tts.revision` is the one in use,
`kitten-tts2 --save-voice` prints it, and `load_voice()` and `kitten-tts2 --load-voice` name the one a voice needs.

After `clone_voice()`, each request's first audio comes about 15 ms later (RTX 4090): supported, not optimal, because
the engine then keeps no voice ready between requests, so that building a voice still fits a 12 GB GPU.

## GPU memory

`memory_profile="auto"` (the default) uses the default profile (about 13,500 MiB with one voice) when it fits and the
low-memory profile (about 9,100 MiB) otherwise, as on most 12 GB cards: supported, not optimal, because requests at
the same time then share smaller buffers; one request at a time runs about as fast (real-time factor 0.050 against
0.051 on a 16 GB card). Sizes for each card class:
[GPU memory](https://github.com/KittenML/kitten-inference/blob/main/tts-2/docs/memory.md).

## Same audio

A seeded complete file (`generate(..., seed=N)`) is bit for bit the same on Linux and Windows with the same GPU
model. Other GPU models give slightly different samples with the same words and voice. AMD Radeon, GeForce GTX 10,
GTX 16 and RTX 20, and 4 GB cards (long streamed text) can differ more
([details](https://github.com/KittenML/kitten-inference/blob/main/tts-2/docs/consumer-gpus.md#output-differences)).
Streamed audio with the same seed can differ from run to run on any GPU.

## License

This Python code is Apache-2.0 AND MIT: KittenML's own code is Apache-2.0, and the model preparation includes ported
MIT-licensed code (see [NOTICE](https://github.com/KittenML/kitten-inference/blob/main/tts-2/NOTICE)). The Kitten TTS 2
model this downloads on first use is under the
[Stellon Labs Community License](https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md), not Apache or
MIT: commercial use needs registration, the license ends once your revenue or funding passes US$1M, training on or
distilling from the outputs is not allowed, and redistribution must include the agreement, a notice file and
"Powered by KittenML". Read it before you build on this.

Docs and numbers: https://github.com/KittenML/kitten-inference/tree/main/tts-2
