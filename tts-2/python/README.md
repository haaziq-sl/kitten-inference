# kitten-inference-tts2

Kitten TTS 2 text-to-speech on NVIDIA and AMD GPUs, from Python or the `kitten-tts2` command. On an RTX PRO 6000 it
streams its first audio after about **11 ms**, and finishes complete files **5 to 7 times faster** than Kitten TTS 2
on vLLM or llama.cpp (82 ms against 453 to 584 ms for a short text; one request at a time).

```sh
python3 -m venv .venv && . .venv/bin/activate   # Windows: py -m venv .venv, then .venv\Scripts\activate
python -m pip install -U pip                    # pip older than 23.3 skips the "[tts-0.8]" extra with only a warning
pip install "kitten-inference[tts-2]"           # NVIDIA: Linux x86-64 or Windows x64, driver R570 or newer
```

- On Ubuntu and Debian, `python3 -m venv` needs `sudo apt-get install -y python3-venv` first.
- AMD Radeon RX 7000 and RX 9000 on Linux x86-64: install ROCm 7.x first (step 1 of
  [AMD Radeon](https://github.com/KittenML/kitten-inference/blob/main/tts-2/docs/amd-radeon.md)), then
  `pip install "kitten-inference[tts-2-rocm]"` in place of the last line. No voice cloning there.
- Python 3.9 or newer. Other systems have no engine: pip then stops with "No matching distribution", or installs the
  old version 0.1.1 of `kitten-inference` instead (`pip show kitten-inference` says which).

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
`cache_dir=` / `--cache-dir` keeps both in one folder instead; on Windows it must be a new folder or one only you can
change. When a newer revision cannot be prepared (for example for lack of disk space), the one prepared before is
used, with a warning. Each start without a pinned revision asks Hugging Face for the latest one (at most 10 s);
without a connection the prepared model is used, and `HF_HUB_OFFLINE=1` skips the question. On Windows, a user folder path with
non-ASCII characters can move the model to a folder of yours in `C:\ProgramData\kitten-inference`: supported, not
optimal (with the usual Windows code page the engine opens only ASCII paths; the command says where the model went).

Voice cloning (NVIDIA only) needs about 11,300 MiB (11 GB) of free GPU memory while the voice is built, and the
system `ffmpeg`. On a 12 GB card other programs may then use at most about 500 MiB of it: cloning worked with
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
After `clone_voice()`, each request starts about 15 ms later (34 ms instead of 19 ms on an RTX 4090): supported, not
optimal. The engine then keeps no voice ready between requests, so that building a voice still fits a 12 GB GPU.

GPU memory: `memory_profile="auto"` (the default) uses the default profile (about 13,500 MiB with one voice) when it
fits and the low-memory profile (about 9,100 MiB) otherwise, as on most 12 GB cards: supported, not optimal, because
requests at the same time then share smaller buffers; one request at a time runs as fast (real-time factor 0.050
against 0.051 on a 16 GB card). Sizes for each card class:
[GPU memory](https://github.com/KittenML/kitten-inference/blob/main/tts-2/docs/memory.md).

## Model license

The package code is Apache-2.0. The Kitten TTS 2 model this downloads on first use is under the
[Stellon Labs Community License](https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md), not Apache or
MIT: commercial use needs registration, the license ends once your revenue or funding passes US$1M, training on or
distilling from the outputs is not allowed, and redistribution must include the agreement, a notice file and
"Powered by KittenML". Read it before you build on this.

Docs and numbers: https://github.com/KittenML/kitten-inference/tree/main/tts-2
