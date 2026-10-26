# kitten-inference-tts2

Kitten TTS 2 text-to-speech on NVIDIA and AMD GPUs, from Python or the `kitten-tts2` command. On an RTX 4090 it
streams its first audio after **19 ms** and speaks a 7-second sentence in **0.2 s** (medians of 27 requests).

```sh
python -m pip install -U pip                 # older pip versions can skip an extra with only a warning
pip install "kitten-inference[tts-2]"        # NVIDIA: Linux x86-64 or Windows x64, driver R570 or newer
pip install "kitten-inference[tts-2-rocm]"   # AMD Radeon: Linux x86-64 with ROCm 7.x (no voice cloning)
```

Python 3.9 or newer. There is no engine for other platforms: there pip installs version 0.1.1 of `kitten-inference`
instead, without an error (`pip show kitten-inference` then says 0.1.1).

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
`revision=` / `--revision` pins one) and prepares it once per revision: 35 to 100 s of preparation on our test
machines, plus the download time. No PyTorch and no Hugging Face account are needed. First use also fetches a few
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
`kitten-tts2 --save-voice` prints it, and `load_voice()` and `kitten-tts2 --load-voice` name the one a voice needs. The Python API gives the same audio and first-audio time as the command (19.0 ms vs 19.1 ms). After
`clone_voice()`, each request starts about 14 ms later (33 ms instead of 19 ms on an RTX 4090): supported, not
optimal. The engine then keeps no voice ready between requests, so that building a voice still fits a 12 GB GPU.

Docs and numbers: https://github.com/KittenML/kitten-inference/tree/main/tts-2
