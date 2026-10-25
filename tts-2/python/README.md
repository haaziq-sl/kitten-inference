# kitten-inference-tts2

Python API and `kitten-tts2` command for Kitten TTS 2 on NVIDIA and AMD GPUs. Install it through the umbrella
package, which also installs the GPU engine:

```sh
pip install "kitten-inference[tts-2]"        # NVIDIA (CUDA engine, driver R570 or newer)
pip install "kitten-inference[tts-2-rocm]"   # AMD Radeon (needs ROCm 7.x; no voice cloning)
```

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

On first use the model is downloaded from Hugging Face (KittenML/kitten-tts-2, latest revision unless
`revision=` / `--revision` pins one) and prepared on your machine, once per model revision, into
`~/.cache/kitten-inference/tts2/`. No PyTorch and no Hugging Face account are needed.

Voice cloning (NVIDIA, about 13 GB of free GPU memory) uses the system `ffmpeg` to read the recording:

```python
voice = tts.clone_voice("me.wav", "The exact words spoken in the recording.")
tts.save_voice(voice, "me.voice")      # later: voice = tts.load_voice("me.voice")
audio = tts.generate("New speech in my voice.", voice=voice)
```

Docs and numbers: https://github.com/KittenML/kitten-inference/tree/main/tts-2
