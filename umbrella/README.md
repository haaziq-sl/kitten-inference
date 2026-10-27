# kitten-inference

Fast inference engines for the Kitten TTS text-to-speech models:

- **Kitten TTS 2 on an NVIDIA GPU:** first streamed audio after about **11 ms**, and complete files **5 to 7 times
  faster** than Kitten TTS 2 on vLLM or llama.cpp (RTX PRO 6000, one request at a time; 2.4 times or more on a 12 GB
  RTX 3060). AMD Radeon RX 7000 and RX 9000 are supported too.
- **Kitten TTS 0.8 on the CPU:** the nano model runs **1.1 to 2.3 times faster** than kittenml 0.9.6 (ONNX Runtime) on
  every machine we tested, and about **60 times faster than real time** on a 12-core desktop CPU.

```sh
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -U pip
pip install "kitten-inference[tts-0.8]"
```

This installs Kitten TTS 0.8 on the CPU. On Windows, run the commands in a Command Prompt, with `py -m venv .venv` and then `.venv\Scripts\activate.bat` in place of the first line. Upgrade pip as shown: pip older than 23.3 skips the `[tts-0.8]` extra with only a warning. On Ubuntu and Debian, `python3 -m venv` needs `sudo apt-get update && sudo apt-get install -y python3-venv` first.

For Kitten TTS 2, install `"kitten-inference[tts-2]"` instead. It needs Python 3.9 to 3.14 and an NVIDIA GPU (driver
R570 or newer) on Linux x86-64 or Windows x64, or an AMD Radeon RX 7000 or RX 9000 on Linux x86-64 (tested inside
WSL2) with ROCm 7 installed ([setup](https://github.com/KittenML/kitten-inference/blob/main/tts-2/docs/amd-radeon.md)).
It uses the GPU it finds. On Linux ARM64, macOS and Windows ARM64 the package installs too, and Kitten TTS 2 says when
it starts that it does not run there. Both engines: `"kitten-inference[tts-0.8,tts-2]"`.

- With a pip you cannot upgrade, write the extra as `"kitten-inference[tts-0-8]"` (this needs pip 20.3 or newer on
  Linux x86-64; older pip finds no wheel of this version there): supported, not optimal, because pip older than 23.3
  (for example 22.0.2 on Ubuntu 22.04, or 21.2.4 with the python3 that macOS includes) skips `[tts-0.8]` with only a
  warning: Kitten TTS 0.8 is then installed without the Python packages it needs, and says so when you import it.
- Python: `from kitten_inference.tts08 import KittenTTS`, `from kitten_inference.tts2 import KittenTTS2`.
- Version 0.1.x of this package was the TTS 0.8 engine itself (`import model_inference`); code that uses it pins
  `kitten-inference==0.1.1`.

Docs, speed tables and examples: https://github.com/KittenML/kitten-inference

Licenses: KittenML's code is Apache-2.0. The engines in this package also contain third-party code under other
licenses, listed in the NOTICE files in its `.dist-info/licenses/` folder (also
[on GitHub](https://github.com/KittenML/kitten-inference/blob/main/NOTICE)). The models each engine downloads keep their own licenses. The Kitten TTS 2
model is under the [Stellon Labs Community License](https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md),
not Apache or MIT: commercial use needs registration, the license ends once your revenue or funding passes US$1M,
training on or distilling from the outputs is not allowed, and redistribution must include the agreement, a notice
file and "Powered by KittenML". Read it before you build on Kitten TTS 2. The Kitten TTS 0.8 model licenses are in
the [tts-0.8 README](https://github.com/KittenML/kitten-inference/tree/main/tts-0.8#license).
