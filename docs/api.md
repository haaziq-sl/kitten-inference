# Python and command-line API

Install with `pip install kitten-inference`; model weights download on first use. See [installation requirements](../README.md#install) for supported platforms and GPU setup.

```python
from kitten_inference import KittenTTS

with KittenTTS("KittenML/kitten-tts-nano-0.8") as tts:
    tts.generate_to_file("Hello!", "hello.wav", voice="Jasper")

with KittenTTS("KittenML/kitten-tts-2") as tts:
    audio = tts.generate("Hello!", voice="Bruno", seed=42)
    tts.save_wav("hello.wav", audio)
```

A model is required. TTS 0.8 accepts [nano/micro/mini FP32 and INT8 names](../engines/tts08/README.md#models-and-voices), aliases, native Hugging Face repositories, and local native configuration directories. TTS 2 accepts `KittenML/kitten-tts-2` or `kitten-tts-2`. Hardware detection selects a backend for the requested model.

## Common operations

- `KittenTTS(model, backend=None, cache_dir=None, **model_options)` loads the selected model. TTS 0.8 backends are `cpu` and `metal`; TTS 2 backends are `cuda` and `rocm`. The backend is selected automatically when omitted.
- `generate(text, voice=None, **options)` returns a mono float32 NumPy array. `sample_rate` gives its rate: normally 24 kHz, or the rate in a custom TTS 0.8 configuration.
- `stream(text, voice=None, **options)` yields arrays. TTS 0.8 yields text chunks; TTS 2 yields incremental audio. Explicitly close the iterator when stopping early, preferably with `contextlib.closing`.
- `generate_to_file(text, output_path, ...)` writes audio using the model's actual sample rate; SoundFile selects the format from the extension. `save_wav(path, audio)` writes PCM16 WAV and accepts an array or a list of chunks.
- `list_voices()` lists available voices. The default voice is Leo for the standard TTS 0.8 models and Bruno for TTS 2.
- `capabilities` describes streaming mode, cloning availability, and normalization span support. `close()` is safe to call repeatedly; closed instances reject synthesis. Instances support context managers. On Metal, reuse each model: its GPU weights stay allocated until the process exits ([details](../engines/tts08/README.md#models-and-voices)).

`speed` and `normalize` are synthesis options for both models. TTS 0.8 applies the model's speed priors; its `generate` and `stream` default to normalization off, while file generation defaults to normalization on. TTS 2 enables normalization by default. Unsupported options raise errors.

## Model-specific operations

TTS 0.8 standalone normalization is available through `tts.normalize_text(text, locale="en-US", return_spans=False)` and the root exports `normalize_text`, `normalize_text_result`, `NormalizedSpan`, and `NormalizedTextResult`. This normalizer is not used for TTS 2.

TTS 2 constructor options are `revision` (latest by default), `device` (GPU 0), `memory_profile` (`auto`), and `voice_cache` (automatic). Synthesis options include `seed`, `preset` (`stable` or `expressive`), `temperature`, `top_p`, `top_k`, `min_p`, `max_new_tokens`/`max_tokens`, and `speed`. `generate(stream=True)` renders through the streaming path and returns all audio.

Use `clone_voice(reference_audio_path, transcript, **options)` to create a voice, `save_voice(voice, path)` to save it, and `load_voice(path)` to reuse it. `Voice` is exported from `kitten_inference`. Check `can_clone()` and the [GPU memory and FFmpeg requirements](../README.md#kitten-tts-2-nvidia-and-amd-gpus) before cloning. Saved voices require the same model revision and preparation version. TTS 2 also exposes `revision`, `model_dir`, `build_info()`, and `stats()` for diagnostics.

Public exceptions are exported from `kitten_inference`: `EngineError`, `EngineNotInstalled`, `EngineMismatch`, `PrepareError`, `BundleError`, `AudioError`, and `FFmpegNotFound`.

## Command line

```sh
kitten-tts --model KittenML/kitten-tts-nano-0.8 --text "Hello!" --voice Jasper --output hello.wav
kitten-tts --model KittenML/kitten-tts-2 --text "Hello!" --voice Bruno --output hello.wav
kitten-tts --model KittenML/kitten-tts-2 --prepare-only --with-cloning
```

`python -m kitten_inference` is equivalent. Use `kitten-tts --model MODEL --help` for model-specific options. TTS 0.8 accepts text or a UTF-8 text file, voice listing, backend/cache selection, speed, and normalization. TTS 2 also supports output formats, reports, streaming, repeat/concurrency/thread controls, sampling, voice-cache limits, and cloning. Use `--revision` to pin a model version and `--cache-dir` to choose its cache location.

## Source builds

`pip install .` builds the TTS 0.8 extension. Source installs do not include TTS 2 GPU binaries; use the PyPI package for GPU inference, or follow the [native build instructions](../engines/tts2/README.md#build-from-source).
