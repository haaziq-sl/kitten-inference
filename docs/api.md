# Python and command-line API

Install `pip install kitten-inference`. Both model families use the same package and dependencies; model weights download on first use. TTS 2 additionally requires a supported GPU and its system runtime.

```python
from kitten_inference import KittenTTS

with KittenTTS("KittenML/kitten-tts-nano-0.8") as tts:
    tts.generate_to_file("Hello!", "hello.wav", voice="Jasper")

with KittenTTS("KittenML/kitten-tts-2") as tts:
    audio = tts.generate("Hello!", voice="Bruno", seed=42)
    tts.save_wav("hello.wav", audio)
```

A model is required. TTS 0.8 accepts nano/micro/mini FP32 and INT8 names, existing aliases, native Hugging Face repositories, and local native configuration directories. TTS 2 accepts `KittenML/kitten-tts-2` or `kitten-tts-2`. Hardware detection selects a backend within that family; it never changes the model family.

## Common operations

- `KittenTTS(model, backend=None, cache_dir=None, **model_options)` loads the selected model. TTS 0.8 backends are `cpu` and `metal`; TTS 2 backends are `cuda` and `rocm`. Omitting the backend retains each engine's automatic selection.
- `generate(text, voice=None, **options)` returns a mono float32 NumPy array. `sample_rate` gives its rate: normally 24 kHz, or the rate in a custom TTS 0.8 configuration.
- `stream(text, voice=None, **options)` yields arrays. TTS 0.8 yields text chunks; TTS 2 yields incremental audio. Explicitly close the iterator when stopping early, preferably with `contextlib.closing`.
- `generate_to_file(text, output_path, ...)` writes audio using the model's actual sample rate; SoundFile selects the format from the extension. `save_wav(path, audio)` writes PCM16 WAV and accepts an array or a list of chunks.
- `list_voices()` lists available voices. Omitting `voice` retains the engine default: TTS 0.8 uses `expr-voice-5-m`; TTS 2 uses Bruno.
- `capabilities` describes streaming mode, cloning availability, and normalization span support. `close()` is idempotent; closed instances reject synthesis. Instances support context managers. Existing Metal native allocation limitations still apply.

`speed` and `normalize` are shared synthesis options. TTS 0.8 preserves its model-specific speed priors. Its `generate` and `stream` default to normalization off; file generation defaults to normalization on. TTS 2 keeps native normalization enabled by default. Unsupported options raise errors.

## Model-specific operations

TTS 0.8 standalone normalization is available through `tts.normalize_text(text, locale="en-US", return_spans=False)` and the root exports `normalize_text`, `normalize_text_result`, `NormalizedSpan`, and `NormalizedTextResult`. This normalizer is not used for TTS 2.

TTS 2 constructor options are `revision`, `device`, `memory_profile`, and `voice_cache`, with their existing defaults. Synthesis retains `seed`, `preset` (`stable` or `expressive`), `temperature`, `top_p`, `top_k`, `min_p`, `max_new_tokens`/`max_tokens`, and `speed`. `generate(stream=True)` renders through the streaming path and returns all audio.

Cloning retains `clone_voice(reference_audio_path, transcript, **options)`, `save_voice(voice, path)`, and `load_voice(path)`. `Voice` is exported from `kitten_inference`. Check `can_clone()` before cloning; GPU memory, backend, and FFmpeg requirements still apply. Saved voices retain model-identity validation. `revision`, `model_dir`, `build_info()`, and `stats()` remain available for TTS 2.

Public exceptions are exported from `kitten_inference`: `EngineError`, `EngineNotInstalled`, `EngineMismatch`, `PrepareError`, `BundleError`, `AudioError`, and `FFmpegNotFound`.

## Command line

```sh
kitten-tts --model KittenML/kitten-tts-nano-0.8 --text "Hello!" --voice Jasper --output hello.wav
kitten-tts --model KittenML/kitten-tts-2 --text "Hello!" --voice Bruno --output hello.wav
kitten-tts --model KittenML/kitten-tts-2 --prepare-only --with-cloning
```

`python -m kitten_inference` is equivalent. Use `--model MODEL --help` for model-specific options. TTS 0.8 accepts text or a UTF-8 text file, voice listing, backend/cache selection, speed, and normalization. TTS 2 forwards all existing native options, including output formats, reports, streaming, repeat/concurrency/thread controls, sampling, voice-cache limits, and cloning. Preparation retains revision pinning, integrity checks, and cached assets.

## Source builds

`pip install .` builds the CPU extension and installs both Python implementations. GPU binaries are incorporated by the release assembly pipeline; a source-only build does not supply them. Release dependencies and version are defined only in the root `pyproject.toml`; internal build inputs derive metadata from it.

The documented interface replaces the previous model-specific imports and `kitten-tts2` command. Internal `tts08`/`tts2` modules remain implementation details. Native C APIs and the standalone native command retain their existing contracts.
