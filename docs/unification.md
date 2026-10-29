# Unification capability inventory

The inventory was taken from PR #1's Python APIs, native command wrapper, package metadata, and tests before changing the public interface. Native inference algorithms and model formats are retained.

| Existing capability | Unified interface | Regression coverage |
|---|---|---|
| Nano/micro/mini, FP32/INT8, model aliases, native Hub and local configurations | `KittenTTS(model)` | `tests/tts08/test_api.py`, CPU audio smoke |
| CPU/Metal, custom voices, aliases and speed priors | Constructor backend and synthesis options | TTS 0.8 API/hybrid tests |
| Full audio, text-chunk streaming, file output, custom sample rates | `generate`, `stream`, `generate_to_file`, `save_wav`, `sample_rate` | `tests/test_unified.py`, TTS 0.8 tests |
| Standalone normalization, locale and span results | Root normalizer exports and instance method | `tests/tts08/test_text_normalization.py` |
| CUDA/ROCm, device selection, automatic/low memory, voice caching | TTS 2 constructor options | TTS 2 backend/GPU-check/API tests |
| Whole-sentence and incremental generation, sampling, presets, seed, speed, normalization | `generate`, `stream`, existing TTS 2 options | TTS 2 arguments/API/engine tests |
| Cancellation, close, context manager | Iterator close, `close`, context manager | Unified contract and TTS 2 engine tests |
| Voice listing, cloning, saved voices and identity checks | Common listing; TTS 2 clone/save/load, `Voice`, `can_clone` | TTS 2 API/audio/CLI tests |
| Downloads, preparation, cache, revision, cloning assets, integrity checks | Existing preparation behind unified model selection | TTS 2 preparation/bundle tests |
| Preparation-only, text files, formats, streaming, repeats, threads, concurrency, reports, sampling and cache limits | `kitten-tts --model ...` dispatch | Unified CLI forwarding and existing native CLI tests |
| Build/runtime diagnostics and errors | TTS 2 diagnostics; root exception exports | TTS 2 engine/backend tests |
| Wheel payloads, ABI, licensing, platform selection, provenance and publish guards | Existing verified assembly with consolidated Python sources | `tests/packaging` |

## Validation and release

Install a root wheel before running `python -m pytest tests --ignore=tests/tts2/gpu`; source-tree tests need the built extension and bundled preparation assets. Run GPU tests explicitly on supported devices. Set `KITTEN_RELEASE_WHEEL=1` when testing assembled release wheels to require their platform-specific GPU payloads.

The release input lock remains historical evidence. The source layout and packaged Python code have changed, so fresh native build/test artifacts and a newly recorded tested tree are required before publishing. The existing source and tag checks intentionally reject the old lock; do not rewrite historical hashes or bypass those checks.

The native engines remain separate because their architectures differ. Both user-facing Python implementations now share one install, public interface, command, and dependency table. Model weights remain outside wheels.

## Local verification (2026-10-30)

- Root wheel and sdist-to-wheel builds passed on Linux x86-64, Python 3.14. Internal metadata, TTS 0.8, and TTS 2 input builds passed; their actual dependency metadata agrees after normalizing package names and markers.
- The installed-package suite passed 341 tests and 15 subtests. Eight tests skip here: seven require Windows, and one requires a machine without an NVIDIA driver.
- A fresh environment installed the plain wheel with dependencies and passed `pip check`. CPU synthesis, CLI file output, and streaming passed. Unified and direct CPU generation were array-identical with the same model, text, voice, speed, and normalization setting.
- The tested source-built CPU wheel was 887,005 bytes. Its installed dependency closure was 143,617,638 bytes across 27 distributions in the test environment; this is an installed-file measurement, not download size. No PyTorch or model weights were included. GPU release wheel sizes were not remeasured.
- CUDA/ROCm/Metal inference and cloning were not run. This host has an 8 GB NVIDIA GPU but no matching TTS 2 engine build or prepared model in the inference checkouts. Full platform release validation and refreshed release inputs remain required before publication.
