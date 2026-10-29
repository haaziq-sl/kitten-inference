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

The release input lock records refreshed native artifacts and their source trees. `tested_packaged_trees` remains unset pending Windows CUDA hardware validation. Record the tested trees and satisfy the repository, source, and tag checks before production publication.

The native engines remain separate because their architectures differ. Both user-facing Python implementations now share one install, public interface, command, and dependency table. Model weights remain outside wheels.

## Validation (2026-10-30)

- Fresh published-package installs following the README passed on Linux, Windows, and macOS, each on x86-64 and ARM64 with Python 3.11 and 3.14: [documented installation results](https://github.com/haaziq-sl/kitten-inference/actions/runs/40973311113).
- All 54 assembled wheels passed dependency, payload, and provenance checks; platform tests covered older Python/pip, minimum dependencies, and missing-driver diagnostics: [package validation](https://github.com/haaziq-sl/kitten-inference/actions/runs/40971905573).
- CPU and Apple Silicon Metal inference passed [native validation](https://github.com/haaziq-sl/kitten-inference/actions/runs/40964306931); torch-free model preparation passed [cross-platform checks](https://github.com/haaziq-sl/kitten-inference/actions/runs/40964307132). Android remains experimental and build-only.
- Published-package hardware checks passed on Linux CUDA (8 GB and 12 GB) and AMD WSL2/ROCm (16 GB), including streaming, cancellation, saved voices, and NVIDIA cloning. Windows CUDA hardware validation remains required; its [native build checks](https://github.com/haaziq-sl/kitten-inference/actions/runs/40963770784) passed.
