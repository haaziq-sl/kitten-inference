# Provenance

This file records where the code in this repository comes from, which upstream files the preparation tool downloads, and what was left out. Licenses and copyright notices are in [NOTICE](NOTICE) and [licenses/](licenses/).

## Origin of the code

The engine was first published as the `native/` directory, together with a command-line front end in `tools/kitten-tts/`, in [KittenML/kitten-tts-2-cpp pull request #6](https://github.com/KittenML/kitten-tts-2-cpp/pull/6), head commit `fd3ab3c40eae3b1f32440bfc6859e7756c11e8a9` ("kitten-tts: add native CUDA inference backend").

This repository starts from that commit:

- The contents of `native/` moved to the repository root. The model file formats, numerics and precision are unchanged, and the C API (`include/kt2n/engine.h`) keeps ABI version 2: the types, functions and configuration fields of `fd3ab3c` keep their layout, and their meaning except for the changes listed at the end of this file.
- The command-line tool was rewritten as `kitten-tts-native` in `cli/`, from `tools/kitten-tts/native_cli.cpp` and related files. It uses the engine's own JSON and SHA-256 code, reads only local files and has no network code.
- Downloading moved into `tools/prepare_assets.py`, which takes the latest model revision and pins every other source listed below.

## Upstream files used by preparation

`tools/prepare_assets.py` downloads and checks these files. Inference reads only the prepared local directory.

| Source | Pin | Used for |
|---|---|---|
| [KittenML/kitten-tts-2](https://huggingface.co/KittenML/kitten-tts-2) (Hugging Face) | latest revision (`main`) unless `--revision` pins one; the prepared manifest records the exact commit | speech language model, tokenizer, built-in voices |
| [ResembleAI/chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) (Hugging Face) | revision `1e4698ca7cbb41ff030c4185f0927a3b42d76924`, `s3gen_meanflow.safetensors` SHA-256 `d65cb687a2ed581ee6cc297e919ffefa63386944f42364ae13b78a594945514f` | speech decoder weights; cloning tokenizer and CAM++ encoder weights |
| [KittenML/kitten-text-processing](https://github.com/KittenML/kitten-text-processing) (GitHub) | commit `1988d61ca82d51ad11cb4a541ac61e1f46677eb4` | English normalization grammar data |
| [nemo_text_processing 1.2.0](https://pypi.org/project/nemo-text-processing/1.2.0/) (PyPI, from [NVIDIA NeMo Text Processing](https://github.com/NVIDIA/NeMo-text-processing)) | wheel SHA-256 `dd78657a0f031023de8b2c8fee6b55d71f444485826f7b507881e407afa5d804` | English word lists, read as data; the package is never installed |
| [pyannote/embedding](https://huggingface.co/pyannote/embedding) (Hugging Face, gated) | revision `4db4899737a38b2d618bbd74350915aa10293cb2`, checkpoint SHA-256 `4bcec986de13da7af7ac88736572692359950df63669989c4f78b294934c9089` | speaker embedding for optional voice cloning |

Build-time sources, also pinned by SHA-256: OpenFst 1.8.3 and PCRE2 10.42 (`tools/build_text_deps.py`) and GoogleTest 1.17.0 for tests (`cmake/Kt2nDeps.cmake`).

## Ported algorithms

These parts re-implement published code. [NOTICE](NOTICE) gives each license.

- Speech decoder (encoder, MeanFlow estimator, HiFT vocoder): S3Gen from Resemble AI's Chatterbox, which builds on CosyVoice and Matcha-TTS.
- Voice cloning: the pyannote.audio speaker-embedding network, the CAM++ speaker encoder (3D-Speaker, as adapted by FunASR) and the S3Tokenizer speech tokenizer.
- Text normalization: the token parser of NVIDIA NeMo Text Processing, driven by grammar data from KittenML/kitten-text-processing. The optional multilingual tables in `src/text/voice/` were exported from ICU 78.2 / CLDR 48 on top of kitten-text-processing at `1988d61`.
- Tempo stretching: FFmpeg's `af_atempo` filter (LGPL-2.1-or-later).
- Random numbers: NumPy 1.26.4's PCG64, SeedSequence and float32 ziggurat.

## What was removed or replaced, and why

- **Tests that depended on non-public code.** The original test suite included a copy of a reference weight packer from a source tree that is not public, and tests that compared the converter against it. That copy was removed. The tests now compare the converter's output with SHA-256 golden vectors recorded once from the reference packer (`tests/data/pack_golden.json`), so they still check exact output and no longer need PyTorch. For the same reason, `tools/kt2n_assets/verify.py` no longer runs its checks against the reference packer; its own repack and unpack checks remain.
- **Comments that pointed to non-public files.** Comments that cited file names or line numbers in non-public code were reworded or removed. The rewording changed no code: for every C, C++ and CUDA file it touched, the preprocessed token stream is the same before and after, apart from a few error messages (including one decoder stage name that appears in an error message), one test printout and some test names that were reworded to drop internal names (status codes are unchanged).
- **An unused test script** (`tests/text/voice_contract.py`) that no test target ran.
- **Network code in the command-line tool.** The old front end downloaded files at run time through llama.cpp's common library and libcurl and parsed JSON with a vendored copy of nlohmann/json. The new tool reads only a prepared local directory, so these dependencies are gone.
- **The CPU backend.** The llama.cpp/GGML speech model and LibTorch decoder stay in kitten-tts-2-cpp and are not part of this repository.
- **Not included:** a Python SDK, wheel or bindings; CI workflow files; a network server.

## What did not change

Model numerics, precision, the built-in voices of the model, streaming and complete (`--stream false`) generation, cloning, cancellation and incremental text sessions are the same as in `fd3ab3c`. The C ABI version and the configuration fields are unchanged, with these exceptions:

- Sampling applies the repetition penalty to the silence token too. Exempting it let the model stall in long silences.
- `levers_json` and `app` are reserved and must be NULL; the engine always uses its built-in defaults.
- A saved voice records the hashes of the prepared model it was built with and loads only with that model, so voices saved by `fd3ab3c` must be built again.
