# Contributing

## Tests

The tests need the [README](README.md)'s Quick start first and a Python with numpy (the `.venv-prepare` environment from Quick start step 3 has it). `cmake --preset dev` makes a second full build in `build/dev`, which takes about 10 minutes with 24 CPU threads. In the repository folder:

```sh
. .venv-prepare/bin/activate
python -m pip install pytest
export PATH=/usr/local/cuda-12.8/bin:$PATH
export KT2N_TEXT_DEPS="$PWD/build-deps/text"
cmake --preset dev
cmake --build --preset dev --parallel "$(nproc)"
ctest --preset cpu
KT2N_TEST_ASSETS="$PWD/models/kitten-tts-2" ctest --preset gpu
KT2N_TEST_GPU=1 KT2N_TEST_ASSETS="$PWD/models/kitten-tts-2" ctest --preset cpu
python -m pytest tests/assets
```

- As in Quick start step 2, type `4` instead of `"$(nproc)"` with less than 16 GB of RAM or in a container with a CPU limit.
- The tests need numpy in the Python that `cmake --preset dev` finds, so activate `.venv-prepare` first. If that Python has no numpy, configure uses `.venv-prepare` in the repository folder by itself when it exists, or stops with `The tests need numpy, and <python> has none`.
- The `cpu` tests need no GPU. The `gpu` tests need an NVIDIA GPU and the prepared model directory in `KT2N_TEST_ASSETS`; without it they fail. They include an end-to-end test of `kitten-tts-native` with its default settings. With the earlier 47-voice model revision they passed on GPUs with 8, 11 and 16 GB; GPUs with less memory were not tested.
- If you prepared the cloning files, add `KT2N_TEST_REQUIRE_CLONING=1` to the `gpu` line: the end-to-end test then also builds a cloned voice (about 13 GB of free GPU memory) and fails, instead of skipping those steps, when the files are missing.
- `ctest` lists skipped tests under "The following tests did not run"; that is expected. Two tests need inputs these steps do not set, and are skipped and counted as passed: `Text.TokenizerAndPrompt` (`KT2N_TOKENIZER_JSON`) and `GpuSmoke.RealEngineFileUnpackMatchesCpu` (`KT2N_REAL_ENGINE_FILE`).
- The third `ctest` line runs the `cpu` tests again with `KT2N_TEST_GPU=1`: this adds the engine's runtime tests that load the real model on the GPU (for example the low-memory fit and the automatic profile). Without `KT2N_TEST_GPU` they are skipped and still counted as passed. One of them needs about 21.5 GB (20 GiB) of free GPU memory, so it is skipped on every card below 24 GB; on 8 GB cards one more is skipped.

## Offline preparation

`tools/prepare_assets.py` takes the latest model revision unless `--revision` pins one, and checks the SHA-256 of every other file it downloads ([PROVENANCE.md](PROVENANCE.md) lists them, from huggingface.co, github.com and raw.githubusercontent.com, and files.pythonhosted.org). The prepared manifest records the model revision it used. Keep the whole prepared directory, including `config.json`. To prepare without network access, fill a cache on a connected machine with `--cache-dir DIR`, then run with `--offline` and the same `--cache-dir`. `--help` lists the options for local copies of each source.

## Troubleshooting

- **`nvcc` not found:** run `export PATH=/usr/local/cuda-12.8/bin:$PATH` in this terminal (add it to `~/.bashrc` to keep it), or pass `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc` to `cmake --preset`.
- **CUDA older than 12.8, or CMake warns that CUDA 13 leaves out GeForce GTX 10 series GPUs:** install CUDA 12.8 as in the README's Quick start, step 2, side by side with the other version, and configure with `cmake --preset default -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc` (or put `/usr/local/cuda-12.8/bin` first on `PATH`). Asking for compute capability 6.1 explicitly with CUDA 13 stops at configure.
- **`apt-get` says "Could not get lock /var/lib/dpkg/lock-frontend":** another program is installing packages, usually the automatic updates that run after a fresh machine starts. Wait until it is done, then run the same command again. To let `apt-get` wait by itself (up to 10 minutes), add `-o DPkg::Lock::Timeout=600` after `apt-get`.
- **`apt-get` fails with "Conflicting values set for option Signed-By"** (NVIDIA's CUDA container images, which already list NVIDIA's repository): run `sudo rm /etc/apt/sources.list.d/cuda.list`, then the commands of the README's Quick start step 2 again.
- **CMake is older than 3.28:** install a newer CMake, for example with `python -m pip install cmake`.
- **Configure fails because OpenFst or PCRE2 is missing:** run `python3 tools/build_text_deps.py --prefix build-deps/text`, then `export KT2N_TEXT_DEPS="$PWD/build-deps/text"` in the same terminal before `cmake --preset`.
- **The build stops with "Killed" or the computer runs out of memory:** run the same `cmake --build` command again with half the `--parallel` number you used (for example `--parallel 2` after `--parallel 4`).
- **"CUDA driver version is insufficient for CUDA runtime version":** the build was made with CUDA 13, which needs NVIDIA driver 580 or newer. Update the driver, or build with CUDA 12.8 as in the README's Quick start, step 2.
- **The engine fails to start on your GPU:** check that its compute capability is 6.1, 7.5, or 8.0 or newer and that `nvidia-smi` shows driver 570 or newer. "not supported by this build" means the library was built without your GPU's architecture: rebuild with `-DKT2N_CUDA_ARCHS` including it (for example `75` for compute capability 7.5).
- **Out of GPU memory, or "exceed the memory budget":** close other GPU programs, keep `--max-streams 1`, leave out `--memory-profile` and `--voice-cache` (the tool then picks settings that fit), or use `--memory-profile low --voice-cache none`. Plain `--voice-cache lru` sets aside room for 16 voices and does not fit on 4 GB cards. For voice cloning, see [Voice cloning](README.md#voice-cloning).
- **"the default memory profile needs about N MiB or more of free GPU memory":** you passed `--memory-profile default` on a GPU without room for it. Leave out `--memory-profile` (the tool then picks the low-memory profile when needed), or pass `--memory-profile low`.
- **"the low-memory profile needs about N MiB of free GPU memory":** the GPU has too little free memory for one stream (the minimum is about 2,583 MiB after the CUDA context). Close other GPU programs and leave out `--voice-cache`; see [Small GPUs](docs/memory.md#small-gpus).
- **`pip` says "No matching distribution found" or a download fails:** usually a short network problem reaching PyPI or download.pytorch.org. Run the same command again. Check that you use Python 3.10 to 3.12 (`python --version`).
- **Preparation cannot find `kt2n_prepare_text_data`:** build the project first (README, Quick start step 2), or pass `--text-converter build/default/bin/kt2n_prepare_text_data`.
- **Preparation says this torch cannot use CUDA:** check `nvidia-smi`, and that you installed torch from the `cu128` index as in the README's Quick start, step 3. On a GTX 10 series GPU, use `--weight-norm-device cpu`.
- **Preparation is refused with a checksum mismatch:** delete the affected file from the cache directory and run the command again.
- **Cloning preparation cannot download pyannote/embedding:** accept the model's conditions on Hugging Face, then run `hf auth login` with the same account.
- **"voice cloning is not available in this library build":** build with the `default` preset, which includes cloning.
- **"this bundle has no voice cloning files":** run the [cloning preparation](README.md#voice-cloning) step on the model directory.
