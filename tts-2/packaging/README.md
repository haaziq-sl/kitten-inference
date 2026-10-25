# Building the CUDA wheels

`kitten-inference-tts2-cu12` holds the compiled engine for NVIDIA GPUs: one wheel for Linux
(`py3-none-manylinux_2_28_x86_64`) and one for Windows (`py3-none-win_amd64`). It contains only the package
`kitten_inference/_tts2_cuda/`:

| File | What it is |
|---|---|
| `__init__.py` | `BACKEND`, `ABI`, `VERSION`, `LIB`, `CLI`, `TEXT_DATA_TOOL`, `BUILD_INFO` (the library's `kt2n_build_info()` text), `CAPS` |
| `lib/libkt2n_engine.so` or `lib/kt2n_engine.dll` | the engine library (C API in `include/kt2n/engine.h`) |
| `bin/kitten-tts-native` (`.exe`) | the command-line tool, which loads the library from `../lib` |
| `bin/kt2n_prepare_text_data` (`.exe`) | the text-data helper used when a model is prepared |

The CUDA runtime is linked in, so running needs only the NVIDIA driver (R570 or newer). On Windows the C runtime is
linked in too, so no Visual C++ redistributable is needed. Voice cloning reads the reference audio with the system's
`ffmpeg` program; the library itself links no FFmpeg libraries (`wheel` presets in `CMakePresets.json`).

## Linux

In the `quay.io/pypa/manylinux_2_28_x86_64` container (Docker):

```sh
tts-2/packaging/build_linux_docker.sh            # writes tts-2/dist/*.whl
```

or, inside such a container, `tts-2/packaging/build_wheel_linux.sh OUT_DIR`. It installs CUDA 12.8
(`install_cuda_linux.sh`), builds the text libraries, runs `cmake --preset wheel`, assembles the wheel
(`make_wheel.py`) and checks it with `auditwheel show`.

## Windows

From a Visual Studio 2022 x64 developer prompt with CUDA 12.8, CMake, Ninja and Python on `PATH`:

```powershell
powershell -File tts-2\packaging\build_wheel_windows.ps1
```

It checks that the DLL and programs import only Windows system DLLs.

## Checks

- `size_check.py WHEEL...` prints the sizes (PyPI's default file limit is 100 MB).
- `smoke_wheel.py` (run with the Python of a virtual environment the wheel is installed in) checks the layout and
  that the programs start; with `--assets MODEL_DIR` it also generates speech on an NVIDIA GPU.

`.github/workflows/tts-2-engine.yml` runs all of this for both wheels.
