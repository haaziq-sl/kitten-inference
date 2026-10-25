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
(`install_cuda_linux.sh`), builds the text libraries, runs `cmake --preset wheel`, checks the GPU code (below),
assembles the wheel (`make_wheel.py`) and checks it with `auditwheel show`. `WHEEL_VERSION=0.2.0.post1` sets the
version.

## Windows

From a Visual Studio 2022 x64 developer prompt with CUDA 12.8, CMake, Ninja and Python on `PATH`:

```powershell
powershell -File tts-2\packaging\build_wheel_windows.ps1
```

It checks the GPU code (below) and that the DLL and programs import only Windows system DLLs. `-Version 0.2.0.post1`
sets the version.

## GPU code check

The wheel's GPU code must be the default build's (`cmake --preset default`), instruction for instruction, so that it
gives the same audio. After `cmake --build`, both build scripts run `sass_manifest.py check` against
`sass-reference.txt`: one hash per GPU function and architecture of the default build of a gated commit (SASS identical
to the reference release on every architecture, seeded audio identical).

nvcc's device front end (cicc, CUDA C++ to PTX) is not deterministic for a few kernels: with the same source, flags
and toolkit, 6 of 44 compiles on a busy Linux build machine gave another SASS for the vocoder's sm_90
`conv1d_tc3_k<64>` (one 16-byte shared-memory load split into a 16-byte and an 8-byte load of the same bytes), and each
Windows build so far had one such function among the CUDA-core decode kernels. ptxas gives the same SASS for the same
PTX every time. So when a
function's SASS differs, the check deletes the object files that hold it, runs the build again (only those files are
compiled again) and checks again, up to 6 times. A build that still differs, or whose set of GPU functions differs,
fails: the GPU code really changed, and the reference has to be made again from a gated build:

```sh
python3 tts-2/packaging/sass_manifest.py dump build/default/lib/libkt2n_engine.so \
  --out tts-2/packaging/sass-reference.txt --note "default build of <commit>, gated <date>"
```

Release only wheels whose build log says `SASS check: IDENTICAL`; every wheel the CI workflow uploads has passed it.
`KT2N_SASS_CHECK=0` skips the check for development builds (for example with fewer architectures); such a wheel is not
for release.

## Checks

- `auditwheel_tag.py` (Linux) checks that `auditwheel show` finds the wheel within its manylinux_2_28 tag.
- `size_check.py WHEEL...` prints the sizes (PyPI's default file limit is 100 MB).
- `smoke_wheel.py` (run with the Python of a virtual environment the wheel is installed in) checks the layout and
  that the programs start; with `--assets MODEL_DIR` it also generates speech on an NVIDIA GPU.

`.github/workflows/tts-2-engine.yml` runs all of this for both wheels (run it by hand with a `version` for a fix
release), plus the reference-audio ffmpeg tests of the command-line tool on Linux and Windows without a GPU.
