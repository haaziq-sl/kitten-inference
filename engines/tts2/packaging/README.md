# Building the Kitten TTS 2 engines

The CUDA and ROCm builds here make internal payload wheels. The kitten-inference wheels are assembled from them
(`.github/scripts/assemble_wheels.py`); a payload wheel is never uploaded on its own. Each holds one package:

| Build | Package | Payload wheels |
|---|---|---|
| CUDA (NVIDIA) | `kitten_inference/_tts2_cuda/` | Linux (`py3-none-manylinux_2_28_x86_64`) and Windows (`py3-none-win_amd64`) |
| ROCm (AMD) | `kitten_inference/_tts2_rocm/` | Linux (`py3-none-manylinux_2_28_x86_64`) |

The Linux x86-64 kitten-inference wheels carry both packages, and Kitten TTS 2 picks one when it starts; the Windows x64
wheels carry the CUDA package. The Python 3.8 wheels carry neither (Kitten TTS 2 needs Python 3.9). Each package holds:

| File | What it is |
|---|---|
| `__init__.py` | `BACKEND`, `ABI`, `VERSION`, `LIB`, `CLI`, `TEXT_DATA_TOOL`, `BUILD_INFO` (the library's `kt2n_build_info()` text), `CAPS`; ROCm also `HIP_ARCHS` |
| `lib/libkt2n_engine.so` or `lib/kt2n_engine.dll` | the engine library (C API in `include/kt2n/engine.h`) |
| `bin/kitten-tts-native` (`.exe`) | the command-line tool, which loads the library from `../lib` |
| `bin/kt2n_prepare_text_data` (`.exe`) | the text-data helper used when a model is prepared |

The ROCm payload also has a `licenses/` folder, which the kitten-inference wheels leave out: their `.dist-info/licenses`
folder covers ROCm. `VERSION` is the engine's version (project VERSION in `engines/tts2/CMakeLists.txt`) and must equal the
kitten-inference version without any `.postN`: a post release reuses the same payloads.

The CUDA runtime is linked in, so running needs only the NVIDIA driver (R570 or newer). On Windows the C runtime is
linked in too, so no Visual C++ redistributable is needed. Voice cloning reads the reference audio with the system's
`ffmpeg` program; the library itself links no FFmpeg libraries (`wheel` presets in `CMakePresets.json`).

The ROCm engine uses the system's ROCm 7 runtime (`libamdhip64.so.7`), which is not in the wheel: users install ROCm 7
first ([AMD Radeon guide](../docs/amd-radeon.md)). It is built for gfx1100, gfx1101, gfx1102, gfx1200 and gfx1201
(Radeon RX 7000 and RX 9000) and has no voice cloning.

## How a release uses them

A release builds no engine. `.github/release-inputs.json` names the CI runs whose engine builds passed the GPU tests,
with the sha256 of each payload wheel and of each engine file in it. `packages.yml` downloads those runs' artifacts,
checks them against that file and copies the engine files byte for byte into the kitten-inference wheels. New engine
builds need new GPU tests and a new `release-inputs.json`.

Never run auditwheel, delvewheel or delocate on an assembled wheel: they would change the copied files, and auditwheel
would also try to bundle `libamdhip64.so.7`. `.github/scripts/check_assembled.py` checks the assembled wheels instead.

## Linux (CUDA)

In the `quay.io/pypa/manylinux_2_28_x86_64` container (Docker), this writes the payload wheel to `engines/tts2/dist/`:

```sh
engines/tts2/packaging/build_linux_docker.sh
```

Inside such a container, run `engines/tts2/packaging/build_wheel_linux.sh OUT_DIR` instead. It installs CUDA 12.8
(`install_cuda_linux.sh`), builds the text libraries, runs `cmake --preset wheel`, checks the GPU code (below),
assembles the wheel (`make_wheel.py`) and checks it with `auditwheel show`.

## Windows (CUDA)

From a Visual Studio 2022 x64 developer prompt with CUDA 12.8, CMake, Ninja and Python on `PATH`:

```powershell
powershell -File tts-2\packaging\build_wheel_windows.ps1
```

It checks the GPU code (below) and that the DLL and programs import only Windows system DLLs.

## Linux (ROCm)

On a machine with Docker, from anywhere in the repository, this writes the payload wheel to `engines/tts2/dist/`:

```sh
engines/tts2/packaging/build_wheel_rocm.sh
```

It builds in the manylinux_2_28 container with ROCm 7.1 from AMD's RHEL 8 repository (inside such a container, set
`KT2N_IN_CONTAINER=1`), runs `auditwheel repair` with `libamdhip64.so.7` excluded, and checks the wheel with
`check_wheel_rocm.py`, a fresh virtual environment and both programs. It needs no AMD GPU. Its settings are listed at
the top of the script.

## GPU code check (CUDA)

The CUDA payload's GPU code must be the default build's (`cmake --preset default`), instruction for instruction, so
that it gives the same audio. After `cmake --build`, both CUDA build scripts run `sass_manifest.py check` against
`sass-reference.txt`: one hash per GPU function and architecture of the default build of a checked commit. Checked
means: its GPU code (SASS) is identical, function for function on all 8 architectures, to the previous reference's
(after a deliberate change of GPU code: only the changed functions differ, and they were reviewed), and 36 seeded test
requests give byte-identical audio before and after on an RTX 4090.

nvcc's device front end (cicc, CUDA C++ to PTX) is not deterministic for a few kernels: with the same source, flags
and toolkit, 6 of 44 compiles on a busy Linux build machine gave another SASS for the vocoder's sm_90
`conv1d_tc3_k<64>` (one 16-byte shared-memory load split into a 16-byte and an 8-byte load of the same bytes). On
Windows nearly every compile of the CUDA-core decode kernels (`src/cuda/legacy/tern_dec_cc.cu`, 39 kernels for
sm_61 and sm_75) differs in 1 to 7 of them: 5 of 65 compiles in six CI runs had all of them identical. ptxas gives
the same SASS for the same PTX every time. So when a function's SASS differs, the check compiles the object files that
hold it again and compares their own GPU code with the reference (up to 40 times on Windows, 25 to 40 s each, and 10
on Linux), then links the library again and checks all of it, up to 6 times. A build that still differs, or whose set of
GPU functions differs, fails: the GPU code really changed. After a deliberate change, make the reference again from
the default build of a commit checked as above, built from a checkout at `/root/src/port` (nvcc puts the source path
into the names of GPU functions in anonymous namespaces; `build_linux_docker.sh` mounts the source there too):

```sh
python3 engines/tts2/packaging/sass_manifest.py dump build/default/lib/libkt2n_engine.so \
  --out engines/tts2/packaging/sass-reference.txt --note "default build of <commit>, checked <date>"
```

Only payloads whose build log says `SASS check: IDENTICAL` may go into a release; every payload the CI workflow keeps
as an artifact has passed it. `KT2N_SASS_CHECK=0` skips the check for development builds (for example with fewer
architectures); such a payload is not for release.

## Checks

- `auditwheel_tag.py` (Linux CUDA) checks that `auditwheel show` finds the wheel within its manylinux_2_28 tag.
- `check_wheel_rocm.py` (ROCm) checks the wheel's tag, and that no program or library needs a glibc or libstdc++
  symbol newer than manylinux_2_28 allows, which `auditwheel show` cannot confirm with the system ROCm library.
  `check_assembled.py` runs its `symbols` check on the assembled wheels too.
- `size_check.py WHEEL...` prints the sizes and fails above PyPI's default file limit of 100 MB, which matters for the
  kitten-inference wheels (about 53 MB on Linux x86-64 with both engines).
- `smoke_wheel.py [--backend cuda|rocm]` (run with the Python of a virtual environment where kitten-inference or a
  payload wheel is installed) checks the layout, the version, the build info and that the programs start. Without the
  ROCm runtime it checks that the ROCm engine names `libamdhip64.so.7` when it fails to load. With `--assets MODEL_DIR`
  it also generates speech on the GPU.

`.github/workflows/tts-2-engine.yml` (CUDA) and `tts-2-rocm.yml` (ROCm) build the payloads and check them; the CUDA
workflow also runs the reference-audio ffmpeg tests of the command-line tool on Linux and Windows without a GPU.
`packages.yml` runs `size_check.py` and `smoke_wheel.py` again on the assembled kitten-inference wheels.
