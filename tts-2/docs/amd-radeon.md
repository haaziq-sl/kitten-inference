# AMD Radeon

Kitten TTS 2 native runs on AMD Radeon RX 7000 (RDNA3) and RX 9000 (RDNA4) GPUs with a ROCm build. These steps set
up ROCm inside WSL2 (the Linux subsystem of Windows), then build and run the engine there. Measured numbers are in
[Consumer GPUs](consumer-gpus.md#amd-radeon).

Tested inside WSL2 on Windows desktops with RX 7000 and RX 9000 cards. Linux without WSL2 (bare metal) was not
tested: there, install ROCm 7 with [AMD's ROCm guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) for your distribution in
place of step 1.

## What works, and what does not yet

- Works: speech with every built-in voice, streaming and `--stream false`, the voice cache and memory options, saved
  voices (`--load-voice`) and the C API.
- No voice cloning: the `hip` build compiles it, but it was not tested on Radeon, and the pip package leaves it out.
  Build and save a voice on an NVIDIA GPU, then load it here with `--load-voice`.
- The audio is close to, not bit for bit the same as, an NVIDIA GPU's; see
  [Output differences](consumer-gpus.md#output-differences). These steps install ROCm 7.2; our measured numbers were
  made with ROCm 7.1 builds. A ROCm 7.2 build differs from a ROCm 7.1 build only at the level of rounding (about -60
  to -65 dB, same words and voice) and runs about 2% slower (real-time factor 0.092 against 0.090 on an RX 9060 XT).
  With the ROCm 7.2 runtime, the first request of each process on an RX 7900 XTX often starts its audio about 17 ms
  later than with ROCm 7.1 (66 against 49 ms); later requests in the same process, complete files and the RX 9070 XT
  are not affected.
- Cards with less free memory: as on NVIDIA, the tool picks the low-memory profile, and on 8 GB cards it fits that
  profile to the card. Supported, not optimal: several requests at once are batched less, and when fitted it serves
  one stream and has no voice cloning. Complete files keep the same audio and speed (real-time factor 0.095 on an
  RX 9060 XT). 8 GB Radeons and the RX 7700 XT were not tested.
- Under WSL2 the fitted profile keeps at least 1 GiB of GPU memory free, because some RX 7900 XT and XTX PCs hang
  below about 0.5 GB free.
- Supported GPUs: gfx1100 and gfx1101 (RX 7900 and RX 7800 / 7700 series) and gfx1200 and gfx1201 (RX 9000). The
  build also includes gfx1102 (RX 7600 and RX 7600 XT), but AMD's WSL2 support does not include these cards and bare
  metal was not tested, so they are not tested. Other AMD GPUs stop at start with one of these
  messages:
  - Radeon RX 6000 and older RDNA GPUs, even in a build made for them: `this GPU (<name>, <gfx id>) is not supported;
    the native engine needs an AMD Radeon RX 7000 (RDNA3) or RX 9000 (RDNA4) GPU`;
  - Instinct (MI series) and older GCN GPUs: `this GPU (<name>, <gfx id>) runs 64-lane wavefronts; the native engine
    needs a 32-lane (wave32) AMD GPU such as Radeon RX 7000 (RDNA3) or RX 9000 (RDNA4)`;
  - RDNA3 and RDNA4 GPUs that the build has no code for, such as the Radeon 780M and other integrated GPUs: `this GPU
    (<name>, <gfx id>) is not supported by this build, which has no kernels for it (built for <list>); rebuild with
    KT2N_HIP_ARCHS including it`. A build with `-DKT2N_HIP_ARCHS=gfx1103` (Radeon 780M) or `gfx1150;gfx1151` (Ryzen AI
    integrated GPUs) compiles, but no such GPU was tested: not supported.

## Requirements

- An AMD Radeon RX 7000 or RX 9000 GPU (not the RX 7600 or RX 7600 XT under WSL2, see above).
- Windows 11 with WSL2 and the AMD Adrenalin driver with WSL2 support (AMD's WSL2 guide names Adrenalin 26.2.2 for
  WSL2 with ROCm 7.2). The Windows part of step 1 is AMD's procedure and is not tested here.
- About 45 GB of free disk space (ROCm takes about 22 GB) and at least 8 GB of RAM inside WSL2. WSL2 gets half of the
  PC's memory by default, so that means 16 GB in the PC.

## 1. Set up WSL2 and ROCm

On Windows (not tested by us): install the AMD Adrenalin driver with WSL2 support from AMD. Then open Windows PowerShell
as administrator and install Ubuntu 24.04 in WSL2:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart Windows if asked, then open "Ubuntu 24.04" from the Start menu and create your Linux user.

In the Ubuntu terminal, install ROCm 7.2 and its WSL2 bridge (AMD's `librocdxg`). If you are `root` and `sudo` is
not installed (for example in a container), first run
`apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y sudo tzdata` (the time zone data keeps a later
step from stopping to ask for your time zone).

```sh
sudo apt-get update
sudo apt-get install -y wget ca-certificates
wget https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/noble/amdgpu-install_7.2.4.70204-1_all.deb
sudo apt-get install -y ./amdgpu-install_7.2.4.70204-1_all.deb
sudo apt-get update
sudo apt-get install -y python3-setuptools python3-wheel
sudo apt-get install -y rocm
wget https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb
sudo apt-get install -y ./rocdxg-roct_1.2.2_amd64.deb
echo 'export HSA_ENABLE_DXG_DETECTION=1' >> ~/.bashrc
export HSA_ENABLE_DXG_DETECTION=1
/opt/rocm/bin/rocminfo | grep -E 'Marketing Name|gfx'
```

The last command should list your Radeon and its gfx id (for example `gfx1200` and `AMD Radeon RX 9060 XT`). ROCm
takes about 22 GB of disk (a 7 GB download) and installed in 5 to 32 minutes in our tests, depending on the connection.
Do not install AMD's kernel driver package (`amdgpu-dkms`) in WSL2: WSL2 uses the Windows driver. Under WSL2, `amd-smi` from these packages does not see the GPU; use `rocminfo` as above.

- `rocminfo` prints `HSA_STATUS_ERROR_OUT_OF_RESOURCES`: `HSA_ENABLE_DXG_DETECTION=1` is not set in this terminal;
  run the `export` line again.
- `rocminfo` prints `Cannot load librocdxg.so`: the `rocdxg-roct` package is missing; run its two lines again.

## Install with pip instead of building

After step 1, pip can install the engine, in place of steps 2 to 4:

```sh
sudo apt-get install -y python3-venv
python3 -m venv ~/kitten-tts
. ~/kitten-tts/bin/activate
python -m pip install -U pip                 # pip older than 23.3 skips the "[tts-0.8]" extra with only a warning
pip install "kitten-inference[tts-2-rocm]"
kitten-tts2 --text "Hello" --voice Bruno --output hello.wav
```

The engine package (`kitten-inference-tts2-rocm`, Linux x86_64 with glibc 2.28 or newer) is built with ROCm 7.1 for
gfx1100, gfx1101, gfx1102, gfx1200 and gfx1201. Its audio is the same as that of a `hip` build made with ROCm 7.1 (the
builds our numbers come from); a build made with the ROCm 7.2 of step 1 differs from it only at the level of rounding
(about -60 to -64 dB in our tests), as described above. The package does not include ROCm: it uses `libamdhip64.so.7`
from the ROCm 7.x of step 1 and stops with an error naming that file when ROCm is missing. The first run downloads and
prepares the model (about a 4.6 GB download; 55 to 168 s in our tests); later runs start in a few seconds. On an RX
9070 XT the real-time factor was 0.06 (10 s of speech in about 0.6 s) and the first streamed audio came after 46 to
50 ms. It has no voice cloning.

## 2. Get the source and build

In the Ubuntu terminal:

```sh
sudo apt-get update
sudo apt-get install -y git ca-certificates build-essential cmake pkg-config python3-venv \
  libavformat-dev libavcodec-dev libavutil-dev libswresample-dev
git clone --branch v0.2.0 https://github.com/KittenML/kitten-inference.git
cd kitten-inference/tts-2
python3 tools/build_text_deps.py --prefix build-deps/text
export KT2N_TEXT_DEPS="$PWD/build-deps/text"
cmake --preset hip
cmake --build --preset hip --parallel "$(nproc)"
```

The `hip` preset builds into `build/hip/` for gfx1100, gfx1101, gfx1102, gfx1200 and gfx1201. With 12 to 32 compile jobs this took 4 to 9 minutes and at most about 5 GB of RAM. If the build stops with "Killed" (out of memory), run the last command again with a smaller number, for example `--parallel 4` (and half that if it stops again). To build for one GPU
only, which is faster, add `-DKT2N_HIP_ARCHS=gfx1100` (your GPU's id, as `rocminfo` prints it) to `cmake --preset
hip`. ROCm must be in `/opt/rocm`; otherwise also pass `-DCMAKE_HIP_COMPILER=<rocm>/llvm/bin/clang++` and
`-DCMAKE_PREFIX_PATH=<rocm>` (not tested). The build prints many compiler warnings, most of them `ignoring return value
of type 'hipError_t'`; they are expected.

## 3. Prepare the model

Preparation needs no GPU and no PyTorch, and gives the same files on every machine:

```sh
python3 -m venv .venv-prepare
. .venv-prepare/bin/activate
python -m pip install -r requirements-prepare.txt
python tools/prepare_assets.py --out models/kitten-tts-2
```

This takes 1 to 2 minutes, more than half of it for the download (about 4.6 GB), and at most 4 GB of RAM.

## 4. Generate speech

```sh
build/hip/bin/kitten-tts-native --assets models/kitten-tts-2 \
  --text "Hello" --voice Bruno --output hello.wav
```

This writes `hello.wav`. The tool also prints a `[decoder] config: ...` line and a one-line JSON report on standard
error; in that report `"backend"` reads `hip`. Every option in the
[README](../README.md#usage) works the same way. `KT2N_AMD_WMMA=0` turns the Radeon
matrix instructions off: supported, not optimal, 3.4 to 4.9 times slower (real-time factor 0.23 instead of 0.068 on an
RX 7900 XTX).

## Tests

The tests need steps 1 to 4 first. In the Ubuntu terminal, in the `kitten-inference/tts-2` folder:

```sh
export HSA_ENABLE_DXG_DETECTION=1
. .venv-prepare/bin/activate
export KT2N_TEXT_DEPS="$PWD/build-deps/text"
cmake --preset hip-dev
cmake --build --preset hip-dev --parallel "$(nproc)"
ctest --preset hip-cpu
KT2N_TEST_ASSETS="$PWD/models/kitten-tts-2" ctest --preset hip-gpu
KT2N_TEST_GPU=1 KT2N_TEST_ASSETS="$PWD/models/kitten-tts-2" ctest --preset hip-cpu
```

The last line runs the `hip-cpu` tests again with `KT2N_TEST_GPU=1`, which adds the engine's runtime tests that load
the real model on the GPU; without it they are skipped and still counted as passed. One of them needs about 20 GB of
free GPU memory and is skipped unless that much is free, which in practice means on every card under 24 GB. A few tests need inputs these steps do not set (for
example `KT2N_TOKENIZER_JSON`) and are skipped.

`cmake --preset hip-dev` needs a Python with numpy: activate `.venv-prepare` first, as above. It builds into
`build/hip-dev`, a second full build (4 to 10 minutes).
