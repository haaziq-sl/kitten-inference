# AMD Radeon

Kitten TTS 2 native runs on AMD Radeon RX 7000 (RDNA3) and RX 9000 (RDNA4) GPUs with a ROCm build. These steps set
up ROCm inside WSL2 (the Linux subsystem of Windows), then build and run the engine there. Measured numbers are in
[Consumer GPUs](consumer-gpus.md#amd-radeon).

Tested inside WSL2 on Windows desktops with an RX 7900 XTX, RX 7900 XT, RX 7800 XT, RX 9070 XT and RX 9060 XT. Linux
without WSL2 (bare metal) was not tested.

## What works, and what does not yet

- Works: speech with every built-in voice, streaming and `--stream false`, the voice cache and memory options, saved
  voices (`--load-voice`) and the C API.
- Voice cloning builds, but was not tested on Radeon.
- The audio is close to, not bit for bit the same as, an NVIDIA GPU's; see
  [Output differences](consumer-gpus.md#output-differences). These steps install ROCm 7.2; our measured numbers were
  made with ROCm 7.1 builds. A ROCm 7.2 build differs from a ROCm 7.1 build only at the level of rounding (about
  -65 dB, same words and voice) and runs about 2% slower (real-time factor 0.092 against 0.090 on an RX 9060 XT).
- Cards with less memory: on an RX 9060 XT 16 GB with only 12 GB free (simulated), the tool picks the low-memory
  profile; with 8 GB or 7 GB free (simulated), it fits that profile to the card, as on an 8 GB NVIDIA card. Supported,
  not optimal: the low profile batches several requests at once less, and when fitted it serves one stream and has no
  voice cloning. The audio of complete files is the same, at the same speed (real-time factor 0.095 fitted and 0.095
  with the standard low profile). Real 8 GB Radeons and the RX 7700 XT were not tested.
- Under WSL2, when the low-memory profile fits itself to the card, it keeps at least 1 GiB of GPU memory free: some
  RX 7900 XT and RX 7900 XTX PCs hang when free memory gets below about 0.5 GB. On a simulated 8 GB card this cost
  nothing (the same fitted sizes and the same audio).
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
- Windows 11 with WSL2 and the AMD Adrenalin driver with WSL2 support. AMD's WSL2 guide names Adrenalin 26.2.2 for
  WSL2 with ROCm 7.2; the PCs we ran on had Adrenalin 26.10.41.01 and 26.20.15.07. We could not run the Windows-side
  setup ourselves, so the Windows part of step 1 below is not tested.
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
takes about 22 GB of disk (a 7 GB download) and installed in 5 to 18 minutes in our tests, depending on the connection.
Do not install AMD's kernel driver package (`amdgpu-dkms`) in WSL2: WSL2 uses the Windows driver. Under WSL2, `amd-smi` from these packages does not see the GPU; use `rocminfo` as above.

- `rocminfo` prints `HSA_STATUS_ERROR_OUT_OF_RESOURCES`: `HSA_ENABLE_DXG_DETECTION=1` is not set in this terminal;
  run the `export` line again.
- `rocminfo` prints `Cannot load librocdxg.so`: the `rocdxg-roct` package is missing; run its two lines again.

## 2. Get the source and build

In the Ubuntu terminal:

```sh
sudo apt-get update
sudo apt-get install -y git ca-certificates build-essential cmake pkg-config python3-venv \
  libavformat-dev libavcodec-dev libavutil-dev libswresample-dev
git clone --branch v0.2.0 https://github.com/KittenML/kitten-tts-2-native.git
cd kitten-tts-2-native
python3 tools/build_text_deps.py --prefix build-deps/text
export KT2N_TEXT_DEPS="$PWD/build-deps/text"
cmake --preset hip
cmake --build --preset hip --parallel "$(nproc)"
```

The `hip` preset builds into `build/hip/` for gfx1100, gfx1101, gfx1102, gfx1200 and gfx1201. With 12 to 24 compile jobs this took 4 to 7 minutes and at most about 5 GB of RAM. If the build stops with "Killed" (out of memory), run the last command again with a smaller number, for example `--parallel 4` (and half that if it stops again). To build for one GPU
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

This took 1 to 2 minutes on two home PCs, more than half of it for the download (about 4.6 GB), and at most 4 GB of RAM.

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

The tests need steps 1 to 4 first. In the Ubuntu terminal, in the `kitten-tts-2-native` folder:

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
`build/hip-dev`, a second full build (4 to 6 minutes).
