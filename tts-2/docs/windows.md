# Windows

Kitten TTS 2 native builds and runs natively on 64-bit Windows with an NVIDIA GPU. These steps take a Windows machine with an NVIDIA GPU and its driver to a first WAV file.

Tested on Windows Server 2022 with an NVIDIA L4 (a 24 GB data-centre GPU run with the same Windows driver model as GeForce cards). Windows 10 (version 1903 or newer) and Windows 11 with a GeForce card should work the same way but have not been tested. Measured numbers are in [Consumer GPUs](consumer-gpus.md#windows).

## What works, and what does not yet

- Works: streaming and `--stream false`, all built-in voices, the voice cache and memory options, WAV or PCM output to a file or to standard output, Unicode text and non-ASCII file and folder names, UTF-8 text files with or without a byte order mark (UTF-16 files are refused), and the C API (`kt2n_engine.dll`, `include/kt2n/engine.h`).
- Voice cloning: with the pip package (`pip install "kitten-inference[tts-2]"`), which decodes the recording with the FFmpeg program; built; passes an automated check, not tested by hand. The source build on this page leaves it out; see [Voice cloning](#voice-cloning).
- Not yet: the MP3, AAC, Opus and FLAC encoders, and the C examples in `examples/`. `kt2n_event_fd()` returns -1 on Windows: wait in `kt2n_poll()` with a timeout instead.

## Requirements

- 64-bit Windows 10 (version 1903 or newer), Windows 11 or Windows Server 2022.
- An NVIDIA GPU of the groups listed in the [README](../README.md#supported-gpus), with at least 4 GB of memory.
- The NVIDIA driver for your GPU, version 570 or newer. To check, open a Command Prompt and run `nvidia-smi`: it should list your GPU and the driver version. If it is not found, install the driver from NVIDIA first (not covered here).
- About 40 GB of free disk space (45 GB with the tests) and at least 8 GB of RAM.

The steps below install everything else: Visual Studio 2022 Build Tools (the C++ compiler, CMake and Ninja), the CUDA Toolkit 12.8, Python 3.12 and Git.

## 1. Install the build tools

Open **Windows PowerShell as administrator** (Start menu: right-click Windows PowerShell, "Run as administrator") and run:

```powershell
$ProgressPreference = 'SilentlyContinue'
New-Item -ItemType Directory -Force C:\kt2-downloads | Out-Null
Set-Location C:\kt2-downloads
curl.exe -fsSL -o vs_buildtools.exe https://aka.ms/vs/17/release/vs_buildtools.exe
(Start-Process -Wait -PassThru -FilePath .\vs_buildtools.exe -ArgumentList '--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.CMake.Project --includeRecommended').ExitCode
curl.exe -fsSL -o cuda_12.8.1_windows.exe https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_572.61_windows.exe
(Start-Process -Wait -PassThru -FilePath .\cuda_12.8.1_windows.exe -ArgumentList '-s cudart_12.8 nvcc_12.8 nvfatbin_12.8 nvjitlink_12.8 cuobjdump_12.8 nvdisasm_12.8 cuxxfilt_12.8 nvprune_12.8 nvrtc_12.8 nvrtc_dev_12.8 nvtx_12.8 nvml_dev_12.8 cupti_12.8 sanitizer_12.8 curand_12.8 curand_dev_12.8 cublas_12.8 cublas_dev_12.8 thrust_12.8 cuda_profiler_api_12.8 visual_studio_integration_12.8 opencl_12.8 -n').ExitCode
curl.exe -fsSL -o python-3.12.10-amd64.exe https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe
(Start-Process -Wait -PassThru -FilePath .\python-3.12.10-amd64.exe -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1').ExitCode
curl.exe -fsSL -o Git-2.47.1-64-bit.exe https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/Git-2.47.1-64-bit.exe
(Start-Process -Wait -PassThru -FilePath .\Git-2.47.1-64-bit.exe -ArgumentList '/VERYSILENT /NORESTART').ExitCode
```

This downloads about 7 GB and takes about 10 to 30 minutes, depending on your connection. Each installer runs without windows and returns when it is done, and its line prints the installer's exit code: `0` means it worked (`3010` from Visual Studio also means it worked and asks for a restart later). Any other number means that installer failed; run its line again. The CUDA Toolkit is installed without its display driver, so your driver stays as it is. Visual Studio must be installed before the CUDA Toolkit, as above. If you already have Visual Studio 2022 with "Desktop development with C++", skip its two lines.

Close PowerShell when it is done.

## 2. Get the source and build

Open a new **Command Prompt** (not as administrator; Start menu: "Command Prompt"), so that it sees the newly installed programs. Load the compiler's environment, get the source and build:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
git clone --branch v0.2.0 https://github.com/KittenML/kitten-inference.git
cd kitten-inference\tts-2
python tools\build_text_deps.py --prefix build-deps\text
set KT2N_TEXT_DEPS=%CD%\build-deps\text
cmake --preset windows
cmake --build --preset windows --parallel
```

If you have the full Visual Studio 2022 instead of the Build Tools, open "x64 Native Tools Command Prompt for VS 2022" from the Start menu instead of the `call` line.

This builds `kitten-tts-native.exe` and `kt2n_engine.dll` in `build\windows\bin`; keep the two files together. The
build prints a few hundred compiler warnings (most of them `floating-point value does not fit`); they are expected. It builds code for every supported NVIDIA GPU, which takes about 20 minutes on 8 cores. `--parallel` runs about one compile job per CPU thread; if the computer runs out of memory, use `--parallel 4` instead.

## 3. Prepare the model

In the same Command Prompt:

```bat
py -3.12 -m venv .venv-prepare
call .venv-prepare\Scripts\activate.bat
python -m pip install -r requirements-prepare.txt
python tools\prepare_assets.py --out models\kitten-tts-2
```

This downloads the latest revision of the model (about 4.6 GB) and converts the model into `models\kitten-tts-2` (about 2 GB); how long it takes depends on your connection. The tool may print warnings that Hugging Face downloads cannot use symlinks on this machine and about unauthenticated
requests; both are harmless. Ignore the `Next: kitten-tts-native ...` line it prints at the end and use step 4. You can also copy a `models\kitten-tts-2` folder prepared on another computer with the same version of this repository instead (Linux or Windows).

## 4. Generate speech

```bat
build\windows\bin\kitten-tts-native.exe --assets models\kitten-tts-2 --text "Hello" --voice Bruno --output hello.wav
```

This writes `hello.wav`. Every option in the [README](../README.md#usage) works the same way; write paths with `\`.

## Tests

The tests need a Python with numpy: the `.venv-prepare` environment from step 3 has it. In a new Command Prompt, in the `kitten-inference\tts-2` folder, first restore the environment of steps 2 and 3:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set KT2N_TEXT_DEPS=%CD%\build-deps\text
call .venv-prepare\Scripts\activate.bat
```

Then:

```bat
cmake --preset windows-dev
cmake --build --preset windows-dev --parallel
ctest --preset windows-cpu
set KT2N_TEST_ASSETS=%CD%\models\kitten-tts-2
ctest --preset windows-gpu
set KT2N_TEST_GPU=1
ctest --preset windows-cpu
```

- `windows-dev` builds into `build\windows-dev`: a second full build, about 20 minutes on 8 cores and about 4 GB more
  disk.
- The `windows-gpu` tests need the prepared model in `KT2N_TEST_ASSETS`; without it they fail.
- The last line runs the `windows-cpu` tests again with `KT2N_TEST_GPU=1`, which adds the engine's runtime tests that
  load the real model on the GPU; without it they are skipped and still counted as passed.
- Two tests need inputs these steps do not set, and are skipped: `Text.TokenizerAndPrompt` and
  `GpuSmoke.RealEngineFileUnpackMatchesCpu`. One GPU runtime test needs about 21.5 GB (20 GiB) of free GPU memory, so
  it is skipped on every card below 24 GB; on 8 GB cards one more is skipped. Skipped tests count as passed.

## Running on another computer

Copy `kitten-tts-native.exe`, `kt2n_engine.dll` and the `models\kitten-tts-2` folder. The other computer needs the NVIDIA driver (570 or newer) and the Microsoft Visual C++ Redistributable for Visual Studio 2015-2022 (x64) from Microsoft, but no CUDA Toolkit. This was not tested.

## Voice cloning

The pip package builds voices on Windows with the FFmpeg program (`winget install --id Gyan.FFmpeg -e`; built; passes an automated check, not tested by hand). The `windows` preset on this page builds without voice cloning. A cloning build compiles with a shared FFmpeg 8.1 build for Windows: add `-DKT2N_BUILD_CLONING=ON -DKT2N_VOICE_AV=ON -DCMAKE_PREFIX_PATH=<FFmpeg folder>` to `cmake --preset windows`, and copy FFmpeg's `bin\*.dll` files next to `kitten-tts-native.exe`. Building a voice that way has not been tested on Windows.

## Tips

- On a card with little memory, set "CUDA - Sysmem Fallback Policy" to "Prefer No Sysmem Fallback" for `kitten-tts-native.exe` in the NVIDIA Control Panel. Otherwise Windows may move GPU memory to system memory when the card is full, and speech gets much slower instead of stopping with an out-of-memory error.
- Under Windows, some display drivers take extra GPU memory once a program has written about 2 GB (about 400 MiB on an RTX 3050 Laptop GPU; none on a GTX 1650). The engine measures this when it starts and fits itself to what is left, so on a 4 GB card it can need up to about 3,000 MiB free rather than 2,583 MiB. If a start is refused because the desktop took memory at the same moment, start again.
- Text typed at the console for `--text-file -` is read in the console's code page; pipe or redirect a UTF-8 file instead.
- To pipe text from Windows PowerShell 5.1, first run `$OutputEncoding = [Text.UTF8Encoding]::new($false)`; otherwise PowerShell turns non-ASCII characters into `?`.
- Programs that load `kt2n_engine.dll` should pass UTF-8 paths and use a UTF-8 code page (or keep paths ASCII), and run very long unpunctuated text on threads with 8 MB stacks (Windows' default 1 MB can run out).
- Only the Ninja presets above (`windows`, `windows-dev`) have been tested.
