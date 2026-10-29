# Windows

Kitten TTS 2 native builds and runs natively on 64-bit Windows with an NVIDIA GPU. These steps take a Windows machine with an NVIDIA GPU and its driver to a first WAV file. Speed is in [Consumer GPUs](consumer-gpus.md#windows).

## What works, and what does not yet

- Works: streaming and `--stream false`, all built-in voices, the voice cache and memory options, WAV or PCM output to a file or to standard output, Unicode text and non-ASCII file and folder names, UTF-8 text files with or without a byte order mark (UTF-16 files are refused), and the C API (`kt2n_engine.dll`, `include/kt2n/engine.h`).
- Voice cloning: with the pip package (`pip install "kitten-inference"`), which decodes the recording with the FFmpeg program. The source build on this page leaves it out; see [Voice cloning](#voice-cloning).
- Not yet: the MP3, AAC, Opus and FLAC encoders, and the C examples in `examples/`. `kt2n_event_fd()` returns -1 on Windows: wait in `kt2n_poll()` with a timeout instead.

## Install with pip instead of building

With Python 3.9 to 3.14 from python.org (3.15 also needs the Visual C++ Build Tools until PyYAML ships ready-made 3.15 packages), the NVIDIA driver 570 or newer and no other tools, in a Command Prompt:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
pip install "kitten-inference"
kitten-tts --model KittenML/kitten-tts-2 --text "Hello" --voice Bruno --output hello.wav
```

The first run downloads about 4.6 GB and prepares the model, in about 1 to 3 minutes; later runs reuse it. Seeded complete files are byte for byte the same as on Linux with the same GPU model. The Python API is in [Python API](../../../docs/api.md).

## Requirements

- 64-bit Windows 10 (version 1903 or newer), Windows 11 or Windows Server 2022.
- An NVIDIA GPU of the groups listed in the [README](../README.md#supported-gpus), with at least 4 GB of memory.
- The NVIDIA driver for your GPU, version 570 or newer. To check, open a Command Prompt and run `nvidia-smi`: it should list your GPU and the driver version. If it is not found, install the driver from NVIDIA first (not covered here).
- About 40 GB of free disk space and at least 8 GB of RAM.

The steps below install everything else: Visual Studio 2022 Build Tools (the C++ compiler, CMake and Ninja), the CUDA Toolkit 12.8, Python 3.12 and Git.

## 1. Install the build tools

Open **Windows PowerShell as administrator** (Start menu: right-click Windows PowerShell, "Run as administrator") and run:

```powershell
$ProgressPreference = 'SilentlyContinue'
$ErrorActionPreference = 'Stop'
# A private folder for this run, under your own profile. Do NOT use a fixed name under C:\ (for example
# C:\kt2-downloads): a folder made directly under C:\ can be read and changed by every signed-in user, so on a
# shared machine someone could swap an installer for their own before it runs with your administrator rights.
$d = (New-Item -ItemType Directory (Join-Path $env:TEMP ([guid]::NewGuid()))).FullName

function Get-Installer($url, $name) {
  $f = Join-Path $d $name
  curl.exe -fsSL -o $f $url
  if ($LASTEXITCODE -ne 0) { throw "download failed: $url" }
  $sig = Get-AuthenticodeSignature $f
  if ($sig.Status -ne 'Valid') { throw "$name is not validly signed ($($sig.Status)); do not run it" }
  Write-Host "$name signed by: $($sig.SignerCertificate.Subject)"
  return $f
}

$vs = Get-Installer https://aka.ms/vs/17/release/vs_buildtools.exe vs_buildtools.exe
(Start-Process -Wait -PassThru -FilePath $vs -ArgumentList '--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.CMake.Project --includeRecommended').ExitCode
$cuda = Get-Installer https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_572.61_windows.exe cuda_12.8.1_windows.exe
(Start-Process -Wait -PassThru -FilePath $cuda -ArgumentList '-s cudart_12.8 nvcc_12.8 nvfatbin_12.8 nvjitlink_12.8 cuobjdump_12.8 nvdisasm_12.8 cuxxfilt_12.8 nvprune_12.8 nvrtc_12.8 nvrtc_dev_12.8 nvtx_12.8 nvml_dev_12.8 cupti_12.8 sanitizer_12.8 curand_12.8 curand_dev_12.8 cublas_12.8 cublas_dev_12.8 thrust_12.8 cuda_profiler_api_12.8 visual_studio_integration_12.8 opencl_12.8 -n').ExitCode
$py = Get-Installer https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe python-3.12.10-amd64.exe
(Start-Process -Wait -PassThru -FilePath $py -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1').ExitCode
$git = Get-Installer https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/Git-2.47.1-64-bit.exe Git-2.47.1-64-bit.exe
(Start-Process -Wait -PassThru -FilePath $git -ArgumentList '/VERYSILENT /NORESTART').ExitCode
```

This downloads about 7 GB and takes about 10 to 30 minutes, depending on your connection. Each installer is checked for a valid Authenticode signature before it runs (the line prints who signed it); a download that fails or a file that is not validly signed stops the script. Each installer's line prints its exit code: `0` means it worked (`3010` from Visual Studio also means it worked and asks for a restart later); any other number means that installer failed, so run the script again. The CUDA Toolkit is installed without its display driver, so your driver stays as it is. Visual Studio must be installed before the CUDA Toolkit, as above. If you already have Visual Studio 2022 with "Desktop development with C++", skip its two lines.

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

This builds `kitten-tts-native.exe` and `kt2n_engine.dll` in `build\windows\bin`; keep the two files together. The build prints a few hundred compiler warnings (most of them `floating-point value does not fit`); they are expected. It builds code for every supported NVIDIA GPU, which takes about 20 minutes on 8 cores. If the computer runs out of memory, use `--parallel 4` instead of `--parallel`.

## 3. Prepare the model

In the same Command Prompt:

```bat
py -3.12 -m venv .venv-prepare
call .venv-prepare\Scripts\activate.bat
python -m pip install -r requirements-prepare.txt
python tools\prepare_assets.py --out models\kitten-tts-2
```

This downloads the latest revision of the model (about 4.6 GB) and converts it into `models\kitten-tts-2` (about 2 GB). Warnings that Hugging Face downloads cannot use symlinks on this machine, and about unauthenticated requests, are harmless. Ignore the `Next: kitten-tts-native ...` line it prints at the end and use step 4. You can also copy a `models\kitten-tts-2` folder prepared on another computer (Linux or Windows) with the same version of this repository instead.

## 4. Generate speech

```bat
build\windows\bin\kitten-tts-native.exe --assets models\kitten-tts-2 --text "Hello" --voice Bruno --output hello.wav
```

This writes `hello.wav`. Every option in the [README](../README.md#usage) works the same way; write paths with `\`.

## Running on another computer

Experimental: copy `kitten-tts-native.exe`, `kt2n_engine.dll` and the `models\kitten-tts-2` folder. The other computer needs the NVIDIA driver (570 or newer) and the Microsoft Visual C++ Redistributable for Visual Studio 2015-2022 (x64) from Microsoft, but no CUDA Toolkit.

## Voice cloning

The pip package builds voices on Windows with the FFmpeg program: `winget install --id Gyan.FFmpeg -e`, then open a new Command Prompt and run `.venv\Scripts\activate` again. Windows 10 and 11 include winget; on Windows Server 2022, install winget (App Installer) first. Voices built on Windows match those built on Linux for WAV and MP3 recordings; with an M4A recording, FFmpeg 9 (what winget installs today) can give a slightly different voice file. The `windows` preset on this page builds without voice cloning.

## Tips

- On a card with little memory, set "CUDA - Sysmem Fallback Policy" to "Prefer No Sysmem Fallback" for `kitten-tts-native.exe` in the NVIDIA Control Panel. Otherwise Windows may move GPU memory to system memory when the card is full, and speech gets much slower instead of stopping with an out-of-memory error.
- On a 4 GB card the engine can need up to about 3,000 MiB of free GPU memory rather than 2,583 MiB, because some Windows display drivers take extra GPU memory once a program has written about 2 GB; the engine measures this when it starts. If a start is refused because the desktop took memory at the same moment, start again.
- Text typed at the console for `--text-file -` is read in the console's code page; pipe or redirect a UTF-8 file instead.
- To pipe text from Windows PowerShell 5.1, first run `$OutputEncoding = [Text.UTF8Encoding]::new($false)`; otherwise PowerShell turns non-ASCII characters into `?`.
- Programs that load `kt2n_engine.dll` should pass UTF-8 paths and use a UTF-8 code page (or keep paths ASCII), and run very long unpunctuated text on threads with 8 MB stacks (Windows' default 1 MB can run out).
