# Kitten TTS 2 native

A fast C++/CUDA inference engine for [Kitten TTS 2](https://huggingface.co/KittenML/kitten-tts-2) on NVIDIA GPUs, with a ROCm build for AMD Radeon RX 7000 and RX 9000. **On an RTX PRO 6000 it streams its first audio after about 11 ms and makes complete files 5 to 7 times faster than Kitten TTS 2 on vLLM or llama.cpp. On smaller GPUs the lead is smaller, but it is still 2.4 times or more on a 12 GB RTX 3060.** It ships as a C library and a command-line tool, and needs no Python, PyTorch or network connection at run time. From Python, install it with pip instead: see [python/README.md](python/README.md).

- Built-in voices, expression tags and vocal events
- Voice cloning from a short recording, with saved, reusable voices
- Streaming output, or complete files with `--stream false`
- A C API with cancellation and incremental text input
- Starts with no options on GPUs from 4 GB up: it picks its memory settings from the free GPU memory

This page covers Linux with an NVIDIA GPU. For Windows see [docs/windows.md](docs/windows.md); for AMD Radeon see [docs/amd-radeon.md](docs/amd-radeon.md). Speed, memory and audio quality on each GPU group are in [docs/consumer-gpus.md](docs/consumer-gpus.md).

## Performance

One request at a time on an NVIDIA RTX PRO 6000 (96 GB) with the current model, through the pip package with its default settings (the bundled command-line tool gives the same within 2%). Every engine speaks the same texts with the same voice and seeds and returns the complete audio; medians of 300 short and 12 long texts.

| Engine | Short text (one chunk) | Long text (about three chunks) | GPU memory |
|---|---:|---:|---:|
| **Kitten TTS 2 native** | **82 ms** | **0.95 s** | 12.8 GiB (default profile, one voice) |
| Kitten TTS 2 on vLLM, 4-bit weights | 453 ms (5.5x) | 4.9 s (5.1x) | 49.5-51.9 GiB (vLLM's default takes half the GPU) |
| Kitten TTS 2 on vLLM, bf16 (its default) | 584 ms (7.2x) | 6.6 s (6.9x) | 49.5-51.4 GiB |
| Kitten TTS 2 on llama.cpp (GPU) | 520 ms (6.4x) | 5.4 s (5.6x) | 2.7-4.8 GiB |

When streaming, the native engine sends its first audio after **11.4 ms** (median; 11.6 ms at the 95th percentile, 300 requests). vLLM and llama.cpp stream one whole sentence at a time, so on a short text their first audio comes with the complete file (481 to 643 ms). Word error rate is 0.02 to 0.05% for all four on the short texts. vLLM ran through `kittenml[vllm]` 0.9.6. The default profile reserves GPU memory up front so that it can serve several streams at once; see [Using less GPU memory](#using-less-gpu-memory).

On smaller GPUs the same comparison gives a smaller lead, because the native engine's speed follows the GPU's memory speed while vLLM and llama.cpp lose a roughly fixed time per token on the processor: complete files 4.5 to 6.9 times faster and first audio after 22 ms on an RTX 4090, 2.4 to 4.3 times and 58 ms on a 12 GB RTX 3060 (with the low-memory profile, which the tool picks there by itself). Tables: [docs/consumer-gpus.md](docs/consumer-gpus.md#against-vllm-and-llamacpp).

## Supported GPUs

| GPU group | Status |
|---|---|
| NVIDIA GeForce RTX 30 series and newer, and other NVIDIA GPUs with compute capability 8.0 or newer | supported |
| NVIDIA GeForce GTX 10, GTX 16 and RTX 20 series (compute capability 6.1 and 7.5) | supported, not optimal: these GPUs have no bf16 tensor cores, so part of the math runs in software. Real-time factor 0.22 to 0.76 with `--stream false` on the cards we measured, against 0.03 to 0.17 for RTX 30 series and newer. The audio is close to, not bit for bit the same as, a newer GPU's |
| AMD Radeon RX 7000 and RX 9000 | supported with a separate build, tested inside WSL2 on Windows (RX 7600 and RX 7600 XT: builds, not tested); no voice cloning; the audio is close to, not bit for bit the same as, an NVIDIA GPU's. See [docs/amd-radeon.md](docs/amd-radeon.md) |
| Other GPUs: NVIDIA compute capability 6.0, 7.0 or older; AMD RX 6000 and older, Instinct | not supported: the engine stops at start and says so |

A real-time factor of 0.25 means 10 seconds of audio take 2.5 seconds to produce.

### GPU memory

`kitten-tts-native` and the pip package pick their memory settings from the free GPU memory, so they start without options on supported GPUs with 4 GB or more. Smaller GPUs give up some speed or features:

| GPU memory | What you get |
|---|---|
| 24 GB or more | Everything, at full speed. |
| 10 to 16 GB | When the default profile does not fit (about 13,500 MiB with one voice), the tool uses the low-memory profile (about 9,100 MiB): supported, not optimal, because several requests at once share smaller buffers. One request at a time runs as fast (real-time factor 0.050 against 0.051 on a 16 GB card). Building a cloned voice needs about 11,300 MiB of free GPU memory with the low-memory profile (15,600 MiB with the default one); it was tested end to end on a 16 GB card. On 12 GB cards it fits when other programs use at most about 500 MiB of GPU memory; on 10 GB cards it does not fit. |
| 4 to 8 GB | Supported, not optimal: one request at a time, and the tool fits its buffers to the card, which makes complete files (`--stream false`) about 1.5% slower. On 4 GB cards, long streaming text is cut into shorter pieces, so the audio can differ from a larger GPU's (on long texts, word error rate 1.43% against 0.58% without the cut). No voice cloning: use a voice saved on a larger GPU. |
| 3 GB | Not tested. |
| 2 GB | Not supported: the engine needs about 2,583 MiB of free GPU memory and says so at start. |

The model comes from the latest revision on Hugging Face, which has 115 built-in voices. The figures on this page were measured with it, except the speed and word error rate figures in [Supported GPUs](#supported-gpus) and [GPU memory](#gpu-memory) (from [docs/consumer-gpus.md](docs/consumer-gpus.md)), which were measured with an earlier revision with 47 voices. The tool's own choices above use only the voice it speaks with, so the voice count matters mainly when all voices are prepared (`--voice-cache all`).

## Requirements

- Linux on x86-64. The commands below are for Ubuntu 24.04; other distributions need the same packages under their own names (not tested).
- A supported GPU with at least 4 GB of memory (see above) and the NVIDIA driver R570 or newer. Running needs only the driver.
- About 30 GB of free disk space and at least 8 GB of RAM.

The Quick start installs everything else: the CUDA Toolkit 12.8, GCC, CMake, `pkg-config`, Python and the FFmpeg development libraries (for voice cloning).

## Quick start

These steps take a fresh Ubuntu 24.04 machine with an NVIDIA GPU and its driver to a first WAV file. Run them in one terminal, in order; in a new terminal, run the `export` lines of step 2 again.

If you are `root` and `sudo` is not installed (for example in a container), install it first, together with the time zone data, so that no later step stops to ask for your time zone:

```sh
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y sudo tzdata
```

### 1. Get the source

```sh
sudo apt-get update
sudo apt-get install -y git ca-certificates wget
git clone --branch v0.2.0 https://github.com/KittenML/kitten-inference.git
cd kitten-inference/tts-2
```

Run every later command in this `kitten-inference/tts-2` folder.

### 2. Build

Check that the NVIDIA driver works:

```sh
nvidia-smi
```

It should list your GPU and a driver version of 570 or newer (under WSL2 on Windows, the Windows driver's version is shown as `KMD Version`). If the command is not found, install the NVIDIA driver for your GPU first and restart the computer (not covered here; on Ubuntu, "Software & Updates" > "Additional Drivers" offers it).

Install the CUDA Toolkit 12.8 from NVIDIA's package repository, and the other build tools:

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
sudo apt-get install -y build-essential cmake pkg-config python3-venv \
  libavformat-dev libavcodec-dev libavutil-dev libswresample-dev
export PATH=/usr/local/cuda-12.8/bin:$PATH
nvcc --version
```

`nvcc --version` should print `release 12.8`. We test and release with CUDA 12.8 (12.9 also builds every GPU, not tested). CUDA 13 also builds, with three differences: the build leaves out GeForce GTX 10 series GPUs (CMake prints a warning, and such a GPU is refused at start), it needs NVIDIA driver 580 or newer, and its audio is not bit for bit the same as the CUDA 12.8 build's, at the same quality (word error rate 1.39% against 1.42% on 72 test requests, RTX 5060).

Build the text-normalization libraries (pinned copies of OpenFst and PCRE2, about a minute) and the engine. The last command runs one compile job per CPU thread (`--parallel "$(nproc)"`); a single job uses up to 2.7 GB of RAM. Before you run it, check two things, and if either holds, type `4` instead of `"$(nproc)"` (the build then takes about 20 minutes; with 4 jobs our build used at most 3.8 GB of RAM in all):

- the machine has less than 16 GB of RAM (`free -g` shows 14 or less under `total`; a 16 GB machine shows 15);
- you are in a container that may use fewer CPUs than `nproc` prints (`cat /sys/fs/cgroup/cpu.max` shows the limit, if set: the first number divided by the second is the number of CPUs).

```sh
python3 tools/build_text_deps.py --prefix build-deps/text
export KT2N_TEXT_DEPS="$PWD/build-deps/text"
cmake --preset default
cmake --build --preset default --parallel "$(nproc)"
```

The build writes everything to `build/default/`. It contains code for every supported NVIDIA GPU and includes voice cloning, and takes 10 to 15 minutes on machines with 16 to 48 CPU threads. If the build stops with "Killed" (out of memory), run the last command again with half the number of jobs you used (for example `--parallel 2` after `--parallel 4`). To build only for your GPU, which is faster, add `-DKT2N_CUDA_ARCHS=86` to `cmake --preset default`, with your GPU's compute capability without the dot.

### 3. Prepare the model

```sh
python3 -m venv .venv-prepare
. .venv-prepare/bin/activate
python -m pip install -r requirements-prepare.txt
python tools/prepare_assets.py --out models/kitten-tts-2
```

This downloads about 4.6 GB from Hugging Face, GitHub and PyPI (the latest revision of the model and its voices, and the text-normalization data) and converts them once into `models/kitten-tts-2` (about 2 GB). `--revision` pins another model revision. No Hugging Face account is needed: you can ignore the warning about unauthenticated requests to the Hugging Face Hub.

- The last line the tool prints, `Next: kitten-tts-native ...`, assumes an installed tool. Before you install it ([below](#use-the-library-in-your-own-program)), use the path in step 4.

### 4. Generate speech

```sh
build/default/bin/kitten-tts-native --assets models/kitten-tts-2 \
  --text "Hello" --voice Bruno --output hello.wav
```

The output is mono 16-bit audio at 24 kHz, and generation needs no network connection. On standard error the tool prints a `[decoder] config: ...` line, then a one-line JSON report with the time to first audio and the real-time factor. On GPUs where the default memory profile does not fit (most cards under 16 GB) it also prints one line saying it uses the low-memory profile, with the free memory and the default profile's need on this card; that is expected. On 4 GB cards long streaming text is also cut into shorter pieces (see [GPU memory](#gpu-memory)).

## Usage

### Voices and expression

```sh
build/default/bin/kitten-tts-native --assets models/kitten-tts-2 --list-voices
```

Pass a voice name to `--voice`. Read the text from a file with `--text-file FILE` (`-` for standard input).

The text can carry an expression tag, vocal events and emphasis, for example `[joyful] That is wonderful <laugh> (((really)))`. Expression tags: `[mundane]`, `[nervous]`, `[tender]`, `[angry]`, `[excited]`, `[stern]`, `[sad]`, `[contemplative]`, `[surprised]`, `[joyful]`. Vocal events: `<pause>`, `<sigh>`, `<gasp>`, `<laugh>`, `<giggle>`, `<sob>`, `<scoff>`, `<growl>`, `<um>`, `<gulp>`.

### Streaming and complete files

Audio streams by default. To get raw 16-bit PCM at 24 kHz as it arrives, use `--format pcm`:

```sh
build/default/bin/kitten-tts-native --assets models/kitten-tts-2 --voice Bruno \
  --text "Streaming speech." --format pcm --output stream.pcm
```

With `--output -` the audio goes to standard output. On a computer with sound output, you can play it as it arrives with `aplay` from the `alsa-utils` package:

```sh
build/default/bin/kitten-tts-native --assets models/kitten-tts-2 --voice Bruno \
  --text "Streaming speech." --format pcm --output - | aplay -f S16_LE -r 24000 -c 1
```

`--stream false` renders complete files instead: the text is split at sentence boundaries and each chunk is generated as a whole.

```sh
build/default/bin/kitten-tts-native --assets models/kitten-tts-2 --voice Bruno \
  --text "A complete passage with natural pauses. It has two sentences." \
  --stream false --output complete.wav
```

### Options

- `--preset stable|expressive`, `--seed N`, and `--temperature`, `--top-p`, `--top-k`, `--min-p`, `--max-tokens` control sampling. With `--stream false` and a fixed seed, the same input gives the same audio on the same GPU. Streamed audio with the same seed can differ between runs, because where it is cut into decoder chunks depends on timing.
- `--device N` selects the GPU; `--max-streams N` lets the engine run several requests at once.
- `--memory-profile` and `--voice-cache`: see [Using less GPU memory](#using-less-gpu-memory).
- `--report FILE` writes the run report (time to first audio, total time, real-time factor) as JSON.
- `--check-assets` validates the prepared model directory; `--version` prints the version and build information; `--help` lists every option.

Ctrl-C cancels a running request; the tool then exits with status 130. The exit status is 0 on success, 1 on an error (the tool prints the reason and the engine's status number, for example `(status 111)`) and 2 for an invalid command line.

## Voice cloning

Cloning needs the `default` build from the Quick start, extra model files, and about 11,300 MiB (11 GB) of free GPU memory while the voice is built. Building, saving and loading a cloned voice was tested end to end on 16 GB and 24 GB cards. On 12 GB cards it fits when other programs use at most about 500 MiB of GPU memory (measured with the Python package: building worked with 11,285 MiB free and failed with 11,157 MiB free). On GPUs with 8 GB or less, cloning is off: the engine starts, and building a voice fails with a message that says so. On those cards and on 10 GB cards, build and save the voice on a larger GPU, then load it.

In the repository folder after the Quick start, add the cloning files to the prepared model once, in the same environment (no Hugging Face account needed):

```sh
. .venv-prepare/bin/activate
python tools/prepare_assets.py --out models/kitten-tts-2 --cloning-only
```

Clone a voice from a 2 to 30 second recording (`reference.wav` stands for yours; any format FFmpeg can decode, at most 10 MiB) and its exact transcript, and save it for later:

```sh
build/default/bin/kitten-tts-native --assets models/kitten-tts-2 \
  --reference-audio reference.wav \
  --reference-transcript "The exact words spoken in the recording." \
  --save-voice my-voice.voice --text "New speech in this voice." --output cloned.wav
build/default/bin/kitten-tts-native --assets models/kitten-tts-2 \
  --load-voice my-voice.voice \
  --text "Using the saved voice." --output saved.wav
```

Building a cloned voice needs about 15,600 MiB of GPU memory with the default profile; with less, the tool uses the low-memory profile for it, which needs about 11,300 MiB (the audio is the same). A saved voice works only with the same prepared model: the same model revision prepared by this version on any machine, or a copy of the same folder. A voice saved with a model prepared by an earlier version does not load with a model prepared by this version: keep that model folder for it, or build and save the voice again. Loading a saved voice needs neither the cloning files nor Hugging Face access. Speaking with it needs about 13,500 MiB of GPU memory with the default profile and 9,200 MiB with the low-memory one; on smaller cards the tool fits its buffers to the card, as with built-in voices. Without a `--voice-cache` option the tool keeps only the voice it speaks with on the GPU, for built-in, cloned and loaded voices alike, so loading needs no extra option on small cards.

## Using less GPU memory

`kitten-tts-native` picks working settings by itself (see [GPU memory](#gpu-memory)); the options below are for several streams, the C API, and using less memory. They can be combined and do not change the audio of `--stream false` requests.

- `--memory-profile auto|default|low`: without the option the tool uses `auto`, which takes the default profile when it fits the free GPU memory and `low` otherwise, and prints one line when it picks `low`. `low` sizes the engine's buffers for the requests it actually serves, and fits itself to cards with about 8 GB or less (supported, not optimal: one stream, `--stream false` about 1.5% slower). An explicit `--memory-profile default` on a GPU too small for it stops at start and says how much it needs.
- `--voice-cache all|lru|none` keeps all built-in voices on the GPU, only the recently used ones, or only the ones in use. Without the option, the tool prepares only the voice it speaks with. `--voice-cache-voices N` and `--voice-cache-mib N` set the cache size and need an explicit `--voice-cache lru` or `none`. Plain `--voice-cache lru` sets aside room for 16 voices (about 3,000 MiB), which does not fit on 4 GB cards.
- In the C API, set `memory_profile` to `KT2N_MEMORY_PROFILE_AUTO` for the same choice; the C API defaults are the default profile and all voices prepared, which need about 33,200 MiB with the 115 voices of the current model, more than a 24 GB card has.

[docs/memory.md](docs/memory.md) explains each setting, the small-GPU fit and the C API fields; [docs/consumer-gpus.md](docs/consumer-gpus.md) has the measured numbers.

## C API and examples

The C API is in [include/kt2n/engine.h](include/kt2n/engine.h): engine setup, streaming and complete requests, cancellation, incremental text sessions, voice cloning, and saving and loading voices. Create the engine once and reuse it to keep the model loaded.

| Example | Shows |
|---|---|
| [stream.c](examples/stream.c) | streaming audio with a built-in voice |
| [complete.c](examples/complete.c) | a complete file |
| [clone.c](examples/clone.c) | building, saving and loading a cloned voice |
| [session.c](examples/session.c) | incremental text input |
| [voice_cache.c](examples/voice_cache.c) | keeping only some voices on the GPU |

The examples build into `build/default/examples/` with the Quick start. They use the automatic memory profile and keep only the voice they speak with on the GPU, so they start on GPUs with 4 GB or more. [examples/README.md](examples/README.md) shows how to run them.

### Use the library in your own program

Install the library, its header, the CMake package files and `kitten-tts-native` into `$HOME/.local`:

```sh
cmake --install build/default --prefix "$HOME/.local"
```

In your own CMake project, find the package and link it:

```cmake
find_package(kt2n CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE kt2n::engine)
```

CMake does not search `$HOME/.local` by itself, so pass the prefix when you configure your project. For example, to build the examples on their own against the installed package:

```sh
cmake -S examples -B build-examples -DCMAKE_PREFIX_PATH="$HOME/.local"
cmake --build build-examples
build-examples/complete models/kitten-tts-2 Bruno "Hello from the installed library." installed.wav
```

To run the installed `kitten-tts-native` by name, `$HOME/.local/bin` must be on your `PATH`:

```sh
export PATH="$HOME/.local/bin:$PATH"
kitten-tts-native --version
```

## Contributing

Running the tests, offline preparation and troubleshooting are covered in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

The code is licensed under [Apache-2.0](LICENSE); some components keep their own licenses (see [NOTICE](NOTICE) and [licenses/](licenses/)). The model weights are downloaded during preparation and keep their publishers' licenses: Kitten TTS 2 is under the [Stellon Labs Community License](https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md), and the decoder and pyannote checkpoints are under MIT.
