# C API examples

Small C programs that drive the engine through its C API
([`include/kt2n/engine.h`](../include/kt2n/engine.h)) and nothing else. Each one
takes the directory written by `tools/prepare_assets.py --out DIR` and finds the
files under `DIR/native/` itself.

| Program    | What it shows |
|------------|---------------|
| `stream`   | Streaming with a built-in voice: raw PCM is written chunk by chunk while the rest is still being generated. Ctrl-C cancels. |
| `complete` | The non-streaming mode (`KT2N_OUTPUT_COMPLETE`): one finished WAV file. Long text is split into sentence-sized chunks, each chunk is rendered in one piece and the whole file is levelled at the end. |
| `clone`    | `build`: a voice from a 2-30 second recording and its transcript, saved to a file. `speak`: load a saved voice and write a WAV file with it. |
| `session`  | Incremental text: lines read from standard input are spoken as they arrive, as with text coming from a chat model. An empty line commits; end of input closes the session. |
| `voice_cache` | Keeping only some voices on the GPU (`KT2N_VOICE_CACHE_LRU` with a voice count and a MiB budget): one short request per voice in the order given, each reported as a hit or a miss with the number of voices dropped for it, then the `voice_cache` object of `kt2n_stats_json()`. |

## Build

The examples are built with the project (option `KT2N_BUILD_EXAMPLES`, on by
default on Linux) and end up in `<build dir>/examples/`, for example
`build/default/examples/` after the README's Quick start. The Windows presets
do not build them.

They also build on their own against an installed package (see
[Use the library in your own program](../README.md#use-the-library-in-your-own-program)):

```sh
cmake -S examples -B build-examples -DCMAKE_PREFIX_PATH="$HOME/.local"
cmake --build build-examples
```

## Run

Run these from the repository folder after the README's Quick start.
`models/kitten-tts-2` is the directory made by `tools/prepare_assets.py`.

The examples use the automatic memory profile (the default profile where it
fits, otherwise the low-memory profile, with one line on standard error that
says so) and keep only the voice they speak with on the GPU (`clone`: no
built-in voice), so they start on GPUs with 4 GB or more. The C API's own
defaults need more; see
[Using less GPU memory](../README.md#using-less-gpu-memory).

```sh
# Stream raw 16-bit little-endian mono PCM at 24 kHz into a file as it is made.
build/default/examples/stream models/kitten-tts-2 Bruno "Hello there." stream.pcm

# One complete WAV file.
build/default/examples/complete models/kitten-tts-2 Bruno "A complete passage with natural pauses." passage.wav

# Text that arrives over time, written as raw PCM.
(echo "Hello there."; sleep 1; echo "This line arrived a second later.") |
  build/default/examples/session models/kitten-tts-2 Bruno session.pcm

# At most 2 voices and 1024 MiB of GPU memory for voices; Bruno is a hit the second time
# (on a 4 GB card the cache has room for one voice only, so every request is a miss).
build/default/examples/voice_cache models/kitten-tts-2 2 1024 Bruno Bella Bruno Kiki Bella Bruno
```

Use `-` instead of a `.pcm` file name to write to standard output. On a
computer with sound output you can then listen straight away with `aplay`
(package `alsa-utils`):

```sh
build/default/examples/stream models/kitten-tts-2 Bruno "Hello there." - | aplay -q -f S16_LE -r 24000 -c 1
```

To make a voice from a recording, save it, then speak with it (`me.wav` stands
for your own recording):

```sh
build/default/examples/clone build models/kitten-tts-2 me.wav "The exact words spoken in the recording." me.voice
build/default/examples/clone speak models/kitten-tts-2 me.voice "Hello in a saved voice." hello.wav
```

`clone build` needs a library built with voice cloning (the `default` preset,
or `-DKT2N_BUILD_CLONING=ON -DKT2N_VOICE_AV=ON`), a model prepared with the
cloning files ([Voice cloning](../README.md#voice-cloning)) and about 11,300 MiB
of free GPU memory. The recording can be WAV, MP3, FLAC, OGG and other common formats, at most 10 MiB,
and the transcript must match what is said. `clone speak` needs neither: a saved
voice file only needs the same prepared model files it was made with.

Sessions accept built-in voices only.
