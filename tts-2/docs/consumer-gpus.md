# Consumer GPUs

Kitten TTS 2 native runs on the GPUs most people own: GeForce cards from the GTX 10 series on, cards with 4 GB of
memory, AMD Radeon RX 7000 and RX 9000 cards, and Windows. On RTX 20-series and newer NVIDIA GPUs the audio of the CUDA 12.8
build is bit for bit the same as with the earlier build of this engine (never released as a package; see
[How these were measured](#how-these-were-measured)), and no setting we measured got slower.

- **Starts with no options on every card we tested, from 4 GB up** (20 GeForce and Radeon cards). `kitten-tts-native`
  picks the memory profile itself and fits its buffers to small cards. Before, only cards with about 24 GB started
  without `--memory-profile low`. In the Steam Hardware Survey (September 2026), about 90% of NVIDIA users have such a
  card; before, under 3% could start the engine without options.
- **Faster command-line start:** one sentence from start to exit in 2.1 s instead of 5.1 s, and first audio 37 ms
  after the request instead of 66 ms (RTX 4060 Laptop GPU).
- **GTX 10, GTX 16 and RTX 20 1.7 to 2.2 times faster, with the same audio:** real-time factor 0.58 → 0.27 on a GTX
  1660 Ti, 0.47 → 0.28 on a GTX 1080 Ti.
- **AMD Radeon RX 7000 and RX 9000** with a ROCm build: real-time factor 0.068 and first audio 0.05 s on an RX 7900
  XTX; five Radeon models tested.
- **4 GB cards** run with buffers fitted to their free memory; the minimum is about 2,583 MiB free (up to about
  3,000 MiB under Windows and WSL2 with some display drivers).
- **Windows** builds natively with Visual Studio 2022.

A real-time factor of 0.27 means 10 seconds of audio take 2.7 seconds to produce.

**Model revision.** The measurements on this page were made with an earlier revision of the model (f2eee74, 47 voices).
The tool now prepares the latest revision, which has 115 voices. Speed and quality figures were not measured
again. Memory figures that depend on preparing all voices are given for the latest revision where we measured them
again, and are marked "47-voice revision" otherwise.

## At a glance

| GPU | Status | Measured on |
|---|---|---|
| GeForce RTX 30, 40 and 50 series | supported; same audio as the earlier build | RTX 4060 Laptop GPU: real-time factor 0.079; RTX 5060 Ti: 0.047 |
| GeForce RTX 20 series | supported, not optimal: no bf16 tensor cores, so part of the math runs in software | RTX 2060: real-time factor 0.45 → 0.22 |
| GeForce GTX 16 series | supported, not optimal: as RTX 20 | GTX 1660 Ti: 0.58 → 0.27 |
| GeForce GTX 10 series | supported, not optimal: as RTX 20 | GTX 1080 Ti: 0.47 → 0.28 |
| Cards with 8 GB or less | supported, not optimal: one stream, buffers fitted to the card (`--stream false` about 1.5% slower), no voice cloning; see [Small-memory cards](#small-memory-cards) | RTX 4060 Laptop GPU 8 GB: 0.079 |
| 4 GB cards (3 GB cards not tested) | supported, not optimal: as above, and long streaming text is cut where the fitted sizes limit it | RTX 3050 Laptop GPU 4 GB: 0.17 |
| 2 GB cards | not supported: the engine needs about 2,583 MiB of free GPU memory and says so at start | — |
| AMD Radeon RX 7000 and RX 9000 | supported; audio close to, not bit for bit the same as, NVIDIA; no voice cloning; RX 7600 and RX 7600 XT not tested; see [AMD Radeon](#amd-radeon) | RX 7900 XTX: 0.068, first audio 0.05 s; RX 9060 XT: 0.090 |
| 64-bit Windows with an NVIDIA GPU | supported; the same audio as Linux on the same GPU model; voice cloning only with the pip package; tested on Windows Server 2022 only; see [Windows](#windows) | NVIDIA L4: 0.074, first audio 46 ms |

Building a cloned voice needs about 11,300 MiB of free GPU memory with the low-memory profile, which the tool picks
when the default profile's 15,600 MiB does not fit; it was tested end to end on 16 GB and 24 GB cards. On 12 GB cards it fits
when other programs use at most about 500 MiB of GPU memory. On cards with 8 GB or less it is off; see
[Small-memory cards](#small-memory-cards).

Other AMD GPUs (RX 6000 and earlier, Instinct) and NVIDIA GPUs other than compute capability 6.1, 7.5 and 8.0 or newer
(for example 6.0 and 7.0) are not supported: the engine stops at start and says so.

## Output differences

**NVIDIA: the same audio as the earlier build (CUDA 12.8 build).** On RTX 20-series and newer NVIDIA GPUs, and on GTX 10
and GTX 16 GPUs with the new decode kernel, a seeded `--stream false` request from the CUDA 12.8 build gives bit for
bit the same audio as the earlier build on the same GPU, with the same prepared model files. The fitted small-memory sizes give the same audio too, except for the
streaming split listed below. Windows and Linux give byte for byte the same audio on the same GPU model (the pip
packages of this version on an NVIDIA L4: 36 of 36 seeded `--stream false` requests).

**Streaming on every GPU.** Where streamed audio is cut into decoder chunks depends on how fast speech is generated,
so streamed audio with the same seed can differ from run to run on any GPU (in one test, 39 of 212 streaming requests
differed between two runs of the same settings). In our tests the words stayed the same. `--stream false` is not
affected.

**Between GPU models.** Two different GPU models usually do not give bit for bit the same samples (the decoder's noise
depends on the number of processors on the GPU), but on the GeForce RTX 30 and 40 cards we measured the words and
voice were the same as on an RTX 4090. The groups below can differ more. Each was measured against an RTX 4090 with
the same model files, using a speech recogniser (word error rate) and a speaker-similarity model. For scale: the
engine without any of these changes ranges from 0.46% to 1.91% word error rate across GPUs and seeds on the same 24
texts.

| GPU group | Why it differs | Measured |
|---|---|---|
| AMD Radeon RX 7000 and RX 9000 (default) | AMD's math library (exp, log, sin, cos) and the Radeon matrix instructions round differently, so sampled speech can differ | 24 texts, 2 seeds: word error rate 1.41% (RX 7900 XTX) and 1.18% (RX 9070 XT), vs 0.60% on the RTX 4090. 2 of 48 and 1 of 48 outputs left out a sentence, which also happens on NVIDIA GPUs (3 of 48 on an RTX 2060 with the earlier build); without those, 0.54% and 0.62%, vs 0.46% and 0.62% on the RTX 4090. The voice is closer to the RTX 4090's than a different seed is (similarity 0.971 and 0.976, vs 0.949). On the 72 test requests: word error rate 1.17% on every RX 7000 card (RX 7900 XTX, RX 7900 XT, RX 7800 XT) and 1.02% on both RX 9000 cards (RX 9070 XT, RX 9060 XT), against 0.99% on the RTX 4090; 4 of 72 outputs dropped a sentence on every card, as on the RTX 4090 |
| AMD Radeon with `KT2N_AMD_WMMA=0` | the matrix math runs in software | word error rate 1.24% on both cards; voice similarity 0.966 |
| AMD Radeon, builds made with different ROCm versions | the compiler rounds differently | ROCm 7.1 and 7.2.4 builds on the same card differ only at the level of rounding (about -65 dB), with the same words and voice |
| Cards where the fitted sizes limit streaming segments (seen on 4 GB cards), streaming only | a long streaming text chunk is split into shorter pieces, each spoken separately ([details](#small-memory-cards)) | Output can differ. Same GPU with and without the split, 24 texts, 2 seeds: 3 of 48 outputs skipped a sentence (without: none), word error rate 1.43% (without: 0.58%), same voice (similarity 0.995). On a real RTX 3050 Laptop GPU 4 GB: 1.71%, 2 of 47 outputs. On the 72 test requests, real 4 GB cards stayed within the usual variation (word error rate 0.80% to 0.99%; RTX 4090: 0.99%). `KT2N_FIT_SPLIT=0` refuses such text instead. `--stream false` is never split |
| NVIDIA, builds made with CUDA 13 | another compiler version rounds differently | RTX 5060, 72 test requests: none bit for bit the same as the CUDA 12.8 build's; word error rate 1.39% against 1.42% |
| GeForce GTX 10, GTX 16 and RTX 20 | no bf16 tensor cores: the engine does the same math in software; not changed by this version | 72 test requests: word error rate 0.80%, vs 0.99% on the RTX 4090; voice similarity 0.957, vs 0.910 for a different seed |

These are speech-recognition and speaker-similarity measurements; no listening test was done.

## Starting without options

`kitten-tts-native` uses `--memory-profile auto` unless you pass `--memory-profile`. It uses the default profile when
that fits the free GPU memory and `low` otherwise, and prints one line when it picks `low`. The audio is the same
either way: a seeded `--stream false` request gives the same bytes with `low` and `default`.

- With the tool's own voice setting (only the spoken voice prepared), the default profile needs about
  13,500 MiB of GPU memory with the current model; with `--voice-cache all`, about 33,200 MiB (see
  [GPU memory](memory.md)).
- **Below that, `auto` uses `low`: supported, not optimal.** One request at a time runs as fast (`--stream false`
  real-time factor 0.050 with `low` and 0.051 with the default profile on an RTX 5060 Ti 16 GB, measured on a
  development build of this version). With several streams (C API), requests that arrive together are batched less. On 8 GB cards and
  smaller, `low` also fits its sizes to the card and the decoder runs without CUDA graphs: `--stream false` about 1.5%
  slower than with them on an 8 GB laptop GPU.
- When even `low` does not fit, the engine does not start and says how much free memory it needs.

The command-line tool also starts sooner than the earlier build: it checks the model files in the background, loads the
text grammars in parallel and prepares only the voice it speaks with, before the request. The audio is the same.
Measured with `--memory-profile low`, running the earlier build and this version in turns on the same machine:

| GPU | One sentence, start to exit | First audio after the request |
|---|---|---|
| RTX 4060 Laptop GPU 8 GB | 5.10 → 2.07 s | 66 → 37 ms |
| RTX 5060 8 GB | 4.80 → 2.06 s | 93 → 69 ms |
| RTX 5060 Ti 16 GB | 6.32 → 2.97 s | 68 → 51 ms |
| RTX 3060 12 GB | 23.00 → 8.27 s | 75 → 43 ms |

The RTX 5060 and RTX 3060 are desktops measured under WSL2 on Windows. The RTX 3060's processor has no SHA
instructions, so checking the model files took longest there, and moving that check to the background helped most. On
the RTX 5060 and RTX 5060 Ti, each generated token also takes 7 to 9% less time (same audio).

With an explicit `--voice-cache all`, which prepares all voices, first audio from a cold command-line start is as
fast as with the earlier build: 19.5 → 19.1 ms on an RTX 4090 (median of 40 rounds in turns).

## GTX 10, GTX 16 and RTX 20

These GPUs have no bf16 tensor-core instructions. **Supported, not optimal:** part of the language model's math runs
in software on them, so they are slower than an RTX 30-series or newer card. When the engine produces the next audio
token, the language model's weight multiplications now run on plain arithmetic units instead of that software path, so
speech as a whole is 1.7 to 2.2 times faster than with the earlier build (about 2 times on GTX 16 and RTX 20, 1.7 times on
GTX 10; table below). With one request those multiplications got about 4 to 5 times faster on the GPUs we measured
(GTX 1080 Ti, GTX 1070, GTX 1660 Ti, RTX 2060), and the whole step 1.8 to 2.7 times faster (GTX 1080 Ti, GTX 1660 Ti,
RTX 2060). Other GPUs in these groups run the same code but were not measured. With more than four requests at once,
the engine keeps the previous method, which is faster there.

On each measured GPU the audio is bit for bit the same as with the earlier build.

Measured with `--memory-profile low`, one request at a time, on 36 English and German requests of different lengths,
each with and without streaming; before = the earlier build on the same card:

| GPU | Real-time factor (`--stream false`), before → now | First audio (streaming, median), before → now |
|---|---|---|
| GTX 1080 Ti 11 GB | 0.47 → 0.28 | 0.30 s → 0.25 s |
| GTX 1070 8 GB | 0.61 → 0.37 | 0.36 s → 0.30 s |
| GTX 1660 Ti 6 GB | 0.58 → 0.27 | 0.32 s → 0.24 s |
| RTX 2060 12 GB | 0.45 → 0.22 | 0.25 s → 0.19 s |

## Small-memory cards

On a card with too little free memory for its standard sizes (8 GB cards and smaller), `--memory-profile low`
measures the free memory when it starts and shrinks its buffers to fit. **Complete files (`--stream false`) do not
change:** on the same GPU they are byte for byte the same as with the standard sizes (checked on an RTX 4090 with
2,700 to 3,680 MiB free, and on real 6 GB cards against the earlier build; on real 4 GB cards the words and voice matched
a larger card's). Streaming audio changes only where long text is cut (below), apart from the run-to-run variation
streaming has with any setting.

The minimum is about **2,583 MiB of free GPU memory** after the CUDA context. Under Windows and WSL2 some display drivers
take extra GPU memory once the engine has written about 2 GB (about 400 MiB on the RTX 3050 Laptop GPU; none on a GTX
1650; 26 to 62 MiB on an RX 9060 XT), so the minimum there can be up to about 3,000 MiB. The engine measures
this when it starts. Below the minimum the engine does not start and says how much it needs. A 2 GB card
is therefore not supported.

Measured with one stream, `--voice-cache none` unless noted, on the same 72 test requests (median real-time factor;
median first audio when streaming; with `--voice-cache none` each request first prepares its voice). The last column
is the tool with no options (8 runs: median real-time factor with `--stream false`, median first audio when
streaming). The cards with a Windows desktop running were measured under WSL2:

| Card | Free after the CUDA context | Streaming text | Real-time factor `--stream false` / streaming, first streamed audio | With no options |
|---|---|---|---|---|
| GTX 1660 Ti 6 GB, `--voice-cache all` | whole card (Linux, no desktop) | no limit (72 of 72 test requests) | 0.27 / 0.29, 0.24 s | 0.26, 0.22 s |
| GTX 1060 6 GB laptop GPU, `--voice-cache all` | 5,173 MiB | no limit (72 of 72) | 0.51 / 0.58, 0.43 s | 0.50, 0.42 s |
| GTX 1650 4 GB | 3,254 MiB | chunks over about 450 characters (after normalisation) are cut | 0.55 / 0.57, 1.5 s | 0.43, 0.35 s |
| GTX 1050 Ti 4 GB | 3,331 MiB | chunks over about 480 characters (after normalisation) are cut; rare | 0.95 / 1.03, 2.5 s | 0.76, 0.62 s |
| RTX 3050 Laptop GPU 4 GB | 3,277 MiB | chunks over about 330 characters (after normalisation) are cut | 0.23 / 0.25, 0.32 s | 0.17, 0.10 s |

With `--voice-cache none`, streaming on the GTX 1050 Ti is slower than real time (real-time factor 1.03), so streamed
playback can pause. With no options, the tool prepares the voice before the request, which is faster on every card
here.

**Supported, not optimal:**
- **One stream, no decoder CUDA graphs**, and a new segment can wait for the previous one's memory. The speed above is
  set mostly by the card's math (GTX 10 and GTX 16 cards have no bf16 tensor cores), not by the smaller buffers.
- **Long streaming text is cut into shorter segments** whenever the fitted sizes limit streaming segments (on the
  cards we measured: 4 GB cards; `kt2n_engine_low_memory_limits()` reports the limit as `streaming_budget_tokens`).
  Text is cut into chunks of up to about 380 characters, which normalisation can lengthen to about 530 (460 in our
  test texts), and a chunk longer than the fitted budget is cut again at sentence, clause or word breaks (a single
  word longer than the limit is cut inside the word). Those pieces are spoken separately, so the audio can differ from
  a larger GPU's (see [Output differences](#output-differences)). Set `KT2N_FIT_SPLIT=0` to refuse such a request
  instead: it then fails with `gpu_memory_limit_exceeded` before the long chunk (`kitten-tts-native` prints the reason
  and `(status 111)`). A piece that still does not fit is refused the same way. `--stream false` segments are never
  cut; with about 4 GB, one whose text needs much more language-model context than usual (for example hundreds of rare
  characters) can be refused with `gpu_memory_limit_exceeded`, which none of our test texts was.
- **No voice cloning on cards where the sizes are fitted (8 GB and smaller):** the cloning model needs about 4.4 GB
  more. The engine starts, and building a voice fails with a message saying so. Building a cloned voice needs about 11.3
  GB of free GPU memory, more than a 10 GB card has. Build and save the voice on a larger GPU, then use it here with
  `--load-voice` (without a `--voice-cache` option the tool keeps only that voice on the GPU). A saved voice with a
  long prompt leaves less room for text, so its streaming segments are shorter.
- **4 GB cards need a small voice cache.** Leave out `--voice-cache` (the tool then prepares only the voice it speaks
  with), or use `--voice-cache none` or `--voice-cache lru --voice-cache-voices 1`. `--voice-cache all` needed about
  4,305 MiB with the 47-voice revision (every built-in voice stays prepared; more with the latest revision), and plain `--voice-cache lru` sets aside room for 16 voices, which
  needs even more; both are refused at start on 4 GB cards.
- **A start can be refused while the desktop grabs memory.** On a laptop whose desktop takes more GPU memory while the
  engine starts, less than the 128 MiB the engine keeps free can be left (1 of 48 starts on the RTX 3050 Laptop GPU).
  Starting again works.
- Host memory: about 0.9 to 1.2 GB at peak (up to 1,189 MiB measured).

### How the sizes are fitted

`low` fits itself to the GPU when it starts:

1. It measures the free memory after the CUDA context.
2. It keeps `min_free_device_bytes` free: by default 128 MiB or 1/64 of the GPU, whichever is larger.
3. When its standard sizes do not fit:
   - it sizes the decoder's working memory for complete files to exactly what its calls need;
   - it keeps one language-model context slot per stream, so a segment waits until the previous segment's slot is
     released;
   - the decoder runs without CUDA graphs (the audio is the same; the graphs would need 6 to 8 MiB each on such a GPU);
   - with one stream, buffers that are never in use at the same time share memory, and tables are sized for the
     longest segment the decoder can hold;
   - it then reduces the language model's context and the longest complete-file segment the vocoder renders in one
     piece.
4. On a GPU with less free memory still (about 4 GB), complete-file segments (`--stream false`) keep their full budget
   of 1,000 tokens, and streaming segments are limited to the largest token budget that still fits. A streaming text
   chunk longer than that budget is split into shorter segments that fit.
5. Under Windows (and WSL2), some display drivers take extra GPU memory once a program has written about 2 GB (414 MiB
   on an RTX 3050 Laptop GPU). The engine measures this when it starts and fits itself to what is left.

With enough memory for it (6 GB cards), it never goes below what one segment needs at its token budget: the longest
streaming segment (532 characters after normalisation, 2,341 tokens) and a complete-file segment at its budget of
1,000 tokens.

Fitted sizes on real 4 GB cards running a Windows desktop (WSL2):

| Free GPU memory | Fitted sizes | GPU memory used by the engine (the GPU's memory in use minus what the desktop held) |
|---|---|---|
| 3,254 MiB (GTX 1650) | context 3,136 tokens, 72 decoder pages, 2,048 vocoder frames, streaming budget 1,965 tokens (about 450 characters) | about 3,180 MiB |
| 3,331 MiB (GTX 1050 Ti) | context 3,264 tokens, 77 decoder pages, 2,048 vocoder frames, streaming budget 2,125 tokens (about 480 characters) | about 3,240 MiB |
| 3,277 MiB (RTX 3050 Laptop GPU; the display driver takes about 400 MiB more) | context about 2,500 tokens, 55 to 56 decoder pages, 2,048 vocoder frames, streaming budget about 1,420 to 1,450 tokens (about 330 characters; it depends on what the desktop uses) | about 2,800 MiB |

A request that needs more than was fitted fails with `gpu_memory_limit_exceeded` and a message with both sizes; it is
never shortened silently. This can happen with a voice prompt and text that need more language-model context than was
fitted, a segment whose decoder pages cannot be reserved, or a segment that runs past its token budget until the
fitted context is full. With the smallest fitted sizes a streaming text chunk is split before this check (above), so a
segment is refused there only when even its pieces do not fit, or with `KT2N_FIT_SPLIT=0`.

## AMD Radeon

Radeon RX 7000 (RDNA3) and RX 9000 (RDNA4) GPUs run the engine with a ROCm build, which uses the Radeon matrix
instructions. The build steps are in [amd-radeon.md](amd-radeon.md). Measured with the command-line tool's default
settings, one request at a time, about 16 seconds of speech:

| GPU | Real-time factor (`--stream false` / streaming) | First audio (streaming) | GPU memory (with `--memory-profile low`) |
|---|---|---|---|
| RX 7900 XTX 24 GB | 0.068 / 0.075 | 0.05 s | 13.0 GB (7.4 GB) |
| RX 7900 XT 20 GB | 0.070 / 0.079 | 0.06 s | 12.9 GB (7.3 GB) |
| RX 7800 XT 16 GB | 0.090 / 0.102 | 0.07 s | 12.8 GB (7.2 GB) |
| RX 9070 XT 16 GB | 0.059 / 0.068 | 0.05 s | 12.8 GB (7.2 GB) |
| RX 9060 XT 16 GB | 0.090 / 0.101 | 0.07 s | 12.6 GB (7.0 GB) |

Under WSL2, the low-memory profile keeps at least 1 GiB free when it fits itself to a Radeon (some RX 7900 XT and XTX
PCs hang when free memory gets below about 0.5 GB); on a simulated 8 GB card this cost nothing. All five cards were
measured inside WSL2 on Windows desktops, with ROCm 7.1 builds; each started with no options 8 of
8 times and served all 72 test requests. On a 16 GB card whose desktop used 2.5 GB, the tool picked the low profile by
itself. Linux without WSL2 was not tested. Other RX 7000 and RX 9000 cards (RX 7700 XT, and 8 GB cards) run the same
code but were not tested; AMD's WSL2 support does not include the RX 7600 or RX 7600 XT. The ROCm build has no voice
cloning: build and save a voice on an NVIDIA GPU, then load it on the Radeon.

`KT2N_AMD_WMMA=0` turns the matrix instructions off. **Supported, not optimal:** all matrix math then runs in
software, 3.4 to 4.9 times slower: real-time factor 0.23 (RX 7900 XTX) to 0.44 (RX 9060 XT).

The audio is close to, not bit for bit the same as, an NVIDIA GPU's; see [Output differences](#output-differences).
With all built-in voices prepared (`--voice-cache all`), the default memory profile needs about 33,200 MiB, so on 16 GB
cards such as the RX 9070 XT the command-line tool then picks the low profile by itself (supported, not optimal; see
[Starting without options](#starting-without-options)). With the one voice it normally prepares, the default profile
fits (see the table).

## Windows

Kitten TTS 2 native builds and runs natively on 64-bit Windows with an NVIDIA GPU. The steps are in
[windows.md](windows.md). It was tested on Windows Server 2022 with an NVIDIA L4 (an Ada data-centre GPU, 24 GB, run
with the same Windows driver model as GeForce cards). Windows 10 and Windows 11 on desktop GeForce cards are expected to
work the same way but have not been tested.

| On the L4 | Windows |
|---|---|
| First audio (streaming, median), default settings | 46 ms (6 runs) |
| Real-time factor, streaming / `--stream false` | 0.082 / 0.074 (12 to 13.5 times faster than real time; 24 test texts, measured on a development build of this version with `--memory-profile low --voice-cache lru --voice-cache-voices 4`) |
| Word error rate | 0.55% (Linux on an RTX 4090: 0.60%; same development build and settings) |

Output against Linux: with the pip packages of this version, the Windows and Linux engines give byte for byte the
same audio on the same GPU model (an NVIDIA L4 on each: 36 of 36 seeded `--stream false` requests), and the model
prepared on Windows is byte for byte the one prepared on Linux. Voices cloned on Windows match Linux too (WAV and MP3
recordings; M4A recordings can differ slightly with FFmpeg 9). First use of the pip package (download and
preparation) took 160 s on the L4 VM.

## How these were measured

- "The earlier build" is this engine as it was before this release (0.2.0), when these measurements were made. It was
  never released as a package, so it cannot be installed (kitten-inference 0.1.x on PyPI is the Kitten TTS 0.8
  engine, another product); where the engine's code comes from is in [PROVENANCE.md](../PROVENANCE.md). "Before → now"
  tables compare it with this version on the same card, run in turns.
- Model files prepared from `KittenML/kitten-tts-2` revision f2eee74 on one NVIDIA GPU, the same files on every GPU.
- Word error rate: Whisper large-v3 transcripts compared with the input text. Voice similarity: cosine similarity of
  speaker embeddings (ECAPA) between this GPU's output and the RTX 4090's for the same text and seed; "a different
  seed" is the same comparison between two seeds on the RTX 4090.
- Test sets: 24 long and short English texts with 2 seeds each, and 72 requests (3 voices in English and German, 6
  texts from 45 to 790 characters, 2 seeds, streaming and `--stream false`).
