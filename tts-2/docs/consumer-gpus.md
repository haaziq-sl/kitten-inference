# Consumer GPUs

Speed, memory and audio for each GPU group. Except for [Against vLLM and llama.cpp](#against-vllm-and-llamacpp), the
figures come from an earlier model revision (47 voices); the tool now prepares the latest revision, which has 115
voices.

A real-time factor of 0.27 means 10 seconds of audio take 2.7 seconds to produce.

## At a glance

| GPU | Status |
|---|---|
| GeForce RTX 30, 40 and 50 series | supported: real-time factor 0.05 to 0.08 with `--stream false` |
| GeForce GTX 10, GTX 16 and RTX 20 series | supported, not optimal: no bf16 tensor cores, so part of the math runs in software (real-time factor 0.22 to 0.37 on desktop cards with 6 GB or more) |
| Cards with 8 GB or less | supported, not optimal: one stream, buffers fitted to the card (`--stream false` about 1.5% slower), no voice cloning; see [Small-memory cards](#small-memory-cards) |
| 4 GB cards (3 GB cards not tested) | supported, not optimal: as above, and long streaming text is cut where the fitted sizes limit it |
| 2 GB cards | not supported: the engine needs about 2,583 MiB of free GPU memory and says so at start |
| AMD Radeon RX 7000 and RX 9000 | supported: real-time factor 0.059 to 0.090; audio close to, not bit for bit the same as, NVIDIA; no voice cloning; RX 7600 and RX 7600 XT not tested; see [AMD Radeon](#amd-radeon) |
| 64-bit Windows with an NVIDIA GPU | supported: the same audio as Linux on the same GPU model; voice cloning only with the pip package; tested on Windows Server 2022 only; see [Windows](#windows) |

Building a cloned voice needs about 11,300 MiB of free GPU memory with the low-memory profile, which the tool picks
when the default profile's 15,600 MiB does not fit; it was tested end to end on 16 GB and 24 GB cards. On 12 GB cards it fits
when other programs use at most about 500 MiB of GPU memory. On cards with 8 GB or less it is off; see
[Small-memory cards](#small-memory-cards).

Other AMD GPUs (RX 6000 and earlier, Instinct) and NVIDIA GPUs other than compute capability 6.1, 7.5 and 8.0 or newer
(for example 6.0 and 7.0) are not supported: the engine stops at start and says so.

## Against vLLM and llama.cpp

With the current model (115 voices), through the pip package with its default settings, one request at a time, each
GPU on its own machine. Every system spoke the same 300 short texts (about 5.5 s of speech each) and 12 long texts
(about 64 s each) with the same voice and seeds. Medians; for vLLM and llama.cpp, their time divided by the native
engine's (short / long texts):

| GPU | Native, short / long | Native, first streamed audio | vLLM, 4-bit | vLLM, bf16 (its default) | llama.cpp (GPU) |
|---|---|---|---|---|---|
| RTX PRO 6000 96 GB | 82 ms / 0.95 s | 11.4 ms | 5.55x / 5.12x | 7.15x / 6.92x | 6.37x / 5.64x |
| RTX 4090 24 GB | 159 ms / 1.80 s | 22.1 ms | 5.22x / 4.49x | 6.89x / 6.37x | 5.43x / 5.04x |
| RTX 3060 12 GB | 459 ms / 5.07 s | 57.5 ms | 2.45x / 2.44x | 4.27x / 4.33x | 2.55x / 2.40x |

- vLLM's and llama.cpp's streaming sends nothing until a whole sentence is done, which on these short texts is the
  complete file (0.5 to 2.0 s), so their first audio is not ranked against the native engine's.
- Word error rate is about the same for every system (0.00 to 0.05% on the short texts).
- The lead shrinks on smaller GPUs: the native engine's speed follows the GPU's memory speed, while vLLM and llama.cpp
  lose a roughly fixed 2 to 5 ms per token on the processor side.
- On the RTX 3060 the tool picked the low-memory profile by itself (the default profile needed about 275 MiB more than
  was free): supported, not optimal, because it batches several requests at once less; at one request at a time, as
  here, that does not show.
- Several requests at once were not measured; vLLM is built for that case.

## Output differences

**NVIDIA.** On the same GPU model, seeded `--stream false` requests give byte for byte the same audio on Linux and
Windows, and with every memory profile, including the sizes fitted to small cards (except the 4 GB streaming split
below).

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
| AMD Radeon RX 7000 and RX 9000 (default) | AMD's math library (exp, log, sin, cos) and the Radeon matrix instructions round differently, so sampled speech can differ | 24 texts, 2 seeds: word error rate 1.41% (RX 7900 XTX) and 1.18% (RX 9070 XT), vs 0.60% on the RTX 4090. 2 of 48 and 1 of 48 outputs left out a sentence, which also happens on NVIDIA GPUs (3 of 48 on an RTX 2060 with an earlier build); without those, 0.54% and 0.62%, vs 0.46% and 0.62% on the RTX 4090. The voice is closer to the RTX 4090's than a different seed is (similarity 0.971 and 0.976, vs 0.949). On the 72 test requests: word error rate 1.17% on every RX 7000 card (RX 7900 XTX, RX 7900 XT, RX 7800 XT) and 1.02% on both RX 9000 cards (RX 9070 XT, RX 9060 XT), against 0.99% on the RTX 4090; 4 of 72 outputs dropped a sentence on every card, as on the RTX 4090 |
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
- **Below that, `auto` uses `low`: supported, not optimal.** One request at a time runs about as fast (`--stream false`
  real-time factor 0.050 with `low` and 0.051 with the default profile on an RTX 5060 Ti 16 GB, measured on an
  earlier build). With several streams (C API), requests that arrive together are batched less. On 8 GB cards and
  smaller, `low` also fits its sizes to the card: `--stream false` about 1.5% slower on an 8 GB laptop GPU.
- When even `low` does not fit, the engine does not start and says how much free memory it needs.

## GTX 10, GTX 16 and RTX 20

**Supported, not optimal:** these GPUs have no bf16 tensor-core instructions, so part of the language model's math
runs in software and they are slower than an RTX 30-series or newer card: real-time factor 0.22 to 0.37 with
`--stream false`, one request at a time, on desktop cards with 6 GB or more (cards with less memory also run with
[fitted sizes](#small-memory-cards)). With more than four requests at once the engine uses a different method,
which is faster there.

## Small-memory cards

On a card with too little free memory for its standard sizes (8 GB cards and smaller), `--memory-profile low`
measures the free memory when it starts and shrinks its buffers to fit. **Complete files (`--stream false`) do not
change:** on the same GPU they are byte for byte the same as with the standard sizes. Streaming audio changes only
where long text is cut (below), apart from the run-to-run variation streaming has with any setting.

The minimum is about **2,583 MiB of free GPU memory** after the CUDA context. Under Windows and WSL2 some display
drivers take up to about 400 MiB more once the engine has written about 2 GB, so the minimum there can be up to about
3,000 MiB. The engine measures this when it starts. Below the minimum it does not start and says how much it needs. A
2 GB card is therefore not supported.

**Supported, not optimal:**
- **One stream**, and a new segment can wait for the previous one's memory. On these cards the speed is set mostly by
  the card's math (GTX 10 and GTX 16 cards have no bf16 tensor cores), not by the smaller buffers.
- **Long streaming text is cut into shorter segments** where the fitted sizes limit them (on 4 GB cards, chunks over
  about 330 to 480 characters after normalisation). Those pieces are spoken separately, so the audio can differ from a
  larger GPU's (see [Output differences](#output-differences)). Set `KT2N_FIT_SPLIT=0` to refuse such a request
  instead: it then fails with `gpu_memory_limit_exceeded` (`kitten-tts-native` prints the reason and `(status 111)`).
  `--stream false` segments are never cut.
- **No voice cloning** where the sizes are fitted (8 GB and smaller): the engine starts, and building a voice fails
  with a message saying so. Building a cloned voice needs about 11,300 MiB of free GPU memory, more than a 10 GB card
  has. Build and save the voice on a larger GPU, then use it here with `--load-voice`. A saved voice with a long prompt
  leaves less room for text, so its streaming segments are shorter.
- **4 GB cards need a small voice cache.** Leave out `--voice-cache` (the tool then prepares only the voice it speaks
  with), or use `--voice-cache none` or `--voice-cache lru --voice-cache-voices 1`. `--voice-cache all` and plain
  `--voice-cache lru` are refused at start on 4 GB cards.
- **A start can be refused while the desktop grabs memory.** On a laptop whose desktop takes more GPU memory while the
  engine starts, starting again works.
- Host memory: about 0.9 to 1.2 GB at peak.

## AMD Radeon

Radeon RX 7000 (RDNA3) and RX 9000 (RDNA4) GPUs run the engine with a ROCm build; see [amd-radeon.md](amd-radeon.md).
With the command-line tool's default settings, one request at a time: real-time factor 0.059 to 0.090 with
`--stream false`, first streamed audio after 0.05 to 0.07 s. Tested inside WSL2 on Windows; Linux without WSL2 was
not tested. Under WSL2, the low-memory profile keeps at least 1 GiB free when it fits itself to a Radeon, because some
RX 7900 XT and XTX PCs hang below about 0.5 GB free.

Voice cloning is not supported on Radeon (not tested in a source build; the pip package has none): build and save a
voice on an NVIDIA GPU, then load it on the Radeon.

`KT2N_AMD_WMMA=0` turns the matrix instructions off. **Supported, not optimal:** all matrix math then runs in
software, 3.4 to 4.9 times slower: real-time factor 0.23 to 0.44.

The audio is close to, not bit for bit the same as, an NVIDIA GPU's; see [Output differences](#output-differences).

## Windows

The steps are in [windows.md](windows.md). Tested on Windows Server 2022 with an NVIDIA L4; Windows 10 and 11 on
GeForce cards are expected to work the same way but were not tested. With the pip packages, Windows and Linux give
byte for byte the same audio on the same GPU model (36 of 36 seeded `--stream false` requests), and voices cloned on
Windows match Linux (WAV and MP3 recordings; M4A recordings can differ slightly with FFmpeg 9). First use of the pip
package (download and preparation) took 160 s; real-time factor 0.074 with `--stream false` on the L4.

## How these were measured

- Word error rate: Whisper large-v3 transcripts compared with the input text. Voice similarity: cosine similarity of
  speaker embeddings (ECAPA) between this GPU's output and the RTX 4090's for the same text and seed; "a different
  seed" is the same comparison between two seeds on the RTX 4090.
- Test sets: 24 long and short English texts with 2 seeds each, and 72 requests (3 voices in English and German, 6
  texts from 45 to 790 characters, 2 seeds, streaming and `--stream false`).
