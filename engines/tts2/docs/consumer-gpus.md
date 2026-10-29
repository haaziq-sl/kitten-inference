# Consumer GPUs

Speed, memory and audio for each GPU group. A real-time factor of 0.27 means 10 seconds of audio take 2.7 seconds to
produce.

## At a glance

| GPU | Status |
|---|---|
| GeForce RTX 30, 40 and 50 series | Supported: real-time factor 0.05 to 0.08 with `--stream false` |
| GeForce GTX 10, GTX 16 and RTX 20 series | Supported, not optimal: no bf16 tensor cores, so part of the math runs in software (real-time factor 0.22 to 0.37 on desktop cards with 6 GB or more) |
| Cards with 8 GB or less | Supported, not optimal: one stream, buffers fitted to the card (`--stream false` about 1.5% slower), no voice cloning; see [Small-memory cards](#small-memory-cards) |
| 4 GB cards | Supported, not optimal: as above, and long streaming text is cut where the fitted sizes limit it |
| Cards under 4 GB | Not supported: with less than about 2,583 MiB of free GPU memory the engine does not start and says so |
| AMD Radeon RX 7000 and RX 9000 | Supported: real-time factor 0.059 to 0.090; audio close to, not bit for bit the same as, NVIDIA; no voice cloning; RX 7600, RX 7600 XT, RX 7700 XT and 8 GB cards experimental; see [AMD Radeon](#amd-radeon) |
| 64-bit Windows with an NVIDIA GPU | Supported on Windows 10, 11 and Server 2022: the same audio as Linux on the same GPU model; voice cloning only with the pip package; see [Windows](#windows) |

Building a cloned voice needs about 11,300 MiB of free GPU memory with the low-memory profile, which the tool picks
when the default profile's 15,600 MiB does not fit. On 12 GB cards it fits when other programs use at most about
500 MiB of GPU memory. On cards with 8 GB or less it is off; see [Small-memory cards](#small-memory-cards). Memory
needs and settings for each card size are in [GPU memory](memory.md).

Other AMD GPUs (RX 6000 and earlier, Instinct) and NVIDIA GPUs other than compute capability 6.1, 7.5 and 8.0 or newer
(for example 6.0 and 7.0) are not supported: the engine stops at start and says so.

## Against vLLM and llama.cpp

Through the pip package with its default settings, one request at a time: the same 300 short texts (about 5.5 s of
speech each) and 12 long texts (about 64 s each) with the same voice and seeds. Medians; for vLLM and llama.cpp,
their time divided by the native engine's (short / long texts):

| GPU | Native, short / long | Native, first streamed audio | vLLM, 4-bit | vLLM, bf16 (its default) | llama.cpp (GPU) |
|---|---|---|---|---|---|
| RTX PRO 6000 96 GB | 82 ms / 0.95 s | 11.4 ms | 5.55x / 5.12x | 7.15x / 6.92x | 6.37x / 5.64x |
| RTX 4090 24 GB | 159 ms / 1.80 s | 22.1 ms | 5.22x / 4.49x | 6.89x / 6.37x | 5.43x / 5.04x |
| RTX 3060 12 GB | 459 ms / 5.07 s | 57.5 ms | 2.45x / 2.44x | 4.27x / 4.33x | 2.55x / 2.40x |

- vLLM and llama.cpp stream one whole sentence at a time, so on these short texts their first audio comes with the
  complete file (0.5 to 2.0 s).
- Word error rate is about the same for every system (0.00 to 0.05% on the short texts).
- On the RTX 3060 the tool picks the low-memory profile by itself: supported, not optimal, because several requests
  at once are batched less; one request at a time, as here, is not affected.

## Output differences

**NVIDIA.** On the same GPU model, seeded `--stream false` requests give byte for byte the same audio on Linux and
Windows, and with every memory profile, including the sizes fitted to small cards.

**Streaming on every GPU.** Where streamed audio is cut into decoder chunks depends on how fast speech is generated,
so streamed audio with the same seed can differ slightly from run to run on any GPU. `--stream false` is not
affected.

**Between GPU models.** Two different GPU models usually do not give bit for bit the same samples (the decoder's noise
depends on the number of processors on the GPU), but on GeForce RTX 30 and 40 cards the words and voice match an
RTX 4090's. The groups below can differ more. Word error rate and voice similarity against an RTX 4090 (RTX 4090
word error rate: 0.99%):

| GPU group | Why it differs | Measured |
|---|---|---|
| AMD Radeon RX 7000 and RX 9000 (default) | AMD's math library (exp, log, sin, cos) and the Radeon matrix instructions round differently, so sampled speech can differ | word error rate 1.17% on RX 7000 and 1.02% on RX 9000 cards; 4 of 72 outputs dropped a sentence, as on the RTX 4090; the voice is closer to the RTX 4090's than a different seed is |
| AMD Radeon with `KT2N_AMD_WMMA=0` | the matrix math runs in software | word error rate 1.24% (RTX 4090 on the same texts: 0.60%); voice similarity 0.966 |
| AMD Radeon, builds made with different ROCm versions | the compiler rounds differently | differ only at the level of rounding (about -65 dB), with the same words and voice |
| 4 GB cards, streaming only | a long streaming text chunk is split into shorter pieces, each spoken separately ([details](memory.md#small-gpus)) | word error rate 0.80% to 0.99%; on long texts on the same GPU, 1.43% with the split against 0.58% without. `KT2N_FIT_SPLIT=0` refuses such text instead. `--stream false` is never split |
| NVIDIA, builds made with CUDA 13 | another compiler version rounds differently | not bit for bit the same as the CUDA 12.8 build's; word error rate 1.39% against 1.42% (RTX 5060) |
| GeForce GTX 10, GTX 16 and RTX 20 | no bf16 tensor cores: the engine does the same math in software | word error rate 0.80%; voice similarity 0.957, against 0.910 for a different seed |

## GTX 10, GTX 16 and RTX 20

**Supported, not optimal:** these GPUs have no bf16 tensor-core instructions, so part of the language model's math
runs in software and they are slower than an RTX 30-series or newer card: real-time factor 0.22 to 0.37 with
`--stream false`, one request at a time, on desktop cards with 6 GB or more (cards with less memory also run with
[fitted sizes](#small-memory-cards)). With more than four requests at once the engine uses a different method,
which is faster there.

## Small-memory cards

Cards with 8 GB or less run the low-memory profile with its buffers fitted to the free memory. **Supported, not
optimal:** one stream, `--stream false` about 1.5% slower, no voice cloning (build and save a voice on a larger GPU,
then use it here with `--load-voice`), and on 4 GB cards long streaming text is cut into shorter pieces, which can
change the audio (see [Output differences](#output-differences)). Complete files (`--stream false`) are byte for byte
the same as with the standard sizes. On these cards the speed is set mostly by the card's math (GTX 10 and GTX 16
cards have no bf16 tensor cores), not by the smaller buffers.

The minimum is about **2,583 MiB of free GPU memory** after the CUDA context (up to about 3,000 MiB under Windows and
WSL2). Cards under 4 GB are not supported. Settings, limits and messages: [Small GPUs](memory.md#small-gpus).

## AMD Radeon

Radeon RX 7000 (RDNA3) and RX 9000 (RDNA4) GPUs run the engine with a ROCm build; see [amd-radeon.md](amd-radeon.md).
With the command-line tool's default settings, one request at a time: real-time factor 0.059 to 0.090 with
`--stream false`, first streamed audio after 0.05 to 0.07 s. Supported inside WSL2 on Windows 11, experimental on
Linux without WSL2.

No voice cloning on Radeon: build and save a voice on an NVIDIA GPU, then load it on the Radeon.

`KT2N_AMD_WMMA=0` turns the matrix instructions off. **Supported, not optimal:** all matrix math then runs in
software, 3.4 to 4.9 times slower: real-time factor 0.23 to 0.44.

The audio is close to, not bit for bit the same as, an NVIDIA GPU's; see [Output differences](#output-differences).

## Windows

The steps are in [windows.md](windows.md). Supported on Windows 10, 11 and Server 2022.
Real-time factor 0.074 with `--stream false` on an NVIDIA L4. With the pip package, Windows and Linux give byte for
byte the same audio on the same GPU model.
