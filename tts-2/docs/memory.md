# Voice cache and memory

The model comes from the latest revision on Hugging Face (115 built-in voices). Figures marked "(47-voice revision)" were measured with an earlier revision that had 47 voices.

`kitten-tts-native` picks working settings by itself, so most users need nothing in this section. It is for running several streams, for the C API, and for using less GPU memory.

By default the engine (the C API) prepares all built-in voices on the GPU when it starts, so every voice is ready for its first request. `kitten-tts-native` speaks with one built-in voice per run, so without a `--voice-cache` option it prepares only that voice, right after the engine starts (`lru` with one voice per stream). That uses much less GPU memory with the default profile (about 13 GB instead of 21 GB with the earlier 47-voice revision; the difference is larger with the 115 voices of the current revision, not measured), and the first audio comes as soon as with all voices prepared. In the C API, `kt2n_voice_prepare()` does the same for a voice cache in `lru` mode.

Two settings make the engine use less GPU memory: the voice cache keeps only some voices on the GPU, and the low memory profile uses smaller working buffers. They can be combined. Neither changes the audio; what changes is the time to the first audio of a voice that is not on the GPU yet. With the same text, voice, seed, model and GPU, `--stream false` output is byte for byte the same with every cache mode and both profiles, when requests run one at a time (as `kitten-tts-native` runs them).

```sh
build/default/bin/kitten-tts-native --assets models/kitten-tts-2 --voice Bruno \
  --text "Speaking with less GPU memory." --voice-cache none --memory-profile low --output low.wav
```

## Where voices are kept

- **On disk:** `voices.kt2n` in the prepared model directory holds the built-in voices. Voices you save with `--save-voice` are in their own files.
- **In host memory:** with `lru` and `none`, the engine reads every built-in voice when it starts and keeps it there, about 7.6 MiB for the 47 voices of the earlier revision (more with 115 voices).
- **On the GPU:** the state the model needs to speak with the voice. One built-in voice took about 180 MiB of GPU memory and the largest 197 MiB (measured with the earlier 47-voice revision).

Preparing a voice copies it from host memory into GPU memory, in about 9 ms (47-voice revision). Removing a voice from the GPU keeps its host copy, so using it again only prepares it again.

## `--voice-cache all|lru|none`

- `all` (the C API default; also `kitten-tts-native`'s for a cloned or loaded voice) prepares every built-in voice when the engine starts and keeps it on the GPU.
- `lru` prepares no voice at startup. A voice is prepared the first time a request uses it and stays on the GPU until the cache needs room for another voice. Then the least recently used voice that no request is using is removed.
- `none` keeps a voice on the GPU only while requests use it. In the C API, an incremental text session keeps its voice between phrases until the session ends or the cache needs the room.

A voice that a request is using is never removed. If the cache is full of voices that are in use, a request for another voice waits until one of them is free. It is queued, not refused. Several requests for the same voice share one preparation.

## `--voice-cache-voices N` and `--voice-cache-mib N`

These limits need an explicit `--voice-cache lru` or `--voice-cache none`. Without a `--voice-cache` option, and with `all`, they are refused.

- `--voice-cache-voices N` keeps at most N voices on the GPU at once (1 to 4096).
- `--voice-cache-mib N` keeps at most N MiB of GPU memory for voices.

With both options, both limits apply. With only `--voice-cache-mib`, as many voices are kept as fit in the budget. With neither, `none` keeps one voice for each stream (`--max-streams`), and `lru` keeps 16 voices, or one for each stream when that is more. The engine sets aside GPU memory for these limits when it starts, so the limits, not the number of voices used so far, decide how much memory the cache takes. Plain `--voice-cache lru` therefore sets aside room for 16 voices (about 3 GB (measured with the earlier 47-voice revision)), which does not fit on 4 GB cards: there, use `--voice-cache none` or `--voice-cache lru --voice-cache-voices 1`. The engine refuses to start when the MiB budget cannot hold the largest built-in voice, and the message gives both sizes, for example `voice_cache_max_bytes (100 MiB) cannot hold the largest voice (197 MiB)` (47-voice revision).

Cloned and loaded voices go through the same cache. A cloned or loaded voice larger than the whole budget is refused when it is built or loaded.

## `--memory-profile auto|default|low`

`auto` is what `kitten-tts-native` uses when you do not pass `--memory-profile`. It uses the default profile when that fits the free GPU memory and `low` otherwise. When it uses `low`, it prints one line that says so and how much memory was free.

- The audio does not depend on the choice: with the same text, voice, seed and GPU, `--stream false` output is byte for byte the same with `low` and `default`.
- With `kitten-tts-native`'s own voice setting (one voice prepared), the default profile needs about 11 to 13 GB of GPU memory (measured with the earlier 47-voice revision), depending on the card (the low-memory line prints the number for yours), and keeps 10% of the GPU free; with `--voice-cache all` it needed about 21 GB with the earlier 47-voice revision, and needs more with the 115 voices of the current one (not measured). Below that, `auto` uses `low`: supported, not optimal. With several streams (C API), requests that arrive together are batched less than with the default profile (not measured). One request at a time runs as fast (`--stream false` real-time factor 0.050 with `low` and 0.051 with the default profile on a 16 GB RTX 5060 Ti; 47-voice revision).
- `auto` decides from the free memory before it loads the model. When the default profile clearly cannot fit, it does not try it. Only when it is close (about 12 to 14 GB free) does it try the default profile first; when that does not fit, starting takes a few seconds longer.
- When even `low` does not fit, the engine does not start, and the message says how much free memory one stream needs.
- With an explicit `--memory-profile default` on a GPU too small for it, the engine stops at start with `the default memory profile needs about N MiB or more of free GPU memory; this GPU has M MiB free after the CUDA context`, followed by what to use instead (exit status 1).
- In the C API, set `memory_profile` to `KT2N_MEMORY_PROFILE_AUTO` for the same choice. The C API default stays `KT2N_MEMORY_PROFILE_DEFAULT`, and `auto` never changes the voice cache setting.

`low` sizes the working buffers for the requests the engine can serve with your settings, such as `--max-streams`, instead of for the largest ones. On a GPU with room for them it also prepares at startup every GPU graph it can need, so it allocates no further GPU memory while it serves requests. All built-in voices and all voices cloned by this engine work with `low`. Cloning keeps at most 10 seconds of the recording as the decoder's reference; a voice whose decoder reference is longer than 250 tokens is refused with `low`.

### Small GPUs

On GPUs with about 8 GB or less, `low` also fits its buffers to the free memory when it starts. Supported, not optimal:

- It serves one stream, and a new segment can wait for the previous segment's memory.
- The decoder runs without CUDA graphs: `--stream false` is about 1.5% slower on an 8 GB laptop GPU (47-voice revision). The audio is the same.
- On about 4 GB, streaming text chunks longer than the fitted limit (about 330 to 480 characters, depending on the card; 47-voice revision) are cut into shorter pieces, which are spoken separately. This is the only setting that changes the audio; see [Output differences](consumer-gpus.md#output-differences). Set the environment variable `KT2N_FIT_SPLIT=0` to refuse such text instead. Complete files (`--stream false`) are never cut.
- Voice cloning is off (see [Voice cloning](../README.md#voice-cloning)). Saved voices (`--load-voice`) work.

The engine keeps 128 MiB or 1/64 of the GPU free, whichever is larger. When even the smallest fit does not fit, it does not start, and the message says how much free memory one stream needs, for example: `the low-memory profile needs about 2583 MiB of free GPU memory for one stream; this GPU has 1730 MiB free after the CUDA context (min_free_device_bytes 128 MiB)` (47-voice revision). A 2 GB GPU is therefore not supported. [docs/consumer-gpus.md](consumer-gpus.md#small-memory-cards) has the measured numbers and how the sizes are fitted.

A request that needs more memory than was fitted fails with the error code `gpu_memory_limit_exceeded` (`KT2N_ERR_GPU_MEMORY_LIMIT`, status 111; `kitten-tts-native` prints the reason and `(status 111)`). The message gives the size the request needs and the size fitted on this GPU. It is never shortened silently. A streaming request can deliver audio for its first segments and then fail this way on a later, longer segment.

## Memory settings in the C API

The fields `voice_cache_mode`, `voice_cache_max_voices`, `voice_cache_max_bytes` and `memory_profile` of `kt2n_engine_config` set the same options (see [include/kt2n/engine.h](../include/kt2n/engine.h)).

- `min_free_device_bytes` sets how much GPU memory must stay free after the engine starts. When it is 0: with the low profile, 128 MiB or 1/64 of the GPU, whichever is larger; with the default profile, 10% of the GPU for engines created with `KT2N_CFG_INFERENCE_ONLY` or `max_custom_voices` > 0, and no check for other engines.
- With `lru` and `none`, `kt2n_stats_json()` includes a `voice_cache` object. It counts hits, misses, removals and waits, gives the time taken to prepare voices, and gives the bytes used on the GPU and in host memory. On an engine fitted to a small GPU it also reports `fit_split_chunks`, the streaming chunks cut to fit.
- `kt2n_engine_low_memory_limits()` returns the fitted sizes, including `streaming_budget_tokens` when streaming segments are limited.
- With the low profile and saved voices, `custom_voice_max_prefix_ids` gives the longest language-model prompt of the voices you will load (`kt2n_voice_artifact_lm_prefix_ids()` reads it from a saved voice; `kitten-tts-native` does this for `--load-voice`). The fit then keeps context for that prompt instead of 1,024 ids, and `kt2n_voice_load()` refuses a longer voice with `gpu_memory_limit_exceeded`. A value above 1,024 is not enforced; the engine then plans for 1,024 ids, as with 0.
