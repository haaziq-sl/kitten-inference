/*
 * kt2n engine C ABI (frozen interface; fields are only appended).
 *
 * One native library, libkt2n_engine.so, serves Kitten TTS 2 end to end:
 * text normalisation and tokens -> LM -> conformer encoder -> 2-step MeanFlow
 * estimator -> HiFT -> postprocessing -> in-process encoders, plus the clone
 * voice build. The same ABI drives a Python adapter
 * (through a binding that releases the GIL on every call that can wait),
 * the command-line tool and the benchmark driver.
 *
 * Plain C (C11 / C++17 and later both compile it). Every struct passed IN
 * starts with `struct_size` so fields can be appended without breaking older
 * callers; the library reads only the bytes the caller declared. Handles are
 * 64-bit generation-checked ids, never pointers: a stale or foreign id fails
 * with KT2N_ERR_INVALID_HANDLE instead of touching freed memory.
 *
 * Threading. Every function is thread-safe. Only kt2n_engine_create,
 * kt2n_engine_destroy, kt2n_poll (with timeout_ms != 0) and
 * kt2n_voice_prepare may wait; every other call returns in microseconds and
 * never waits for the GPU. Callers own the engine pointer's lifetime: stop new
 * calls and finish non-poll calls before destroy, and return all previously
 * borrowed events. Already blocked polls and voice preparations may overlap
 * destroy; they wake (zero events, KT2N_ERR_NOT_READY) before it frees the
 * engine. No call may begin once destroy starts. No call
 * allocates device memory after create (one device arena + one pinned arena
 * are allocated there); arena exhaustion is admission back-pressure.
 *
 * Clock. Every time is CLOCK_MONOTONIC in nanoseconds. On Linux CPython's
 * time.perf_counter() reads the same clock, so a Python caller turns an event's
 * at_ns into the reference timeline `at_ms` exactly as the reference does:
 * at_ms = (at_ns / 1e9 - the timeline's start in seconds) * 1000.
 * On Windows the clock is QueryPerformanceCounter in nanoseconds: the clock of
 * std::chrono::steady_clock (MSVC) and of CPython's time.perf_counter_ns().
 *
 * "The reference" below means the reference implementation whose behaviour
 * this engine reproduces.
 */
#ifndef KT2N_ENGINE_H_
#define KT2N_ENGINE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
/* Windows: the engine DLL exports the API (KT2N_BUILDING_ENGINE); programs
 * import it. KT2N_ENGINE_STATIC: linked from the static library instead. */
#if defined(KT2N_ENGINE_STATIC)
#define KT2N_API
#elif defined(KT2N_BUILDING_ENGINE)
#define KT2N_API __declspec(dllexport)
#else
#define KT2N_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define KT2N_API __attribute__((visibility("default")))
#else
#define KT2N_API
#endif

/* Bumped on any incompatible change (a field removed, moved or re-typed, an
 * enum value renumbered). Appending fields to a struct_size struct or adding
 * enum values does not bump it. */
#define KT2N_ABI_VERSION 2u

/* ------------------------------------------------------------------------- */
/* Handles                                                                   */
/* ------------------------------------------------------------------------- */
typedef struct kt2n_engine kt2n_engine; /* opaque */
typedef uint64_t kt2n_request_id;       /* one synthesis request */
typedef uint64_t kt2n_session_id;       /* one incremental text session */
typedef uint64_t kt2n_voice_id;         /* a loaded voice (built-in, clone, saved) */
typedef uint64_t kt2n_job_id;           /* an asynchronous voice build */
#define KT2N_INVALID_ID ((uint64_t)0)

/* ------------------------------------------------------------------------- */
/* Status codes and the reference public error codes                          */
/* ------------------------------------------------------------------------- */
/* Every function returns a kt2n_status (0 = OK). Codes >= 100 are public
 * errors: kt2n_error_lookup() gives the exact `error.code` string, HTTP status
 * and `error.type` the reference answers with (its custom-voice errors,
 * HTTP status error codes and OpenAI-style error types;
 * its error envelope). Codes below
 * 100 are internal: a caller should not show them to a client as such. */
typedef enum kt2n_status {
  KT2N_OK = 0,
  /* internal */
  KT2N_ERR_INVALID_ARGUMENT = 1, /* caller bug (NULL, bad struct_size); public: internal_error */
  KT2N_ERR_INVALID_HANDLE = 2,   /* unknown or released id */
  KT2N_ERR_ABI_MISMATCH = 3,     /* config.abi_version != KT2N_ABI_VERSION */
  KT2N_ERR_AGAIN = 4,            /* admission full right now; wait on the event fd and retry */
  KT2N_ERR_CANCELLED = 5,        /* cancelled (client gone / shutdown): no public body */
  KT2N_ERR_NOT_READY = 6,        /* create still warming / destroy in progress */
  KT2N_ERR_UNSUPPORTED = 7,      /* a lever value or path the native engine does not implement:
                                    create() fails closed instead of serving it differently */
  KT2N_ERR_TOO_SMALL = 8,        /* output buffer too small (size returned) */

  /* public: request validation (HTTP 4xx) */
  KT2N_ERR_INVALID_REQUEST = 100,         /* invalid_request_error 400; `param` names the field
                                             (sampler fields as in the reference) */
  KT2N_ERR_INVALID_INPUT = 101,           /* invalid_input 400 ("input must not be blank") */
  KT2N_ERR_INVALID_VOICE = 102,           /* invalid_voice 400 */
  KT2N_ERR_INVALID_RESPONSE_FORMAT = 103, /* invalid_response_format 400 */
  KT2N_ERR_INVALID_STREAM_FORMAT = 104,   /* invalid_stream_format 400 */
  KT2N_ERR_REQUEST_TOO_LARGE = 105,       /* request_too_large 413 (input > 40,000 chars,
                                             reference audio > 10 MiB decoded) */
  KT2N_ERR_CUSTOM_VOICE_REALTIME_NOT_SUPPORTED = 106, /* custom_voice_realtime_not_supported 400 */
  KT2N_ERR_INPUT_ALREADY_CLOSED = 107,    /* input_already_closed 400; WS close 1008 */
  KT2N_ERR_INVALID_SESSION_CONFIGURATION = 108, /* invalid_session_configuration 400 */
  KT2N_ERR_MODEL_NOT_FOUND = 109,         /* model_not_found 404 */
  KT2N_ERR_VOICE_NOT_FOUND = 110,         /* voice_not_found 404 */
  KT2N_ERR_GPU_MEMORY_LIMIT = 111,        /* gpu_memory_limit_exceeded 413: with KT2N_MEMORY_PROFILE_LOW
                                             on a GPU too small for the standard sizes, a segment needs more
                                             language-model context, decoder pages or vocoder frames than the
                                             engine fitted to this GPU at create (kt2n_engine_low_memory_limits) */

  /* public: voice build / clone paths */
  KT2N_ERR_INVALID_AUDIO = 200,                 /* invalid_audio 400 (no decodable audio) */
  KT2N_ERR_INVALID_AUDIO_DURATION = 201,        /* invalid_audio_duration 400 (2-30 s) */
  KT2N_ERR_INVALID_AUDIO_SIGNAL = 202,          /* invalid_audio_signal 400 (silent / too quiet) */
  KT2N_ERR_INVALID_REFERENCE_AUDIO = 203,       /* invalid_reference_audio 400 */
  KT2N_ERR_REFERENCE_ALIGNMENT_FAILED = 204,    /* invalid_reference_audio 422 */
  KT2N_ERR_SPEAKER_EMBEDDING_FAILED = 205,      /* speaker_embedding_failed 422 */
  KT2N_ERR_SPEAKER_VERIFICATION_FAILED = 206,   /* speaker_verification_failed 422 */
  KT2N_ERR_CONSENT_PHRASE_MISMATCH = 207,       /* consent_phrase_mismatch 400 */
  KT2N_ERR_UNSUPPORTED_CONSENT_LANGUAGE = 208,  /* unsupported_consent_language 400 */
  KT2N_ERR_CONSENT_REQUIRES_SPEAKER_VERIFICATION = 209, /* consent_requires_speaker_verification 409 */
  KT2N_ERR_VOICE_ARTIFACT_VERSION_MISMATCH = 210, /* voice_artifact_version_mismatch 409 */
  KT2N_ERR_VOICE_ARTIFACT_REBUILDING = 211,     /* voice_artifact_rebuilding 503 */
  KT2N_ERR_CUSTOM_VOICE_CREATION_BUSY = 212,    /* custom_voice_creation_busy 429 (4 slots / queue 16) */
  KT2N_ERR_CUSTOM_VOICE_SERVICE_ERROR = 213,    /* custom_voice_service_error 503 */
  KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE = 214, /* custom_voice_service_unavailable 503 */
  KT2N_ERR_INVALID_REFERENCE_TRANSCRIPT = 215,  /* invalid_reference_transcript 400, param reference_transcript
                                                   (missing, > 4096 chars / 16 KiB, control characters:
                                                   the reference transcript rule) */
  KT2N_ERR_INVALID_CONSENT_TRANSCRIPT = 216,    /* invalid_consent_transcript 400, param consent_transcript
                                                   (same rule, the consent path) */

  /* public: server (HTTP 5xx) */
  KT2N_ERR_SPEECH_GENERATION_FAILED = 300, /* speech_generation_failed 503 before any audio;
                                              after audio: the transfer is aborted;
                                              WS: error event + close 1011 */
  KT2N_ERR_DEADLINE_EXCEEDED = 301,        /* public code speech_generation_failed (the reference raises
                                              TimeoutError -> same envelope); kept apart for stats */
  KT2N_ERR_SERVICE_UNAVAILABLE = 302,      /* service_unavailable 503 (engine owner failed) */
  KT2N_ERR_INTERNAL = 303                  /* internal_error 500 */
} kt2n_status;

typedef struct kt2n_error_info {
  int32_t status;      /* kt2n_status */
  int32_t http_status; /* 400, 404, 409, 413, 422, 429, 500, 503; 0 for internal codes */
  const char* code;    /* the reference `error.code`, e.g. "invalid_request_error" (static storage) */
  const char* type;    /* the reference `error.type`: invalid_request_error | authentication_error |
                          rate_limit_error | server_error (static storage) */
} kt2n_error_info;

/* Static table lookup; never NULL (unknown status -> internal_error 500). */
KT2N_API const kt2n_error_info* kt2n_error_lookup(int32_t status);

/* Detail of one failure. Optional out-parameter of most calls (NULL = not wanted). */
typedef struct kt2n_error {
  int32_t status;      /* kt2n_status */
  int32_t http_status; /* as kt2n_error_lookup(status)->http_status */
  char code[48];       /* public error.code, NUL-terminated */
  char param[48];      /* public error.param ("" = null), e.g. "temperature", "voice", "input" */
  char message[400];   /* public error.message, the reference wording where it exists */
} kt2n_error;

/* ------------------------------------------------------------------------- */
/* Engine                                                                    */
/* ------------------------------------------------------------------------- */
enum kt2n_config_flags {
  KT2N_CFG_TIMELINE_ALLOWED = 1u << 0, /* lab: requests may enable the benchmark timeline */
  KT2N_CFG_X_SCHED = 1u << 1,          /* experimental option X-SCHED (deadline / urgent-class dispatch).
                                          NEVER in a release build: releases keep the reference timing rules */
  KT2N_CFG_NO_GRAPHS = 1u << 2,        /* debug: no CUDA graphs (eager launches) */
  KT2N_CFG_SPIN_KEEPWARM = 1u << 3,    /* keep the engine awake between requests */
  KT2N_CFG_INFERENCE_ONLY = 1u << 4   /* reserve only configured voices; disable clone builds */
};

typedef struct kt2n_engine_config {
  uint32_t struct_size;  /* sizeof(kt2n_engine_config) */
  uint32_t abi_version;  /* KT2N_ABI_VERSION */
  uint32_t flags;        /* kt2n_config_flags */
  int32_t device;        /* CUDA ordinal */

  /* Files (the container format in tools/kt2n_assets/format.py). */
  /* File and directory paths are UTF-8. On Windows, non-ASCII paths need a UTF-8 process code page (an
   * application manifest with activeCodePage UTF-8); create refuses them otherwise. */
  const char* lm_file;        /* model.kt2n: packed language model */
  const char* s3gen_file;     /* s3gen.kt2n: FP32 encoder, BF16 estimator, folded FP32 HiFT */
  const char* voices_file;    /* voices.kt2n: prepared voice conditioning */
  const char* tokenizer_json; /* tokenizer matching the language model */
  const char* text_data_dir;  /* prepared normalization grammars and regex tables */

  const char* reserved_ptr0; /* reserved; must be NULL */
  const char* reserved_ptr1; /* reserved; must be NULL */

  /* Capacity. 0 = default. */
  int32_t max_streams;          /* concurrent requests/phrases; inference-only default is 1 */
  int32_t max_lm_rows;          /* LM decode rows (step graphs for buckets 1-16, 20, 24, 32, 40, 48, 64) */
  int32_t render_pages;         /* 64-frame K/V pages; 0 = voice prefixes plus 40 pages per stream */
  int32_t inline_voice_lru;     /* inline clone LRU entries (reference default: 32) */
  int32_t clone_slots;          /* concurrent voice builds (reference default: 4) */
  int32_t clone_queue;          /* queued voice builds before custom_voice_creation_busy (reference default: 16) */
  int32_t clone_timeout_ms;     /* reference default: 120,000 */
  uint64_t device_arena_bytes;  /* initialization GPU allocation budget; 0 = available VRAM */
  uint64_t pinned_arena_bytes;  /* reserved; must be 0 */

  /* CPU threads (sized for 4 physical cores / 8 vCPU). The conductor
   * and the render dispatcher are always 1 thread each. 0 = default. */
  int32_t threads_text;  /* 0 = default 4 */
  int32_t threads_post;  /* default 2 (post + encoders) */
  int32_t threads_voice; /* default 1 */
  const char* cpuset;    /* optional "0-7" style list the engine pins its threads to; NULL = inherit */

  /* Appended (append-only, no ABI bump): the clone-build networks until s3gen.kt2n carries
   * them. A directory holding s3gen.safetensors (the S3Gen weights: speech tokenizer + speaker encoder),
   * voicebuild_consts.safetensors and voicebuild_pyannote.safetensors (cloning converters). NULL = clone
   * builds unavailable (kt2n_voice_build -> custom_voice_service_unavailable). */
  const char* voice_build_dir;

  /* Optional custom-voice capacity. 0 preserves the reference cache reserve;
   * a positive value reserves this many additional LM/render voice slots.
   * Prefix storage uses the reference allowance of 512 tokens per slot;
   * unusually long references can exhaust that pool before the slot count.
   * Ignored with KT2N_CFG_INFERENCE_ONLY. */
  int32_t max_custom_voices;

  /* Voice cache and memory controls (appended; no ABI bump). The library reads
   * these five fields only when struct_size covers all of them (up to and
   * including min_free_device_bytes); a smaller struct_size uses the defaults
   * (every field 0). */
  int32_t voice_cache_mode;       /* kt2n_voice_cache_mode; 0 = KT2N_VOICE_CACHE_ALL */
  int32_t voice_cache_max_voices; /* LRU / NONE: voices resident on the GPU at once (0 = see below) */
  uint64_t voice_cache_max_bytes; /* LRU / NONE: GPU bytes for resident voices (LM prefix store, renderer
                                     prefix pages and voice table rows); 0 = see below */
  int32_t memory_profile;         /* kt2n_memory_profile; 0 = KT2N_MEMORY_PROFILE_DEFAULT */
  uint64_t min_free_device_bytes; /* create fails when less device memory than this stays free after
                                     initialization; 0 = the built-in rule */

  /* Appended (no ABI bump; read only when struct_size covers it). KT2N_MEMORY_PROFILE_LOW with custom voices:
   * > 0 = the custom voices this engine will use have language-model prompts of at most this many ids
   * (kt2n_voice_artifact_lm_prefix_ids of a saved voice), so the low-memory fit keeps context and voice-cache room
   * for that instead of 1,024 ids; kt2n_voice_load refuses a longer voice with KT2N_ERR_GPU_MEMORY_LIMIT (every
   * voice-cache mode). 0 = 1,024 (no explicit bound). A value above 1,024 is not enforced at load; the fit then
   * plans for 1,024 ids, as with 0. Ignored by the default profile and with KT2N_CFG_INFERENCE_ONLY. */
  int32_t custom_voice_max_prefix_ids;
} kt2n_engine_config;

/* Where voice conditioning lives on the GPU.
 *   KT2N_VOICE_CACHE_ALL   every catalogue voice is prepared at create and stays
 *                          resident (the default, and the fastest first audio).
 *   KT2N_VOICE_CACHE_LRU   a voice is prepared on its first use and kept until
 *                          the cache needs room; the least recently used voice
 *                          that no request is using is dropped first.
 *   KT2N_VOICE_CACHE_NONE  a voice is prepared for the requests that use it and
 *                          dropped as soon as none does (smallest footprint).
 *                          A realtime session's voice stays prepared between
 *                          its phrases until the session ends or the cache
 *                          needs the room.
 * In LRU and NONE modes the cache holds at most voice_cache_max_voices voices
 * and voice_cache_max_bytes bytes. With both 0, NONE keeps one voice per
 * stream the engine serves at once (max_streams, or fewer when the GPU cannot
 * serve that many) and LRU keeps 16 voices, or one per stream when that is
 * more. A voice that is in use is never
 * dropped; a request whose voice does not fit yet waits until a voice is free
 * (it is queued, not failed). A voice stays in use while new requests for it
 * arrive before the earlier ones finish, so when every voice the cache holds
 * is kept busy that way, a waiting request waits until that traffic pauses
 * (or its deadline passes). Create fails when the byte budget cannot hold
 * the largest catalogue voice. Cloned and loaded voices are budgeted at up to
 * 1,024 LM prefix ids and 250 decoder prompt tokens each; one larger than the
 * whole budget (possible with a budget of one voice) is refused when it is
 * loaded or built, where KT2N_VOICE_CACHE_ALL would accept it. For requests
 * served one at a time, results are identical in every mode for every voice
 * the cache accepts; only first-audio latency of a voice that is not resident
 * differs. With several requests in flight, a request waiting for its voice
 * changes the order in which the others start, which can change their audio
 * as running a different mix of requests does. */
typedef enum kt2n_voice_cache_mode {
  KT2N_VOICE_CACHE_ALL = 0,
  KT2N_VOICE_CACHE_LRU = 1,
  KT2N_VOICE_CACHE_NONE = 2
} kt2n_voice_cache_mode;

typedef enum kt2n_memory_profile {
  KT2N_MEMORY_PROFILE_DEFAULT = 0,
  KT2N_MEMORY_PROFILE_LOW = 1, /* size buffers for small GPUs; see below */
  KT2N_MEMORY_PROFILE_AUTO = 2 /* the default profile when it fits this GPU, otherwise the low profile; see below */
} kt2n_memory_profile;
/* KT2N_MEMORY_PROFILE_AUTO creates the engine with the default profile, exactly as KT2N_MEMORY_PROFILE_DEFAULT,
 * when it fits the free GPU memory, and otherwise with the low profile, exactly as KT2N_MEMORY_PROFILE_LOW (one line
 * on stderr says so, with the free memory). It first runs the low profile's fit: when that has to shrink its sizes,
 * or when the free memory is below the default profile's smallest possible need (the low profile's standard need +
 * 2 GiB + the default profile's free-memory floor), the default profile cannot fit and the engine is the low profile;
 * otherwise it tries the default profile and falls back to the low profile if that create fails. When even the low
 * profile does not fit, create fails with its KT2N_ERR_TOO_SMALL message (how much free memory one stream needs).
 * The voice cache mode is never changed. kt2n_stats_json then has "memory_profile_auto": "default" or "low".
 * kitten-tts-native uses it unless --memory-profile is given. */
/* KT2N_MEMORY_PROFILE_DEFAULT (also a zero-initialised config) on a GPU with less than 22 GiB free: create first runs
 * the low profile's fit without allocating (it reads the tokenizer and voices file, which the create then reuses) and
 * fails with KT2N_ERR_TOO_SMALL, "the default memory profile needs about N MiB ...", before any model memory is
 * allocated when the free memory is below the default profile's smallest possible need (as above, plus the pool pages
 * of every voice prefix with KT2N_VOICE_CACHE_ALL). It does not check when render_pages is set. */
/* KT2N_MEMORY_PROFILE_LOW sizes buffers for the calls the engine can make
 * instead of the largest ones. Requests served one at a time give the same
 * results as the default profile. Requests that arrive together may start in
 * smaller prefill batches than in the default profile, which changes how they
 * share decoding steps, as different arrival times do in either profile. Voices
 * whose decoder prompt is longer than 250 tokens or the longest catalogue
 * prompt, whichever is larger, are refused (the default profile accepts them).
 * The low profile also fits itself to the GPU at create: it measures the free
 * memory after the CUDA context, keeps min_free_device_bytes free (0 = the
 * larger of 128 MiB and 1/64 of the GPU), and when the standard low sizes do not
 * fit it sizes the decoder's complete-file workspace to the exact need of its
 * calls, keeps one language-model context slot per stream (a new segment waits
 * for the previous one's slot), then shrinks the vocoder's frame limit and the
 * language model's context, never below what the longest segment needs at its
 * token budget (streaming: 2,341 tokens, 532 characters after normalisation;
 * complete-file: 1,000), and runs the decoder without CUDA graphs (same
 * results, no graph memory). On a GPU too small for that (about 4 GB), complete-
 * file segments keep their full 1,000-token budget and streaming segments are
 * limited to the largest budget the fitted context and page pool hold
 * (streaming_budget_tokens below). There, a streaming text chunk whose budget is
 * larger is split at sentence, clause or word breaks into more, shorter segments
 * that fit (not the same audio as on a larger GPU; kt2n_stats_json counts them
 * as "fit_split_chunks"; the environment variable KT2N_FIT_SPLIT=0 refuses them
 * instead). A segment that still does not fit fails with
 * KT2N_ERR_GPU_MEMORY_LIMIT before any of its audio, never silently shortened;
 * when even the minimum does not fit, create fails with KT2N_ERR_TOO_SMALL and
 * says how much free memory one stream needs. Voice cloning is off on a fitted
 * GPU, also with voice_build_dir set: the cloning model is not loaded and
 * kt2n_voice_build returns KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE (it needs
 * about 4.4 GB more); saved voices load. kt2n_engine_low_memory_limits reports
 * the fit. */
typedef struct kt2n_low_memory_limits {
  uint32_t struct_size;
  int32_t fitted;              /* 1 = sizes were reduced to fit this GPU; 0 = standard sizes (or default profile) */
  int32_t lm_context_tokens;   /* language-model context per segment: prompt (voice prefix + text) + budget */
  int32_t decoder_pages;       /* decoder page pool (64 frames = 32 tokens each, shared by voice prefixes) */
  int32_t vocoder_frames;      /* longest complete-file segment the vocoder renders in one call, in frames
                                  (2 per token, plus 12) */
  int32_t render_graphs;       /* 1 = decoder runs as CUDA graphs, 0 = eager launches */
  uint64_t free_after_context; /* free device bytes measured at create, after the CUDA context */
  uint64_t min_free_bytes;     /* the floor create kept free */
  int32_t streaming_budget_tokens; /* > 0: the largest streaming segment budget (tokens) this GPU holds; a longer
                                      streaming text chunk is split into shorter segments (KT2N_FIT_SPLIT=0:
                                      refused with KT2N_ERR_GPU_MEMORY_LIMIT). 0 = the standard cap (2,341
                                      tokens, 532 characters) */
} kt2n_low_memory_limits;
KT2N_API int32_t kt2n_engine_low_memory_limits(kt2n_engine* engine, kt2n_low_memory_limits* out);

KT2N_API uint32_t kt2n_abi_version(void);
/* Static string: git sha, build time, CUDA version, compiled archs. */
KT2N_API const char* kt2n_build_info(void);
/* Capability tokens an engine includes (as plain substrings) in kt2n_build_info().
 * KT2N_CAP_AUDIO_OPENING_SKIP: every AUDIO event carries kt2n_event.opening_skipped_tokens. */
#define KT2N_CAP_AUDIO_OPENING_SKIP "cap:audio.opening_skipped_tokens"
#define KT2N_CAP_VOICE_CLONING "cap:voice.cloning"

/* Loads every file, allocates both arenas, builds the voice prefixes, captures
 * the graphs and warms up. Blocks (seconds). */
KT2N_API int32_t kt2n_engine_create(const kt2n_engine_config* config, kt2n_engine** out, kt2n_error* err);
/* Cancels everything (KT2N_ERR_CANCELLED to each owner), waits for the device
 * to go idle, wakes and drains already blocked polls and voice preparations
 * (kt2n_voice_prepare returns KT2N_ERR_NOT_READY), then frees everything.
 * The pointer lifetime rules above apply. The event fd is closed. Safe on NULL. */
KT2N_API void kt2n_engine_destroy(kt2n_engine* engine);

/* ------------------------------------------------------------------------- */
/* Voices                                                                    */
/* ------------------------------------------------------------------------- */
typedef enum kt2n_voice_kind {
  KT2N_VOICE_BUILTIN = 0, /* voices.kt2n; immutable and never removed (kt2n_voice_evict refuses it). With
                             KT2N_VOICE_CACHE_LRU / _NONE its GPU state may be dropped and prepared again on
                             a later use; its id stays valid */
  KT2N_VOICE_INLINE = 1,  /* one-shot clone from a request (LRU of inline_voice_lru) */
  KT2N_VOICE_SAVED = 2    /* a stored custom voice loaded from its artifact */
} kt2n_voice_kind;

typedef struct kt2n_voice_info {
  uint32_t struct_size;
  int32_t kind;            /* kt2n_voice_kind */
  char name[96];           /* internal voice id, e.g. "03_radio_dj_smooth_male"; clones: the build's key */
  char language[16];       /* e.g. "en" (catalogue language code, lower-case primary subtag) */
  int32_t prompt_tokens;   /* decoder prompt tokens Tp (built-ins: 201..250) */
  int32_t prompt_frames;   /* 2 * Tp */
  int32_t lm_prefix_ids;   /* LM reference prompt length */
  int32_t hidden;          /* 1 = the 48th voice: nearest built-in voice projection only, not public */
  float peak_gain_db;      /* streamed peak-norm start gain (custom: the catalogue median) */
  int32_t refs;            /* in-flight requests holding it */
} kt2n_voice_info;

/* Built-in id or public alias (the reference catalogue + legacy voice names)
 * -> id. Unknown: KT2N_ERR_INVALID_VOICE. */
KT2N_API int32_t kt2n_voice_lookup(kt2n_engine* engine, const char* name, kt2n_voice_id* out, kt2n_error* err);
KT2N_API int32_t kt2n_voice_info_get(kt2n_engine* engine, kt2n_voice_id voice, kt2n_voice_info* out);
/* The voice's 512-d speaker embedding (pyannote; consent verification and
 * nearest built-in voice lookup use it). */
KT2N_API int32_t kt2n_voice_speaker(kt2n_engine* engine, kt2n_voice_id voice, float out[512]);

enum kt2n_voice_build_flags {
  KT2N_VB_INLINE = 1u << 0,       /* keep in the inline LRU under identity_key */
  KT2N_VB_SAVED = 1u << 1,        /* the result will be exported and stored (kt2n_voice_export) */
  KT2N_VB_DECODED_PCM = 1u << 2,  /* input is float PCM (pcm/pcm_samples/pcm_rate), not encoded bytes */
  KT2N_VB_CONSENT = 1u << 3,      /* this audio is a consent recording: validate its transcript/language,
                                     then decode + pyannote only (as the reference consent create does).
                                     The VOICE_BUILT voice is speaker-only: read kt2n_voice_speaker()
                                     for the store, then kt2n_voice_evict(); submit with it fails
                                     (KT2N_ERR_INVALID_VOICE) */
  KT2N_VB_VERIFY_SPEAKER = 1u << 4 /* compare against consent_speaker (speaker_verification_failed) */
};

/* Clone build inputs. Every documented clone path (consent, no consent, admin,
 * inline, saved) maps onto these flags; the API layer keeps auth/ownership. */
typedef struct kt2n_voice_build_params {
  uint32_t struct_size;
  uint32_t flags;                   /* kt2n_voice_build_flags */
  const void* audio;                /* encoded reference audio (<= 10 MiB decoded) */
  size_t audio_len;
  const char* audio_format;         /* "mp3"|"wav"|"ogg"|"aac"|"flac"|"webm"|"mp4" (the reference inline audio formats) */
  const float* pcm;                 /* with KT2N_VB_DECODED_PCM: mono float PCM */
  size_t pcm_samples;
  int32_t pcm_rate;
  int32_t consent_speaker_dim;      /* 512 with KT2N_VB_VERIFY_SPEAKER, else 0 */
  const float* consent_speaker;     /* the consent's stored speaker embedding */
  const char* reference_transcript; /* REQUIRED, 1..4096 chars and <= 16 KiB, no control characters.
                                       The caller transcribes the reference with an
                                       ASR when the client sent none (the reference
                                       reference_transcription_failed), so the engine always gets one.
                                       With KT2N_VB_CONSENT it is the consent transcript, checked against
                                       the consent phrase (the reference consent_phrase_mismatch) */
  const char* language;             /* BCP 47 tag; consent: the reference supports English only */
  const char* identity_key;         /* inline LRU key (in the reference: request-bound lease identity) */
  int64_t deadline_ns;              /* absolute; 0 = clone_timeout_ms from now */
  uint64_t user_data;               /* echoed in the KT2N_EVENT_VOICE_BUILT event */
} kt2n_voice_build_params;

/* Queues a build (admission: clone_slots running + clone_queue waiting, else
 * KT2N_ERR_CUSTOM_VOICE_CREATION_BUSY). The result arrives as one
 * KT2N_EVENT_VOICE_BUILT event (voice id, or status + error). An inline key
 * already in the LRU completes immediately with the cached voice. */
KT2N_API int32_t kt2n_voice_build(kt2n_engine* engine, const kt2n_voice_build_params* params, kt2n_job_id* out,
                                  kt2n_error* err);
/* Optional additive ABI: atomically cancel a build only while it remains queued.
 * *cancelled is 1 if marked cancelled under the dequeue mutex, otherwise 0.
 * An admitted, terminal or unknown job is unchanged. Repeated calls are safe.
 * Older libraries may omit this symbol; bindings must then report false. */
KT2N_API int32_t kt2n_voice_build_cancel_queued(kt2n_engine* engine, kt2n_job_id job, int32_t* cancelled);
/* Saved voices: the artifact is a one-voice voices.kt2n blob (the container format in tools/kt2n_assets/format.py).
 * Reference-format artifacts in the voice store are converted offline or
 * rebuilt (the reference voice_artifact_rebuilding path); load never guesses. */
/* The language-model prompt length (ids) of a saved-voice artifact, for kt2n_engine_config::
 * custom_voice_max_prefix_ids before an engine exists; a negative value when the artifact cannot be read. */
KT2N_API int32_t kt2n_voice_artifact_lm_prefix_ids(const void* artifact, size_t len);
KT2N_API int32_t kt2n_voice_load(kt2n_engine* engine, const void* artifact, size_t len, kt2n_voice_id* out,
                                 kt2n_error* err);
/* Writes the one-voice artifact. buf == NULL or too small: *len = bytes needed,
 * returns KT2N_ERR_TOO_SMALL. */
KT2N_API int32_t kt2n_voice_export(kt2n_engine* engine, kt2n_voice_id voice, void* buf, size_t cap, size_t* len);
/* Drops the engine's reference; requests already holding the voice finish
 * with it. Built-ins cannot be evicted (KT2N_ERR_INVALID_ARGUMENT). */
KT2N_API int32_t kt2n_voice_evict(kt2n_engine* engine, kt2n_voice_id voice);
/* KT2N_VOICE_CACHE_LRU: prepares the voice on the GPU now instead of on its first request, and waits until it is
 * ready (usually milliseconds, at most about 30 s), so that request's first audio does not include preparing it. It
 * may wait, so call it before kt2n_engine_destroy, like any other call; one still waiting when destroy starts returns
 * KT2N_ERR_NOT_READY. Nothing is dropped to make room: when the cache is full, the voice is prepared on first use as
 * before (KT2N_OK). With KT2N_VOICE_CACHE_ALL every voice is prepared at create, and with _NONE a voice is prepared for
 * each request, so this returns KT2N_OK at once. Unknown voice: KT2N_ERR_INVALID_VOICE; a load that failed (after its
 * retries) or a failed engine: KT2N_ERR_SERVICE_UNAVAILABLE. kitten-tts-native calls it right after create. */
KT2N_API int32_t kt2n_voice_prepare(kt2n_engine* engine, kt2n_voice_id voice, kt2n_error* err);

/* ------------------------------------------------------------------------- */
/* Requests                                                                  */
/* ------------------------------------------------------------------------- */
typedef enum kt2n_audio_format {
  KT2N_FORMAT_PCM = 0, /* raw s16le, 24 kHz mono */
  KT2N_FORMAT_WAV = 1, /* streaming WAV header (the reference streaming_wav_header) or complete header */
  KT2N_FORMAT_MP3 = 2, /* libmp3lame 128k, -write_xing 0 */
  KT2N_FORMAT_OPUS = 3,/* libopus 96k */
  KT2N_FORMAT_AAC = 4,
  KT2N_FORMAT_FLAC = 5,
  /* Appended (an appended enum value, no ABI bump). Host float32 LE,
   * 24 kHz mono, at the reference engine -> transport boundary: what
   * the reference engine's audio stream yields (after the engine's trims, speed stretch and
   * startup hold, BEFORE normalize_audio + peak norm + PCM16 / encoder).
   * A caller that levels and encodes the audio itself asks for it (levelling and encoding
   * then stay exactly as in the reference). The in-process post/encoder path is the bytes formats
   * above. */
  KT2N_FORMAT_F32 = 6
} kt2n_audio_format;

typedef enum kt2n_output_mode {
  KT2N_OUTPUT_STREAM = 0,      /* progressive chunks (HTTP audio / SSE / binary transports) */
  KT2N_OUTPUT_STREAM_FILE = 1, /* the buffered complete WAV of stream_format=audio + wav: the streaming
                                  path, then the demo peak formula over the whole file
                                  (as the reference levels a full file) */
  KT2N_OUTPUT_COMPLETE = 2     /* `stream: false`: one complete file in any format, DEMO path
                                  (the reference complete-file path; whole render + 3 x 4299) */
} kt2n_output_mode;

typedef enum kt2n_speech_mode {
  KT2N_SPEECH_MODE_DEFAULT = 0, /* the default speech mode */
  KT2N_SPEECH_MODE_FAST = 1,
  KT2N_SPEECH_MODE_QUALITY = 2,
  KT2N_SPEECH_MODE_DEMO = 3
} kt2n_speech_mode;

/* Public sampler fields. Only the fields whose
 * bit is set in `present` were sent; the engine validates them (ranges below)
 * and resolves the preset exactly like the reference resolve with the default
 * mode, expressive. "Advanced" settings = explicit fields. */
typedef enum kt2n_sampler_mode {
  KT2N_SAMPLER_UNSET = 0,
  KT2N_SAMPLER_STABLE = 1,    /* temperature 0.8, top_p 0.8, top_k 50, min_p 0 */
  KT2N_SAMPLER_EXPRESSIVE = 2 /* temperature 0.9, top_p 0.9, top_k 50, min_p 0 */
} kt2n_sampler_mode;

enum kt2n_sampler_present {
  KT2N_SAMPLER_HAS_MODE = 1u << 0,
  KT2N_SAMPLER_HAS_TEMPERATURE = 1u << 1, /* 0.3 .. 2.0 */
  KT2N_SAMPLER_HAS_TOP_P = 1u << 2,       /* 0.5 .. 1.0 */
  KT2N_SAMPLER_HAS_TOP_K = 1u << 3,       /* integer 0 .. 200 (0 = no top-k) */
  KT2N_SAMPLER_HAS_MIN_P = 1u << 4,       /* 0.0 .. 0.5 */
  KT2N_SAMPLER_HAS_MAX_NEW_TOKENS = 1u << 5 /* integer 500 .. 12000 */
};

typedef struct kt2n_sampler {
  uint32_t present; /* kt2n_sampler_present */
  int32_t mode;     /* kt2n_sampler_mode */
  double temperature;
  double top_p;
  double min_p;
  int64_t top_k;
  int64_t max_new_tokens;
} kt2n_sampler;

enum kt2n_request_flags {
  KT2N_REQ_TIMELINE = 1u << 0,   /* emit KT2N_EVENT_TIMELINE (benchmark trace; needs KT2N_CFG_TIMELINE_ALLOWED) */
  KT2N_REQ_SEED = 1u << 1,       /* use `seed` as the segment-seed base instead of the request-id salt (paired runs) */
  KT2N_REQ_DEMO_CHECK = 1u << 2, /* DEMO+CHECK retries (the reference demo-check header) */
  KT2N_REQ_NORMALIZE_OFF = 1u << 4 /* raw text for requests / every phrase of a session */
};

typedef struct kt2n_request_params {
  uint32_t struct_size;
  uint32_t flags;          /* kt2n_request_flags */
  const char* text;        /* UTF-8 input as the engine transport receives it (before normalisation) */
  size_t text_len;         /* UTF-8 bytes */
  kt2n_voice_id voice;
  const char* request_id;  /* the reference request id: segment seed = salt + ordinal, salt =
                              (BE32(sha256(request_id)[0:4]) & 0x3FFFFFFF) * 1024 */
  uint64_t seed;           /* with KT2N_REQ_SEED only */
  kt2n_sampler sampler;
  double speed;            /* 0.25 .. 4.0; 1.0 = no stretch */
  int32_t format;          /* kt2n_audio_format */
  int32_t output;          /* kt2n_output_mode */
  int32_t speech_mode;     /* kt2n_speech_mode */
  int32_t reserved0;
  int64_t deadline_ns;     /* absolute CLOCK_MONOTONIC; 0 = none */
  uint64_t user_data;      /* echoed in every event of this request */
  /* Appended (read only when struct_size covers it, NULL = none).
   * The request's reference generation-control policy fields as one
   * JSON object, the reference names, only the fields the transport set (not null). A caller
   * that runs its own WS phrase planner submits each realtime phrase as one request
   * carrying the overrides the reference gives a phrase: "startup_hold_seconds": 0,
   * "initial_audio_buffer_seconds": 0, "concurrent_initial_audio_buffer_seconds": 0, and the
   * turn's EMO decision "emotion" (bool) / "emotion_carry_tag" (decided per request by default).
   * Also: "skip_opening_silence", "q_skip_limiter" (DEMO, set by the speech mode),
   * "demo_check", "phrase" (true for a WS phrase), and the traced lab overrides
   * (startup hold, delivery, preview, normalization, chunking and forced-id overrides).
   * The engine applies every key it knows exactly as the reference does; a key
   * it does not implement fails the submit with KT2N_ERR_UNSUPPORTED, param = the key, so a
   * request is never served under a different policy. */
  const char* control_json;
} kt2n_request_params;

/* Validates (public 400s come back here, synchronously, before any work),
 * then queues. Never waits for the GPU. */
KT2N_API int32_t kt2n_submit(kt2n_engine* engine, const kt2n_request_params* params, kt2n_request_id* out,
                             kt2n_error* err);

typedef enum kt2n_cancel_reason {
  KT2N_CANCEL_REQUEST = 0,  /* the reference "request_cancelled" (client closed) */
  KT2N_CANCEL_DEADLINE = 1, /* "deadline_exceeded" */
  KT2N_CANCEL_SHUTDOWN = 2
} kt2n_cancel_reason;

/* Cancels a request, a session (all its phrases) or a voice build. Idempotent.
 * The owner still receives a terminal event (KT2N_EVENT_DONE with status
 * KT2N_ERR_CANCELLED / KT2N_ERR_DEADLINE_EXCEEDED). */
KT2N_API int32_t kt2n_cancel(kt2n_engine* engine, uint64_t id, int32_t reason);

/* Frees a finished request / session / build id (after its terminal event).
 * Until then the id stays valid for kt2n_cancel. */
KT2N_API int32_t kt2n_release(kt2n_engine* engine, uint64_t id);

/* ------------------------------------------------------------------------- */
/* Realtime WebSocket sessions                                               */
/* ------------------------------------------------------------------------- */
/* The engine owns the phrase planner (the reference realtime_phrase_cut,
 * per-turn emotion decision and phrases): every phrase becomes an
 * internal generation with its own phrase_index; events carry it. */
typedef struct kt2n_session_params {
  uint32_t struct_size;
  uint32_t flags;          /* kt2n_request_flags */
  kt2n_voice_id voice;     /* clones: KT2N_ERR_CUSTOM_VOICE_REALTIME_NOT_SUPPORTED, as the reference */
  const char* request_id;  /* session id for seeds/logs (phrases derive their own as the reference does) */
  uint64_t seed;
  kt2n_sampler sampler;    /* from the socket query (text spellings validated by the caller) */
  double speed;
  int32_t format;          /* KT2N_FORMAT_PCM or KT2N_FORMAT_F32 */
  int32_t speech_mode;
  int64_t deadline_ns;
  uint64_t user_data;
} kt2n_session_params;

KT2N_API int32_t kt2n_session_open(kt2n_engine* engine, const kt2n_session_params* params, kt2n_session_id* out,
                                   kt2n_error* err);
/* Appends client text (a text frame). After kt2n_session_done:
 * KT2N_ERR_INPUT_ALREADY_CLOSED. */
KT2N_API int32_t kt2n_session_text_append(kt2n_engine* engine, kt2n_session_id session, const char* text,
                                          size_t len, kt2n_error* err);
/* The client's commit / flush: everything appended so far is spoken without
 * waiting for more text (the reference records the turn end). */
KT2N_API int32_t kt2n_session_commit(kt2n_engine* engine, kt2n_session_id session, kt2n_error* err);
/* Input closed: flush, finish every phrase, then KT2N_EVENT_DONE for the session. */
KT2N_API int32_t kt2n_session_done(kt2n_engine* engine, kt2n_session_id session, kt2n_error* err);

/* ------------------------------------------------------------------------- */
/* Events                                                                    */
/* ------------------------------------------------------------------------- */
typedef enum kt2n_event_kind {
  KT2N_EVENT_AUDIO = 1,        /* encoded audio bytes in the request's format */
  KT2N_EVENT_TIMELINE = 2,     /* one reference timeline point (name + JSON fields) */
  KT2N_EVENT_METRICS = 4,      /* end-of-generation engine metrics (the reference metrics snapshot) */
  KT2N_EVENT_PHRASE_DONE = 5,  /* WS: one phrase finished (phrase_index) */
  KT2N_EVENT_DONE = 6,         /* terminal: request / session finished; status = OK or the failure */
  KT2N_EVENT_VOICE_BUILT = 7,  /* terminal for a build job: voice, or status + error */
  KT2N_EVENT_HEADERS = 8       /* response metadata before the first audio (fields_json: speech_mode,
                                  sample_rate, complete-file byte length when known) */
} kt2n_event_kind;

/* Every pointer member points into engine-owned memory that stays valid until
 * kt2n_events_release() for that event. Unused members are 0 / NULL / -1. */
typedef struct kt2n_event {
  int32_t kind;            /* kt2n_event_kind */
  int32_t status;          /* DONE / VOICE_BUILT / mid-stream failure: kt2n_status; else 0 */
  uint64_t id;             /* request, session or job id */
  uint64_t user_data;      /* from the submit / open / build params */
  uint64_t seq;            /* per-id sequence number from 0, no gaps */
  int64_t at_ns;           /* CLOCK_MONOTONIC when it happened (not when it was polled) */
  int32_t segment_index;   /* the reference segment_index, -1 = none */
  int32_t phrase_index;    /* WS phrase, -1 = none */

  /* KT2N_EVENT_AUDIO */
  const uint8_t* data;     /* encoded bytes (PCM16 LE for KT2N_FORMAT_PCM) */
  size_t data_len;
  int64_t sample_begin;    /* first 24 kHz sample of this chunk in the request's (post-speed) output */
  int32_t samples;         /* samples this chunk carries */
  int32_t engine_chunk_index; /* the reference engine_chunk_index, from 0 */
  int32_t is_final_chunk;  /* 1 on the last audio event of the request / phrase */
  /* (was reserved0: same offset, no ABI bump) the reference opening-skipped token count
   * for this request / WS phrase: the opening-silence ids the skip dropped on segment 0 (the reference:
   * "final when the opening closes, before the first publication"). Set on EVERY audio event,
   * so the first one already carries the final count. A caller's realtime reserve can read it before it
   * releases a phrase's first audio (the reference's skip-aware per-listener term).
   * An engine that fills it says KT2N_CAP_AUDIO_OPENING_SKIP in
   * kt2n_build_info(); older builds leave 0 here, and a caller should then refuse to serve while a
   * skip-aware hold is configured (a silent timing-policy change otherwise). */
  int32_t opening_skipped_tokens;

  /* KT2N_EVENT_TIMELINE / METRICS / HEADERS */
  const char* name;        /* the reference event name (KT2N_TL_* below; static storage) */
  const char* fields_json; /* JSON object: the reference field names and units for that event */

  /* DONE / VOICE_BUILT with an error */
  const kt2n_error* error;

  /* KT2N_EVENT_VOICE_BUILT */
  kt2n_voice_id voice;
} kt2n_event;

/* An eventfd (EFD_NONBLOCK | EFD_CLOEXEC) owned by the engine: readable while
 * events are queued. read() it to re-arm, then kt2n_poll until it returns 0
 * events. One fd for the whole engine; demultiplex by id / user_data.
 * Windows has no eventfd: this returns -1 there; wait in kt2n_poll instead. */
KT2N_API int kt2n_event_fd(kt2n_engine* engine);
/* Moves up to `max` queued events into `out` (*n = count). timeout_ms: 0 =
 * never wait, < 0 = wait until at least one event, > 0 = wait at most that
 * long. Shutdown wakes blocked polls with zero events. A Python binding releases
 * the GIL around it and retains the engine until copied buffers are returned. */
KT2N_API int32_t kt2n_poll(kt2n_engine* engine, kt2n_event* out, int32_t max, int32_t timeout_ms, int32_t* n);
/* Returns the events' buffers to the engine (pinned / post-pool memory). */
KT2N_API void kt2n_events_release(kt2n_engine* engine, const kt2n_event* events, int32_t n);

/* ------------------------------------------------------------------------- */
/* Stats                                                                     */
/* ------------------------------------------------------------------------- */
typedef struct kt2n_stats {
  uint32_t struct_size;
  int32_t active_requests;   /* holding a seat */
  int32_t queued_requests;   /* admitted to the engine, waiting for a seat */
  int32_t active_sessions;
  int32_t lm_rows;           /* rows in the current decode step */
  int32_t render_queue_depth;/* render batches queued (all render streams) */
  int32_t render_inflight;
  int32_t pages_total;       /* 64-frame K/V pages */
  int32_t pages_free;
  int32_t voices_loaded;     /* KT2N_VOICE_CACHE_ALL: voices registered; LRU / NONE: voices resident on the GPU */
  int32_t voice_builds_active;
  int32_t voice_builds_queued;
  uint64_t device_arena_bytes;
  uint64_t device_arena_used;
  uint64_t pinned_arena_bytes;
  uint64_t pinned_arena_used;
  uint64_t requests_done;
  uint64_t requests_failed;
  uint64_t requests_cancelled;
  uint64_t lm_steps;
  uint64_t renders;          /* render calls */
  uint64_t render_rows;      /* rows over all render calls */
  uint64_t underrun_risks;   /* renders that published after their underrun deadline (engine's own view) */
} kt2n_stats;

KT2N_API int32_t kt2n_stats_get(kt2n_engine* engine, kt2n_stats* out);
/* Health snapshot as JSON (the reference /health engine block shape where one exists).
 * buf too small: *len = bytes needed (incl. NUL), KT2N_ERR_TOO_SMALL. */
KT2N_API int32_t kt2n_stats_json(kt2n_engine* engine, char* buf, size_t cap, size_t* len);

/* ------------------------------------------------------------------------- */
/* Timeline event names (KT2N_EVENT_TIMELINE `name`)                         */
/* ------------------------------------------------------------------------- */
/* Exactly the reference names (its timeline marks), so the
 * reference analysis scripts run unchanged, and the reference service's engine stage events stamp the same origin
 * stages (normalized, admitted, prefill, first_commit, hold_released,
 * engine_chunk). "Required fields" are the ones those scripts read; the native
 * engine must emit them with the same names and units (ms floats, counts ints).
 * Other reference fields are emitted when the native engine has the value. */

/* Text and admission */
#define KT2N_TL_GENERATE_STREAM_ENTERED "gpu.generate_stream_entered"
#define KT2N_TL_EMOTION_DECIDED "gpu.emotion_decided"       /* emo_* */
#define KT2N_TL_TEXT_SEGMENTED "gpu.text_segmented"         /* characters, segments, wfst */
#define KT2N_TL_TEXT_NORMALIZED "gpu.text_normalized"       /* stage `normalized`; characters, fast_path,
                                                               incremental, normalized_segments,
                                                               pending_segments */
#define KT2N_TL_SEGMENT_NORMALIZED "gpu.segment_normalized" /* stage `normalized`; characters, chunks,
                                                               normalize_ms, segment_index */
#define KT2N_TL_LM_STARTED "cpu.lm_started"
#define KT2N_TL_SCHEDULER_ENQUEUED "gpu.scheduler_enqueued"
#define KT2N_TL_SCHEDULER_REGISTERED "gpu.scheduler_registered"
#define KT2N_TL_SCHEDULER_EXTENDED "gpu.scheduler_extended"
#define KT2N_TL_SCHEDULER_ADMITTED "gpu.scheduler_admitted" /* stage `admitted`. Required: admission_batch,
                                                               active_before, queue_ms, segment_index
                                                               */
#define KT2N_TL_SCHEDULER_REST "gpu.scheduler_rest"
#define KT2N_TL_SCHEDULER_RESUME "gpu.scheduler_resume"
#define KT2N_TL_SCHEDULER_FINISHED "gpu.scheduler_finished"
#define KT2N_TL_SEGMENT_PREFETCH_RELEASED "gpu.segment_prefetch_released"
#define KT2N_TL_DEMO_PREFETCH_RELEASED "gpu.demo_prefetch_released"

/* LM */
#define KT2N_TL_LM_PREFILL_FINISHED "gpu.lm_prefill_finished" /* stage `prefill`. Required: batch_size,
                                                                 segment_index; also lm_share_ms,
                                                                 lm_wall_ms */
#define KT2N_TL_NATIVE_PREFILL "gpu.kt2n_native_prefill"      /* lab native-prefill marker (the reference's
                                                                 owner events) */
#define KT2N_TL_LM_DECODE_STEP "gpu.lm_decode_step"           /* batch_size, cuda_ms, generated_tokens,
                                                                 segment_index, wall_ms */
#define KT2N_TL_N_OPENING_SKIP "gpu.n_opening_skip"           /* Required: skipped (list of ids); cap, closing,
                                                                 lm_head, pending, skipped_in_set
                                                                 */
#define KT2N_TL_LM_TOKENS_FINALIZED "gpu.lm_tokens_finalized" /* audio_token_count, audio_token_sha256,
                                                                 segment_index */
#define KT2N_TL_LM_SEGMENT_FINISHED "gpu.lm_segment_finished" /* Required: lm_steps, segment_index; budget,
                                                                 finish_reason, audio_tokens, silent_tokens,
                                                                 longest_silent_run_* */
#define KT2N_TL_Q_PRE_EOS "gpu.q_pre_eos"

/* Commit planning and renders */
#define KT2N_TL_Q_COMMIT_PLAN "gpu.q_commit_plan"
#define KT2N_TL_D_COMMIT_PLAN "gpu.d_commit_plan"   /* buffer_ms, commit_index, next_commit_tokens, render_ms,
                                                       segment_index, token_ms */
#define KT2N_TL_D_EMERGENCY_COMMIT "gpu.d_emergency_commit"
#define KT2N_TL_D_JOIN_CAP "gpu.d_join_cap"
#define KT2N_TL_Q_ALIGN_COMMIT "gpu.q_align_commit"
#define KT2N_TL_Q_PAUSE_FLUSH "gpu.q_pause_flush"
#define KT2N_TL_Q_PAUSE_TAIL "gpu.q_pause_tail"
#define KT2N_TL_Q_PAUSE "gpu.q_pause"
#define KT2N_TL_Q_SEAM_SNAP "gpu.q_seam_snap"
#define KT2N_TL_Q_FAST_SEAM_SNAP "gpu.q_fast_seam_snap"
#define KT2N_TL_T9_DEFERRAL_ENDED "gpu.t9_acoustic_deferral_ended" /* Required: deferred_turns, reservoir_s,
                                                                      segment_index */
#define KT2N_TL_DW_QUEUED "gpu.dw_queued"
#define KT2N_TL_DW_RENDER "gpu.dw_render"
#define KT2N_TL_RG_FIRST_HANDOFF "gpu.rg_first_handoff"
#define KT2N_TL_S3GEN_STARTED "gpu.s3gen_started"   /* render launched. Required: commit_tokens, prefix_tokens,
                                                       segment_index; also batch_size, async_launch,
                                                       cached_encoder, cached_flow */
#define KT2N_TL_PREVIEW_PUBLICATION "gpu.preview_publication" /* stage `first_commit` (1-step preview);
                                                                 audio_ms, audio_samples, batch_size,
                                                                 flow_steps, preview_share_ms,
                                                                 preview_tokens, preview_wall_ms,
                                                                 segment_index */
#define KT2N_TL_S3GEN_PUBLICATION "gpu.s3gen_publication" /* stage `first_commit` (first one). Required:
                                                             s3gen_wall_ms, s3gen_gpu_ms, s3gen_launch_ms,
                                                             s3gen_queue_depth, batch_size, commit_tokens,
                                                             audio_ms, segment_index; also audio_samples,
                                                             flow_cuda_ms, hift_cuda_ms, lm_ms_total,
                                                             s3gen_share_ms */
#define KT2N_TL_S3GEN_FINAL_PUBLICATION "gpu.s3gen_final_publication" /* the segment's final flush; same
                                                                         fields as s3gen_publication
                                                                         */
#define KT2N_TL_CPU_S3GEN_FINAL_PUBLICATION "cpu.s3gen_final_publication"
#define KT2N_TL_CPU_LM_TOKENS_FINALIZED "cpu.lm_tokens_finalized"

/* Post-processing inside the engine (the reference emits these from its runtime
 * despite the "api." prefix) */
#define KT2N_TL_STARTUP_HOLD_RELEASED "api.startup_hold_released" /* stage `hold_released` (HTTP/SSE).
                                                                     Required: target_ms, held_ms, sharing,
                                                                     mode; chunks, reached */
#define KT2N_TL_INITIAL_RESERVOIR_SELECTED "api.initial_reservoir_selected"
#define KT2N_TL_LEADING_SILENCE_TRIMMED "api.leading_silence_trimmed"
#define KT2N_TL_SEAM_TRIMMED "api.seam_trimmed"
#define KT2N_TL_SPEED_STRETCH "api.speed_stretch"
#define KT2N_TL_SILENCE_CAPPED "api.silence_capped"
#define KT2N_TL_Q_HOST_AUDIO "api.q_host_audio"
#define KT2N_TL_Q_DCG_VERDICT "api.q_dcg_verdict"
#define KT2N_TL_Q_DCG_FOLLOWUP "api.q_dcg_followup"
#define KT2N_TL_LB_HEAD_BUFFER "lb.head_buffer"
#define KT2N_TL_LB_PHRASE_PLAN "lb.phrase_plan"
#define KT2N_TL_LB_VERIFY "lb.verify"

/* Transport events: emitted by a Python adapter / the reference service around the engine
 * (they need the socket). The engine never emits them; a caller derives
 * engine_publication_ready from KT2N_EVENT_AUDIO (engine_chunk_index, samples,
 * data_len). Listed so every name the analysis scripts read has one home. */
#define KT2N_TL_API_REQUEST_READY "api.request_ready"                   /* stage `engine_ready` */
#define KT2N_TL_API_WEBSOCKET_CONNECTED "api.websocket_connected"       /* stage `engine_ready` (WS) */
#define KT2N_TL_API_STREAM_STARTED "api.stream_started"
#define KT2N_TL_API_ENGINE_PUBLICATION_READY "api.engine_publication_ready" /* stage `engine_chunk`. Required:
                                                                              engine_chunk_index */
#define KT2N_TL_API_SSE_PUBLICATION "api.sse_publication"               /* Required: pcm_bytes, engine_chunk_index */
#define KT2N_TL_API_WEBSOCKET_PUBLICATION "api.websocket_publication"   /* stage `engine_chunk` (WS). Required:
                                                                           pcm_bytes, engine_chunk_index */
#define KT2N_TL_API_WEBSOCKET_RESERVE_RELEASED "api.websocket_reserve_released" /* WS hold. Required:
                                                                                  target_ms, held_ms,
                                                                                  sharing, mode */
#define KT2N_TL_API_WEBSOCKET_RESERVE_REARMED "api.websocket_reserve_rearmed"
#define KT2N_TL_API_PHRASE_GENERATION_STARTED "api.phrase_generation_started"
#define KT2N_TL_API_R_WS_CUT_SETTLED "api.r_ws_cut_settled"
#define KT2N_TL_API_COMPLETE_BOUNDARY "api.complete_boundary"
#define KT2N_TL_API_COMPLETE_SLOT_WAIT "api.complete_slot_wait"
#define KT2N_TL_API_COMPLETE_SLOT_ACQUIRED "api.complete_slot_acquired"

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* KT2N_ENGINE_H_ */
