/*
 * voice_cache.c: keep only some voices on the GPU and switch between them.
 *
 *   voice_cache ASSETS_DIR MAX_VOICES MAX_MIB VOICE [VOICE...]
 *
 * The engine starts with KT2N_VOICE_CACHE_LRU: no voice is prepared at
 * startup. A voice is prepared on the GPU the first time a request uses it and
 * stays there until the cache needs its room; then the least recently used
 * voice that no request is using is dropped. MAX_VOICES caps how many voices
 * are on the GPU at once and MAX_MIB caps the GPU memory they use (0 = no limit
 * of that kind; both 0 = 16 voices, or one per stream when that is more).
 *
 * One short request (complete mode, audio discarded) is spoken with each VOICE
 * in the order given, so a voice can be repeated to see a hit, for example:
 *
 *   voice_cache models/kitten 2 1024 Bruno Bella Bruno Kiki Bella Bruno
 *
 * After each request the program prints whether the voice was already on the
 * GPU (hit) or had to be prepared (miss) and how many voices were dropped for
 * it. At the end it prints the "voice_cache" object of kt2n_stats_json(). The
 * audio does not depend on the cache: every mode gives the same result for a
 * voice it accepts; only the time to the first audio of a voice that is not on
 * the GPU differs.
 */
#define _POSIX_C_SOURCE 200809L
#include "example_util.h"

/* kt2n_stats_json() into a malloc'd, NUL-terminated string (NULL on failure). */
static char* read_stats(kt2n_engine* engine) {
  size_t cap = 4096;
  for (int attempt = 0; attempt < 4; ++attempt) {
    char* buf = (char*)malloc(cap);
    if (!buf) return NULL;
    size_t len = 0;
    const int32_t status = kt2n_stats_json(engine, buf, cap, &len);
    if (status == KT2N_OK) return buf;
    free(buf);
    if (status != KT2N_ERR_TOO_SMALL || len == 0) return NULL;
    cap = len + 1024; /* the text can grow between two calls */
  }
  return NULL;
}

/* Copies the JSON object stored under "key" (e.g. "voice_cache") into a
 * malloc'd string. Returns NULL when the key is missing. */
static char* json_object(const char* json, const char* key) {
  char pattern[64];
  snprintf(pattern, sizeof pattern, "\"%s\":{", key);
  const char* begin = strstr(json, pattern);
  if (!begin) return NULL;
  begin += strlen(pattern) - 1; /* at the '{' */
  int depth = 0, in_string = 0;
  for (const char* p = begin; *p; ++p) {
    if (in_string) {
      if (*p == '\\' && p[1]) ++p;
      else if (*p == '"') in_string = 0;
    } else if (*p == '"') {
      in_string = 1;
    } else if (*p == '{') {
      ++depth;
    } else if (*p == '}' && --depth == 0) {
      const size_t n = (size_t)(p - begin) + 1;
      char* out = (char*)malloc(n + 1);
      if (!out) return NULL;
      memcpy(out, begin, n);
      out[n] = '\0';
      return out;
    }
  }
  return NULL;
}

/* An unsigned integer member of a flat JSON object ("hits", "misses", ...); 0 if absent. */
static unsigned long long json_count(const char* object, const char* key) {
  char pattern[64];
  snprintf(pattern, sizeof pattern, "\"%s\":", key);
  const char* p = object ? strstr(object, pattern) : NULL;
  return p ? strtoull(p + strlen(pattern), NULL, 10) : 0;
}

typedef struct cache_counts {
  unsigned long long hits, misses, joins, evictions;
} cache_counts;

/* Reads the cache counters; prints the whole voice_cache object when `print` is set. */
static int read_cache(kt2n_engine* engine, cache_counts* out, int print) {
  char* stats = read_stats(engine);
  char* cache = stats ? json_object(stats, "voice_cache") : NULL;
  free(stats);
  if (!cache) {
    fprintf(stderr, "error: kt2n_stats_json has no voice_cache object\n");
    return 0;
  }
  out->hits = json_count(cache, "hits");
  out->misses = json_count(cache, "misses");
  out->joins = json_count(cache, "joins");
  out->evictions = json_count(cache, "evictions");
  if (print) printf("voice_cache: %s\n", cache);
  free(cache);
  return 1;
}

/* Speaks `text` with `voice` in complete mode and waits for the result. Returns
 * the request's final status; *ms is the time from submit to the finished audio. */
static int32_t speak(kt2n_engine* engine, kt2n_voice_id voice, const char* text, double* ms, kt2n_error* error) {
  kt2n_request_params params;
  memset(&params, 0, sizeof params);
  params.struct_size = sizeof params;
  params.text = text;
  params.text_len = strlen(text);
  params.voice = voice;
  params.speed = 1.0;
  params.format = KT2N_FORMAT_PCM;
  params.output = KT2N_OUTPUT_COMPLETE;

  const int64_t started_ns = example_now_ns();
  kt2n_request_id request = KT2N_INVALID_ID;
  int32_t status = kt2n_submit(engine, &params, &request, error);
  if (status != KT2N_OK) return status;
  int done = 0, cancelled = 0;
  int32_t final_status = KT2N_OK;
  while (!done) {
    if (example_interrupted && !cancelled) {
      kt2n_cancel(engine, request, KT2N_CANCEL_REQUEST);
      cancelled = 1;
    }
    kt2n_event events[16];
    int32_t count = 0;
    status = kt2n_poll(engine, events, 16, 100 /* ms */, &count);
    if (status != KT2N_OK) return status;
    for (int32_t i = 0; i < count; ++i) {
      if (events[i].id != request || events[i].kind != KT2N_EVENT_DONE) continue;
      done = 1;
      final_status = events[i].status;
      if (events[i].error) *error = *events[i].error;
    }
    kt2n_events_release(engine, events, count);
  }
  kt2n_release(engine, request);
  *ms = (double)(example_now_ns() - started_ns) / 1e6;
  return final_status;
}

static int parse_limit(const char* text, unsigned long long max, unsigned long long* out) {
  char* end = NULL;
  errno = 0;
  const unsigned long long value = strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || text[0] == '-' || value > max) return 0;
  *out = value;
  return 1;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr,
            "usage: %s ASSETS_DIR MAX_VOICES MAX_MIB VOICE [VOICE...]\n"
            "  MAX_VOICES  voices kept on the GPU at once (0 = no count limit)\n"
            "  MAX_MIB     GPU memory for voices in MiB (0 = no byte limit; both 0 = 16 voices or one per stream)\n",
            argv[0]);
    return 2;
  }
  const char* assets_dir = argv[1];
  unsigned long long max_voices = 0, max_mib = 0;
  if (!parse_limit(argv[2], 4096, &max_voices) || !parse_limit(argv[3], UINT64_MAX >> 20, &max_mib)) {
    fprintf(stderr, "error: MAX_VOICES must be 0-4096 and MAX_MIB a whole number of MiB\n");
    return 2;
  }

  example_bundle bundle;
  if (!example_bundle_init(&bundle, assets_dir)) return 1;

  kt2n_engine_config config;
  memset(&config, 0, sizeof config);
  config.struct_size = sizeof config;
  config.abi_version = KT2N_ABI_VERSION;
  config.flags = KT2N_CFG_INFERENCE_ONLY; /* built-in voices only */
  config.device = 0;
  config.max_streams = 1;
  config.lm_file = bundle.lm;
  config.s3gen_file = bundle.s3gen;
  config.voices_file = bundle.voices;
  config.tokenizer_json = bundle.tokenizer;
  config.text_data_dir = bundle.text_dir;
  config.voice_cache_mode = KT2N_VOICE_CACHE_LRU;
  config.voice_cache_max_voices = (int32_t)max_voices;
  config.voice_cache_max_bytes = (uint64_t)max_mib << 20;
  /* The automatic memory profile: the default profile where it fits the GPU, otherwise the low-memory profile, which
   * also shrinks the working buffers (one line on stderr says so). */
  config.memory_profile = KT2N_MEMORY_PROFILE_AUTO;

  kt2n_error error;
  memset(&error, 0, sizeof error);
  kt2n_engine* engine = NULL;
  fprintf(stderr, "loading the engine...\n");
  int32_t status = kt2n_engine_create(&config, &engine, &error);
  if (status != KT2N_OK) return example_fail("engine create", status, &error);

  cache_counts before;
  if (!read_cache(engine, &before, 1)) {
    kt2n_engine_destroy(engine);
    return 1;
  }

  example_catch_signals();
  const char* text = "This sentence is spoken with each voice in turn.";
  int exit_code = 0;
  for (int i = 4; i < argc && !example_interrupted; ++i) {
    kt2n_voice_id voice = KT2N_INVALID_ID;
    status = kt2n_voice_lookup(engine, argv[i], &voice, &error);
    if (status != KT2N_OK) {
      exit_code = example_fail("voice lookup", status, &error);
      break;
    }
    double ms = 0;
    status = speak(engine, voice, text, &ms, &error);
    if (status != KT2N_OK) {
      if (status == KT2N_ERR_CANCELLED) {
        fprintf(stderr, "cancelled\n");
        exit_code = 130;
      } else {
        exit_code = example_fail("speech", status, &error);
      }
      break;
    }
    cache_counts after;
    if (!read_cache(engine, &after, 0)) {
      exit_code = 1;
      break;
    }
    const char* how = after.misses > before.misses ? "miss" : after.joins > before.joins ? "joined a load" : "hit";
    printf("%-12s %-14s dropped %llu  %.1f ms\n", argv[i], how, after.evictions - before.evictions, ms);
    before = after;
  }

  if (exit_code == 0 && !read_cache(engine, &before, 1)) exit_code = 1;
  kt2n_engine_destroy(engine);
  return exit_code;
}
