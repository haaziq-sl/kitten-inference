/*
 * stream.c: speak TEXT with a built-in voice and write the audio as it is made.
 *
 *   stream ASSETS_DIR VOICE TEXT OUT.pcm
 *
 * OUT.pcm receives raw 16-bit little-endian mono PCM at 24 kHz, one chunk at a
 * time, while the rest of the text is still being generated. Use "-" for
 * standard output, for example to listen straight away:
 *
 *   stream models/kitten Bruno "Hello there." - | aplay -q -f S16_LE -r 24000 -c 1
 *
 * Ctrl-C cancels the request; the audio written so far stays in OUT.pcm. The
 * engine still ends a cancelled request with a KT2N_EVENT_DONE event, so the
 * loop below always runs until DONE.
 */
#define _POSIX_C_SOURCE 200809L
#include "example_util.h"

int main(int argc, char** argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s ASSETS_DIR VOICE TEXT OUT.pcm|-\n", argv[0]);
    return 2;
  }
  const char* assets_dir = argv[1];
  const char* voice_name = argv[2];
  const char* text = argv[3];
  const char* out_path = argv[4];

  example_bundle bundle;
  if (!example_bundle_init(&bundle, assets_dir)) return 1;

  /* Engine settings: built-in voices only, one request at a time. */
  kt2n_engine_config config;
  memset(&config, 0, sizeof config);
  config.struct_size = sizeof config;
  config.abi_version = KT2N_ABI_VERSION;
  config.flags = KT2N_CFG_INFERENCE_ONLY; /* no memory reserved for cloned voices */
  config.device = 0;                      /* CUDA device ordinal */
  config.max_streams = 1;
  config.lm_file = bundle.lm;
  config.s3gen_file = bundle.s3gen;
  config.voices_file = bundle.voices;
  config.tokenizer_json = bundle.tokenizer;
  config.text_data_dir = bundle.text_dir;
  /* Memory: the automatic profile (the default profile where it fits the GPU, otherwise the low-memory profile; one
   * line on stderr says so), and only the voice this program speaks with on the GPU, as kitten-tts-native does by
   * default. Together they start on GPUs from 4 GB up; the C API's own defaults (the default profile with every
   * built-in voice prepared) need about 21 GB of free GPU memory. Complete-file audio is the same in all of these
   * settings; on GPUs of about 4 GB, long streamed text may be split into shorter pieces, which changes its audio. */
  config.memory_profile = KT2N_MEMORY_PROFILE_AUTO;
  config.voice_cache_mode = KT2N_VOICE_CACHE_LRU;
  config.voice_cache_max_voices = 1;

  kt2n_error error;
  memset(&error, 0, sizeof error);
  kt2n_engine* engine = NULL;
  fprintf(stderr, "loading the engine...\n");
  int32_t status = kt2n_engine_create(&config, &engine, &error); /* blocks for a few seconds */
  if (status != KT2N_OK) return example_fail("engine create", status, &error);

  kt2n_voice_id voice = KT2N_INVALID_ID;
  status = kt2n_voice_lookup(engine, voice_name, &voice, &error);
  if (status != KT2N_OK) {
    example_fail("voice lookup", status, &error);
    kt2n_engine_destroy(engine);
    return 1;
  }
  /* Prepare the voice now, so the first request does not wait for it (best effort: if it cannot be prepared now, the
   * request prepares it and reports any error). */
  (void)kt2n_voice_prepare(engine, voice, NULL);

  FILE* out = strcmp(out_path, "-") == 0 ? stdout : fopen(out_path, "wb");
  if (!out) {
    fprintf(stderr, "error: cannot write %s: %s\n", out_path, strerror(errno));
    kt2n_engine_destroy(engine);
    return 1;
  }

  /* One streaming request. Fields left at zero keep their defaults. */
  kt2n_request_params params;
  memset(&params, 0, sizeof params);
  params.struct_size = sizeof params;
  params.text = text;
  params.text_len = strlen(text);
  params.voice = voice;
  params.speed = 1.0;
  params.format = KT2N_FORMAT_PCM;
  params.output = KT2N_OUTPUT_STREAM;

  example_catch_signals();
  const int64_t started_ns = example_now_ns();
  kt2n_request_id request = KT2N_INVALID_ID;
  status = kt2n_submit(engine, &params, &request, &error); /* validates, queues, returns at once */
  if (status != KT2N_OK) {
    example_fail("submit", status, &error);
    if (out != stdout) fclose(out);
    kt2n_engine_destroy(engine);
    return 1;
  }

  int done = 0, cancelled = 0, write_failed = 0;
  int32_t final_status = KT2N_OK;
  int64_t first_audio_ns = -1, samples = 0;
  while (!done) {
    if (example_interrupted && !cancelled) {
      kt2n_cancel(engine, request, KT2N_CANCEL_REQUEST);
      cancelled = 1;
    }
    kt2n_event events[16];
    int32_t count = 0;
    status = kt2n_poll(engine, events, 16, 100 /* ms */, &count);
    if (status != KT2N_OK) {
      example_fail("poll", status, NULL);
      break;
    }
    for (int32_t i = 0; i < count; ++i) {
      const kt2n_event* ev = &events[i];
      if (ev->id != request) continue;
      if (ev->kind == KT2N_EVENT_AUDIO && ev->data_len > 0 && !cancelled) {
        if (first_audio_ns < 0) first_audio_ns = ev->at_ns;
        if (fwrite(ev->data, 1, ev->data_len, out) != ev->data_len || fflush(out) != 0) {
          kt2n_cancel(engine, request, KT2N_CANCEL_REQUEST); /* e.g. the reading end of a pipe closed */
          cancelled = write_failed = 1;
        }
        samples += ev->samples;
      } else if (ev->kind == KT2N_EVENT_DONE) {
        done = 1;
        final_status = ev->status;
        if (ev->error) error = *ev->error;
      }
    }
    /* Event data stays valid until it is released; give it back after every poll. */
    kt2n_events_release(engine, events, count);
  }
  if (done) kt2n_release(engine, request);

  if (out != stdout && fclose(out) != 0) write_failed = 1;
  const double seconds = (double)samples / EXAMPLE_SAMPLE_RATE;
  int exit_code = 0;
  if (write_failed) {
    fprintf(stderr, "error: writing %s failed\n", out_path);
    exit_code = 1;
  } else if (final_status == KT2N_ERR_CANCELLED) {
    fprintf(stderr, "cancelled after %.2f s of audio\n", seconds);
    exit_code = 130;
  } else if (!done || final_status != KT2N_OK) {
    exit_code = done ? example_fail("speech", final_status, &error) : 1;
  } else {
    fprintf(stderr, "wrote %.2f s of audio to %s (first audio after %.0f ms)\n", seconds, out_path,
            first_audio_ns < 0 ? 0.0 : (double)(first_audio_ns - started_ns) / 1e6);
  }
  kt2n_engine_destroy(engine);
  return exit_code;
}
