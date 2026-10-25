/*
 * complete.c: speak TEXT with a built-in voice and save one finished WAV file.
 *
 *   complete ASSETS_DIR VOICE TEXT OUT.wav
 *
 * This is the non-streaming mode (KT2N_OUTPUT_COMPLETE, `stream: false`). The
 * engine splits long text into sentence-sized chunks, renders each chunk in one
 * piece and levels the whole file at the end, then hands over the finished WAV.
 * The result is not the same as joining the chunks of a streaming request.
 * Nothing arrives until the whole file is ready.
 *
 * Ctrl-C cancels the request; no file is written then.
 */
#define _POSIX_C_SOURCE 200809L
#include "example_util.h"

int main(int argc, char** argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s ASSETS_DIR VOICE TEXT OUT.wav\n", argv[0]);
    return 2;
  }
  const char* assets_dir = argv[1];
  const char* voice_name = argv[2];
  const char* text = argv[3];
  const char* out_path = argv[4];

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
  /* Memory: the automatic profile (the default profile where it fits the GPU, otherwise the low-memory profile; one
   * line on stderr says so), and only the voice this program speaks with on the GPU, as kitten-tts-native does by
   * default. Together they start on GPUs from 4 GB up; the C API's own defaults (the default profile with every
   * built-in voice prepared) need about 21 GB of free GPU memory. The audio is the same in all of these settings. */
  config.memory_profile = KT2N_MEMORY_PROFILE_AUTO;
  config.voice_cache_mode = KT2N_VOICE_CACHE_LRU;
  config.voice_cache_max_voices = 1;

  kt2n_error error;
  memset(&error, 0, sizeof error);
  kt2n_engine* engine = NULL;
  fprintf(stderr, "loading the engine...\n");
  int32_t status = kt2n_engine_create(&config, &engine, &error);
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

  char part[EXAMPLE_PATH_MAX];
  FILE* out = example_open_output(out_path, part, sizeof part);
  if (!out) {
    kt2n_engine_destroy(engine);
    return 1;
  }

  kt2n_request_params params;
  memset(&params, 0, sizeof params);
  params.struct_size = sizeof params;
  params.text = text;
  params.text_len = strlen(text);
  params.voice = voice;
  params.speed = 1.0;
  params.format = KT2N_FORMAT_WAV;      /* a complete WAV file, header included */
  params.output = KT2N_OUTPUT_COMPLETE; /* one finished file instead of a stream */

  example_catch_signals();
  const int64_t started_ns = example_now_ns();
  kt2n_request_id request = KT2N_INVALID_ID;
  status = kt2n_submit(engine, &params, &request, &error);
  if (status != KT2N_OK) {
    example_fail("submit", status, &error);
    example_finish_output(out, out_path, part, 0);
    kt2n_engine_destroy(engine);
    return 1;
  }

  int done = 0, cancelled = 0, write_failed = 0;
  int32_t final_status = KT2N_OK;
  int64_t samples = 0;
  size_t bytes = 0;
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
      if (ev->kind == KT2N_EVENT_AUDIO && ev->data_len > 0 && !write_failed) {
        /* Usually one event carries the whole file; writing every chunk in
         * order also covers files that arrive in several pieces. */
        if (fwrite(ev->data, 1, ev->data_len, out) != ev->data_len) write_failed = 1;
        bytes += ev->data_len;
        samples += ev->samples;
      } else if (ev->kind == KT2N_EVENT_DONE) {
        done = 1;
        final_status = ev->status;
        if (ev->error) error = *ev->error;
      }
    }
    kt2n_events_release(engine, events, count);
  }
  if (done) kt2n_release(engine, request);

  const int ok = done && final_status == KT2N_OK && !write_failed && bytes > 0;
  const int saved = example_finish_output(out, out_path, part, ok);
  int exit_code = 0;
  if (final_status == KT2N_ERR_CANCELLED) {
    fprintf(stderr, "cancelled; %s was not written\n", out_path);
    exit_code = 130;
  } else if (done && final_status != KT2N_OK) {
    exit_code = example_fail("speech", final_status, &error);
  } else if (!saved) {
    if (write_failed) fprintf(stderr, "error: writing %s failed\n", out_path);
    else if (done && bytes == 0) fprintf(stderr, "error: the engine returned no audio\n");
    exit_code = 1;
  } else {
    fprintf(stderr, "wrote %.2f s of audio (%zu bytes) to %s in %.2f s\n", (double)samples / EXAMPLE_SAMPLE_RATE,
            bytes, out_path, (double)(example_now_ns() - started_ns) / 1e9);
  }
  kt2n_engine_destroy(engine);
  return exit_code;
}
