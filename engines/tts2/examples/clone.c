/*
 * clone.c: make a voice from a short recording, save it, and speak with it later.
 *
 *   clone build ASSETS_DIR REFERENCE_AUDIO TRANSCRIPT OUT.voice
 *   clone speak ASSETS_DIR VOICE_FILE TEXT OUT.wav
 *
 * build  turns REFERENCE_AUDIO (2-30 seconds of one speaker, at most 10 MiB, in
 *        a common format such as WAV, MP3, FLAC or OGG) plus TRANSCRIPT (the exact
 *        words spoken in it) into a voice, and saves that voice to OUT.voice.
 *        It needs a library built with voice cloning (the default preset, or
 *        KT2N_BUILD_CLONING=ON with KT2N_VOICE_AV=ON) and
 *        a bundle prepared with cloning files (native/cloning/).
 *
 * speak  loads a saved voice file and writes TEXT as a complete WAV file. It needs
 *        neither the cloning files nor a cloning build: a saved voice only needs
 *        the same bundle it was made with.
 *
 * Ctrl-C cancels either step.
 */
#define _POSIX_C_SOURCE 200809L
#include "example_util.h"

#define REFERENCE_LIMIT (10u * 1024u * 1024u) /* the engine accepts up to 10 MiB */
#define VOICE_FILE_LIMIT (256u * 1024u * 1024u)

/* Settings shared by both steps: room for one custom voice next to the built-ins. */
static void clone_config(kt2n_engine_config* config, const example_bundle* bundle) {
  memset(config, 0, sizeof *config);
  config->struct_size = sizeof *config;
  config->abi_version = KT2N_ABI_VERSION;
  config->flags = 0; /* not KT2N_CFG_INFERENCE_ONLY: that mode reserves built-in voices only */
  config->device = 0;
  config->max_streams = 1;
  config->max_custom_voices = 1; /* memory for one custom voice */
  config->lm_file = bundle->lm;
  config->s3gen_file = bundle->s3gen;
  config->voices_file = bundle->voices;
  config->tokenizer_json = bundle->tokenizer;
  config->text_data_dir = bundle->text_dir;
  /* Memory: the automatic profile (the default profile where it fits the GPU, otherwise the low-memory profile; one
   * line on stderr says so), and no built-in voice kept on the GPU (this program speaks only with its own voice).
   * Speaking with a saved voice then starts on GPUs from 4 GB up. Building a voice needs about 13 GB of free GPU
   * memory: the low-memory profile turns cloning off when it has to shrink its sizes to fit the GPU. */
  config->memory_profile = KT2N_MEMORY_PROFILE_AUTO;
  config->voice_cache_mode = KT2N_VOICE_CACHE_NONE;
}

static int library_can_clone(void) {
  return strstr(kt2n_build_info(), KT2N_CAP_VOICE_CLONING) != NULL;
}

/* build: reference recording + transcript -> saved voice file. */
static int build_voice(const char* assets_dir, const char* audio_path, const char* transcript,
                       const char* out_path) {
  if (!library_can_clone()) {
    fprintf(stderr, "error: this library was built without voice cloning (configure with the default preset, "
            "or -DKT2N_BUILD_CLONING=ON -DKT2N_VOICE_AV=ON)\n");
    return 1;
  }
  example_bundle bundle;
  if (!example_bundle_init(&bundle, assets_dir)) return 1;
  char check[EXAMPLE_PATH_MAX];
  const int n = snprintf(check, sizeof check, "%s/s3gen.safetensors", bundle.cloning_dir);
  if (n <= 0 || (size_t)n >= sizeof check || !example_exists(check)) {
    fprintf(stderr,
            "error: %s has no cloning files.\n"
            "Add them with: python3 tools/prepare_assets.py --cloning-only --out %s\n",
            assets_dir, assets_dir);
    return 1;
  }

  size_t audio_len = 0;
  void* audio = example_read_file(audio_path, REFERENCE_LIMIT, &audio_len);
  if (!audio) return 1;

  kt2n_engine_config config;
  clone_config(&config, &bundle);
  config.voice_build_dir = bundle.cloning_dir; /* the networks that turn a recording into a voice */
  config.clone_slots = 1;                      /* one build at a time */
  config.inline_voice_lru = 1;                 /* keep at most one one-off clone (the default is 32) */

  kt2n_error error;
  memset(&error, 0, sizeof error);
  kt2n_engine* engine = NULL;
  fprintf(stderr, "loading the engine...\n");
  int32_t status = kt2n_engine_create(&config, &engine, &error);
  if (status != KT2N_OK) {
    free(audio);
    return example_fail("engine create", status, &error);
  }

  kt2n_voice_build_params params;
  memset(&params, 0, sizeof params);
  params.struct_size = sizeof params;
  params.flags = KT2N_VB_SAVED; /* the voice will be exported and kept */
  params.audio = audio;         /* encoded file bytes; the container is detected */
  params.audio_len = audio_len;
  params.reference_transcript = transcript;
  params.language = "en";

  example_catch_signals();
  kt2n_job_id job = KT2N_INVALID_ID;
  status = kt2n_voice_build(engine, &params, &job, &error); /* queues the build and returns */
  free(audio);                                              /* the engine keeps its own copy */
  if (status != KT2N_OK) {
    example_fail("voice build", status, &error);
    kt2n_engine_destroy(engine);
    return 1;
  }

  fprintf(stderr, "building the voice...\n");
  int done = 0, cancelled = 0;
  int32_t final_status = KT2N_OK;
  kt2n_voice_id voice = KT2N_INVALID_ID;
  while (!done) {
    if (example_interrupted && !cancelled) {
      kt2n_cancel(engine, job, KT2N_CANCEL_REQUEST);
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
      if (ev->id == job && ev->kind == KT2N_EVENT_VOICE_BUILT) { /* the one terminal event of a build */
        done = 1;
        final_status = ev->status;
        voice = ev->voice;
        if (ev->error) error = *ev->error;
      }
    }
    kt2n_events_release(engine, events, count);
  }
  if (done) kt2n_release(engine, job);

  int exit_code = 1;
  if (!done) {
    /* poll failed; already reported */
  } else if (final_status == KT2N_ERR_CANCELLED || (cancelled && final_status == KT2N_OK)) {
    fprintf(stderr, "cancelled; %s was not written\n", out_path);
    exit_code = 130;
  } else if (final_status != KT2N_OK || voice == KT2N_INVALID_ID) {
    example_fail("voice build", final_status != KT2N_OK ? final_status : KT2N_ERR_INTERNAL, &error);
  } else {
    /* Ask for the size first, then copy the voice out. */
    size_t len = 0;
    status = kt2n_voice_export(engine, voice, NULL, 0, &len);
    void* blob = (status == KT2N_ERR_TOO_SMALL && len > 0) ? malloc(len) : NULL;
    if (blob) status = kt2n_voice_export(engine, voice, blob, len, &len);
    if (!blob || status != KT2N_OK) {
      example_fail("voice export", blob ? status : KT2N_ERR_INTERNAL, NULL);
    } else {
      char part[EXAMPLE_PATH_MAX];
      FILE* out = example_open_output(out_path, part, sizeof part);
      if (out) {
        const int written = fwrite(blob, 1, len, out) == len;
        if (example_finish_output(out, out_path, part, written)) {
          fprintf(stderr, "saved the voice (%zu bytes) to %s\n", len, out_path);
          exit_code = 0;
        }
      }
    }
    free(blob);
    kt2n_voice_evict(engine, voice); /* done with it in this process */
  }
  kt2n_engine_destroy(engine);
  return exit_code;
}

/* speak: saved voice file + text -> complete WAV file. */
static int speak(const char* assets_dir, const char* voice_path, const char* text, const char* out_path) {
  example_bundle bundle;
  if (!example_bundle_init(&bundle, assets_dir)) return 1;
  size_t blob_len = 0;
  void* blob = example_read_file(voice_path, VOICE_FILE_LIMIT, &blob_len);
  if (!blob) return 1;

  kt2n_engine_config config;
  clone_config(&config, &bundle); /* no voice_build_dir: loading a saved voice needs no cloning files */

  kt2n_error error;
  memset(&error, 0, sizeof error);
  kt2n_engine* engine = NULL;
  fprintf(stderr, "loading the engine...\n");
  int32_t status = kt2n_engine_create(&config, &engine, &error);
  if (status != KT2N_OK) {
    free(blob);
    return example_fail("engine create", status, &error);
  }

  kt2n_voice_id voice = KT2N_INVALID_ID;
  status = kt2n_voice_load(engine, blob, blob_len, &voice, &error); /* checks the file against this bundle */
  free(blob);
  if (status != KT2N_OK) {
    example_fail("voice load", status, &error);
    kt2n_engine_destroy(engine);
    return 1;
  }

  char part[EXAMPLE_PATH_MAX];
  FILE* out = example_open_output(out_path, part, sizeof part);
  if (!out) {
    kt2n_voice_evict(engine, voice);
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
  params.format = KT2N_FORMAT_WAV;
  params.output = KT2N_OUTPUT_COMPLETE;

  example_catch_signals();
  kt2n_request_id request = KT2N_INVALID_ID;
  status = kt2n_submit(engine, &params, &request, &error);
  int done = 0, cancelled = 0, write_failed = 0;
  int32_t final_status = status;
  int64_t samples = 0;
  if (status != KT2N_OK) example_fail("submit", status, &error);
  while (status == KT2N_OK && !done) {
    if (example_interrupted && !cancelled) {
      kt2n_cancel(engine, request, KT2N_CANCEL_REQUEST);
      cancelled = 1;
    }
    kt2n_event events[16];
    int32_t count = 0;
    status = kt2n_poll(engine, events, 16, 100 /* ms */, &count);
    if (status != KT2N_OK) {
      example_fail("poll", status, NULL);
      final_status = status;
      break;
    }
    for (int32_t i = 0; i < count; ++i) {
      const kt2n_event* ev = &events[i];
      if (ev->id != request) continue;
      if (ev->kind == KT2N_EVENT_AUDIO && ev->data_len > 0 && !write_failed) {
        if (fwrite(ev->data, 1, ev->data_len, out) != ev->data_len) write_failed = 1;
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
  kt2n_voice_evict(engine, voice); /* the voice's memory is free for another one */

  const int ok = done && final_status == KT2N_OK && !write_failed && samples > 0;
  const int saved = example_finish_output(out, out_path, part, ok);
  int exit_code = 1;
  if (final_status == KT2N_ERR_CANCELLED) {
    fprintf(stderr, "cancelled; %s was not written\n", out_path);
    exit_code = 130;
  } else if (done && final_status != KT2N_OK) {
    example_fail("speech", final_status, &error);
  } else if (saved) {
    fprintf(stderr, "wrote %.2f s of audio to %s\n", (double)samples / EXAMPLE_SAMPLE_RATE, out_path);
    exit_code = 0;
  } else if (write_failed) {
    fprintf(stderr, "error: writing %s failed\n", out_path);
  }
  kt2n_engine_destroy(engine);
  return exit_code;
}

int main(int argc, char** argv) {
  if (argc == 6 && strcmp(argv[1], "build") == 0) return build_voice(argv[2], argv[3], argv[4], argv[5]);
  if (argc == 6 && strcmp(argv[1], "speak") == 0) return speak(argv[2], argv[3], argv[4], argv[5]);
  fprintf(stderr,
          "usage: %s build ASSETS_DIR REFERENCE_AUDIO TRANSCRIPT OUT.voice\n"
          "       %s speak ASSETS_DIR VOICE_FILE TEXT OUT.wav\n",
          argv[0], argv[0]);
  return 2;
}
