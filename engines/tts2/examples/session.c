/*
 * session.c: speak text that arrives a piece at a time, for example from a
 * chat model writing its answer.
 *
 *   session ASSETS_DIR VOICE OUT.pcm < text
 *
 * Standard input is read as it arrives. Every complete line is passed on with
 * kt2n_session_text_append(); the engine cuts the text into phrases and starts
 * speaking before the input is finished. An empty line commits: everything sent
 * so far is spoken without waiting for more text. End of input closes the
 * session. Audio goes to OUT.pcm ("-" for standard output) as raw 16-bit
 * little-endian mono PCM at 24 kHz. Try:
 *
 *   (echo "Hello there."; sleep 1; echo "This line arrived a second later.") |
 *     session models/kitten Bruno - | aplay -q -f S16_LE -r 24000 -c 1
 *
 * The loop waits on two file descriptors at once with poll(2): standard input and
 * the engine's event fd (kt2n_event_fd), which is readable while events are
 * queued. Ctrl-C cancels the session. Sessions use built-in voices.
 */
#define _POSIX_C_SOURCE 200809L
#include "example_util.h"

#include <poll.h>
#include <unistd.h>

#define LINE_MAX_BYTES 4096

/* Length of the longest prefix of buf that does not end inside a UTF-8 sequence. */
static size_t utf8_complete_prefix(const char* buf, size_t len) {
  size_t back = 0;
  while (back < len && back < 4 && ((unsigned char)buf[len - 1 - back] & 0xC0) == 0x80) ++back;
  if (back < len) {
    const unsigned char lead = (unsigned char)buf[len - 1 - back];
    const size_t need = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
    if (need > back + 1) return len - 1 - back; /* the last sequence is cut short */
  }
  return len;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s ASSETS_DIR VOICE OUT.pcm|- < text\n", argv[0]);
    return 2;
  }
  const char* assets_dir = argv[1];
  const char* voice_name = argv[2];
  const char* out_path = argv[3];

  example_bundle bundle;
  if (!example_bundle_init(&bundle, assets_dir)) return 1;

  kt2n_engine_config config;
  memset(&config, 0, sizeof config);
  config.struct_size = sizeof config;
  config.abi_version = KT2N_ABI_VERSION;
  config.flags = KT2N_CFG_INFERENCE_ONLY;
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
   * built-in voice prepared) need about 21 GB of free GPU memory. Complete-file audio is the same in all of these
   * settings; on GPUs of about 4 GB, long streamed text may be split into shorter pieces, which changes its audio. */
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

  FILE* out = strcmp(out_path, "-") == 0 ? stdout : fopen(out_path, "wb");
  if (!out) {
    fprintf(stderr, "error: cannot write %s: %s\n", out_path, strerror(errno));
    kt2n_engine_destroy(engine);
    return 1;
  }

  kt2n_session_params params;
  memset(&params, 0, sizeof params);
  params.struct_size = sizeof params;
  params.voice = voice;
  params.speed = 1.0;
  params.format = KT2N_FORMAT_PCM; /* sessions produce PCM16 (or float32) */

  example_catch_signals();
  kt2n_session_id session = KT2N_INVALID_ID;
  status = kt2n_session_open(engine, &params, &session, &error);
  if (status != KT2N_OK) {
    example_fail("session open", status, &error);
    if (out != stdout) fclose(out);
    kt2n_engine_destroy(engine);
    return 1;
  }

  const int event_fd = kt2n_event_fd(engine);
  char line[LINE_MAX_BYTES];
  size_t have = 0;
  int input_open = 1, done = 0, cancelled = 0, write_failed = 0;
  int32_t final_status = KT2N_OK;
  int64_t samples = 0;
  while (!done) {
    if (example_interrupted && !cancelled) {
      kt2n_cancel(engine, session, KT2N_CANCEL_REQUEST); /* stops every phrase of the session */
      cancelled = 1;
      input_open = 0;
    }

    struct pollfd fds[2] = {{event_fd, POLLIN, 0}, {STDIN_FILENO, POLLIN, 0}};
    if (poll(fds, input_open ? 2 : 1, 100 /* ms */) < 0 && errno != EINTR) {
      fprintf(stderr, "error: poll: %s\n", strerror(errno));
      break;
    }

    /* 1. New text: pass on complete lines; an empty line commits. */
    if (input_open && (fds[1].revents & (POLLIN | POLLHUP | POLLERR))) {
      const ssize_t got = read(STDIN_FILENO, line + have, sizeof line - have);
      if (got > 0) {
        have += (size_t)got;
        char* start = line;
        char* newline;
        while ((newline = memchr(start, '\n', have - (size_t)(start - line))) != NULL) {
          const size_t len = (size_t)(newline - start);
          if (len == 0) {
            status = kt2n_session_commit(engine, session, &error); /* speak what has arrived */
          } else {
            *newline = ' '; /* keep a space between this line and the next */
            status = kt2n_session_text_append(engine, session, start, len + 1, &error);
          }
          if (status != KT2N_OK) example_fail("session input", status, &error);
          start = newline + 1;
        }
        have -= (size_t)(start - line);
        memmove(line, start, have);
        if (have == sizeof line) { /* a very long line: pass on what is complete so far */
          const size_t len = utf8_complete_prefix(line, have);
          status = kt2n_session_text_append(engine, session, line, len, &error);
          if (status != KT2N_OK) example_fail("session input", status, &error);
          have -= len;
          memmove(line, line + len, have);
        }
      } else if (got == 0 || (errno != EINTR && errno != EAGAIN)) {
        if (got < 0) fprintf(stderr, "error: reading standard input: %s\n", strerror(errno));
        /* End of input: pass on the rest, then close the session. */
        if (have > 0) kt2n_session_text_append(engine, session, line, have, &error);
        have = 0;
        status = kt2n_session_done(engine, session, &error); /* finish every phrase, then DONE */
        if (status != KT2N_OK) example_fail("session done", status, &error);
        input_open = 0;
      }
    }

    /* 2. Events: read the event fd to reset it, then drain the queue without waiting. */
    if (fds[0].revents & POLLIN) {
      uint64_t ignored;
      if (read(event_fd, &ignored, sizeof ignored) < 0 && errno != EAGAIN) break;
    }
    for (;;) {
      kt2n_event events[16];
      int32_t count = 0;
      status = kt2n_poll(engine, events, 16, 0 /* never wait */, &count);
      if (status != KT2N_OK || count == 0) break;
      for (int32_t i = 0; i < count; ++i) {
        const kt2n_event* ev = &events[i];
        if (ev->id != session) continue;
        if (ev->kind == KT2N_EVENT_AUDIO && ev->data_len > 0 && !cancelled) {
          if (fwrite(ev->data, 1, ev->data_len, out) != ev->data_len || fflush(out) != 0) {
            kt2n_cancel(engine, session, KT2N_CANCEL_REQUEST);
            cancelled = write_failed = 1;
            input_open = 0;
          }
          samples += ev->samples;
        } else if (ev->kind == KT2N_EVENT_PHRASE_DONE) {
          fprintf(stderr, "phrase %d done (%.2f s of audio so far)\n", ev->phrase_index,
                  (double)samples / EXAMPLE_SAMPLE_RATE);
        } else if (ev->kind == KT2N_EVENT_DONE) {
          done = 1;
          final_status = ev->status;
          if (ev->error) error = *ev->error;
        }
      }
      kt2n_events_release(engine, events, count);
    }
  }
  if (done) kt2n_release(engine, session);

  if (out != stdout && fclose(out) != 0) write_failed = 1;
  int exit_code = 0;
  if (write_failed) {
    fprintf(stderr, "error: writing %s failed\n", out_path);
    exit_code = 1;
  } else if (final_status == KT2N_ERR_CANCELLED) {
    fprintf(stderr, "cancelled after %.2f s of audio\n", (double)samples / EXAMPLE_SAMPLE_RATE);
    exit_code = 130;
  } else if (!done || final_status != KT2N_OK) {
    exit_code = done ? example_fail("session", final_status, &error) : 1;
  } else {
    fprintf(stderr, "wrote %.2f s of audio to %s\n", (double)samples / EXAMPLE_SAMPLE_RATE, out_path);
  }
  kt2n_engine_destroy(engine);
  return exit_code;
}
