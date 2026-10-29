/*
 * Small helpers shared by the examples: bundle paths, error messages, Ctrl-C,
 * clocks and files. Nothing here calls the engine except kt2n_error_lookup();
 * each example shows its own engine calls in full.
 */
#ifndef KT2N_EXAMPLE_UTIL_H_
#define KT2N_EXAMPLE_UTIL_H_

#include "kt2n/engine.h"

#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define EXAMPLE_PATH_MAX 4096
#define EXAMPLE_SAMPLE_RATE 24000 /* every output format is 24 kHz mono */

/* The files of a bundle made by `tools/prepare_assets.py --out ASSETS_DIR`. */
typedef struct example_bundle {
  char lm[EXAMPLE_PATH_MAX];          /* ASSETS_DIR/native/model.kt2n     */
  char s3gen[EXAMPLE_PATH_MAX];       /* ASSETS_DIR/native/s3gen.kt2n     */
  char voices[EXAMPLE_PATH_MAX];      /* ASSETS_DIR/native/voices.kt2n    */
  char tokenizer[EXAMPLE_PATH_MAX];   /* ASSETS_DIR/native/tokenizer.json */
  char text_dir[EXAMPLE_PATH_MAX];    /* ASSETS_DIR/native/text           */
  char cloning_dir[EXAMPLE_PATH_MAX]; /* ASSETS_DIR/native/cloning (optional) */
} example_bundle;

static inline int example_join(char* out, const char* dir, const char* name) {
  const int n = snprintf(out, EXAMPLE_PATH_MAX, "%s/native/%s", dir, name);
  return n > 0 && n < EXAMPLE_PATH_MAX;
}

static inline int example_exists(const char* path) {
  struct stat st;
  return stat(path, &st) == 0;
}

/* Fills every path and checks that the core files are there. Returns 1 on success. */
static inline int example_bundle_init(example_bundle* b, const char* assets_dir) {
  if (!example_join(b->lm, assets_dir, "model.kt2n") || !example_join(b->s3gen, assets_dir, "s3gen.kt2n") ||
      !example_join(b->voices, assets_dir, "voices.kt2n") ||
      !example_join(b->tokenizer, assets_dir, "tokenizer.json") || !example_join(b->text_dir, assets_dir, "text") ||
      !example_join(b->cloning_dir, assets_dir, "cloning")) {
    fprintf(stderr, "error: assets path is too long\n");
    return 0;
  }
  const char* required[] = {b->lm, b->s3gen, b->voices, b->tokenizer, b->text_dir};
  for (size_t i = 0; i < sizeof required / sizeof required[0]; ++i) {
    if (!example_exists(required[i])) {
      fprintf(stderr,
              "error: %s is missing.\n"
              "Prepare the files first: python3 tools/prepare_assets.py --out %s\n",
              required[i], assets_dir);
      return 0;
    }
  }
  return 1;
}

/* Prints "<what> failed: <message>" and returns the exit code 1. */
static inline int example_fail(const char* what, int32_t status, const kt2n_error* err) {
  const char* message = (err && err->message[0]) ? err->message : kt2n_error_lookup(status)->code;
  fprintf(stderr, "error: %s failed: %s (status %d)\n", what, message, (int)status);
  return 1;
}

/* Ctrl-C (SIGINT) and SIGTERM set this flag; the examples then cancel their work. */
static volatile sig_atomic_t example_interrupted = 0;

static inline void example_on_signal(int signal_number) {
  (void)signal_number;
  example_interrupted = 1;
}

static inline void example_catch_signals(void) {
  struct sigaction action;
  memset(&action, 0, sizeof action);
  action.sa_handler = example_on_signal;
  sigemptyset(&action.sa_mask);
  sigaction(SIGINT, &action, NULL);
  sigaction(SIGTERM, &action, NULL);
  signal(SIGPIPE, SIG_IGN); /* a closed output pipe becomes a write error instead */
}

/* CLOCK_MONOTONIC in nanoseconds: the clock of kt2n_event.at_ns. */
static inline int64_t example_now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

/* Reads a whole file of at most `limit` bytes. Returns a malloc'd buffer or NULL. */
static inline void* example_read_file(const char* path, size_t limit, size_t* len) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "error: cannot open %s: %s\n", path, strerror(errno));
    return NULL;
  }
  unsigned char* data = (unsigned char*)malloc(limit + 1);
  size_t n = data ? fread(data, 1, limit + 1, f) : 0;
  const int failed = !data || ferror(f);
  fclose(f);
  if (failed || n == 0 || n > limit) {
    fprintf(stderr, "error: %s is %s\n", path,
            failed ? "unreadable" : (n == 0 ? "empty" : "too large"));
    free(data);
    return NULL;
  }
  *len = n;
  return data;
}

/* Opens "<path>.part" for writing; example_finish_output() renames it to <path>
 * only after a complete result, so a failed or cancelled run never leaves a
 * truncated file under the final name. "-" means standard output. */
static inline FILE* example_open_output(const char* path, char* part, size_t part_cap) {
  if (strcmp(path, "-") == 0) return stdout;
  const int n = snprintf(part, part_cap, "%s.part", path);
  if (n <= 0 || (size_t)n >= part_cap) {
    fprintf(stderr, "error: output path is too long\n");
    return NULL;
  }
  FILE* f = fopen(part, "wb");
  if (!f) fprintf(stderr, "error: cannot write %s: %s\n", part, strerror(errno));
  return f;
}

/* Closes the output; keeps it (rename) when ok != 0, otherwise deletes it. Returns 1 on success. */
static inline int example_finish_output(FILE* f, const char* path, const char* part, int ok) {
  if (f == stdout) return fflush(stdout) == 0 && ok;
  const int closed = fclose(f) == 0;
  if (ok && closed && rename(part, path) == 0) return 1;
  if (ok) fprintf(stderr, "error: cannot write %s: %s\n", path, strerror(errno));
  remove(part);
  return 0;
}

#endif /* KT2N_EXAMPLE_UTIL_H_ */
