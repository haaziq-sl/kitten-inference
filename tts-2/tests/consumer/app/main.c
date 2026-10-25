/* Links against the installed library and calls the parts of the C API that
 * need no GPU and no model files. */
#include "kt2n/engine.h"

#include <stdio.h>
#include <string.h>

int consumer_cxx_abi_version_matches(void);

static int failures = 0;

static void expect(int ok, const char* what) {
  if (!ok) {
    fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

int main(void) {
  expect(kt2n_abi_version() == KT2N_ABI_VERSION, "kt2n_abi_version() matches the header");
  expect(consumer_cxx_abi_version_matches(), "the header works from C++");

  const char* info = kt2n_build_info();
  expect(info != NULL && info[0] != '\0', "kt2n_build_info() is a non-empty string");
  if (info) printf("build: %s\n", info);

  const kt2n_error_info* voice = kt2n_error_lookup(KT2N_ERR_INVALID_VOICE);
  expect(voice != NULL && strcmp(voice->code, "invalid_voice") == 0, "error lookup gives invalid_voice");
  expect(kt2n_error_lookup(-12345) != NULL, "error lookup never returns NULL");

  /* A config from a different ABI version is refused before anything is loaded. */
  kt2n_engine_config config;
  memset(&config, 0, sizeof config);
  config.struct_size = sizeof config;
  config.abi_version = KT2N_ABI_VERSION + 1;
  kt2n_engine* engine = NULL;
  kt2n_error error;
  memset(&error, 0, sizeof error);
  expect(kt2n_engine_create(&config, &engine, &error) == KT2N_ERR_ABI_MISMATCH && engine == NULL,
         "engine create refuses a mismatched abi_version");

  kt2n_event event;
  int32_t count = 0;
  expect(kt2n_poll(NULL, &event, 1, 0, &count) == KT2N_ERR_INVALID_ARGUMENT, "poll rejects a NULL engine");
  kt2n_engine_destroy(NULL); /* safe on NULL */

  if (failures) return 1;
  printf("ok\n");
  return 0;
}
