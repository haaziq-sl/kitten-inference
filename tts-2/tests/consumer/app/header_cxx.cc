// The public header must also compile as C++ and keep C linkage.
#include "kt2n/engine.h"

extern "C" int consumer_cxx_abi_version_matches(void) {
  static_assert(sizeof(kt2n_voice_id) == 8, "handles are 64-bit ids");
  return kt2n_abi_version() == KT2N_ABI_VERSION ? 1 : 0;
}
