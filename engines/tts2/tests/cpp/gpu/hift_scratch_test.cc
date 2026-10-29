// kt2n_hift_create4 (the vocoder's run scratch carved from a caller buffer; the low-memory profile's shared
// workspace): the size it needs, and the refusals. That the output is the same as with the handle's own scratch is
// checked end to end in Runtime.GpuLowMemoryFittedAudioMatchesStandard (seeded PCM, standard vs fitted sizes).
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "kt2n/render/hift.h"
#include "test_util.h"

namespace {

TEST(HiftScratch, CompactScratchBytes) {
  // 2,048 frame-rows, 2 rows: 196,932 B per frame-row (f0, noise and sines live in later buffers) plus the 4 Mi-float
  // split-K partial buffer, each buffer rounded up to 256 B.
  const size_t b = kt2n_hift_compact_scratch_bytes(2, 2048);
  EXPECT_GE(b, size_t(2048) * 196932 + (size_t(4) << 22));
  EXPECT_LE(b, size_t(2048) * 196932 + (size_t(4) << 22) + (size_t(64) << 10));  // + per-row terms and rounding
  EXPECT_LT(b, size_t(2048) * 235588);  // the compact handle's own allocations per frame-row
  // the partial-sum buffer grows with the rows per launch above its floor (300,000 floats per row: 32 rows = 36.6 MiB)
  EXPECT_GE(kt2n_hift_compact_scratch_bytes(32, 2048) - b, size_t(32) * 300000 * 4 - (size_t(4) << 22));
}

TEST(HiftScratch, RefusesTooSmallOrNonCompact) {
  KT2N_GPU_OR_SKIP();
  void* dummy = nullptr;
  ASSERT_EQ(cudaMalloc(&dummy, 256), cudaSuccess);
  // The refusals come before any weight is read, so placeholder weight pointers are enough.
  std::vector<const void*> w(319, dummy);
  const size_t need = kt2n_hift_compact_scratch_bytes(2, 2048);
  void* scratch = nullptr;
  ASSERT_EQ(cudaMalloc(&scratch, need), cudaSuccess);
  EXPECT_EQ(kt2n_hift_create4(w.data(), int(w.size()), 2, 2048, 2048, KT2N_HIFT_COMPACT, scratch, need - 1), nullptr);
  EXPECT_NE(std::string(kt2n_hift_last_error()).find("smaller than"), std::string::npos) << kt2n_hift_last_error();
  EXPECT_EQ(kt2n_hift_create4(w.data(), int(w.size()), 2, 2048, 2048, 0, scratch, need), nullptr);
  EXPECT_NE(std::string(kt2n_hift_last_error()).find("KT2N_HIFT_COMPACT"), std::string::npos) << kt2n_hift_last_error();
  cudaFree(scratch);
  cudaFree(dummy);
}

}  // namespace
