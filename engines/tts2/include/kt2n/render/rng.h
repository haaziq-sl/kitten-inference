// Render state: numpy-exact flow-noise RNG and the render seed plumbing.
//
// Ports (the reference implementation; contract in include/kt2n/render/ops.h "RNG contract"):
//   * numpy 1.26 SeedSequence (bit_generator.pyx) + PCG64 XSL-RR 128/64 (src/pcg64/pcg64.h) with
//     numpy's buffered 32-bit half word (pcg64_next32), i.e. np.random.default_rng(seed);
//   * Generator.standard_normal(dtype=np.float32) = numpy's float ziggurat
//     (random_standard_normal_f, distributions.c), filled in C order;
//   * the reference state-noise draw (per-voice prompt noise) and the
//     the reference noise extension stream;
//   * the reference request seed salt and the HiFT seed.
// Host only (the draws are sequential: the ziggurat consumes a variable number of words).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "kt2n/render/ops.h"

namespace kt2n::render {

#if defined(__SIZEOF_INT128__) && !defined(KT2N_PORTABLE_U128)
// 128-bit PCG state (gcc/clang extension; __extension__ keeps -Wpedantic quiet).
__extension__ typedef unsigned __int128 u128;
#else
// Compilers without a 128-bit integer (MSVC): the operations PCG64 uses, exact modulo 2^128 like the built-in type.
// -DKT2N_PORTABLE_U128 selects it on gcc/clang too, to check it against unsigned __int128.
struct u128 {
  uint64_t hi = 0, lo = 0;
  constexpr u128() = default;
  constexpr u128(uint64_t v) : hi(0), lo(v) {}  // implicit, like the built-in type
  constexpr u128(uint64_t h, uint64_t l) : hi(h), lo(l) {}
  explicit constexpr operator uint64_t() const { return lo; }
  explicit constexpr operator uint32_t() const { return uint32_t(lo); }
  friend constexpr u128 operator|(u128 a, u128 b) { return {a.hi | b.hi, a.lo | b.lo}; }
  friend constexpr u128 operator+(u128 a, u128 b) {
    const uint64_t lo = a.lo + b.lo;
    return {a.hi + b.hi + (lo < a.lo ? 1u : 0u), lo};
  }
  u128& operator+=(u128 b) { return *this = *this + b; }
  // (a.hi 2^64 + a.lo)(b.hi 2^64 + b.lo) mod 2^128 = a.lo b.lo + 2^64 (a.hi b.lo + a.lo b.hi).
  friend constexpr u128 operator*(u128 a, u128 b) {
    const uint64_t a0 = a.lo & 0xffffffffu, a1 = a.lo >> 32, b0 = b.lo & 0xffffffffu, b1 = b.lo >> 32;
    const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const uint64_t mid = (p00 >> 32) + (p01 & 0xffffffffu) + (p10 & 0xffffffffu);
    const uint64_t lo = (mid << 32) | (p00 & 0xffffffffu);
    const uint64_t hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return {hi + a.hi * b.lo + a.lo * b.hi, lo};
  }
  friend constexpr u128 operator<<(u128 a, unsigned n) {
    return n == 0 ? a
           : n >= 128 ? u128{}
           : n >= 64  ? u128{a.lo << (n - 64), 0}
                      : u128{(a.hi << n) | (a.lo >> (64 - n)), a.lo << n};
  }
  friend constexpr u128 operator>>(u128 a, unsigned n) {
    return n == 0 ? a
           : n >= 128 ? u128{}
           : n >= 64  ? u128{0, a.hi >> (n - 64)}
                      : u128{a.hi >> n, (a.lo >> n) | (a.hi << (64 - n))};
  }
};
#endif

// numpy SeedSequence(entropy).generate_state(n_words, np.uint32) for a non-negative integer entropy and
// no spawn key (pool size 4). Entropy words are the integer's 32-bit words, lowest first ([0] for 0).
void seed_sequence_generate_state(uint64_t entropy, uint32_t* out, int32_t n_words);

// np.random.default_rng(seed)'s PCG64 bit generator.
class Pcg64 {
 public:
  explicit Pcg64(uint64_t seed);
  uint64_t next64();
  uint32_t next32();      // numpy pcg64_next32: low half first, high half buffered
  float next_float();     // (next32 >> 8) * 2^-24
  float standard_normal_f();
  // C-order fill, == rng.standard_normal(n, dtype=np.float32)
  void fill_standard_normal_f(float* out, size_t n);
  // Draw and discard n float32 normals (advances exactly as a fill would).
  void skip_standard_normal_f(size_t n);
  Pcg64State state() const;

 private:
  u128 state_ = 0;
  u128 inc_ = 0;
  uint32_t has_uint32_ = 0;
  uint32_t uinteger_ = 0;
};

// ---- flow noise (per-voice prompt noise, the default mode) -----------------------------------------
// Prompt-region noise shared by every segment of a voice: default_rng(4,000,000)
// .standard_normal((80, prompt_frames), float32). `out` row-major [80][stride], stride >= prompt_frames.
void draw_voice_prompt_noise(int32_t prompt_frames, float* out, int32_t stride);

// A row's target noise: rng = default_rng(4,000,000 + seed); rng's own (80, prompt_frames) draw is
// discarded (it still advances rng); then (80, W) with W = target_noise_width(budget) =
// (budget + 3) * 2. `out` row-major [80][stride] (StateView::target_noise slot, stride = noise_cap),
// frames [0, W) written. Returns W. Throws std::invalid_argument when W > stride.
int32_t draw_row_target_noise(uint64_t seed, int32_t prompt_frames, int32_t budget, float* out, int32_t stride);

// The extension of a row that outlives its budget (as the reference; once per row,
// triggered when the LM's new tokens reach the budget): frames [W, F) with
// F = noise_extension_frames(budget) from default_rng(5,000,000 + seed).standard_normal((80, F - W)),
// appended per channel. Returns F (or W when F <= W: nothing drawn). When F > stride, each channel keeps its first
// stride - W values (the rest are drawn and discarded, so the kept ones are the reference's) and stride is returned.
inline constexpr int32_t kSilenceFreeBudgetCeiling = 2;  // the reference silence-free budget ceiling
constexpr int32_t noise_extension_frames(int32_t budget) {
  return (budget * kSilenceFreeBudgetCeiling + kLookahead) * kMelPerToken;
}
int32_t draw_row_noise_extension(uint64_t seed, int32_t budget, float* out, int32_t stride);

// ---- seeds ------------------------------------------------------------------------------------------
// The reference salt: (big-endian u32 of sha256(request_id)[0:4] & 0x3FFFFFFF) * 1024.
uint64_t request_seed_salt(std::string_view request_id);
// Segment seed = salt + ordinal.
constexpr uint64_t segment_seed(uint64_t salt, int32_t ordinal) { return salt + uint64_t(ordinal); }
// Preview group RNG seed: the group's FIRST row's seed + 1,000,000.
constexpr uint64_t preview_seed(uint64_t first_row_seed) { return first_row_seed + kPreviewSeedOffset; }

}  // namespace kt2n::render
