// The portable 128-bit integer behind the PCG64 state on compilers without one (MSVC; include/kt2n/render/rng.h).
// On gcc/clang this translation unit selects it with KT2N_PORTABLE_U128 and checks every operation PCG64 uses
// against unsigned __int128; with MSVC it checks multiplication against _umul128. The test links no rng.cc, so the
// two u128 definitions never meet in one program.
#if !defined(_MSC_VER)
#define KT2N_PORTABLE_U128 1
#endif
#include "kt2n/render/rng.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace {

using kt2n::render::u128;

#if !defined(_MSC_VER)
__extension__ typedef unsigned __int128 native128;
native128 nat(u128 v) { return (native128(v.hi) << 64) | v.lo; }
#endif

TEST(U128, MatchesExactArithmetic) {
  std::mt19937_64 g(20261022);
  for (int i = 0; i < 200000; ++i) {
    const u128 a{g(), g()}, b{g(), g()};
    const unsigned s = unsigned(g() % 128);
#if !defined(_MSC_VER)
    const native128 na = nat(a), nb = nat(b);
    ASSERT_EQ(nat(a * b), na * nb);
    ASSERT_EQ(nat(a + b), na + nb);
    ASSERT_EQ(nat(a | b), na | nb);
    ASSERT_EQ(nat(a << s), na << s);
    ASSERT_EQ(nat(a >> s), na >> s);
    ASSERT_EQ(static_cast<uint64_t>(a), uint64_t(na));
    ASSERT_EQ(static_cast<uint32_t>(a >> 122u), uint32_t(na >> 122));
#else
    uint64_t hi = 0;
    const uint64_t lo = _umul128(a.lo, b.lo, &hi);
    const u128 m = a * b;
    ASSERT_EQ(m.lo, lo);
    ASSERT_EQ(m.hi, hi + a.hi * b.lo + a.lo * b.hi);
    const u128 sum = a + b;
    ASSERT_EQ(sum.lo, a.lo + b.lo);
    ASSERT_EQ(sum.hi, a.hi + b.hi + (sum.lo < a.lo ? 1u : 0u));
    const u128 l = a << s, r = a >> s;
    if (s == 0) {
      ASSERT_EQ(l.hi, a.hi);
      ASSERT_EQ(r.lo, a.lo);
    } else if (s < 64) {
      ASSERT_EQ(l.lo, a.lo << s);
      ASSERT_EQ(r.hi, a.hi >> s);
    } else {
      ASSERT_EQ(l.lo, 0u);
      ASSERT_EQ(r.hi, 0u);
    }
#endif
  }
}

}  // namespace
