// log1p(float) by the fdlibm algorithm, as in glibc's s_log1pf.c, which the numpy reference and the Linux build call
// for the flow noise (src/render/rng.cc, ziggurat tail). Windows only: the MSVC runtime's log1pf differs from glibc's in
// the last bit for about 7% of the 2^24 arguments that code can pass (-k * 2^-24); this one matches glibc on all of them.
//
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
// Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
#pragma once

#include <cstdint>
#include <cstring>
#include <limits>

namespace kt2n::render {

inline float log1pf_fdlibm(float x) {
  constexpr float ln2_hi = 6.9313812256e-01f;  // 0x3f317180
  constexpr float ln2_lo = 9.0580006145e-06f;  // 0x3717f7d1
  constexpr float Lp1 = 6.6666668653e-01f;     // 3F2AAAAB
  constexpr float Lp2 = 4.0000000596e-01f;     // 3ECCCCCD
  constexpr float Lp3 = 2.8571429849e-01f;     // 3E924925
  constexpr float Lp4 = 2.2222198546e-01f;     // 3E638E29
  constexpr float Lp5 = 1.8183572590e-01f;     // 3E3A3325
  constexpr float Lp6 = 1.5313838422e-01f;     // 3E1CD04F
  constexpr float Lp7 = 1.4798198640e-01f;     // 3E178897
  auto word = [](float v) { int32_t w; std::memcpy(&w, &v, 4); return w; };
  auto from = [](int32_t w) { float v; std::memcpy(&v, &w, 4); return v; };
  float hfsq, f = 0.0f, c = 0.0f, s, z, R, u;
  int32_t k, hu = 0;
  const int32_t hx = word(x);
  const int32_t ax = hx & 0x7fffffff;
  k = 1;
  if (hx < 0x3ed413d7) {  // x < 0.41422
    if (ax >= 0x3f800000) {  // x <= -1.0
      if (x == -1.0f) return -std::numeric_limits<float>::infinity();
      return std::numeric_limits<float>::quiet_NaN();
    }
    if (ax < 0x31000000) {  // |x| < 2**-29
      if (ax < 0x24800000) return x;  // |x| < 2**-54
      return x - x * x * 0.5f;
    }
    if (hx > 0 || hx <= int32_t(0xbe95f61f)) {  // -0.2929 < x < 0.41422
      k = 0;
      f = x;
      hu = 1;
    }
  } else if (hx >= 0x7f800000) {
    return x + x;
  }
  if (k != 0) {
    if (hx < 0x5a000000) {
      u = 1.0f + x;
      hu = word(u);
      k = (hu >> 23) - 127;
      c = (k > 0) ? 1.0f - (u - x) : x - (u - 1.0f);  // correction term
      c /= u;
    } else {
      u = x;
      hu = word(u);
      k = (hu >> 23) - 127;
      c = 0;
    }
    hu &= 0x007fffff;
    if (hu < 0x3504f7) {
      u = from(hu | 0x3f800000);  // normalize u
    } else {
      k += 1;
      u = from(hu | 0x3f000000);  // normalize u/2
      hu = (0x00800000 - hu) >> 2;
    }
    f = u - 1.0f;
  }
  hfsq = 0.5f * f * f;
  if (hu == 0) {  // |f| < 2**-20
    if (f == 0.0f) {
      if (k == 0) return 0.0f;
      c += float(k) * ln2_lo;
      return float(k) * ln2_hi + c;
    }
    R = hfsq * (1.0f - 0.66666666666666666f * f);
    if (k == 0) return f - R;
    return float(k) * ln2_hi - ((R - (float(k) * ln2_lo + c)) - f);
  }
  s = f / (2.0f + f);
  z = s * s;
  R = z * (Lp1 + z * (Lp2 + z * (Lp3 + z * (Lp4 + z * (Lp5 + z * (Lp6 + z * Lp7))))));
  if (k == 0) return f - (hfsq - s * (hfsq + R));
  return float(k) * ln2_hi - ((hfsq - (s * (hfsq + R) + (float(k) * ln2_lo + c))) - f);
}

}  // namespace kt2n::render
