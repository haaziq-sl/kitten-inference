// HIP builds only (RDNA3 / RDNA4): the decode attention and the LM's decode / prefill parity in the default
// configuration, where the decode step's projections and attention and the prefill's attention run on the matrix unit
// (KT2N_AMD_WMMA_DECODE=2, src/hip/attn_wmma.cuh). Also valid on the software path (KT2N_AMD_WMMA=0); skipped (exit
// 77) for the test-only mixed modes KT2N_AMD_WMMA_DECODE=0 / 1, which keep prefill and decode on different arithmetic.
//
//   A. kt2n_attn_decode_rows (the prefill's per-token attention in this mode) vs an fp64 CPU reference: rows of 1 to
//      1000 keys (1 to 16 chunks, 1 to 8 splits, one row at a key offset), within 2 bf16 ulp (floor: 2^-16 of the
//      terms' size where the result cancels to ~0); each row alone == the row in the batch, bitwise.
//   B. the real model (KT2N_REAL_ENGINE_FILE, else KT2N_TEST_ASSETS/native/model.kt2n; skipped without it), 64 prompt
//      tokens + 640 teacher-forced audio tokens (keys up to 704):
//      - one-pass prefill of 360 and 704 tokens == the incremental prefill's logits at that position, bitwise;
//      - decode step (kt2n_step_forward_o1) with the engine's own per-bucket options (kt2n::lm::Engine defaults:
//        SPLIT64 for buckets <= 2 rows, no fused norm, no KV_NEXT) for buckets of 1, 2, 3, 16 and 20 rows:
//        every row of a bucket == row 0, bitwise; decode logits == incremental prefill logits bitwise at every
//        position in the buckets without SPLIT64. SPLIT64 buckets (<= 2 rows) fold the softmax of rows with <= 640
//        keys in another order (as on NVIDIA), and the K/V those steps write carry the difference to every later
//        position, so there: bucket 2 == bucket 1 bitwise (same options, same bits per row) and agreement with the
//        prefill as between two GPUs (sanity bounds: top-1 >= 90%, mean KL <= 2e-2; measured ~97%, ~4.4e-3, the same
//        on the software path).
// Exit 0 = pass, 1 = fail, 77 = skipped.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <random>
#include <string>
#include <vector>

#include "kt2n/hip/wmma_kernels.h"
#include "kt2n/kernels/attn.h"
#include "kt2n/kernels/step.h"
#include "kt2n/kernels/step_o1.h"
#include "kt2n/prefill.h"

namespace {

int g_fail = 0;

void check(bool ok, const char* name, const std::string& detail) {
  std::printf("%-58s %s  %s\n", name, ok ? "PASS" : "FAIL", detail.c_str());
  if (!ok) ++g_fail;
}

#define CK(x)                                                                                    \
  do {                                                                                           \
    const cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                                     \
      std::printf("%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));             \
      std::exit(1);                                                                              \
    }                                                                                            \
  } while (0)
#define PF(x)                                                                                    \
  do {                                                                                           \
    const int r_ = (x);                                                                          \
    if (r_) {                                                                                    \
      std::printf("%s:%d %s: rc %d: %s\n", __FILE__, __LINE__, #x, r_, kt2n_pf_last_error());     \
      std::exit(1);                                                                              \
    }                                                                                            \
  } while (0)

// Release device memory / streams at the end of a check (errors there do not change a result).
void dfree(void* p) { (void)cudaFree(p); }

float bf2f(uint16_t h) {
  const uint32_t u = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
uint16_t f2bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
// One bf16 ulp at |x| (8 significant bits).
double bf16_ulp(double x) {
  const double a = std::fabs(x);
  if (a < 1e-30) return std::ldexp(1.0, -133);
  return std::ldexp(1.0, std::ilogb(a) - 7);
}

// ------------------------------------------------------------------------------------------- A: decode attention
void test_attention() {
  constexpr int HQ = KT2N_ATTN_HEADS_Q, HKV = KT2N_ATTN_HEADS_KV, D = KT2N_ATTN_HEAD_DIM;
  const std::vector<int> lens = {1, 7, 64, 65, 128, 129, 300, 640, 641, 1000};
  const std::vector<int> starts = {0, 0, 0, 0, 0, 3, 0, 0, 0, 0};  // one row at a column offset
  const int rows = int(lens.size()), ctx = 1024;
  const int64_t head_stride = int64_t(ctx) * D, slot_stride = int64_t(HKV) * head_stride, qs = int64_t(HQ) * D;
  std::mt19937 rng(20261022);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  std::vector<uint16_t> q(size_t(rows) * qs), k(size_t(rows) * slot_stride), v(size_t(rows) * slot_stride);
  for (auto& x : q) x = f2bf(4.f * u(rng));
  for (auto& x : k) x = f2bf(u(rng));
  for (auto& x : v) x = f2bf(u(rng));
  std::vector<int32_t> slot(rows), start(starts), len(lens);
  for (int r = 0; r < rows; ++r) slot[r] = r;

  void *dq, *dk, *dv, *dout, *dout1, *ws;
  int32_t *dslot, *dstart, *dlen;
  const size_t wsb = kt2n_attn_decode_workspace_bytes(rows, ctx);
  CK(cudaMalloc(&dq, q.size() * 2));
  CK(cudaMalloc(&dk, k.size() * 2));
  CK(cudaMalloc(&dv, v.size() * 2));
  CK(cudaMalloc(&dout, size_t(rows) * qs * 2));
  CK(cudaMalloc(&dout1, size_t(rows) * qs * 2));
  CK(cudaMalloc(&ws, wsb));
  CK(cudaMalloc(&dslot, rows * 4));
  CK(cudaMalloc(&dstart, rows * 4));
  CK(cudaMalloc(&dlen, rows * 4));
  CK(cudaMemset(ws, 0, wsb));
  CK(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dk, k.data(), k.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dv, v.data(), v.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dslot, slot.data(), rows * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dstart, start.data(), rows * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dlen, len.data(), rows * 4, cudaMemcpyHostToDevice));
  int rc = kt2n_attn_decode_rows(dq, qs, dk, dv, slot_stride, head_stride, dslot, dstart, dlen, rows, ctx, dout, qs,
                                 ws, wsb, 0u, nullptr);
  // each row alone (same kernel, rows = 1)
  for (int r = 0; r < rows && rc == 0; ++r)
    rc = kt2n_attn_decode_rows(static_cast<uint16_t*>(dq) + r * qs, qs, dk, dv, slot_stride, head_stride, dslot + r,
                               dstart + r, dlen + r, 1, ctx, static_cast<uint16_t*>(dout1) + r * qs, qs, ws, wsb, 0u,
                               nullptr);
  CK(cudaDeviceSynchronize());
  check(rc == 0, "attention: kt2n_attn_decode_rows launches", "rc " + std::to_string(rc));
  std::vector<uint16_t> out(size_t(rows) * qs), out1(size_t(rows) * qs);
  CK(cudaMemcpy(out.data(), dout, out.size() * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(out1.data(), dout1, out1.size() * 2, cudaMemcpyDeviceToHost));

  // fp64 reference: s_j = q.k_j / sqrt(128), softmax over the row's keys, out = sum p_j v_j / sum p_j
  double max_ulp = 0;
  long n1 = 0, n = 0, bad = 0;
  for (int r = 0; r < rows; ++r)
    for (int h = 0; h < HQ; ++h) {
      const int kvh = h / 2;
      const uint16_t* qh = q.data() + size_t(r) * qs + size_t(h) * D;
      std::vector<double> s(len[r]);
      double m = -1e300;
      for (int j = 0; j < len[r]; ++j) {
        const uint16_t* kj = k.data() + size_t(r) * slot_stride + size_t(kvh) * head_stride + size_t(start[r] + j) * D;
        double acc = 0;
        for (int d = 0; d < D; ++d) acc += double(bf2f(qh[d])) * double(bf2f(kj[d]));
        s[j] = acc / std::sqrt(double(D));
        m = std::max(m, s[j]);
      }
      double l = 0;
      for (auto& x : s) l += (x = std::exp(x - m));
      for (int d = 0; d < D; ++d) {
        double o = 0, oabs = 0;
        for (int j = 0; j < len[r]; ++j) {
          const double vj = bf2f(v[size_t(r) * slot_stride + size_t(kvh) * head_stride + size_t(start[r] + j) * D + d]);
          o += s[j] * vj;
          oabs += s[j] * std::fabs(vj);
        }
        o /= l;
        oabs /= l;
        // error in bf16 ulp of the result, with a floor of 2^-16 of the terms' size for results that cancel to ~0
        // (fp32-level accumulation noise there; a real kernel error is at the 2^-8 level of the terms)
        const double got = bf2f(out[size_t(r) * qs + size_t(h) * D + d]);
        const double e = std::fabs(got - o) / (bf16_ulp(o) + std::ldexp(oabs, -16));
        max_ulp = std::max(max_ulp, e);
        n1 += e <= 1.0;
        bad += e > 2.0;
        ++n;
      }
    }
  char buf[160];
  std::snprintf(buf, sizeof buf, "%ld outputs, max %.2f bf16 ulp, %.3f%% within 1 ulp", n, max_ulp, 100.0 * n1 / n);
  check(bad == 0, "attention: 1..1000 keys vs fp64 reference (<= 2 ulp)", buf);
  long diff = 0;
  for (size_t i = 0; i < out.size(); ++i) diff += out[i] != out1[i];
  check(diff == 0, "attention: each row alone == in the batch (bitwise)", std::to_string(diff) + " elements differ");
  for (void* p : {dq, dk, dv, dout, dout1, ws, static_cast<void*>(dslot), static_cast<void*>(dstart),
                  static_cast<void*>(dlen)})
    dfree(p);
}

// -------------------------------------------------------------------------------------------- B: real model LM
const char* model_path() {
  static const std::string p = [] {
    const char* e = std::getenv("KT2N_REAL_ENGINE_FILE");
    if (e && *e) return std::string(e);
    const char* root = std::getenv("KT2N_TEST_ASSETS");
    return (root && *root) ? std::string(root) + "/native/model.kt2n" : std::string();
  }();
  return p.empty() ? nullptr : p.c_str();
}

// KL(p || q) of two logit rows (softmax in fp64) and whether their argmax agrees.
void kl_top1(const uint16_t* a, const uint16_t* b, int n, double* kl, bool* top1) {
  double ma = -1e300, mb = -1e300;
  int ia = 0, ib = 0;
  for (int i = 0; i < n; ++i) {
    if (bf2f(a[i]) > ma) ma = bf2f(a[i]), ia = i;
    if (bf2f(b[i]) > mb) mb = bf2f(b[i]), ib = i;
  }
  double za = 0, zb = 0;
  for (int i = 0; i < n; ++i) za += std::exp(bf2f(a[i]) - ma), zb += std::exp(bf2f(b[i]) - mb);
  double k = 0;
  for (int i = 0; i < n; ++i) {
    const double la = bf2f(a[i]) - ma - std::log(za), lb = bf2f(b[i]) - mb - std::log(zb);
    k += std::exp(la) * (la - lb);
  }
  *kl = k;
  *top1 = ia == ib;
}

void test_lm(const char* path) {
  constexpr int kAudioLow = KT2N_PF_AUDIO_LOW, kAudioN = KT2N_PF_AUDIO_N;
  constexpr int kP = 64, kT = 640, kCtx = 1024, kRmax = 20;
  kt2n_pf_model* model = nullptr;
  PF(kt2n_pf_model_load(path, 0, nullptr, &model));
  const kt2n_pf_weights* w = kt2n_pf_model_weights(model);
  const int L = w->layers;
  kt2n_pf_ctx* ctx = nullptr;
  PF(kt2n_pf_ctx_create(w, 0, 16, 1024, kCtx, &ctx));
  cudaStream_t s;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  const unsigned pf_flags = KT2N_PF_FLAGS_FAST | KT2N_PF_FLAG_NO_PDL;

  std::vector<int64_t> ids(kP + kT);
  std::vector<int32_t> audio(kT);
  {
    std::mt19937 rng(7);
    for (int i = 0; i < kP; ++i) ids[i] = 1000 + int64_t(rng() % 29000u);
    for (int i = 0; i < kT; ++i) audio[i] = int32_t(rng() % 6561u);  // audio codes
    for (int i = 0; i < kT; ++i) ids[kP + i] = kAudioLow + audio[i];
  }
  const int64_t row_stride = int64_t(2) * L * 8 * kCtx * 128;
  uint16_t *kv = nullptr, *dlog = nullptr;
  CK(cudaMalloc(&kv, size_t(kRmax) * row_stride * 2));
  CK(cudaMalloc(&dlog, size_t(kRmax) * kAudioN * 2));

  // incremental prefill: the prompt, then one audio token per call; logits after each
  std::vector<uint16_t> pf(size_t(kT + 1) * kAudioN);
  for (int i = 0; i <= kT; ++i) {
    const int32_t nt = i == 0 ? kP : 1, past = i == 0 ? 0 : kP + i - 1;
    PF(kt2n_pf_run(ctx, 1, &nt, ids.data() + (i == 0 ? 0 : kP + i - 1), &past, nullptr, nullptr, kv, row_stride, kCtx,
                   nullptr, nullptr, dlog, kAudioN, pf_flags, s));
    CK(cudaMemcpyAsync(pf.data() + size_t(i) * kAudioN, dlog, kAudioN * 2, cudaMemcpyDeviceToHost, s));
  }
  CK(cudaStreamSynchronize(s));

  // one-pass prefill == incremental at the same position
  for (int n : {360, kP + kT}) {
    const int32_t nt = n, past = 0;
    PF(kt2n_pf_run(ctx, 1, &nt, ids.data(), &past, nullptr, nullptr, kv, row_stride, kCtx, nullptr, nullptr, dlog,
                   kAudioN, pf_flags, s));
    std::vector<uint16_t> b(kAudioN);
    CK(cudaMemcpyAsync(b.data(), dlog, kAudioN * 2, cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    const bool same = std::memcmp(b.data(), pf.data() + size_t(n - kP) * kAudioN, kAudioN * 2) == 0;
    check(same, ("LM: one-pass prefill of " + std::to_string(n) + " tokens == incremental").c_str(),
          same ? "bitwise" : "logits differ");
  }

  // the engine's per-bucket step options (kt2n::lm::Engine defaults, src/lm/lm_engine.cu)
  for (int r = 1; r <= KT2N_STEP_MAX_ROWS_EXT; ++r) {
    unsigned o = kt2n_step_o1_options_for(r) & ~(KT2N_STEP_O1_SPLIT64 | KT2N_STEP_O1_FUSE_NORM | KT2N_STEP_O1_KV_NEXT);
    if (r <= 2) o |= KT2N_STEP_O1_SPLIT64;
    kt2n_step_o1_set_bucket_options(r, o);
  }
  std::vector<kt2n_step_layer> layers(L);
  for (int l = 0; l < L; ++l) {
    const kt2n_pf_layer& p = w->layer[l];
    kt2n_step_layer& q = layers[l];
    std::memset(&q, 0, sizeof(q));
    q.qkv_codes = p.qkv_codes, q.qkv_scales = p.qkv_scales, q.o_codes = p.o_codes, q.o_scales = p.o_scales;
    q.gate_up_codes = p.gu_codes, q.gate_up_scales = p.gu_scales, q.down_codes = p.down_codes;
    q.down_scales = p.down_scales, q.ln_in = p.input_norm, q.ln_post = p.post_norm, q.q_norm = p.q_norm;
    q.k_norm = p.k_norm;
    q.k_cache = kv + size_t(l) * 8 * kCtx * 128;
    q.v_cache = kv + (size_t(L) + l) * 8 * kCtx * 128;
  }
  std::vector<uint16_t> split64_ref;  // row-0 logits of the first SPLIT64 bucket, every step
  for (int R : {1, 2, 3, 16, 20}) {
    for (int r0 = 0; r0 < R; r0 += 16) {  // the prompt's K/V into rows 0..R-1
      const int nr = std::min(16, R - r0);
      std::vector<int32_t> nts(nr, kP), pasts(nr, 0), dest(nr);
      std::vector<int64_t> pid;
      for (int r = 0; r < nr; ++r) {
        dest[r] = r0 + r;
        pid.insert(pid.end(), ids.begin(), ids.begin() + kP);
      }
      PF(kt2n_pf_run(ctx, nr, nts.data(), pid.data(), pasts.data(), nullptr, nullptr, kv, row_stride, kCtx,
                     dest.data(), nullptr, nullptr, 0, pf_flags, s));
    }
    kt2n_step_model m;
    std::memset(&m, 0, sizeof(m));
    m.abi = KT2N_STEP_ABI;
    m.layers = L;
    m.vocab = w->embed_rows ? w->embed_rows : w->vocab;
    m.embed_row0 = w->embed_row0;
    m.rms_eps = w->rms_eps;
    m.head_n = kAudioN;
    m.embed = w->embed;
    m.final_norm = w->final_norm;
    m.head_w = static_cast<const char*>(w->embed) + size_t(kAudioLow - w->embed_row0) * 2048 * 2;
    m.slots = R;
    m.max_ctx = kCtx;
    m.kv_slot_stride = row_stride;
    m.kv_head_stride = int64_t(kCtx) * 128;
    m.layer = layers.data();
    const size_t sb = kt2n_step_scratch_bytes(R), wb = kt2n_step_attn_ws_bytes(R, 2 * kCtx);
    void *scratch = nullptr, *ws = nullptr;
    int64_t* dvec = nullptr;
    CK(cudaMalloc(&scratch, sb));
    CK(cudaMalloc(&ws, wb));
    CK(cudaMalloc(&dvec, 2 * R * sizeof(int64_t)));
    if (const int rc = kt2n_step_attn_ws_init(ws, wb, s)) {
      std::printf("kt2n_step_attn_ws_init rc %d\n", rc);
      std::exit(1);
    }
    kt2n_step_io io;
    std::memset(&io, 0, sizeof(io));
    io.rows = R;
    io.flags = KT2N_STEP_FLAG_ROW_KV;
    io.tokens = dvec;
    io.positions = dvec + R;
    io.cursor = dvec + R;
    io.scratch = scratch;
    io.scratch_bytes = sb;
    io.attn_ws = ws;
    io.attn_ws_bytes = wb;
    io.logits = dlog;
    io.ld_logits = kAudioN;
    const bool split64 = (kt2n_step_o1_options_for(R) & KT2N_STEP_O1_SPLIT64) != 0;
    std::vector<uint16_t> hl(size_t(R) * kAudioN), st(size_t(kT) * kAudioN);
    std::vector<int64_t> hv(2 * R);
    long row_diff = 0, bitwise = 0, top1 = 0;
    double kl_sum = 0;
    int rc = 0;
    for (int i = 0; i < kT && rc == 0; ++i) {
      for (int r = 0; r < R; ++r) hv[r] = kAudioLow + audio[i], hv[R + r] = kP + i;
      CK(cudaMemcpyAsync(dvec, hv.data(), hv.size() * 8, cudaMemcpyHostToDevice, s));
      rc = kt2n_step_forward_o1(&m, &io, s);
      CK(cudaMemcpyAsync(hl.data(), dlog, hl.size() * 2, cudaMemcpyDeviceToHost, s));
      CK(cudaStreamSynchronize(s));
      for (int r = 1; r < R; ++r) row_diff += std::memcmp(hl.data(), hl.data() + size_t(r) * kAudioN, kAudioN * 2) != 0;
      std::memcpy(st.data() + size_t(i) * kAudioN, hl.data(), kAudioN * 2);
      const uint16_t* ref = pf.data() + size_t(i + 1) * kAudioN;  // the prefill's logits after the same token
      bitwise += std::memcmp(hl.data(), ref, kAudioN * 2) == 0;
      double kl;
      bool t1;
      kl_top1(ref, hl.data(), kAudioN, &kl, &t1);
      top1 += t1, kl_sum += kl;
    }
    const std::string b = "bucket " + std::to_string(R) + (split64 ? " (SPLIT64)" : "");
    check(rc == 0, ("LM decode " + b + ": kt2n_step_forward_o1").c_str(), "rc " + std::to_string(rc));
    check(row_diff == 0, ("LM decode " + b + ": every row == row 0").c_str(),
          std::to_string(row_diff) + " (step, row) pairs differ");
    char buf[160];
    std::snprintf(buf, sizeof buf, "bitwise %ld/%d, top-1 %ld/%d, mean KL %.2e", bitwise, kT, top1, kT, kl_sum / kT);
    if (!split64) {
      check(bitwise == kT, ("LM decode " + b + ": == prefill logits (bitwise)").c_str(), buf);
    } else {
      check(top1 >= 0.90 * kT && kl_sum / kT <= 2e-2, ("LM decode " + b + ": ~ prefill logits").c_str(), buf);
      if (split64_ref.empty()) {
        split64_ref = st;
      } else {
        long d = 0;
        for (int i = 0; i < kT; ++i)
          d += std::memcmp(st.data() + size_t(i) * kAudioN, split64_ref.data() + size_t(i) * kAudioN, kAudioN * 2) != 0;
        check(d == 0, ("LM decode " + b + ": == the first SPLIT64 bucket (bitwise)").c_str(),
              std::to_string(kT - d) + "/" + std::to_string(kT) + " steps bitwise");
      }
    }
    for (void* p : {scratch, ws, static_cast<void*>(dvec)}) dfree(p);
  }
  dfree(kv);
  dfree(dlog);
  (void)cudaStreamDestroy(s);
  kt2n_pf_ctx_free(ctx);
  kt2n_pf_model_free(model);
}

}  // namespace

int main() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
    std::printf("no GPU: skipped\n");
    return 77;
  }
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));
  const int wmma = kt2n_wmma_enabled(), mode = kt2n_wmma_decode_mode();
  std::printf("device %s (%s), WMMA %s, decode mode %d\n", prop.name, prop.gcnArchName, wmma ? "on" : "off", mode);
  if (wmma && mode != 2) {
    std::printf("KT2N_AMD_WMMA_DECODE=%d is a test switch (prefill and decode on different arithmetic): skipped\n",
                mode);
    return 77;
  }
  test_attention();
  if (const char* p = model_path()) test_lm(p);
  else std::printf("LM part skipped: set KT2N_TEST_ASSETS or KT2N_REAL_ENGINE_FILE\n");
  if (g_fail) {
    std::printf("%d check(s) failed\n", g_fail);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
