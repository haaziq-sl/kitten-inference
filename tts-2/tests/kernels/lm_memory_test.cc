// Memory options of the native LM give the same bits as the default layout (needs the real engine file:
// KT2N_REAL_ENGINE_FILE=/path/to/model.kt2n, otherwise skipped).
//   - prefill with the host-gathered embedding (KT2N_PF_LOAD_HOST_EMBED) vs the full device table: K/V of every
//     slot row, last hidden state and logits compared byte for byte (voice prefix build, a burst over voice
//     prefixes of every attention path, a speaker row, ids outside the vocabulary);
//   - a prompt prefilled in consecutive calls vs one call (what max_prefill_tokens below the prompt does);
//   - kt2n::lm::Engine with host_embedding, and with a 64-token prefill limit, vs the default: every sampled token
//     and the logits it was drawn from (debug_logits), for rows over a voice and a row with a speaker vector.
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "kt2n/engine_file.h"
#include "kt2n/lm.h"
#include "kt2n/prefill.h"
#include "test_util.h"

namespace {

constexpr int kL = 28, kKvh = 8, kHd = 128, kH = 2048;

// KT2N_REAL_ENGINE_FILE, else the prepared model under KT2N_TEST_ASSETS (as the documented gpu test run sets).
const char* real_file() {
  static const std::string path = [] {
    const char* p = std::getenv("KT2N_REAL_ENGINE_FILE");
    if (p && *p) return std::string(p);
    const char* root = std::getenv("KT2N_TEST_ASSETS");
    return (root && *root) ? std::string(root) + "/native/model.kt2n" : std::string();
  }();
  return path.empty() ? nullptr : path.c_str();
}

#define REAL_OR_SKIP()                                                       \
  KT2N_GPU_OR_SKIP();                                                        \
  if (!real_file()) GTEST_SKIP() << "set KT2N_TEST_ASSETS or KT2N_REAL_ENGINE_FILE to run";

// Deterministic pseudo-random stream (splitmix64).
struct Rng {
  uint64_t s;
  uint64_t next() {
    uint64_t z = (s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
  double unit() { return double(next() >> 11) / double(1ull << 53); }
};

uint16_t to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// A prompt of text ids, special ids and audio ids (the mix a voice prefix has).
std::vector<int64_t> prompt_ids(Rng& r, int n) {
  std::vector<int64_t> v(n);
  for (int i = 0; i < n; ++i) {
    const uint64_t k = r.next() % 10;
    if (k < 6) v[i] = int64_t(r.next() % 151643);                     // text
    else if (k < 7) v[i] = 151643 + int64_t(r.next() % 32);            // specials
    else v[i] = 151675 + int64_t(r.next() % 6561);                     // audio codes
  }
  return v;
}

std::vector<uint16_t> speaker(Rng& r) {
  std::vector<uint16_t> s(kH);
  for (auto& x : s) x = to_bf16(float((r.unit() - 0.5) * 0.2));
  return s;
}

struct Dev {
  void* p = nullptr;
  size_t n = 0;
  explicit Dev(size_t bytes) : n(bytes) {
    EXPECT_EQ(cudaMalloc(&p, bytes), cudaSuccess);
    EXPECT_EQ(cudaMemset(p, 0, bytes), cudaSuccess);
  }
  ~Dev() { cudaFree(p); }
  std::vector<uint8_t> host() const {
    std::vector<uint8_t> h(n);
    EXPECT_EQ(cudaMemcpy(h.data(), p, n, cudaMemcpyDeviceToHost), cudaSuccess);
    return h;
  }
};

struct Model {
  kt2n_pf_model* m = nullptr;
  kt2n_pf_ctx* c = nullptr;
  explicit Model(unsigned flags) {
    EXPECT_EQ(kt2n_pf_model_load_ex(real_file(), 0, nullptr, flags, &m), 0) << kt2n_pf_last_error();
    if (m) {
      EXPECT_EQ(kt2n_pf_ctx_create(kt2n_pf_model_weights(m), 0, 16, 4096, 4096, &c), 0) << kt2n_pf_last_error();
    }
  }
  ~Model() {
    if (c) kt2n_pf_ctx_free(c);
    if (m) kt2n_pf_model_free(m);
  }
};

int64_t row_elems(int width) { return int64_t(2) * kL * kKvh * width * kHd; }

// Everything one model produces for the fixed inputs below.
struct Outputs {
  std::vector<uint8_t> voice_kv, slots, last_hidden, logits, spk_slot, spk_logits;
};

Outputs run_all(Model& md, unsigned flags) {
  Outputs o;
  Rng r{12345};
  const int p = 300, W = 1024;
  const std::vector<int64_t> prefix = prompt_ids(r, p);
  const std::vector<uint16_t> spk = speaker(r);
  Dev spk_dev(kH * 2);
  EXPECT_EQ(cudaMemcpy(spk_dev.p, spk.data(), kH * 2, cudaMemcpyHostToDevice), cudaSuccess);
  // voice prefix build: past 0, the speaker at position 0, K/V into a contiguous [2][L][8][p][128] entry
  Dev voice(size_t(row_elems(p)) * 2);
  {
    const int32_t nt = p, past = 0;
    EXPECT_EQ(kt2n_pf_run(md.c, 1, &nt, prefix.data(), &past, nullptr, spk_dev.p, voice.p, row_elems(p), p, nullptr,
                          nullptr, nullptr, 0, flags, nullptr),
              0)
        << kt2n_pf_last_error();
  }
  // burst over the voice prefix: suffix lengths that take the decode-rows, q8 and prefill attention paths; one id
  // outside the vocabulary (zero row)
  const std::vector<int32_t> nts = {7, 25, 180};
  std::vector<int64_t> ids;
  for (int n : nts) {
    auto v = prompt_ids(r, n);
    ids.insert(ids.end(), v.begin(), v.end());
  }
  ids[3] = 158241;  // == vocab: out of range
  const std::vector<int32_t> past(nts.size(), p);
  std::vector<const void*> pkv(nts.size(), voice.p);
  const std::vector<int32_t> dest = {2, 0, 1};
  Dev slots(size_t(row_elems(W)) * 2 * 3), hid(size_t(3) * kH * 2), lg(size_t(3) * KT2N_PF_AUDIO_N * 2);
  EXPECT_EQ(kt2n_pf_run(md.c, 3, nts.data(), ids.data(), past.data(), pkv.data(), nullptr, slots.p, row_elems(W), W,
                        dest.data(), hid.p, lg.p, KT2N_PF_AUDIO_N, flags, nullptr),
            0)
      << kt2n_pf_last_error();
  // a row without a prefix: speaker at position 0 of its suffix
  Dev slot1(size_t(row_elems(W)) * 2), lg1(size_t(KT2N_PF_AUDIO_N) * 2);
  {
    const auto v = prompt_ids(r, 40);
    const int32_t nt = 40, p0 = 0;
    EXPECT_EQ(kt2n_pf_run(md.c, 1, &nt, v.data(), &p0, nullptr, spk_dev.p, slot1.p, row_elems(W), W, nullptr, nullptr,
                          lg1.p, KT2N_PF_AUDIO_N, flags, nullptr),
              0)
        << kt2n_pf_last_error();
  }
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  o.voice_kv = voice.host();
  o.slots = slots.host();
  o.last_hidden = hid.host();
  o.logits = lg.host();
  o.spk_slot = slot1.host();
  o.spk_logits = lg1.host();
  return o;
}

void expect_same(const Outputs& a, const Outputs& b) {
  EXPECT_TRUE(a.voice_kv == b.voice_kv) << "voice prefix K/V differ";
  EXPECT_TRUE(a.slots == b.slots) << "slot K/V differ";
  EXPECT_TRUE(a.last_hidden == b.last_hidden) << "last hidden differs";
  EXPECT_TRUE(a.logits == b.logits) << "logits differ";
  EXPECT_TRUE(a.spk_slot == b.spk_slot) << "speaker row K/V differ";
  EXPECT_TRUE(a.spk_logits == b.spk_logits) << "speaker row logits differ";
}

TEST(LmMemory, PrefillHostEmbeddingSameBits) {
  REAL_OR_SKIP();
  Model full(0u), host(KT2N_PF_LOAD_HOST_EMBED);
  ASSERT_TRUE(full.c && host.c);
  const kt2n_pf_weights* w = kt2n_pf_model_weights(host.m);
  EXPECT_EQ(w->embed_row0, KT2N_PF_AUDIO_LOW);
  EXPECT_EQ(w->embed_rows, KT2N_PF_AUDIO_N);
  EXPECT_NE(w->host_embed, nullptr);
  EXPECT_EQ(kt2n_pf_model_bytes(full.m) - kt2n_pf_model_bytes(host.m),
            size_t(158241 - KT2N_PF_AUDIO_N) * kH * 2);  // 592 MiB, give or take alignment
  for (unsigned flags : {unsigned(KT2N_PF_FLAGS_FAST), 0u}) {
    const Outputs a = run_all(full, flags), b = run_all(host, flags);
    expect_same(a, b);
  }
}

// One row's prompt (prefix + suffix) prefilled in one call vs in consecutive calls of at most 64 ids.
TEST(LmMemory, PrefillInConsecutiveCallsSameBits) {
  REAL_OR_SKIP();
  Model md(KT2N_PF_LOAD_HOST_EMBED);
  ASSERT_TRUE(md.c);
  Rng r{777};
  const int n = 300, W = 512;
  const auto ids = prompt_ids(r, n);
  const auto spk = speaker(r);
  Dev spk_dev(kH * 2);
  ASSERT_EQ(cudaMemcpy(spk_dev.p, spk.data(), kH * 2, cudaMemcpyHostToDevice), cudaSuccess);
  Dev one(size_t(row_elems(W)) * 2), many(size_t(row_elems(W)) * 2);
  Dev lg_one(KT2N_PF_AUDIO_N * 2), lg_many(KT2N_PF_AUDIO_N * 2);
  const int32_t nt = n, p0 = 0;
  ASSERT_EQ(kt2n_pf_run(md.c, 1, &nt, ids.data(), &p0, nullptr, spk_dev.p, one.p, row_elems(W), W, nullptr, nullptr,
                        lg_one.p, KT2N_PF_AUDIO_N, KT2N_PF_FLAGS_FAST, nullptr),
            0)
      << kt2n_pf_last_error();
  for (int done = 0; done < n;) {
    const int32_t k = std::min(64, n - done), past = done;
    const bool last = done + k == n;
    ASSERT_EQ(kt2n_pf_run(md.c, 1, &k, ids.data() + done, &past, nullptr, done == 0 ? spk_dev.p : nullptr, many.p,
                          row_elems(W), W, nullptr, nullptr, last ? lg_many.p : nullptr, KT2N_PF_AUDIO_N,
                          KT2N_PF_FLAGS_FAST, nullptr),
              0)
        << kt2n_pf_last_error();
    done += k;
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_TRUE(one.host() == many.host()) << "K/V differ";
  EXPECT_TRUE(lg_one.host() == lg_many.host()) << "logits differ";
}

struct RowResult {
  std::vector<int32_t> tokens;
  std::vector<std::vector<uint16_t>> logits;
  int finish = 0;
};

// one_row: the low-memory fit's LM (max_rows 1, no scratch-slot K/V, prefill attention workspace for this GPU's
// one-row split shapes, 512-token prefill calls, host embedding left lazy).
std::map<uint64_t, RowResult> run_engine(bool host_embedding, int max_prefill_tokens, bool graphs, bool one_row = false,
                                         bool one_row_fit = false) {
  kt2n::lm::Config cfg;
  cfg.engine_path = real_file();
  cfg.max_rows = one_row ? 1 : 3;
  cfg.debug_logits = true;
  cfg.host_embedding = host_embedding;
  cfg.max_prefill_tokens = max_prefill_tokens;
  cfg.graphs = graphs;
  if (one_row_fit) {
    cfg.scratch_kv = false;
    cfg.prefill_fit_attn_ws = true;
    cfg.host_embedding_resident = false;
  }
  cfg.voice_store_bytes = size_t(64) << 20;
  std::string err;
  auto e = kt2n::lm::Engine::create(cfg, &err);
  EXPECT_TRUE(e) << err;
  std::map<uint64_t, RowResult> out;
  if (!e) return out;
  Rng r{4242};
  // the projected speaker from a raw speaker vector (spk_proj's input width from the engine file)
  const kt2n::EngineFile f = kt2n::EngineFile::open(real_file());
  const int spk_in = int(f.find("spk_proj.0.weight")->cols());
  std::vector<float> raw(spk_in);
  for (auto& x : raw) x = float(r.unit() - 0.5);
  std::vector<uint16_t> spk;
  EXPECT_EQ(e->project_speaker(raw.data(), spk_in, &spk, &err), 0) << err;
  const auto prefix = prompt_ids(r, 300);
  const int voice = e->build_voice(spk, prefix, &err);
  EXPECT_GE(voice, 0) << err;
  struct Sub {
    uint64_t tag;
    int voice;
    int n;
  };
  // One row at a time, so every decode step runs the 1-row bucket in every arm (bucket composition is part of
  // the decode arithmetic; batching is not what this test compares).
  int finished = 0;
  for (const Sub& s : {Sub{1, voice, 30}, Sub{2, voice, 180}, Sub{3, -1, 50}, Sub{4, voice, 3}}) {
    kt2n::lm::RowRequest row;
    row.tag = s.tag;
    row.voice = s.voice;
    row.suffix = prompt_ids(r, s.n);
    row.suffix.back() = kt2n::lm::kSpeechStart;
    if (s.voice < 0) row.speaker_bf16 = spk;
    row.seed = 1000 + s.tag;
    row.budget = 40;
    std::vector<kt2n::lm::TokenEvent> ev;
    // One slot (one_row): the previous row's slot comes back once the LM has read its last ring entry; until then
    // submit answers "engine full" (back-pressure), as the engine's admission retries.
    int rc = e->submit(row, &err);
    for (int guard = 0; rc != 0 && guard < 1000; ++guard) {
      ev.clear();
      if (e->pump(&ev, false, &err) < 0) break;
      rc = e->submit(row, &err);
    }
    EXPECT_EQ(rc, 0) << err;
    bool done = false;
    for (int guard = 0; !done && guard < 100000; ++guard) {
      ev.clear();
      if (e->pump(&ev, true, &err) < 0) {
        ADD_FAILURE() << err;
        break;
      }
      for (const auto& t : ev) {
        RowResult& rr = out[t.tag];
        rr.tokens.push_back(t.token);
        std::vector<uint16_t> lg;
        if (t.token >= 0 && e->debug_logits(t.tag, t.index, &lg)) rr.logits.push_back(lg);
        if (t.finish != kt2n::lm::Finish::kNone) {
          rr.finish = int(t.finish);
          done = true;
          ++finished;
        }
      }
    }
  }
  EXPECT_EQ(finished, 4);
  return out;
}

TEST(LmMemory, EngineHostEmbeddingAndShortPrefillSameTokens) {
  REAL_OR_SKIP();
  const auto ref = run_engine(false, 4096, true);
  ASSERT_EQ(ref.size(), 4u);
  for (const auto& [tag, rr] : ref) {
    EXPECT_GT(rr.tokens.size(), 1u) << tag;
    EXPECT_EQ(rr.logits.size(), rr.tokens.size()) << tag;
  }
  struct Arm {
    bool host;
    int tokens;
    bool graphs;
  };
  for (const Arm& a : {Arm{true, 4096, true}, Arm{true, 64, true}, Arm{false, 64, true}, Arm{true, 4096, false}}) {
    const auto got = run_engine(a.host, a.tokens, a.graphs);
    for (const auto& [tag, rr] : ref) {
      auto it = got.find(tag);
      ASSERT_NE(it, got.end()) << tag;
      EXPECT_EQ(it->second.tokens, rr.tokens) << "tag " << tag << " host " << a.host << " tokens " << a.tokens;
      EXPECT_TRUE(it->second.logits == rr.logits) << "tag " << tag << " host " << a.host << " tokens " << a.tokens;
      EXPECT_EQ(it->second.finish, rr.finish) << tag;
    }
  }
}

// The low-memory fit's one-row LM (no scratch-slot K/V, the fitted prefill attention workspace, 512-token prefill
// calls, lazy host embedding) vs a one-row LM with the default layout: every sampled token and its logits.
TEST(LmMemory, EngineOneRowFittedSameTokens) {
  REAL_OR_SKIP();
  const auto ref = run_engine(true, 4096, true, true, false);
  ASSERT_EQ(ref.size(), 4u);
  for (bool graphs : {true, false}) {
    const auto got = run_engine(true, 512, graphs, true, true);
    for (const auto& [tag, rr] : ref) {
      auto it = got.find(tag);
      ASSERT_NE(it, got.end()) << tag;
      EXPECT_EQ(it->second.tokens, rr.tokens) << "tag " << tag << " graphs " << graphs;
      EXPECT_TRUE(it->second.logits == rr.logits) << "tag " << tag << " graphs " << graphs;
      EXPECT_EQ(it->second.finish, rr.finish) << tag;
    }
  }
}

// scratch_kv = false and prefill_fit_attn_ws need a one-row LM.
TEST(LmMemory, OneRowOptionsRefusedWithMoreRows) {
  REAL_OR_SKIP();
  for (int which = 0; which < 2; ++which) {
    kt2n::lm::Config cfg;
    cfg.engine_path = real_file();
    cfg.max_rows = 2;
    cfg.host_embedding = true;
    cfg.voice_store_bytes = size_t(16) << 20;
    if (which == 0) cfg.scratch_kv = false;
    else cfg.prefill_fit_attn_ws = true;
    std::string err;
    auto e = kt2n::lm::Engine::create(cfg, &err);
    EXPECT_FALSE(e) << which;
    EXPECT_NE(err.find("max_rows == 1"), std::string::npos) << err;
  }
}

// KT2N_PF_CTX_FIT_ATTN_WS: one-row calls over every attention path (prefix build, decode rows, q8 tiles of 4 and 8
// queries, the prefill kernel) give the same K/V and logits as with the default workspace, and the context is
// not larger.
TEST(LmMemory, PrefillFitAttnWorkspaceSameBits) {
  REAL_OR_SKIP();
  kt2n_pf_model* m = nullptr;
  ASSERT_EQ(kt2n_pf_model_load_ex(real_file(), 0, nullptr, KT2N_PF_LOAD_HOST_EMBED, &m), 0) << kt2n_pf_last_error();
  kt2n_pf_ctx *full = nullptr, *fit = nullptr;
  ASSERT_EQ(kt2n_pf_ctx_create(kt2n_pf_model_weights(m), 0, 16, 512, 3008, &full), 0) << kt2n_pf_last_error();
  ASSERT_EQ(kt2n_pf_ctx_create2(kt2n_pf_model_weights(m), 0, 16, 512, 3008, KT2N_PF_CTX_FIT_ATTN_WS, &fit), 0)
      << kt2n_pf_last_error();
  EXPECT_LE(kt2n_pf_ctx_bytes(fit), kt2n_pf_ctx_bytes(full));
  EXPECT_EQ(kt2n_pf_ctx_workspace_bytes2(16, 512, 3008, KT2N_PF_CTX_FIT_ATTN_WS, 0), kt2n_pf_ctx_bytes(fit));
  auto run = [&](kt2n_pf_ctx* c) {
    Rng r{2024};
    const int p = 400, W = 3008;
    const auto prefix = prompt_ids(r, p);
    const auto spk = speaker(r);
    Dev spk_dev(kH * 2);
    EXPECT_EQ(cudaMemcpy(spk_dev.p, spk.data(), kH * 2, cudaMemcpyHostToDevice), cudaSuccess);
    Dev voice(size_t(row_elems(p)) * 2);
    const int32_t np = p, p0 = 0;
    EXPECT_EQ(kt2n_pf_run(c, 1, &np, prefix.data(), &p0, nullptr, spk_dev.p, voice.p, row_elems(p), p, nullptr, nullptr,
                          nullptr, 0, KT2N_PF_FLAGS_FAST, nullptr),
              0)
        << kt2n_pf_last_error();
    std::vector<std::vector<uint8_t>> out;
    out.push_back(voice.host());
    for (int n : {7, 25, 60, 100, 160, 300}) {
      const auto ids = prompt_ids(r, n);
      Dev slot(size_t(row_elems(W)) * 2), lg(size_t(KT2N_PF_AUDIO_N) * 2);
      const int32_t nt = n, past = p;
      const void* pkv = voice.p;
      EXPECT_EQ(kt2n_pf_run(c, 1, &nt, ids.data(), &past, &pkv, nullptr, slot.p, row_elems(W), W, nullptr, nullptr, lg.p,
                            KT2N_PF_AUDIO_N, KT2N_PF_FLAGS_FAST, nullptr),
                0)
          << kt2n_pf_last_error();
      EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      out.push_back(slot.host());
      out.push_back(lg.host());
    }
    return out;
  };
  const auto a = run(full), b = run(fit);
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) EXPECT_TRUE(a[i] == b[i]) << "output " << i << " differs";
  kt2n_pf_ctx_free(full);
  kt2n_pf_ctx_free(fit);
  kt2n_pf_model_free(m);
}

}  // namespace
