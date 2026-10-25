// The voice cache through the C ABI on the CPU stubs (no GPU, no model files): synthetic catalogues, LRU / NONE
// against ALL, a 47-voice round-robin through an LRU of 10, loads in flight at cancel / deadline / shutdown, load
// failures, saved voices, config validation. Stub audio is a deterministic function of the seeded tokens, so equal
// bytes across modes check the plumbing (the GPU byte-identity runs separately).
#include <gtest/gtest.h>
#if defined(_WIN32)
#include "kt2n/posix_compat.h"
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "kt2n/engine.h"
#include "voice_blob.h"

namespace {

using namespace kt2n::voice;

VoiceRecord synth(const std::string& id, int tp, int p, unsigned seed, const char* kind) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd;
  std::vector<int32_t> prefix(static_cast<size_t>(p)), tok(static_cast<size_t>(tp));
  for (auto& x : prefix) x = int32_t(rng() % 158241);
  for (auto& x : tok) x = int32_t(rng() % 6561);
  std::vector<float> raw(512), proj(2048), feat(static_cast<size_t>(tp) * 160), emb(192);
  for (auto* v : {&raw, &proj, &feat, &emb})
    for (auto& x : *v) x = nd(rng);
  double nn = 0;
  for (float x : raw) nn += double(x) * x;
  for (float& x : raw) x = float(x / std::sqrt(nn));
  return make_clone_record(id, kind, prefix, raw, proj, tok, feat, emb, "sig-" + id);
}

std::vector<uint8_t> blob_of(std::vector<VoiceRecord> v) {
  VoicesBlob b;
  b.top = default_top_meta(std::string(64, 'c'), std::string(64, 'd'));
  b.voices = std::move(v);
  return write_voices_blob(b);
}

// A catalogue of n built-in voices (prompt 200..249 tokens, prefixes 228..497 ids) written once per n and process
// (ctest runs each test in its own process, possibly at the same time as another test writing the same n).
std::string catalogue(int n, std::vector<VoiceRecord>* out = nullptr) {
  const std::string path =
      testing::TempDir() + "kt2n_vcache_" + std::to_string(long(::getpid())) + "_" + std::to_string(n) + ".kt2n";
  std::vector<VoiceRecord> cat;
  for (int i = 0; i < n; ++i) cat.push_back(synth("v" + std::to_string(i), 200 + i % 50, 228 + (i * 37) % 270, 100u + unsigned(i), "builtin"));
  const auto bytes = blob_of(cat);
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
  if (out) *out = std::move(cat);
  return path;
}

struct Done {
  int status = -1;
  std::string bytes;
};

class Eng {
 public:
  Eng(const std::string& voices, int mode, int max_voices, int max_streams, uint64_t max_bytes = 0) {
    kt2n_engine_config cfg{};
    cfg.struct_size = sizeof cfg;
    cfg.abi_version = KT2N_ABI_VERSION;
    cfg.max_streams = max_streams;
    cfg.voices_file = voices.c_str();
    cfg.voice_cache_mode = mode;
    cfg.voice_cache_max_voices = max_voices;
    cfg.voice_cache_max_bytes = max_bytes;
    kt2n_error err{};
    status = kt2n_engine_create(&cfg, &e, &err);
    error = err.message;
  }
  ~Eng() { kt2n_engine_destroy(e); }
  uint64_t submit(int voice, uint64_t seed, const std::string& text = "A short line for the voice cache test.",
                  int64_t deadline_ns = 0) {
    kt2n_request_params p{};
    p.struct_size = sizeof p;
    p.flags = KT2N_REQ_SEED;
    p.text = text.data();
    p.text_len = text.size();
    p.voice = kt2n_voice_id(voice);
    p.request_id = "t";
    p.seed = seed;
    p.speed = 1.0;
    p.format = KT2N_FORMAT_PCM;
    p.output = KT2N_OUTPUT_STREAM;
    p.deadline_ns = deadline_ns;
    kt2n_request_id id = 0;
    kt2n_error err{};
    EXPECT_EQ(kt2n_submit(e, &p, &id, &err), KT2N_OK) << err.message;
    return id;
  }
  // Polls until `n` more requests are done; the finished ones are released. DONE events beyond `n` that arrive in the
  // same poll count toward the next wait.
  void wait(size_t n, int timeout_s = 60) {
    const auto t0 = std::chrono::steady_clock::now();
    size_t got = std::min(n, spare);
    spare -= got;
    kt2n_event ev[64];
    while (got < n && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(timeout_s)) {
      int32_t k = 0;
      kt2n_poll(e, ev, 64, 100, &k);
      for (int i = 0; i < k; ++i) {
        Done& d = done[ev[i].id];
        if (ev[i].kind == KT2N_EVENT_AUDIO) d.bytes.append(reinterpret_cast<const char*>(ev[i].data), ev[i].data_len);
        if (ev[i].kind == KT2N_EVENT_DONE) {
          d.status = ev[i].status;
          ++got;
          finished.push_back(ev[i].id);
        }
      }
      kt2n_events_release(e, ev, k);
      for (uint64_t id : finished) kt2n_release(e, id);
      finished.clear();
    }
    if (got > n) spare += got - n;
    ASSERT_GE(got, n) << "timed out";
  }
  size_t spare = 0;
  std::string stats() {
    std::vector<char> b(1 << 20);
    size_t len = 0;
    kt2n_stats_json(e, b.data(), b.size(), &len);
    return std::string(b.data(), len ? len - 1 : 0);
  }
  long violations() {  // stub backends: a voice's GPU state used after it was given back, or given back while read
    const std::string s = stats();
    const size_t at = s.find("\"violations\":");
    return at == std::string::npos ? -1 : std::atol(s.c_str() + at + 13);
  }
  long stat(const std::string& key) {  // first "key":N inside the voice_cache object
    const std::string s = stats();
    const size_t vc = s.find("\"voice_cache\":");
    if (vc == std::string::npos) return -1;
    const size_t at = s.find("\"" + key + "\":", vc);
    if (at == std::string::npos) return -1;
    return std::atol(s.c_str() + at + key.size() + 3);
  }
  int status = -1;
  std::string error;
  kt2n_engine* e = nullptr;
  std::map<uint64_t, Done> done;
  std::vector<uint64_t> finished;
};

struct EnvGuard {
  std::vector<std::string> keys;
  void set(const char* k, const char* v) {
    setenv(k, v, 1);
    keys.push_back(k);
  }
  ~EnvGuard() {
    for (const auto& k : keys) unsetenv(k.c_str());
  }
};

TEST(VoiceCacheEngine, ConfigValidationAndDefaults) {
  const std::string path = catalogue(4);
  {
    Eng g(path, 7, 0, 2);
    EXPECT_EQ(g.status, KT2N_ERR_INVALID_ARGUMENT);
  }
  {
    Eng g(path, KT2N_VOICE_CACHE_LRU, -1, 2);
    EXPECT_EQ(g.status, KT2N_ERR_INVALID_ARGUMENT);
  }
  {
    Eng g(path, KT2N_VOICE_CACHE_LRU, 0, 2, 1 << 20);  // 1 MiB cannot hold a voice
    EXPECT_EQ(g.status, KT2N_ERR_TOO_SMALL) << g.error;
  }
  {
    Eng g(path, KT2N_VOICE_CACHE_ALL, 0, 2);  // ALL: the stats text has no voice_cache object
    ASSERT_EQ(g.status, KT2N_OK) << g.error;
    EXPECT_EQ(g.stats().find("voice_cache"), std::string::npos);
  }
  {
    // An older caller's struct_size (up to max_custom_voices) never reads the new block.
    kt2n_engine_config cfg{};
    cfg.struct_size = uint32_t(offsetof(kt2n_engine_config, max_custom_voices) + sizeof(int32_t));
    cfg.abi_version = KT2N_ABI_VERSION;
    cfg.max_streams = 2;
    cfg.voices_file = path.c_str();
    cfg.voice_cache_mode = KT2N_VOICE_CACHE_LRU;  // beyond struct_size: ignored
    kt2n_engine* e = nullptr;
    kt2n_error err{};
    ASSERT_EQ(kt2n_engine_create(&cfg, &e, &err), KT2N_OK) << err.message;
    std::vector<char> b(1 << 20);
    size_t len = 0;
    kt2n_stats_json(e, b.data(), b.size(), &len);
    EXPECT_EQ(std::string(b.data()).find("voice_cache"), std::string::npos);
    kt2n_engine_destroy(e);
  }
  {
    Eng g(path, KT2N_VOICE_CACHE_LRU, 0, 2);
    ASSERT_EQ(g.status, KT2N_OK) << g.error;
    EXPECT_EQ(g.stat("max_voices"), 16);  // LRU default
    EXPECT_EQ(g.stat("resident"), 0);     // nothing prepared at create
  }
  {
    Eng g(path, KT2N_VOICE_CACHE_NONE, 0, 3);
    ASSERT_EQ(g.status, KT2N_OK) << g.error;
    EXPECT_EQ(g.stat("max_voices"), 3);  // NONE default: max_streams
  }
}

// Same seeded request, same bytes: ALL, LRU miss, LRU hit, NONE.
TEST(VoiceCacheEngine, OutputsIdenticalAcrossModes) {
  const std::string path = catalogue(6);
  std::map<std::pair<int, int>, std::string> ref;
  {
    Eng g(path, KT2N_VOICE_CACHE_ALL, 0, 2);
    ASSERT_EQ(g.status, KT2N_OK) << g.error;
    for (int v = 1; v <= 6; ++v) {
      const uint64_t id = g.submit(v, 1234 + uint64_t(v));
      g.wait(1);
      ASSERT_EQ(g.done[id].status, KT2N_OK);
      ref[std::make_pair(v, 0)] = g.done[id].bytes;
    }
    // Stub audio depends on the voice: the same seed and text on another voice gives other bytes, so a row rendered
    // with another voice's state would not match below.
    const uint64_t x = g.submit(2, 1234 + 1);
    g.wait(1);
    ASSERT_EQ(g.done[x].status, KT2N_OK);
    EXPECT_NE(g.done[x].bytes, ref[std::make_pair(1, 0)]);
  }
  for (int mode : {int(KT2N_VOICE_CACHE_LRU), int(KT2N_VOICE_CACHE_NONE)}) {
    Eng g(path, mode, 2, 2);
    ASSERT_EQ(g.status, KT2N_OK) << g.error;
    for (int pass = 0; pass < 2; ++pass)  // LRU: pass 0 misses, pass 1 of the last voices hits
      for (int v = 1; v <= 6; ++v) {
        const uint64_t id = g.submit(v, 1234 + uint64_t(v));
        g.wait(1);
        ASSERT_EQ(g.done[id].status, KT2N_OK);
        EXPECT_EQ(g.done[id].bytes, ref[std::make_pair(v, 0)]) << "mode " << mode << " voice " << v << " pass " << pass;
      }
    const uint64_t id = g.submit(6, 1240);  // voice 6 was the last one used
    g.wait(1);
    EXPECT_EQ(g.done[id].bytes, ref[std::make_pair(6, 0)]);
    EXPECT_EQ(g.violations(), 0) << g.stats();
    if (mode == KT2N_VOICE_CACHE_LRU) {
      EXPECT_GE(g.stat("hits"), 1);
      EXPECT_GE(g.stat("evictions"), 1);
    } else {
      EXPECT_EQ(g.stat("hits"), 0);
      EXPECT_EQ(g.stat("resident"), 0);
    }
  }
}

// 47 voices round-robin through an LRU of 10 at concurrency 1, 2 and 4.
TEST(VoiceCacheEngine, RoundRobin47VoicesLru10) {
  const std::string path = catalogue(47);
  EnvGuard env;
  env.set("KT2N_STUB_LM_VOICE_BUILD_US", "2000");  // loads take 2 ms (asynchronous)
  for (int conc : {1, 2, 4}) {
    Eng g(path, KT2N_VOICE_CACHE_LRU, 10, conc);
    ASSERT_EQ(g.status, KT2N_OK) << g.error;
    const int total = 47 * 2;
    int next = 0;
    size_t finished = 0;
    std::vector<uint64_t> ids;
    while (finished < size_t(total)) {
      while (next < total && int(ids.size() - finished) < conc) ids.push_back(g.submit(1 + next++ % 47, 7 + uint64_t(next)));
      g.wait(1);
      ++finished;
      ASSERT_LE(g.stat("resident"), 10);
      kt2n_stats st{};
      st.struct_size = sizeof st;
      kt2n_stats_get(g.e, &st);
      ASSERT_LE(st.voices_loaded, 10);
    }
    for (uint64_t id : ids) EXPECT_EQ(g.done[id].status, KT2N_OK) << "concurrency " << conc;
    EXPECT_GE(g.stat("evictions"), total - 10) << g.stats();
    EXPECT_EQ(g.stat("load_failures"), 0);
    EXPECT_EQ(g.stat("pinned"), 0);
    EXPECT_GE(g.stat("loads"), total - 10);
    EXPECT_EQ(g.violations(), 0) << g.stats();
  }
}

// A request cancelled, or past its deadline, while its voice loads ends with that status; the next request on the
// voice is served; destroy with a load in flight waits for it.
TEST(VoiceCacheEngine, CancelDeadlineAndShutdownDuringLoad) {
  const std::string path = catalogue(4);
  EnvGuard env;
  env.set("KT2N_STUB_LM_VOICE_BUILD_US", "300000");  // 300 ms loads
  {
    Eng g(path, KT2N_VOICE_CACHE_LRU, 2, 2);
    ASSERT_EQ(g.status, KT2N_OK) << g.error;
    const uint64_t a = g.submit(1, 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(g.stat("loading"), 1);
    EXPECT_EQ(kt2n_cancel(g.e, a, KT2N_CANCEL_REQUEST), KT2N_OK);
    g.wait(1);
    EXPECT_EQ(g.done[a].status, KT2N_ERR_CANCELLED);
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    const int64_t now = int64_t(ts.tv_sec) * 1000000000ll + ts.tv_nsec;
    const uint64_t b = g.submit(2, 2, "A short line for the voice cache test.", now + 20000000);  // 20 ms
    g.wait(1);
    EXPECT_EQ(g.done[b].status, KT2N_ERR_DEADLINE_EXCEEDED);
    const uint64_t c = g.submit(1, 3);
    g.wait(1);
    EXPECT_EQ(g.done[c].status, KT2N_OK);
    EXPECT_EQ(g.stat("pinned"), 0);
  }
  {
    Eng g(path, KT2N_VOICE_CACHE_NONE, 2, 2);
    ASSERT_EQ(g.status, KT2N_OK) << g.error;
    g.submit(3, 1);
    g.submit(4, 2);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    EXPECT_GE(g.stat("loading"), 1);
    const auto t0 = std::chrono::steady_clock::now();
    kt2n_engine_destroy(g.e);  // destroy with loads in flight: waits for their queued builds (300 ms), no longer
    g.e = nullptr;
    EXPECT_LT(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), 3.0);
  }
}

// Destroy while a load waits for render pages that only an open row can give back (the row never closes once the
// conductor has stopped): the load is given back at once instead of polled until the 30 s drain timeout.
TEST(VoiceCacheEngine, DestroyDoesNotWaitForALoadBlockedOnPages) {
  const std::string path = catalogue(3);
  EnvGuard env;
  env.set("KT2N_TEST_STUB_RENDER_PREFIX_WAIT", "1");  // a prefix build waits while any row is open
  env.set("KT2N_STUB_LM_STEP_US", "20000");            // the first request's row stays open for seconds
  Eng g(path, KT2N_VOICE_CACHE_LRU, 3, 2);
  ASSERT_EQ(g.status, KT2N_OK) << g.error;
  g.submit(1, 1, "A much longer line, so that the first request keeps its row open for a good few seconds of decoding.");
  for (int i = 0; i < 200 && g.stat("resident") < 1; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  ASSERT_EQ(g.stat("resident"), 1);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));  // its row is open
  g.submit(2, 2);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  ASSERT_EQ(g.stat("loading"), 1) << g.stats();  // waiting for pages
  const auto t0 = std::chrono::steady_clock::now();
  kt2n_engine_destroy(g.e);
  g.e = nullptr;
  EXPECT_LT(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), 1.0);
}

// An engine failure while a voice loads: every request ends (service unavailable), new submits are refused, and
// destroy returns promptly.
TEST(VoiceCacheEngine, EngineFailureWithALoadInFlight) {
  const std::string path = catalogue(3);
  EnvGuard env;
  env.set("KT2N_STUB_LM_VOICE_BUILD_US", "400000");      // 400 ms loads
  env.set("KT2N_TEST_STUB_LM_FAIL_AFTER_STEPS", "60");   // the LM fails while the second voice loads
  env.set("KT2N_STUB_LM_STEP_US", "5000");
  for (int mode : {int(KT2N_VOICE_CACHE_LRU), int(KT2N_VOICE_CACHE_NONE)}) {
    Eng g(path, mode, 3, 2);
    ASSERT_EQ(g.status, KT2N_OK) << g.error;
    const uint64_t a = g.submit(1, 1, "A much longer line, so that the first request is still decoding when the LM fails.");
    for (int i = 0; i < 200 && g.stat("resident") < 1; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    const uint64_t b = g.submit(2, 2);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_EQ(g.stat("loading"), 1) << g.stats();
    g.wait(2, 30);
    EXPECT_NE(g.done[a].status, KT2N_OK);
    EXPECT_NE(g.done[b].status, KT2N_OK);
    const auto t0 = std::chrono::steady_clock::now();
    kt2n_engine_destroy(g.e);
    g.e = nullptr;
    EXPECT_LT(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), 3.0);
  }
}

// More distinct voices pinned at once than the cache holds: the extra requests wait (queued), all finish.
TEST(VoiceCacheEngine, PinnedVoicesAboveTheBudgetWaitAndFinish) {
  const std::string path = catalogue(8);
  EnvGuard env;
  env.set("KT2N_STUB_LM_VOICE_BUILD_US", "2000");
  for (int mode : {int(KT2N_VOICE_CACHE_LRU), int(KT2N_VOICE_CACHE_NONE)}) {
    Eng g(path, mode, 2, 6);
    ASSERT_EQ(g.status, KT2N_OK) << g.error;
    std::vector<uint64_t> ids;
    for (int r = 0; r < 3; ++r)
      for (int v = 1; v <= 6; ++v) ids.push_back(g.submit(v, uint64_t(100 * r + v)));
    for (size_t i = 0; i < ids.size(); ++i) {
      g.wait(1);
      ASSERT_LE(g.stat("resident"), 2);
    }
    for (uint64_t id : ids) EXPECT_EQ(g.done[id].status, KT2N_OK) << "mode " << mode;
    EXPECT_GE(g.stat("waits"), 1) << g.stats();
    EXPECT_EQ(g.stat("pinned"), 0);
    EXPECT_EQ(g.violations(), 0) << g.stats();
  }
}

// A realtime session keeps its voice between phrases (NONE would otherwise drop and prepare it for every phrase) and
// gives it back when it ends.
TEST(VoiceCacheEngine, SessionKeepsItsVoiceBetweenPhrases) {
  const std::string path = catalogue(3);
  for (int mode : {int(KT2N_VOICE_CACHE_LRU), int(KT2N_VOICE_CACHE_NONE)}) {
    Eng g(path, mode, 1, 1);
    ASSERT_EQ(g.status, KT2N_OK) << g.error;
    kt2n_session_params sp{};
    sp.struct_size = sizeof sp;
    sp.voice = 2;
    sp.request_id = "ws-cache";
    sp.speed = 1.0;
    kt2n_session_id sid = 0;
    kt2n_error err{};
    ASSERT_EQ(kt2n_session_open(g.e, &sp, &sid, &err), KT2N_OK) << err.message;
    const std::string a = "Hello from the socket. ", b = "And here is a second turn of text. ", c = "A third one.";
    ASSERT_EQ(kt2n_session_text_append(g.e, sid, a.data(), a.size(), &err), KT2N_OK);
    ASSERT_EQ(kt2n_session_commit(g.e, sid, &err), KT2N_OK);
    ASSERT_EQ(kt2n_session_text_append(g.e, sid, b.data(), b.size(), &err), KT2N_OK);
    ASSERT_EQ(kt2n_session_commit(g.e, sid, &err), KT2N_OK);
    ASSERT_EQ(kt2n_session_text_append(g.e, sid, c.data(), c.size(), &err), KT2N_OK);
    ASSERT_EQ(kt2n_session_done(g.e, sid, &err), KT2N_OK);
    int phrases = 0, status = -1;
    bool done = false;
    kt2n_event ev[32];
    const auto t0 = std::chrono::steady_clock::now();
    while (!done && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(30)) {
      int32_t k = 0;
      kt2n_poll(g.e, ev, 32, 200, &k);
      for (int i = 0; i < k; ++i) {
        if (ev[i].id != sid) continue;
        if (ev[i].kind == KT2N_EVENT_PHRASE_DONE) ++phrases;
        if (ev[i].kind == KT2N_EVENT_DONE) {
          done = true;
          status = ev[i].status;
        }
      }
      kt2n_events_release(g.e, ev, k);
    }
    EXPECT_EQ(status, KT2N_OK);
    EXPECT_GE(phrases, 2);
    EXPECT_EQ(g.stat("loads"), 1) << "mode " << mode << " " << g.stats();  // one preparation for every phrase
    EXPECT_EQ(kt2n_release(g.e, sid), KT2N_OK);
    // Another voice still gets the room (a kept voice is not pinned).
    const uint64_t q = g.submit(1, 5);
    g.wait(1);
    EXPECT_EQ(g.done[q].status, KT2N_OK);
    if (mode == KT2N_VOICE_CACHE_NONE) {
      for (int i = 0; i < 100 && g.stat("resident") > 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
      EXPECT_EQ(g.stat("resident"), 0) << g.stats();  // the session ended: nothing keeps a voice
    }
    EXPECT_EQ(g.violations(), 0) << g.stats();
  }
}

TEST(VoiceCacheEngine, LoadFailureIsRetriedThenReported) {
  const std::string path = catalogue(3);
  {
    EnvGuard env;
    env.set("KT2N_TEST_STUB_VOICE_FAIL_FIRST", "2");
    Eng g(path, KT2N_VOICE_CACHE_LRU, 2, 1);
    ASSERT_EQ(g.status, KT2N_OK) << g.error;
    const uint64_t a = g.submit(1, 1);
    g.wait(1);
    EXPECT_EQ(g.done[a].status, KT2N_OK);
    EXPECT_EQ(g.stat("load_failures"), 2);
  }
  {
    EnvGuard env;
    env.set("KT2N_TEST_STUB_VOICE_FAIL_FIRST", "3");
    Eng g(path, KT2N_VOICE_CACHE_LRU, 2, 1);
    ASSERT_EQ(g.status, KT2N_OK) << g.error;
    const uint64_t a = g.submit(1, 1);
    g.wait(1);
    EXPECT_EQ(g.done[a].status, KT2N_ERR_SPEECH_GENERATION_FAILED);
    const uint64_t b = g.submit(1, 2);  // every pin is gone: the next request loads it again
    g.wait(1);
    EXPECT_EQ(g.done[b].status, KT2N_OK);
  }
}

// kt2n_voice_prepare (LRU): the voice is resident before any request, which then hits it; a load that fails after its
// retries returns KT2N_ERR_SERVICE_UNAVAILABLE (and a later request loads the voice again); NONE: nothing to do; an
// unknown voice is refused.
TEST(VoiceCacheEngine, PrepareLoadsTheVoiceBeforeTheRequest) {
  const std::string path = catalogue(3);
  {
    Eng g(path, KT2N_VOICE_CACHE_LRU, 2, 1);
    ASSERT_EQ(g.status, KT2N_OK) << g.error;
    kt2n_error err{};
    EXPECT_EQ(kt2n_voice_prepare(g.e, 1, &err), KT2N_OK) << err.message;
    EXPECT_EQ(g.stat("resident"), 1) << g.stats();
    EXPECT_EQ(kt2n_voice_prepare(g.e, 1, &err), KT2N_OK) << err.message;  // already resident
    const uint64_t a = g.submit(1, 1);
    g.wait(1);
    EXPECT_EQ(g.done[a].status, KT2N_OK);
    EXPECT_EQ(g.stat("hits"), 1) << g.stats();
    EXPECT_EQ(g.stat("misses"), 0) << g.stats();
    EXPECT_EQ(kt2n_voice_prepare(g.e, 999, &err), KT2N_ERR_INVALID_VOICE);
    EXPECT_EQ(kt2n_voice_prepare(nullptr, 1, &err), KT2N_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(g.violations(), 0) << g.stats();
  }
  {
    EnvGuard env;
    env.set("KT2N_TEST_STUB_VOICE_FAIL_FIRST", "3");  // the first load fails on every attempt
    Eng g(path, KT2N_VOICE_CACHE_LRU, 2, 1);
    ASSERT_EQ(g.status, KT2N_OK) << g.error;
    kt2n_error err{};
    EXPECT_EQ(kt2n_voice_prepare(g.e, 1, &err), KT2N_ERR_SERVICE_UNAVAILABLE) << g.stats();
    EXPECT_EQ(g.stat("load_failures"), 3) << g.stats();
    EXPECT_EQ(g.stat("resident"), 0) << g.stats();
    const uint64_t b = g.submit(1, 2);  // a later request loads it again
    g.wait(1);
    EXPECT_EQ(g.done[b].status, KT2N_OK);
  }
  {
    EnvGuard env;
    env.set("KT2N_TEST_STUB_VOICE_FAIL_FIRST", "1");  // one failed attempt, then the retry loads it: not a failure
    Eng g(path, KT2N_VOICE_CACHE_LRU, 2, 1);
    ASSERT_EQ(g.status, KT2N_OK) << g.error;
    kt2n_error err{};
    EXPECT_EQ(kt2n_voice_prepare(g.e, 1, &err), KT2N_OK) << err.message << " " << g.stats();
    EXPECT_EQ(g.stat("load_failures"), 1) << g.stats();
    EXPECT_EQ(g.stat("resident"), 1) << g.stats();
  }
  {
    Eng g(path, KT2N_VOICE_CACHE_NONE, 0, 1);
    ASSERT_EQ(g.status, KT2N_OK) << g.error;
    EXPECT_EQ(kt2n_voice_prepare(g.e, 1, nullptr), KT2N_OK);
    EXPECT_EQ(g.stat("resident"), 0) << g.stats();
  }
}

// A destroy that starts while kt2n_voice_prepare waits for a slow load wakes it (KT2N_ERR_NOT_READY) and frees the
// engine only after it has returned (it counts as a poller).
TEST(VoiceCacheEngine, DestroyWakesAWaitingPrepare) {
  const std::string path = catalogue(3);
  EnvGuard env;
  env.set("KT2N_STUB_LM_VOICE_BUILD_US", "3000000");  // the voice's load takes 3 s
  Eng g(path, KT2N_VOICE_CACHE_LRU, 2, 1);
  ASSERT_EQ(g.status, KT2N_OK) << g.error;
  int32_t rc = -1;
  std::thread t([&] { rc = kt2n_voice_prepare(g.e, 1, nullptr); });
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  const auto t0 = std::chrono::steady_clock::now();
  kt2n_engine_destroy(g.e);
  g.e = nullptr;
  t.join();
  EXPECT_EQ(rc, KT2N_ERR_NOT_READY);
  EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::seconds(20));
}

// A saved voice artifact accepted by kt2n_voice_load against `cat` (prompt tp tokens, prefix p ids).
std::vector<uint8_t> saved_voice(const std::vector<VoiceRecord>& cat, int tp = 210, int p = 300) {
  VoiceRecord r = synth("voice_s", tp, p, 20, "saved");
  r.raw_speaker = cat[1].raw_speaker;
  r.raw_speaker[0] += 1e-3f;
  CloneIdentityInputs in{std::string(64, 'a'), std::string(64, 'b'), std::string(64, 'c'), "v1"};
  return blob_of({make_clone_record("voice_s", "saved", r.lm_prefix, r.raw_speaker, r.projected_speaker,
                                    r.prompt_token, r.prompt_feat, r.embedding, "sig", clone_identity(in))});
}

// Load, submit and evict a saved voice before the conductor has registered it: the request still finishes with the
// voice ("requests holding it finish with it"), as in ALL mode.
TEST(VoiceCacheEngine, LoadSubmitEvictAtOnceFinishesTheRequest) {
  std::vector<VoiceRecord> cat;
  const std::string path = catalogue(2, &cat);
  const auto saved = saved_voice(cat);
  EnvGuard env;
  env.set("KT2N_STUB_LM_STEP_US", "3000");  // the conductor spends its turns in blocking LM pumps
  for (int mode : {int(KT2N_VOICE_CACHE_LRU), int(KT2N_VOICE_CACHE_NONE)}) {
    Eng g(path, mode, 2, 4);
    ASSERT_EQ(g.status, KT2N_OK) << g.error;
    const uint64_t busy = g.submit(1, 1, "A longer line that keeps the conductor busy with decoding steps for a while.");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::vector<uint64_t> ids;
    for (int c = 0; c < 4; ++c) {
      kt2n_voice_id v = 0;
      kt2n_error err{};
      ASSERT_EQ(kt2n_voice_load(g.e, saved.data(), saved.size(), &v, &err), KT2N_OK) << err.message;
      ids.push_back(g.submit(int(v), 42 + uint64_t(c)));
      EXPECT_EQ(kt2n_voice_evict(g.e, v), KT2N_OK);
    }
    g.wait(ids.size() + 1, 30);
    EXPECT_EQ(g.done[busy].status, KT2N_OK);
    for (uint64_t id : ids) EXPECT_EQ(g.done[id].status, KT2N_OK) << "mode " << mode;
    for (int i = 0; i < 100 && g.stat("retired") < 4; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_EQ(g.stat("retired"), 4) << g.stats();
    EXPECT_EQ(g.violations(), 0) << g.stats();
  }
}

// A submit racing kt2n_voice_evict either fails at once (the voice is gone) or ends with a DONE event (served, or failed
// because the voice was given back before the request took its hold); it never waits forever.
TEST(VoiceCacheEngine, SubmitRacingEvictNeverHangs) {
  std::vector<VoiceRecord> cat;
  const std::string path = catalogue(2, &cat);
  const auto saved = saved_voice(cat);
  for (int mode : {int(KT2N_VOICE_CACHE_LRU), int(KT2N_VOICE_CACHE_NONE)}) {
    Eng g(path, mode, 2, 4);
    ASSERT_EQ(g.status, KT2N_OK) << g.error;
    size_t submitted = 0;
    for (int c = 0; c < 40; ++c) {
      kt2n_voice_id v = 0;
      kt2n_error err{};
      ASSERT_EQ(kt2n_voice_load(g.e, saved.data(), saved.size(), &v, &err), KT2N_OK) << err.message;
      std::this_thread::sleep_for(std::chrono::microseconds(200 * (c % 8)));  // registered or not yet
      std::thread evictor([&] {
        std::this_thread::sleep_for(std::chrono::microseconds(20 * (c % 5)));
        kt2n_voice_evict(g.e, v);
      });
      const std::string text = "A short line for the voice cache test.";
      kt2n_request_params p{};
      p.struct_size = sizeof p;
      p.flags = KT2N_REQ_SEED;
      p.text = text.data();
      p.text_len = text.size();
      p.voice = v;
      p.request_id = "race";
      p.seed = uint64_t(c);
      p.speed = 1.0;
      p.format = KT2N_FORMAT_PCM;
      p.output = KT2N_OUTPUT_STREAM;
      kt2n_request_id id = 0;
      kt2n_error serr{};
      const int32_t rc = kt2n_submit(g.e, &p, &id, &serr);
      evictor.join();
      if (rc == KT2N_OK)
        ++submitted;
      else
        EXPECT_EQ(rc, KT2N_ERR_INVALID_VOICE);
    }
    g.wait(submitted, 60);  // every submitted request ends
    EXPECT_EQ(g.stat("pinned"), 0);
    EXPECT_EQ(g.violations(), 0) << g.stats();
  }
}

// In LRU / NONE a saved voice larger than the whole cache budget is refused when it is loaded.
TEST(VoiceCacheEngine, LoadRefusesAVoiceLargerThanTheBudget) {
  std::vector<VoiceRecord> cat;
  const std::string path = catalogue(2, &cat);
  const auto big = saved_voice(cat, 240, 1500);  // 1,500 prefix ids: more than a one-voice budget (1,024)
  Eng g(path, KT2N_VOICE_CACHE_LRU, 1, 1);
  ASSERT_EQ(g.status, KT2N_OK) << g.error;
  kt2n_voice_id v = 0;
  kt2n_error err{};
  EXPECT_EQ(kt2n_voice_load(g.e, big.data(), big.size(), &v, &err), KT2N_ERR_CUSTOM_VOICE_SERVICE_ERROR);
  EXPECT_NE(std::string(err.message).find("voice cache budget"), std::string::npos) << err.message;
  const auto ok = saved_voice(cat);
  EXPECT_EQ(kt2n_voice_load(g.e, ok.data(), ok.size(), &v, &err), KT2N_OK) << err.message;
}

// Saved voices go through the same cache; evicting the handle retires its GPU state.
TEST(VoiceCacheEngine, SavedVoicesGoThroughTheCache) {
  std::vector<VoiceRecord> cat;
  const std::string path = catalogue(2, &cat);
  const auto saved = saved_voice(cat);
  for (int mode : {int(KT2N_VOICE_CACHE_LRU), int(KT2N_VOICE_CACHE_NONE)}) {
    Eng g(path, mode, 1, 2);
    ASSERT_EQ(g.status, KT2N_OK) << g.error;
    for (int c = 0; c < 3; ++c) {
      kt2n_voice_id v = 0;
      kt2n_error err{};
      ASSERT_EQ(kt2n_voice_load(g.e, saved.data(), saved.size(), &v, &err), KT2N_OK) << err.message;
      const uint64_t a = g.submit(int(v), 42);
      const uint64_t b = g.submit(1, 43);  // a built-in competes for the single cache entry
      g.wait(2);
      EXPECT_EQ(g.done[a].status, KT2N_OK);
      EXPECT_EQ(g.done[b].status, KT2N_OK);
      EXPECT_EQ(kt2n_voice_evict(g.e, v), KT2N_OK);
    }
    for (int i = 0; i < 100 && g.stat("retired") < 3; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_EQ(g.stat("retired"), 3) << g.stats();
    EXPECT_LE(g.stat("resident"), 1);
    EXPECT_EQ(g.violations(), 0) << g.stats();
  }
}

}  // namespace
