// Engine runtime tests on the CPU stubs (no GPU, no model files): the C ABI contract, the
// conductor's turn loop, admission back-pressure, cancel, stream and complete outputs.
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "kt2n/engine.h"
#if defined(_WIN32)
#include "kt2n/posix_compat.h"  // setenv / unsetenv
#endif
#include "voice_blob.h"  // the voices.kt2n writer: a synthetic catalogue + saved voice for the voice-release test

namespace {

struct Result {
  int status = -1;
  std::string bytes;
  int64_t samples = 0;
  int audio_events = 0;
  uint64_t last_seq = 0;
  bool seq_ok = true;
  bool final_seen = false;
  std::vector<int32_t> opening_skipped;  // kt2n_event.opening_skipped_tokens per AUDIO event
};

class Eng {
 public:
  explicit Eng(int max_streams = 10) {
    kt2n_engine_config cfg{};
    cfg.struct_size = sizeof cfg;
    cfg.abi_version = KT2N_ABI_VERSION;
    cfg.flags = KT2N_CFG_TIMELINE_ALLOWED;
    cfg.max_streams = max_streams;
    kt2n_error err{};
    const int status = kt2n_engine_create(&cfg, &e, &err);
    if (status != KT2N_OK) throw std::runtime_error(std::string("engine fixture initialization failed: ") + err.message);
  }
  ~Eng() { kt2n_engine_destroy(e); }
  uint64_t submit(const std::string& text, int output = KT2N_OUTPUT_STREAM, int format = KT2N_FORMAT_PCM) {
    kt2n_request_params p{};
    p.struct_size = sizeof p;
    p.flags = KT2N_REQ_SEED;
    p.text = text.data();
    p.text_len = text.size();
    p.voice = 1;
    p.request_id = "t";
    p.seed = 42;
    p.speed = 1.0;
    p.format = format;
    p.output = output;
    kt2n_request_id id = 0;
    kt2n_error err{};
    EXPECT_EQ(kt2n_submit(e, &p, &id, &err), KT2N_OK) << err.message;
    return id;
  }
  std::map<uint64_t, Result> run(size_t n, int timeout_s = 30) {
    std::map<uint64_t, Result> out;
    const auto t0 = std::chrono::steady_clock::now();
    size_t done = 0;
    kt2n_event ev[32];
    while (done < n && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(timeout_s)) {
      int32_t k = 0;
      kt2n_poll(e, ev, 32, 200, &k);
      for (int i = 0; i < k; ++i) {
        Result& r = out[ev[i].id];
        if (ev[i].seq != 0 && ev[i].seq != r.last_seq + 1) r.seq_ok = false;
        r.last_seq = ev[i].seq;
        if (ev[i].kind == KT2N_EVENT_AUDIO) {
          r.bytes.append(reinterpret_cast<const char*>(ev[i].data), ev[i].data_len);
          r.samples += ev[i].samples;
          r.audio_events++;
          r.final_seen |= ev[i].is_final_chunk != 0;
          r.opening_skipped.push_back(ev[i].opening_skipped_tokens);
        } else if (ev[i].kind == KT2N_EVENT_DONE) {
          r.status = ev[i].status;
          ++done;
        }
      }
      kt2n_events_release(e, ev, k);
    }
    return out;
  }
  kt2n_engine* e = nullptr;
};

TEST(Runtime, ErrorTableMatchesReferenceCodes) {
  EXPECT_STREQ(kt2n_error_lookup(KT2N_ERR_INVALID_INPUT)->code, "invalid_input");
  EXPECT_EQ(kt2n_error_lookup(KT2N_ERR_CUSTOM_VOICE_CREATION_BUSY)->http_status, 429);
  EXPECT_STREQ(kt2n_error_lookup(KT2N_ERR_DEADLINE_EXCEEDED)->code, "speech_generation_failed");
  EXPECT_STREQ(kt2n_error_lookup(9999)->code, "internal_error");
}

TEST(Runtime, ValidationIsSynchronous) {
  Eng g;
  kt2n_request_params p{};
  p.struct_size = sizeof p;
  p.text = "   ";
  p.text_len = 3;
  p.voice = 1;
  p.speed = 1.0;
  kt2n_request_id id;
  kt2n_error err{};
  EXPECT_EQ(kt2n_submit(g.e, &p, &id, &err), KT2N_ERR_INVALID_INPUT);
  EXPECT_STREQ(err.param, "input");
  p.text = "Hello.";
  p.text_len = 6;
  p.speed = 5.0;
  EXPECT_EQ(kt2n_submit(g.e, &p, &id, &err), KT2N_ERR_INVALID_REQUEST);
  EXPECT_STREQ(err.param, "speed");
  p.speed = 1.0;
  p.sampler.present = KT2N_SAMPLER_HAS_TEMPERATURE;
  p.sampler.temperature = 3.0;
  EXPECT_EQ(kt2n_submit(g.e, &p, &id, &err), KT2N_ERR_INVALID_REQUEST);
  EXPECT_STREQ(err.param, "temperature");
  EXPECT_STREQ(err.message, "temperature must be a number between 0.3 and 2.0.");
  p.sampler.present = 0;
  p.voice = 99;
  EXPECT_EQ(kt2n_submit(g.e, &p, &id, &err), KT2N_ERR_INVALID_VOICE);
}

TEST(Runtime, RejectsInvalidCustomVoiceCapacity) {
  kt2n_engine_config cfg{};
  cfg.struct_size = sizeof cfg;
  cfg.abi_version = KT2N_ABI_VERSION;
  kt2n_engine* e = nullptr;
  kt2n_error err{};
  for (int capacity : {-1, 4097}) {
    cfg.max_custom_voices = capacity;
    EXPECT_EQ(kt2n_engine_create(&cfg, &e, &err), KT2N_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(e, nullptr);
    EXPECT_STREQ(err.param, "max_custom_voices");
  }
  // An older config ends before the appended capacity field. Its trailing
  // bytes must not be interpreted as a capacity override.
  cfg.struct_size = offsetof(kt2n_engine_config, max_custom_voices);
  EXPECT_EQ(kt2n_engine_create(&cfg, &e, &err), KT2N_OK) << err.message;
  kt2n_engine_destroy(e);
}

// KT2N_MEMORY_PROFILE_AUTO with the CPU stubs (no decoder file, so the low profile's fit does not run): the probe stops
// at once and the default-profile create is used; stats say so. Other profiles keep their stats text unchanged. The
// low profile (and so AUTO's first attempt) measures the GPU's free memory, so this needs a GPU: KT2N_TEST_GPU=1.
TEST(Runtime, GpuAutoMemoryProfileChoosesDefaultWhenItFits) {
  if (!std::getenv("KT2N_TEST_GPU")) GTEST_SKIP() << "set KT2N_TEST_GPU=1 (the low profile measures the GPU)";
  for (int profile : {int(KT2N_MEMORY_PROFILE_AUTO), int(KT2N_MEMORY_PROFILE_DEFAULT), int(KT2N_MEMORY_PROFILE_LOW)}) {
    kt2n_engine_config cfg{};
    cfg.struct_size = sizeof cfg;
    cfg.abi_version = KT2N_ABI_VERSION;
    cfg.max_streams = 2;
    cfg.memory_profile = profile;
    kt2n_engine* e = nullptr;
    kt2n_error err{};
    ASSERT_EQ(kt2n_engine_create(&cfg, &e, &err), KT2N_OK) << profile << ": " << err.message;
    std::string stats(1 << 16, '\0');
    size_t len = 0;
    ASSERT_EQ(kt2n_stats_json(e, stats.data(), stats.size(), &len), KT2N_OK);
    stats.resize(len ? len - 1 : 0);
    const bool marked = stats.find("\"memory_profile_auto\":\"default\"") != std::string::npos;
    EXPECT_EQ(marked, profile == KT2N_MEMORY_PROFILE_AUTO) << stats;
    kt2n_engine_destroy(e);
  }
  kt2n_engine_config bad{};
  bad.struct_size = sizeof bad;
  bad.abi_version = KT2N_ABI_VERSION;
  bad.memory_profile = 3;
  kt2n_engine* e = nullptr;
  kt2n_error err{};
  EXPECT_EQ(kt2n_engine_create(&bad, &e, &err), KT2N_ERR_INVALID_ARGUMENT);
  EXPECT_STREQ(err.param, "memory_profile");
}

TEST(Runtime, StreamOneRequest) {
  Eng g;
  const uint64_t id = g.submit("Hello there. This is a small test of the native conductor.");
  auto r = g.run(1);
  ASSERT_EQ(r[id].status, KT2N_OK);
  EXPECT_GT(r[id].samples, 24000 / 2);
  EXPECT_EQ(int64_t(r[id].bytes.size()), r[id].samples * 2);
  EXPECT_TRUE(r[id].seq_ok);
  EXPECT_TRUE(r[id].final_seen);
  EXPECT_GT(r[id].audio_events, 1);  // streamed in pieces (hold, then commits)
  EXPECT_EQ(kt2n_release(g.e, id), KT2N_OK);
}

// Opening skip count: every AUDIO event carries the reference
// opening-skipped token count, final at the FIRST audio event; build_info names the capability.
// The CPU stub LM opens every row with 3 sub-gate silence ids (backend.h opening_quiet_tokens), then
// non-set ids close the opening: 3 skipped (policy_test OpeningSkip* gives the same count). A complete
// file is a complete-mode request, which never skips (row skip_armed), so its count stays the reference default 0.
TEST(Runtime, AudioEventsCarryOpeningSkipCount) {
  EXPECT_NE(std::strstr(kt2n_build_info(), KT2N_CAP_AUDIO_OPENING_SKIP), nullptr) << kt2n_build_info();
  Eng g;
  const uint64_t a = g.submit("Hello there. This is a small test of the native conductor.");
  const uint64_t b = g.submit("A second request, so two rows share the conductor.", KT2N_OUTPUT_COMPLETE,
                              KT2N_FORMAT_WAV);
  auto r = g.run(2);
  for (uint64_t id : {a, b}) {
    ASSERT_EQ(r[id].status, KT2N_OK);
    ASSERT_FALSE(r[id].opening_skipped.empty());
    EXPECT_EQ(r[id].opening_skipped.front(), id == a ? 3 : 0) << (id == a ? "stream" : "complete (DEMO)");
    for (int32_t v : r[id].opening_skipped) EXPECT_EQ(v, r[id].opening_skipped.front());
    EXPECT_EQ(kt2n_release(g.e, id), KT2N_OK);
  }
  EXPECT_GT(r[a].opening_skipped.size(), 1u);  // streamed: the count rides on every chunk
}

TEST(Runtime, CompleteFileWav) {
  Eng g;
  const uint64_t id = g.submit("A complete file, please. Two sentences long.", KT2N_OUTPUT_COMPLETE, KT2N_FORMAT_WAV);
  auto r = g.run(1);
  ASSERT_EQ(r[id].status, KT2N_OK);
  ASSERT_GT(r[id].bytes.size(), 44u);
  EXPECT_EQ(r[id].bytes.substr(0, 4), "RIFF");
  EXPECT_EQ(r[id].audio_events, 1);
  EXPECT_EQ(int64_t(r[id].bytes.size()) - 44, r[id].samples * 2);
}

TEST(Runtime, ConcurrencyAboveSeatsQueuesAndFinishesAll) {
  Eng g(4);
  std::vector<uint64_t> ids;
  for (int i = 0; i < 12; ++i) ids.push_back(g.submit("Request number " + std::to_string(i) + " is here, and it talks."));
  auto r = g.run(ids.size(), 60);
  for (uint64_t id : ids) {
    EXPECT_EQ(r[id].status, KT2N_OK) << id;
    EXPECT_GT(r[id].samples, 0) << id;
  }
  kt2n_stats s{};
  s.struct_size = sizeof s;
  kt2n_stats_get(g.e, &s);
  EXPECT_EQ(s.requests_done, 12u);
  EXPECT_GT(s.renders, 12u);
  for (uint64_t id : ids) EXPECT_EQ(kt2n_release(g.e, id), KT2N_OK);
  // The conductor frees released requests on its next loop.
  std::string js(2048, '\0');
  size_t len = 0;
  bool freed = false;
  for (int i = 0; i < 100 && !freed; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    kt2n_stats_json(g.e, js.data(), js.size(), &len);
    freed = js.find("\"requests_held\":0,") != std::string::npos;
  }
  EXPECT_TRUE(freed) << js;
}

TEST(Runtime, LongTextManySegments) {
  Eng g;
  std::string text;
  for (int i = 0; i < 12; ++i) text += "This sentence number " + std::to_string(i) + " is part of a longer text that keeps going. ";
  const uint64_t id = g.submit(text);
  auto r = g.run(1, 60);
  ASSERT_EQ(r[id].status, KT2N_OK);
  EXPECT_GT(r[id].samples, 24000);
}

TEST(Runtime, CancelGivesTerminalEvent) {
  Eng g;
  const uint64_t id = g.submit("This one will be cancelled before it finishes speaking all of these words, sadly.");
  EXPECT_EQ(kt2n_cancel(g.e, id, KT2N_CANCEL_REQUEST), KT2N_OK);
  auto r = g.run(1);
  EXPECT_EQ(r[id].status, KT2N_ERR_CANCELLED);
  EXPECT_EQ(kt2n_cancel(g.e, 12345, KT2N_CANCEL_REQUEST), KT2N_ERR_INVALID_HANDLE);
}

}  // namespace

namespace {

TEST(Runtime, WsSessionTwoCommittedTurns) {
  Eng g;
  kt2n_session_params sp{};
  sp.struct_size = sizeof sp;
  sp.voice = 1;
  sp.request_id = "ws-1";
  sp.speed = 1.0;
  kt2n_session_id sid = 0;
  kt2n_error err{};
  ASSERT_EQ(kt2n_session_open(g.e, &sp, &sid, &err), KT2N_OK) << err.message;
  const std::string a = "Hello from the socket. ";
  const std::string b = "And here is a second turn of text.";
  ASSERT_EQ(kt2n_session_text_append(g.e, sid, a.data(), a.size(), &err), KT2N_OK);
  ASSERT_EQ(kt2n_session_commit(g.e, sid, &err), KT2N_OK);
  ASSERT_EQ(kt2n_session_text_append(g.e, sid, b.data(), b.size(), &err), KT2N_OK);
  ASSERT_EQ(kt2n_session_done(g.e, sid, &err), KT2N_OK);
  int phrases = 0, done = 0, status = -1;
  int64_t samples = 0;
  uint64_t last = 0;
  bool seq_ok = true;
  kt2n_event ev[32];
  const auto t0 = std::chrono::steady_clock::now();
  while (!done && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(30)) {
    int32_t k = 0;
    kt2n_poll(g.e, ev, 32, 200, &k);
    for (int i = 0; i < k; ++i) {
      if (ev[i].id != sid) continue;
      if (ev[i].seq != 0 && ev[i].seq != last + 1) seq_ok = false;
      last = ev[i].seq;
      if (ev[i].kind == KT2N_EVENT_AUDIO) samples += ev[i].samples;
      if (ev[i].kind == KT2N_EVENT_PHRASE_DONE) ++phrases;
      if (ev[i].kind == KT2N_EVENT_DONE) {
        done = 1;
        status = ev[i].status;
      }
    }
    kt2n_events_release(g.e, ev, k);
  }
  EXPECT_EQ(status, KT2N_OK);
  EXPECT_GE(phrases, 2);
  EXPECT_GT(samples, 0);
  EXPECT_TRUE(seq_ok);
  EXPECT_EQ(kt2n_release(g.e, sid), KT2N_OK);
}

TEST(Runtime, SessionFloat32AndFormatValidation) {
  Eng g;
  kt2n_session_params p{};
  p.struct_size = sizeof p;
  p.voice = 1;
  p.speed = 1.0;
  p.format = KT2N_FORMAT_WAV;
  kt2n_session_id id = 0;
  kt2n_error err{};
  EXPECT_EQ(kt2n_session_open(g.e, &p, &id, &err), KT2N_ERR_INVALID_RESPONSE_FORMAT);
  p.format = KT2N_FORMAT_F32;
  p.struct_size = sizeof p - 1;
  EXPECT_EQ(kt2n_session_open(g.e, &p, &id, &err), KT2N_ERR_INVALID_ARGUMENT);
  p.struct_size = sizeof p;
  ASSERT_EQ(kt2n_session_open(g.e, &p, &id, &err), KT2N_OK) << err.message;
  const std::string text = "An incremental phrase in float32 format.";
  ASSERT_EQ(kt2n_session_text_append(g.e, id, text.data(), text.size(), &err), KT2N_OK);
  ASSERT_EQ(kt2n_session_done(g.e, id, &err), KT2N_OK);
  auto result = g.run(1);
  ASSERT_EQ(result[id].status, KT2N_OK);
  ASSERT_GT(result[id].samples, 0);
  EXPECT_EQ(int64_t(result[id].bytes.size()), result[id].samples * 4);
}

}  // namespace

namespace {

TEST(Runtime, OversizedMessageEndsSessionLikeReference) {
  Eng g;
  kt2n_session_params sp{};
  sp.struct_size = sizeof sp;
  sp.voice = 1;
  sp.speed = 1.0;
  kt2n_session_id sid = 0;
  kt2n_error err{};
  ASSERT_EQ(kt2n_session_open(g.e, &sp, &sid, &err), KT2N_OK);
  const std::string big(8193, 'a');
  EXPECT_EQ(kt2n_session_text_append(g.e, sid, big.data(), big.size(), &err), KT2N_ERR_SPEECH_GENERATION_FAILED);
  EXPECT_STREQ(err.code, "speech_generation_failed");
  int status = -1;
  kt2n_event ev[16];
  const auto t0 = std::chrono::steady_clock::now();
  while (status < 0 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5)) {
    int32_t k = 0;
    kt2n_poll(g.e, ev, 16, 100, &k);
    for (int i = 0; i < k; ++i)
      if (ev[i].id == sid && ev[i].kind == KT2N_EVENT_DONE) status = ev[i].status;
    kt2n_events_release(g.e, ev, k);
  }
  EXPECT_EQ(status, KT2N_ERR_SPEECH_GENERATION_FAILED);
}

}  // namespace

namespace {

TEST(Runtime, WsInputAfterDoneEndsSessionInputAlreadyClosed) {
  Eng g;
  kt2n_session_params sp{};
  sp.struct_size = sizeof sp;
  sp.voice = 1;
  sp.speed = 1.0;
  kt2n_session_id sid = 0;
  kt2n_error err{};
  ASSERT_EQ(kt2n_session_open(g.e, &sp, &sid, &err), KT2N_OK);
  const std::string a = "Some words.";
  ASSERT_EQ(kt2n_session_text_append(g.e, sid, a.data(), a.size(), &err), KT2N_OK);
  ASSERT_EQ(kt2n_session_done(g.e, sid, &err), KT2N_OK);
  EXPECT_EQ(kt2n_session_text_append(g.e, sid, a.data(), a.size(), &err), KT2N_ERR_INPUT_ALREADY_CLOSED);
  EXPECT_STREQ(err.code, "input_already_closed");
  int status = -1;
  kt2n_event ev[16];
  const auto t0 = std::chrono::steady_clock::now();
  while (status < 0 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10)) {
    int32_t k = 0;
    kt2n_poll(g.e, ev, 16, 100, &k);
    for (int i = 0; i < k; ++i)
      if (ev[i].id == sid && ev[i].kind == KT2N_EVENT_DONE) status = ev[i].status;
    kt2n_events_release(g.e, ev, k);
  }
  EXPECT_EQ(status, KT2N_ERR_INPUT_ALREADY_CLOSED);
}

}  // namespace

namespace {

// Random mix: stream / complete / WAV, cancels at random times, releases at DONE; every request must end with
// exactly one terminal event and the engine must hold nothing afterwards.
TEST(Runtime, StressRandomCancelsAndOutputs) {
  Eng g(6);
  uint64_t rng = 12345;
  auto next = [&] {
    rng = rng * 6364136223846793005ull + 1442695040888963407ull;
    return uint32_t(rng >> 33);
  };
  std::map<uint64_t, int> done;
  std::vector<uint64_t> ids;
  const char* texts[] = {"Short.", "A medium sentence that has a few words in it.",
                         "A longer text. It has two sentences, and it keeps going for a little while longer."};
  for (int i = 0; i < 60; ++i) {
    const int kind = int(next() % 3);
    ids.push_back(g.submit(texts[next() % 3], kind == 2 ? KT2N_OUTPUT_COMPLETE : KT2N_OUTPUT_STREAM,
                           kind == 1 ? KT2N_FORMAT_WAV : KT2N_FORMAT_PCM));
    if (next() % 4 == 0) kt2n_cancel(g.e, ids.back(), KT2N_CANCEL_REQUEST);
  }
  kt2n_event ev[64];
  const auto t0 = std::chrono::steady_clock::now();
  while (done.size() < ids.size() && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(60)) {
    int32_t k = 0;
    kt2n_poll(g.e, ev, 64, 100, &k);
    for (int i = 0; i < k; ++i) {
      if (ev[i].kind == KT2N_EVENT_DONE) {
        done[ev[i].id] += 1;
        EXPECT_TRUE(ev[i].status == KT2N_OK || ev[i].status == KT2N_ERR_CANCELLED) << ev[i].status;
      }
    }
    kt2n_events_release(g.e, ev, k);
    // Cancel a random live request now and then.
    if (next() % 8 == 0) kt2n_cancel(g.e, ids[next() % ids.size()], KT2N_CANCEL_REQUEST);
  }
  ASSERT_EQ(done.size(), ids.size());
  for (auto& [id, n] : done) EXPECT_EQ(n, 1) << id;
  for (uint64_t id : ids) EXPECT_EQ(kt2n_release(g.e, id), KT2N_OK);
  std::string js(4096, '\0');
  size_t len = 0;
  bool freed = false;
  for (int i = 0; i < 200 && !freed; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    kt2n_stats_json(g.e, js.data(), js.size(), &len);
    freed = js.find("\"requests_held\":0,") != std::string::npos && js.find("\"active_requests\":0,") != std::string::npos;
  }
  EXPECT_TRUE(freed) << js;
}

}  // namespace

namespace {

// Render failures (stub: every 7th call fails): every request still ends with exactly one terminal event
// (OK or speech_generation_failed) and nothing stays held.
TEST(Runtime, RenderFailuresNeverStick) {
  setenv("KT2N_TEST_STUB_RENDER_FAIL_EVERY", "7", 1);
  Eng g(4);
  unsetenv("KT2N_TEST_STUB_RENDER_FAIL_EVERY");
  std::vector<uint64_t> ids;
  for (int i = 0; i < 20; ++i) ids.push_back(g.submit("Render failures must never leave a request hanging, ever."));
  auto r = g.run(ids.size(), 60);
  int failed = 0;
  for (uint64_t id : ids) {
    EXPECT_TRUE(r[id].status == KT2N_OK || r[id].status == KT2N_ERR_SPEECH_GENERATION_FAILED) << r[id].status;
    failed += r[id].status == KT2N_ERR_SPEECH_GENERATION_FAILED;
    EXPECT_EQ(kt2n_release(g.e, id), KT2N_OK);
  }
  EXPECT_GT(failed, 0);
  std::string js(4096, '\0');
  size_t len = 0;
  bool freed = false;
  for (int i = 0; i < 300 && !freed; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    kt2n_stats_json(g.e, js.data(), js.size(), &len);
    freed = js.find("\"requests_held\":0,") != std::string::npos;
  }
  EXPECT_TRUE(freed) << js;
}

// Native admission back-pressure: a renderer that cannot open more rows (pool pages / slots) makes jobs wait in
// admission; every request still completes, none fails, and the renderer never held more rows than it allowed.
TEST(Runtime, RenderBackpressureWaitsNeverFails) {
  setenv("KT2N_TEST_STUB_RENDER_MAX_ROWS", "3", 1);
  Eng g(8);
  unsetenv("KT2N_TEST_STUB_RENDER_MAX_ROWS");
  std::vector<uint64_t> ids;
  for (int i = 0; i < 12; ++i) ids.push_back(g.submit("Back-pressure waits in admission, and it never fails a request."));
  auto r = g.run(ids.size(), 60);
  for (uint64_t id : ids) {
    EXPECT_EQ(r[id].status, KT2N_OK);
    EXPECT_EQ(kt2n_release(g.e, id), KT2N_OK);
  }
  std::string js(8192, '\0');
  size_t len = 0;
  ASSERT_EQ(kt2n_stats_json(g.e, js.data(), js.size(), &len), KT2N_OK);
  js.resize(len ? len - 1 : 0);
  EXPECT_NE(js.find("\"open_rows_max\":3"), std::string::npos) << js;
  EXPECT_EQ(js.find("\"render_backpressure\":0,"), std::string::npos) << js;
}

// More seats than rows: admitted requests beyond the rows wait in admission
// (they are never opened and closed again at the renderer) and all of them finish.
TEST(Runtime, SeatsAboveRowsWaitInAdmission) {
  setenv("KT2N_ENGINE_ROWS", "3", 1);
  Eng g(64);
  unsetenv("KT2N_ENGINE_ROWS");
  std::vector<uint64_t> ids;
  for (int i = 0; i < 40; ++i) ids.push_back(g.submit("Seat " + std::to_string(i) + " waits for a row, then talks."));
  auto r = g.run(ids.size(), 60);
  for (uint64_t id : ids) {
    EXPECT_EQ(r[id].status, KT2N_OK) << id;
    EXPECT_GT(r[id].samples, 0) << id;
    EXPECT_EQ(kt2n_release(g.e, id), KT2N_OK);
  }
  std::string js(8192, '\0');
  size_t len = 0;
  ASSERT_EQ(kt2n_stats_json(g.e, js.data(), js.size(), &len), KT2N_OK);
  js.resize(len ? len - 1 : 0);
  EXPECT_NE(js.find("\"max_streams\":64,\"engine_rows\":3,\"engine_rows_why\":\"KT2N_ENGINE_ROWS\""),
            std::string::npos) << js;
  // LM rows <= the cap. Render rows also hold rows whose LM finished until their final render (the real
  // renderer has cap + 4 slots for that), so they may exceed the cap by a few, never by the queue.
  auto num = [&js](const std::string& key) {
    const size_t at = js.find("\"" + key + "\":");
    return at == std::string::npos ? -1 : std::atoi(js.c_str() + at + key.size() + 3);
  };
  EXPECT_GE(num("turn_max_rows"), 1) << js;
  EXPECT_LE(num("turn_max_rows"), 3) << js;
  EXPECT_LE(num("open_rows_max"), 3 + 4) << js;
  EXPECT_NE(js.find("\"render_backpressure\":0,"), std::string::npos) << js;
  // Without the knob, CPU stubs (no device memory) keep the 60-row limit; the release's seats are unchanged.
  Eng big(64);
  Eng ten(10);
  for (Eng* x : {&big, &ten}) {
    std::string j2(8192, '\0');
    ASSERT_EQ(kt2n_stats_json(x->e, j2.data(), j2.size(), &len), KT2N_OK);
    j2.resize(len ? len - 1 : 0);
    EXPECT_NE(j2.find(x == &big ? "\"engine_rows\":60,\"engine_rows_why\":\"row limit\""
                                : "\"engine_rows\":10,\"engine_rows_why\":\"seats\""),
              std::string::npos) << j2;
  }
}

// Capacity mode admission order, end to end on the CPU stubs: closed loop as the capacity graph (every short
// request that finishes is replaced by a new one, 40 stay queued while one long request plays); everything
// finishes with the order off and on, and the stats say which ran.
TEST(Runtime, CapacityOrderAdmitsContinuingRequestsFirst) {
  std::string longtext;
  for (int i = 0; i < 6; ++i) longtext += "This sentence number " + std::to_string(i) + " is part of a longer text that keeps going. ";
  int before[2] = {-1, -1};  // short requests finished before the long one, order off / on
  for (int order = 0; order < 2; ++order) {
    setenv("KT2N_ENGINE_ROWS", "2", 1);
    setenv("KT2N_CAPACITY_ORDER", order ? "1" : "0", 1);
    Eng g(64);
    unsetenv("KT2N_ENGINE_ROWS");
    unsetenv("KT2N_CAPACITY_ORDER");
    const uint64_t lid = g.submit(longtext);
    int submitted = 0, outstanding = 0, shorts_done = 0;
    auto add_short = [&] {
      g.submit("Short request " + std::to_string(submitted++) + " here.");
      ++outstanding;
    };
    for (int i = 0; i < 40; ++i) add_short();
    bool long_done = false;
    kt2n_event ev[32];
    const auto t0 = std::chrono::steady_clock::now();
    while ((!long_done || outstanding > 0) && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(120)) {
      int32_t k = 0;
      kt2n_poll(g.e, ev, 32, 200, &k);
      std::vector<uint64_t> finished;
      for (int i = 0; i < k; ++i)
        if (ev[i].kind == KT2N_EVENT_DONE) {
          EXPECT_EQ(ev[i].status, KT2N_OK);
          finished.push_back(ev[i].id);
        }
      kt2n_events_release(g.e, ev, k);
      for (uint64_t id : finished) {
        EXPECT_EQ(kt2n_release(g.e, id), KT2N_OK);
        if (id == lid) {
          long_done = true;
          before[order] = shorts_done;
          continue;
        }
        ++shorts_done;
        --outstanding;
        if (!long_done && submitted < 400) add_short();
      }
    }
    EXPECT_TRUE(long_done);
    EXPECT_EQ(outstanding, 0);
    std::string js(8192, '\0');
    size_t len = 0;
    ASSERT_EQ(kt2n_stats_json(g.e, js.data(), js.size(), &len), KT2N_OK);
    EXPECT_NE(js.find(order ? "\"capacity_order\":true" : "\"capacity_order\":false"), std::string::npos);
  }
  // Printed, not asserted: on the CPU stubs the long request's next segment is often not ready when a row frees
  // (the reference admits it only after its previous segment is acknowledged), so the order rarely gets to act and
  // the counts depend on machine load (16 / 16 and 54 / 16 in runs on the test machine). The ordering itself is checked
  // exactly in policy_test (Admission.CapacityModeOffersContinuingRequestsFirst).
  std::printf("[capacity order] short requests finished before the long one: %d (round-robin) vs %d "
              "(continuing first)\n", before[0], before[1]);
}

// Regression test: a streamed segment of 466+ characters (tagged budget
// >= 2,047 tokens) was refused by the render service's open_row ("budget exceeds the noise table") after segment 0,
// on every voice. Needs a GPU (the RenderService with its stand-in stages: the real open_row, no weights, ~3 GB):
// run with KT2N_TEST_GPU=1; skipped otherwise (the CPU stub renderer has no noise table).
TEST(Runtime, GpuLongStreamedSegmentOpensItsRenderRow) {
  if (!std::getenv("KT2N_TEST_GPU")) GTEST_SKIP() << "set KT2N_TEST_GPU=1 on a GPU slot";
  const std::string text =
      "Good evening, and welcome to the last session of this year's community lecture series. Tonight we are talking "
      "about bridges: why they stand up, why a few of them fall down, and what engineers have learned from both. We "
      "will start with the oldest stone arches in the region, some of which have carried traffic for more than four "
      "hundred years without a single repair. Then we will look at the iron bridges of the railway age, which were "
      "lighter and cheaper, and occasionally far less forgiving. Finally, we will spend some time on the suspension "
      "bridge across the estuary, which many of you drive over every morning. I promise you that it is perfectly "
      "safe, and by the end of the evening I hope you will understand exactly why. Please keep your questions until "
      "the end; there will be plenty of time, and the coffee will still be warm.";
  kt2n_engine_config cfg{};
  cfg.struct_size = sizeof cfg;
  cfg.abi_version = KT2N_ABI_VERSION;
  cfg.flags = KT2N_CFG_TIMELINE_ALLOWED;
  cfg.max_streams = 2;
  cfg.s3gen_file = "l7-standin";
  setenv("KT2N_L7_POOL_PAGES", "64", 1);
  kt2n_engine* e = nullptr;
  kt2n_error err{};
  ASSERT_EQ(kt2n_engine_create(&cfg, &e, &err), KT2N_OK) << err.message;
  unsetenv("KT2N_L7_POOL_PAGES");
  std::map<int, int> status;
  for (int output : {int(KT2N_OUTPUT_STREAM), int(KT2N_OUTPUT_COMPLETE)}) {
    kt2n_request_params p{};
    p.struct_size = sizeof p;
    p.flags = KT2N_REQ_SEED;
    p.text = text.data();
    p.text_len = text.size();
    p.voice = 1;
    p.request_id = "c02";
    p.seed = 42;
    p.speed = 1.0;
    p.format = KT2N_FORMAT_PCM;
    p.output = output;
    kt2n_request_id id = 0;
    ASSERT_EQ(kt2n_submit(e, &p, &id, &err), KT2N_OK) << err.message;
    kt2n_event ev[32];
    bool done = false;
    const auto t0 = std::chrono::steady_clock::now();
    while (!done && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(60)) {
      int32_t k = 0;
      kt2n_poll(e, ev, 32, 200, &k);
      for (int i = 0; i < k; ++i)
        if (ev[i].kind == KT2N_EVENT_DONE && ev[i].id == id) {
          done = true;
          status[output] = ev[i].status;
          EXPECT_EQ(ev[i].status, KT2N_OK) << (output == KT2N_OUTPUT_STREAM ? "stream: " : "complete: ")
                                           << (ev[i].error ? ev[i].error->message : "");
        }
      kt2n_events_release(e, ev, k);
    }
    EXPECT_TRUE(done);
    kt2n_release(e, id);
  }
  kt2n_engine_destroy(e);
}

}  // namespace

namespace {

// The caller boundary: F32 frames (held, not levelled) and control_json policy fields.
TEST(Runtime, F32BoundaryAndControlJson) {
  Eng g;
  kt2n_request_params p{};
  p.struct_size = sizeof p;
  const std::string text = "A phrase for the adapter boundary.";
  p.text = text.data();
  p.text_len = text.size();
  p.voice = 1;
  p.speed = 1.0;
  p.format = KT2N_FORMAT_F32;
  p.control_json = "{\"startup_hold_seconds\":0,\"phrase\":true,\"emotion\":false,\"initial_audio_buffer_seconds\":0}";
  kt2n_request_id id = 0;
  kt2n_error err{};
  ASSERT_EQ(kt2n_submit(g.e, &p, &id, &err), KT2N_OK) << err.message;
  auto r = g.run(1);
  ASSERT_EQ(r[id].status, KT2N_OK);
  EXPECT_EQ(int64_t(r[id].bytes.size()), r[id].samples * 4);
  p.control_json = "{\"delivery_override\":{\"x\":1}}";
  EXPECT_EQ(kt2n_submit(g.e, &p, &id, &err), KT2N_ERR_UNSUPPORTED);
  EXPECT_STREQ(err.param, "delivery_override");
  p.control_json = "{\"demo_check\":true}";
  EXPECT_EQ(kt2n_submit(g.e, &p, &id, &err), KT2N_ERR_UNSUPPORTED);
}

// Evicted custom voices give their LM store space and render table row back once their last request is freed
// (release_voices on the conductor; LM release_voice, render remove_voice). CPU stubs: the stats count the releases.
TEST(Runtime, EvictedSavedVoicesAreReleased) {
  using namespace kt2n::voice;
  auto synth = [](const std::string& id, int tp, int p, unsigned seed, const char* kind) {
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
  };
  auto blob_of = [](std::vector<VoiceRecord> v) {
    VoicesBlob b;
    b.top = default_top_meta(std::string(64, 'c'), std::string(64, 'd'));
    b.voices = std::move(v);
    return write_voices_blob(b);
  };
  std::vector<VoiceRecord> cat = {synth("v0", 7, 30, 10, "builtin"), synth("v1", 7, 30, 11, "builtin")};
  const auto cat_bytes = blob_of(cat);
  const std::string path = testing::TempDir() + "kt2n_release_test_voices.kt2n";
  {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(cat_bytes.data()), std::streamsize(cat_bytes.size()));
  }
  VoiceRecord r = synth("voice_s", 9, 40, 20, "saved");
  r.raw_speaker = cat[1].raw_speaker;
  r.raw_speaker[0] += 1e-3f;
  CloneIdentityInputs in{std::string(64, 'a'), std::string(64, 'b'), std::string(64, 'c'), "v1"};
  const auto saved = blob_of({make_clone_record("voice_s", "saved", r.lm_prefix, r.raw_speaker, r.projected_speaker,
                                                r.prompt_token, r.prompt_feat, r.embedding, "sig",
                                                clone_identity(in))});
  kt2n_engine_config cfg{};
  cfg.struct_size = sizeof cfg;
  cfg.abi_version = KT2N_ABI_VERSION;
  cfg.max_streams = 4;
  cfg.voices_file = path.c_str();
  Eng g(0);  // the helper's engine is unused; this test needs a catalogue
  kt2n_engine* e = nullptr;
  kt2n_error err{};
  ASSERT_EQ(kt2n_engine_create(&cfg, &e, &err), KT2N_OK) << err.message;
  std::swap(g.e, e);
  kt2n_engine_destroy(e);
  const int cycles = 5;
  for (int c = 0; c < cycles; ++c) {
    kt2n_voice_id v = 0;
    ASSERT_EQ(kt2n_voice_load(g.e, saved.data(), saved.size(), &v, &err), KT2N_OK) << err.message << " " << err.code;
    kt2n_request_params p{};
    p.struct_size = sizeof p;
    p.flags = KT2N_REQ_SEED;
    const std::string text = "A short line on a saved voice.";
    p.text = text.data();
    p.text_len = text.size();
    p.voice = v;
    p.request_id = "t";
    p.seed = 42;
    p.speed = 1.0;
    p.format = KT2N_FORMAT_PCM;
    p.output = KT2N_OUTPUT_STREAM;
    kt2n_request_id id = 0;
    ASSERT_EQ(kt2n_submit(g.e, &p, &id, &err), KT2N_OK) << err.message;
    auto res = g.run(1);
    EXPECT_EQ(res[id].status, KT2N_OK);
    EXPECT_EQ(kt2n_voice_evict(g.e, v), KT2N_OK);
    EXPECT_EQ(kt2n_release(g.e, id), KT2N_OK);
  }
  std::string stats;
  for (int i = 0; i < 100; ++i) {  // the conductor frees released requests, then sweeps evicted voices
    std::vector<char> b(1 << 20);
    size_t len = 0;
    kt2n_stats_json(g.e, b.data(), b.size(), &len);
    stats.assign(b.data(), len);
    if (stats.find("\"render_voices_released\":" + std::to_string(cycles)) != std::string::npos) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  EXPECT_NE(stats.find("\"lm_voices_released\":" + std::to_string(cycles)), std::string::npos) << stats.substr(0, 400);
  EXPECT_NE(stats.find("\"render_voices_released\":" + std::to_string(cycles)), std::string::npos);
  EXPECT_NE(stats.find("\"voice_store_full\":0"), std::string::npos);
  std::remove(path.c_str());
}

}  // namespace

#if defined(KT2N_HIP)
#include <hip/hip_runtime_api.h>
#define cudaMemGetInfo(free_bytes, total_bytes) static_cast<int>(hipMemGetInfo((free_bytes), (total_bytes)))
#else
// cudart is linked statically into the engine library and this test; declared here so the test needs no CUDA headers.
extern "C" int cudaMemGetInfo(size_t* free_bytes, size_t* total_bytes);
#endif
#if defined(KT2N_HIP) && !defined(_WIN32)
#include <unistd.h>  // access
#endif
// The HIP runtime keeps some device memory from a process's first memset on (measured: 32 MiB on an RX 9060 XT, 90 MiB
// on an RX 7900 XTX under WSL2; once, never returned, a second memset costs nothing). A test that checks that a create
// kept no device memory takes that cost first, so it is not counted as the engine's.
static void warm_gpu_runtime() {
#if defined(KT2N_HIP)
  void* p = nullptr;
  if (hipMalloc(&p, size_t(1) << 20) == hipSuccess) {
    (void)hipMemset(p, 0, size_t(1) << 20);
    (void)hipDeviceSynchronize();
    (void)hipFree(p);
  }
#endif
}

// Under WSL2 the free memory a process reads is the whole GPU's, the Windows desktop's included, and for up to about
// half a second after another process releases a lot of GPU memory (in ctest order, the test before the refusal test
// loads the model) it falls back by ~100 MiB with no call from this process, presumably Windows making memory resident
// again that it had moved out of VRAM (measured on an RX 7900 XT: 104.75 MiB, 0.2-0.4 s after the exit, also when the
// process before was a plain allocator, and with no test process running at all). A test that checks that a create
// kept no device memory first waits until free memory has been steady for a second (at most 15 s), so that only this
// process moves it while the test measures. Without /dev/dxg nothing waits.
static void settle_gpu_free_memory() {
#if defined(KT2N_HIP) && !defined(_WIN32)
  if (access("/dev/dxg", F_OK) != 0) return;
  using steady = std::chrono::steady_clock;
  const auto start = steady::now();
  auto steady_since = start;
  size_t ref = 0, now = 0, total = 0;
  if (hipMemGetInfo(&ref, &total) != hipSuccess) return;
  while (steady::now() - steady_since < std::chrono::seconds(1) && steady::now() - start < std::chrono::seconds(15)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (hipMemGetInfo(&now, &total) != hipSuccess) return;
    if ((now > ref ? now - ref : ref - now) >= (size_t(1) << 20)) {
      ref = now;
      steady_since = steady::now();
    }
  }
#endif
}

// Attempts a "kept no device memory" measurement gets: under WSL2 another process (a browser, the desktop) can take
// memory during the measurement, so a drop is measured again, up to 3 times. A create that keeps memory every time
// fails every attempt; one that keeps memory only on its first call would pass here, and fails on CUDA and on native
// Linux, which measure once.
static int free_memory_attempts() {
#if defined(KT2N_HIP) && !defined(_WIN32)
  if (access("/dev/dxg", F_OK) == 0) return 3;
#endif
  return 1;
}

namespace {


// Simulated free memory (KT2N_LOW_FIT_FREE_MB) for a small GPU whose free-memory floor is floor_mb. The sizes in these
// tests were set on a GPU without WSL2. On AMD under WSL2 the low-memory fit keeps at least 1 GiB free (engine.cc:
// driving free memory below about 0.5 GB hangs some PCs), so the simulated value grows by the difference, and the
// display driver's extra (measured by a probe, varying by tens of MiB between PCs and runs) is set to 0 with the
// engine's test knob: the fit then has exactly the room it has there, on every PC.
// The knob is unset again with the simulated size (clear_sim_free), so a later create on real free memory measures
// the driver's extra as users' creates do.
std::string sim_free_mb(long long mb, long long floor_mb) {
#if defined(KT2N_HIP) && !defined(_WIN32)
  if (access("/dev/dxg", F_OK) == 0 && floor_mb < 1024) {
    mb += 1024 - floor_mb;
    setenv("KT2N_FIT_DRIVER_EXTRA_MB", "0", 1);
  }
#else
  (void)floor_mb;
#endif
  return std::to_string(mb);
}

// Simulated free memory used as given, for a create that must be refused: no WSL2 raise (it would only shrink the
// refusal margin of a check that does not use the 1 GiB floor), but on AMD under WSL2 the driver's extra is still 0,
// so that a fitted check does not run the probe, which fills the GPU down to the floor.
void set_sim_free_exact(const char* mb) {
  setenv("KT2N_LOW_FIT_FREE_MB", mb, 1);
#if defined(KT2N_HIP) && !defined(_WIN32)
  if (access("/dev/dxg", F_OK) == 0) setenv("KT2N_FIT_DRIVER_EXTRA_MB", "0", 1);
#endif
}

void clear_sim_free() {
  unsetenv("KT2N_LOW_FIT_FREE_MB");
  unsetenv("KT2N_FIT_DRIVER_EXTRA_MB");
}

// Low-memory profile: create prewarms every render graph the profile can use, so serving a streamed and a
// stream:false request captures no graph, and (on a GPU no other process uses) device free memory is the same before
// and after serving. Needs a prepared bundle and a GPU: KT2N_TEST_GPU=1 KT2N_TEST_ASSETS=<bundle root with native/>;
// set KT2N_TEST_EXCLUSIVE_GPU=1 to also compare device free memory. Skipped otherwise.
TEST(Runtime, GpuLowMemoryProfileServesWithoutDeviceAllocations) {
  const char* root = std::getenv("KT2N_TEST_ASSETS");
  if (!std::getenv("KT2N_TEST_GPU") || !root || !*root) GTEST_SKIP() << "set KT2N_TEST_GPU=1 and KT2N_TEST_ASSETS";
  const std::string n = std::string(root) + "/native/";
  const std::string lm = n + "model.kt2n", s3 = n + "s3gen.kt2n", vo = n + "voices.kt2n", tok = n + "tokenizer.json",
                    txt = n + "text";
  kt2n_engine_config cfg{};
  cfg.struct_size = sizeof cfg;
  cfg.abi_version = KT2N_ABI_VERSION;
  cfg.flags = KT2N_CFG_INFERENCE_ONLY;
  cfg.max_streams = 1;
  cfg.lm_file = lm.c_str();
  cfg.s3gen_file = s3.c_str();
  cfg.voices_file = vo.c_str();
  cfg.tokenizer_json = tok.c_str();
  cfg.text_data_dir = txt.c_str();
  cfg.memory_profile = KT2N_MEMORY_PROFILE_LOW;
  cfg.min_free_device_bytes = 64ull << 20;
  kt2n_engine* e = nullptr;
  kt2n_error err{};
  ASSERT_EQ(kt2n_engine_create(&cfg, &e, &err), KT2N_OK) << err.message;
  kt2n_voice_id voice = 0;
  ASSERT_EQ(kt2n_voice_lookup(e, "Bruno", &voice, &err), KT2N_OK) << err.message;
  size_t free0 = 0, free1 = 0, total = 0;
  ASSERT_EQ(cudaMemGetInfo(&free0, &total), 0);
  const std::string text =
      "The old lighthouse keeper climbed the spiral stairs every evening, just as the sun slipped behind the hills. "
      "He trimmed the wick, polished the great lens, and watched the beam sweep across the dark water.";
  for (int output : {int(KT2N_OUTPUT_STREAM), int(KT2N_OUTPUT_COMPLETE)}) {
    kt2n_request_params p{};
    p.struct_size = sizeof p;
    p.flags = KT2N_REQ_SEED;
    p.seed = 1234;
    p.text = text.data();
    p.text_len = text.size();
    p.voice = voice;
    p.speed = 1.0;
    p.format = KT2N_FORMAT_PCM;
    p.output = output;
    kt2n_request_id id = 0;
    ASSERT_EQ(kt2n_submit(e, &p, &id, &err), KT2N_OK) << err.message;
    bool done = false;
    size_t bytes = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (!done && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(120)) {
      kt2n_event ev[32];
      int32_t k = 0;
      kt2n_poll(e, ev, 32, 200, &k);
      for (int i = 0; i < k; ++i) {
        if (ev[i].id != id) continue;
        if (ev[i].kind == KT2N_EVENT_AUDIO) bytes += ev[i].data_len;
        if (ev[i].kind == KT2N_EVENT_DONE) {
          done = true;
          EXPECT_EQ(ev[i].status, KT2N_OK) << (ev[i].error ? ev[i].error->message : "");
        }
      }
      kt2n_events_release(e, ev, k);
    }
    ASSERT_TRUE(done);
    EXPECT_GT(bytes, 0u);
    kt2n_release(e, id);
  }
  ASSERT_EQ(cudaMemGetInfo(&free1, &total), 0);
  size_t need = 0;
  kt2n_stats_json(e, nullptr, 0, &need);
  std::string stats(need + 4096, '\0');
  size_t got = 0;
  ASSERT_EQ(kt2n_stats_json(e, stats.data(), stats.size(), &got), KT2N_OK);
  stats.resize(got ? got - 1 : 0);
  EXPECT_NE(stats.find("\"live_captures\":0"), std::string::npos) << stats;
  if (std::getenv("KT2N_TEST_EXCLUSIVE_GPU")) EXPECT_EQ(free0, free1) << "device free memory changed while serving";
  kt2n_engine_destroy(e);
}

// KT2N_MEMORY_PROFILE_AUTO on a GPU: with a simulated 6 GB of free memory (KT2N_LOW_FIT_FREE_MB) the low profile has
// to fit itself, so the automatic profile is that fitted low profile at once (same limits, same seeded audio as an
// explicit KT2N_MEMORY_PROFILE_LOW). With simulated free memory between the standard low sizes and the default
// profile's smallest need it takes the standard low profile without a default attempt. Without the simulation it picks
// whichever fits this GPU and says which. kt2n_voice_prepare: a one-voice LRU engine prepares the voice before any
// request, and the request's audio is unchanged. Needs KT2N_TEST_GPU=1 and KT2N_TEST_ASSETS; skipped otherwise.
TEST(Runtime, GpuAutoMemoryProfile) {
  const char* root = std::getenv("KT2N_TEST_ASSETS");
  if (!std::getenv("KT2N_TEST_GPU") || !root || !*root) GTEST_SKIP() << "set KT2N_TEST_GPU=1 and KT2N_TEST_ASSETS";
  const std::string n = std::string(root) + "/native/";
  const std::string lm = n + "model.kt2n", s3 = n + "s3gen.kt2n", vo = n + "voices.kt2n", tok = n + "tokenizer.json",
                    txt = n + "text";
  auto make = [&](int profile, kt2n_engine** e, kt2n_error* err, int cache = KT2N_VOICE_CACHE_NONE, int voices = 0) {
    kt2n_engine_config cfg{};
    cfg.struct_size = sizeof cfg;
    cfg.abi_version = KT2N_ABI_VERSION;
    cfg.flags = KT2N_CFG_INFERENCE_ONLY;
    cfg.max_streams = 1;
    cfg.lm_file = lm.c_str();
    cfg.s3gen_file = s3.c_str();
    cfg.voices_file = vo.c_str();
    cfg.tokenizer_json = tok.c_str();
    cfg.text_data_dir = txt.c_str();
    cfg.memory_profile = profile;
    cfg.voice_cache_mode = cache;
    cfg.voice_cache_max_voices = voices;
    return kt2n_engine_create(&cfg, e, err);
  };
  auto stats_of = [](kt2n_engine* e) {
    std::string s(1 << 16, '\0');
    size_t len = 0;
    if (kt2n_stats_json(e, s.data(), s.size(), &len) != KT2N_OK) return std::string();
    s.resize(len ? len - 1 : 0);
    return s;
  };
  auto pcm = [](kt2n_engine* e) {
    kt2n_voice_id voice = 0;
    kt2n_error err{};
    if (kt2n_voice_lookup(e, "Bruno", &voice, &err) != KT2N_OK) return std::string("no voice");
    const std::string text = "Hello there, how are you doing today? This is a small GPU.";
    kt2n_request_params p{};
    p.struct_size = sizeof p;
    p.flags = KT2N_REQ_SEED;
    p.seed = 7;
    p.text = text.data();
    p.text_len = text.size();
    p.voice = voice;
    p.speed = 1.0;
    p.format = KT2N_FORMAT_PCM;
    p.output = KT2N_OUTPUT_COMPLETE;
    kt2n_request_id id = 0;
    if (kt2n_submit(e, &p, &id, &err) != KT2N_OK) return std::string("submit failed");
    std::string bytes;
    int status = -2;
    const auto t0 = std::chrono::steady_clock::now();
    while (status == -2 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(120)) {
      kt2n_event ev[32];
      int32_t k = 0;
      kt2n_poll(e, ev, 32, 200, &k);
      for (int i = 0; i < k; ++i) {
        if (ev[i].id != id) continue;
        if (ev[i].kind == KT2N_EVENT_AUDIO) bytes.append(reinterpret_cast<const char*>(ev[i].data), ev[i].data_len);
        if (ev[i].kind == KT2N_EVENT_DONE) status = ev[i].status;
      }
      kt2n_events_release(e, ev, k);
    }
    kt2n_release(e, id);
    return status == KT2N_OK ? bytes : std::string("request failed");
  };
  setenv("KT2N_LOW_FIT_FREE_MB", "5671", 1);
  std::string low_pcm, auto_pcm;
  kt2n_low_memory_limits low_lim{}, auto_lim{};
  low_lim.struct_size = auto_lim.struct_size = sizeof(kt2n_low_memory_limits);
  for (int profile : {int(KT2N_MEMORY_PROFILE_LOW), int(KT2N_MEMORY_PROFILE_AUTO)}) {
    kt2n_engine* e = nullptr;
    kt2n_error err{};
    ASSERT_EQ(make(profile, &e, &err), KT2N_OK) << err.message;
    const std::string st = stats_of(e);
    EXPECT_EQ(st.find("\"memory_profile_auto\":\"low\"") != std::string::npos, profile == KT2N_MEMORY_PROFILE_AUTO) << st;
    ASSERT_EQ(kt2n_engine_low_memory_limits(e, profile == KT2N_MEMORY_PROFILE_LOW ? &low_lim : &auto_lim), KT2N_OK);
    (profile == KT2N_MEMORY_PROFILE_LOW ? low_pcm : auto_pcm) = pcm(e);
    kt2n_engine_destroy(e);
  }
  clear_sim_free();
  EXPECT_EQ(auto_lim.fitted, 1);
  EXPECT_EQ(auto_lim.lm_context_tokens, low_lim.lm_context_tokens);
  EXPECT_EQ(auto_lim.decoder_pages, low_lim.decoder_pages);
  EXPECT_EQ(auto_lim.vocoder_frames, low_lim.vocoder_frames);
  EXPECT_GT(low_pcm.size(), 1000u) << low_pcm;
  EXPECT_TRUE(auto_pcm == low_pcm);
  // Skip rule: with (simulated) free memory that holds the standard low sizes but not the default profile's smallest
  // need (the standard estimate + 2 GiB + its 10% floor), AUTO takes the low profile at its standard sizes without a
  // default attempt. On a GPU where the default profile fits, that attempt would succeed and the stats would say
  // "default". The simulated free memory is what a standard low engine took + its floor + (2 GiB + 10% - floor -
  // 256 MiB): above the fit's estimate (which is above what create takes) and below the default's smallest need.
  {
    kt2n_engine* e = nullptr;
    kt2n_error err{};
    size_t free0 = 0, free1 = 0, total = 0;
    ASSERT_EQ(cudaMemGetInfo(&free0, &total), 0);
    ASSERT_EQ(make(KT2N_MEMORY_PROFILE_LOW, &e, &err), KT2N_OK) << err.message;
    kt2n_low_memory_limits std_lim{};
    std_lim.struct_size = sizeof std_lim;
    ASSERT_EQ(kt2n_engine_low_memory_limits(e, &std_lim), KT2N_OK);
    ASSERT_EQ(cudaMemGetInfo(&free1, &total), 0);
    const std::string std_pcm = pcm(e);
    kt2n_engine_destroy(e);
    if (std_lim.fitted) {
      std::printf("[ note ] the standard low sizes do not fit this GPU: skip-rule case not run\n");
    } else {
      const uint64_t MiB = uint64_t(1) << 20, used = free0 > free1 ? free0 - free1 : 0;
      const uint64_t extra = (uint64_t(2048) << 20) + total / 10 - std_lim.min_free_bytes - 256 * MiB;
      const uint64_t sim = std::min<uint64_t>(free0, used + std_lim.min_free_bytes + extra);
      setenv("KT2N_LOW_FIT_FREE_MB", std::to_string(sim / MiB).c_str(), 1);
      e = nullptr;
      ASSERT_EQ(make(KT2N_MEMORY_PROFILE_AUTO, &e, &err), KT2N_OK) << err.message;
      clear_sim_free();
      const std::string st = stats_of(e);
      EXPECT_NE(st.find("\"memory_profile_auto\":\"low\""), std::string::npos) << "simulated " << sim / MiB << " MiB: " << st;
      kt2n_low_memory_limits lim{};
      lim.struct_size = sizeof lim;
      ASSERT_EQ(kt2n_engine_low_memory_limits(e, &lim), KT2N_OK);
      EXPECT_EQ(lim.fitted, 0);
      EXPECT_EQ(lim.lm_context_tokens, std_lim.lm_context_tokens);
      EXPECT_EQ(lim.decoder_pages, std_lim.decoder_pages);
      EXPECT_EQ(lim.render_graphs, std_lim.render_graphs);
      EXPECT_GT(std_pcm.size(), 1000u) << std_pcm;
      EXPECT_TRUE(pcm(e) == std_pcm);
      kt2n_engine_destroy(e);
    }
  }
  // kt2n_voice_prepare with the CLI's one-voice LRU cache: the voice is resident before any request (voices_loaded 1),
  // a second call returns at once, and the request's audio equals the one of an engine where the request prepared it.
  // With voice cache NONE it does nothing (returns OK, nothing resident); an unknown voice is refused.
  {
    std::string got[2];
    for (int prepare : {0, 1}) {
      kt2n_engine* e = nullptr;
      kt2n_error err{};
      ASSERT_EQ(make(KT2N_MEMORY_PROFILE_LOW, &e, &err, KT2N_VOICE_CACHE_LRU, 1), KT2N_OK) << err.message;
      kt2n_voice_id voice = 0;
      ASSERT_EQ(kt2n_voice_lookup(e, "Bruno", &voice, &err), KT2N_OK);
      kt2n_stats st{};
      st.struct_size = sizeof st;
      ASSERT_EQ(kt2n_stats_get(e, &st), KT2N_OK);
      EXPECT_EQ(st.voices_loaded, 0);
      if (prepare) {
        EXPECT_EQ(kt2n_voice_prepare(e, voice, &err), KT2N_OK) << err.message;
        EXPECT_EQ(kt2n_voice_prepare(e, voice, &err), KT2N_OK) << err.message;
        ASSERT_EQ(kt2n_stats_get(e, &st), KT2N_OK);
        EXPECT_EQ(st.voices_loaded, 1);
        EXPECT_EQ(kt2n_voice_prepare(e, kt2n_voice_id(1u << 30), &err), KT2N_ERR_INVALID_VOICE);
      }
      got[prepare] = pcm(e);
      kt2n_engine_destroy(e);
    }
    EXPECT_GT(got[0].size(), 1000u) << got[0];
    EXPECT_TRUE(got[1] == got[0]);
    kt2n_engine* e = nullptr;
    kt2n_error err{};
    ASSERT_EQ(make(KT2N_MEMORY_PROFILE_LOW, &e, &err), KT2N_OK) << err.message;  // voice cache NONE
    kt2n_voice_id voice = 0;
    ASSERT_EQ(kt2n_voice_lookup(e, "Bruno", &voice, &err), KT2N_OK);
    EXPECT_EQ(kt2n_voice_prepare(e, voice, &err), KT2N_OK);
    kt2n_stats st{};
    st.struct_size = sizeof st;
    ASSERT_EQ(kt2n_stats_get(e, &st), KT2N_OK);
    EXPECT_EQ(st.voices_loaded, 0);
    kt2n_engine_destroy(e);
  }
  // The real GPU: whichever profile fits, and the stats name it.
  kt2n_engine* e = nullptr;
  kt2n_error err{};
  ASSERT_EQ(make(KT2N_MEMORY_PROFILE_AUTO, &e, &err), KT2N_OK) << err.message;
  const std::string st = stats_of(e);
  EXPECT_TRUE(st.find("\"memory_profile_auto\":\"default\"") != std::string::npos ||
              st.find("\"memory_profile_auto\":\"low\"") != std::string::npos)
      << st;
  kt2n_engine_destroy(e);
}

// A create that fails on a GPU allocation leaves no CUDA error behind for the next create on the same thread. The
// first create asks for a decoder pool far larger than any GPU (render_pages: 1.65 TB), so it fails in
// cudaMalloc(render arena), after the language model has loaded; the CUDA runtime keeps that out-of-memory error as the
// thread's last error. Before the fix the next create read it at its first kernel-launch check and failed ("LM:
// kt2n_pf_model_load: repack qkv failed (rc 2)"); now the low, automatic and default profiles all start after it, with
// the same seeded audio as without the failed create. Needs KT2N_TEST_GPU=1 and KT2N_TEST_ASSETS; skipped otherwise.
TEST(Runtime, GpuCreateAfterAFailedCreate) {
  const char* root = std::getenv("KT2N_TEST_ASSETS");
  if (!std::getenv("KT2N_TEST_GPU") || !root || !*root) GTEST_SKIP() << "set KT2N_TEST_GPU=1 and KT2N_TEST_ASSETS";
  const std::string n = std::string(root) + "/native/";
  const std::string lm = n + "model.kt2n", s3 = n + "s3gen.kt2n", vo = n + "voices.kt2n", tok = n + "tokenizer.json",
                    txt = n + "text";
  auto make = [&](int profile, int render_pages, kt2n_engine** e, kt2n_error* err) {
    kt2n_engine_config cfg{};
    cfg.struct_size = sizeof cfg;
    cfg.abi_version = KT2N_ABI_VERSION;
    cfg.flags = KT2N_CFG_INFERENCE_ONLY;
    cfg.max_streams = 1;
    cfg.lm_file = lm.c_str();
    cfg.s3gen_file = s3.c_str();
    cfg.voices_file = vo.c_str();
    cfg.tokenizer_json = tok.c_str();
    cfg.text_data_dir = txt.c_str();
    cfg.memory_profile = profile;
    cfg.render_pages = render_pages;
    cfg.voice_cache_mode = KT2N_VOICE_CACHE_LRU;
    cfg.voice_cache_max_voices = 1;
    return kt2n_engine_create(&cfg, e, err);
  };
  auto pcm = [](kt2n_engine* e) {
    kt2n_voice_id voice = 0;
    kt2n_error err{};
    if (kt2n_voice_lookup(e, "Bruno", &voice, &err) != KT2N_OK) return std::string("no voice");
    const std::string text = "A second engine on the same thread.";
    kt2n_request_params p{};
    p.struct_size = sizeof p;
    p.flags = KT2N_REQ_SEED;
    p.seed = 11;
    p.text = text.data();
    p.text_len = text.size();
    p.voice = voice;
    p.speed = 1.0;
    p.format = KT2N_FORMAT_PCM;
    p.output = KT2N_OUTPUT_COMPLETE;
    kt2n_request_id id = 0;
    if (kt2n_submit(e, &p, &id, &err) != KT2N_OK) return std::string("submit failed");
    std::string bytes;
    int status = -2;
    const auto t0 = std::chrono::steady_clock::now();
    while (status == -2 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(120)) {
      kt2n_event ev[32];
      int32_t k = 0;
      kt2n_poll(e, ev, 32, 200, &k);
      for (int i = 0; i < k; ++i) {
        if (ev[i].id != id) continue;
        if (ev[i].kind == KT2N_EVENT_AUDIO) bytes.append(reinterpret_cast<const char*>(ev[i].data), ev[i].data_len);
        if (ev[i].kind == KT2N_EVENT_DONE) status = ev[i].status;
      }
      kt2n_events_release(e, ev, k);
    }
    kt2n_release(e, id);
    return status == KT2N_OK ? bytes : std::string("request failed");
  };
  for (int profile : {int(KT2N_MEMORY_PROFILE_LOW), int(KT2N_MEMORY_PROFILE_AUTO), int(KT2N_MEMORY_PROFILE_DEFAULT)}) {
    kt2n_engine* e = nullptr;
    kt2n_error err{};
    ASSERT_EQ(make(profile, 0, &e, &err), KT2N_OK) << "profile " << profile << ": " << err.message;
    const std::string clean = pcm(e);  // this profile's audio without a failed create before it
    kt2n_engine_destroy(e);
    e = nullptr;
    ASSERT_GT(clean.size(), 1000u) << clean;
    ASSERT_NE(make(KT2N_MEMORY_PROFILE_DEFAULT, 100000, &e, &err), KT2N_OK) << "a 1.65 TB decoder pool started";
    ASSERT_EQ(e, nullptr);
    EXPECT_NE(std::string(err.message).find("render arena"), std::string::npos) << err.message;
    err = kt2n_error{};
    ASSERT_EQ(make(profile, 0, &e, &err), KT2N_OK) << "profile " << profile << " after a failed create: " << err.message;
    const std::string audio = pcm(e);
    kt2n_engine_destroy(e);
    EXPECT_EQ(audio, clean) << "profile " << profile;
  }
}

#if !defined(KT2N_HIP)
// The driver's entry point for cuModuleGetLoadingMode, through the runtime (no CUDA headers or driver library here).
extern "C" int cudaGetDriverEntryPointByVersion(const char* symbol, void** fn, unsigned int cuda_version,
                                                unsigned long long flags, int* query_result);
#endif

// A create asks for eager module loading (CUDA_MODULE_LOADING=EAGER: every kernel's module loads when the context is
// made) before its first CUDA call. Any CUDA call before that, a cudaGetLastError included, initialises CUDA in the
// driver's lazy mode, and the first request then loads its modules inside its launches: on an RTX 4090 the first audio
// of a voice prepared inside the request went from 35 to 44 ms. Checks how this process's driver was initialised, so
// it has to run in a process of its own (ctest runs every test alone) with the CPU stubs' create as the first CUDA
// user. Needs KT2N_TEST_GPU=1; skipped when the environment sets the loading mode; CUDA builds only.
TEST(Runtime, GpuCreateLoadsModulesEagerly) {
#if defined(KT2N_HIP)
  GTEST_SKIP() << "CUDA only (ROCm has its own switch, HIP_ENABLE_DEFERRED_LOADING)";
#else
  if (!std::getenv("KT2N_TEST_GPU")) GTEST_SKIP() << "set KT2N_TEST_GPU=1";
  if (std::getenv("CUDA_MODULE_LOADING") || std::getenv("KT2N_CUDA_EAGER"))
    GTEST_SKIP() << "CUDA_MODULE_LOADING or KT2N_CUDA_EAGER is set in the environment";
  Eng eng(1);
  void* fn = nullptr;
  int query = -1;
  ASSERT_EQ(cudaGetDriverEntryPointByVersion("cuModuleGetLoadingMode", &fn, 12000, 0, &query), 0);
  ASSERT_EQ(query, 0) << "cuModuleGetLoadingMode not found";
  int mode = 0;  // CUmoduleLoadingMode: 1 eager, 2 lazy
  ASSERT_EQ(reinterpret_cast<int (*)(int*)>(fn)(&mode), 0);
  EXPECT_EQ(mode, 1) << "the driver was initialised in lazy module-loading mode";
#endif
}

// An explicit KT2N_MEMORY_PROFILE_DEFAULT on a GPU too small for it (simulated free memory, KT2N_LOW_FIT_FREE_MB) is
// refused with KT2N_ERR_TOO_SMALL and how much it needs, before any model memory is allocated, instead of failing in
// whichever allocation runs out first: a 4 GB card (the low profile has to fit itself), a 16 GB card with every voice
// prepared (the default pool holds every voice prefix), and a card too small even for the low profile. Needs
// KT2N_TEST_GPU=1 and KT2N_TEST_ASSETS; skipped otherwise.
TEST(Runtime, GpuDefaultMemoryProfileRefusesBeforeAllocating) {
  const char* root = std::getenv("KT2N_TEST_ASSETS");
  if (!std::getenv("KT2N_TEST_GPU") || !root || !*root) GTEST_SKIP() << "set KT2N_TEST_GPU=1 and KT2N_TEST_ASSETS";
  const std::string n = std::string(root) + "/native/";
  const std::string lm = n + "model.kt2n", s3 = n + "s3gen.kt2n", vo = n + "voices.kt2n", tok = n + "tokenizer.json",
                    txt = n + "text";
  struct Case {
    const char* free_mb;
    int cache;
    int lm_rows;
    const char* says;
  };
  // The last case: more LM rows (max_lm_rows only raises them, in both profiles, and the estimate counts them) on a
  // simulated 12 GB card: still refused before allocating, not by the LM's allocation.
  warm_gpu_runtime();
  for (const Case& c : {Case{"3900", KT2N_VOICE_CACHE_NONE, 0, "the default memory profile needs about "},
                        Case{"15000", KT2N_VOICE_CACHE_ALL, 0, "voice cache all"},
                        Case{"1500", KT2N_VOICE_CACHE_NONE, 0, "neither does the low-memory profile"},
                        Case{"12000", KT2N_VOICE_CACHE_NONE, 16, "the default memory profile needs about "}}) {
    kt2n_engine_config cfg{};
    cfg.struct_size = sizeof cfg;
    cfg.abi_version = KT2N_ABI_VERSION;
    cfg.flags = KT2N_CFG_INFERENCE_ONLY;
    cfg.max_streams = 1;
    cfg.lm_file = lm.c_str();
    cfg.s3gen_file = s3.c_str();
    cfg.voices_file = vo.c_str();
    cfg.tokenizer_json = tok.c_str();
    cfg.text_data_dir = txt.c_str();
    cfg.memory_profile = KT2N_MEMORY_PROFILE_DEFAULT;
    cfg.voice_cache_mode = c.cache;
    cfg.max_lm_rows = c.lm_rows;
    cfg.min_free_device_bytes = uint64_t(128) << 20;  // the floor of a small GPU, whatever this GPU's size
    set_sim_free_exact(c.free_mb);
    kt2n_engine* e = nullptr;
    kt2n_error err{};
    int32_t rc = KT2N_OK;
    double s = 0;
    size_t used = 0;
    for (int attempt = 1; attempt <= free_memory_attempts(); ++attempt) {
      settle_gpu_free_memory();
      size_t free0 = 0, free1 = 0, total = 0;
      ASSERT_EQ(cudaMemGetInfo(&free0, &total), 0);
      err = kt2n_error{};
      const auto t0 = std::chrono::steady_clock::now();
      rc = kt2n_engine_create(&cfg, &e, &err);
      s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      ASSERT_EQ(cudaMemGetInfo(&free1, &total), 0);
      used = free0 > free1 ? free0 - free1 : 0;
      if (e || used < (size_t(64) << 20)) break;
      std::printf("[ note ] %s: free GPU memory fell %zu MiB during a refused create (attempt %d)\n", c.free_mb,
                  used >> 20, attempt);
    }
    clear_sim_free();
    EXPECT_EQ(rc, KT2N_ERR_TOO_SMALL) << c.free_mb << ": " << err.message;
    EXPECT_EQ(e, nullptr);
    if (e) kt2n_engine_destroy(e);
    EXPECT_STREQ(err.param, "memory_profile");
    EXPECT_NE(std::string(err.message).find(c.says), std::string::npos) << err.message;
    EXPECT_NE(std::string(err.message).find("default memory profile"), std::string::npos) << err.message;
    // Nothing of the model was loaded: no device memory kept, and no seconds spent loading.
    EXPECT_LT(used, size_t(64) << 20) << c.free_mb;
    EXPECT_LT(s, 5.0);
  }
}

// A seeded complete-file request's PCM (empty on failure).
static std::string seeded_complete_pcm(kt2n_engine* e) {
  kt2n_voice_id voice = 0;
  kt2n_error err{};
  if (kt2n_voice_lookup(e, "Bruno", &voice, &err) != KT2N_OK) return {};
  const std::string text = "Hello there, how are you doing today? This is a small GPU.";
  kt2n_request_params p{};
  p.struct_size = sizeof p;
  p.flags = KT2N_REQ_SEED;
  p.seed = 7;
  p.text = text.data();
  p.text_len = text.size();
  p.voice = voice;
  p.speed = 1.0;
  p.format = KT2N_FORMAT_PCM;
  p.output = KT2N_OUTPUT_COMPLETE;
  kt2n_request_id id = 0;
  if (kt2n_submit(e, &p, &id, &err) != KT2N_OK) return {};
  std::string bytes;
  int status = -2;
  const auto t0 = std::chrono::steady_clock::now();
  while (status == -2 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(120)) {
    kt2n_event ev[32];
    int32_t k = 0;
    kt2n_poll(e, ev, 32, 200, &k);
    for (int i = 0; i < k; ++i) {
      if (ev[i].id != id) continue;
      if (ev[i].kind == KT2N_EVENT_AUDIO) bytes.append(reinterpret_cast<const char*>(ev[i].data), ev[i].data_len);
      if (ev[i].kind == KT2N_EVENT_DONE) status = ev[i].status;
    }
    kt2n_events_release(e, ev, k);
  }
  kt2n_release(e, id);
  return status == KT2N_OK ? bytes : std::string();
}

// The DEFAULT memory check prices the pool the default profile chooses itself, so it is skipped when the caller sets
// render_pages (a smaller pool the default create accepts by shrinking the voice cache budget): no refusal from the
// check. When the check passes, the create uses the tokenizer and voices file the check read: the same audio as a
// create without the check. Needs KT2N_TEST_GPU=1, KT2N_TEST_ASSETS and 20 GB of free GPU memory; skipped otherwise.
TEST(Runtime, GpuDefaultMemoryProfileCheckSkipsRenderPagesAndKeepsAudio) {
  const char* root = std::getenv("KT2N_TEST_ASSETS");
  if (!std::getenv("KT2N_TEST_GPU") || !root || !*root) GTEST_SKIP() << "set KT2N_TEST_GPU=1 and KT2N_TEST_ASSETS";
  size_t free0 = 0, total = 0;
  ASSERT_EQ(cudaMemGetInfo(&free0, &total), 0);
  if (free0 < (size_t(20) << 30))
    GTEST_SKIP() << "needs 20 GB of free GPU memory (two default-profile engines in turn; the render_pages case is "
                    "skipped with them), has "
                 << (free0 >> 20) << " MiB";
  const std::string n = std::string(root) + "/native/";
  const std::string lm = n + "model.kt2n", s3 = n + "s3gen.kt2n", vo = n + "voices.kt2n", tok = n + "tokenizer.json",
                    txt = n + "text";
  auto config = [&](int cache, int voices, int pages) {
    kt2n_engine_config cfg{};
    cfg.struct_size = sizeof cfg;
    cfg.abi_version = KT2N_ABI_VERSION;
    cfg.flags = KT2N_CFG_INFERENCE_ONLY;
    cfg.max_streams = 1;
    cfg.lm_file = lm.c_str();
    cfg.s3gen_file = s3.c_str();
    cfg.voices_file = vo.c_str();
    cfg.tokenizer_json = tok.c_str();
    cfg.text_data_dir = txt.c_str();
    cfg.memory_profile = KT2N_MEMORY_PROFILE_DEFAULT;
    cfg.voice_cache_mode = cache;
    cfg.voice_cache_max_voices = voices;
    cfg.render_pages = pages;
    cfg.min_free_device_bytes = uint64_t(128) << 20;
    return cfg;
  };
  {
    // LRU of 16 voices in 80 pages: the estimate (about 226 pages) would refuse at this simulated free memory.
    const kt2n_engine_config cfg = config(KT2N_VOICE_CACHE_LRU, 16, 80);
    setenv("KT2N_LOW_FIT_FREE_MB", sim_free_mb(3900, 128).c_str(), 1);
    kt2n_engine* e = nullptr;
    kt2n_error err{};
    const int32_t rc = kt2n_engine_create(&cfg, &e, &err);
    clear_sim_free();
    EXPECT_EQ(std::string(err.message).find("the default memory profile needs about"), std::string::npos) << err.message;
    EXPECT_EQ(rc, KT2N_OK) << err.message;
    if (e) kt2n_engine_destroy(e);
  }
  std::string checked, unchecked;
  for (const bool check : {true, false}) {
    const kt2n_engine_config cfg = config(KT2N_VOICE_CACHE_NONE, 0, 0);
    setenv("KT2N_LOW_FIT_FREE_MB", sim_free_mb(20000, 128).c_str(), 1);  // below 22 GiB: the check runs (and passes)
    if (!check) setenv("KT2N_DEFAULT_CHECK", "0", 1);
    kt2n_engine* e = nullptr;
    kt2n_error err{};
    const int32_t rc = kt2n_engine_create(&cfg, &e, &err);
    clear_sim_free();
    unsetenv("KT2N_DEFAULT_CHECK");
    ASSERT_EQ(rc, KT2N_OK) << err.message;
    (check ? checked : unchecked) = seeded_complete_pcm(e);
    kt2n_engine_destroy(e);
  }
  EXPECT_FALSE(checked.empty());
  EXPECT_TRUE(checked == unchecked) << "the checked create's audio differs";
}

// Low-memory profile on a small GPU (simulated with the KT2N_LOW_FIT_FREE_MB test knob: the RTX A2000 6 GB's 5,671 MiB
// free after the context): create fits itself (smaller context, eager decoder), serves a streamed and a complete-file
// request, and a segment beyond a fitted context (KT2N_LM_MAX_CTX lab knob, 1,024) fails with
// KT2N_ERR_GPU_MEMORY_LIMIT instead of being shortened. Needs KT2N_TEST_GPU=1 and KT2N_TEST_ASSETS; skipped otherwise.
TEST(Runtime, GpuLowMemoryProfileFitsSmallGpu) {
  const char* root = std::getenv("KT2N_TEST_ASSETS");
  if (!std::getenv("KT2N_TEST_GPU") || !root || !*root) GTEST_SKIP() << "set KT2N_TEST_GPU=1 and KT2N_TEST_ASSETS";
  const std::string n = std::string(root) + "/native/";
  const std::string lm = n + "model.kt2n", s3 = n + "s3gen.kt2n", vo = n + "voices.kt2n", tok = n + "tokenizer.json",
                    txt = n + "text";
  auto make = [&](kt2n_engine** e, kt2n_error* err) {
    kt2n_engine_config cfg{};
    cfg.struct_size = sizeof cfg;
    cfg.abi_version = KT2N_ABI_VERSION;
    cfg.flags = KT2N_CFG_INFERENCE_ONLY;
    cfg.max_streams = 1;
    cfg.lm_file = lm.c_str();
    cfg.s3gen_file = s3.c_str();
    cfg.voices_file = vo.c_str();
    cfg.tokenizer_json = tok.c_str();
    cfg.text_data_dir = txt.c_str();
    cfg.memory_profile = KT2N_MEMORY_PROFILE_LOW;
    cfg.voice_cache_mode = KT2N_VOICE_CACHE_NONE;
    cfg.min_free_device_bytes = 128ull << 20;
    return kt2n_engine_create(&cfg, e, err);
  };
  std::string last_error;  // the DONE event's error message of the last run
  auto run = [&last_error](kt2n_engine* e, kt2n_voice_id voice, const std::string& text, int output, size_t* bytes) {
    last_error.clear();
    kt2n_request_params p{};
    p.struct_size = sizeof p;
    p.flags = KT2N_REQ_SEED;
    p.seed = 7;
    p.text = text.data();
    p.text_len = text.size();
    p.voice = voice;
    p.speed = 1.0;
    p.format = KT2N_FORMAT_PCM;
    p.output = output;
    kt2n_request_id id = 0;
    kt2n_error err{};
    if (kt2n_submit(e, &p, &id, &err) != KT2N_OK) return -1;
    int status = -2;
    const auto t0 = std::chrono::steady_clock::now();
    while (status == -2 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(120)) {
      kt2n_event ev[32];
      int32_t k = 0;
      kt2n_poll(e, ev, 32, 200, &k);
      for (int i = 0; i < k; ++i) {
        if (ev[i].id != id) continue;
        if (ev[i].kind == KT2N_EVENT_AUDIO) *bytes += ev[i].data_len;
        if (ev[i].kind == KT2N_EVENT_DONE) {
          status = ev[i].status;
          if (ev[i].error) last_error = ev[i].error->message;
        }
      }
      kt2n_events_release(e, ev, k);
    }
    kt2n_release(e, id);
    return status;
  };
  const std::string text = "Hello there, how are you doing today? This is a small GPU.";
  setenv("KT2N_LOW_FIT_FREE_MB", sim_free_mb(5671, 128).c_str(), 1);
  kt2n_engine* e = nullptr;
  kt2n_error err{};
  const int rc = make(&e, &err);
  clear_sim_free();
  ASSERT_EQ(rc, KT2N_OK) << err.message;
  kt2n_low_memory_limits lim{};
  lim.struct_size = sizeof lim;
  ASSERT_EQ(kt2n_engine_low_memory_limits(e, &lim), KT2N_OK);
  EXPECT_EQ(lim.fitted, 1);
  EXPECT_EQ(lim.render_graphs, 0);
  EXPECT_LE(lim.lm_context_tokens, 4096);
  EXPECT_GE(lim.lm_context_tokens, 2400);  // the corpora's longest prompt + budget is 2,399
  EXPECT_EQ(lim.streaming_budget_tokens, 0);  // 6 GB: streaming segments up to the standard cap
  kt2n_voice_id voice = 0;
  ASSERT_EQ(kt2n_voice_lookup(e, "Bruno", &voice, &err), KT2N_OK) << err.message;
  for (int output : {int(KT2N_OUTPUT_STREAM), int(KT2N_OUTPUT_COMPLETE)}) {
    size_t bytes = 0;
    EXPECT_EQ(run(e, voice, text, output, &bytes), KT2N_OK);
    EXPECT_GT(bytes, 0u);
  }
  kt2n_engine_destroy(e);
  // A fitted context of 1,024: a complete-file segment's budget (1,000) plus its prompt does not fit.
  setenv("KT2N_LM_MAX_CTX", "1024", 1);
  e = nullptr;
  const int rc2 = make(&e, &err);
  unsetenv("KT2N_LM_MAX_CTX");
  ASSERT_EQ(rc2, KT2N_OK) << err.message;
  ASSERT_EQ(kt2n_voice_lookup(e, "Bruno", &voice, &err), KT2N_OK) << err.message;
  size_t bytes = 0;
  EXPECT_EQ(run(e, voice, text, KT2N_OUTPUT_COMPLETE, &bytes), KT2N_ERR_GPU_MEMORY_LIMIT);
  EXPECT_EQ(bytes, 0u);
  const kt2n_error_info* info = kt2n_error_lookup(KT2N_ERR_GPU_MEMORY_LIMIT);
  ASSERT_NE(info, nullptr);
  EXPECT_STREQ(info->code, "gpu_memory_limit_exceeded");
  kt2n_engine_destroy(e);
  // 4 GB cards (simulated free memory after the CUDA context; KT2N_LOW_FIT_FREE_MB): 3,680 MiB (the A2000 with
  // ballast) and 3,300 MiB (a real 4 GB card with a desktop running). The fit keeps complete-file segments at their
  // 1,000-token budget and streaming segments at least at the budget of a 380-character chunk (4.4 tokens per
  // character: 1,672); a longer streaming segment the fit cannot hold is refused with KT2N_ERR_GPU_MEMORY_LIMIT (no
  // audio), never shortened.
  const std::string long380 =
      "The ferry left the harbour a little after dawn, and for the first hour the water was so calm that the wake "
      "behind us looked like a seam in grey silk, opening slowly and closing again behind the boat while the gulls "
      "followed us out past the breakwater and the lighthouse and the long line of moored fishing boats that were "
      "still asleep at their ropes in the early cold morning light today.";
  for (const char* free_mb : {"3680", "3300"}) {
    setenv("KT2N_LOW_FIT_FREE_MB", sim_free_mb(std::atoll(free_mb), 128).c_str(), 1);
    e = nullptr;
    const int rc3 = make(&e, &err);
    clear_sim_free();
    ASSERT_EQ(rc3, KT2N_OK) << free_mb << ": " << err.message;
    kt2n_low_memory_limits lim4{};
    lim4.struct_size = sizeof lim4;
    ASSERT_EQ(kt2n_engine_low_memory_limits(e, &lim4), KT2N_OK);
    EXPECT_EQ(lim4.fitted, 1) << free_mb;
    // 0 = the standard streaming cap (2,341); otherwise at least a 380-character chunk's budget.
    EXPECT_TRUE(lim4.streaming_budget_tokens == 0 || lim4.streaming_budget_tokens >= 1672)
        << free_mb << ": " << lim4.streaming_budget_tokens;
    EXPECT_GE(lim4.vocoder_frames, 2 * (1000 + 6)) << free_mb;
    ASSERT_EQ(kt2n_voice_lookup(e, "Bruno", &voice, &err), KT2N_OK) << err.message;
    bytes = 0;
    EXPECT_EQ(run(e, voice, long380, KT2N_OUTPUT_COMPLETE, &bytes), KT2N_OK) << free_mb;
    EXPECT_GT(bytes, 0u);
    bytes = 0;
    EXPECT_EQ(run(e, voice, long380, KT2N_OUTPUT_STREAM, &bytes), KT2N_OK) << free_mb;  // a 380-character chunk
    EXPECT_GT(bytes, 0u);
    bytes = 0;
    EXPECT_EQ(run(e, voice, text, KT2N_OUTPUT_STREAM, &bytes), KT2N_OK) << free_mb;
    EXPECT_GT(bytes, 0u);
    // Two 380-character chunks in one request: served, or refused with the documented error (never another failure).
    bytes = 0;
    const int st = run(e, voice, long380 + " " + long380, KT2N_OUTPUT_STREAM, &bytes);
    EXPECT_TRUE(st == KT2N_OK || st == KT2N_ERR_GPU_MEMORY_LIMIT) << free_mb << ": " << st;
    kt2n_engine_destroy(e);
  }
  // Smallest tier (simulated 2,700 MiB): a 312-character sentence and a short one are one text-plan chunk (up to 380
  // characters) whose budget (4.4 tokens per character, ~1,450) is beyond the fitted streaming budget. It is re-split
  // into pieces that fit and served (stats fit_split_chunks 1), or, with KT2N_FIT_SPLIT=0, refused before any audio as
  // before. The short sentence checks that the text plan's merging of short pieces does not undo the split.
  const std::string long312 =
      "The ferry left the harbour a little after dawn, and for the first hour the water was so calm that the wake "
      "behind us looked like a seam in grey silk, opening slowly and closing again behind the boat while the gulls "
      "followed us out past the breakwater and the lighthouse and the long line of moored fishing boats. It was cold.";
  for (const char* split : {"1", "0"}) {
    setenv("KT2N_LOW_FIT_FREE_MB", sim_free_mb(2700, 128).c_str(), 1);
    setenv("KT2N_FIT_SPLIT", split, 1);
    e = nullptr;
    const int rc5 = make(&e, &err);
    clear_sim_free();
    unsetenv("KT2N_FIT_SPLIT");
    ASSERT_EQ(rc5, KT2N_OK) << err.message;
    kt2n_low_memory_limits lim5{};
    lim5.struct_size = sizeof lim5;
    ASSERT_EQ(kt2n_engine_low_memory_limits(e, &lim5), KT2N_OK);
    EXPECT_GT(lim5.streaming_budget_tokens, 0);
    ASSERT_LT(lim5.streaming_budget_tokens, 1300) << "the 312-character chunk must be beyond the fitted budget";
    ASSERT_EQ(kt2n_voice_lookup(e, "Bruno", &voice, &err), KT2N_OK) << err.message;
    bytes = 0;
    const int st5 = run(e, voice, long312, KT2N_OUTPUT_STREAM, &bytes);
    if (split[0] == '1') {
      EXPECT_EQ(st5, KT2N_OK);
      EXPECT_GT(bytes, 0u);
    } else {
      EXPECT_EQ(st5, KT2N_ERR_GPU_MEMORY_LIMIT);
      EXPECT_EQ(bytes, 0u);
      // The refusal names the fitted streaming limit (decoder pages) or the fitted context.
      EXPECT_TRUE(last_error.find("streaming segments fit up to a budget of") != std::string::npos ||
                  last_error.find("language-model context positions") != std::string::npos)
          << last_error;
    }
    std::string js(1 << 16, '\0');
    size_t len = 0;
    ASSERT_EQ(kt2n_stats_json(e, js.data(), js.size(), &len), KT2N_OK);
    EXPECT_NE(js.find(std::string("\"fit_split_chunks\":") + (split[0] == '1' ? "1," : "0,")), std::string::npos)
        << js.substr(0, std::min<size_t>(len, 600));
    kt2n_engine_destroy(e);
  }
  // 2 GB class (simulated 1,600 MiB free): create refuses with KT2N_ERR_TOO_SMALL and the free memory it needs.
  setenv("KT2N_LOW_FIT_FREE_MB", sim_free_mb(1600, 128).c_str(), 1);
  e = nullptr;
  err = kt2n_error{};
  const int rc4 = make(&e, &err);
  clear_sim_free();
  EXPECT_EQ(rc4, KT2N_ERR_TOO_SMALL);
  EXPECT_NE(std::string(err.message).find("MiB of free GPU memory for one stream"), std::string::npos) << err.message;
  if (rc4 == KT2N_OK) kt2n_engine_destroy(e);
}


// The fitted sizes give the same audio as the standard low sizes on the same GPU (seeded complete files, byte for
// byte): at the 4 GB sizes, at a size where a complete-file row must give its voice's prefix pages back (checked in
// the stats), and with a saved voice (not inference-only, so the clone-voice page and context allowances apply).
// Needs a GPU where the standard low sizes fit (about 9 GB free).
TEST(Runtime, GpuLowMemoryFittedAudioMatchesStandard) {
  const char* root = std::getenv("KT2N_TEST_ASSETS");
  if (!std::getenv("KT2N_TEST_GPU") || !root || !*root) GTEST_SKIP() << "set KT2N_TEST_GPU=1 and KT2N_TEST_ASSETS";
  const std::string n = std::string(root) + "/native/";
  const std::string lm = n + "model.kt2n", s3 = n + "s3gen.kt2n", vo = n + "voices.kt2n", tok = n + "tokenizer.json",
                    txt = n + "text";
  // The built-in voice with the longest decoder prompt (its complete file needs the most pages, so at the smallest
  // size below it must take its prefix's pages), and a saved voice made from it (what --save-voice writes).
  std::vector<uint8_t> saved, saved_long;
  std::string vid;
  {
    using namespace kt2n::voice;
    std::ifstream f(vo, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const VoicesBlob cat = parse_voices_blob(bytes.data(), bytes.size(), true);
    const VoiceRecord* src = nullptr;
    for (const auto& r : cat.voices)
      if (r.raw_speaker.size() == 512 && (!src || r.prompt_token.size() > src->prompt_token.size())) src = &r;
    ASSERT_NE(src, nullptr);
    vid = src->id();
    CloneIdentityInputs in{std::string(64, '0'), std::string(64, '0'), std::string(64, '0'), vid};
    const Catalogue c = Catalogue::from_blob(cat);
    const int nearest = c.nearest(src->raw_speaker.data());
    ASSERT_GE(nearest, 0);
    in.nearest_voice_id = c.ids[size_t(nearest)];
    VoicesBlob b;
    b.top = default_top_meta(c.checkpoint, c.s3gen_sha256);
    b.voices.push_back(make_clone_record("saved-" + vid, "saved", src->lm_prefix, src->raw_speaker, src->projected_speaker,
                                         src->prompt_token, src->prompt_feat, src->embedding, "sig",
                                         clone_identity(in)));
    saved = write_voices_blob(b);
    // The same voice with a language-model prompt of 1,100 ids (longer than the 1,024 the fit plans for by default).
    std::vector<int32_t> long_prefix;
    while (long_prefix.size() < 1100) long_prefix.insert(long_prefix.end(), src->lm_prefix.begin(), src->lm_prefix.end());
    long_prefix.resize(1100);
    b.voices.clear();
    b.voices.push_back(make_clone_record("saved-long-" + vid, "saved", long_prefix, src->raw_speaker,
                                         src->projected_speaker, src->prompt_token, src->prompt_feat, src->embedding,
                                         "sig", clone_identity(in)));
    saved_long = write_voices_blob(b);
  }
  ASSERT_GT(kt2n_voice_artifact_lm_prefix_ids(saved.data(), saved.size()), 0);
  ASSERT_EQ(kt2n_voice_artifact_lm_prefix_ids(saved_long.data(), saved_long.size()), 1100);
  int tight_prefix = 0;  // > 0: custom_voice_max_prefix_ids for the next make()
  bool cache_all = false;  // the next make() uses voice cache mode ALL
  const char* build_dir = nullptr;  // voice_build_dir for the next make() (the CLI's --reference-audio sets one)
  auto make = [&](kt2n_engine** e, kt2n_error* err, bool with_saved) {
    kt2n_engine_config cfg{};
    cfg.struct_size = sizeof cfg;
    cfg.abi_version = KT2N_ABI_VERSION;
    cfg.flags = with_saved ? 0u : uint32_t(KT2N_CFG_INFERENCE_ONLY);
    if (with_saved) {
      cfg.max_custom_voices = cache_all ? 2 : 1;  // ALL: two 512-id allowances hold the 1,100-id voice
      cfg.custom_voice_max_prefix_ids =
          tight_prefix > 0 ? tight_prefix : kt2n_voice_artifact_lm_prefix_ids(saved.data(), saved.size());
    }
    cfg.max_streams = 1;
    cfg.lm_file = lm.c_str();
    cfg.s3gen_file = s3.c_str();
    cfg.voices_file = vo.c_str();
    cfg.tokenizer_json = tok.c_str();
    cfg.text_data_dir = txt.c_str();
    cfg.memory_profile = KT2N_MEMORY_PROFILE_LOW;
    cfg.voice_cache_mode = cache_all ? KT2N_VOICE_CACHE_ALL : KT2N_VOICE_CACHE_NONE;
    if (build_dir) cfg.voice_build_dir = build_dir;
    cfg.min_free_device_bytes = 128ull << 20;
    return kt2n_engine_create(&cfg, e, err);
  };
  auto run = [](kt2n_engine* e, kt2n_voice_id voice, const std::string& text, int output, std::string* pcm) {
    kt2n_request_params p{};
    p.struct_size = sizeof p;
    p.flags = KT2N_REQ_SEED;
    p.seed = 11;
    p.text = text.data();
    p.text_len = text.size();
    p.voice = voice;
    p.speed = 1.0;
    p.format = KT2N_FORMAT_PCM;
    p.output = output;
    kt2n_request_id id = 0;
    kt2n_error err{};
    if (kt2n_submit(e, &p, &id, &err) != KT2N_OK) return -1;
    int status = -2;
    const auto t0 = std::chrono::steady_clock::now();
    while (status == -2 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(180)) {
      kt2n_event ev[32];
      int32_t k = 0;
      kt2n_poll(e, ev, 32, 200, &k);
      for (int i = 0; i < k; ++i) {
        if (ev[i].id != id) continue;
        if (ev[i].kind == KT2N_EVENT_AUDIO) pcm->append(reinterpret_cast<const char*>(ev[i].data), ev[i].data_len);
        if (ev[i].kind == KT2N_EVENT_DONE) status = ev[i].status;
      }
      kt2n_events_release(e, ev, k);
    }
    kt2n_release(e, id);
    return status;
  };
  auto stat = [](kt2n_engine* e, const char* key) {
    std::string js(1 << 20, '\0');
    size_t len = 0;
    kt2n_stats_json(e, js.data(), js.size(), &len);
    js.resize(len);
    const auto at = js.find(std::string("\"") + key + "\":");
    return at == std::string::npos ? -1L : std::atol(js.c_str() + at + std::strlen(key) + 3);
  };
  const std::string short_text = "Hello there, how are you doing today? This is a small GPU.";
  const std::string long380 =
      "The ferry left the harbour a little after dawn, and for the first hour the water was so calm that the wake "
      "behind us looked like a seam in grey silk, opening slowly and closing again behind the boat while the gulls "
      "followed us out past the breakwater and the lighthouse and the long line of moored fishing boats that were "
      "still asleep at their ropes in the early cold morning light today.";
  struct Out {
    std::string a, b, saved_a;
  };
  // free_mb 0: no simulation (the standard low sizes, which must not be fitted on the test GPU).
  auto serve = [&](const char* free_mb, Out* o, long* givebacks, int* fitted) {
    if (free_mb) setenv("KT2N_LOW_FIT_FREE_MB", sim_free_mb(std::atoll(free_mb), 128).c_str(), 1);
    for (bool with_saved : {false, true}) {
      kt2n_engine* e = nullptr;
      kt2n_error err{};
      // Fitted sizes with a voice_build_dir that does not exist: a fitted engine never loads the cloning model, so
      // create still succeeds (and kt2n_voice_build answers 214 below). A build without voice cloning (e.g. the
      // Windows presets) refuses any voice_build_dir at create, so there the engine gets none; its kt2n_voice_build
      // answers 214 as well.
      const bool cloning_built = std::strstr(kt2n_build_info(), KT2N_CAP_VOICE_CLONING) != nullptr;
      build_dir = (free_mb && with_saved && cloning_built) ? "/nonexistent/kt2n-voice-build-dir" : nullptr;
      const int rc = make(&e, &err, with_saved);
      build_dir = nullptr;
      ASSERT_EQ(rc, KT2N_OK) << (free_mb ? free_mb : "standard") << ": " << err.message;
      kt2n_low_memory_limits lim{};
      lim.struct_size = sizeof lim;
      ASSERT_EQ(kt2n_engine_low_memory_limits(e, &lim), KT2N_OK);
      if (!with_saved) *fitted = lim.fitted;
      kt2n_voice_id voice = 0;
      if (with_saved) {
        ASSERT_EQ(kt2n_voice_load(e, saved.data(), saved.size(), &voice, &err), KT2N_OK) << err.message;
        // no cloning on this engine's GPU budget (or no clone groups in this model): 214 either way
        kt2n_voice_build_params bp{};
        bp.struct_size = sizeof bp;
        kt2n_job_id job = 0;
        if (lim.fitted) EXPECT_EQ(kt2n_voice_build(e, &bp, &job, &err), KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE);
        EXPECT_EQ(run(e, voice, long380, KT2N_OUTPUT_COMPLETE, &o->saved_a), KT2N_OK) << free_mb;
      } else {
        ASSERT_EQ(kt2n_voice_lookup(e, vid.c_str(), &voice, &err), KT2N_OK) << err.message;
        std::string st1, st2;
        EXPECT_EQ(run(e, voice, short_text, KT2N_OUTPUT_STREAM, &st1), KT2N_OK) << free_mb;
        EXPECT_EQ(run(e, voice, long380, KT2N_OUTPUT_COMPLETE, &o->a), KT2N_OK) << free_mb;
        EXPECT_EQ(run(e, voice, short_text, KT2N_OUTPUT_STREAM, &st2), KT2N_OK) << free_mb;
        EXPECT_EQ(run(e, voice, short_text, KT2N_OUTPUT_COMPLETE, &o->b), KT2N_OK) << free_mb;
        EXPECT_GT(st1.size(), 0u);
        EXPECT_GT(st2.size(), 0u);
        *givebacks = stat(e, "demo_prefix_givebacks");
      }
      kt2n_engine_destroy(e);
      if (with_saved && !free_mb) {
        // custom_voice_max_prefix_ids below this voice's prompt: kt2n_voice_load refuses it (engine.h).
        kt2n_engine* e2 = nullptr;
        kt2n_error err2{};
        tight_prefix = kt2n_voice_artifact_lm_prefix_ids(saved.data(), saved.size()) - 1;
        const int rc2 = make(&e2, &err2, true);
        tight_prefix = 0;
        ASSERT_EQ(rc2, KT2N_OK) << err2.message;
        kt2n_voice_id v2 = 0;
        EXPECT_EQ(kt2n_voice_load(e2, saved.data(), saved.size(), &v2, &err2), KT2N_ERR_GPU_MEMORY_LIMIT);
        EXPECT_NE(std::string(err2.message).find("custom_voice_max_prefix_ids"), std::string::npos) << err2.message;
        kt2n_engine_destroy(e2);
        // A declared bound above 1,024 (the CLI passes the loaded voice's own length) is not enforced: the standard low
        // sizes with voice cache ALL load a 1,100-id saved voice, as before the bound existed; a bound of 1,024 refuses it.
        for (const int bound : {1100, 1024}) {
          tight_prefix = bound;
          cache_all = true;
          e2 = nullptr;
          err2 = kt2n_error{};
          const int rc3 = make(&e2, &err2, true);
          tight_prefix = 0;
          cache_all = false;
          ASSERT_EQ(rc3, KT2N_OK) << err2.message;
          EXPECT_EQ(kt2n_voice_load(e2, saved_long.data(), saved_long.size(), &v2, &err2),
                    bound > 1024 ? KT2N_OK : KT2N_ERR_GPU_MEMORY_LIMIT)
              << bound << ": " << err2.message;
          kt2n_engine_destroy(e2);
        }
      }
    }
    clear_sim_free();
  };
  Out ref;
  long gb = -1;
  int fitted = -1;
  serve(nullptr, &ref, &gb, &fitted);
  if (fitted != 0) GTEST_SKIP() << "the standard low sizes do not fit this GPU; nothing to compare against";
  EXPECT_EQ(gb, -1) << "the fit's counters must not appear in the stats of an engine that was not fitted";
  ASSERT_GT(ref.a.size(), 0u);
  ASSERT_GT(ref.saved_a.size(), 0u);
  // 3,680 / 3,300: the 4 GB sizes; 2,700: the pool holds the prefix + a streaming row or a complete-file row, so the
  // complete file must give the prefix pages back.
  for (const char* mb : {"3680", "3300", "2700"}) {
    Out o;
    serve(mb, &o, &gb, &fitted);
    EXPECT_EQ(fitted, 1) << mb;
    EXPECT_TRUE(o.a == ref.a) << mb << ": complete-file audio differs from the standard sizes";
    EXPECT_TRUE(o.b == ref.b) << mb << ": complete-file audio differs from the standard sizes";
    EXPECT_TRUE(o.saved_a == ref.saved_a) << mb << ": saved-voice audio differs from the standard sizes";
    if (std::string(mb) == "2700") EXPECT_GE(gb, 1) << "the complete file did not give the prefix pages back";
  }
}
}  // namespace

// kt2n_voice_artifact_lm_prefix_ids: a negative value (never a positive status code) when the artifact cannot be read.
TEST(Runtime, VoiceArtifactPrefixIdsNegativeOnError) {
  EXPECT_LT(kt2n_voice_artifact_lm_prefix_ids(nullptr, 0), 0);
  const std::string garbage = "not a saved voice";
  EXPECT_LT(kt2n_voice_artifact_lm_prefix_ids(garbage.data(), garbage.size()), 0);
}

// The engine's event fd (engine.h): an eventfd on Linux; Windows has none, so kt2n_event_fd returns -1 there and
// callers wait in kt2n_poll.
TEST(Runtime, EventFdPerPlatform) {
  Eng eng;
#if defined(_WIN32)
  EXPECT_EQ(kt2n_event_fd(eng.e), -1);
#else
  EXPECT_GE(kt2n_event_fd(eng.e), 0);
#endif
  EXPECT_EQ(kt2n_event_fd(nullptr), -1);
}
