// The native engine runtime behind include/kt2n/engine.h.
//
// Threads (sized for 4 physical cores):
//   text pool (threads_text)   normaliser/text plan front end: plan + normalize segments, tokens, budgets
//   conductor (1)              the reference owner loop: collect renders, DW gate,
//                              admit, LM steps (LM pump, run-ahead), T9 / D / groups / async / finals,
//                              publication accounting (DeliveryClock, prefetch)
//   post pool (threads_post)   post-processing ResponsePost -> StartupHold -> level_publication -> PCM16/WAV bytes
// The conductor makes the reference decisions (policy.h); renders run on the render backend's streams
// while the LM keeps decoding (execution overlap only).
#include "build_flags.h"
#include <cuda_runtime.h>
#if defined(_WIN32)
#include "kt2n/platform.h"
#include "kt2n/posix_compat.h"
#else
#include <pthread.h>
#include <sched.h>
#include <sys/eventfd.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cassert>
#include <cctype>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <malloc.h>

#include "backend.h"
#include "kt2n/cuda_caps.h"
#include "kt2n/prefill.h"
#include "kt2n/engine.h"
#include "kt2n/post/encoder.h"
#include "kt2n/post/post.h"
#include "kt2n/post/response.h"
#include "kt2n/sha256.h"
#include "policy.h"
#include "text_front.h"
#include "json.h"
#include "voice_blob.h"
#include "voice_cache.h"
#if KT2N_ENGINE_WITH_VOICE
#include "clone_weights.h"
#include "voice_service.h"
#endif

// Lab switches for diagnosis (low-memory fit; each default on): KT2N_FIT_<NAME>=0 turns one fitted-size measure off.
// The fit's estimate does not follow, so a switch can make create fail on a GPU the fit just fills.
static bool fit_on(const char* k) {
  const char* v = std::getenv(k);
  return !(v && *v == '0');
}

// Lab diagnostic (KT2N_DEBUG_MEM=1): device memory in use (all processes) after each create phase, to stderr.
static void debug_mem(const char* phase) {
  static const bool on = [] {
    const char* v = std::getenv("KT2N_DEBUG_MEM");
    return v && *v == '1';
  }();
  if (!on) return;
  size_t free_b = 0, total_b = 0;
  if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess)
    std::fprintf(stderr, "[mem] %-28s used %zu MiB\n", phase, (total_b - free_b) >> 20);
}

// Windows display-driver model (WDDM: Windows, and Linux under WSL2, where the GPU is /dev/dxg): free device memory
// can fall by more than an allocation once the GPU writes it. Measured on an RTX 3050 Laptop GPU under WSL2: 415 MiB
// more once the process has written 2 GiB, given back when freed and the same on every pass; a GTX 1650 under WSL2
// shows none. The low-memory fit measures it before sizing: 128 MiB blocks are allocated and zeroed until free memory
// is within a block of `keep`, the drop beyond the bytes allocated is the extra, then the blocks are freed. Elsewhere
// (Linux without WSL2, Windows TCC) there is no probe and the result is 0.
static uint64_t wddm_written_extra(uint64_t keep, int device) {
#if defined(_WIN32)
  int tcc = 0;
  if (cudaDeviceGetAttribute(&tcc, cudaDevAttrTccDriver, device) != cudaSuccess || tcc != 0) return 0;
#else
  (void)device;
  if (access("/dev/dxg", F_OK) != 0) return 0;
#endif
  const size_t block = size_t(128) << 20;
  size_t f0 = 0, f = 0, total = 0;
  if (cudaMemGetInfo(&f0, &total) != cudaSuccess) return 0;
  cudaStream_t s = nullptr;
  if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) return 0;
  std::vector<void*> blocks;
  f = f0;
  while (f > keep + block && blocks.size() < 1024) {
    void* p = nullptr;
    if (cudaMalloc(&p, block) != cudaSuccess) break;
    blocks.push_back(p);
    if (cudaMemsetAsync(p, 0, block, s) != cudaSuccess || cudaStreamSynchronize(s) != cudaSuccess ||
        cudaMemGetInfo(&f, &total) != cudaSuccess)
      break;
  }
  const uint64_t written = uint64_t(blocks.size()) * block, drop = f0 > f ? f0 - f : 0;
  for (void* p : blocks) cudaFree(p);
  cudaStreamDestroy(s);
  (void)cudaGetLastError();
  return drop > written + (uint64_t(16) << 20) ? drop - written : 0;
}

namespace kt2n::engine {
namespace pol = kt2n::engine::policy;

namespace {

// Timed condition waits. Release builds wait on the steady clock (pthread_cond_clockwait); the TSan build
// (KT2N_TSAN_COMPAT, GCC 11's TSan does not intercept pthread_cond_clockwait and then reports false races)
// waits on the system clock through pthread_cond_timedwait, which it does intercept.
template <class Rep, class Period>
void cv_wait_for(std::condition_variable& cv, std::unique_lock<std::mutex>& lk, std::chrono::duration<Rep, Period> d) {
#if KT2N_TSAN_COMPAT
  cv.wait_until(lk, std::chrono::system_clock::now() + d);
#else
  cv.wait_for(lk, d);
#endif
}
template <class Rep, class Period, class Pred>
bool cv_wait_for(std::condition_variable& cv, std::unique_lock<std::mutex>& lk, std::chrono::duration<Rep, Period> d,
                 Pred pred) {
#if KT2N_TSAN_COMPAT
  return cv.wait_until(lk, std::chrono::system_clock::now() + d, pred);
#else
  return cv.wait_for(lk, d, pred);
#endif
}

int64_t mono_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return int64_t(ts.tv_sec) * 1000000000ll + ts.tv_nsec;
}
double mono_s() { return double(mono_ns()) / 1e9; }

std::string jnum(double v) {
  char b[64];
  std::snprintf(b, sizeof b, "%.3f", v);
  return b;
}

// Custom voices that can be loaded at the same time, for the LM voice store and the render voice table: the
// engine's inline LRU + a caller's saved-voice cache (default 32,
// KT2N_CUSTOM_VOICE_CACHE_ENTRIES in the process environment) + 8 in transit (consent voices, loads racing an eviction). Earlier
// both stores held 32, so a server with a full saved-voice cache refused its first inline clone. Evicted voices
// give their space back (release_voices), so this bounds what is live at once, not a lifetime total.
int custom_voice_headroom(int inline_lru) {
  int cache = 32;
  if (const char* v = std::getenv("KT2N_CUSTOM_VOICE_CACHE_ENTRIES")) cache = std::max(1, std::atoi(v));
  return std::max(1, inline_lru) + cache + 8;
}

// Rows the engine serves at once: LM decode rows, render-state slots and the render pool's per-row pages.
// Every seat gets its own row while the GPU can hold them, as before: the release's 10 / 20 seats are unchanged
// (this function returns max_streams there and never queries the GPU). With more seats than that, the rows are
// capped and the admitted requests beyond them wait in admission, in arrival order; that wait counts in their
// first audio.
// Before this cap, 300 seats sized the render pool at 40 pages x 300 = 12,000
// pages x 16.5 MB = 198 GB and engine create failed in cudaMalloc(render arena).
// KT2N_ENGINE_ROWS sets the cap (capacity only, never a timing decision). Otherwise, above 32 seats, the cap is
// the smaller of
//  - the throughput knee, 32 rows: prior measurements gave the same
//    chars/s at 32 and 48 concurrent rows (3,738-3,837 vs 3,807-3,836) but 0 underruns at 32 and underruns in
//    351-411 of 530 requests (38-65 s of stalls) at 48. Past the knee more rows only add stalls; queued requests
//    wait for a row instead, before their first audio;
//  - the GPU memory fit: per row one LM K/V slot (max_ctx x 112 KiB) + 40 pool pages (kPageBytes, render/ops.h);
//    a fixed 36 GB for everything else (measured: 54.2 GB at 20 seats, minus 22 LM slots and 800 row pages,
//    plus ~4 GB of lazy growth after create); 10% of the card kept free (memory headroom). A 96 GB
//    RTX PRO 6000 fits 49 rows (48 rows measured at 82.3 GB peak); an 80 GB H100 36.
int engine_rows(int max_streams, int device, bool gpu, std::string* why) {
  constexpr int kMaxRows = 60;   // LM rows (rows + 2) and render slots (rows + 4) stop at 64
  constexpr int kKneeRows = 32;  // see above
  int cap = kMaxRows;
  *why = "row limit";
  if (const char* r = std::getenv("KT2N_ENGINE_ROWS"); r && *r && std::atoi(r) > 0) {
    cap = std::min(kMaxRows, std::atoi(r));
    *why = "KT2N_ENGINE_ROWS";
  } else if (max_streams > kKneeRows && gpu) {  // gpu: a real LM or renderer (the CPU stubs hold no device memory)
    int lm_max_ctx = 4096;
    if (const char* v = std::getenv("KT2N_LM_MAX_CTX")) lm_max_ctx = std::max(1024, std::atoi(v) / 64 * 64);
    size_t free_b = 0, total_b = 0;
    if (cudaSetDevice(device) == cudaSuccess && cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
      const double per_row = double(lm_max_ctx) * 114688.0 + 40.0 * 16515072.0;
      const double room = 0.9 * double(total_b) - 36e9;
      const int fit = std::max(4, std::min(kMaxRows, int(room / per_row)));
      cap = std::min(fit, kKneeRows);
      *why = fit < kKneeRows ? "GPU memory fit" : "throughput knee";
    }
  }
  if (max_streams <= cap) *why = "seats";
  return std::min(max_streams, cap);
}
// ------------------------------------------------------------------------------------- errors
struct ErrRow {
  int32_t status;
  int32_t http;
  const char* code;
  const char* type;
};
const ErrRow kErrors[] = {
    {KT2N_OK, 0, "", ""},
    {KT2N_ERR_INVALID_ARGUMENT, 500, "internal_error", "server_error"},
    {KT2N_ERR_INVALID_HANDLE, 500, "internal_error", "server_error"},
    {KT2N_ERR_ABI_MISMATCH, 500, "internal_error", "server_error"},
    {KT2N_ERR_AGAIN, 0, "", ""},
    {KT2N_ERR_CANCELLED, 0, "", ""},
    {KT2N_ERR_NOT_READY, 503, "service_unavailable", "server_error"},
    {KT2N_ERR_UNSUPPORTED, 500, "internal_error", "server_error"},
    {KT2N_ERR_TOO_SMALL, 0, "", ""},
    {KT2N_ERR_INVALID_REQUEST, 400, "invalid_request_error", "invalid_request_error"},
    {KT2N_ERR_INVALID_INPUT, 400, "invalid_input", "invalid_request_error"},
    {KT2N_ERR_INVALID_VOICE, 400, "invalid_voice", "invalid_request_error"},
    {KT2N_ERR_INVALID_RESPONSE_FORMAT, 400, "invalid_response_format", "invalid_request_error"},
    {KT2N_ERR_INVALID_STREAM_FORMAT, 400, "invalid_stream_format", "invalid_request_error"},
    {KT2N_ERR_REQUEST_TOO_LARGE, 413, "request_too_large", "invalid_request_error"},
    {KT2N_ERR_CUSTOM_VOICE_REALTIME_NOT_SUPPORTED, 400, "custom_voice_realtime_not_supported", "invalid_request_error"},
    {KT2N_ERR_INPUT_ALREADY_CLOSED, 400, "input_already_closed", "invalid_request_error"},
    {KT2N_ERR_INVALID_SESSION_CONFIGURATION, 400, "invalid_session_configuration", "invalid_request_error"},
    {KT2N_ERR_MODEL_NOT_FOUND, 404, "model_not_found", "invalid_request_error"},
    {KT2N_ERR_VOICE_NOT_FOUND, 404, "voice_not_found", "invalid_request_error"},
    {KT2N_ERR_GPU_MEMORY_LIMIT, 413, "gpu_memory_limit_exceeded", "invalid_request_error"},
    {KT2N_ERR_INVALID_AUDIO, 400, "invalid_audio", "invalid_request_error"},
    {KT2N_ERR_INVALID_AUDIO_DURATION, 400, "invalid_audio_duration", "invalid_request_error"},
    {KT2N_ERR_INVALID_AUDIO_SIGNAL, 400, "invalid_audio_signal", "invalid_request_error"},
    {KT2N_ERR_INVALID_REFERENCE_AUDIO, 400, "invalid_reference_audio", "invalid_request_error"},
    {KT2N_ERR_REFERENCE_ALIGNMENT_FAILED, 422, "invalid_reference_audio", "invalid_request_error"},
    {KT2N_ERR_SPEAKER_EMBEDDING_FAILED, 422, "speaker_embedding_failed", "invalid_request_error"},
    {KT2N_ERR_SPEAKER_VERIFICATION_FAILED, 422, "speaker_verification_failed", "invalid_request_error"},
    {KT2N_ERR_CONSENT_PHRASE_MISMATCH, 400, "consent_phrase_mismatch", "invalid_request_error"},
    {KT2N_ERR_UNSUPPORTED_CONSENT_LANGUAGE, 400, "unsupported_consent_language", "invalid_request_error"},
    {KT2N_ERR_CONSENT_REQUIRES_SPEAKER_VERIFICATION, 409, "consent_requires_speaker_verification",
     "invalid_request_error"},
    {KT2N_ERR_VOICE_ARTIFACT_VERSION_MISMATCH, 409, "voice_artifact_version_mismatch", "invalid_request_error"},
    {KT2N_ERR_VOICE_ARTIFACT_REBUILDING, 503, "voice_artifact_rebuilding", "server_error"},
    {KT2N_ERR_CUSTOM_VOICE_CREATION_BUSY, 429, "custom_voice_creation_busy", "rate_limit_error"},
    {KT2N_ERR_CUSTOM_VOICE_SERVICE_ERROR, 503, "custom_voice_service_error", "server_error"},
    {KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE, 503, "custom_voice_service_unavailable", "server_error"},
    {KT2N_ERR_INVALID_REFERENCE_TRANSCRIPT, 400, "invalid_reference_transcript", "invalid_request_error"},
    {KT2N_ERR_INVALID_CONSENT_TRANSCRIPT, 400, "invalid_consent_transcript", "invalid_request_error"},
    {KT2N_ERR_SPEECH_GENERATION_FAILED, 503, "speech_generation_failed", "server_error"},
    {KT2N_ERR_DEADLINE_EXCEEDED, 503, "speech_generation_failed", "server_error"},
    {KT2N_ERR_SERVICE_UNAVAILABLE, 503, "service_unavailable", "server_error"},
    {KT2N_ERR_INTERNAL, 500, "internal_error", "server_error"},
};

void set_err(kt2n_error* err, int32_t status, const std::string& message, const std::string& param = "") {
  if (!err) return;
  const kt2n_error_info* info = kt2n_error_lookup(status);
  err->status = status;
  err->http_status = info->http_status;
  std::snprintf(err->code, sizeof err->code, "%s", info->code);
  std::snprintf(err->param, sizeof err->param, "%s", param.c_str());
  std::snprintf(err->message, sizeof err->message, "%s", message.c_str());
}

// ------------------------------------------------------------------------------------- state
struct HostVoice;

struct Voice {
  std::string id;
  std::vector<std::string> names;  // id + public names
  std::string language = "en";
  int prompt_tokens = 0;
  int prompt_frames = 0;
  int lm_prefix_ids = 0;
  bool hidden = false;
  double peak_gain_db = 0.0;
  std::vector<float> raw_speaker;
  std::atomic<int> lm_voice{-1};  // -1 until the conductor built its LM prefix (saved voices: asynchronously)
  int render_voice = -1;
  int64_t geometry = 0;
  std::atomic<int> refs{0};
  int kind = KT2N_VOICE_BUILTIN;
  std::atomic<bool> evicted{false};
  bool lm_released = false;      // conductor only: its LM store space was given back (release_voices)
  bool render_released = false;  // conductor only: its render table row + prefix pages were given back
  std::vector<uint8_t> artifact;  // saved voices: the one-voice voices.kt2n blob (kt2n_voice_export)
  bool consent_only = false;      // a consent recording's speaker embedding: never speaks
  // Voice cache (LRU / NONE modes only): the record its GPU state is prepared from (kept on the host for the voice's
  // lifetime) and its cache key (-1 = not in the cache yet, -2 = refused: it can never fit the cache budget).
  std::shared_ptr<const HostVoice> host;
  int cache_key = -1;
};

// The inputs of a voice's LM prefix and render table row (voices.kt2n tensors), kept on the host by the voice cache.
struct HostVoice {
  std::vector<int32_t> prompt_token;
  std::vector<float> prompt_feat, embedding, projected;
  std::vector<int32_t> lm_prefix;
  uint64_t bytes() const {
    return prompt_token.size() * 4 + prompt_feat.size() * 4 + embedding.size() * 4 + projected.size() * 4 +
           lm_prefix.size() * 4;
  }
};

struct Job {  // one LM chunk (a segment job in the reference); segment_index = ordinal
  int ordinal = 0;
  int raw_segment = 0;
  std::u32string text;
  std::vector<int32_t> suffix;
  int budget = 200;
  bool charged = false, acknowledged = false;
  bool lm_done = false, done = false;
  bool preview_ready = false;
  int64_t published_samples = 0;
  std::optional<double> first_pub;
};

struct PostItem {
  int job = -1;
  std::vector<float> pcm;
  bool final = false;   // the job's last publication
  bool has_more = false;
  int lane_jobs = 0;    // jobs the lane holds now (SeamContext)
  bool lane_sealed = false;
  bool end = false;     // response complete (after the last job's final)
  int status = 0;       // end with a failure
};

struct Request;

struct EventRec {
  kt2n_event ev{};
  std::string data, name, fields;
  kt2n_error error{};
};

struct Request {
  uint64_t id = 0;
  uint64_t user_data = 0;
  uint32_t flags = 0;
  bool timeline = false;
  Voice* voice = nullptr;
  std::string raw;
  uint64_t salt = 0;
  lm::SamplerSettings sampler;
  double speed = 1.0;
  int format = KT2N_FORMAT_PCM;
  int output = KT2N_OUTPUT_STREAM;
  bool demo = false;  // speech mode DEMO (stream:false)
  int64_t deadline_ns = 0;
  double started_s = 0;

  // Text (conductor-owned after hand-off).
  std::unique_ptr<PlannedRequest> plan;
  std::vector<std::unique_ptr<Job>> jobs;      // conductor only (absorbed from new_jobs)
  std::vector<std::unique_ptr<Job>> new_jobs;  // text thread -> conductor, under e->mu
  size_t jobs_created = 0;                     // under e->mu
  bool c_sealed = false;                       // conductor's snapshot of `sealed` (taken with the jobs)
  size_t next_index = 0;
  int inflight = 0;                // lane.inflight
  size_t segments_normalized = 0;  // planned segments turned into jobs
  bool normalizing = false;
  bool sealed = false;             // every planned segment normalized
  int total_chunks = 0;            // the request's demo chunk count
  int text_status = 0;
  std::string text_error;

  // Lifecycle.
  std::atomic<bool> cancelled{false};
  int cancel_status = KT2N_ERR_CANCELLED;
  std::atomic<bool> release_requested{false};
  std::string fail_reason;  // set by the conductor before the failure's post item (happens-before via q->mu)
  bool in_lanes = false;  // conductor
  bool cache_pinned = false;  // conductor: holds a voice cache pin on cache_key (LRU / NONE modes)
  int cache_key = -1;
  int text_tasks = 0;     // under e->mu: text-pool tasks queued or running for this request
  int live_rows = 0;      // conductor: rows (decoding or rendering) that point at this request
  bool generation_complete = false;  // conductor sent the end item
  std::atomic<bool> terminal{false};  // DONE emitted
  int opening_skipped = 0, kept_tokens = 0;
  // True from create until the startup hold releases (a streamed response with a hold). The post thread clears it,
  // the conductor reads it (arm KT2N_HOLD_URGENT: this request's commits are on its first-audio path).
  std::atomic<bool> awaiting_hold{false};

  // Post side (guarded by mu).
  std::mutex mu;
  std::deque<PostItem> post_q;
  bool post_scheduled = false;
  std::mutex clock_mu;            // DeliveryClock only: the conductor never waits on post processing
  pol::DeliveryClock clock;       // under clock_mu
  std::atomic<int64_t> dropped_samples{0};  // the onset trimmer's dropped samples (post writes, conductor reads)
  std::unique_ptr<post::ResponsePost> rpost;
  std::unique_ptr<post::StartupHold> hold;
  std::optional<post::PeakNorm> peak;
  post::PeakNorm* peak_ptr = nullptr;  // a WS phrase levels with its session's state
#if KT2N_ENGINE_WITH_AV
  std::unique_ptr<post::StreamEncoder> enc;  // mp3 / aac / opus / flac (the reference CLI settings)
#endif
  std::vector<float> whole;  // complete output (stream:false): the whole utterance
  int64_t samples_out = 0;
  int chunk_index = 0;
  bool header_sent = false;
  bool headers_sent = false;
  int post_begun_job = -1;
  // Event identity: own id/seq for an HTTP request; a WS phrase emits as its session.
  uint64_t ev_id = 0;
  std::mutex own_ev_lock;
  uint64_t own_seq = 0;
  std::mutex* ev_lock = &own_ev_lock;
  uint64_t* ev_seq = &own_seq;
  int phrase_index = -1;
  struct Session* session = nullptr;
  bool realtime = false;  // a WS phrase: startup hold pinned 0, T9 start target without the hold
  bool skip_opening = true;  // control skip_opening_silence (DEMO: off)
  bool normalize = true;
  std::optional<bool> emotion_preset;
  int post_cur_job = 0;                         // the consumer reads jobs in order
  std::map<int, std::deque<PostItem>> post_later;  // publications of later jobs, held until their turn
  std::optional<PostItem> post_end;
  bool hold_released_marked = false;
  int hold_chunks = 0;
  // Metrics (conductor writes before the final publication; read by the post stage after it).
  int64_t m_audio_tokens = 0, m_lm_steps = 0, m_segments = 0;
  std::optional<double> hold_pinned;  // control_json startup_hold_seconds (a WS phrase: 0)
};

// A realtime WebSocket session (the reference socket input rules + sentence batcher): the text plan WsInputGate
// and WsPlanner, one generation slot; each phrase is an internal request emitting as the session.
struct Session {
  bool normalize = true;
  uint64_t id = 0;
  uint64_t user_data = 0;
  uint32_t flags = 0;
  Voice* voice = nullptr;
  std::string request_id;
  uint64_t salt = 0;
  bool seeded = false;
  lm::SamplerSettings sampler;
  double speed = 1.0;
  int format = KT2N_FORMAT_PCM;
  int64_t deadline_ns = 0;
  double opened_s = 0;
  std::mutex ev_lock;
  uint64_t seq = 0;
  // Planner state (under kt2n_engine::sess_mu).
  textplan::WsInputGate gate;
  textplan::WsPlanner planner;
  std::atomic<bool> phrase_running{false};  // set by the post thread without sess_mu (lock order)
  int phrases = 0;
  double wake_ms = 0;  // next planner step (ms since opened); 0 = now
  std::atomic<bool> dirty{true};
  bool planner_done = false;
  bool terminal = false;
  bool cancelled = false;
  int cancel_status = KT2N_ERR_CANCELLED;
  std::vector<uint64_t> phrase_ids;
  std::optional<post::PeakNorm> peak;  // one streaming peak-norm state per session, across phrases
  std::atomic<bool> ended{false};  // set with `terminal` (read by the conductor without sess_mu)
  int cache_keep_key = -1;          // conductor only: the voice cache key this session keeps (VoiceCache::keep)
};

void session_phrase_done(kt2n_engine* e, Request* q);

// A queued clone build (kt2n_voice_build -> cloning VoiceService on a voice thread).
struct BuildJob {
  uint64_t id = 0;
  uint64_t user_data = 0;
  uint32_t flags = 0;
  std::vector<uint8_t> audio;
  std::string audio_format;
  std::vector<float> pcm;
  int pcm_rate = 24000;
  std::vector<float> consent;
  std::string transcript, language, identity_key;
  int64_t deadline_ns = 0;
  double queued_s = 0;
  std::atomic<bool> cancelled{false};
  bool terminal = false;
  std::mutex ev_lock;
  uint64_t seq = 0;
};

struct Row {
  uint64_t key = 0;
  Request* req = nullptr;
  int job = 0;
  std::vector<int32_t> audio;  // S3Gen tokens (vocoder view)
  pol::OpeningSkip skip;
  bool skip_armed = false;
  pol::CommitState c;
  bool finished = false;
  lm::Finish finish = lm::Finish::kNone;
  int lm_steps = 0;
  int prompt_ids = 0;  // LM prompt (voice prefix + suffix): the low-memory fit's context check
  bool in_active = true;
  bool final_launched = false;
  bool closed = false;
  uint64_t next_seq = 0, pub_seq = 0;
  struct Done {
    std::vector<float> pcm;
    bool final = false;
    bool failed = false;  // the render failed: nothing is published, but the row's order advances
    int commit_tokens = 0;
    double gpu_ms = 0, launch_ms = 0, launched_s = 0;
    size_t batch = 0;
    int graph_nodes = 0;
    bool graph_captured = false;
  };
  std::map<uint64_t, Done> reorder;  // seq -> finished render output, published in seq order
  int async_inflight = 0;
  double admitted_s = 0;
  bool first_token_marked = false;
  int t9_deferred_turns = 0;
  bool noise_extended = false;
  std::vector<int32_t> lm_head;  // first 64 generated ids (gpu.n_opening_skip lm_head)
};

struct Inflight {  // what the conductor keeps of a launched call (not its token lists)
  struct RowRef {
    uint64_t row = 0;
    uint64_t seq = 0;
    int commit_tokens = 0;
  };
  std::vector<RowRef> rows;
  std::vector<bool> finals;
  std::vector<bool> asyncs;
  double launched_s = 0;
  render::StreamRole stream = render::StreamRole::kSUrgent;
};

}  // namespace
}  // namespace kt2n::engine

// =====================================================================================================
struct kt2n_engine {
  // Config.
  kt2n::engine::policy::Levers lv;
  uint32_t flags = 0;
  uint64_t device_reserved_bytes = 0;
  // Low-memory fit (memory_profile LOW; engine.h kt2n_low_memory_limits). fit_ctx > 0: the LM context was fitted
  // below 4,096 (admission refuses a segment whose prompt + budget does not fit, and a row the slot stops fails);
  // fit_frames > 0: the vocoder's complete-file frame limit was fitted (DEMO segments beyond it are refused).
  kt2n_low_memory_limits fit{};
  int fit_ctx = 0;
  int fit_frames = 0;
  // KT2N_MEMORY_PROFILE_AUTO: the profile it chose ("" = not automatic): "default" or "low".
  std::string auto_profile;
  // The fit left no room for the cloning model, so the model file's clone groups were not loaded (kt2n_voice_build).
  bool cloning_off_fitted = false;
  int max_streams = 10;
  int row_cap = 10;  // rows served at once (engine_rows); <= max_streams
  std::string row_cap_why = "seats";
  std::unique_ptr<kt2n::engine::LmBackend> lm;
  std::unique_ptr<kt2n::engine::RenderBackend> render;
  std::unique_ptr<kt2n::engine::TextFront> text;
  std::vector<std::unique_ptr<kt2n::engine::Voice>> voices;  // append-only; id = index + 1 (under voices_mu)
  std::mutex voices_mu;
  std::optional<kt2n::voice::Catalogue> catalogue;  // built-ins (cloning load gate, nearest built-in voice)
  std::deque<std::function<void()>> admin_q;         // conductor jobs (owner loop step 2), under mu

  // Shared state.
  std::mutex mu;
  std::condition_variable cv;  // conductor wake-up
  std::unordered_map<uint64_t, std::unique_ptr<kt2n::engine::Request>> requests;
  std::deque<kt2n::engine::Request*> incoming;  // submitted, waiting for the conductor
  uint64_t next_id = 1;
  std::atomic<bool> stop{false};
  std::atomic<int> sharing{1};      // startup_sharing()
  std::atomic<int> steady{0};       // d_steady_requests()

  // Events.
  std::mutex ev_mu;
  std::condition_variable ev_cv;
  std::condition_variable ev_drained;
  size_t pollers = 0;  // entered polls, under ev_mu; destroy drains them before freeing ev_cv
  std::deque<kt2n::engine::EventRec*> events;
  std::map<std::pair<uint64_t, uint64_t>, std::unique_ptr<kt2n::engine::EventRec>> outstanding;
  int efd = -1;
#if defined(_WIN32)
  kt2n::platform::TimerResolution timer_resolution;  // 1 ms timer while the engine lives (short timed waits)
#endif

  // Pools.
  std::mutex text_mu, post_mu;
  std::condition_variable text_cv, post_cv;
  std::deque<std::function<void()>> text_q;
  std::deque<kt2n::engine::Request*> post_q;
  std::vector<std::thread> threads;

  // Conductor-owned.
  std::vector<kt2n::engine::Request*> lanes;
  size_t lane_cursor = 0;
  size_t lane_cursor_cont = 0;  // capacity mode: the round-robin cursor over continuing requests
  bool capacity_order = false;  // KT2N_CAPACITY_ORDER=1, capacity mode only (seats > rows): continuing requests first
  std::unordered_map<uint64_t, std::unique_ptr<kt2n::engine::Row>> rows;
  std::vector<uint64_t> active;
  std::unordered_map<uint64_t, kt2n::engine::Inflight> inflight;
  std::deque<std::pair<uint64_t, double>> dw_queue;  // (row key, queued at)
  std::optional<double> last_whole_render;
  uint64_t next_row = 1, next_call = 1;
  int inflight_segments = 0;

  // Clone builds (voice threads; the conductor grants each build its turn: clone-interleave gate).
#if KT2N_ENGINE_WITH_VOICE
  std::unique_ptr<kt2n::voice::VoiceBuilder> vbuilder;
  std::unique_ptr<kt2n::voice::VoiceService> vservice;
  // Optional host executor; an empty executor selects the concrete VoiceService.
  std::function<kt2n::voice::VoiceBuildOutcome(const kt2n::voice::VoiceBuildRequest&)> voice_build_exec;
  // identity_key -> voice id: the reference's in-use-safe LRU (inline_voice_lru entries, in-use voices never evicted;
  // cloning InlineVoiceIndex). Constructed with the configured size in kt2n_engine_create.
  std::unique_ptr<kt2n::voice::InlineVoiceIndex> inline_index;
#endif
  std::mutex build_mu;
  std::condition_variable build_cv;
  std::deque<std::shared_ptr<kt2n::engine::BuildJob>> build_q;
  std::unordered_map<uint64_t, std::shared_ptr<kt2n::engine::BuildJob>> builds;
  // Registered voices awaiting conductor initialization, under mu. Shared owners
  // survive worker return and let fatal closure finish every accepted build.
  std::unordered_map<uint64_t, std::vector<std::shared_ptr<kt2n::engine::BuildJob>>> pending_voice_builds;
  int builds_running = 0;
  bool build_waiting = false;    // a voice thread waits for the gate
  double build_wait_since = 0;
  bool build_go = false;
  std::optional<double> last_build_done_s;
  int clone_slots = 4, clone_queue = 16;
  double clone_timeout_s = 120.0;

  // Realtime sessions (session thread).
  std::mutex sess_mu;
  std::condition_variable sess_cv;
  std::unordered_map<uint64_t, std::unique_ptr<kt2n::engine::Session>> sessions;

  // Stats.
  std::atomic<uint64_t> done{0}, failed{0}, cancelled{0}, renders{0}, render_rows{0}, lm_steps{0};
  std::atomic<int> lm_rows{0}, render_inflight{0}, active_requests{0}, queued_requests{0};
  std::atomic<uint64_t> previews_due{0};     // 1-step preview condition met (no default path fires it)
  std::atomic<uint64_t> lm_backpressure{0};  // LM submits refused (slot release lag); the job retries
  std::atomic<uint64_t> render_backpressure{0};  // open_row refused (render slot / pool pages); the job retries
  // Low-memory fit, smallest tier (fit.streaming_budget_tokens > 0): streaming chunks beyond the fitted budget are
  // re-split into pieces that fit instead of being refused (normalize_into_jobs; KT2N_FIT_SPLIT=0 refuses them).
  bool fit_split = false;
  std::atomic<uint64_t> fit_split_chunks{0};
  // custom_voice_max_prefix_ids as the low-memory profile applied it (0 = no bound): kt2n_voice_load refuses longer.
  int custom_prefix_limit = 0;
  bool poll = true;                      // conductor poll mode (KT2N_CONDUCTOR_POLL=0 turns it off; execution only)
  // Arm (execution only, off by default): a commit group that mixes second commits (commit 1, which the startup hold
  // waits for) with later commits is launched as two calls, the commit-1 rows on S_urgent and the rest on S_bulk,
  // instead of one call that stream_for() puts on S_urgent because of its most urgent row. Same rows, same commit
  // boundaries, same turn. KT2N_SPLIT_URGENT=1.
  bool split_urgent = false;
  std::atomic<uint64_t> split_urgent_calls{0};
  // Option (execution only, off by default; found in review under the release-candidate settings): a commit of a request whose startup
  // hold has not released yet (its first audio waits for it) runs on S_urgent even when its commit index is >= 2.
  // stream_for() sends commit 2+ to S_bulk; with the release-candidate settings (decode priority 2 above S_bulk, bulk call merging) those
  // renders took 156-190 ms wall p50 for 39-51 ms of GPU work in one measured window, and the ~4% of requests whose hold
  // waited for them set the N=20 p99 (270-299 ms vs 176-201 ms with the reference defaults). A group that mixes such
  // rows with later commits of released requests (and has no commit-0/1 row) is launched as two calls, as in
  // split_urgent. Same rows, same commit boundaries, same turn. KT2N_HOLD_URGENT=1 (the render service also
  // prewarms the urgent twins of single-row steady graphs, so the promoted calls capture no graph live).
  // Level 2 (KT2N_HOLD_URGENT=2, measured separately): also the FINAL flush of a row whose request is
  // still held (a one- or two-word input whose hold can only release on its final render).
  int hold_urgent = 0;
  // KT2N_BUFFER_URGENT_MS (execution-only arm, 0 = off): a commit or final of a playing stream whose delivery clock
  // (DeliveryClock) holds less than this much audio runs on S_urgent.
  double buffer_urgent_ms = 0;
  std::atomic<uint64_t> buffer_urgent_rows{0}, buffer_urgent_finals{0};
  // KT2N_BUFFER_ROLE=aux (two-level option): those commits go to S_aux, which the renderer then runs at a
  // middle priority (KT2N_RENDER_AUX_PRIO_LEVEL), instead of S_urgent; finals stay on their natural stream.
  bool buffer_to_aux = false;
  std::atomic<uint64_t> buffer_mid_calls{0};
  std::atomic<uint64_t> hold_urgent_calls{0}, hold_urgent_rows{0}, hold_urgent_splits{0}, hold_urgent_finals{0};
  // Default-off execution experiment: briefly let an already-launched bulk render compete at the LM's priority.
  // It never changes commits, T9, admission order, the startup hold, or any model setting.
  bool lm_render_yield = false;
  int lm_render_yield_base_priority = 0;
  bool lm_render_yield_active = false;  // conductor only
  uint64_t lm_render_yield_call = 0;
  double lm_render_yield_since = 0, lm_render_yield_cooldown = 0, lm_render_yield_checked = 0;
  std::atomic<uint64_t> lm_render_yield_episodes{0}, lm_render_yield_switches{0};
  int text_nice = 0;                     // nice of the text threads (KT2N_TEXT_NICE; experimental, execution only)
  std::atomic<uint64_t> poll_idle{0};    // poll-mode passes that found no finished LM step
  int render_voice_errors = 0;               // render add_voice failures at create (stats)
  std::atomic<long> voice_store_full{0};     // loaded / built voices refused by a full LM or render voice store
  std::string render_voice_error;
  std::atomic<uint64_t> turn_ns{0}, turns{0}, turn_ns_max{0};  // conductor CPU time per turn (host on the token path)
  std::atomic<uint64_t> turns_over_1ms{0}, turns_over_5ms{0}, turn_max_decide_ns{0};
  // Render calls per turn (diagnostic): turns that launched commit groups, the groups, and the groups beyond
  // the first that went to the same stream in the same turn (could have been one bigger call); the same for finals.
  std::atomic<uint64_t> commit_turns{0}, commit_groups{0}, commit_same_stream_extra{0}, commit_extra_rows{0};
  std::atomic<uint64_t> commit_mixed_groups{0}, commit_mixed_later_rows{0};  // commit-1 rows + later rows in one group
  std::atomic<uint64_t> final_turns{0}, final_calls{0}, final_extra_calls{0};
  std::atomic<int> turn_max_rows{0};
  // LM step interval while rows stay live (diagnostic: the LM's ms per step under the engine's real load, prefills and
  // GPU sharing included, idle gaps excluded), split by rows per step: 1-4, 5-8, 9-12, 13-16, 17-20, 21+.
  int64_t lm_prev_ev_ns = 0;  // conductor only
  std::atomic<uint64_t> lm_iv_ns[8]{}, lm_iv_steps[8]{};  // 1-4 ... 17-20, 21-32, 33-48, 49+ rows
  std::atomic<bool> backend_failed{false};  // latched under mu with request admission
  std::string fatal;  // under mu; diagnostic only
  std::atomic<bool> release_pending{false};
  std::atomic<bool> voice_sweep{false};       // an evicted voice may have no request left: release_voices()
  std::atomic<long> lm_voices_released{0};    // LM voice store ranges given back (LM release_voice)
  std::atomic<long> render_voices_released{0};  // render table rows + prefix pages given back (decoder remove_voice)
  // Voice cache (kt2n_voice_cache_mode LRU / NONE; both null in ALL mode, which keeps the original code path).
  int voice_cache_mode = KT2N_VOICE_CACHE_ALL;
  std::unique_ptr<kt2n::engine::VoiceLoader> vloader;  // declared before vcache: destroyed after it
  std::unique_ptr<kt2n::engine::VoiceCache> vcache;
  std::vector<kt2n::engine::Session*> cache_kept_sessions;  // conductor only: sessions keeping a voice (cache keep)
};

namespace kt2n::engine {
namespace {

using Engine = kt2n_engine;

// ------------------------------------------------------------------------------------- voice cache
// The GPU side of a voice cache load (LRU / NONE modes): render table row + prefix, then the LM prefix K/V, both queued
// without a host wait; the voice is resident once the LM's build completes. Conductor thread only (VoiceCache::pump).
class CacheLoader final : public VoiceLoader {
 public:
  explicit CacheLoader(Engine* e) : e_(e) {}
  int add(Voice* v) {
    slots_.push_back(Slot{v});
    return int(slots_.size()) - 1;
  }
  // A retired voice with no GPU state left can never load again: drop its host tensors (the cache's lock is held,
  // and only the loader reads the record after registration).
  void forget(int key) override { slots_[size_t(key)].v->host.reset(); }
  int start_load(int key, std::string* err) override {
    Slot& s = slots_[size_t(key)];
    s.stage = 0;
    s.rv = s.lv = -1;
    return poll_load(key, err);
  }
  int poll_load(int key, std::string* err) override {
    Slot& s = slots_[size_t(key)];
    if (!s.v->host) {  // retired voices never load again (the cache does not start them); defensive
      if (err) *err = "voice was released";
      return kFailed;
    }
    const HostVoice& h = *s.v->host;
    if (s.stage == 0) {
      const int rv = e_->render->add_voice(h.prompt_token, h.prompt_feat, h.embedding, err);
      if (rv < 0) return kFailed;
      s.rv = rv;
      s.stage = 1;
    }
    if (s.stage == 1) {
      const int rc = e_->render->prepare_prefix(s.rv, err);
      // The pool's free pages are held by open rows' reservations: they come back when those rows finish (the cache's
      // page budget already counts this voice), so wait instead of dropping other voices.
      if (rc == kRenderBackpressure) return kPending;
      if (rc != 0) return kFailed;
      s.stage = 2;
    }
    if (s.stage == 2) {
      int lv = e_->lm->add_voice_async(h.projected, h.lm_prefix, err);
      if (lv == -2 && e_->lm->live_rows() == 0) {
        // A dropped voice's store range is freed once the LM releases the slots of its last rows, which happens in
        // the LM's pump; with no row live the conductor is not pumping it, so let it catch up and try again. With no
        // row live (pending or running) the pump has no row to report on, so it returns no events.
        std::vector<lm::TokenEvent> none;
        std::string perr;
        const int pr = e_->lm->pump(&none, false, &perr);
        if (!none.empty())
          std::fprintf(stderr, "[kt2n voice cache] LM pump without live rows returned %zu events (dropped)\n",
                       none.size());
        assert(none.empty());
        if (pr >= 0 && none.empty()) lv = e_->lm->add_voice_async(h.projected, h.lm_prefix, err);
      }
      if (lv == -2) return kWaitRoom;
      if (lv < 0) return kFailed;
      s.lv = lv;
      s.stage = 3;
    }
    if (s.stage == 3) {
      const int r = e_->lm->voice_ready(s.lv);
      if (r == 0) return kPending;
      if (r < 0) {
        if (err) *err = "LM voice prefix build failed";
        return kFailed;
      }
      // Resident only once the decoder prefix is complete too, so the voice's first rows find its GPU state as a
      // preloaded voice would (no prefix build still running beside their first renders).
      const int p = e_->render->prefix_ready(s.rv);
      if (p == 0) return kPending;
      if (p < 0) {
        if (err) *err = "render voice prefix build failed";
        return kFailed;
      }
      s.v->render_voice = s.rv;
      s.v->lm_voice = s.lv;
      s.stage = 4;
    }
    return kDone;
  }
  // Shutdown (VoiceCache::drain): only a load whose LM prefix build is queued must finish first; earlier stages have
  // queued nothing that must complete (a render prefix build in flight makes the unload busy instead).
  bool load_in_flight(int key) override { return slots_[size_t(key)].stage == 3; }
  // Only for voices no row uses (unpinned: their requests' rows are closed and dropped) or never made resident.
  int unload(int key, std::string* err) override {
    Slot& s = slots_[size_t(key)];
    if (s.rv >= 0) {
      const int rc = e_->render->remove_voice(s.rv, err);
      if (rc == -2) return kBusy;  // a closing row's call or the prefix build
      if (rc == 0)
        e_->render_voices_released++;
      else  // not a voice the renderer holds (nothing to give back); the cache's accounting is not affected
        std::fprintf(stderr, "[kt2n voice cache] render remove_voice(%d) failed: %s\n", s.rv, err ? err->c_str() : "");
      s.rv = -1;
    }
    if (s.lv >= 0) {
      // LM frees the range once no slot reads it (or its build is done).
      if (e_->lm->release_voice(s.lv) == 0) e_->lm_voices_released++;
      s.lv = -1;
    }
    s.v->lm_voice = -1;
    s.v->render_voice = -1;
    s.stage = 0;
    return 0;
  }

 private:
  struct Slot {
    Voice* v = nullptr;
    int stage = 0;
    int rv = -1, lv = -1;
  };
  Engine* e_;
  std::vector<Slot> slots_;  // by cache key
};

// Voice cache cost of a voice: LM store range, renderer prefix pages (upper bound) and voice table row.
constexpr uint64_t kVoiceTableRowBytes = 400u << 10;  // render arena per-voice tables (~0.4 MiB, estimate)
VoiceCost voice_cost(int prompt_tokens, size_t lm_prefix_ids) {
  VoiceCost c;
  c.lm_bytes = (uint64_t(lm_prefix_ids) * 114688 + 255) / 256 * 256;  // [2][28][8][128] bf16 per prefix id
  c.pages = (2 * prompt_tokens + 63) / 64 + 1;                       // L7Render::prefix_pages_bound
  c.page_bytes = uint64_t(c.pages) * render::kPageBytes;
  c.table_bytes = kVoiceTableRowBytes;
  return c;
}

// Registers a voice with the cache (conductor, or create before the threads start). Returns its key.
int cache_add(Engine* e, Voice* v, std::shared_ptr<const HostVoice> host) {
  const VoiceCost c = voice_cost(int(host->prompt_token.size()), host->lm_prefix.size());
  const uint64_t hb = host->bytes();
  v->host = std::move(host);
  const int k = static_cast<CacheLoader*>(e->vloader.get())->add(v);
  const int k2 = e->vcache->add(c, !v->hidden, hb);
  (void)k2;  // both append one entry per call
  v->cache_key = k;
  return k;
}

// Admission gate: the voice's LM prefix and render state are usable by new rows.
bool voice_usable(Engine* e, const Voice* v) {
  if (v->lm_voice.load() < 0) return false;
  return !e->vcache || (v->cache_key >= 0 && e->vcache->ready(v->cache_key));
}

// LRU / NONE: a request in the lanes pins its voice (once registered with the cache) ...
void cache_pin(Engine* e, Request* q) {
  if (!e->vcache || q->cache_pinned || q->generation_complete) return;
  const int k = q->voice->cache_key;
  if (k < 0) return;
  e->vcache->pin(k);
  q->cache_pinned = true;
  q->cache_key = k;
  // A realtime session keeps its voice between phrases (NONE mode would drop it after every phrase); ended with the
  // session (cache_end_session_keeps). Kept voices still make room for other voices' loads.
  if (Session* s = q->session; s && s->cache_keep_key < 0 && !s->ended.load()) {
    e->vcache->keep(k);
    s->cache_keep_key = k;
    e->cache_kept_sessions.push_back(s);
  }
}

// Conductor: sessions that ended give their voice keep back.
void cache_end_session_keeps(Engine* e) {
  auto& v = e->cache_kept_sessions;
  for (auto it = v.begin(); it != v.end();) {
    if ((*it)->ended.load()) {
      e->vcache->unkeep((*it)->cache_keep_key);
      it = v.erase(it);
    } else {
      ++it;
    }
  }
}

// ... until its generation is complete and none of its rows is left (or it is freed).
void cache_unpin(Engine* e, Request* q, bool force = false) {
  if (!q->cache_pinned || (!force && (!q->generation_complete || q->live_rows > 0))) return;
  e->vcache->unpin(q->cache_key);
  q->cache_pinned = false;
}

// ------------------------------------------------------------------------------------- events
void push_event(Engine* e, EventRec* r) {
  {
    std::lock_guard<std::mutex> lk(e->ev_mu);
    e->events.push_back(r);
  }
  e->ev_cv.notify_all();
  const uint64_t one = 1;
  if (e->efd >= 0) (void)!write(e->efd, &one, sizeof one);
}

EventRec* new_event(Request* q, int kind) {
  auto* r = new EventRec();
  r->ev.kind = kind;
  r->ev.user_data = q->user_data;
  r->ev.at_ns = mono_ns();
  r->ev.segment_index = -1;
  r->ev.phrase_index = q->phrase_index;
  r->ev.sample_begin = -1;
  r->ev.engine_chunk_index = -1;
  return r;
}

// Assigns the per-id sequence number and queues the event under the id's emit lock, so seq order
// is queue order even when the conductor and a post thread emit for the same id.
void emit(Engine* e, Request* q, EventRec* r) {
  std::lock_guard<std::mutex> lk(*q->ev_lock);
  if (q->terminal.load() && r->ev.kind != KT2N_EVENT_DONE && r->ev.kind != KT2N_EVENT_PHRASE_DONE) {
    delete r;
    return;
  }
  r->ev.id = q->ev_id;
  r->ev.seq = (*q->ev_seq)++;
  push_event(e, r);
}

void timeline(Engine* e, Request* q, const char* name, const std::string& fields, int segment = -1) {
  if (!q->timeline) return;
  EventRec* r = new_event(q, KT2N_EVENT_TIMELINE);
  r->ev.segment_index = segment;
  r->name = name;
  r->fields = "{" + fields + "}";
  emit(e, q, r);
}

// ------------------------------------------------------------------------------------- post
void schedule_post(Engine* e, Request* q) {
  // q->mu held by the caller.
  if (q->post_scheduled) return;
  q->post_scheduled = true;
  {
    std::lock_guard<std::mutex> lk(e->post_mu);
    e->post_q.push_back(q);
  }
  e->post_cv.notify_one();
}

bool encoded_format(int f);
const char* format_name(int f);

void emit_audio(Engine* e, Request* q, const std::string& bytes, int64_t samples, bool final) {
  // Post thread.
  if (!q->headers_sent) {
    // Response metadata before the first audio (engine.h KT2N_EVENT_HEADERS).
    q->headers_sent = true;
    EventRec* h = new_event(q, KT2N_EVENT_HEADERS);
    h->name = "headers";
    h->fields = std::string("{\"sample_rate\":24000,\"format\":\"") + format_name(q->format) +
                "\",\"speech_mode\":" + (q->demo ? "\"demo\"" : "null") +
                (q->output == KT2N_OUTPUT_COMPLETE ? ",\"complete_bytes\":" + std::to_string(bytes.size()) : "") + "}";
    emit(e, q, h);
  }
  EventRec* r = new_event(q, KT2N_EVENT_AUDIO);
  r->data = bytes;
  r->ev.sample_begin = q->samples_out;
  r->ev.samples = int32_t(samples);
  r->ev.engine_chunk_index = q->chunk_index++;
  r->ev.is_final_chunk = final ? 1 : 0;
  // engine.h opening_skipped_tokens: the reference opening-skipped token count, final before
  // the first audio (the conductor sets it when segment 0's opening closes, before the vocoder sees a token;
  // post_push gave this thread happens-before). A caller's realtime reserve can read it at the first chunk (the
  // skip-aware per-listener term). Set on every AUDIO event.
  r->ev.opening_skipped_tokens = q->opening_skipped;
  q->samples_out += samples;
  emit(e, q, r);
}

void emit_done(Engine* e, Request* q, int status, const std::string& message) {
  // Post thread.
  if (q->terminal) return;
  q->terminal = true;
  EventRec* r = new_event(q, q->session ? KT2N_EVENT_PHRASE_DONE : KT2N_EVENT_DONE);
  r->ev.status = status;
  if (status != KT2N_OK) {
    set_err(&r->error, status, message);
    r->ev.error = &r->error;
  }
  if (status == KT2N_OK)
    e->done++;
  else if (status == KT2N_ERR_CANCELLED || status == KT2N_ERR_DEADLINE_EXCEEDED)
    e->cancelled++;
  else
    e->failed++;
  e->active_requests--;
  emit(e, q, r);
  if (q->session) session_phrase_done(e, q);
}

bool encoded_format(int f) { return f == KT2N_FORMAT_MP3 || f == KT2N_FORMAT_OPUS || f == KT2N_FORMAT_AAC || f == KT2N_FORMAT_FLAC; }
const char* format_name(int f) {
  switch (f) {
    case KT2N_FORMAT_MP3: return "mp3";
    case KT2N_FORMAT_OPUS: return "opus";
    case KT2N_FORMAT_AAC: return "aac";
    case KT2N_FORMAT_FLAC: return "flac";
    case KT2N_FORMAT_WAV: return "wav";
    case KT2N_FORMAT_F32: return "f32";
    default: return "pcm";
  }
}

// One levelled publication -> the response's bytes (PCM16 LE; streaming WAV header first; or the encoder's
// header first, then whatever it finalises now).
std::string bytes_of(Request* q, const post::Pcm& lvl) {
  std::string bytes;
#if KT2N_ENGINE_WITH_AV
  if (q->enc) {
    if (!q->header_sent) bytes = q->enc->take_header();
    q->header_sent = true;
    std::string error;
    q->enc->push(lvl, bytes, &error);
    return bytes;
  }
#endif
  if (q->format == KT2N_FORMAT_WAV && !q->header_sent) bytes = post::streaming_wav_header();
  q->header_sent = true;
  post::pcm16_bytes_from_float(lvl, bytes);
  return bytes;
}
std::string bytes_end(Request* q) {
  std::string bytes;
#if KT2N_ENGINE_WITH_AV
  if (q->enc) {
    if (!q->header_sent) bytes = q->enc->take_header();
    q->header_sent = true;
    std::string error;
    q->enc->finish(bytes, &error);
    return bytes;
  }
#endif
  if (q->format == KT2N_FORMAT_WAV && !q->header_sent) bytes = post::streaming_wav_header();
  q->header_sent = true;
  return bytes;
}

// Startup hold release event: held/target ms, chunks, reached, mode + sharing (adaptive).
void mark_hold(Engine* e, Request* q, int64_t held_samples, bool reached) {
  if (!q->timeline || q->realtime) return;
  timeline(e, q, KT2N_TL_STARTUP_HOLD_RELEASED,
           "\"held_ms\":" + jnum(double(held_samples) * 1000.0 / 24000.0) + ",\"target_ms\":" +
               jnum(double(q->hold->target_samples()) * 1000.0 / 24000.0) + ",\"chunks\":" +
               std::to_string(q->hold_chunks) + ",\"reached\":" + (reached ? "true" : "false") +
               ",\"mode\":\"adaptive\",\"sharing\":" + std::to_string(q->hold->peak_sharing()));
}

std::string f32_bytes(const std::vector<float>& v) {
  return std::string(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
}

void publish_streamed(Engine* e, Request* q, std::vector<float> pub) {
  // One reference publication (after ResponsePost): startup hold -> level -> bytes.
  if (pub.empty()) return;
  if (q->format == KT2N_FORMAT_F32 && q->output == KT2N_OUTPUT_COMPLETE) {
    // KT2N_FORMAT_F32 (adapter): the reference engine -> the reference boundary. A complete file leaves the
    // engine unheld and unlevelled; the reference takes the whole-file peak and writes the header.
    emit_audio(e, q, f32_bytes(pub), int64_t(pub.size()), false);
    return;
  }
  if (q->format != KT2N_FORMAT_F32 && (q->output == KT2N_OUTPUT_COMPLETE || q->output == KT2N_OUTPUT_STREAM_FILE)) {
    q->whole.insert(q->whole.end(), pub.begin(), pub.end());
    return;
  }
  post::HoldObservation obs;
  obs.listeners = std::max(1, e->sharing.load());
  obs.steady_joiner = e->steady.load() > 0;
  obs.trimmed_ms_so_far = q->rpost ? q->rpost->trimmed_ms() : std::nullopt;
  obs.opening_skipped_tokens = q->opening_skipped;
  q->hold_chunks += 1;
  post::Pcm out = q->hold->feed(pub, obs);
  // KT2N_HOLD_URGENT: the first audio has left the hold, so later commits no longer gate first audio. This also
  // covers a pinned-0 hold (WS phrases, startup_hold_seconds=0), which is a passthrough and never reports
  // released() (before this fix, every WS phrase commit was promoted to S_urgent).
  if (!out.empty()) q->awaiting_hold.store(false, std::memory_order_relaxed);
  if (q->hold->released() && !q->hold_released_marked) {
    q->hold_released_marked = true;
    q->awaiting_hold.store(false, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> ck(q->clock_mu);
      q->clock.release(mono_s());
    }
    mark_hold(e, q, int64_t(out.size()), true);
  }
  if (out.empty()) return;
  if (q->format == KT2N_FORMAT_F32) {
    // Held but not levelled: the reference service's normalize_audio + peak norm + PCM16 / encoders run as in the reference.
    emit_audio(e, q, f32_bytes(out), int64_t(out.size()), false);
    return;
  }
  post::Pcm lvl;
  if (post::level_publication(out, (q->peak_ptr ? q->peak_ptr : (q->peak ? &*q->peak : nullptr)), lvl) != post::NormalizeStatus::kOk) return;
  emit_audio(e, q, bytes_of(q, lvl), int64_t(lvl.size()), false);
}

// KT2N_EVENT_METRICS (the reference metrics snapshot, the scalars the reference reads / logs).
void emit_metrics(Engine* e, Request* q) {
  // Post thread.
  EventRec* r = new_event(q, KT2N_EVENT_METRICS);
  r->name = "metrics";
  int und;
  double und_ms;
  {
    std::lock_guard<std::mutex> ck(q->clock_mu);
    und = q->clock.underruns();
    und_ms = q->clock.underrun_ms();
  }
  std::ostringstream o;
  o << "{\"producer_seconds\":" << jnum(mono_s() - q->started_s) << ",\"audio_tokens\":" << q->m_audio_tokens
    << ",\"lm_steps\":" << q->m_lm_steps << ",\"segments\":" << q->m_segments
    << ",\"opening_silence_tokens_skipped\":" << std::max(0, q->opening_skipped - q->kept_tokens)
    << ",\"emo_applied\":" << ((q->plan && q->plan->decision.emotion) ? 1 : 0)
    << ",\"audio_seconds\":" << jnum(double(q->samples_out) / 24000.0)
    << ",\"d_underruns\":" << und << ",\"d_underrun_ms\":" << jnum(und_ms) << "}";
  r->fields = o.str();
  emit(e, q, r);
}

void finish_response(Engine* e, Request* q, int status) {
  // Post thread.
  if (status != KT2N_OK) {
    emit_done(e, q, status, status == KT2N_ERR_CANCELLED ? "request_cancelled" : "speech generation failed");
    return;
  }
  emit_metrics(e, q);
  if (q->format == KT2N_FORMAT_F32) {
    std::string bytes;
    int64_t n = 0;
    if (q->output != KT2N_OUTPUT_COMPLETE) {
      post::Pcm rest = q->hold->finish();
      if (!q->hold_released_marked && !rest.empty()) {
        q->hold_released_marked = true;
        q->awaiting_hold.store(false, std::memory_order_relaxed);
        {
          std::lock_guard<std::mutex> ck(q->clock_mu);
          q->clock.release(mono_s());
        }
        mark_hold(e, q, int64_t(rest.size()), false);
      }
      bytes = f32_bytes(rest);
      n = int64_t(rest.size());
    }
    emit_audio(e, q, bytes, n, true);  // the final marker (possibly empty)
  } else if (q->output == KT2N_OUTPUT_COMPLETE || q->output == KT2N_OUTPUT_STREAM_FILE) {
    // The whole-file levelling (the reference whole-file peak normalisation, demo peak formula).
    const float peak = post::demo_peak(q->whole);
    std::string bytes;
    int64_t n = int64_t(q->whole.size());
    if (encoded_format(q->format)) {
#if KT2N_ENGINE_WITH_AV
      post::Pcm lv;
      if (q->format == KT2N_FORMAT_FLAC)
        post::whole_file_flac(q->whole, peak, lv);
      else
        post::whole_file_lossy(q->whole, peak, 0.891, lv);
      bytes = bytes_of(q, lv);  // two statements: the push must come before the finish
      bytes += bytes_end(q);
#endif
    } else {
      std::vector<int16_t> pcm16;
      post::demo_int16_with_peak(q->whole, peak, pcm16);
      if (q->format == KT2N_FORMAT_WAV) bytes = post::complete_wav_header(uint64_t(pcm16.size()) * 2);
      bytes.append(reinterpret_cast<const char*>(pcm16.data()), pcm16.size() * 2);
    }
    emit_audio(e, q, bytes, n, true);
  } else {
    post::Pcm rest = q->hold->finish();
    if (!q->hold_released_marked && !rest.empty()) {
      q->hold_released_marked = true;
      q->awaiting_hold.store(false, std::memory_order_relaxed);
      {
        std::lock_guard<std::mutex> ck(q->clock_mu);
        q->clock.release(mono_s());
      }
      mark_hold(e, q, int64_t(rest.size()), false);
    }
    std::string bytes;
    int64_t n = 0;
    if (!rest.empty()) {
      post::Pcm lvl;
      if (post::level_publication(rest, (q->peak_ptr ? q->peak_ptr : (q->peak ? &*q->peak : nullptr)), lvl) ==
          post::NormalizeStatus::kOk) {
        bytes = bytes_of(q, lvl);
        n = int64_t(lvl.size());
      }
    }
    bytes += bytes_end(q);
    emit_audio(e, q, bytes, n, true);
  }
  emit_done(e, q, KT2N_OK, "");
}

// Conductor (the LM's driver thread, and the thread that drives the decoder): give the LM store space, the render table row and
// the prefix pages of evicted voices back once no request holds them (refs 0). Requests hold a ref from
// register_request to their release, so every row of the voice was submitted and closed by then; LM keeps the LM
// range until those rows' slots are released, and the decoder refuses (-2, retried on a later turn) while a row or a prefix
// build of the voice is still in flight. lm_voice / render_voice are set to -1 so a request that raced the eviction
// (looked the voice up just before it) fails at admission (custom_voice_service_error) instead of using a released
// id.
void release_voices(Engine* e) {
  bool retry = false;
  std::lock_guard<std::mutex> vlk(e->voices_mu);
  for (auto& v : e->voices) {
    if (!v->evicted.load() || v->refs.load() != 0) continue;
    if (e->vcache) {
      // Voice cache: no request holds the voice, so it has no pin either; the cache drops its GPU state (now, or once
      // its readers are done) and never loads it again (the loader counts what it gives back).
      if (!v->lm_released && v->consent_only) {  // a consent embedding: never registered, no GPU state to drop
        v->lm_released = v->render_released = true;
        continue;
      }
      if (!v->lm_released) {
        if (v->cache_key == -1) {  // its registration job has not run yet: retire it once it has a key
          retry = true;
          continue;
        }
        if (v->cache_key >= 0) e->vcache->retire(v->cache_key);
        v->lm_released = v->render_released = true;
      }
      continue;
    }
    if (!v->lm_released) {
      const int lmv = v->lm_voice.load();
      if (lmv >= 0) {  // < 0: never built (store full / failed build), or still building in an admin job
        if (e->lm->release_voice(lmv) == 0) e->lm_voices_released++;
        v->lm_voice = -1;
        v->lm_released = true;
      }
    }
    if (!v->render_released && v->render_voice >= 0) {
      std::string err;
      const int rc = e->render->remove_voice(v->render_voice, &err);
      if (rc == -2) {
        retry = true;  // busy: a row or a prefix build of it is still in flight
      } else {
        if (rc == 0) e->render_voices_released++;
        v->render_voice = -1;
        v->render_released = true;
      }
    }
  }
  if (retry) e->voice_sweep = true;
}

void post_item(Engine* e, Request* q, PostItem it) {
  // Post thread (one per request at a time); it.job == q->post_cur_job.
  if (q->post_begun_job < 0 && q->kept_tokens > 0) q->rpost->set_kept_opening_tokens(q->kept_tokens);
  if (it.job > q->post_begun_job) {
    q->post_begun_job = it.job;
    post::SeamContext seam;
    seam.demo_request = q->demo;
    seam.demo_chunked = true;
    seam.dc_mode = q->plan ? q->plan->decision.dc_mode : 0;
    seam.lane_sealed = it.lane_sealed;
    seam.lane_jobs = it.lane_jobs;
    seam.lane_known = true;
    q->rpost->begin_segment(it.job, seam);
  }
  std::vector<post::Pcm> pubs;
  if (!it.pcm.empty()) pubs.push_back(q->rpost->push(it.pcm));
  if (it.final) {
    const post::SeamEdgeTrimmer* seam = q->rpost->seam();
    const bool seam_finishing = seam && !seam->finished();
    for (auto& p : q->rpost->end_segment(it.has_more)) pubs.push_back(std::move(p));
    if (seam_finishing && q->timeline) {
      // the reference seam note (made right after the seam finishes): what this segment's edges
      // lost, on a traced request. seam_released_ms only when a hold-release window released audio (0 by default: never).
      timeline(e, q, KT2N_TL_SEAM_TRIMMED,
               "\"segment_index\":" + std::to_string(it.job) +
                   ",\"seam_head_trimmed_ms\":" + jnum(double(seam->head_trimmed_samples()) * 1000.0 / 24000.0) +
                   ",\"seam_tail_trimmed_ms\":" + jnum(double(seam->tail_trimmed_samples()) * 1000.0 / 24000.0),
               it.job);
    }
    if (!it.has_more) pubs.push_back(q->rpost->finish());
    q->post_cur_job += 1;
  }
  q->dropped_samples.store(q->rpost->dropped_samples());
  for (auto& p : pubs) publish_streamed(e, q, std::move(p));
}

void post_run(Engine* e, Request* q) {
  // Items are popped under q->mu and processed without it: post state is touched only by the one post thread
  // that holds this request (post_scheduled), so the conductor's post_push never waits on trims / encoders.
  for (;;) {
    PostItem it;
    {
      std::lock_guard<std::mutex> lk(q->mu);
      if (q->post_q.empty()) {
        q->post_scheduled = false;
        return;
      }
      it = std::move(q->post_q.front());
      q->post_q.pop_front();
    }
    if (q->terminal) continue;
    if (q->cancelled.load()) {
      emit_done(e, q, q->cancel_status,
                q->cancel_status == KT2N_ERR_CANCELLED ? "request_cancelled"
                : q->cancel_status == KT2N_ERR_GPU_MEMORY_LIMIT ? q->fail_reason
                : q->fail_reason.empty()               ? "speech generation failed"
                                                       : "speech generation failed: " + q->fail_reason);
      continue;
    }
    if (it.end && it.status != KT2N_OK) {
      finish_response(e, q, it.status);
      continue;
    }
    if (it.end) {
      q->post_end = std::move(it);
    } else if (it.job != q->post_cur_job) {
      q->post_later[it.job].push_back(std::move(it));
    } else {
      post_item(e, q, std::move(it));
    }
    // Drain the held publications of the jobs whose turn came.
    for (;;) {
      auto later = q->post_later.find(q->post_cur_job);
      if (later == q->post_later.end() || later->second.empty()) break;
      PostItem next = std::move(later->second.front());
      later->second.pop_front();
      if (later->second.empty()) q->post_later.erase(later);
      post_item(e, q, std::move(next));
    }
    if (q->post_end && q->post_cur_job >= q->post_end->lane_jobs) {
      q->post_end.reset();
      finish_response(e, q, KT2N_OK);
    }
  }
}

void post_worker(Engine* e) {
  for (;;) {
    Request* q;
    {
      std::unique_lock<std::mutex> lk(e->post_mu);
      e->post_cv.wait(lk, [&] { return e->stop.load() || !e->post_q.empty(); });
      if (e->stop.load() && e->post_q.empty()) return;
      q = e->post_q.front();
      e->post_q.pop_front();
    }
    post_run(e, q);
  }
}

// ------------------------------------------------------------------------------------- text
void text_worker(Engine* e) {
  // Experimental option: text threads below the conductor / post threads under CPU contention (a thread's own nice).
#if !defined(_WIN32)  // not available on Windows
  if (e->text_nice > 0) setpriority(PRIO_PROCESS, id_t(syscall(SYS_gettid)), e->text_nice);
#endif
  for (;;) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lk(e->text_mu);
      e->text_cv.wait(lk, [&] { return e->stop.load() || !e->text_q.empty(); });
      if (e->stop.load() && e->text_q.empty()) return;
      task = std::move(e->text_q.front());
      e->text_q.pop_front();
    }
    task();
  }
}

// A text task for request q: counted so a released request is never freed under a running task.
void text_enqueue(Engine* e, std::function<void()> f) {
  {
    std::lock_guard<std::mutex> lk(e->text_mu);
    e->text_q.push_back(std::move(f));
  }
  e->text_cv.notify_one();
}

// e->mu held by the caller.
void text_task_locked(Engine* e, Request* q, std::function<void()> f) {
  q->text_tasks += 1;
  text_enqueue(e, [e, q, f = std::move(f)] {
    if (!q->cancelled.load() && !e->backend_failed.load()) f();
    {
      std::lock_guard<std::mutex> lk(e->mu);
      q->text_tasks -= 1;
    }
    e->release_pending = true;
    e->cv.notify_all();
  });
}

// Normalize planned segment `index` into jobs (text thread). Appends under e->mu.
void normalize_into_jobs(Engine* e, Request* q, size_t index) {
  if (q->cancelled.load() || e->backend_failed.load()) return;
  std::vector<std::u32string> chunks;
  std::string error;
  try {
    chunks = e->text->normalize_segment(*q->plan, index);
    // Low-memory fit, smallest tier: a streaming chunk whose token budget is beyond the fitted streaming budget (the
    // worst case over prompt lengths 1..250, so a voice with a short prompt could sometimes have held it whole) is
    // re-split by the text plan's rule (sentence, then clause, then word breaks; a single word longer than the limit
    // is cut inside the word) into pieces whose budget fits. Only such chunks change: their audio is cut into more
    // segments than on a larger GPU. A piece that still does not fit is refused at admission. KT2N_FIT_SPLIT=0: no
    // split, such chunks are refused at admission.
    if (e->fit_split && !q->demo) {
      const int fit_budget = e->fit.streaming_budget_tokens;
      const int max_chars = int(double(fit_budget) / textplan::Settings{}.tagged_budget_per_char) - 1;
      std::vector<std::u32string> fitted;
      for (auto& c : chunks) {
        if (max_chars < 64 || textplan::tagged_segment_budget(c.size()) <= fit_budget) {
          fitted.push_back(std::move(c));
          continue;
        }
        e->fit_split_chunks++;
        for (auto& piece : e->text->split_chunk(*q->plan, c, max_chars)) fitted.push_back(std::move(piece));
      }
      chunks.swap(fitted);
    }
  } catch (const std::exception& ex) {
    error = ex.what();
  }
  std::vector<std::unique_ptr<Job>> made;
  if (error.empty()) {
    const bool demo = q->demo;
    std::vector<size_t> lens;
    for (auto& c : chunks) lens.push_back(c.size());
    std::vector<int> budgets;
    std::unique_lock<std::mutex> lk(e->mu);
    if (q->cancelled.load() || e->backend_failed.load()) return;
    if (index == 0) q->total_chunks = int(q->plan->raw_segments.size()) - 1 + int(chunks.size());
    const int start = int(q->jobs_created);
    q->jobs_created += chunks.size();
    lk.unlock();
    if (demo) budgets = textplan::demo_budgets(lens, q->total_chunks, start);
    for (size_t i = 0; i < chunks.size(); ++i) {
      auto j = std::make_unique<Job>();
      j->ordinal = start + int(i);
      j->raw_segment = int(index);
      j->text = chunks[i];
      j->suffix = e->text->suffix(chunks[i], q->plan->decision.emotion);
      j->budget = demo ? budgets[i] : textplan::tagged_segment_budget(chunks[i].size());
      made.push_back(std::move(j));
    }
  }
  {
    std::lock_guard<std::mutex> lk(e->mu);
    if (q->cancelled.load() || e->backend_failed.load()) return;
    if (!error.empty()) {
      q->text_status = KT2N_ERR_SPEECH_GENERATION_FAILED;
      q->text_error = error;
    }
    for (auto& j : made) q->new_jobs.push_back(std::move(j));
    q->segments_normalized = index + 1;
    q->sealed = q->segments_normalized >= q->plan->raw_segments.size();
    q->normalizing = false;
  }
  e->cv.notify_all();
}

void plan_request(Engine* e, Request* q) {
  if (q->cancelled.load() || e->backend_failed.load()) return;
  std::unique_ptr<PlannedRequest> plan;
  std::string error;
  try {
    plan = std::make_unique<PlannedRequest>(e->text->plan(q->raw, q->voice->language, q->emotion_preset, q->normalize));
  } catch (const std::exception& ex) {
    error = ex.what();
  }
  {
    std::lock_guard<std::mutex> lk(e->mu);
    if (q->cancelled.load() || e->backend_failed.load()) return;
    if (!plan || plan->raw_segments.empty()) {
      q->text_status = KT2N_ERR_SPEECH_GENERATION_FAILED;
      q->text_error = error.empty() ? "empty plan" : error;
      q->sealed = true;
    } else {
      q->plan = std::move(plan);
      q->normalizing = true;
    }
  }
  if (q->plan) {
    timeline(e, q, KT2N_TL_TEXT_SEGMENTED,
             "\"characters\":" + std::to_string(q->raw.size()) +
                 ",\"segments\":" + std::to_string(q->plan->raw_segments.size()));
    normalize_into_jobs(e, q, 0);  // segment 0 on the request's own text task
    timeline(e, q, KT2N_TL_TEXT_NORMALIZED, "\"segment_index\":0", 0);
  } else {
    e->cv.notify_all();
  }
}

// ------------------------------------------------------------------------------------- conductor
std::optional<double> d_buffer_ms(Request* q, double now) {
  std::optional<double> b;
  {
    std::lock_guard<std::mutex> ck(q->clock_mu);
    b = q->clock.buffer_ms(now);
  }
  if (!b) return b;
  return *b - double(q->dropped_samples.load()) * 1000.0 / pol::kSampleRate;
}

void acknowledge(Engine* e, Request* q, Job* j) {
  if (j->acknowledged) return;
  j->acknowledged = true;
  if (j->charged) {
    q->inflight -= 1;
    e->inflight_segments -= 1;
  }
}

void post_push(Engine* e, Request* q, PostItem it) {
  std::lock_guard<std::mutex> lk(q->mu);
  q->post_q.push_back(std::move(it));
  schedule_post(e, q);
}

void fail_request(Engine* e, Request* q, int status, const std::string& reason = "") {
  if (q->generation_complete) return;
  q->generation_complete = true;
  q->fail_reason = reason;  // internal detail for logs / the lab (the public message stays the reference wording)
  // Its rows stop decoding and rendering (the next turn cancels them); DONE carries this status.
  q->cancel_status = status;
  q->cancelled = true;
  PostItem it;
  it.end = true;
  it.status = status;
  post_push(e, q, std::move(it));
}

// Publication of one row's render output, in the row's order (conductor).
void publish_row(Engine* e, Row* row, std::vector<float> pcm, bool final) {
  Request* q = row->req;
  Job* job = q->jobs[size_t(row->job)].get();
  const double now = mono_s();
  if (q->cancelled.load()) return;
  if (!pcm.empty()) {
    if (!job->first_pub) job->first_pub = now;
    job->published_samples += int64_t(pcm.size());
    job->preview_ready = true;
    {
      std::lock_guard<std::mutex> ck(q->clock_mu);
      q->clock.publish(now, int64_t(pcm.size()));
    }
  }
  const bool last_job = q->c_sealed && size_t(job->ordinal) + 1 >= q->jobs.size();
  PostItem it;
  it.lane_jobs = int(q->jobs.size());
  it.lane_sealed = q->c_sealed;
  it.job = job->ordinal;
  it.pcm = std::move(pcm);
  it.final = final;
  it.has_more = !last_job;
  post_push(e, q, std::move(it));
  if (final) {
    job->done = true;
    acknowledge(e, q, job);
    if (last_job) {
      q->generation_complete = true;
      PostItem end;
      end.end = true;
      end.lane_jobs = int(q->jobs.size());
      post_push(e, q, std::move(end));
    }
  } else if (!job->acknowledged && job->charged) {
    // Consider the next text segment, on every engine chunk.
    if (pol::segment_prefetch_due(last_job, job->done, job->published_samples, job->text.size(),
                                  e->inflight_segments, e->lv)) {
      acknowledge(e, q, job);
      timeline(e, q, KT2N_TL_SEGMENT_PREFETCH_RELEASED,
               "\"segment_index\":" + std::to_string(job->ordinal) +
                   ",\"received_ms\":" + jnum(double(job->published_samples) * 1000.0 / 24000.0),
               job->ordinal);
    }
  }
}

void close_row_if_done(Engine* e, Row* row) {
  if (row->in_active || !row->final_launched || row->async_inflight > 0 || !row->reorder.empty()) return;
  if (row->pub_seq < row->next_seq) return;
  if (!row->closed) {
    e->render->close_row(row->key);
    row->closed = true;
  }
}

void drop_closed_rows(Engine* e) {
  for (auto it = e->rows.begin(); it != e->rows.end();) {
    if (it->second->closed) {
      Request* q = it->second->req;
      q->live_rows -= 1;
      if (e->vcache) cache_unpin(e, q);
      it = e->rows.erase(it);
    } else {
      ++it;
    }
  }
}

void process_outs(Engine* e, std::vector<RenderOut>& outs, bool drop_closed) {
  for (RenderOut& o : outs) {
    auto it = e->inflight.find(o.call_id);
    if (it == e->inflight.end()) continue;
    Inflight& inf = it->second;
    e->render_inflight--;
    for (size_t i = 0; i < inf.rows.size(); ++i) {
      const Inflight::RowRef& rr = inf.rows[i];
      auto rit = e->rows.find(rr.row);
      if (rit == e->rows.end()) continue;
      Row* row = rit->second.get();
      if (inf.asyncs[i]) {
        row->async_inflight -= 1;
        if (row->async_inflight == 0) row->c.inflight = false;
      }
      std::vector<float> pcm = i < o.pcm.size() ? std::move(o.pcm[i]) : std::vector<float>{};
      Row::Done d;
      if (o.status != 0) {
        if (e->fit.fitted && (o.error.find("max_t") != std::string::npos || o.error.find("frame-rows") != std::string::npos ||
                              o.error.find("pages") != std::string::npos || o.error.find("workspace") != std::string::npos))
          fail_request(e, row->req, KT2N_ERR_GPU_MEMORY_LIMIT,
                       "this segment needs more decoder or vocoder memory than the low-memory profile fitted on this "
                       "GPU (" + o.error + ")");
        else
        fail_request(e, row->req, KT2N_ERR_SPEECH_GENERATION_FAILED,
                     "render call failed (status " + std::to_string(o.status) + (o.error.empty() ? "" : ": " + o.error) + ")");
        d.failed = true;
      }
      d.pcm = std::move(pcm);
      d.final = bool(inf.finals[i]);
      d.commit_tokens = rr.commit_tokens;
      d.gpu_ms = o.gpu_ms;
      d.launch_ms = o.launch_ms;
      d.launched_s = inf.launched_s;
      d.batch = inf.rows.size();
      d.graph_nodes = o.graph_nodes;
      d.graph_captured = o.graph_captured;
      row->reorder[rr.seq] = std::move(d);
      while (!row->reorder.empty() && row->reorder.begin()->first == row->pub_seq) {
        auto node = std::move(row->reorder.begin()->second);
        row->reorder.erase(row->reorder.begin());
        row->pub_seq += 1;
        const bool fin = node.final;
        const size_t n = node.pcm.size();
        if (node.failed) continue;
        if (row->req->timeline) {
          timeline(e, row->req, fin ? KT2N_TL_S3GEN_FINAL_PUBLICATION : KT2N_TL_S3GEN_PUBLICATION,
                   "\"s3gen_wall_ms\":" + jnum((mono_s() - node.launched_s) * 1000.0) +
                       ",\"s3gen_gpu_ms\":" + jnum(node.gpu_ms) + ",\"s3gen_launch_ms\":" + jnum(node.launch_ms) +
                       ",\"s3gen_queue_depth\":" + std::to_string(e->render_inflight.load()) +
                       ",\"batch_size\":" + std::to_string(node.batch) +
                       ",\"commit_tokens\":" + std::to_string(node.commit_tokens) +
                       ",\"graph_nodes\":" + std::to_string(node.graph_nodes) +
                       ",\"graph_captured\":" + (node.graph_captured ? "true" : "false") +
                       ",\"audio_ms\":" + jnum(double(n) * 1000.0 / 24000.0) +
                       ",\"audio_samples\":" + std::to_string(n) +
                       ",\"segment_index\":" + std::to_string(row->job),
                   row->job);
        }
        publish_row(e, row, std::move(node.pcm), fin);  // after its timeline mark: DONE may follow at once
      }
      if (!e->backend_failed.load()) close_row_if_done(e, row);
    }
    e->inflight.erase(it);
  }
  if (drop_closed) drop_closed_rows(e);  // not inside a launch: the caller may still hold Row pointers
}

void collect_renders(Engine* e, bool wait) {
  std::vector<RenderOut> outs;
  e->render->collect(&outs, wait);
  process_outs(e, outs, true);
}

// KT2N_BUFFER_URGENT_MS: true when the request is already playing (its first audio has left the hold) and its
// delivery clock's buffer (d_buffer: published audio not yet played on the listener's timeline) is below the arm's
// threshold. The audible-pause rule counts a >= 250 ms gap between chunks while < 1 s is buffered (one
// measured run at N=20: 706 of 6,457 streams, 697 of them a 2nd-publication render queued behind ~690 ms of bulk work).
bool buffer_low(Engine* e, Request* q) {
  if (e->buffer_urgent_ms <= 0 || q->awaiting_hold.load(std::memory_order_relaxed)) return false;
  std::lock_guard<std::mutex> ck(q->clock_mu);
  if (!q->clock.started()) return false;
  const auto b = q->clock.buffer_ms(mono_s());
  return b && *b < e->buffer_urgent_ms;
}

void launch(Engine* e, std::vector<Row*> rows, render::Phase phase, bool async, bool finals,
            bool force_urgent = false, int force_role = -1) {
  if (rows.empty()) return;
  RenderCall call;
  call.call_id = e->next_call++;
  call.async = async;
  Inflight inf;
  int min_commit = 1 << 30;
  render::Phase call_phase = phase;
  for (Row* row : rows) {
    RenderRowReq rr;
    rr.row = row->key;
    rr.voice = row->req->voice->render_voice;
    rr.commit_index = row->c.commit_index;
    rr.committed_tokens = row->c.committed;
    rr.seq = row->next_seq++;
    if (finals) {
      rr.phase = row->req->demo ? render::Phase::kWholeRender : render::Phase::kFinalFlush;
      rr.finalize = true;
      rr.tokens = row->audio;
      rr.tokens.insert(rr.tokens.end(), 3, pol::kSilenceToken);
      rr.commit_tokens = int(row->audio.size());
      call_phase = rr.phase;
    } else {
      const bool first = row->c.commit_index == 0 && row->c.committed == 0;
      rr.phase = first ? render::Phase::kFirstCommit : render::Phase::kSteadyCommit;
      rr.commit_tokens = row->c.next;
      rr.tokens.assign(row->audio.begin(), row->audio.begin() + (row->c.next + pol::kLookahead));
      if (!first) call_phase = render::Phase::kSteadyCommit;
    }
    min_commit = std::min(min_commit, rr.commit_index);
    inf.finals.push_back(finals);
    inf.asyncs.push_back(async);
    if (async) {
      row->async_inflight += 1;
      row->c.inflight = !finals || row->c.inflight;
    }
    if (row->req->timeline)
      timeline(e, row->req, KT2N_TL_S3GEN_STARTED,
               "\"commit_tokens\":" + std::to_string(rr.commit_tokens) +
                   ",\"prefix_tokens\":" + std::to_string(rr.committed_tokens) +
                   ",\"segment_index\":" + std::to_string(row->job) +
                   ",\"batch_size\":" + std::to_string(rows.size()) +
                   ",\"async_launch\":" + (async ? "true" : "false"),
               row->job);
    call.rows.push_back(std::move(rr));
    if (!finals) pol::after_commit(row->c, e->lv);  // the launch advances the row's state at once
  }
  call.stream = render::stream_for(call_phase, min_commit);
  if (force_urgent) {  // KT2N_HOLD_URGENT: commits (level 1) or finals (level 2) a startup hold still waits for
    call.stream = render::StreamRole::kSUrgent;
    e->hold_urgent_calls++;
  }
  if (force_role >= 0) call.stream = render::StreamRole(force_role);  // KT2N_BUFFER_ROLE=aux
  inf.stream = call.stream;
  inf.launched_s = mono_s();
  std::string err;
  const uint64_t id = call.call_id;
  for (const auto& r : call.rows) inf.rows.push_back({r.row, r.seq, r.commit_tokens});
  e->inflight.emplace(id, std::move(inf));
  e->render_inflight++;
  e->renders++;
  e->render_rows += call.rows.size();
  if (e->render->launch(std::move(call), &err) != 0) {
    std::vector<RenderOut> failed(1);
    failed[0].call_id = id;
    failed[0].status = KT2N_ERR_SPEECH_GENERATION_FAILED;
    failed[0].error = "launch: " + err;
    process_outs(e, failed, false);
  }
}

int startup_sharing(Engine* e) {
  // admission snapshot queued_requests: requests holding a lane, not complete.
  int n = 0;
  for (Request* q : e->lanes)
    if (!q->generation_complete && !q->cancelled.load()) ++n;
  return std::max(1, n);
}

void dw_service(Engine* e) {
  if (e->dw_queue.empty()) return;
  const double now = mono_s();
  pol::DwInput in;
  in.now_s = now;
  in.oldest_queued_s = e->dw_queue.front().second;
  in.last_whole_render_s = e->last_whole_render;
  std::optional<double> low;
  for (uint64_t k : e->active) {
    Row* r = e->rows[k].get();
    if (r->req->demo) continue;
    in.any_realtime = true;
    auto b = d_buffer_ms(r->req, now);
    if (!b) in.waiting = true;
    Job* j = r->req->jobs[size_t(r->job)].get();
    if (j->ordinal == 0 && !j->preview_ready) in.waiting = true;
    const double v = b.value_or(0.0);
    low = low ? std::min(*low, v) : v;
  }
  in.low_buffer_ms = low;
  const pol::DwReason why = pol::dw_due(in, e->lv);
  if (why == pol::DwReason::kNone) return;
  std::vector<Row*> batch;
  if (why == pol::DwReason::kIdle) {
    for (auto& [k, t] : e->dw_queue) batch.push_back(e->rows[k].get());
    e->dw_queue.clear();
  } else {
    // The oldest + the queued renders of the same voice in its length bucket (final_flush_buckets).
    Row* oldest = e->rows[e->dw_queue.front().first].get();
    std::vector<std::pair<int, int>> same;
    std::unordered_map<int, Row*> by_id;
    int idx = 0;
    for (auto& [k, t] : e->dw_queue) {
      Row* r = e->rows[k].get();
      if (r->req->voice == oldest->req->voice) {
        same.push_back({idx, int(r->audio.size())});
        by_id[idx] = r;
      }
      ++idx;
    }
    for (const auto& b : pol::final_flush_buckets(same, e->lv)) {
      bool has_oldest = false;
      for (int i : b) has_oldest |= by_id[i] == oldest;
      if (!has_oldest) continue;
      for (int i : b) batch.push_back(by_id[i]);
    }
    std::set<uint64_t> taken;
    for (Row* r : batch) taken.insert(r->key);
    std::deque<std::pair<uint64_t, double>> rest;
    for (auto& p : e->dw_queue)
      if (!taken.count(p.first)) rest.push_back(p);
    e->dw_queue.swap(rest);
  }
  for (Row* r : batch)
    timeline(e, r->req, KT2N_TL_DW_RENDER, "\"reason\":\"" + std::string(pol::dw_reason_name(why)) + "\"", r->job);
  launch(e, batch, render::Phase::kWholeRender, true, true);
  e->last_whole_render = mono_s();
}

void admit(Engine* e) {
  // Lanes in arrival order; prune completed ones.
  {
    std::lock_guard<std::mutex> lk(e->mu);
    while (!e->incoming.empty()) {
      e->lanes.push_back(e->incoming.front());
      e->lanes.back()->in_lanes = true;
      e->incoming.pop_front();
    }
  }
  std::vector<Request*> keep, gone;
  if (e->vcache) cache_end_session_keeps(e);
  for (Request* q : e->lanes) {
    bool live;
    {
      std::lock_guard<std::mutex> lk(e->mu);
      for (auto& j : q->new_jobs) q->jobs.push_back(std::move(j));
      q->new_jobs.clear();
      q->c_sealed = q->sealed;
      // Voice cache: a voice is dead when it can never fit the cache budget or its loads failed while pinned.
      // A request that looked a saved voice up just before kt2n_voice_evict and took its ref after release_voices gave
      // the voice back is dead too (lm_released: conductor only); in ALL mode lm_voice is -1 then.
      const bool voice_dead =
          e->vcache ? (q->voice->cache_key == -2 || (q->voice->evicted.load() && q->voice->lm_released) ||
                       (q->voice->cache_key >= 0 && e->vcache->failed(q->voice->cache_key)))
                    : q->voice->evicted.load() && q->voice->lm_voice.load() < 0;
      live = !q->generation_complete && !q->cancelled.load() && q->text_status == 0 && !voice_dead;
      if (voice_dead && !q->generation_complete)
        fail_request(e, q, e->vcache && q->voice->kind == KT2N_VOICE_BUILTIN
                               ? KT2N_ERR_SPEECH_GENERATION_FAILED
                               : KT2N_ERR_CUSTOM_VOICE_SERVICE_ERROR);  // its LM prefix could not be built
      else if (q->cancelled.load() && !q->generation_complete)
        fail_request(e, q, q->cancel_status);  // cancelled before or while waiting for a row
      else if (q->text_status != 0 && !q->generation_complete)
        fail_request(e, q, q->text_status);
    }
    if (live) {
      keep.push_back(q);
      if (e->vcache) cache_pin(e, q);
    } else {
      q->in_lanes = false;
      gone.push_back(q);
    }
  }
  // Unpin after every live request pinned: a finished request must not drop (NONE) a voice a newer request in the
  // lanes uses in the same turn.
  if (e->vcache)
    for (Request* q : gone) cache_unpin(e, q);
  e->lanes.swap(keep);
  e->sharing = startup_sharing(e);
  int active_count = int(e->active.size());
  const int capacity = std::min(e->max_streams, e->row_cap) - active_count;
  if (capacity <= 0 || e->lanes.empty()) return;
  std::vector<pol::LaneView> views;
  std::vector<bool> continuing;
  {
    std::lock_guard<std::mutex> lk(e->mu);
    for (Request* q : e->lanes) {
      views.push_back({q->next_index < q->jobs.size() && voice_usable(e, q->voice), q->inflight});
      continuing.push_back(q->next_index > 0);
    }
  }
  std::vector<int> taken;
  if (e->capacity_order && e->row_cap < e->max_streams) {
    // Experimental option (KT2N_CAPACITY_ORDER=1; off by default = the reference single round-robin), capacity mode
    // only (more seats than rows; the reference never runs this way: its seats = its decode rows). With many
    // queued requests, round-robin makes a request that is already playing wait for its NEXT segment behind
    // the queued new ones (a short check with a 96-page test pool, ~1 row: 5 takes ran dry before their next
    // segment was admitted). Continuing requests are offered first, with the reference round-robin among them;
    // new requests share what is left with the reference round-robin and cursor. A/B (128 pages,
    // 3 rows, N=32): 0 dry takes in BOTH settings, so not shown to help yet; measured only in an exclusive window.
    taken = pol::admission_take_capacity(views, continuing, capacity, &e->lane_cursor_cont, &e->lane_cursor, e->lv);
  } else {
    taken = pol::admission_take(views, capacity, &e->lane_cursor, e->lv);
  }
  for (int li : taken) {
    Request* q = e->lanes[size_t(li)];
    Job* j;
    {
      std::lock_guard<std::mutex> lk(e->mu);
      j = q->jobs[q->next_index].get();
      q->next_index += 1;
    }
    if (!j->acknowledged) {
      q->inflight += 1;
      j->charged = true;
      e->inflight_segments += 1;
    }
    auto row = std::make_unique<Row>();
    row->key = e->next_row++;
    row->req = q;
    row->job = j->ordinal;
    row->c = pol::new_commit_state(e->lv, q->demo);
    row->skip_armed = j->ordinal == 0 && !q->demo && q->skip_opening;  // opening-silence skip: segment 0, not DEMO
    row->admitted_s = mono_s();
    lm::RowRequest rr;
    rr.tag = row->key;
    rr.voice = q->voice->lm_voice;
    rr.suffix.assign(j->suffix.begin(), j->suffix.end());
    rr.sampler = q->sampler;
    rr.seed = q->salt + uint64_t(j->ordinal);
    rr.budget = j->budget;
    rr.demo = q->demo;
    rr.demo_chunked = true;  // every request is demo-chunked by default
    std::string err;
    auto put_back = [&] {
      std::lock_guard<std::mutex> lk(e->mu);
      q->next_index -= 1;
      if (j->charged) {
        q->inflight -= 1;
        j->charged = false;
        e->inflight_segments -= 1;
      }
    };
    if (e->fit_ctx > 0 || e->fit_frames > 0) {
      // Low-memory fit: a segment the fitted sizes cannot hold at its budget is refused before any of its work.
      row->prompt_ids = q->voice->lm_prefix_ids + int(rr.suffix.size());
      static const bool dbg = [] {
        const char* v = std::getenv("KT2N_DEBUG_MEM");
        return v && *v == '1';
      }();
      if (dbg)
        std::fprintf(stderr, "[fit] segment %d demo %d prompt %d budget %d chars %zu\n", j->ordinal, q->demo ? 1 : 0,
                     row->prompt_ids, j->budget, j->text.size());
      std::string why;
      if (e->fit_ctx > 0 && row->prompt_ids + j->budget + 1 > e->fit_ctx)
        why = "this segment needs " + std::to_string(row->prompt_ids + j->budget + 1) +
              " language-model context positions (prompt " + std::to_string(row->prompt_ids) + " + budget " +
              std::to_string(j->budget) + "); the low-memory profile fitted " + std::to_string(e->fit_ctx) +
              " on this GPU";
      else if (e->fit_frames > 0 && q->demo && kt2n::render::kMelPerToken * (j->budget + 6) > e->fit_frames)
        why = "this complete-file segment needs " + std::to_string(kt2n::render::kMelPerToken * (j->budget + 6)) +
              " vocoder frames; the low-memory profile fitted " + std::to_string(e->fit_frames) + " on this GPU";
      if (!why.empty()) {
        fail_request(e, q, KT2N_ERR_GPU_MEMORY_LIMIT, why + " (shorter segments, or a GPU with more memory)");
        continue;
      }
    }
    // Render state first: a row the renderer cannot take yet (no slot, or its pool pages cannot be reserved)
    // waits in admission instead of failing mid-request (native back-pressure; the reference has no page pool).
    const int orc = e->render->open_row(row->key, q->voice->render_voice, rr.seed, j->budget, q->demo, &err);
    if (orc == kRenderBackpressure) {
      put_back();
      e->render_backpressure++;
      break;  // keep arrival order: nothing behind it is admitted this turn
    }
    if (orc != 0) {
      if (e->fit.fitted)
        fail_request(e, q, KT2N_ERR_GPU_MEMORY_LIMIT,
                     "this segment needs more decoder pages than the low-memory profile fitted on this GPU (" + err +
                         (e->fit.streaming_budget_tokens > 0 && !q->demo
                              ? "; streaming segments fit up to a budget of " +
                                    std::to_string(e->fit.streaming_budget_tokens) + " tokens, about " +
                                    std::to_string(e->fit.streaming_budget_tokens * 10 / 44) + " characters"
                              : std::string()) +
                         ")");
      else
        fail_request(e, q, KT2N_ERR_SPEECH_GENERATION_FAILED, "render open_row: " + err);
      continue;
    }
    if (e->lm->submit(rr, &err) != 0) {
      // LM back-pressure (a slot not yet released after read-back): put the job back, retry next loop.
      e->render->close_row(row->key);
      put_back();
      e->lm_backpressure++;
      continue;
    }
    timeline(e, q, KT2N_TL_SCHEDULER_ADMITTED,
             "\"admission_batch\":" + std::to_string(taken.size()) + ",\"active_before\":" +
                 std::to_string(active_count) + ",\"queue_ms\":" + jnum((mono_s() - q->started_s) * 1000.0) +
                 ",\"segment_index\":" + std::to_string(j->ordinal),
             j->ordinal);
    e->active.push_back(row->key);
    q->live_rows += 1;
    e->rows.emplace(row->key, std::move(row));
    ++active_count;
  }
}

// Normaliser thread trigger: segment k >= 1 after segment 0 published >= 1.5 s (or is done),
// at most 1 normalized segment ahead of admission.
void maybe_normalize_next(Engine* e, Request* q) {
  std::lock_guard<std::mutex> lk(e->mu);
  if (!q->plan || q->sealed || q->normalizing || q->cancelled.load() || q->jobs.empty()) return;
  Job* j0 = q->jobs[0].get();
  const bool first_ok = j0->done || j0->published_samples >= int64_t(e->lv.normalize_start_audio_s * 24000) ||
                        (mono_s() - q->started_s) >= e->lv.normalize_wait_max_s;
  if (!first_ok) return;
  if (q->jobs_created - q->next_index > size_t(e->lv.normalize_lookahead)) return;
  q->normalizing = true;
  const size_t index = q->segments_normalized;
  text_task_locked(e, q, [e, q, index] { normalize_into_jobs(e, q, index); });
}

void finalize_rows(Engine* e, const std::vector<Row*>& done, int decoded) {
  std::vector<Row*> stream_finals;
  for (Row* r : done) {
    pol::opening_restore(r->skip, r->audio);
    r->in_active = false;
    r->final_launched = true;
    if (r->req->timeline && !r->audio.empty()) {
      std::string bytes(r->audio.size() * 4, '\0');
      for (size_t i = 0; i < r->audio.size(); ++i) {
        const uint32_t v = uint32_t(r->audio[i]);
        for (int k = 0; k < 4; ++k) bytes[i * 4 + size_t(k)] = char((v >> (8 * k)) & 0xff);
      }
      timeline(e, r->req, KT2N_TL_LM_TOKENS_FINALIZED,
               "\"segment_index\":" + std::to_string(r->job) + ",\"audio_token_count\":" +
                   std::to_string(r->audio.size()) + ",\"audio_token_sha256\":\"" +
                   kt2n::sha256_hex(bytes.data(), bytes.size()) + "\"",
               r->job);
    }
    if (r->audio.empty()) {
      // "generated no audio tokens" fails the request.
      fail_request(e, r->req, KT2N_ERR_SPEECH_GENERATION_FAILED, "generated no audio tokens");
      close_row_if_done(e, r);
      continue;
    }
    Job* j = r->req->jobs[size_t(r->job)].get();
    j->lm_done = true;
    r->req->m_audio_tokens += int64_t(r->audio.size());
    r->req->m_lm_steps += r->lm_steps;
    r->req->m_segments += 1;
    timeline(e, r->req, KT2N_TL_LM_SEGMENT_FINISHED,
             "\"lm_steps\":" + std::to_string(r->lm_steps) + ",\"segment_index\":" + std::to_string(r->job) +
                 ",\"finish_reason\":\"" + lm::finish_name(r->finish) + "\",\"audio_tokens\":" +
                 std::to_string(r->audio.size()),
             r->job);
    if (r->req->demo) {
      e->dw_queue.push_back({r->key, mono_s()});
      // Prefetch at language-model completion: acknowledge at LM end so the next chunk may be admitted.
      Request* q = r->req;
      const bool last = q->c_sealed && size_t(j->ordinal) + 1 >= q->jobs.size();
      if (!last && e->inflight_segments + 1 + 1 <= e->lv.request_batch_size) acknowledge(e, q, j);
    } else {
      stream_finals.push_back(r);
    }
  }
  if (stream_finals.empty()) return;
  // Group by geometry owner, bucket by length.
  std::map<int64_t, std::vector<Row*>> by_owner;
  for (Row* r : stream_finals) by_owner[r->req->voice->geometry].push_back(r);
  const bool async = pol::launch_async(decoded, e->lv);
  uint64_t calls = 0;
  for (auto& [owner, rows] : by_owner) {
    std::vector<std::pair<int, int>> items;
    for (size_t i = 0; i < rows.size(); ++i) items.push_back({int(i), int(rows[i]->audio.size())});
    for (const auto& b : pol::final_flush_buckets(items, e->lv)) {
      std::vector<Row*> batch;
      for (int i : b) batch.push_back(rows[size_t(i)]);
      if (e->hold_urgent >= 2 || e->buffer_urgent_ms > 0) {
        // KT2N_HOLD_URGENT=2: finals of requests whose startup hold has not released go to S_urgent (one call), the
        // rest of the bucket stays one S_bulk call. Same rows, same buckets, same turn. KT2N_BUFFER_URGENT_MS: the
        // same for finals of playing streams whose buffer is low.
        std::vector<Row*> held, rest;
        int low = 0;
        for (Row* r : batch) {
          const bool h = e->hold_urgent >= 2 && r->req->awaiting_hold.load(std::memory_order_relaxed);
          const bool lowbuf = !h && !e->buffer_to_aux && buffer_low(e, r->req);
          low += lowbuf;
          (h || lowbuf ? held : rest).push_back(r);
        }
        if (low) {
          e->buffer_urgent_rows += uint64_t(low);
          e->buffer_urgent_finals += uint64_t(low);
        }
        if (!held.empty()) {
          e->hold_urgent_rows += uint64_t(held.size()) - uint64_t(low);
          e->hold_urgent_finals += uint64_t(held.size()) - uint64_t(low);
          if (!rest.empty()) e->hold_urgent_splits++;
          launch(e, held, render::Phase::kFinalFlush, async, true, true);
          ++calls;
          if (rest.empty()) continue;
          batch.swap(rest);
        }
      }
      launch(e, batch, render::Phase::kFinalFlush, async, true);
      ++calls;
    }
  }
  if (calls) {
    e->final_turns++;
    e->final_calls += calls;
    e->final_extra_calls += calls - 1;
  }
}

// One owner turn after one decode step (owner loop steps 10-13).
void turn(Engine* e, const std::vector<lm::TokenEvent>& step) {
  const double now = mono_s();
  int decoded = 0;
  for (const auto& ev : step) {
    auto it = e->rows.find(ev.tag);
    if (it == e->rows.end()) continue;
    Row* r = it->second.get();
    if (!r->in_active) continue;
    if (ev.finish != lm::Finish::kCancelled) ++decoded;  // rows decoded this turn
    r->lm_steps = ev.index;
    if (!r->first_token_marked) {
      r->first_token_marked = true;
      timeline(e, r->req, KT2N_TL_LM_PREFILL_FINISHED,
               "\"batch_size\":1,\"segment_index\":" + std::to_string(r->job) +
                   ",\"lm_wall_ms\":" + jnum((now - r->admitted_s) * 1000.0),
               r->job);
    }
    if (ev.finish == lm::Finish::kCancelled) {
      r->finished = true;
      r->finish = ev.finish;
      continue;
    }
    if (r->lm_head.size() < 64) r->lm_head.push_back(ev.token - lm::kAudioIdBase);
    if (ev.token >= lm::kAudioIdBase && ev.token < lm::kAudioIdBase + lm::kAudioVocab) {
      const int32_t s3 = ev.token - lm::kAudioIdBase;
      const bool was_closed = r->skip.closed;
      const bool traced = !was_closed && r->req->timeline && r->skip_armed && r->audio.empty();
      std::vector<int32_t> skipped_before, pending_before;
      if (traced) {
        skipped_before = r->skip.skipped;
        pending_before = r->skip.pending;
      }
      pol::opening_admit(r->skip, r->skip_armed, s3, r->audio, e->lv);
      if (traced && r->skip.closed) {
        auto list = [](const std::vector<int32_t>& v) {
          std::string o = "[";
          for (size_t i = 0; i < v.size(); ++i) o += (i ? "," : "") + std::to_string(v[i]);
          return o + "]";
        };
        std::string in_set = "[";
        for (size_t i = 0; i < skipped_before.size(); ++i)
          in_set += std::string(i ? "," : "") + (pol::opening_quiet_id(skipped_before[i]) || skipped_before[i] == 1466 ||
                                                         skipped_before[i] == 3653
                                                     ? "true"
                                                     : "false");
        in_set += "]";
        timeline(e, r->req, KT2N_TL_N_OPENING_SKIP,
                 "\"cap\":" + std::to_string(e->lv.skip_cap) + ",\"skipped\":" + list(skipped_before) +
                     ",\"pending\":" + list(pending_before) + ",\"closing\":" + std::to_string(s3) +
                     ",\"skipped_in_set\":" + in_set + ",\"lm_head\":" + list(r->lm_head) +
                     ",\"segment_index\":" + std::to_string(r->job),
                 r->job);
      }
      if (!was_closed && r->skip.closed && r->job == 0) {
        // Known before any audio of the request exists (the opening closes before the vocoder sees a token);
        // the post stage applies it at its first item (post_item), after a post_push gave happens-before.
        r->req->opening_skipped = r->skip.skipped_tokens;
        r->req->kept_tokens = r->skip.kept_tokens;
      }
    }
    if (ev.finish != lm::Finish::kNone) {
      r->finished = true;
      r->finish = ev.finish;
      // Low-memory fit: the default profile's context (4,096) ends this row only at 2 x budget or at the demo-chunked
      // margin; a fitted context can be full first. Fail instead of silently ending the segment early.
      if (e->fit_ctx > 0 && ev.finish == lm::Finish::kCeiling && r->prompt_ids + r->lm_steps + 1 >= e->fit_ctx &&
          r->lm_steps < 2 * r->req->jobs[size_t(r->job)]->budget)
        fail_request(e, r->req, KT2N_ERR_GPU_MEMORY_LIMIT,
                     "this segment ran past its token budget to the language-model context the low-memory profile "
                     "fitted on this GPU (" + std::to_string(e->fit_ctx) + " positions)");
    } else if (!r->noise_extended && r->lm_steps >= r->req->jobs[size_t(r->job)]->budget) {
      // a row that outlives its budget gets flow noise for the extra frames.
      r->noise_extended = true;
      std::string err;
      e->render->extend_noise(r->key, &err);
    }
  }
  e->lm_steps++;
  // Active rows in admission order; cancelled rows leave (step 6).
  std::vector<Row*> act;
  for (uint64_t k : e->active) {
    Row* r = e->rows[k].get();
    // The reference adds an admitted row to `active` after its prefill (which samples its first token); LM
    // prefills beside decode, so a row joins the turn's decisions with its first token.
    if (!r->first_token_marked && !r->req->cancelled.load()) continue;
    if (r->req->cancelled.load() && !r->finished) {
      e->lm->cancel(r->key);
      r->finished = true;
      r->finish = lm::Finish::kCancelled;
    }
    if (r->finish == lm::Finish::kCancelled) continue;  // step 6: cancelled rows leave before any decision
    act.push_back(r);
  }
  // D steady count (one-commit steady age): live requests (a job not done) whose
  // listener has played for >= 1000 ms (the D join window).
  {
    int steady = 0;
    for (Request* q : e->lanes) {
      if (q->cancelled.load()) continue;
      std::lock_guard<std::mutex> ck(q->clock_mu);
      const auto first = q->clock.first_at();
      if (first && (now - *first) * 1000.0 >= 1000.0) ++steady;
    }
    e->steady = steady;
  }
  // Step 12: T9 rows.
  std::vector<pol::T9Row> t9;
  const int sharing_rows = int(act.size());
  for (Row* r : act) {
    Job* j = r->req->jobs[size_t(r->job)].get();
    pol::T9Row t;
    t.finished = r->finished;
    t.held_short = r->c.held;
    t.segment_index = j->ordinal;
    t.preview_ready = j->preview_ready;
    t.published_s = double(j->published_samples) / 24000.0;
    t.start_target_s = r->req->hold_pinned ? std::max(0.0, *r->req->hold_pinned)
                                           : pol::start_target_seconds(sharing_rows, r->req->realtime, e->lv);
    if (j->first_pub) t.reservoir_s = double(j->published_samples) / 24000.0 - (now - *j->first_pub);
    t.dropped_s = double(r->req->dropped_samples.load()) / 24000.0;
    const int n = int(r->audio.size());
    const bool prev = pol::preview_due(j->preview_ready, n, r->c.committed, r->c.held, r->c.next, r->c.commit_index,
                                       e->lv);
    t.row_ready = prev || pol::commit_ready(r->c, n);
    t.backlog = n - r->c.next - pol::kLookahead;
    t9.push_back(t);
  }
  const bool xsched = (e->flags & KT2N_CFG_X_SCHED) != 0;  // experimental option: no T9 deferral
  pol::T9Result sel;
  if (xsched) {
    for (int i = 0; i < int(t9.size()); ++i) sel.selected.push_back(i);
  } else {
    sel = pol::t9_acoustic_rows(t9, e->lv);
  }
  for (int i : sel.deferred_rows) act[size_t(i)]->t9_deferred_turns += 1;
  std::vector<Row*> rows;
  for (int i : sel.selected) {
    Row* r = act[size_t(i)];
    rows.push_back(r);
    if (r->t9_deferred_turns) {
      const auto& t = t9[size_t(i)];
      timeline(e, r->req, KT2N_TL_T9_DEFERRAL_ENDED,
               "\"segment_index\":" + std::to_string(r->job) + ",\"deferred_turns\":" +
                   std::to_string(r->t9_deferred_turns) + ",\"reservoir_s\":" +
                   (t.reservoir_s ? jnum(*t.reservoir_s) : std::string("null")),
               r->job);
      r->t9_deferred_turns = 0;
    }
  }
  // Previews: ported condition; no default path fires it. Native renders it
  // as a 1-step kPreview call once the decoder serves that phase; until then the condition is counted.
  for (Row* r : rows) {
    Job* j = r->req->jobs[size_t(r->job)].get();
    if (pol::preview_due(j->preview_ready, int(r->audio.size()), r->c.committed, r->c.held, r->c.next,
                         r->c.commit_index, e->lv))
      e->previews_due++;
  }
  // Step 12b: choose the row schedule on the rows passed to the acoustic step, sharing read once.
  const int sharing = startup_sharing(e);
  for (Row* r : rows) pol::d_prepare(r->c, int(r->audio.size()), sharing, e->lv);
  // Step 12c: ready commits -> groups -> order -> launch.
  std::vector<pol::CommitRow> ready, partners;
  std::unordered_map<int, Row*> by_id;
  int id = 0;
  for (Row* r : rows) {
    if (r->finish == lm::Finish::kCancelled) continue;
    pol::CommitRow cr;
    cr.id = id;
    by_id[id] = r;
    ++id;
    Job* j = r->req->jobs[size_t(r->job)].get();
    cr.first = r->c.commit_index == 0 && r->c.committed == 0;
    cr.commit_index = r->c.commit_index;
    cr.token_count = int(r->audio.size());
    cr.next = r->c.next;
    cr.geometry = r->req->voice->geometry;
    if (j->first_pub) cr.reservoir_s = double(j->published_samples) / 24000.0 - (now - *j->first_pub);
    cr.d_buffer_ms = d_buffer_ms(r->req, now);
    if (pol::commit_ready(r->c, cr.token_count))
      ready.push_back(cr);
    else if (!r->c.inflight && !r->c.held)
      partners.push_back(cr);
  }
  int held = 0;
  const auto groups = xsched ? pol::plan_commit_groups_xsched(ready, e->lv)
                             : pol::plan_commit_groups(ready, partners, e->lv, &held);
  const bool async = pol::launch_async(decoded, e->lv);
  if (!groups.empty()) {
    int per_stream[render::kStreamRoles] = {};
    for (const auto& g : groups) {
      bool all_first = true;
      int min_ci = 1 << 30, n_urgent = 0, n_later = 0;
      for (int rid : g.rows) {
        const Row* r = by_id[rid];
        const bool first = r->c.commit_index == 0 && r->c.committed == 0;
        all_first &= first;
        min_ci = std::min(min_ci, r->c.commit_index);
        (first || r->c.commit_index <= 1 ? n_urgent : n_later) += 1;
      }
      if (n_urgent && n_later) {
        e->commit_mixed_groups++;
        e->commit_mixed_later_rows += uint64_t(n_later);
      }
      const int s = int(render::stream_for(all_first ? render::Phase::kFirstCommit : render::Phase::kSteadyCommit, min_ci));
      if (per_stream[s]++ > 0) {
        e->commit_same_stream_extra++;
        e->commit_extra_rows += g.rows.size();
      }
    }
    e->commit_turns++;
    e->commit_groups += groups.size();
  }
  for (const auto& g : groups) {
    std::vector<Row*> batch;
    for (int rid : g.rows) batch.push_back(by_id[rid]);
    if (e->buffer_to_aux && e->buffer_urgent_ms > 0) {
      // Two-level arm: low-buffer rows of playing streams (not commit 0 / 1, not held) leave the group as one S_aux
      // call at the middle priority; the rest of the group launches as it would without them.
      std::vector<Row*> mid, rest;
      for (Row* r : batch) {
        const bool natural = (r->c.commit_index == 0 && r->c.committed == 0) || r->c.commit_index <= 1;
        const bool in_hold = e->hold_urgent > 0 && r->req->awaiting_hold.load(std::memory_order_relaxed);
        (!natural && !in_hold && buffer_low(e, r->req) ? mid : rest).push_back(r);
      }
      if (!mid.empty()) {
        e->buffer_urgent_rows += uint64_t(mid.size());
        e->buffer_mid_calls++;
        launch(e, mid, render::Phase::kSteadyCommit, async, false, false, int(render::StreamRole::kSAux));
        batch.swap(rest);
        if (batch.empty()) continue;
      }
    }
    if (e->split_urgent || e->hold_urgent > 0 || e->buffer_urgent_ms > 0) {
      // urgent = commit 0 / 1 rows (stream_for) + with KT2N_HOLD_URGENT the rows of requests still under the startup
      // hold, + with KT2N_BUFFER_URGENT_MS the rows of playing streams whose buffer is low (promoted); later = the rest.
      std::vector<Row*> urgent, later;
      int natural = 0, promoted = 0, low = 0;
      for (Row* r : batch) {
        const bool first = r->c.commit_index == 0 && r->c.committed == 0;
        if (first || r->c.commit_index <= 1) {
          ++natural;
          urgent.push_back(r);
        } else if (e->hold_urgent > 0 && r->req->awaiting_hold.load(std::memory_order_relaxed)) {
          ++promoted;
          urgent.push_back(r);
        } else if (!e->buffer_to_aux && buffer_low(e, r->req)) {
          ++promoted;
          ++low;
          urgent.push_back(r);
        } else {
          later.push_back(r);
        }
      }
      if (low) e->buffer_urgent_rows += uint64_t(low);
      if (promoted > low) e->hold_urgent_rows += uint64_t(promoted - low);
      // Split when the arm asks for it (split_urgent), or when only promoted rows make the group urgent (a commit-0/1
      // row already puts the whole call on S_urgent, as without the arm).
      if (!urgent.empty() && !later.empty() && (e->split_urgent || natural == 0)) {
        if (e->split_urgent) e->split_urgent_calls++;
        if (natural == 0) e->hold_urgent_splits++;
        launch(e, urgent, render::Phase::kSteadyCommit, async, false, promoted > 0 && natural == 0);
        launch(e, later, render::Phase::kSteadyCommit, async, false);
        continue;
      }
      if (later.empty() && natural == 0 && promoted > 0) {
        launch(e, batch, render::Phase::kSteadyCommit, async, false, true);
        continue;
      }
    }
    launch(e, batch, render::Phase::kSteadyCommit, async, false);
  }
  // Step 13: finished rows -> finals; they leave `active`.
  std::vector<Row*> done;
  std::vector<uint64_t> still;
  for (uint64_t k : e->active) {
    Row* r = e->rows[k].get();
    if (r->finished) {
      if (r->finish == lm::Finish::kCancelled) {
        r->in_active = false;
        r->final_launched = true;
        close_row_if_done(e, r);
      } else {
        done.push_back(r);
      }
    } else {
      still.push_back(r->key);
    }
  }
  e->active.swap(still);
  finalize_rows(e, done, decoded);
  e->lm_rows = int(e->active.size());
}

void build_emit_locked(Engine* e, BuildJob* j, int status, uint64_t voice, const std::string& code,
                       const std::string& param, const std::string& message);

// Close admission atomically with the snapshot of accepted requests. The conductor
// stays alive after an LM failure to drain renders and reclaim released owners.
void fail_engine(Engine* e, const std::string& reason) {
  {
    std::lock_guard<std::mutex> lk(e->mu);
    e->fatal = reason;
    e->backend_failed = true;
    for (auto& [id, q] : e->requests) {
      fail_request(e, q.get(), KT2N_ERR_SERVICE_UNAVAILABLE, reason);
      q->in_lanes = false;
    }
    e->incoming.clear();
    for (const auto& [voice, jobs] : e->pending_voice_builds)
      for (const auto& job : jobs)
        build_emit_locked(e, job.get(), KT2N_ERR_SERVICE_UNAVAILABLE, 0, "", "", reason);
    e->pending_voice_builds.clear();
    e->admin_q.clear();
  }
  if (e->vcache) e->vcache->clear_pending();  // loads already running finish (destroy waits for them)
  e->lanes.clear();
  e->active.clear();
  e->dw_queue.clear();
  e->lm_rows = 0;
  e->queued_requests = 0;
  // These admission paths take their wait mutex before mu, so never nest them
  // inside mu here. Waking with the wait mutex prevents a missed notification.
  {
    std::lock_guard<std::mutex> lk(e->build_mu);
    e->build_cv.notify_all();
  }
  {
    std::lock_guard<std::mutex> lk(e->sess_mu);
    e->sess_cv.notify_all();
  }
}

void drain_failed_rows(Engine* e) {
  // Keep a row until every already launched render call has been collected.
  // Cancelled publications need no ordering or final flush after LM failure.
  std::set<uint64_t> rendering;
  for (const auto& [id, call] : e->inflight)
    for (const auto& row : call.rows) rendering.insert(row.row);
  for (auto& [id, row] : e->rows) {
    row->in_active = false;
    row->final_launched = true;
    row->reorder.clear();
    row->pub_seq = row->next_seq;
    if (!rendering.count(id)) close_row_if_done(e, row.get());
  }
  drop_closed_rows(e);
}

// A saturated GPU has no spare throughput: this only tests whether bounded LM priority relief reduces the long
// bulk-render waits. Existing runtime switching orders later work with an event; it does not preempt queued work.
// The eight-millisecond limit is a host-request interval, not a guarantee of eight milliseconds on the GPU.
void maybe_yield_lm_to_render(Engine* e) {
  if (!e->lm_render_yield) return;
  const double now = mono_s();
  if (e->lm_render_yield_active) {
    if (e->inflight.count(e->lm_render_yield_call) && now - e->lm_render_yield_since < 0.008) return;
    e->lm->set_decode_priority(e->lm_render_yield_base_priority);
    e->lm_render_yield_active = false;
    e->lm_render_yield_cooldown = now + 0.008;
    ++e->lm_render_yield_switches;
    return;
  }
  if (now < e->lm_render_yield_cooldown || now - e->lm_render_yield_checked < 0.001) return;
  e->lm_render_yield_checked = now;
  uint64_t oldest_id = 0;
  double oldest_s = now;
  for (const auto& [id, call] : e->inflight) {
    if (call.stream != render::StreamRole::kSBulk || now - call.launched_s < 0.080) continue;
    bool low_buffer = false;
    for (const auto& ref : call.rows) {
      const auto row = e->rows.find(ref.row);
      if (row == e->rows.end()) continue;
      Request* q = row->second->req;
      if (q->demo || q->awaiting_hold.load(std::memory_order_relaxed)) continue;
      const auto b = d_buffer_ms(q, now);
      if (b && *b < 1000.0) {
        low_buffer = true;
        break;
      }
    }
    if (low_buffer && call.launched_s < oldest_s) {
      oldest_id = id;
      oldest_s = call.launched_s;
    }
  }
  if (!oldest_id) return;
  e->lm->set_decode_priority(0);  // equal to S_bulk; first/held renders and prefill keep their existing priorities
  e->lm_render_yield_active = true;
  e->lm_render_yield_call = oldest_id;
  e->lm_render_yield_since = now;
  ++e->lm_render_yield_episodes;
  ++e->lm_render_yield_switches;
}

void conductor(Engine* e) {
  std::vector<lm::TokenEvent> events;
  while (!e->stop.load()) {
    // Step 1: collect finished renders (publish in row order).
    collect_renders(e, false);
    if (e->backend_failed.load()) drain_failed_rows(e);
    // Free released requests the conductor and the post stage no longer touch.
    if (e->release_pending.exchange(false)) {
      std::lock_guard<std::mutex> lk(e->mu);
      bool again = false;
      for (auto it = e->requests.begin(); it != e->requests.end();) {
        Request* q = it->second.get();
        bool idle = false;
        if (q->release_requested.load() && !q->in_lanes && q->live_rows == 0 && q->text_tasks == 0) {
          std::lock_guard<std::mutex> qlk(q->mu);
          idle = !q->post_scheduled && q->post_q.empty();
        }
        if (idle) {
          if (e->vcache) cache_unpin(e, q, true);  // normally released already (generation complete, no rows)
          if (--q->voice->refs == 0 && q->voice->evicted.load()) e->voice_sweep = true;
          it = e->requests.erase(it);
        } else {
          again |= q->release_requested.load();
          ++it;
        }
      }
      if (again) e->release_pending = true;
    }
    if (e->backend_failed.load()) {
      // No more LM calls, admissions or builds; retain the release sweep above.
      std::unique_lock<std::mutex> lk(e->mu);
      cv_wait_for(e->cv, lk, std::chrono::milliseconds(5));
      continue;
    }
    if (e->voice_sweep.exchange(false)) release_voices(e);
    // Deadlines and cancels of requests without rows.
    const int64_t now_ns = mono_ns();
    for (Request* q : e->lanes) {
      if (q->deadline_ns && now_ns > q->deadline_ns && !q->cancelled.load()) {
        q->cancel_status = KT2N_ERR_DEADLINE_EXCEEDED;
        q->cancelled = true;
      }
      if (q->cancelled.load() && !q->generation_complete) {
        q->generation_complete = true;
        PostItem it;
        it.end = true;
        it.status = q->cancel_status;
        post_push(e, q, std::move(it));
      }
    }
    // Step 2: non-blocking admin jobs (voice preparation: LM prefix builds of loaded voices).
    {
      std::deque<std::function<void()>> jobs;
      {
        std::lock_guard<std::mutex> lk(e->mu);
        jobs.swap(e->admin_q);
      }
      for (auto& j : jobs) j();
    }
    // Voice cache (LRU / NONE): finish loads, drop voices, start queued loads.
    if (e->vcache) e->vcache->pump();
    // Step 3: the interleaved clone build, if due.
    {
      std::lock_guard<std::mutex> lk(e->build_mu);
      if (e->build_waiting && !e->build_go) {
        const double now = mono_s();
        bool before_first = false;
        for (uint64_t k : e->active) {
          Row* r = e->rows[k].get();
          if (r->req->demo) continue;  // the waiting realtime rows exclude whole-render rows
          if (!r->req->jobs[size_t(r->job)]->preview_ready) before_first = true;
        }
        std::optional<double> since;
        if (e->last_build_done_s) since = (now - *e->last_build_done_s) * 1000.0;
        if (pol::cf_interleave_due(!e->active.empty(), before_first, (now - e->build_wait_since) * 1000.0, since,
                                   e->lv)) {
          e->build_go = true;
          e->build_cv.notify_all();
        }
      }
    }
    // Step 4: whole-render gate.
    dw_service(e);
    // Step 8-9: admit (the LM prefills beside decode).
    admit(e);
    for (Request* q : e->lanes) maybe_normalize_next(e, q);
    // Step 10: LM. Block for a step only when rows are live; else wait for renders or new work.
    events.clear();
    std::string err;
    // One reference decision pass per LM step: a tag seen twice starts the next step.
    auto run_steps = [&] {
      size_t begin = 0;
      std::set<uint64_t> seen;
      for (size_t i = 0; i <= events.size(); ++i) {
        if (i == events.size() || seen.count(events[i].tag)) {
          std::vector<lm::TokenEvent> step(events.begin() + long(begin), events.begin() + long(i));
          if (!step.empty()) {
            const int64_t t0 = mono_ns();
            turn(e, step);
            const int64_t t1 = mono_ns();
            collect_renders(e, false);
            const uint64_t dt = uint64_t(mono_ns() - t0);
            e->turn_ns += dt;
            e->turns++;
            // Stall diagnostics: how often a turn (every stream's next LM step waits on it) is long, and for the
            // longest one, how it split between the decision pass + launches (turn) and the collect.
            if (dt > 1000000) e->turns_over_1ms++;
            if (dt > 5000000) e->turns_over_5ms++;
            if (dt > e->turn_ns_max.load()) {
              e->turn_ns_max = dt;
              e->turn_max_decide_ns = uint64_t(t1 - t0);
              e->turn_max_rows = int(step.size());
            }
          }
          seen.clear();
          begin = i;
        }
        if (i < events.size()) seen.insert(events[i].tag);
      }
    };
    const bool lm_live = e->lm->live_rows() > 0 || !e->active.empty();
    if (!lm_live) e->lm_prev_ev_ns = 0;
    maybe_yield_lm_to_render(e);
    // Steps in `events` (a tag seen twice starts the next step) -> the LM step-interval stats.
    auto note_lm_interval = [&] {
      size_t steps = 1;
      std::set<uint64_t> seen;
      for (const auto& ev : events) {
        if (seen.count(ev.tag)) {
          ++steps;
          seen.clear();
        }
        seen.insert(ev.tag);
      }
      const int64_t now = mono_ns();
      if (e->lm_prev_ev_ns > 0) {
        const double rows = double(events.size()) / double(steps);
        const int b = rows <= 4    ? 0
                      : rows <= 8  ? 1
                      : rows <= 12 ? 2
                      : rows <= 16 ? 3
                      : rows <= 20 ? 4
                      : rows <= 32 ? 5
                      : rows <= 48 ? 6
                                   : 7;
        e->lm_iv_ns[b] += uint64_t(now - e->lm_prev_ev_ns);
        e->lm_iv_steps[b] += steps;
      }
      e->lm_prev_ev_ns = now;
    };
    if (e->poll && (lm_live || e->render->inflight() > 0)) {
      // Poll mode (execution only): never block inside one backend. The LM is pumped without waiting, and when
      // no step finished the loop goes round again after a short pause, so a finished render is published and a
      // new request admitted within ~25 us instead of after the next LM step (or after a blocking render wait).
      if (lm_live && e->lm->pump(&events, false, &err) < 0) {
        fail_engine(e, err);
        continue;
      }
      if (!events.empty()) note_lm_interval();
      if (events.empty()) {
        e->poll_idle++;
        for (int i = 0; i < 64; ++i) {  // ~5-10 us of pause, then a 20 us sleep
#if defined(__x86_64__)
          __builtin_ia32_pause();
#endif
        }
        const timespec ts{0, 20000};
        nanosleep(&ts, nullptr);
        continue;
      }
      run_steps();
    } else if (lm_live) {
      if (e->lm->pump(&events, true, &err) < 0) {
        fail_engine(e, err);
        continue;
      }
      if (!events.empty()) note_lm_interval();
      if (events.empty()) {
        // Nothing decoded (rows still prefilling, or none live after all): never spin.
        if (e->render->inflight() > 0) {
          collect_renders(e, true);
        } else {
          std::unique_lock<std::mutex> lk(e->mu);
          cv_wait_for(e->cv, lk, std::chrono::microseconds(200));
        }
        continue;
      }
      run_steps();
    } else if (e->render->inflight() > 0) {
      collect_renders(e, true);
    } else {
      std::unique_lock<std::mutex> lk(e->mu);
      bool work = !e->incoming.empty() || !e->admin_q.empty();
      for (Request* q : e->lanes)
        work |= (q->next_index < q->jobs.size() && (!e->vcache || voice_usable(e, q->voice))) ||
                !q->new_jobs.empty() || q->cancelled.load();
      if (!work && !e->dw_queue.empty()) work = true;
      if (!work && e->vcache && e->vcache->busy())
        cv_wait_for(e->cv, lk, std::chrono::microseconds(100));  // a voice load in flight: check it again soon
      else if (!work)
        cv_wait_for(e->cv, lk, std::chrono::milliseconds(5));
    }
  }
}

}  // namespace
}  // namespace kt2n::engine

namespace kt2n::engine {
namespace {

Request* register_request(Engine* e, std::unique_ptr<Request> q, uint64_t* id = nullptr) {
  Request* raw = q.get();
  {
    std::lock_guard<std::mutex> lk(e->mu);
    if (e->backend_failed.load()) return nullptr;
    q->id = e->next_id++;
    if (!q->session) q->ev_id = q->id;
    e->requests.emplace(q->id, std::move(q));
    e->incoming.push_back(raw);
    if (id) *id = raw->id;
    e->active_requests++;
    raw->voice->refs++;
    timeline(e, raw, KT2N_TL_GENERATE_STREAM_ENTERED, "");
    text_task_locked(e, raw, [e, raw] { plan_request(e, raw); });
  }
  e->cv.notify_all();
  return raw;
}

// mu is the common fatal/admission/publication boundary. The caller retains a
// shared owner through asynchronous voice preparation and the final callback.
void build_emit_locked(Engine* e, BuildJob* j, int status, uint64_t voice, const std::string& code,
                       const std::string& param, const std::string& message = "") {
  std::lock_guard<std::mutex> lk(j->ev_lock);
  if (j->terminal) return;
  if (status == KT2N_OK && e->backend_failed.load()) {
    status = KT2N_ERR_SERVICE_UNAVAILABLE;
    voice = 0;
  }
  auto* r = new EventRec();
  r->ev.kind = KT2N_EVENT_VOICE_BUILT;
  r->ev.status = status;
  r->ev.user_data = j->user_data;
  r->ev.at_ns = mono_ns();
  r->ev.segment_index = -1;
  r->ev.phrase_index = -1;
  r->ev.sample_begin = -1;
  r->ev.engine_chunk_index = -1;
  r->ev.voice = voice;
  if (status != KT2N_OK) {
    set_err(&r->error, status, message.empty() ? code : message, param);
    std::snprintf(r->error.code, sizeof r->error.code, "%s", code.empty() ? kt2n_error_lookup(status)->code : code.c_str());
    r->ev.error = &r->error;
  }
  j->terminal = true;
  r->ev.id = j->id;
  r->ev.seq = j->seq++;
  push_event(e, r);
}

void build_emit(Engine* e, BuildJob* j, int status, uint64_t voice, const std::string& code,
                const std::string& param, const std::string& message = "") {
  std::lock_guard<std::mutex> lk(e->mu);
  build_emit_locked(e, j, status, voice, code, param, message);
}

// mu held. Loaded voices return an id immediately; builds complete only after
// both model stores are initialized. Fatal closure finishes all pending owners.
uint64_t register_voice_locked(Engine* e, std::unique_ptr<Voice> v, std::vector<int32_t> prompt_token,
                               std::vector<float> prompt_feat, std::vector<float> embedding,
                               std::vector<float> projected, std::vector<int32_t> lm_prefix,
                               const std::shared_ptr<BuildJob>& job = {}) {
  if (e->backend_failed.load()) return 0;
  Voice* raw = v.get();
  uint64_t id;
  {
    std::lock_guard<std::mutex> lk(e->voices_mu);
    e->voices.push_back(std::move(v));
    id = e->voices.size();
  }
  if (job) e->pending_voice_builds[id].push_back(job);
  if (e->vcache) {
    // Voice cache (LRU / NONE): the record stays on the host and the cache prepares the GPU state when a request uses
    // the voice (LRU: also right away, without dropping anything). Registration runs on the conductor (cache owner).
    auto host = std::make_shared<HostVoice>();
    host->prompt_token = std::move(prompt_token);
    host->prompt_feat = std::move(prompt_feat);
    host->embedding = std::move(embedding);
    host->projected = std::move(projected);
    host->lm_prefix = std::move(lm_prefix);
    e->admin_q.push_back([e, raw, id, host = std::shared_ptr<const HostVoice>(std::move(host))] {
      const VoiceCost c = voice_cost(int(host->prompt_token.size()), host->lm_prefix.size());
      const bool fits = e->vcache->fits_alone(c);
      std::string error;
      if (fits) {
        // Always registered: a request may already hold the voice (load, submit, evict before this ran), and it
        // finishes with it. A voice evicted before this ran is not warmed (nothing may need it).
        const int k = cache_add(e, raw, host);
        if (!raw->evicted.load()) e->vcache->warm(k);
      } else {
        error = "the voice needs " + std::to_string(c.total() >> 20) + " MiB of GPU memory, more than the voice cache budget";
      }
      std::lock_guard<std::mutex> lk(e->mu);
      if (!fits) {
        raw->cache_key = -2;
        raw->evicted = true;
        e->voice_store_full++;
        e->voice_sweep = true;
      }
      auto pending = e->pending_voice_builds.find(id);
      if (pending != e->pending_voice_builds.end()) {
        const int status = e->backend_failed.load() ? KT2N_ERR_SERVICE_UNAVAILABLE
                           : !fits                  ? KT2N_ERR_CUSTOM_VOICE_SERVICE_ERROR
                                                    : KT2N_OK;
        for (const auto& owner : pending->second)
          build_emit_locked(e, owner.get(), status, status == KT2N_OK ? id : 0, "", fits ? "" : "voice", error);
        e->pending_voice_builds.erase(pending);
      }
    });
    e->cv.notify_all();
    return id;
  }
  e->admin_q.push_back([e, raw, id, pt = std::move(prompt_token), pf = std::move(prompt_feat),
                        em = std::move(embedding), pj = std::move(projected), lp = std::move(lm_prefix)] {
    std::string error;
    const int render_voice = e->render->add_voice(pt, pf, em, &error);
    const int lm_voice = render_voice >= 0 ? e->lm->add_voice(pj, lp, &error) : -1;
    std::lock_guard<std::mutex> lk(e->mu);
    raw->render_voice = render_voice;
    raw->lm_voice = lm_voice;
    if (lm_voice < 0) {
      raw->evicted = true;
      e->voice_store_full++;
    }
    if (raw->evicted.load()) e->voice_sweep = true;
    auto pending = e->pending_voice_builds.find(id);
    if (pending != e->pending_voice_builds.end()) {
      const int status = e->backend_failed.load() ? KT2N_ERR_SERVICE_UNAVAILABLE
                         : lm_voice < 0 ? KT2N_ERR_CUSTOM_VOICE_SERVICE_ERROR : KT2N_OK;
      for (const auto& owner : pending->second)
        build_emit_locked(e, owner.get(), status, status == KT2N_OK ? id : 0, "", "", error);
      e->pending_voice_builds.erase(pending);
    }
  });
  e->cv.notify_all();
  return id;
}

uint64_t register_voice(Engine* e, std::unique_ptr<Voice> v, std::vector<int32_t> prompt_token,
                        std::vector<float> prompt_feat, std::vector<float> embedding,
                        std::vector<float> projected, std::vector<int32_t> lm_prefix) {
  std::lock_guard<std::mutex> lk(e->mu);
  return register_voice_locked(e, std::move(v), std::move(prompt_token), std::move(prompt_feat),
                               std::move(embedding), std::move(projected), std::move(lm_prefix));
}

#if KT2N_ENGINE_WITH_VOICE
// Marks voices the inline index dropped (kt2n_voice_evict semantics: requests holding them finish with them).
void evict_inline(Engine* e, const std::vector<uint64_t>& ids) {
  std::lock_guard<std::mutex> lk(e->voices_mu);
  for (uint64_t id : ids)
    if (id >= 1 && id <= e->voices.size()) e->voices[size_t(id - 1)]->evicted = true;
  if (!ids.empty()) e->voice_sweep = true;
}

void run_build(Engine* e, const std::shared_ptr<BuildJob>& owner) {
  BuildJob* j = owner.get();
  namespace kv = kt2n::voice;
  const bool inline_key = (j->flags & KT2N_VB_INLINE) && !j->identity_key.empty();
  // The engine index may contain a voice whose initialization is still pending.
  // Such hits join its completion owners rather than publishing an unusable id.
  {
    std::lock_guard<std::mutex> lk(e->mu);
    if (e->backend_failed.load()) {
      build_emit_locked(e, j, KT2N_ERR_SERVICE_UNAVAILABLE, 0, "", "");
      return;
    }
    if (inline_key && e->inline_index) {
      const uint64_t hit = e->inline_index->lookup(j->identity_key);
      bool live = false;
      if (hit) {
        std::lock_guard<std::mutex> vlk(e->voices_mu);
        live = !e->voices[size_t(hit - 1)]->evicted.load();
      }
      if (live) {
        auto pending = e->pending_voice_builds.find(hit);
        if (pending != e->pending_voice_builds.end()) pending->second.push_back(owner);
        else build_emit_locked(e, j, KT2N_OK, hit, "", "");
        return;
      }
      if (hit) e->inline_index->erase_voice(hit);
    }
  }
  kv::VoiceBuildRequest req;
  req.flags = j->flags;
  req.audio = j->audio.empty() ? nullptr : j->audio.data();
  req.audio_len = j->audio.size();
  req.pcm = j->pcm.empty() ? nullptr : j->pcm.data();
  req.pcm_samples = j->pcm.size();
  req.pcm_rate = j->pcm_rate;
  req.consent_speaker = j->consent.empty() ? nullptr : j->consent.data();
  req.reference_transcript = j->transcript;
  req.language = j->language;  // consent create: the reference language check (208) needs it
  req.identity_key = j->identity_key;
  if (j->deadline_ns) {
    const int64_t left = j->deadline_ns - mono_ns();
    req.deadline = kv::Clock::now() + std::chrono::nanoseconds(std::max<int64_t>(0, left));
  }
  req.cancelled = [e, j] { return e->backend_failed.load() || j->cancelled.load(); };
  // Clone interleave: the conductor's grant covers the GPU work only (in the reference, admission and the decode
  // happen before it). The cloning code calls gpu_turn after admission + decode, under the builder's GPU mutex.
  req.gpu_turn = [e, j]() -> bool {
    std::unique_lock<std::mutex> lk(e->build_mu);
    e->build_waiting = true;
    e->build_wait_since = mono_s();
    e->build_go = false;
    e->cv.notify_all();
    e->build_cv.wait(lk, [&] {
      return e->build_go || e->stop.load() || e->backend_failed.load() || j->cancelled.load();
    });
    e->build_waiting = false;
    e->build_go = false;
    return !(e->stop.load() || e->backend_failed.load() || j->cancelled.load());
  };
  req.gpu_done = [e] {
    std::lock_guard<std::mutex> lk(e->build_mu);
    e->last_build_done_s = mono_s();
  };
  kv::VoiceBuildOutcome out;
  try {
    out = e->voice_build_exec ? e->voice_build_exec(req) : e->vservice->build(req);
  } catch (const std::exception& ex) {
    out.status = KT2N_ERR_CUSTOM_VOICE_SERVICE_ERROR;
    out.code = "custom_voice_service_error";
  }
  std::lock_guard<std::mutex> completion(e->mu);
  if (e->backend_failed.load()) {
    build_emit_locked(e, j, KT2N_ERR_SERVICE_UNAVAILABLE, 0, "", "");
    return;
  }
  if (out.status == KT2N_ERR_CANCELLED) {
    build_emit_locked(e, j, KT2N_ERR_CANCELLED, 0, "", "");
    return;
  }
  if (out.status != 0 || !out.result) {
    build_emit_locked(e, j, out.status ? out.status : KT2N_ERR_CUSTOM_VOICE_SERVICE_ERROR, 0, out.code, out.param,
               out.message);
    return;
  }
  const kv::CloneResult& r = *out.result;
  auto v = std::make_unique<Voice>();
  v->kind = (j->flags & KT2N_VB_SAVED) ? KT2N_VOICE_SAVED : KT2N_VOICE_INLINE;
  v->id = j->identity_key.empty() ? ("clone-" + std::to_string(j->id)) : j->identity_key;
  v->names = {v->id};
  v->peak_gain_db = kv::kCustomPeakGainDb;
  v->prompt_tokens = int(r.decoder.prompt_token.size());
  v->prompt_frames = int(r.decoder.prompt_feat.size() / 80);
  v->lm_prefix_ids = int(r.lm_prefix.size());
  v->raw_speaker = r.raw_speaker;
  v->geometry = policy::geometry_owner(v->prompt_tokens, v->prompt_frames, int(r.decoder.xvector.size()));
  v->artifact = out.artifact;
  std::vector<float> projected;
  if (out.projected_speaker) projected.assign(out.projected_speaker, out.projected_speaker + 2048);
  uint64_t vid;
  if (j->flags & KT2N_VB_CONSENT) {
    // Consent recordings carry only the speaker embedding: no LM prefix, no render state.
    std::lock_guard<std::mutex> lk(e->voices_mu);
    v->consent_only = true;
    e->voices.push_back(std::move(v));
    vid = e->voices.size();
  } else {
    // A concurrent build of the same inline key may have registered first: serve that one (the reference
    // single flight), and do not register a duplicate.
    if (inline_key && e->inline_index) {
      const uint64_t first = e->inline_index->lookup(j->identity_key);
      if (first) {
        auto pending = e->pending_voice_builds.find(first);
        if (pending != e->pending_voice_builds.end()) pending->second.push_back(owner);
        else build_emit_locked(e, j, KT2N_OK, first, "", "");
        return;
      }
    }
    vid = register_voice_locked(e, std::move(v), r.decoder.prompt_token, r.decoder.prompt_feat, r.decoder.xvector,
                         std::move(projected), r.lm_prefix, owner);
  }
  if (!vid) {
    build_emit_locked(e, j, KT2N_ERR_SERVICE_UNAVAILABLE, 0, "", "");
    return;
  }
  if (inline_key && e->inline_index) {
    std::vector<uint64_t> evict;
    auto in_use = [e](uint64_t id) {
      std::lock_guard<std::mutex> lk(e->voices_mu);
      return e->pending_voice_builds.count(id) ||
             (id >= 1 && id <= e->voices.size() && e->voices[size_t(id - 1)]->refs.load() > 0);
    };
    vid = e->inline_index->insert(j->identity_key, vid, in_use, &evict);
    evict_inline(e, evict);
  }
  if (j->flags & KT2N_VB_CONSENT) build_emit_locked(e, j, KT2N_OK, vid, "", "");
}
#endif

void voice_worker(Engine* e) {
  for (;;) {
    std::shared_ptr<BuildJob> j;
    {
      std::unique_lock<std::mutex> lk(e->build_mu);
      e->build_cv.wait(lk, [&] { return e->stop.load() || !e->build_q.empty(); });
      if (e->stop.load()) return;
      j = e->build_q.front();
      e->build_q.pop_front();
      e->builds_running += 1;
    }
#if KT2N_ENGINE_WITH_VOICE
    if (e->backend_failed.load())
      build_emit(e, j.get(), KT2N_ERR_SERVICE_UNAVAILABLE, 0, "", "");
    else if (j->cancelled.load())
      build_emit(e, j.get(), KT2N_ERR_CANCELLED, 0, "", "");
    else
      run_build(e, j);
#else
    build_emit(e, j.get(), KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE, 0, "", "");
#endif
    std::lock_guard<std::mutex> lk(e->build_mu);
    e->builds_running -= 1;
  }
}

void session_emit(Engine* e, Session* s, int kind, int status, const std::string& message) {
  auto* r = new EventRec();
  r->ev.kind = kind;
  r->ev.status = status;
  r->ev.user_data = s->user_data;
  r->ev.at_ns = mono_ns();
  r->ev.segment_index = -1;
  r->ev.phrase_index = -1;
  r->ev.sample_begin = -1;
  r->ev.engine_chunk_index = -1;
  if (status != KT2N_OK) {
    set_err(&r->error, status, message);
    r->ev.error = &r->error;
  }
  std::lock_guard<std::mutex> lk(s->ev_lock);
  r->ev.id = s->id;
  r->ev.seq = s->seq++;
  push_event(e, r);
}

// A phrase from the planner -> one internal generation emitting as the session (sess_mu held).
void start_phrase(Engine* e, Session* s, const textplan::WsPlanner::Phrase& ph) {
  auto q = std::make_unique<Request>();
  q->user_data = s->user_data;
  q->flags = s->flags & ~uint32_t(KT2N_REQ_SEED);
  q->timeline = (s->flags & KT2N_REQ_TIMELINE) != 0;
  q->voice = s->voice;
  q->raw = textplan::to_utf8(ph.text);
  q->sampler = s->sampler;
  q->speed = s->speed;
  q->format = s->format;
  q->output = KT2N_OUTPUT_STREAM;
  q->deadline_ns = s->deadline_ns;
  q->started_s = mono_s();
  q->realtime = true;
  q->normalize = s->normalize;
  q->emotion_preset = ph.emotion;
  q->session = s;
  q->phrase_index = s->phrases + 1;  // the reference phrase index counts from 1
  q->ev_id = s->id;
  q->ev_lock = &s->ev_lock;
  q->ev_seq = &s->seq;
  // Phrase seed: the reference phrase control has request_id "<session request id>:phrase:<index>"
  // and the salt is request_seed_salt of it. Seeded runs (KT2N_REQ_SEED): seed + 1000 x (index-1).
  q->salt = s->seeded ? s->salt + uint64_t(s->phrases) * 1000ull
                      : policy::request_seed_salt(s->request_id + ":phrase:" + std::to_string(q->phrase_index));
  kt2n::post::ResponsePostConfig rc;
  rc.speed = s->speed;
  q->rpost = std::make_unique<kt2n::post::ResponsePost>(rc);
  kt2n::post::HoldPolicy hp;
  hp.pinned = 0.0;  // a realtime phrase pins startup_hold_seconds = 0; the WS reserve lives in the wrapper
  q->hold = std::make_unique<kt2n::post::StartupHold>(hp);
  q->awaiting_hold.store(true, std::memory_order_relaxed);  // until its first publication releases the pinned-0 hold
  q->peak_ptr = &*s->peak;
  s->phrases += 1;
  s->phrase_running = true;
  Request* raw = register_request(e, std::move(q));
  if (!raw) {
    s->phrase_running = false;
    s->cancelled = true;
    s->cancel_status = KT2N_ERR_SERVICE_UNAVAILABLE;
    return;
  }
  s->phrase_ids.push_back(raw->id);
  timeline(e, raw, KT2N_TL_API_PHRASE_GENERATION_STARTED,
           "\"phrase_index\":" + std::to_string(raw->phrase_index) + ",\"characters\":" + std::to_string(ph.text.size()));
}

void session_phrase_done(Engine* e, Request* q) {
  // Called from emit_done with q->mu held: never take sess_mu here (sess_mu -> e->mu -> q->mu order).
  q->session->phrase_running = false;
  q->session->dirty = true;
  e->sess_cv.notify_all();
}

void session_thread(Engine* e) {
  std::unique_lock<std::mutex> lk(e->sess_mu);
  while (!e->stop.load()) {
    double next_wake = 50.0;  // ms
    for (auto& [id, sp] : e->sessions) {
      Session* s = sp.get();
      if (s->terminal) continue;
      if (e->backend_failed.load()) {
        s->cancelled = true;
        s->cancel_status = KT2N_ERR_SERVICE_UNAVAILABLE;
      }
      const double now_ms = (mono_s() - s->opened_s) * 1000.0;
      if (!s->cancelled && s->deadline_ns && mono_ns() > s->deadline_ns) {
        // The session deadline (as in the reference) ends it: deadline_exceeded.
        s->cancelled = true;
        s->cancel_status = KT2N_ERR_DEADLINE_EXCEEDED;
        std::lock_guard<std::mutex> elk(e->mu);
        for (uint64_t pid : s->phrase_ids) {
          auto it = e->requests.find(pid);
          if (it != e->requests.end() && !it->second->cancelled.load()) {
            it->second->cancel_status = KT2N_ERR_DEADLINE_EXCEEDED;
            it->second->cancelled = true;
          }
        }
        e->cv.notify_all();
      }
      if (s->cancelled) {
        if (!s->phrase_running) {
          s->terminal = true;
          s->ended = true;
          session_emit(e, s, KT2N_EVENT_DONE, s->cancel_status,
                       s->cancel_status == KT2N_ERR_INPUT_ALREADY_CLOSED ? "Realtime TTS input is already closed"
                       : s->cancel_status == KT2N_ERR_SPEECH_GENERATION_FAILED ? "Speech generation failed"
                                                                               : "request_cancelled");
        }
        continue;
      }
      if (!s->dirty && now_ms < s->wake_ms) {
        next_wake = std::min(next_wake, s->wake_ms - now_ms);
        continue;
      }
      s->dirty = false;
      if (s->planner_done) {
        if (!s->phrase_running) {
          s->terminal = true;
          s->ended = true;
          session_emit(e, s, KT2N_EVENT_DONE, KT2N_OK, "");
        }
        continue;
      }
      for (int guard = 0; guard < 64; ++guard) {
        auto act = s->planner.step(now_ms, !s->phrase_running);
        if (act.kind == textplan::WsPlanner::Action::Kind::kPhrase) {
          start_phrase(e, s, act.phrase);
          if (s->cancelled) break;
          continue;  // the next step waits for the slot
        }
        if (act.kind == textplan::WsPlanner::Action::Kind::kDone) {
          s->planner_done = true;
          if (!s->phrase_running) {
            s->terminal = true;
            s->ended = true;
            session_emit(e, s, KT2N_EVENT_DONE, KT2N_OK, "");
          }
          break;
        }
        s->wake_ms = std::isinf(act.until_ms) ? 1e18 : act.until_ms;
        if (!std::isinf(act.until_ms)) next_wake = std::min(next_wake, std::max(0.0, act.until_ms - now_ms));
        break;
      }
    }
    cv_wait_for(e->sess_cv, lk, std::chrono::microseconds(int64_t(std::max(0.2, next_wake) * 1000.0)));
  }
}

Session* session_of(Engine* e, uint64_t id) {
  auto it = e->sessions.find(id);
  return it == e->sessions.end() ? nullptr : it->second.get();
}

}  // namespace
}  // namespace kt2n::engine

// =====================================================================================================
// C ABI
// =====================================================================================================
using namespace kt2n::engine;

extern "C" {

KT2N_API const kt2n_error_info* kt2n_error_lookup(int32_t status) {
  static kt2n_error_info table[sizeof(kErrors) / sizeof(kErrors[0])];
  static std::once_flag once;
  std::call_once(once, [] {
    for (size_t i = 0; i < sizeof(kErrors) / sizeof(kErrors[0]); ++i)
      table[i] = {kErrors[i].status, kErrors[i].http, kErrors[i].code, kErrors[i].type};
  });
  for (const auto& t : table)
    if (t.status == status) return &t;
  for (const auto& t : table)
    if (t.status == KT2N_ERR_INTERNAL) return &t;
  return &table[0];
}

KT2N_API uint32_t kt2n_abi_version(void) { return KT2N_ABI_VERSION; }

KT2N_API const char* kt2n_build_info(void) {
  // Capability tokens (engine.h): a caller can refuse the skip-aware WS hold without the first one.
  return "Kitten TTS 2 native " KT2N_VERSION_STRING " " KT2N_CAP_AUDIO_OPENING_SKIP
#if KT2N_ENGINE_WITH_VOICE
      " " KT2N_CAP_VOICE_CLONING
#if !KT2N_VOICE_DECODE
      " " KT2N_CAP_VOICE_PCM_ONLY
#endif
#endif
      ;
}

static int text_threads(const kt2n_engine_config* cfg) { return cfg->threads_text > 0 ? cfg->threads_text : 4; }

// KT2N_MEMORY_PROFILE_AUTO (kt2n_engine_create below): the first attempt is a low-profile create run with this probe
// set. When the low profile's fit finds that its standard sizes fit and the free memory reaches the default profile's
// smallest possible need (see kAutoDefaultMargin), the default profile may fit too: the create stops there (before any
// model memory is allocated, *out stays NULL, status OK) and reports it here, so the caller can try the default
// profile. Otherwise the default profile cannot fit (when the fit has to shrink the sizes, its buffers are at least the
// standard low ones), so that create carries on as the low profile.
struct AutoProbe {
  bool reached = false;   // the fit decision ran
  bool stopped = false;   // stopped there (or before it, with no decoder to fit): the default profile may fit
  bool fitted = false;    // the fit shrank the sizes
  uint64_t free_bytes = 0, floor_bytes = 0;
  uint64_t std_need_bytes = 0;       // the fit's estimate for the standard low sizes
  uint64_t fit_free_bytes = 0;       // the free memory the fit used (free_bytes, or less with KT2N_LOW_FIT_FREE_MB)
  uint64_t default_floor_bytes = 0;  // the free-memory floor a default-profile create would keep
  // An explicit DEFAULT create's memory check (kt2n_engine_create): stop at the fit decision in every outcome, before
  // any model memory is allocated, or right after the CUDA start when the free memory is at least
  // kDefaultCheckBelow (the check is skipped there).
  bool decide_only = false;
  uint64_t default_pool_extra_bytes = 0;  // voice cache ALL: the default pool's every-voice prefix pages beyond low's
  // decide_only, once the check has read them: the text front end (tokenizer loaded, grammars not started) and the
  // parsed voices file, handed to the DEFAULT create that follows (CreateHandoff) so it does not read them again.
  std::unique_ptr<TextFront> text;
  std::optional<kt2n::voice::VoicesBlob> vblob;
};
// The DEFAULT create after its memory check (default_create_checked) takes over what the check read for the same
// config: the same tokenizer, grammar settings and voices file, so the same engine as reading them again.
struct CreateHandoff {
  std::unique_ptr<TextFront> text;
  std::optional<kt2n::voice::VoicesBlob> vblob;
};
thread_local CreateHandoff* g_create_handoff = nullptr;
// The default profile needs at least this much more than the low profile's standard-size estimate: measured
// 3.9 GB more on an RTX 5060 Ti (default 12,736 MiB with one voice vs an estimate of 8,827 MiB) — its vocoder handle for
// 32 rows x 1,024 frames, the token embedding on the GPU and larger workspaces, whatever the GPU. Below the estimate +
// this margin + the default profile's 10% floor, AUTO does not try the default profile at all (on a 12 GB RTX 3060
// a default create that fails only at its final memory check took 8.5 s).
constexpr uint64_t kAutoDefaultMargin = uint64_t(2) << 30;
// An explicit DEFAULT create checks its memory before allocating only below this much free memory: the default
// profile with every built-in voice prepared and one stream took 21,180 MiB on an RTX 4090, so on GPUs with more free
// memory it fits and the check (which reads the tokenizer and the voices file once more) is skipped.
constexpr uint64_t kDefaultCheckBelow = uint64_t(22) << 30;
thread_local AutoProbe* g_auto_probe = nullptr;

static int32_t engine_create_attempt(const kt2n_engine_config* cfg, kt2n_engine** out, kt2n_error* err) {
  if (!cfg || !out || cfg->struct_size < offsetof(kt2n_engine_config, voice_build_dir)) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "config is NULL or too small");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  if (cfg->abi_version != KT2N_ABI_VERSION) {
    set_err(err, KT2N_ERR_ABI_MISMATCH, "abi_version mismatch");
    return KT2N_ERR_ABI_MISMATCH;
  }
#if defined(_WIN32)
  // Paths are UTF-8 (engine.h). Most loaders open them through the C runtime, which reads them in the process code
  // page: refuse non-ASCII paths unless that code page is UTF-8, instead of failing later with "cannot open".
  if (!kt2n::platform::narrow_paths_are_utf8()) {
    // voice_build_dir only exists in configs at least that long (the same check as where it is read below).
    const char* vbd = cfg->struct_size >= offsetof(kt2n_engine_config, voice_build_dir) + sizeof(const char*)
                          ? cfg->voice_build_dir
                          : nullptr;
    for (const char* p :
         {cfg->lm_file, cfg->s3gen_file, cfg->voices_file, cfg->tokenizer_json, cfg->text_data_dir, vbd}) {
      for (const char* c = p; c && *c; ++c) {
        if (static_cast<unsigned char>(*c) >= 0x80) {
          set_err(err, KT2N_ERR_INVALID_ARGUMENT,
                  "a file path has non-ASCII characters, which need a UTF-8 process code page on Windows "
                  "(an application manifest with activeCodePage UTF-8, Windows 10 1903 or newer); or use an ASCII path");
          return KT2N_ERR_INVALID_ARGUMENT;
        }
      }
    }
  }
#endif
  *out = nullptr;
  const bool inference_only = (cfg->flags & KT2N_CFG_INFERENCE_ONLY) != 0;
  const int custom_capacity = cfg->struct_size >= offsetof(kt2n_engine_config, max_custom_voices) + sizeof(int32_t)
                                  ? cfg->max_custom_voices : 0;
  if (custom_capacity < 0 || custom_capacity > 4096) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "max_custom_voices must be 0..4096", "max_custom_voices");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  // Voice cache and memory controls: read only when struct_size covers the whole appended block (engine.h).
  const bool mem_block = cfg->struct_size >= offsetof(kt2n_engine_config, min_free_device_bytes) + sizeof(uint64_t);
  // Appended: the custom voices' LM prompt bound (low-memory profile only; 0 = the 1,024-id allowance).
  const int custom_prefix_ids =
      cfg->struct_size >= offsetof(kt2n_engine_config, custom_voice_max_prefix_ids) + sizeof(int32_t) &&
              mem_block && cfg->memory_profile == KT2N_MEMORY_PROFILE_LOW && cfg->custom_voice_max_prefix_ids > 0
          ? std::min(1024, cfg->custom_voice_max_prefix_ids)
          : 1024;
  int cache_mode = mem_block ? cfg->voice_cache_mode : KT2N_VOICE_CACHE_ALL;
  int cache_voices = mem_block ? cfg->voice_cache_max_voices : 0;
  uint64_t cache_bytes = mem_block ? cfg->voice_cache_max_bytes : 0;
  // Lab / test overrides (the C ABI fields are the interface): KT2N_VOICE_CACHE=all|lru|none,
  // KT2N_VOICE_CACHE_VOICES=N, KT2N_VOICE_CACHE_MB=N.
  if (const char* v = std::getenv("KT2N_VOICE_CACHE"); v && *v) {
    const std::string m = v;
    cache_mode = m == "lru" ? KT2N_VOICE_CACHE_LRU : m == "none" ? KT2N_VOICE_CACHE_NONE : m == "all" ? KT2N_VOICE_CACHE_ALL : -1;
  }
  if (const char* v = std::getenv("KT2N_VOICE_CACHE_VOICES"); v && *v) cache_voices = std::atoi(v);
  if (const char* v = std::getenv("KT2N_VOICE_CACHE_MB"); v && *v) cache_bytes = uint64_t(std::max(0ll, std::atoll(v))) << 20;
  if (cache_mode < KT2N_VOICE_CACHE_ALL || cache_mode > KT2N_VOICE_CACHE_NONE) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "voice_cache_mode must be KT2N_VOICE_CACHE_ALL, _LRU or _NONE", "voice_cache_mode");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  if (cache_voices < 0 || cache_voices > 4096) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "voice_cache_max_voices must be 0..4096", "voice_cache_max_voices");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  if (mem_block && (cfg->memory_profile < KT2N_MEMORY_PROFILE_DEFAULT || cfg->memory_profile > KT2N_MEMORY_PROFILE_LOW)) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "memory_profile must be KT2N_MEMORY_PROFILE_DEFAULT, _LOW or _AUTO",
            "memory_profile");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  const bool cached_voices = cache_mode != KT2N_VOICE_CACHE_ALL;
  const bool bounded_capacity = inference_only || custom_capacity > 0;
  bool require_assets = inference_only;
#ifndef KT2N_ALLOW_TEST_STUBS
  require_assets = true;
#endif
  if (require_assets && (!cfg->lm_file || !cfg->s3gen_file || !cfg->voices_file ||
                         !cfg->tokenizer_json || !cfg->text_data_dir ||
                         std::string(cfg->s3gen_file) == "l7-standin")) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "inference requires complete model, voice, tokenizer and text assets");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  if (cfg->max_streams < 0 || cfg->max_streams > 64 || cfg->render_pages < 0) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "max_streams must be 0..64 and render_pages must be nonnegative");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  if (cfg->pinned_arena_bytes) {
    set_err(err, KT2N_ERR_UNSUPPORTED, "custom pinned-memory limits are not supported", "pinned_arena_bytes");
    return KT2N_ERR_UNSUPPORTED;
  }
  // Memory profile and free-memory floor (validated above with the voice cache fields).
  const bool mem_fields = mem_block;
  const bool low_mem = mem_block && cfg->memory_profile == KT2N_MEMORY_PROFILE_LOW;
  // Load every CUDA module at context creation instead of at a kernel's first launch (decoder measurement: the first live
  // render of a process otherwise pays lazy module loading inside launch, 6.3 ms vs 1.1 ms). Execution only.
  // Takes effect only if this process has not initialised CUDA yet; an explicit CUDA_MODULE_LOADING wins.
  // KT2N_CUDA_EAGER=0 keeps the driver's lazy default. The low-memory profile loads eagerly too: lazy loading saved
  // 10 MiB per process here (one-arch build; 12,748 -> 12,738 MiB) and moves module loads into the first calls.
  if (const char* v = std::getenv("KT2N_CUDA_EAGER"); !(v && std::atoi(v) == 0)) setenv("CUDA_MODULE_LOADING", "EAGER", 0);
#if defined(KT2N_HIP)
  // ROCm ignores CUDA_MODULE_LOADING; its switch for loading code objects at start is HIP_ENABLE_DEFERRED_LOADING=0.
  if (const char* v = std::getenv("KT2N_CUDA_EAGER"); !(v && std::atoi(v) == 0)) setenv("HIP_ENABLE_DEFERRED_LOADING", "0", 0);
#endif
  size_t free_before = 0, device_total = 0;
  // The free-memory floor applies to every engine that sets min_free_device_bytes; the built-in 10% rule only to
  // engines of bounded capacity (as before).
  const bool floor_check = bounded_capacity || (mem_block && cfg->min_free_device_bytes > 0) || low_mem;
  if (floor_check) {
    cudaError_t status = cudaSetDevice(cfg->device);
    if (status == cudaSuccess) status = cudaFree(nullptr);
    if (status == cudaSuccess) status = cudaMemGetInfo(&free_before, &device_total);
    if (status != cudaSuccess) {
      set_err(err, KT2N_ERR_UNSUPPORTED, std::string("CUDA initialization: ") + cudaGetErrorString(status), "device");
      return KT2N_ERR_UNSUPPORTED;
    }
  }
  // GPU architecture: SM 6.1, 7.5 or 8.0+ with kernels in this build; anything else fails here with a clear message
  // instead of an internal error from the first kernel launch (the CPU stubs use no GPU).
  if (cfg->lm_file || cfg->s3gen_file) {
#ifndef KT2N_CUDA_ARCH_LIST
#define KT2N_CUDA_ARCH_LIST ""  // the build's KT2N_CUDA_ARCHS (cmake/Kt2nCuda.cmake); "" = arch rule only
#endif
    std::string why;
    const int rc = kt2n::cuda::arch_check(cfg->device, KT2N_CUDA_ARCH_LIST, &why);
    if (rc != 0) {
      set_err(err, KT2N_ERR_UNSUPPORTED, rc < 0 ? why : "CUDA initialization: " + why, "device");
      return KT2N_ERR_UNSUPPORTED;
    }
  }
  if (g_auto_probe != nullptr && g_auto_probe->decide_only) {
    uint64_t check_free = free_before;  // the test knob KT2N_LOW_FIT_FREE_MB counts here as in the fit
    if (const char* v = std::getenv("KT2N_LOW_FIT_FREE_MB"); v && std::atoll(v) > 0)
      check_free = std::min<uint64_t>(check_free, uint64_t(std::atoll(v)) << 20);
    if (free_before == 0 || check_free >= kDefaultCheckBelow) {
      g_auto_probe->stopped = true;  // not reached: no check, the default create runs as before
      return KT2N_OK;
    }
  }
  auto e = std::make_unique<kt2n_engine>();
  e->flags = cfg->flags;
  // Conductor poll mode (execution only), on by default after an A/B (first audio lower at N=1/4/8 in 10
  // of 12 interleaved pairs, ~25% less process CPU). KT2N_CONDUCTOR_POLL=0 restores the blocking
  // waits.
  e->poll = true;
  if (const char* v = std::getenv("KT2N_CONDUCTOR_POLL")) e->poll = std::atoi(v) != 0;
  if (const char* v = std::getenv("KT2N_SPLIT_URGENT")) e->split_urgent = std::atoi(v) != 0;
  if (const char* v = std::getenv("KT2N_HOLD_URGENT")) e->hold_urgent = std::max(0, std::min(2, std::atoi(v)));
  if (const char* v = std::getenv("KT2N_BUFFER_URGENT_MS")) e->buffer_urgent_ms = std::max(0.0, std::atof(v));
  if (const char* v = std::getenv("KT2N_BUFFER_ROLE")) e->buffer_to_aux = std::string(v) == "aux";
  if (const char* v = std::getenv("KT2N_TEXT_NICE")) e->text_nice = std::max(0, std::min(19, std::atoi(v)));
  if (cfg->reserved_ptr0 || cfg->reserved_ptr1) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "reserved fields must be NULL", cfg->reserved_ptr0 ? "reserved_ptr0" : "reserved_ptr1");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  e->lv = policy::Levers{};
  e->max_streams = cfg->max_streams > 0 ? cfg->max_streams : (inference_only ? 1 : 10);
  e->row_cap = engine_rows(e->max_streams, cfg->device, cfg->lm_file || cfg->s3gen_file, &e->row_cap_why);
  // request_batch_size = the reference decode rows (= its seats). Here the decode rows are row_cap: the same as
  // max_streams unless the seats exceed the GPU's row fit (capacity mode).
  e->lv.request_batch_size = e->row_cap;
  if (const char* v = std::getenv("KT2N_CAPACITY_ORDER")) e->capacity_order = std::atoi(v) != 0;
  if (e->row_cap < e->max_streams)
    std::fprintf(stderr, "[kt2n engine] seats %d, rows %d (%s): admitted requests beyond %d rows wait in admission\n",
                 e->max_streams, e->row_cap, e->row_cap_why.c_str(), e->row_cap);
  std::string msg;
  // Text.
  // Text threads = grammar copies. Default 4 = the reference's text worker count (measured: with 2 copies the split
  // budget is 2 and 1 of 4,054 segments joins its pieces differently from the reference; 3-4 match all). engine.h's
  // comment still says 2 (frozen header text).
  // The normaliser's grammars (0.5-0.9 s of CPU on a laptop) load in a background thread while the models are
  // uploaded and the voices prepared on the GPU; wait_ready() below, before the workers start, joins it and reports
  // a grammar error with the same code and message (after the GPU loads instead of before them). The tokenizer the
  // sizing below needs loads here. In the automatic profile's probe the load starts once the probe carries on.
  if (g_create_handoff && g_create_handoff->text) {  // read by the memory check just before (default_create_checked)
    e->text = std::move(g_create_handoff->text);
    e->text->start_loading();  // the grammars now load in the background, as with TextFront::Load::kBackground
  } else {
    e->text = TextFront::create(cfg->text_data_dir, cfg->tokenizer_json, text_threads(cfg), &msg,
                                g_auto_probe ? TextFront::Load::kDeferred : TextFront::Load::kBackground);
  }
  if (!e->text) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "text front end: " + msg, "text_data_dir");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  debug_mem("text front end (tokenizer; grammars loading)");
  // Voices file first: the LM's voice store is sized from the prefixes it must hold.
  std::optional<kt2n::voice::VoicesBlob> vblob;
  size_t prefix_tokens = 0;
  if (cfg->voices_file && g_create_handoff && g_create_handoff->vblob) {
    vblob = std::move(g_create_handoff->vblob);
    for (const auto& rec : vblob->voices) prefix_tokens += rec.lm_prefix.size();
  } else if (cfg->voices_file) {
    try {
      std::ifstream f(cfg->voices_file, std::ios::binary);
      std::vector<uint8_t> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      if (blob.empty()) throw std::runtime_error("cannot read voices_file");
      vblob = kt2n::voice::parse_voices_blob(blob.data(), blob.size());
      for (const auto& rec : vblob->voices) prefix_tokens += rec.lm_prefix.size();
    } catch (const std::exception& ex) {
      set_err(err, KT2N_ERR_INVALID_ARGUMENT, std::string("voices: ") + ex.what(), "voices_file");
      return KT2N_ERR_INVALID_ARGUMENT;
    }
  }
  // Voice cache budget (LRU / NONE). Each voice costs its LM prefix range, its renderer prefix pages and a voice table
  // row (voice_cost). The cache enforces the count and byte budgets plus the two derived store sizes below, which are
  // the most any resident set within the budgets can use, so the LM store and the render pool are sized from the
  // budget instead of the catalogue.
  kt2n::engine::VoiceCacheLimits cache_lim;
  int cache_table_rows = 0, cache_row_pages = 0, cache_max_voice_pages = 0;
  if (cached_voices) {
    std::vector<kt2n::engine::VoiceCost> costs;
    int max_tp = 0, hidden_voices = 0;
    if (vblob) {
      for (const auto& rec : vblob->voices) {
        costs.push_back(kt2n::engine::voice_cost(int(rec.prompt_token.size()), rec.lm_prefix.size()));
        max_tp = std::max(max_tp, int(rec.prompt_token.size()));
        if (const auto* h = rec.meta.get("hidden"); h && h->is_bool() && h->as_bool()) ++hidden_voices;
      }
    } else {
      costs.push_back(kt2n::engine::voice_cost(200, 10));  // the stub catalogue (tests)
      max_tp = 200;
    }
    const kt2n::engine::VoiceCost largest_cat = *std::max_element(
        costs.begin(), costs.end(), [](const auto& a, const auto& b) { return a.total() < b.total(); });
    // The voice with the most bytes is not always the one with the most pages or LM bytes: keep each maximum.
    int max_cat_pages = 0;
    uint64_t max_cat_lm = 0;
    for (const auto& c : costs) {
      max_cat_pages = std::max(max_cat_pages, c.pages);
      max_cat_lm = std::max(max_cat_lm, c.lm_bytes);
    }
    // Clones go through the same cache: bound them at 1,024 prefix ids (twice the reference allowance of 512 per
    // slot: a 30 s reference with its transcript is ~670) and 250 prompt tokens (10 s); a larger one is refused at
    // registration with an explicit error.
    if (!inference_only) {
      costs.push_back(kt2n::engine::voice_cost(250, size_t(custom_prefix_ids)));
      max_tp = std::max(max_tp, 250);
    }
    uint64_t max_lm = 0, min_total = ~uint64_t{0};
    int max_pages = 0;
    double r_lm = 0, r_pg = 0;
    for (const auto& c : costs) {
      max_lm = std::max(max_lm, c.lm_bytes);
      max_pages = std::max(max_pages, c.pages);
      min_total = std::min(min_total, c.total());
      r_lm = std::max(r_lm, double(c.lm_bytes) / double(c.total()));
      r_pg = std::max(r_pg, double(c.page_bytes) / double(c.total()));
    }
    int n = cache_voices;
    // No limits given: LRU keeps 16 voices, NONE one voice per row; both at least one per row, since every request
    // waiting for a row or running holds its voice.
    if (n == 0 && cache_bytes == 0) n = std::max(cache_mode == KT2N_VOICE_CACHE_LRU ? 16 : 1, e->row_cap);
    if (cache_bytes && cache_bytes < largest_cat.total()) {
      set_err(err, KT2N_ERR_TOO_SMALL,
              "voice_cache_max_bytes (" + std::to_string(cache_bytes >> 20) + " MiB) cannot hold the largest voice (" +
                  std::to_string((largest_cat.total() + (1 << 20) - 1) >> 20) + " MiB)",
              "voice_cache_max_bytes");
      return KT2N_ERR_TOO_SMALL;
    }
    if (n == 0) {  // a byte budget only: as many voices as the smallest one allows, at most the catalogue + clones
      const uint64_t by_bytes = cache_bytes / min_total;
      n = int(std::min<uint64_t>(by_bytes, uint64_t(costs.size()) + (inference_only ? 0 : 72)));
      n = std::max(1, n);
    }
    cache_lim.max_voices = n;
    cache_lim.max_bytes = cache_bytes;
    cache_lim.max_lm_bytes = uint64_t(n) * max_lm;
    cache_lim.max_pages = n * max_pages;
    if (cache_bytes) {
      cache_lim.max_lm_bytes = std::min<uint64_t>(cache_lim.max_lm_bytes, uint64_t(double(cache_bytes) * r_lm) + 1);
      cache_lim.max_pages = std::min<int>(cache_lim.max_pages, int(double(cache_bytes) * r_pg / double(kt2n::render::kPageBytes)));
    }
    cache_lim.max_lm_bytes = std::max(cache_lim.max_lm_bytes, max_cat_lm);
    cache_lim.max_pages = std::max(cache_lim.max_pages, max_cat_pages);
    cache_max_voice_pages = max_cat_pages;
    cache_table_rows = n + hidden_voices + 1;  // + the hidden voices (not counted) + one spare
    // Rows' pages: the same 40 pages per row as the catalogue pool, and at least one row at its largest budget
    // (chars x 4.4 capped at 2,341 tokens, from position 0), so one row always fits beside a full cache.
    cache_row_pages = std::max(40 * e->row_cap, (2 * (max_tp + 2341 + 3) + 63) / 64);
  }
  // Low-memory fit (engine.h KT2N_MEMORY_PROFILE_LOW): decided with the LM's sizes below, applied to the LM context and
  // to the decoder (pool, complete-file workspace, vocoder frame limit, CUDA graphs).
  struct LowFit {
    bool fitted = false;
    int ctx = 4096, pool = 0, ws_mb = 1024, frames = 4096;
    int rows = 0;           // > 0: LM max_rows (K/V slots = rows + 1)
    int stream_budget = 0;  // > 0: tier 0, the largest streaming segment budget the context and pool hold
    bool give_back = false;  // complete-file rows give their voice's prefix pages back (L7RenderOptions)
    uint64_t floor = 0;
  } fit;
  const bool fit_real_decoder = low_mem && cfg->s3gen_file && std::string(cfg->s3gen_file) != "l7-standin";
  if (g_auto_probe != nullptr && !(fit_real_decoder && cfg->lm_file)) {  // no fit to run (stand-in parts): try default
    g_auto_probe->stopped = true;
    return KT2N_OK;
  }
  // LM.
  if (cfg->lm_file) {
    kt2n::lm::Config lc;
    lc.engine_path = cfg->lm_file;
    lc.device = cfg->device;
    // +2 slots: LM releases a finished row's slot only after its last ring entry is read (lm.h), so a new
    // row admitted in the same loop may need one more slot for a moment.
    lc.max_rows = std::min(64, std::max(e->row_cap + 2, cfg->max_lm_rows > 0 ? cfg->max_lm_rows : 0));
    lc.graphs = !(cfg->flags & KT2N_CFG_NO_GRAPHS);
    // 112 KB of bf16 K/V per prefix token ([2][28][8][128]), plus 1/8 slack for alignment.
    // Headroom for loaded saved / inline voices: custom_voice_headroom() prefixes of 512 ids (72 by default, ~4.8 GB).
    // Lab knobs for fitting a 12 GB GPU (defaults unchanged): KT2N_LM_VOICE_HEADROOM = loaded-voice prefixes
    // of headroom (32); KT2N_LM_MAX_CTX = K/V columns per LM slot (4096; capacity only, the ceiling rule uses
    // ctx_limit, which stays 4096).
    size_t headroom = inference_only ? 0 : size_t(custom_capacity > 0 ? custom_capacity :
        custom_voice_headroom(cfg->inline_voice_lru > 0 ? cfg->inline_voice_lru : 32));
    if (custom_capacity == 0)
      if (const char* v = std::getenv("KT2N_LM_VOICE_HEADROOM")) headroom = size_t(std::max(0, std::atoi(v)));
    // Memory controls (results unchanged by any of them; low_mem and cached_voices come from the appended config
    // block, read only when struct_size covers it).
    if (vblob && headroom == 0 && low_mem && !cached_voices) {
      // Exact store: only the catalogue prefixes, built once at create into consecutive ranges (114,688 B per token,
      // already 256-byte aligned), so they need no slack and no 256 MiB floor.  With custom-voice headroom the
      // store keeps the default size: a clone prefix can exceed the 512-token allowance (a 30 s reference is
      // ~850 ids) and today fits in the slack.
      lc.voice_store_bytes = prefix_tokens * 114688;
    } else {
      lc.voice_store_bytes = std::max<size_t>(size_t{256} << 20, (prefix_tokens + headroom * 512) * 114688 * 9 / 8);
    }
    // Voice cache: the store holds the cache's LM budget (+1/8 for alignment and fragmentation), not the catalogue.
    if (cached_voices) lc.voice_store_bytes = size_t(cache_lim.max_lm_bytes + cache_lim.max_lm_bytes / 8 + (size_t{1} << 20));
    if (low_mem) {
      // Only the audio slice of the embedding on the GPU; prefill gathers its rows on the host (-592 MiB).
      lc.host_embedding = true;
      // Prefill workspace (36 KiB per token) for the longest prefill this engine runs, instead of 4,096 tokens.
      // Without clone builds: the longest catalogue prefix, and the longest suffix target_suffix() can make from an
      // LM chunk: its fixed ids (emotion control + text start + final segment + speech start) plus at most one id per
      // UTF-8 byte of the chunk (byte-level BPE never yields more ids than bytes), at most 4 bytes per character, and
      // a chunk is at most demo_chunk_characters + 1 characters (resplit_segment + terminal punctuation).  Clone
      // builds make prefixes of up to max_ctx - 1 ids, so they keep 4,096.  A longer prompt still works: it runs
      // as consecutive prefill calls (same results).
      if (inference_only) {
        size_t longest = 1;
        if (vblob)
          for (const auto& rec : vblob->voices) longest = std::max(longest, rec.lm_prefix.size());
        const size_t fixed_ids = e->text->suffix(std::u32string(), true).size();
        const size_t chunk_chars = size_t(kt2n::textplan::Settings{}.demo_chunk_characters) + 1;
        longest = std::max(longest, fixed_ids + 4 * chunk_chars);
        lc.max_prefill_tokens = int(std::min<size_t>(4096, (longest + 63) / 64 * 64));
      }
    }
    if (fit_real_decoder && free_before > 0) {
      // Sizes in bytes. Measured on the RTX A2000 (SM86) and RTX PRO 6000 (SM120), low profile, one stream:
      //   LM: ~520 MiB + voice store + (max_rows + 1) x 114,688 B per context position (K/V slots + prefill);
      //   decoder: ~840 MiB of weights and tables; the vocoder's compact handle ~235.6 KB per frame-row + ~60 MiB;
      //   S_urgent's workspace (sized by the renderer, ~112 MiB); S_bulk's workspace; the page pool (kPageBytes each);
      //   CUDA graphs captured by the standard prewarm: up to ~2 GiB on a 6 GB SM86 card (6-8 MiB per graph).
      const uint64_t MiB = uint64_t(1) << 20;
      fit.floor = mem_fields && cfg->min_free_device_bytes > 0 ? cfg->min_free_device_bytes
                                                               : std::max<uint64_t>(128 * MiB, device_total / 64);
#if defined(KT2N_HIP) && !defined(_WIN32)
      // AMD under WSL2 (/dev/dxg): driving free memory below about 0.5 GB hangs the allocating process on some PCs
      // (RX 7900 XT, RX 7900 XTX: the next 128 MiB block of the display-driver probe never returned) and fails the
      // allocation on others (RX 7800 XT, RX 9060 XT: at about 0.4 GB free). Keep at least 1 GiB free there, whatever
      // min_free_device_bytes asks for: the fit sizes below it and the probe stops above it.
      if (access("/dev/dxg", F_OK) == 0) fit.floor = std::max<uint64_t>(fit.floor, uint64_t(1) << 30);
#endif
      // Test knob (labelled "simulated" wherever used): fit as if only this much were free (KT2N_LOW_FIT_FREE_MB).
      uint64_t fit_free = free_before;
      if (const char* v = std::getenv("KT2N_LOW_FIT_FREE_MB"); v && std::atoll(v) > 0)
        fit_free = std::min<uint64_t>(fit_free, uint64_t(std::atoll(v)) * MiB);
      uint64_t avail = fit_free > fit.floor ? fit_free - fit.floor : 0;
      uint64_t wddm_extra = 0;  // fitted on WDDM: what writing the memory costs beyond the allocations
      const kt2n::textplan::Settings ts{};
      // The longest segment at its budget: a 380-character chunk grows in normalisation (the corpora reach 460
      // characters), and a streaming segment's budget is 4.4 tokens per character up to 2,341 (532 characters);
      // a complete-file segment's DEMO budget is at most 1,000. The pool always holds one such row.
      const int seg_chars = int(ts.demo_chunk_characters * 1.4);
      const int seg_budget = std::max(kt2n::textplan::tagged_segment_budget(size_t(1) << 20), ts.demo_max_new_tokens);
      int max_tp = inference_only ? 0 : 250, max_pref = inference_only ? 0 : custom_prefix_ids, max_pp = 1;
      if (vblob)
        for (const auto& rec : vblob->voices) {
          max_tp = std::max(max_tp, int(rec.prompt_token.size()));
          max_pref = std::max(max_pref, int(rec.lm_prefix.size()));
          max_pp = std::max(max_pp, (2 * int(rec.prompt_token.size()) + 63) / 64 + 1);
        }
      const int prompt_cap = std::max(250, max_tp);
      auto row_pages = [&](int budget) {
        return (kt2n::render::kMelPerToken * (prompt_cap + budget + kt2n::render::kLookahead) + kt2n::render::kPageFrames - 1) /
               kt2n::render::kPageFrames;
      };
      // Fitted sizes (fit.fitted): a row's pages exactly as the renderer reserves them (render/state.cc
      // row_pages_needed): a streaming row shares its voice prefix's complete pages, so it needs the span of
      // prompt + budget + look-ahead minus those shared pages; a complete-file row starts at position 0 and shares
      // none. Taken over every prompt length a row can have: the catalogue's, and 1..250 for loaded / built voices.
      std::vector<int> row_tps;
      if (vblob)
        for (const auto& rec : vblob->voices) row_tps.push_back(int(rec.prompt_token.size()));
      if (!inference_only || row_tps.empty())
        for (int tp = 1; tp <= 250; ++tp) row_tps.push_back(tp);
      std::sort(row_tps.begin(), row_tps.end());
      row_tps.erase(std::unique(row_tps.begin(), row_tps.end()), row_tps.end());
      auto span_pages = [](int tokens) {
        return (kt2n::render::kMelPerToken * tokens + kt2n::render::kPageFrames - 1) / kt2n::render::kPageFrames;
      };
      auto shared_pages = [](int tp) {  // render/service.cu build_voice_prefix: complete pages of 2 x stable frames
        // render/state.h stable_prefix_tokens: whole 25-token blocks whose look-ahead stays in the prompt
        const int stable = tp < kt2n::render::kLookahead
                               ? 0
                               : (tp - kt2n::render::kLookahead) / kt2n::render::kTokenChunk * kt2n::render::kTokenChunk;
        return stable < kt2n::render::kTokenChunk ? 0 : kt2n::render::kMelPerToken * stable / kt2n::render::kPageFrames;
      };
      auto stream_row_pages = [&](int budget) {
        int m = 0;
        for (int tp : row_tps) m = std::max(m, span_pages(tp + budget + kt2n::render::kLookahead) - shared_pages(tp));
        return m;
      };
      int demo_row_pages = 0;  // a complete-file row at its DEMO budget, from position 0
      for (int tp : row_tps)
        demo_row_pages = std::max(demo_row_pages, span_pages(tp + ts.demo_max_new_tokens + kt2n::render::kLookahead));
      auto fitted_row_pages = [&](int budget) { return std::max(stream_row_pages(budget), demo_row_pages); };
      // One cached voice (NONE, or LRU of one voice) at one stream: a complete-file row gives its voice's prefix pages
      // back when it needs them (L7RenderOptions::demo_gives_back_prefix), so the pool holds the larger of the prefix
      // plus a streaming row and a complete-file row, not their sum.
      const bool give_back = cached_voices && cache_lim.max_voices == 1 && e->row_cap == 1;
      const int big_budget = seg_budget;
      int pool_std, pool_min;
      if (cached_voices) {
        pool_std = cache_lim.max_pages + cache_row_pages;
        pool_min = cache_lim.max_pages + std::max(40 * e->row_cap, row_pages(seg_budget));
      } else {
        pool_std = max_pp + row_pages(big_budget) * e->row_cap;
        pool_min = max_pp + std::max(40, row_pages(seg_budget)) * e->row_cap;
      }
      // Context: the longest voice prefix + the text of the longest segment (fixed ids + one id per character; the
      // corpora's longest prompt, prefix included, is 380 ids) + the segment budget. A longer prompt is refused at
      // admission (KT2N_ERR_GPU_MEMORY_LIMIT).
      const int fixed_ids = int(e->text->suffix(std::u32string(), true).size());
      auto ctx_for = [&](int budget) {
        return std::min(4096, (max_pref + fixed_ids + seg_chars + budget + 1 + 63) / 64 * 64);
      };
      // Fitted pool: the voice prefixes' pages plus each row's exact need (fitted_row_pages).
      auto fitted_pool_for = [&](int budget) {
        if (give_back) return std::max(cache_lim.max_pages + stream_row_pages(budget), demo_row_pages);
        return cached_voices ? cache_lim.max_pages + std::max(40 * e->row_cap, fitted_row_pages(budget) * e->row_cap)
                             : max_pp + fitted_row_pages(budget) * e->row_cap;
      };
      int ctx_min = ctx_for(seg_budget);
      // K/V slots: the standard low sizes keep max_rows + 1; a fitted engine keeps one slot per row + the scratch
      // slot (fit_rows below), so per context position it costs (fit_rows + 1) x 114,688 B; with one row, no scratch
      // slot K/V (lm.h Config::scratch_kv), so 114,688 B.
      const int fit_rows = std::min(64, std::max(e->row_cap, cfg->max_lm_rows > 0 ? cfg->max_lm_rows : 0));
      uint64_t per_ctx = uint64_t(lc.max_rows + 1) * 114688;
      const int frames_min = (kt2n::render::kMelPerToken * (ts.demo_max_new_tokens + 6) + 63) / 64 * 64;
      // Encoder position tables (fitted: render::RendererConfig::pos_tables_from_pool): 6 token-rate and 4 frame-rate
      // tables of (2 maxd - 1) x 512 floats, maxd = the longest row the pool holds (32 tokens per page).
      auto table_bytes = [](int maxd_tok) {
        return (uint64_t(kt2n::render::kEnc1Layers) * uint64_t(2 * maxd_tok - 1) +
                uint64_t(kt2n::render::kEnc2Layers) * uint64_t(4 * maxd_tok - 1)) * uint64_t(kt2n::render::kEncDim) * 4;
      };
      // One stream (fitted): both flow workspaces and the vocoder's run scratch are one buffer of the largest of the
      // three (render::RendererConfig::alias_workspaces): the vocoder's scratch is ~235.6 KB per frame-row plus its
      // partial sums (32 x 300,000 floats); S_urgent's workspace ~128 MiB; S_bulk's ws_mb.
      const bool alias_ws = e->max_streams == 1;
      // Fitted: prefill calls of at most this many new tokens (the prefill workspace, ~37 KB per token). A longer voice
      // prefix or prompt runs as consecutive calls, which give the same bits as one call (prefill.h, lm_engine.cu).
      constexpr int kFitPrefill = 512;
      uint64_t prefill_saved_bytes = 0;
      {
        const size_t full = kt2n_pf_ctx_workspace_bytes2(16, 1536, 4096, 0u, cfg->device);
        const size_t fit_pf = kt2n_pf_ctx_workspace_bytes2(16, kFitPrefill, 4096, KT2N_PF_CTX_FIT_ATTN_WS, cfg->device);
        if (full > fit_pf && fit_pf > 0) prefill_saved_bytes = full - fit_pf;
        if (fit_rows != 1) prefill_saved_bytes = uint64_t(1536 - kFitPrefill) * 37152;  // no fitted attention workspace
      }
      bool fitted_sizes = false;  // need() with the fitted sizes (set with fit.fitted below)
      auto need = [&](int ctx, int pool, int ws_mb, int frames, bool graphs) {
        const uint64_t tables_saved =
            fitted_sizes ? table_bytes(kt2n::render::kMaxRowTokens) -
                               table_bytes(std::min(kt2n::render::kMaxRowTokens, kt2n::render::kPageTokens * pool))
                         : 0;
        // The shared buffer holds the vocoder's scratch with its dead source-stage buffers folded into later ones
        // (hift.cu compact_scratch_floats): 196,932 B per frame-row instead of 235,588.
        const uint64_t part = fitted_sizes && alias_ws ? uint64_t(4) << 22 : uint64_t(32) * 300000 * 4;  // 4 Mi floats
        const uint64_t vocoder = uint64_t(frames) * (fitted_sizes && alias_ws ? 196932 : 235588) + part;
        const uint64_t scratch = fitted_sizes && alias_ws
                                     ? std::max({vocoder, 128 * MiB, uint64_t(ws_mb) * MiB})
                                     : vocoder + 128 * MiB + uint64_t(ws_mb) * MiB;
        // Fitted: prefill calls of at most kFitPrefill tokens and its split-attention workspace for this GPU's SMs
        // (below) instead of the 1,536-token, 17-tile context the 520 MiB was measured with.
        const uint64_t prefill_saved = fitted_sizes ? prefill_saved_bytes : 0;
        // Fitted: the decoder's fixed allocations measured on a GTX 1080 Ti and an RTX 4090 (low profile, one stream;
        // allocation log): encoder 157 + estimator 126 + vocoder weights and tensor-core copies 171 + state and staging
        // ~45 = ~500 MiB plus the position tables, and 60-70 MiB of driver overhead on all allocations; the 840 MiB
        // the standard sizes were planned with left the whole estimate 80-100 MiB above what create takes. 780 keeps
        // 20-40 MiB of that margin.
        const uint64_t decoder_fixed = fitted_sizes ? 780 * MiB : 840 * MiB;
        return 520 * MiB - prefill_saved + uint64_t(lc.voice_store_bytes) + per_ctx * uint64_t(ctx) + decoder_fixed - tables_saved +
               scratch + (60 * MiB - uint64_t(32) * 300000 * 4) + uint64_t(pool) * kt2n::render::kPageBytes +
               (graphs ? 2048 * MiB : 0);
      };
      fit.pool = pool_std;
      if (g_auto_probe != nullptr) {
        g_auto_probe->std_need_bytes = need(4096, pool_std, 1024, 4096, true);
        g_auto_probe->fit_free_bytes = fit_free;
        if (g_auto_probe->decide_only && !cached_voices && vblob) {
          // Voice cache ALL: the default profile's pool holds every catalogue voice's prefix pages plus 40 pages per
          // row (render_pages below), the low profile's one prefix plus a full row each (pool_std).
          uint64_t default_pages = 40 * uint64_t(e->row_cap);
          for (const auto& rec : vblob->voices) default_pages += (2 * uint64_t(rec.prompt_token.size()) + 63) / 64 + 1;
          if (default_pages > uint64_t(pool_std))
            g_auto_probe->default_pool_extra_bytes = (default_pages - uint64_t(pool_std)) * kt2n::render::kPageBytes;
        }
      }
      if (need(4096, pool_std, 1024, 4096, true) > avail) {
        // Smaller than the standard low sizes: no CUDA graphs in the decoder, S_bulk's workspace sized by the renderer
        // to its calls' exact need (renderer.h bulk_ws_row_tokens: 247 MiB measured on SM86; 256 planned), one LM
        // K/V slot per row (a new row whose slot is not released yet waits: LM back-pressure, admit()), then shrink
        // in order.
        fit.fitted = true;
        fitted_sizes = true;
#if defined(KT2N_HIP)
        // Test knob (with KT2N_LOW_FIT_FREE_MB): the display driver's extra in MiB instead of the probe, so a
        // simulated small GPU gets the same sizes on every PC (the probe's result varies by tens of MiB under WSL2).
        if (const char* v = std::getenv("KT2N_FIT_DRIVER_EXTRA_MB"); v && *v)
          wddm_extra = uint64_t(std::atoll(v)) << 20;
        else
#endif
        wddm_extra = wddm_written_extra(fit.floor, cfg->device);
        avail = avail > wddm_extra ? avail - wddm_extra : 0;
        fit.give_back = give_back;
        fit.ws_mb = 256;
        fit.rows = fit_rows;
        per_ctx = uint64_t(fit_rows == 1 ? 1 : fit_rows + 1) * 114688;
        pool_min = fitted_pool_for(seg_budget);  // the standard pool (fit.pool) stays while it fits: spare pages for rows past
                                                 // their budget (the noise extension), as before
        if (need(ctx_min, pool_min, fit.ws_mb, frames_min, false) > avail) {
          // Tier 0 (smaller cards): complete-file segments at their DEMO budget (1,000) always fit; streaming
          // segments up to the largest budget whose context and pool still fit (fit.stream_budget). A segment beyond
          // it is refused at admission with KT2N_ERR_GPU_MEMORY_LIMIT (context or pages), never cut short.
          int b = seg_budget;
          while (b > ts.demo_max_new_tokens && need(ctx_for(b), fitted_pool_for(b), fit.ws_mb, frames_min, false) > avail)
            b = std::max(ts.demo_max_new_tokens, b - 8);
          fit.stream_budget = b;
          ctx_min = ctx_for(b);
          pool_min = fitted_pool_for(b);
          fit.pool = pool_min;
          fit.frames = frames_min;
        }
        if (need(fit.ctx, fit.pool, fit.ws_mb, fit.frames, false) > avail) fit.pool = std::min(fit.pool, pool_min);
        if (need(4096, fit.pool, fit.ws_mb, fit.frames, false) > avail && fit.frames > frames_min &&
            need(ctx_min, fit.pool, fit.ws_mb, fit.frames, false) > avail)
          fit.frames = frames_min;
        auto fit_ctx = [&] {
          const uint64_t n0 = need(0, fit.pool, fit.ws_mb, fit.frames, false);
          const int room = n0 >= avail ? 0 : int(std::min<uint64_t>(4096, (avail - n0) / per_ctx));
          fit.ctx = std::max(ctx_min, std::min(4096, room / 64 * 64));
        };
        fit_ctx();
        if (need(fit.ctx, fit.pool, fit.ws_mb, fit.frames, false) > avail) {
          fit.frames = frames_min;  // a complete-file segment at its DEMO budget in one vocoder call
          fit_ctx();
        }
        const uint64_t n_min = need(ctx_min, pool_min, fit.ws_mb, frames_min, false);
        if (need(fit.ctx, fit.pool, fit.ws_mb, fit.frames, false) > avail) {
          set_err(err, KT2N_ERR_TOO_SMALL,
                  "the low-memory profile needs about " + std::to_string((n_min + fit.floor) >> 20) +
                      " MiB of free GPU memory for one stream; this GPU has " + std::to_string(fit_free >> 20) +
                      " MiB free after the CUDA context (min_free_device_bytes " + std::to_string(fit.floor >> 20) +
                      " MiB" +
                      (wddm_extra > 0 ? "; the display driver takes " + std::to_string(wddm_extra >> 20) +
                                            " MiB more once memory is written"
                                      : std::string()) +
                      ")",
                  "memory_profile");
          return KT2N_ERR_TOO_SMALL;
        }
      }
      if (fit.fitted) lc.max_ctx = fit.ctx;
      if (fit.fitted) lc.max_prefill_tokens = std::min(lc.max_prefill_tokens, kFitPrefill);
      if (fit.fitted && fit.rows == 1 && fit_on("KT2N_FIT_PREFILL_WS")) lc.prefill_fit_attn_ws = true;  // one-row calls (lm.h)
      if (fit.fitted && fit.rows > 0) lc.max_rows = fit.rows;
      if (fit.fitted && lc.max_rows == 1 && fit_on("KT2N_FIT_SCRATCH_KV")) lc.scratch_kv = false;  // no padding row (lm.h)
      // Fitted (small GPUs, often on hosts with little RAM): the host embedding table becomes resident row by row as
      // prefill reads it, instead of all ~600 MiB at load (lm.h host_embedding_resident; same results).
      if (fit.fitted && fit_on("KT2N_FIT_LAZY_EMBED")) lc.host_embedding_resident = false;
      if (const char* v = std::getenv("KT2N_DEBUG_MEM"); v && *v == '1')
        std::fprintf(stderr,
                     "[mem] low-memory fit: free %zu MiB floor %zu MiB -> %s ctx %d (min %d) pool %d (std %d, min %d) "
                     "bulk workspace %d MiB vocoder frames %d (min %d) graphs %d lm rows %d streaming budget %d; "
                     "estimate %zu MiB; display driver extra %zu MiB\n",
                     size_t(fit_free >> 20), size_t(fit.floor >> 20), fit.fitted ? "fitted" : "standard", fit.ctx,
                     ctx_min, fit.pool, pool_std, pool_min, fit.ws_mb, fit.frames, frames_min, fit.fitted ? 0 : 1,
                     lc.max_rows, fit.stream_budget > 0 ? fit.stream_budget : seg_budget,
                     size_t(need(fit.ctx, fit.pool, fit.ws_mb, fit.frames, !fit.fitted) >> 20), size_t(wddm_extra >> 20));
    }
    if (g_auto_probe != nullptr) {
      g_auto_probe->reached = true;
      g_auto_probe->fitted = fit.fitted;
      g_auto_probe->free_bytes = free_before;
      g_auto_probe->floor_bytes = fit.floor;
      // The standard low sizes fit: stop here so that the caller tries the default profile first -- unless even the
      // default profile's smallest possible need (the standard estimate + kAutoDefaultMargin + the free-memory floor
      // a default create would keep: min_free_device_bytes, or 10% for an engine of bounded capacity) is more than
      // is free, then carry on as the standard low profile. "Free" is what the fit used: the measured free memory,
      // or less with the test knob KT2N_LOW_FIT_FREE_MB (so a test on a large GPU can take this branch).
      const uint64_t default_floor = mem_fields && cfg->min_free_device_bytes > 0 ? cfg->min_free_device_bytes
                                     : bounded_capacity                           ? device_total / 10
                                                                                  : 0;
      g_auto_probe->default_floor_bytes = default_floor;
      if (g_auto_probe->decide_only) {  // DEFAULT's check: nothing allocated yet; the caller decides
        g_auto_probe->stopped = true;
        g_auto_probe->text = std::move(e->text);
        g_auto_probe->vblob = std::move(vblob);
        return KT2N_OK;
      }
      const bool default_may_fit =
          g_auto_probe->std_need_bytes == 0 ||
          g_auto_probe->fit_free_bytes >= g_auto_probe->std_need_bytes + kAutoDefaultMargin + default_floor;
      if (!fit.fitted && default_may_fit) {
        g_auto_probe->stopped = true;
        return KT2N_OK;
      }
      e->text->start_loading();  // carrying on as the low profile (fitted, or standard sizes)
    }
    // Lab knobs (A/B and tests): KT2N_LM_HOST_EMBED=0|1 overrides the embedding placement, KT2N_LM_PREFILL_TOKENS
    // the prefill call size (64..4096).
    if (const char* v = std::getenv("KT2N_LM_HOST_EMBED"); v && *v) lc.host_embedding = std::atoi(v) != 0;
    if (const char* v = std::getenv("KT2N_LM_PREFILL_TOKENS"); v && *v)
      lc.max_prefill_tokens = std::max(64, std::min(4096, std::atoi(v)));
    if (const char* v = std::getenv("KT2N_LM_MAX_CTX")) lc.max_ctx = std::max(1024, std::atoi(v) / 64 * 64);
    // A/B knob (execution only): decode steps of split_min_rows..16 rows run as two half-batch graph
    // branches (LM's default 14; 0 = off, one step). Unset = LM's default.
    if (const char* v = std::getenv("KT2N_LM_SPLIT_MIN_ROWS"); v && *v) lc.split_min_rows = std::max(0, std::atoi(v));
    // ... and the largest split bucket (measured: 78% of N=20 steps have 17-20 rows; 20 = bucket 20 as 10 + 10).
    if (const char* v = std::getenv("KT2N_LM_SPLIT_MAX_ROWS"); v && *v) lc.split_max_rows = std::max(0, std::atoi(v));
    // ... and the decode stream's CUDA priority level above the lowest (execution only; 0 = LM's default).
    if (const char* v = std::getenv("KT2N_LM_DECODE_PRIORITY"); v && *v) lc.decode_priority_level = std::max(0, std::atoi(v));
    // Graph replay takes the launch stream's priority. Eager split branches keep their original stream priorities,
    // so this experiment is restricted to the existing graph path and a positive restore priority.
    if (const char* v = std::getenv("KT2N_LM_RENDER_YIELD"); v && std::atoi(v) != 0) {
      if (!lc.graphs || lc.decode_priority_level <= 0) {
        set_err(err, KT2N_ERR_INVALID_ARGUMENT,
                "the render-yield option requires graphs and a positive LM decode priority");
        return KT2N_ERR_INVALID_ARGUMENT;
      }
      e->lm_render_yield = true;
      e->lm_render_yield_base_priority = lc.decode_priority_level;
    }
    if (const char* v = std::getenv("KT2N_DEBUG_MEM"); v && *v == '1')
      std::fprintf(stderr, "[mem] lm sizing: host_embedding %d max_prefill_tokens %d voice_store %zu MiB max_rows %d\n",
                   lc.host_embedding ? 1 : 0, lc.max_prefill_tokens, lc.voice_store_bytes >> 20, lc.max_rows);
    e->lm = make_l3_lm(lc, &msg);
    if (!e->lm) {
      set_err(err, KT2N_ERR_INVALID_ARGUMENT, "LM: " + msg, "lm_file");
      return KT2N_ERR_INVALID_ARGUMENT;
    }
    // Low profile with a context below the default's 4,096 (fitted, or the KT2N_LM_MAX_CTX lab knob): admission checks
    // prompt + budget, and a row stopped by the full slot fails (engine.h KT2N_ERR_GPU_MEMORY_LIMIT).
    if (low_mem && lc.max_ctx < 4096) e->fit_ctx = lc.max_ctx;
    e->fit.lm_context_tokens = lc.max_ctx;
    if (e->lm_render_yield) {
      // Create the alternate stream before any request exists, so a live relief episode allocates no CUDA stream.
      std::vector<kt2n::lm::TokenEvent> warm;
      for (int level : {0, e->lm_render_yield_base_priority}) {
        e->lm->set_decode_priority(level);
        if (e->lm->pump(&warm, false, &msg) < 0) {
          set_err(err, KT2N_ERR_INVALID_ARGUMENT, "LM render-yield prewarm: " + msg, "lm_file");
          return KT2N_ERR_INVALID_ARGUMENT;
        }
      }
    }
  } else {
    StubLmConfig lm_cfg;  // stub timing knobs (lab simulation only; the real LM ignores them)
    if (const char* v = std::getenv("KT2N_STUB_LM_STEP_US")) lm_cfg.step_us = std::atoi(v);
    // Tests (voice cache): simulated asynchronous voice build time, and failures of the first n builds.
    if (const char* v = std::getenv("KT2N_STUB_LM_VOICE_BUILD_US")) lm_cfg.voice_build_us = std::atoi(v);
    if (const char* v = std::getenv("KT2N_TEST_STUB_LM_FAIL_AFTER_STEPS")) lm_cfg.fail_after_steps = std::atoi(v);
    if (const char* v = std::getenv("KT2N_TEST_STUB_VOICE_FAIL_FIRST")) lm_cfg.voice_build_fail_first = std::atoi(v);
    lm_cfg.voice_required = cached_voices;
    e->lm = make_stub_lm(lm_cfg);
  }
  debug_mem("lm create");
  // Render. NULL = the CPU stub. "l7-standin" (lab) = the decoder's RenderService with its stand-in stages: the real
  // render state and call path, placeholder audio. A path = the real renderer (decoder Renderer over s3gen.kt2n).
  if (cfg->s3gen_file && std::string(cfg->s3gen_file) == "l7-standin") {
#if KT2N_ENGINE_WITH_L7
    // Pool pages: lab knob KT2N_L7_POOL_PAGES (16.5 MB each; voice prefixes take shared pages too).
    int pages = 160;  // 2.6 GB; 64 ran out at N=10 (no_pages)
    if (const char* v = std::getenv("KT2N_L7_POOL_PAGES")) pages = std::max(8, std::atoi(v));
    e->render = make_l7_standin_render(cfg->device, std::min(64, e->row_cap + 4), pages, 64, &msg);
    if (!e->render) {
      set_err(err, KT2N_ERR_INVALID_ARGUMENT, "decoder: " + msg, "s3gen_file");
      return KT2N_ERR_INVALID_ARGUMENT;
    }
#else
    set_err(err, KT2N_ERR_UNSUPPORTED, "built without the decoder", "s3gen_file");
    return KT2N_ERR_UNSUPPORTED;
#endif
  } else if (cfg->s3gen_file) {
    // The real renderer: the decoder's Renderer over s3gen.kt2n (encoder + estimator + HiFT).
    L7RenderOptions ro;
    ro.s3gen_path = cfg->s3gen_file;
    ro.device = cfg->device;
    // Low-memory profile: one render-state slot beyond the rows (a call's rows are open rows, so the rows bucket cap
    // follows); an admission that finds no free slot waits (back-pressure), as when the rows are full.
    ro.slots = std::min(64, e->row_cap + (low_mem ? 1 : 4));
    // A call's rows bucket rounds its rows up (kRowBuckets), and the caller refuses a bucket above max_rows: with one
    // slot beyond the rows (low-memory profile) max_rows is the bucket of the slots, so any call of open rows fits.
    ro.max_rows = low_mem ? std::min(64, int(kt2n::render::bucket_of(kt2n::render::kRowBuckets, ro.slots))) : ro.slots;
    // Voice table and pool. Voices are append-only in the decoder (and the LM) today: a loaded / built voice keeps its table
    // row and prefix pages for the process lifetime. Headroom: 32 voices
    // (the inline LRU size), the same as the LM voice store below.
    // Pool: every catalogue voice's prefix (built on first use, kept) + the headroom voices' prefixes + 40 pages
    // per stream (~640 tokens per row; a row reserves its budget's pages at admission and waits when they are
    // not free). Lab knobs override (for fitting a 12 GB GPU). Earlier the default was 64 + 40 x streams,
    // so a mix of more than ~8 voices let first-use prefixes eat the rows' pages.
    // Table rows for custom voices: custom_voice_headroom() (72 by default; KT2N_L7_VOICE_HEADROOM = lab / test
    // knob). The pool keeps pages for 32 custom prefixes (below): more live custom prefixes than that take pages
    // from the rows' share, which is back-pressure (a first-use prefix waits for reservations), never a failure.
    const int extra_voices = inference_only ? 0 : custom_capacity > 0 ? custom_capacity : std::max(1, [cfg] {
      const char* v = std::getenv("KT2N_L7_VOICE_HEADROOM");
      return v && *v ? std::atoi(v) : custom_voice_headroom(cfg->inline_voice_lru > 0 ? cfg->inline_voice_lru : 32);
    }());
    const int pool_extra_voices = std::min(extra_voices, 32);
    int prefix_pages = 0, n_cat = 0, max_prefix = 0, max_prompt = 0;
    if (vblob)
      for (const auto& rec : vblob->voices) {
        const int pp = (2 * int(rec.prompt_token.size()) + 63) / 64 + 1;
        prefix_pages += pp;
        max_prefix = std::max(max_prefix, pp);
        max_prompt = std::max(max_prompt, int(rec.prompt_token.size()));
        ++n_cat;
      }
    const int per_voice = n_cat ? (prefix_pages + n_cat - 1) / n_cat : 10;
    ro.voices = n_cat + extra_voices;
    ro.pool_pages = prefix_pages + pool_extra_voices * per_voice + 40 * e->row_cap;
    if (cached_voices) {
      // Voice cache: table rows and prefix pages for the cache's budget, plus the rows' pages; the engine's cache (not
      // the backend's LRU) decides which voices keep prefixes.
      ro.voices = cache_table_rows;
      ro.pool_pages = cache_lim.max_pages + cache_row_pages;
      ro.engine_managed_prefixes = true;
    } else if (low_mem) {
      // Low-memory profile with cache mode ALL: the pool holds one prefix plus the rows' pages, not every catalogue
      // prefix. Prefixes are built on a voice's first use; an idle one is unbuilt (and rebuilt on its next use, the
      // same pages' contents) when a prefix or a row reservation needs its pages. Each row gets the largest token
      // budget a row can be admitted with (a streaming segment's cap or a stream:false chunk's, from position 0, for
      // the longest catalogue or clone prompt), so every request the default pool admits is admitted. A row that runs
      // past its budget (the noise extension) uses the pool's free pages as with any pool; render_pages sets the pool.
      const int prompt_cap = std::max(250, max_prompt);  // clones: the decoder reference is cropped to 10 s (250 tokens)
      const int budget_cap = std::max(kt2n::textplan::tagged_segment_budget(size_t(1) << 20), kt2n::textplan::Settings{}.demo_max_new_tokens);
      const int row_pages = (kt2n::render::kMelPerToken * (prompt_cap + budget_cap + kt2n::render::kLookahead) +
                             kt2n::render::kPageFrames - 1) / kt2n::render::kPageFrames;
      ro.pool_pages = std::max(1, max_prefix) + row_pages * e->row_cap;
    }
    auto env_int = [](const char* k, int d) {
      const char* v = std::getenv(k);
      return v && *v ? std::atoi(v) : d;
    };
    if (fit.fitted) ro.pool_pages = fit.pool;
    ro.pool_pages = std::max(16, cfg->render_pages > 0 ? cfg->render_pages : env_int("KT2N_L7_POOL_PAGES", ro.pool_pages));
    if (cached_voices) {
      // An explicit pool: the rows keep their share (at least 40 pages each) and the cache gets the rest. A fitted pool
      // with the complete-file give-back (fit.give_back) holds the prefix plus a streaming row at the fitted budget
      // and, without the prefix, a complete-file row: its exact sizes, checked by the fit.
      const int row_share = std::min(cache_row_pages, ro.pool_pages - cache_max_voice_pages);
      if (row_share < 40 * e->row_cap && !(fit.fitted && fit.give_back && cfg->render_pages <= 0)) {
        set_err(err, KT2N_ERR_INVALID_ARGUMENT, "render_pages cannot hold a voice prefix and the requested streams",
                "render_pages");
        return KT2N_ERR_INVALID_ARGUMENT;
      }
      cache_lim.max_pages = std::min(cache_lim.max_pages, ro.pool_pages - row_share);
    } else if (ro.pool_pages < (low_mem ? max_prefix : prefix_pages) + 40 * e->row_cap &&
               !(fit.fitted && cfg->render_pages <= 0)) {  // a fitted pool holds the exact row need (the fit)
      set_err(err, KT2N_ERR_INVALID_ARGUMENT, "render_pages cannot hold voice prefixes and requested streams", "render_pages");
      return KT2N_ERR_INVALID_ARGUMENT;
    }
    ro.workspace_mb = env_int("KT2N_L7_WORKSPACE_MB", fit.fitted ? fit.ws_mb : ro.workspace_mb);
    ro.hift_max_frame_rows = env_int("KT2N_L7_HIFT_FRAME_ROWS", 0);
    ro.hift_bulk_max_t = env_int("KT2N_L7_HIFT_BULK_T", fit.fitted && fit.frames < 4096 ? fit.frames : 0);  // lab: fitting a 12 GB GPU
    ro.hift_bulk_max_frame_rows = env_int("KT2N_L7_HIFT_BULK_FRAME_ROWS", fit.fitted && fit.frames < 4096 ? fit.frames : 0);
    ro.prewarm = env_int("KT2N_L7_PREWARM", fit.fitted ? 0 : 1) != 0;
    // Fitted low profile: the decoder launches eagerly (the same kernels and arguments as its graphs; no graph memory,
    // which is 6-8 MiB per captured graph on a 6 GB SM86 card).
    ro.render_graphs = env_int("KT2N_L7_RENDER_GRAPHS", fit.fitted ? 0 : 1) != 0;
    // Fitted: the encoder's position tables for the longest row the pool holds, and (one stream) one buffer for both
    // flow workspaces and the vocoder's scratch (render::RendererConfig).
    // Lab switches (fit_on): KT2N_FIT_POS_TABLES, KT2N_FIT_ALIAS, KT2N_FIT_PCM_POOL, KT2N_FIT_HIFT_MAXB, KT2N_FIT_GIVEBACK.
    ro.pos_tables_from_pool = fit.fitted && fit_on("KT2N_FIT_POS_TABLES");
    ro.alias_workspaces = fit.fitted && e->max_streams == 1 && fit_on("KT2N_FIT_ALIAS");
    ro.pcm_from_pool = fit.fitted && fit_on("KT2N_FIT_PCM_POOL");
    ro.demo_gives_back_prefix = fit.fitted && fit.give_back && fit_on("KT2N_FIT_GIVEBACK");
    ro.fitted = fit.fitted;  // its counters in the stats JSON
    // Fitted, one stream: HiFT launches of at most the rows a call can have (the rows bucket of the slots, 2) instead
    // of 32. Rows are grouped by width as before (a group never exceeds the call's rows), and the split-K partial
    // buffer shrinks to its 4 Mi-float floor, which no split choice reaches: SIMT splits stay below 512 blocks x
    // 512 outputs and tensor-core splits below 2 x (2 x SMs) blocks x 64 x 64 outputs (hift.cu launch_conv), i.e.
    // below 4 Mi for any GPU of up to 256 SMs, so every launch picks the same splits.
    if (fit.fitted && e->max_streams == 1 && fit_on("KT2N_FIT_HIFT_MAXB")) ro.hift_max_b = ro.max_rows;
    if (fit.fitted && !std::getenv("KT2N_L7_WORKSPACE_MB")) {
      // S_bulk's workspace: the exact need of its calls for the fitted rows (renderer.h bulk_ws_row_tokens).
      const kt2n::textplan::Settings ts{};
      ro.bulk_ws_row_tokens = fit.stream_budget > 0 ? fit.stream_budget
                                                    : std::max(kt2n::textplan::tagged_segment_budget(size_t(1) << 20),
                                                               ts.demo_max_new_tokens);
      ro.bulk_ws_demo_tokens = ts.demo_max_new_tokens;
    }
    if (low_mem) {
      e->fit.fitted = fit.fitted ? 1 : 0;
      e->fit.decoder_pages = ro.pool_pages;
      e->fit.vocoder_frames = ro.hift_bulk_max_t > 0 ? ro.hift_bulk_max_t : 4096;
      e->fit.render_graphs = ro.render_graphs ? 1 : 0;
      e->fit.free_after_context = free_before;
      e->fit.min_free_bytes = fit.floor;
      e->fit.streaming_budget_tokens = fit.stream_budget;
      if (ro.hift_bulk_max_t > 0 && ro.hift_bulk_max_t < 4096) e->fit_frames = ro.hift_bulk_max_t;
      e->fit_split = fit.fitted && fit.stream_budget > 0 && fit_on("KT2N_FIT_SPLIT");
      if (!inference_only && cfg->struct_size >= offsetof(kt2n_engine_config, custom_voice_max_prefix_ids) + sizeof(int32_t) &&
          cfg->custom_voice_max_prefix_ids > 0)
        e->custom_prefix_limit = cfg->custom_voice_max_prefix_ids <= 1024 ? cfg->custom_voice_max_prefix_ids : 0;
    }
    ro.prewarm_rows = env_int("KT2N_L7_PREWARM_ROWS", 0);
    ro.prewarm_tokens = env_int("KT2N_L7_PREWARM_TOKENS", ro.prewarm_tokens);
    ro.reserve_pages = env_int("KT2N_L7_RESERVE", 1) != 0;
    const bool cache_pool = low_mem || cached_voices;
    if (bounded_capacity && !cache_pool) ro.lazy_prefix = false;
    if (const char* v = std::getenv("KT2N_L7_PREFIXES")) ro.lazy_prefix = std::string(v) != "all";
    if (low_mem) {
      // Low-memory profile (none of these changes what a call renders; render/renderer.h):
      //   S_urgent's workspace sized to the calls it can get; S_bulk keeps workspace_mb (it decides how a
      //   stream:false batch is split);
      //   one HiFT handle for S_urgent and S_bulk when there is one stream (runs ordered on the GPU);
      //   the encoder's load-time sinusoid tables freed;
      //   graphs prewarmed for every voice prompt geometry and for rows up to the longest row, so serving captures
      //   (and allocates for) no graph.
      ro.urgent_ws_prompt_tokens = std::max(250, max_prompt);
      // Several rows decoding: a commit group can mix a commit 0 / 1 row with a row deep into a long segment, and the
      // whole group runs on S_urgent (stream_for of its most urgent row).
      ro.urgent_ws_long_rows = e->row_cap > 1;
      ro.hift_share = e->max_streams == 1;
      // Several streams (own S_urgent HiFT handle): S_urgent's HiFT rows are commit rows (a call with a commit 0 / 1
      // row; finals and whole renders go to S_bulk), so a row is at most the mel cache plus two frames per token of
      // the widest commit; a call has at most `slots` rows. Grouping is unchanged (up to max_b rows of 2,048 / T
      // frame-rows), so the handle needs T x min(rows, max_b) frame-rows. Conductor experiment switches that send
      // finals to S_urgent keep the default.
      const bool urgent_routing = std::getenv("KT2N_HOLD_URGENT") || std::getenv("KT2N_BUFFER_URGENT_MS") ||
                               std::getenv("KT2N_BUFFER_ROLE");
      if (!ro.hift_share && !urgent_routing && ro.hift_max_frame_rows == 0) {
        int widest = e->lv.preview_tokens;
        for (const auto* sched : {&e->lv.helper_schedule, &e->lv.alone_schedule})
          for (int w : *sched) widest = std::max(widest, w);
        const int t = kt2n::render::kMelCacheFrames + kt2n::render::kMelPerToken * widest;
        if (t <= 1024) {
          ro.hift_max_t = t;
          ro.hift_max_frame_rows = std::min(2048, t * std::min(ro.slots, 32));
        }
      }
      ro.free_pe_tables = true;
      ro.prewarm_geometries = true;
      ro.evict_while_reserved = true;
      ro.staging_from_rows = true;
      // Prefixes are built on a voice's first use here: finish the build before the row's LM work starts, so the
      // first commit does not queue behind it (commit timing then matches a voice built at create).
      ro.sync_prefix_build = !cached_voices;  // the voice cache builds prefixes before admission itself
      ro.prewarm_tokens = env_int("KT2N_L7_PREWARM_TOKENS", kt2n::render::kMaxRowTokens - std::max(250, max_prompt));
    }
    if (!cached_voices && !low_mem) {
      // Default profile, cache mode ALL: voice prefixes never take the rows' 40 pages per stream. Voices registered
      // beyond the pool's prefix room (the catalogue + 32 custom voices) build on first use, evicting idle prefixes
      // (least recently used) to stay under the cap, so requests are not admitted one at a time behind idle prefixes.
      ro.prefix_cap_pages = ro.pool_pages - 40 * e->row_cap;
    }
#if KT2N_ENGINE_WITH_L7
    e->render = make_l7_render(ro, &msg);
#else
    msg = "built without the decoder";
#endif
    if (!e->render) {
      set_err(err, KT2N_ERR_INVALID_ARGUMENT, "decoder: " + msg, "s3gen_file");
      return KT2N_ERR_INVALID_ARGUMENT;
    }
  }
  if (!e->render) {
    StubRenderConfig stub_cfg;
    // Tests only (the stub renderer): every n-th render call fails, to exercise the failure paths.
    if (const char* f = std::getenv("KT2N_TEST_STUB_RENDER_FAIL_EVERY")) stub_cfg.fail_every = std::atoi(f);
    if (const char* f = std::getenv("KT2N_TEST_STUB_RENDER_MAX_ROWS")) stub_cfg.max_open_rows = std::atoi(f);
    if (const char* f = std::getenv("KT2N_TEST_STUB_RENDER_PREFIX_WAIT")) stub_cfg.prefix_waits_for_rows = std::atoi(f) != 0;
    if (const char* v = std::getenv("KT2N_STUB_RENDER_CALL_US")) stub_cfg.call_us = std::atoi(v);
    if (const char* v = std::getenv("KT2N_STUB_RENDER_TOKEN_US")) stub_cfg.token_us = std::atof(v);
    stub_cfg.voice_required = cached_voices;
    e->render = make_stub_render(stub_cfg);
  }
  debug_mem("render create");
  if (cached_voices) {
    e->voice_cache_mode = cache_mode;
    e->vloader = std::make_unique<CacheLoader>(e.get());
    e->vcache = std::make_unique<VoiceCache>(cache_mode, cache_lim, e->vloader.get());
  }
  // Voices.
  try {
    if (vblob && cached_voices) {
      // Voice cache (LRU / NONE): keep every record on the host; the cache prepares a voice's GPU state on its first
      // use. Capture the render graphs of every prompt geometry now (a temporary voice per geometry, removed again),
      // so a voice's first use captures nothing live.
      e->catalogue = kt2n::voice::Catalogue::from_blob(*vblob);
      std::set<std::pair<int, int>> geometries;
      const bool prewarm = [] {
        const char* v = std::getenv("KT2N_VOICE_CACHE_PREWARM");
        return !(v && std::atoi(v) == 0);
      }();
      for (auto& rec : vblob->voices) {
        auto v = std::make_unique<Voice>();
        v->id = rec.id();
        v->names.push_back(v->id);
        if (const auto* pn = rec.meta.get("public_names"))
          for (const auto& n : pn->items()) v->names.push_back(n.as_string());
        if (const auto* l = rec.meta.get("language"); l && l->is_string() && !l->as_string().empty())
          v->language = l->as_string();
        if (const auto* h = rec.meta.get("hidden"); h && h->is_bool()) v->hidden = h->as_bool();
        if (const auto* g = rec.meta.get("peak_gain_db"); g && !g->is_null()) v->peak_gain_db = g->as_double();
        v->prompt_tokens = int(rec.prompt_token.size());
        v->prompt_frames = int(rec.prompt_feat.size() / 80);
        v->lm_prefix_ids = int(rec.lm_prefix.size());
        v->raw_speaker = rec.raw_speaker;
        v->geometry = policy::geometry_owner(v->prompt_tokens, v->prompt_frames, int(rec.embedding.size()));
        if (prewarm && geometries.insert({v->prompt_tokens, v->prompt_frames}).second) {
          if (e->render->prewarm_geometry(rec.prompt_token, rec.prompt_feat, rec.embedding, &msg) != 0) {
            if (require_assets) throw std::runtime_error("render voice " + v->id + ": " + msg);
            e->render_voice_errors++;
            e->render_voice_error = msg;
          }
        }
        auto host = std::make_shared<HostVoice>();
        host->prompt_token = std::move(rec.prompt_token);
        host->prompt_feat = std::move(rec.prompt_feat);
        host->embedding = std::move(rec.embedding);
        host->projected = std::move(rec.projected_speaker);
        host->lm_prefix = std::move(rec.lm_prefix);
        // A catalogue voice that cannot fit the budget alone would wait forever and block every voice behind it.
        const auto vc = kt2n::engine::voice_cost(int(host->prompt_token.size()), host->lm_prefix.size());
        if (!e->vcache->fits_alone(vc))
          throw std::runtime_error("voice " + v->id + " needs " + std::to_string(vc.pages) + " decoder pages and " +
                                   std::to_string((vc.total() + (1 << 20) - 1) >> 20) +
                                   " MiB, more than the voice cache budget");
        cache_add(e.get(), v.get(), std::move(host));
        e->voices.push_back(std::move(v));
      }
      debug_mem("voice cache prewarm");
    } else if (cached_voices) {
      for (const char* name : {"stub_voice_a", "stub_voice_b"}) {  // stub catalogue (tests)
        auto v = std::make_unique<Voice>();
        v->id = name;
        v->names = {name};
        v->prompt_tokens = 200;
        v->prompt_frames = 400;
        v->geometry = policy::geometry_owner(200, 400, 192);
        auto host = std::make_shared<HostVoice>();
        host->projected.assign(2048, 0.f);
        host->lm_prefix.assign(10, 1);
        host->prompt_token.assign(200, 0);
        cache_add(e.get(), v.get(), std::move(host));
        e->voices.push_back(std::move(v));
      }
    } else if (vblob) {
      e->catalogue = kt2n::voice::Catalogue::from_blob(*vblob);
      for (auto& rec : vblob->voices) {
        auto v = std::make_unique<Voice>();
        v->id = rec.id();
        v->names.push_back(v->id);
        if (const auto* pn = rec.meta.get("public_names"))
          for (const auto& n : pn->items()) v->names.push_back(n.as_string());
        // The voice's language: the reference voice catalogue's language code, casefolded,
        // before the first '-' (as the reference reads it; built in when the voices file is
        // prepared). An empty value (an older cloning check catalogue left every voice blank) falls back to
        // "en", and that fallback is NOT neutral: normalizer.cc prepare() takes code "en" as English before
        // language detection ever looks at the text, so a blank catalogue normalizes Hindi or other non-English
        // text with the English WFST.
        if (const auto* l = rec.meta.get("language"); l && l->is_string() && !l->as_string().empty())
          v->language = l->as_string();
        if (const auto* h = rec.meta.get("hidden"); h && h->is_bool()) v->hidden = h->as_bool();
        if (const auto* g = rec.meta.get("peak_gain_db"); g && !g->is_null()) v->peak_gain_db = g->as_double();
        v->prompt_tokens = int(rec.prompt_token.size());
        v->prompt_frames = int(rec.prompt_feat.size() / 80);
        v->lm_prefix_ids = int(rec.lm_prefix.size());
        v->raw_speaker = rec.raw_speaker;
        v->geometry = policy::geometry_owner(v->prompt_tokens, v->prompt_frames, int(rec.embedding.size()));
        v->lm_voice = e->lm->add_voice(rec.projected_speaker, rec.lm_prefix, &msg);
        if (v->lm_voice.load() < 0) throw std::runtime_error("LM voice " + v->id + ": " + msg);
        debug_mem(("lm voice " + std::to_string(rec.lm_prefix.size())).c_str());
        v->render_voice = e->render->add_voice(rec.prompt_token, rec.prompt_feat, rec.embedding, &msg);
        if (v->render_voice < 0) {
          if (require_assets) throw std::runtime_error("render voice " + v->id + ": " + msg);
          e->render_voice_errors++;
          e->render_voice_error = msg;
        }
        e->voices.push_back(std::move(v));
      }
    } else {
      // Stub catalogue (tests): two fake voices.
      for (const char* name : {"stub_voice_a", "stub_voice_b"}) {
        auto v = std::make_unique<Voice>();
        v->id = name;
        v->names = {name};
        v->prompt_tokens = 200;
        v->prompt_frames = 400;
        v->geometry = policy::geometry_owner(200, 400, 192);
        v->lm_voice = e->lm->add_voice(std::vector<float>(2048, 0.f), std::vector<int32_t>(10, 1), &msg);
        v->render_voice = e->render->add_voice({}, {}, {}, &msg);
        e->voices.push_back(std::move(v));
      }
    }
  } catch (const std::exception& ex) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, std::string("voices: ") + ex.what(), "voices_file");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  debug_mem("voices");
  // Clone builds (cloning): VoiceBuilder (its device arena is allocated here) + VoiceService on voice threads.
  e->clone_slots = cfg->clone_slots > 0 ? cfg->clone_slots : 4;
  e->clone_queue = cfg->clone_queue > 0 ? cfg->clone_queue : 16;
  e->clone_timeout_s = cfg->clone_timeout_ms > 0 ? cfg->clone_timeout_ms / 1000.0 : 120.0;
  const char* vbd = cfg->struct_size >= offsetof(kt2n_engine_config, voice_build_dir) + sizeof(const char*)
                        ? cfg->voice_build_dir
                        : nullptr;
  // Clone weights: from s3gen.kt2n when it carries the four clone groups, else from the
  // lab voice_build_dir (three files). voice_build_dir wins when both are given.
  bool kt2n_clone = false;
#if KT2N_ENGINE_WITH_VOICE
  if (!inference_only && !vbd && cfg->s3gen_file && std::string(cfg->s3gen_file) != "l7-standin") {
    try {
      kt2n::voice::CloneWeightsFile::open(cfg->s3gen_file);
      kt2n_clone = true;
    } catch (const std::exception&) {
      // an s3gen.kt2n without the clone groups: no clone service (kt2n_voice_build answers 214), as before
    }
  }
  // A low-memory engine fitted to a small GPU has no room for the cloning model (building a voice took ~4.4 GB more
  // than the low profile alone on an RTX PRO 6000: 12,408 vs 8,052 MiB): no clone service from the model file's clone
  // groups or from voice_build_dir (the CLI's --reference-audio), so saved voices (kt2n_voice_load) work instead of
  // create failing with an out-of-memory error; kt2n_voice_build answers 214 with that reason.
  if ((kt2n_clone || (vbd && !inference_only)) && e->fit.fitted) {
    std::fprintf(stderr, "[kt2n engine] voice cloning is off on this GPU: the low-memory profile uses the memory the "
                         "cloning model needs (about 4.4 GB more); build and save voices on a larger GPU, saved voices "
                         "load as usual\n");
    kt2n_clone = false;
    vbd = nullptr;
    e->cloning_off_fitted = true;
  }
#endif
  if (!inference_only && (vbd || kt2n_clone)) {
#if KT2N_ENGINE_WITH_VOICE
    if (!e->catalogue || e->catalogue->size() == 0 || !e->text->real_tokenizer()) {
      set_err(err, KT2N_ERR_INVALID_ARGUMENT, "clone builds need a raw speaker catalogue and tokenizer_json", "voice_build_dir");
      return KT2N_ERR_INVALID_ARGUMENT;
    }
    try {
      if (kt2n_clone) {
        e->vbuilder = std::make_unique<kt2n::voice::VoiceBuilder>(kt2n::voice::S3genKt2n{cfg->s3gen_file});
      } else {
        const std::string d = vbd;
        e->vbuilder = std::make_unique<kt2n::voice::VoiceBuilder>(d + "/s3gen.safetensors",
                                                                  d + "/voicebuild_consts.safetensors",
                                                                  d + "/voicebuild_pyannote.safetensors");
      }
      e->vbuilder->warmup();
      kt2n::engine::TextFront* tf = e->text.get();
      e->vservice = std::make_unique<kt2n::voice::VoiceService>(
          *e->vbuilder,
          [tf](const std::string& t) { return tf->encode(t); },
          e->clone_slots, e->clone_queue, e->clone_timeout_s, cfg->inline_voice_lru > 0 ? cfg->inline_voice_lru : 32);
      e->vservice->set_catalogue(&*e->catalogue);
      e->inline_index = std::make_unique<kt2n::voice::InlineVoiceIndex>(
          cfg->inline_voice_lru > 0 ? cfg->inline_voice_lru : 32);
    } catch (const std::exception& ex) {
      set_err(err, KT2N_ERR_INVALID_ARGUMENT, std::string("voice build: ") + ex.what(), "voice_build_dir");
      return KT2N_ERR_INVALID_ARGUMENT;
    }
#else
    set_err(err, KT2N_ERR_UNSUPPORTED, "built without voice cloning (KT2N_BUILD_CLONING, KT2N_VOICE_AV)", "voice_build_dir");
    return KT2N_ERR_UNSUPPORTED;
#endif
  }
  debug_mem("clone service");
  if (floor_check) {
    size_t free_after = 0, total_after = 0;
    const auto status = cudaMemGetInfo(&free_after, &total_after);
    if (status != cudaSuccess) {
      set_err(err, KT2N_ERR_INTERNAL, "cannot query initialized device memory");
      return KT2N_ERR_INTERNAL;
    }
    e->device_reserved_bytes = free_before > free_after ? free_before - free_after : 0;
    // Free-memory floor after initialization: min_free_device_bytes, or 10% of the device when it is 0 (the low
    // profile: the floor its fit kept, engine.h).
    const uint64_t min_free = mem_fields && cfg->min_free_device_bytes > 0 ? cfg->min_free_device_bytes
                              : low_mem && fit.floor > 0                   ? fit.floor
                                                                           : device_total / 10;
    if (low_mem && free_after < min_free && !(cfg->device_arena_bytes && e->device_reserved_bytes > cfg->device_arena_bytes)) {
      set_err(err, KT2N_ERR_TOO_SMALL,
              "the low-memory profile left " + std::to_string(free_after >> 20) + " MiB free after create, below the " +
                  std::to_string(min_free >> 20) + " MiB it keeps free (min_free_device_bytes); free more GPU memory",
              "memory_profile");
      return KT2N_ERR_TOO_SMALL;
    }
    if ((cfg->device_arena_bytes && e->device_reserved_bytes > cfg->device_arena_bytes) || free_after < min_free) {
      set_err(err, KT2N_ERR_TOO_SMALL,
              "model, streams and optional cloning exceed the memory budget; reduce capacity", "max_streams");
      return KT2N_ERR_TOO_SMALL;
    }
  }
  // Low-memory profile (small GPUs, often hosts with little RAM): the model loaders' temporary host copies (weights
  // rearranged before upload) are freed by now; give the freed heap back to the system instead of keeping it resident.
#if !defined(_WIN32)
  if (low_mem) malloc_trim(0);
#endif
#if defined(_WIN32)
  e->efd = -1;  // no event fd on Windows: kt2n_event_fd() returns -1 and callers wait in kt2n_poll()
#else
  e->efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
#endif
  // cpuset ("0-7,12"): every engine thread is pinned to it (plan: an 8-vCPU budget).
  cpu_set_t cpus;
  CPU_ZERO(&cpus);
  bool pin = false;
  if (cfg->cpuset && *cfg->cpuset) {
    std::stringstream ss(cfg->cpuset);
    std::string part;
    while (std::getline(ss, part, ',')) {
      const auto dash = part.find('-');
      const int lo = std::atoi(part.substr(0, dash).c_str());
      const int hi = dash == std::string::npos ? lo : std::atoi(part.substr(dash + 1).c_str());
      for (int c = lo; c <= hi && c < CPU_SETSIZE; ++c) {
        CPU_SET(c, &cpus);
        pin = true;
      }
    }
    if (!pin) {
      set_err(err, KT2N_ERR_INVALID_ARGUMENT, "cpuset is empty or malformed", "cpuset");
      return KT2N_ERR_INVALID_ARGUMENT;
    }
  }
  if (std::string why; !e->text->wait_ready(&why)) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "text front end: " + why, "text_data_dir");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  debug_mem("text front end ready");
  kt2n_engine* raw = e.get();
  const int nt = text_threads(cfg);
  const int np = cfg->threads_post > 0 ? cfg->threads_post : 2;
  for (int i = 0; i < nt; ++i) raw->threads.emplace_back([raw] { text_worker(raw); });
  for (int i = 0; i < np; ++i) raw->threads.emplace_back([raw] { post_worker(raw); });
  raw->threads.emplace_back([raw] { conductor(raw); });
  raw->threads.emplace_back([raw] { session_thread(raw); });
  const int nv = cfg->threads_voice > 0 ? cfg->threads_voice : 1;
  for (int i = 0; i < nv; ++i) raw->threads.emplace_back([raw] { voice_worker(raw); });
  if (pin)
    for (auto& t : raw->threads) pthread_setaffinity_np(t.native_handle(), sizeof cpus, &cpus);
  *out = e.release();
  return KT2N_OK;
}

// One create attempt on the caller's thread, starting and (when it fails) ending with no CUDA error pending on that
// thread. The CUDA runtime keeps the last error per host thread, and a failed create can leave one there (an
// out-of-memory cudaMalloc, e.g. "decoder: cudaMalloc(render arena)"). The next create on the same thread would read it
// at its first kernel-launch check (cudaGetLastError) and fail with an unrelated message ("LM: kt2n_pf_model_load:
// repack qkv failed (rc 2)"): the automatic profile's low-profile retry, or a caller retrying with other settings. A
// failed create's error is reported through kt2n_error, so nothing is lost by clearing it.
static int32_t engine_create_impl(const kt2n_engine_config* cfg, kt2n_engine** out, kt2n_error* err) {
  (void)cudaGetLastError();
  const int32_t rc = engine_create_attempt(cfg, out, err);
  if (rc != KT2N_OK) (void)cudaGetLastError();
  return rc;
}

// KT2N_MEMORY_PROFILE_AUTO: the default profile exactly as before when it fits, otherwise the low profile (which fits
// itself to the GPU). Each attempt is an ordinary DEFAULT or LOW create of the same config:
//   1. a LOW create with the probe set: when its fit has to shrink the sizes, the default profile cannot fit, and that
//      create simply finishes as the low profile; when the standard low sizes fit, it stops at the fit decision;
//   2. then a DEFAULT create; if it fails (it does not fit), a LOW create.
// One line on stderr says so whenever the low profile is chosen. Any other memory_profile value goes straight through.
// An explicit DEFAULT profile on a GPU that is too small for it: refuse with how much it needs before any model
// memory is allocated, instead of an allocation error from whichever part runs out first. Below kDefaultCheckBelow of
// free memory, the low profile's fit runs first without allocating (as AUTO's first step does) and the default profile
// is refused when that fit has to shrink its sizes, refuses itself, or the free memory is below the default profile's
// smallest possible need: the low profile's standard estimate + kAutoDefaultMargin + (voice cache ALL) the pages of
// every voice prefix the default pool holds + the default profile's free-memory floor. Otherwise the create runs as
// before. Lab switch: KT2N_DEFAULT_CHECK=0 skips the check.
// Lab switches that change the default profile's sizes (KT2N_L7_*, KT2N_LM_*: pool pages, context, workspaces,
// graphs, ...), which the default profile's estimate does not follow.
static bool size_lab_switch_set() {
#if defined(_WIN32)
  for (const char* k : {"KT2N_L7_POOL_PAGES", "KT2N_L7_WORKSPACE_MB", "KT2N_L7_PREWARM", "KT2N_L7_RENDER_GRAPHS",
                        "KT2N_L7_HIFT_FRAME_ROWS", "KT2N_L7_HIFT_BULK_T", "KT2N_L7_HIFT_BULK_FRAME_ROWS",
                        "KT2N_L7_PREWARM_ROWS", "KT2N_L7_PREWARM_TOKENS", "KT2N_LM_MAX_CTX", "KT2N_LM_HOST_EMBED",
                        "KT2N_LM_PREFILL_TOKENS"})
    if (std::getenv(k)) return true;
#else
  for (char** v = environ; v && *v; ++v)
    if (std::strncmp(*v, "KT2N_L7_", 8) == 0 || std::strncmp(*v, "KT2N_LM_", 8) == 0) return true;
#endif
  return false;
}

static int32_t default_create_checked(const kt2n_engine_config* cfg, kt2n_engine** out, kt2n_error* err) {
  const char* off = std::getenv("KT2N_DEFAULT_CHECK");
  // The estimate prices the pool the default profile chooses itself: an explicit render_pages (the default create
  // honours it and shrinks the voice cache budget to fit) or a lab switch for the sizes can make the engine smaller
  // than the estimate, so then the create runs unchecked, as before. (max_lm_rows only raises the LM rows, in both
  // profiles, and the estimate counts them.)
  if ((off && *off == '0') || !cfg->lm_file || !cfg->s3gen_file || std::strcmp(cfg->s3gen_file, "l7-standin") == 0 ||
      cfg->render_pages > 0 || size_lab_switch_set())
    return engine_create_impl(cfg, out, err);
  kt2n_engine_config c{};
  std::memcpy(&c, cfg, std::min<size_t>(sizeof c, cfg->struct_size));
  c.memory_profile = KT2N_MEMORY_PROFILE_LOW;
  AutoProbe probe;
  probe.decide_only = true;
  kt2n_error low_err{};
  int32_t rc;
  g_auto_probe = &probe;
  try {
    rc = engine_create_impl(&c, out, &low_err);
  } catch (...) {
    g_auto_probe = nullptr;
    throw;
  }
  g_auto_probe = nullptr;
  if (rc == KT2N_OK && *out) {  // cannot happen: the check stops before the engine exists
    kt2n_engine_destroy(*out);
    *out = nullptr;
  }
  const char* hint = c.voice_cache_mode == KT2N_VOICE_CACHE_ALL
                         ? "; with every voice prepared at start (voice cache all): one voice per stream (voice cache "
                           "none) needs less"
                         : "";
  const char* use = ". Use the automatic memory profile, which takes the low-memory profile on this GPU "
                    "(kitten-tts-native: leave out --memory-profile), or the low-memory profile";
  if (rc == KT2N_ERR_TOO_SMALL && std::strcmp(low_err.param, "memory_profile") == 0) {
    set_err(err, rc,
            std::string("the default memory profile does not fit this GPU, and neither does the low-memory profile: ") +
                low_err.message + hint,
            "memory_profile");
    return rc;
  }
  if (rc == KT2N_OK && probe.reached) {
    const uint64_t need =
        probe.std_need_bytes + kAutoDefaultMargin + probe.default_pool_extra_bytes + probe.default_floor_bytes;
    if (probe.fitted || probe.fit_free_bytes < need) {
      set_err(err, KT2N_ERR_TOO_SMALL,
              "the default memory profile needs about " + std::to_string(need >> 20) +
                  " MiB or more of free GPU memory; this GPU has " + std::to_string(probe.fit_free_bytes >> 20) +
                  " MiB free after the CUDA context" + (probe.default_pool_extra_bytes > 0 ? hint : "") + use,
              "memory_profile");
      return KT2N_ERR_TOO_SMALL;
    }
  }
  // Checked (or not checkable: the check stopped early, or the low create reported something else, which the
  // default create reports in its own words): create as before, with the tokenizer and voices file the check read.
  CreateHandoff handoff;
  if (rc == KT2N_OK && probe.reached) {
    handoff.text = std::move(probe.text);
    handoff.vblob = std::move(probe.vblob);
  }
  g_create_handoff = &handoff;
  try {
    rc = engine_create_impl(cfg, out, err);
  } catch (...) {
    g_create_handoff = nullptr;
    throw;
  }
  g_create_handoff = nullptr;
  return rc;
}

KT2N_API int32_t kt2n_engine_create(const kt2n_engine_config* cfg, kt2n_engine** out, kt2n_error* err) {
  const bool mem_block =
      cfg && cfg->struct_size >= offsetof(kt2n_engine_config, min_free_device_bytes) + sizeof(uint64_t);
  if (mem_block && cfg->memory_profile == KT2N_MEMORY_PROFILE_DEFAULT && out) return default_create_checked(cfg, out, err);
  if (!mem_block || cfg->memory_profile != KT2N_MEMORY_PROFILE_AUTO || !out) return engine_create_impl(cfg, out, err);
  kt2n_engine_config c{};
  std::memcpy(&c, cfg, std::min<size_t>(sizeof c, cfg->struct_size));
  // Whenever the low profile is chosen, one line says so: it is a fallback, supported but not optimal.
  auto say_low = [](kt2n_engine* e, const std::string& why) {
    kt2n_low_memory_limits lim{};
    lim.struct_size = sizeof lim;
    const bool fitted = kt2n_engine_low_memory_limits(e, &lim) == KT2N_OK && lim.fitted;
    std::fprintf(stderr, "kt2n: memory profile auto: using the low-memory profile, supported but not optimal: %s (%s)\n",
                 fitted ? "its sizes are fitted to this GPU and the decoder runs without CUDA graphs: complete-file "
                          "requests about 1.5% slower on an 8 GB GPU"
                        : "requests that arrive together are batched less than with the default profile; one at a "
                          "time runs as fast (real-time factor 0.050 vs 0.051 on a 16 GB GPU)",
                 why.c_str());
  };
  AutoProbe probe;
  c.memory_profile = KT2N_MEMORY_PROFILE_LOW;
  struct ProbeScope {  // the probe is set for this create only, also when it throws
    explicit ProbeScope(AutoProbe* p) { g_auto_probe = p; }
    ~ProbeScope() { g_auto_probe = nullptr; }
    ProbeScope(const ProbeScope&) = delete;
    ProbeScope& operator=(const ProbeScope&) = delete;
  };
  int32_t rc;
  {
    ProbeScope scope(&probe);
    rc = engine_create_impl(&c, out, err);
  }
  if (rc != KT2N_OK) {
    // The low profile does not fit, so neither does the default profile (its buffers are at least the low ones).
    // Its error says how much memory one stream needs; say that both profiles were considered, and name the voice
    // cache when every voice is prepared at start (one voice per stream needs less: the CLI's own default).
    if (err && rc == KT2N_ERR_TOO_SMALL && std::strcmp(err->param, "memory_profile") == 0) {
      std::string m = std::string("memory profile auto: the default profile does not fit, and ") + err->message;
      if (c.voice_cache_mode == KT2N_VOICE_CACHE_ALL)
        m += "; with every voice prepared at start (voice cache all): one voice per stream (voice cache none) needs less";
      set_err(err, rc, m, "memory_profile");
    }
    return rc;
  }
  const uint64_t free_mib = probe.free_bytes >> 20;
  if (!probe.stopped && !probe.reached) {  // finished without a fit decision (cannot happen with real files): retry
    kt2n_engine_destroy(*out);
    *out = nullptr;
    probe.stopped = true;
  }
  if (!probe.stopped) {  // the default profile cannot fit: the low profile carried on
    (*out)->auto_profile = "low";
    say_low(*out, std::to_string(free_mib) + " MiB free after the CUDA context; the default profile needs about " +
                      std::to_string((probe.std_need_bytes + kAutoDefaultMargin + probe.default_floor_bytes) >> 20) +
                      " MiB or more");
    return KT2N_OK;
  }
  c.memory_profile = KT2N_MEMORY_PROFILE_DEFAULT;
  kt2n_error default_err{};
  rc = engine_create_impl(&c, out, &default_err);
  if (rc == KT2N_OK) {
    (*out)->auto_profile = "default";
    return KT2N_OK;
  }
  c.memory_profile = KT2N_MEMORY_PROFILE_LOW;  // engine_create_impl clears the failed attempt's CUDA error
  rc = engine_create_impl(&c, out, err);
  if (rc == KT2N_OK) {
    (*out)->auto_profile = "low";
    say_low(*out, std::to_string(free_mib) + " MiB free after the CUDA context; the default profile did not fit: " +
                      (default_err.message[0] ? std::string(default_err.message) : std::string("create failed")));
  } else if (err) {  // both failed: keep both reasons
    set_err(err, rc,
            std::string("memory profile auto: the default profile failed (") +
                (default_err.message[0] ? default_err.message : "create failed") + "), then the low profile: " +
                err->message,
            err->param);
  }
  return rc;
}

KT2N_API void kt2n_engine_destroy(kt2n_engine* e) {
  if (!e) return;
  {
    // Publish shutdown under the same mutex as the event wait predicate: no lost wake-up.
    std::lock_guard<std::mutex> lk(e->ev_mu);
    e->stop = true;
  }
  e->ev_cv.notify_all();
  // Pair every indefinite wait with its mutex. Otherwise a worker can observe stop=false,
  // miss the notification just before entering wait, and keep destroy's join blocked.
  auto wake = [](std::mutex& mu, std::condition_variable& cv) {
    std::lock_guard<std::mutex> lk(mu);
    cv.notify_all();
  };
  wake(e->mu, e->cv);
  wake(e->text_mu, e->text_cv);
  wake(e->post_mu, e->post_cv);
  wake(e->sess_mu, e->sess_cv);
  wake(e->build_mu, e->build_cv);
  for (auto& t : e->threads) t.join();
  // Voice cache: loads still in flight write into the LM store and the render pool; let them finish first.
  if (e->vcache) e->vcache->drain(30.0);
  e->render.reset();
  e->lm.reset();
  {
    std::unique_lock<std::mutex> lk(e->ev_mu);
    e->ev_drained.wait(lk, [&] { return e->pollers == 0; });
    for (EventRec* r : e->events) delete r;
    e->events.clear();
  }
  if (e->efd >= 0) close(e->efd);
  delete e;
}

static Voice* voice_of(kt2n_engine* e, kt2n_voice_id id) {
  std::lock_guard<std::mutex> lk(e->voices_mu);
  if (id == KT2N_INVALID_ID || id > e->voices.size()) return nullptr;
  Voice* v = e->voices[size_t(id - 1)].get();
  return v->evicted.load() ? nullptr : v;
}

KT2N_API int32_t kt2n_voice_lookup(kt2n_engine* e, const char* name, kt2n_voice_id* out, kt2n_error* err) {
  if (!e || !name || !out) return KT2N_ERR_INVALID_ARGUMENT;
  {
    std::lock_guard<std::mutex> lk(e->voices_mu);
    for (size_t i = 0; i < e->voices.size(); ++i)
      if (e->voices[i]->kind == KT2N_VOICE_BUILTIN)
        for (const auto& n : e->voices[i]->names)
          if (n == name) {
            *out = i + 1;
            return KT2N_OK;
          }
  }
  set_err(err, KT2N_ERR_INVALID_VOICE, std::string("Voice '") + name + "' is not available.", "voice");
  return KT2N_ERR_INVALID_VOICE;
}

KT2N_API int32_t kt2n_voice_info_get(kt2n_engine* e, kt2n_voice_id id, kt2n_voice_info* out) {
  if (!e || !out) return KT2N_ERR_INVALID_ARGUMENT;
  Voice* v = voice_of(e, id);
  if (!v) return KT2N_ERR_INVALID_HANDLE;
  kt2n_voice_info info{};
  info.struct_size = sizeof info;
  info.kind = v->kind;
  std::snprintf(info.name, sizeof info.name, "%s", v->id.c_str());
  std::snprintf(info.language, sizeof info.language, "%s", v->language.c_str());
  info.prompt_tokens = v->prompt_tokens;
  info.prompt_frames = v->prompt_frames;
  info.lm_prefix_ids = v->lm_prefix_ids;
  info.hidden = v->hidden;
  info.peak_gain_db = float(v->peak_gain_db);
  info.refs = v->refs.load();
  std::memcpy(out, &info, std::min<size_t>(out->struct_size ? out->struct_size : sizeof info, sizeof info));
  return KT2N_OK;
}

KT2N_API int32_t kt2n_voice_speaker(kt2n_engine* e, kt2n_voice_id id, float out[512]) {
  Voice* v = e ? voice_of(e, id) : nullptr;
  if (!v) return KT2N_ERR_INVALID_HANDLE;
  if (v->raw_speaker.size() != 512) return KT2N_ERR_UNSUPPORTED;
  std::memcpy(out, v->raw_speaker.data(), 512 * sizeof(float));
  return KT2N_OK;
}

KT2N_API int32_t kt2n_voice_build(kt2n_engine* e, const kt2n_voice_build_params* p, kt2n_job_id* out,
                                  kt2n_error* err) {
  if (!e || !p || !out) return KT2N_ERR_INVALID_ARGUMENT;
#if !KT2N_ENGINE_WITH_VOICE
  set_err(err, KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE, "built without voice cloning");
  return KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE;
#else
  if (!e->vservice && !e->voice_build_exec) {
    if (e->cloning_off_fitted) {
      set_err(err, KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE,
              "voice cloning is off on this GPU: the low-memory profile uses the memory the cloning model needs (about "
              "4.4 GB more); build and save the voice on a larger GPU, then load it with kt2n_voice_load");
      return KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE;
    }
    set_err(err, KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE, "clone builds need voice_build_dir");
    return KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE;
  }
  // A keyless KT2N_VB_INLINE build is never cached (cloning VoiceService) and never enters the inline LRU, so the caller
  // owns its eviction (kt2n_voice_evict; the conductor then releases its store space). The reference always
  // sends a key. The reference cloning tests exercise the keyless path (builds succeed, the caller evicts).
  // Inline: transcript rules first, synchronously (the reference checks them before the slot).
  // Saved and consent creates take the slot first in the reference (both creates run inside
  // the management slot), so the cloning VoiceService checks their transcript after admission (429 before 400).
  std::string stripped;
  const std::string why = (p->flags & KT2N_VB_INLINE) && !(p->flags & KT2N_VB_CONSENT)
                              ? kt2n::voice::VoiceService::validate_transcript(
                                    p->reference_transcript ? p->reference_transcript : "", &stripped)
                              : std::string();
  if (!why.empty()) {
    const bool consent = (p->flags & KT2N_VB_CONSENT) != 0;
    set_err(err, consent ? KT2N_ERR_INVALID_CONSENT_TRANSCRIPT : KT2N_ERR_INVALID_REFERENCE_TRANSCRIPT, why,
            consent ? "consent_transcript" : "reference_transcript");
    return consent ? KT2N_ERR_INVALID_CONSENT_TRANSCRIPT : KT2N_ERR_INVALID_REFERENCE_TRANSCRIPT;
  }
  if ((p->flags & KT2N_VB_VERIFY_SPEAKER) && (p->consent_speaker_dim != 512 || !p->consent_speaker)) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "consent_speaker must be 512 floats");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  auto j = std::make_shared<BuildJob>();
  j->user_data = p->user_data;
  j->flags = p->flags;
  if (p->flags & KT2N_VB_DECODED_PCM) {
    if (!p->pcm || !p->pcm_samples) {
      set_err(err, KT2N_ERR_INVALID_AUDIO, "no audio", "file");
      return KT2N_ERR_INVALID_AUDIO;
    }
    j->pcm.assign(p->pcm, p->pcm + p->pcm_samples);
    j->pcm_rate = p->pcm_rate;
  } else {
#if !KT2N_VOICE_DECODE
    set_err(err, KT2N_ERR_UNSUPPORTED, kt2n::voice::kNoDecodeMessage, "audio");
    return KT2N_ERR_UNSUPPORTED;
#else
    if (!p->audio || !p->audio_len) {
      set_err(err, KT2N_ERR_INVALID_AUDIO, "no audio", "file");
      return KT2N_ERR_INVALID_AUDIO;
    }
    j->audio.assign(static_cast<const uint8_t*>(p->audio), static_cast<const uint8_t*>(p->audio) + p->audio_len);
#endif
  }
  if (p->flags & KT2N_VB_VERIFY_SPEAKER) j->consent.assign(p->consent_speaker, p->consent_speaker + 512);
  j->transcript = p->reference_transcript ? p->reference_transcript : "";
  j->language = p->language ? p->language : "";
  j->identity_key = p->identity_key ? p->identity_key : "";
  j->deadline_ns = p->deadline_ns ? p->deadline_ns : mono_ns() + int64_t(e->clone_timeout_s * 1e9);
  j->queued_s = mono_s();
  {
    std::lock_guard<std::mutex> lk(e->build_mu);
    // Admission (reference default: 4 running + 16 waiting, else custom_voice_creation_busy).
    if (e->builds_running + int(e->build_q.size()) >= e->clone_slots + e->clone_queue) {
      set_err(err, KT2N_ERR_CUSTOM_VOICE_CREATION_BUSY, "Custom voice creation is busy. Retry shortly.");
      return KT2N_ERR_CUSTOM_VOICE_CREATION_BUSY;
    }
    {
      std::lock_guard<std::mutex> ilk(e->mu);
      if (e->backend_failed.load()) {
        set_err(err, KT2N_ERR_SERVICE_UNAVAILABLE, "engine backend failed");
        return KT2N_ERR_SERVICE_UNAVAILABLE;
      }
      j->id = e->next_id++;
    }
    *out = j->id;
    e->build_q.push_back(j);
    e->builds.emplace(j->id, std::move(j));
  }
  e->build_cv.notify_all();
  return KT2N_OK;
#endif
}
KT2N_API int32_t kt2n_voice_artifact_lm_prefix_ids(const void* artifact, size_t len) {
  // engine.h: a negative value when the artifact cannot be read (a status code would be positive).
  if (!artifact || len == 0) return -KT2N_ERR_INVALID_ARGUMENT;
  try {
    const kt2n::voice::VoicesBlob b = kt2n::voice::parse_voices_blob(artifact, len, true);
    if (b.voices.size() != 1) return -KT2N_ERR_INVALID_ARGUMENT;
    return int32_t(b.voices[0].lm_prefix.size());
  } catch (const std::exception&) {
    return -KT2N_ERR_INVALID_ARGUMENT;
  }
}

KT2N_API int32_t kt2n_voice_load(kt2n_engine* e, const void* artifact, size_t len, kt2n_voice_id* out,
                                 kt2n_error* err) {
  if (!e || !artifact || !out) return KT2N_ERR_INVALID_ARGUMENT;
  if (!e->catalogue) {
    set_err(err, KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE, "no built-in catalogue loaded (voices_file)");
    return KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE;
  }
  // the reference saved-voice load gate via cloning check_saved_voice.
  kt2n::voice::VoicesBlob parsed;
  const auto verdict = kt2n::voice::check_saved_voice(artifact, len, *e->catalogue, &parsed);
  if (verdict.status != 0) {
    std::string why = verdict.code;
    for (const auto& m : verdict.mismatched) why += " " + m;
    set_err(err, verdict.status, why, verdict.param);
    return verdict.status;
  }
  const auto& rec = parsed.voices[0];
  auto v = std::make_unique<Voice>();
  v->kind = KT2N_VOICE_SAVED;
  v->id = rec.id();
  v->names = {v->id};
  v->peak_gain_db = kt2n::voice::kCustomPeakGainDb;  // custom voices: the catalogue median
  if (const auto* l = rec.meta.get("language"); l && l->is_string() && !l->as_string().empty()) v->language = l->as_string();
  v->prompt_tokens = int(rec.prompt_token.size());
  v->prompt_frames = int(rec.prompt_feat.size() / 80);
  v->lm_prefix_ids = int(rec.lm_prefix.size());
  if (e->custom_prefix_limit > 0 && v->lm_prefix_ids > e->custom_prefix_limit) {
    // engine.h custom_voice_max_prefix_ids: the low-memory profile sized its context and voice store for saved voices of
    // at most this many prefix ids, in every voice-cache mode.
    set_err(err, KT2N_ERR_GPU_MEMORY_LIMIT,
            "this saved voice has " + std::to_string(v->lm_prefix_ids) +
                " language-model prompt ids; the low-memory profile was sized for at most " +
                std::to_string(e->custom_prefix_limit) + " (custom_voice_max_prefix_ids)",
            "custom_voice_max_prefix_ids");
    return KT2N_ERR_GPU_MEMORY_LIMIT;
  }
  v->raw_speaker = rec.raw_speaker;
  v->geometry = policy::geometry_owner(v->prompt_tokens, v->prompt_frames, int(rec.embedding.size()));
  v->artifact.assign(static_cast<const uint8_t*>(artifact), static_cast<const uint8_t*>(artifact) + len);
  if (e->vcache) {
    // Voice cache (LRU / NONE): a voice larger than the whole budget could never be prepared; say so now instead of
    // returning an id that the next submit finds evicted. (fits_alone reads only limits fixed at create.)
    const auto c = kt2n::engine::voice_cost(int(rec.prompt_token.size()), rec.lm_prefix.size());
    if (!e->vcache->fits_alone(c)) {
      set_err(err, KT2N_ERR_CUSTOM_VOICE_SERVICE_ERROR,
              "the voice needs " + std::to_string(c.total() >> 20) + " MiB of GPU memory, more than the voice cache budget",
              "voice");
      return KT2N_ERR_CUSTOM_VOICE_SERVICE_ERROR;
    }
  }
  *out = register_voice(e, std::move(v), rec.prompt_token, rec.prompt_feat, rec.embedding, rec.projected_speaker,
                        rec.lm_prefix);
  if (!*out) {
    set_err(err, KT2N_ERR_SERVICE_UNAVAILABLE, "engine backend failed");
    return KT2N_ERR_SERVICE_UNAVAILABLE;
  }
  return KT2N_OK;
}

KT2N_API int32_t kt2n_voice_export(kt2n_engine* e, kt2n_voice_id id, void* buf, size_t cap, size_t* len) {
  if (!e || !len) return KT2N_ERR_INVALID_ARGUMENT;
  Voice* v = voice_of(e, id);
  if (!v) return KT2N_ERR_INVALID_HANDLE;
  if (v->artifact.empty()) return KT2N_ERR_INVALID_ARGUMENT;  // built-ins are not exported
  *len = v->artifact.size();
  if (!buf || cap < v->artifact.size()) return KT2N_ERR_TOO_SMALL;
  std::memcpy(buf, v->artifact.data(), v->artifact.size());
  return KT2N_OK;
}

KT2N_API int32_t kt2n_voice_evict(kt2n_engine* e, kt2n_voice_id id) {
  Voice* v = e ? voice_of(e, id) : nullptr;
  if (!v) return KT2N_ERR_INVALID_HANDLE;
  if (v->kind == KT2N_VOICE_BUILTIN) return KT2N_ERR_INVALID_ARGUMENT;
  v->evicted = true;  // new submits fail; requests holding it finish with it
  e->voice_sweep = true;  // the conductor gives its LM store space back once no request holds it
  e->cv.notify_all();
  return KT2N_OK;
}

// Voice cache LRU: prepare a voice's GPU state now (the conductor owns the cache: the warm and every state check run
// as conductor jobs) and wait until it is resident, its load failed, or the cache could not take it without dropping a
// voice (a warm load never drops anything; the voice is then prepared on first use, as before). ALL: prepared at
// create; NONE: prepared per request. Either way nothing to do. While it waits it counts as a poller, so a destroy that
// starts meanwhile wakes it (KT2N_ERR_NOT_READY) and frees the engine only after it has left.
KT2N_API int32_t kt2n_voice_prepare(kt2n_engine* e, kt2n_voice_id id, kt2n_error* err) {
  if (!e) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "engine is NULL");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  {
    std::lock_guard<std::mutex> lk(e->ev_mu);
    if (e->stop.load()) {
      set_err(err, KT2N_ERR_NOT_READY, "the engine is shutting down");
      return KT2N_ERR_NOT_READY;
    }
    ++e->pollers;
  }
  struct Leave {  // destroy waits for pollers == 0 before it frees the engine
    kt2n_engine* e;
    ~Leave() {
      std::lock_guard<std::mutex> lk(e->ev_mu);
      if (--e->pollers == 0) e->ev_drained.notify_all();
    }
  } leave{e};
  Voice* v = voice_of(e, id);
  if (!v || v->consent_only) {
    set_err(err, KT2N_ERR_INVALID_VOICE, "voice is not available", "voice");
    return KT2N_ERR_INVALID_VOICE;
  }
  if (!e->vcache || e->voice_cache_mode != KT2N_VOICE_CACHE_LRU) return KT2N_OK;
  enum : int { kPending = -1, kNotLoading = 0, kResident = 1, kLoading = 2, kFailed = 3 };
  auto give_up = [&](int32_t status, const char* message) {
    set_err(err, status, message, status == KT2N_ERR_NOT_READY ? "" : "voice");
    return status;
  };
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  for (bool first = true;; first = false) {
    auto state = std::make_shared<std::atomic<int>>(kPending);
    {
      std::lock_guard<std::mutex> lk(e->mu);
      if (e->backend_failed.load()) return give_up(KT2N_ERR_SERVICE_UNAVAILABLE, "the engine failed");
      e->admin_q.push_back([e, v, state, first] {
        const int k = v->cache_key;
        if (k < 0 || v->evicted.load()) {  // not registered with the cache (yet): prepared on first use
          state->store(kNotLoading);
          return;
        }
        if (first) e->vcache->warm(k);
        using Res = kt2n::engine::VoiceCache::Res;
        const Res r = e->vcache->res(k);
        state->store(r == Res::kResident                       ? kResident
                     : r == Res::kQueued || r == Res::kLoading ? kLoading
                     : e->vcache->load_failed(k)               ? kFailed
                                                               : kNotLoading);
      });
    }
    e->cv.notify_all();
    while (state->load() == kPending) {
      if (e->stop.load()) return give_up(KT2N_ERR_NOT_READY, "the engine is shutting down");
      if (e->backend_failed.load()) return give_up(KT2N_ERR_SERVICE_UNAVAILABLE, "the engine failed");
      if (std::chrono::steady_clock::now() > deadline) return KT2N_OK;  // still queued: the request waits for it
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    const int s = state->load();
    if (s == kFailed) return give_up(KT2N_ERR_SERVICE_UNAVAILABLE, "the voice could not be prepared on the GPU");
    if (s != kLoading || std::chrono::steady_clock::now() > deadline) return KT2N_OK;
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
}

KT2N_API int32_t kt2n_submit(kt2n_engine* e, const kt2n_request_params* p, kt2n_request_id* out, kt2n_error* err) {
  if (!e || !p || !out || p->struct_size < offsetof(kt2n_request_params, user_data)) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "params");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  std::string text(p->text ? p->text : "", p->text ? p->text_len : 0);
  const auto u32 = kt2n::textplan::from_utf8(text);
  if (kt2n::textplan::py_strip(u32).empty()) {
    set_err(err, KT2N_ERR_INVALID_INPUT, "input must not be blank", "input");
    return KT2N_ERR_INVALID_INPUT;
  }
  if (u32.size() > 40000) {
    set_err(err, KT2N_ERR_REQUEST_TOO_LARGE, "input must be at most 40000 characters", "input");
    return KT2N_ERR_REQUEST_TOO_LARGE;
  }
  Voice* v = voice_of(e, p->voice);
  if (!v || v->consent_only) {
    set_err(err, KT2N_ERR_INVALID_VOICE, "voice is not available", "voice");
    return KT2N_ERR_INVALID_VOICE;
  }
  if (!(p->speed >= 0.25 && p->speed <= 4.0)) {
    set_err(err, KT2N_ERR_INVALID_REQUEST, "speed must be between 0.25 and 4.0", "speed");
    return KT2N_ERR_INVALID_REQUEST;
  }
  if (p->format < KT2N_FORMAT_PCM || (p->format > KT2N_FORMAT_FLAC && p->format != KT2N_FORMAT_F32)) {
    set_err(err, KT2N_ERR_INVALID_RESPONSE_FORMAT, "unsupported response_format", "response_format");
    return KT2N_ERR_INVALID_RESPONSE_FORMAT;
  }
#if !KT2N_ENGINE_WITH_AV
  if (encoded_format(p->format)) {
    set_err(err, KT2N_ERR_UNSUPPORTED, "built without the in-process encoders (KT2N_AV)", "response_format");
    return KT2N_ERR_UNSUPPORTED;
  }
#endif
  if (p->speech_mode == KT2N_SPEECH_MODE_FAST || p->speech_mode == KT2N_SPEECH_MODE_QUALITY ||
      (p->flags & KT2N_REQ_DEMO_CHECK)) {
    // Only DEFAULT and complete generation (stream:false) are supported; the others fail closed.
    set_err(err, KT2N_ERR_UNSUPPORTED, "this speech mode is not supported; use KT2N_SPEECH_MODE_DEFAULT", "speech_mode");
    return KT2N_ERR_UNSUPPORTED;
  }
  if ((p->flags & KT2N_REQ_TIMELINE) && !(e->flags & KT2N_CFG_TIMELINE_ALLOWED)) {
    set_err(err, KT2N_ERR_INVALID_REQUEST, "timeline not allowed on this engine", "timeline");
    return KT2N_ERR_INVALID_REQUEST;
  }
  // Sampler (the reference resolve with the default mode = expressive).
  kt2n::lm::SamplerRequest sr;
  const kt2n_sampler& s = p->sampler;
  if (s.present & KT2N_SAMPLER_HAS_MODE) sr.mode = s.mode == KT2N_SAMPLER_STABLE ? "stable" : "expressive";
  if (s.present & KT2N_SAMPLER_HAS_TEMPERATURE) sr.temperature = s.temperature;
  if (s.present & KT2N_SAMPLER_HAS_TOP_P) sr.top_p = s.top_p;
  if (s.present & KT2N_SAMPLER_HAS_TOP_K) sr.top_k = double(s.top_k);
  if (s.present & KT2N_SAMPLER_HAS_MIN_P) sr.min_p = s.min_p;
  if (s.present & KT2N_SAMPLER_HAS_MAX_NEW_TOKENS) sr.max_new_tokens = double(s.max_new_tokens);
  kt2n::lm::SamplerSettings settings;
  std::string param;
  const std::string msg = kt2n::lm::resolve_sampler(sr, kt2n::lm::SamplerMode::kExpressive, &settings, &param);
  if (!msg.empty()) {
    set_err(err, KT2N_ERR_INVALID_REQUEST, msg, param);
    return KT2N_ERR_INVALID_REQUEST;
  }
  // control_json (appended): the request's generation-control policy fields. Known keys are applied as
  // the reference applies them; any other key fails closed so a request is never served under another policy.
  std::optional<double> hold_pinned;
  std::optional<bool> emo_preset, skip_open, phrase, normalize;
  const char* cj = p->struct_size >= offsetof(kt2n_request_params, control_json) + sizeof(const char*)
                       ? p->control_json
                       : nullptr;
  if (cj && *cj) {
    kt2n::json::Value doc;
    try {
      doc = kt2n::json::parse(cj);
    } catch (const std::exception&) {
      set_err(err, KT2N_ERR_INVALID_ARGUMENT, "control_json is not valid JSON", "control_json");
      return KT2N_ERR_INVALID_ARGUMENT;
    }
    if (!doc.is_object()) {
      set_err(err, KT2N_ERR_INVALID_ARGUMENT, "control_json must be an object", "control_json");
      return KT2N_ERR_INVALID_ARGUMENT;
    }
    for (const auto& [k, cv] : doc.members()) {
      bool ok = true;
      if (cv.is_null()) {
        ok = true;  // "not set"
      } else if (k == "startup_hold_seconds") {
        ok = cv.is_number();
        if (ok) hold_pinned = cv.as_double();
      } else if (k == "initial_audio_buffer_seconds" || k == "concurrent_initial_audio_buffer_seconds") {
        ok = cv.is_number() && cv.as_double() == 0.0;  // 0 by default
      } else if (k == "normalize") {
        ok = cv.is_bool();
        if (ok) normalize = cv.as_bool();
      } else if (k == "emotion") {
        ok = cv.is_bool();
        if (ok) emo_preset = cv.as_bool();
      } else if (k == "emotion_carry_tag") {
        ok = cv.is_string();  // read only with leading-tag carry (off by default)
      } else if (k == "skip_opening_silence") {
        ok = cv.is_bool();
        if (ok) skip_open = cv.as_bool();
      } else if (k == "q_skip_limiter") {
        ok = cv.is_bool();  // the internal-silence limiter is off by default either way
      } else if (k == "demo_check") {
        ok = cv.is_bool() && !cv.as_bool();  // DEMO+CHECK is unreachable at the default settings
      } else if (k == "phrase") {
        ok = cv.is_bool();
        if (ok) phrase = cv.as_bool();
      } else {
        ok = false;
      }
      if (!ok) {
        set_err(err, KT2N_ERR_UNSUPPORTED, "control field not served by the native engine: " + k, k);
        return KT2N_ERR_UNSUPPORTED;
      }
    }
  }
  auto q = std::make_unique<Request>();
  q->user_data = p->user_data;
  q->flags = p->flags;
  q->hold_pinned = hold_pinned;
  q->emotion_preset = emo_preset;
  q->realtime = phrase.value_or(false);
  q->skip_opening = skip_open.value_or(true);
  q->normalize = normalize.value_or((p->flags & KT2N_REQ_NORMALIZE_OFF) == 0);
  q->timeline = (p->flags & KT2N_REQ_TIMELINE) != 0;
  q->voice = v;
  q->raw = text;
  q->sampler = settings;
  q->speed = p->speed;
  q->format = p->format;
  q->output = p->output;
  q->demo = p->output == KT2N_OUTPUT_COMPLETE || p->speech_mode == KT2N_SPEECH_MODE_DEMO;
  q->deadline_ns = p->deadline_ns;
  q->started_s = mono_s();
  q->salt = (p->flags & KT2N_REQ_SEED) ? p->seed : policy::request_seed_salt(p->request_id ? p->request_id : "");
  kt2n::post::ResponsePostConfig rc;
  rc.speed = p->speed;
  rc.onset_trim = !q->demo;
  rc.withhold = !q->demo;
  q->rpost = std::make_unique<kt2n::post::ResponsePost>(rc);
  kt2n::post::HoldPolicy hold_policy;
  if (q->hold_pinned) hold_policy.pinned = *q->hold_pinned;
  q->hold = std::make_unique<kt2n::post::StartupHold>(hold_policy);
  // A streamed non-DEMO response publishes through the hold (complete files and DEMO rows have no commits on it).
  q->awaiting_hold.store(q->output == KT2N_OUTPUT_STREAM && !q->demo, std::memory_order_relaxed);
#if KT2N_ENGINE_WITH_AV
  if (encoded_format(p->format)) {
    std::string why;
    q->enc = kt2n::post::StreamEncoder::create(format_name(p->format), &why);
    if (!q->enc) {
      set_err(err, KT2N_ERR_INTERNAL, "encoder: " + why, "response_format");
      return KT2N_ERR_INTERNAL;
    }
  }
#endif
  if (q->output == KT2N_OUTPUT_STREAM) {
    // Ceiling 0.999, lossy formats 0.891 (PeakNorm.for_request).
    q->peak.emplace(false, v->peak_gain_db, kt2n::post::is_lossy_format(format_name(p->format)) ? 0.891 : 0.999, 5.0);
    if (!(q->hold_pinned && *q->hold_pinned <= 0.0)) q->clock.hold();  // the startup hold withholds the first byte
  }
  if (!register_request(e, std::move(q), out)) {
    set_err(err, KT2N_ERR_SERVICE_UNAVAILABLE, "engine backend failed");
    return KT2N_ERR_SERVICE_UNAVAILABLE;
  }
  return KT2N_OK;
}

KT2N_API int32_t kt2n_voice_build_cancel_queued(kt2n_engine* e, kt2n_job_id id, int32_t* cancelled) {
  if (!e || !cancelled) return KT2N_ERR_INVALID_ARGUMENT;
  *cancelled = 0;
  // Queue membership and voice_worker dequeue use the same mutex. No later
  // cancellation can interrupt an admitted stored create through this API.
  std::lock_guard<std::mutex> lk(e->build_mu);
  for (const auto& job : e->build_q) {
    if (job->id == id) {
      job->cancelled = true;
      *cancelled = 1;
      return KT2N_OK;
    }
  }
  return KT2N_OK;
}

KT2N_API int32_t kt2n_cancel(kt2n_engine* e, uint64_t id, int32_t reason) {
  if (!e) return KT2N_ERR_INVALID_ARGUMENT;
  {
    std::lock_guard<std::mutex> blk(e->build_mu);
    auto bit = e->builds.find(id);
    if (bit != e->builds.end()) {
      bit->second->cancelled = true;
      e->build_cv.notify_all();
      return KT2N_OK;
    }
  }
  {
    std::lock_guard<std::mutex> slk(e->sess_mu);
    if (Session* s = session_of(e, id)) {
      if (!s->cancelled) {
        s->cancelled = true;
        s->cancel_status = reason == KT2N_CANCEL_DEADLINE ? KT2N_ERR_DEADLINE_EXCEEDED : KT2N_ERR_CANCELLED;
      }
      std::vector<uint64_t> phrases = s->phrase_ids;
      std::lock_guard<std::mutex> lk(e->mu);
      for (uint64_t pid : phrases) {
        auto it = e->requests.find(pid);
        if (it != e->requests.end() && !it->second->cancelled.load()) {
          it->second->cancel_status = s->cancel_status;
          it->second->cancelled = true;
        }
      }
      e->cv.notify_all();
      e->sess_cv.notify_all();
      return KT2N_OK;
    }
  }
  std::lock_guard<std::mutex> lk(e->mu);
  auto it = e->requests.find(id);
  if (it == e->requests.end()) return KT2N_ERR_INVALID_HANDLE;
  Request* q = it->second.get();
  if (!q->cancelled.load()) {
    q->cancel_status = reason == KT2N_CANCEL_DEADLINE ? KT2N_ERR_DEADLINE_EXCEEDED : KT2N_ERR_CANCELLED;
    q->cancelled = true;
  }
  e->cv.notify_all();
  return KT2N_OK;
}


KT2N_API int32_t kt2n_release(kt2n_engine* e, uint64_t id) {
  if (!e) return KT2N_ERR_INVALID_ARGUMENT;
  {
    std::lock_guard<std::mutex> blk(e->build_mu);
    auto bit = e->builds.find(id);
    if (bit != e->builds.end()) {
      // The worker retains ownership through its final callback and loop accounting. Keep a local
      // owner too, so erasing the map cannot destroy the mutex while jl still holds it.
      const auto j = bit->second;
      std::lock_guard<std::mutex> jl(j->ev_lock);
      if (!j->terminal) return KT2N_ERR_INVALID_ARGUMENT;
      e->builds.erase(bit);
      return KT2N_OK;
    }
  }
  {
    std::lock_guard<std::mutex> slk(e->sess_mu);
    if (Session* s = session_of(e, id)) {
      if (!s->terminal) return KT2N_ERR_INVALID_ARGUMENT;
      // Phrases emit through the session object, so it is kept (small) until engine destroy; its phrase
      // requests are released for the conductor's sweep.
      std::lock_guard<std::mutex> lk(e->mu);
      for (uint64_t pid : s->phrase_ids) {
        auto pit = e->requests.find(pid);
        if (pit != e->requests.end()) pit->second->release_requested = true;
      }
      s->phrase_ids.clear();
      e->release_pending = true;
      return KT2N_OK;
    }
  }
  std::lock_guard<std::mutex> lk(e->mu);
  auto it = e->requests.find(id);
  if (it == e->requests.end()) return KT2N_ERR_INVALID_HANDLE;
  {
    std::lock_guard<std::mutex> qlk(it->second->mu);
    if (!it->second->terminal) return KT2N_ERR_INVALID_ARGUMENT;
  }
  // The conductor frees it once no row or post item points at it (its loop sweeps released requests).
  if (it->second->release_requested.exchange(true)) return KT2N_ERR_INVALID_HANDLE;
  e->release_pending = true;
  e->cv.notify_all();
  return KT2N_OK;
}

KT2N_API int32_t kt2n_session_open(kt2n_engine* e, const kt2n_session_params* p, kt2n_session_id* out,
                                   kt2n_error* err) {
  if (!e || !p || !out || p->struct_size < sizeof(kt2n_session_params)) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "session params are NULL or too small");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  if (p->format != KT2N_FORMAT_PCM && p->format != KT2N_FORMAT_F32) {
    set_err(err, KT2N_ERR_INVALID_RESPONSE_FORMAT, "sessions support PCM16 or float32 audio", "format");
    return KT2N_ERR_INVALID_RESPONSE_FORMAT;
  }
  Voice* v = voice_of(e, p->voice);
  if (!v || v->consent_only) {
    set_err(err, KT2N_ERR_INVALID_VOICE, "voice is not available", "voice");
    return KT2N_ERR_INVALID_VOICE;
  }
  if (v->kind != KT2N_VOICE_BUILTIN) {
    set_err(err, KT2N_ERR_CUSTOM_VOICE_REALTIME_NOT_SUPPORTED, "Custom voices are not supported on realtime sessions.",
            "voice");
    return KT2N_ERR_CUSTOM_VOICE_REALTIME_NOT_SUPPORTED;
  }
  if (!(p->speed >= 0.25 && p->speed <= 4.0)) {
    set_err(err, KT2N_ERR_INVALID_SESSION_CONFIGURATION, "speed must be between 0.25 and 4.0", "speed");
    return KT2N_ERR_INVALID_SESSION_CONFIGURATION;
  }
  kt2n::lm::SamplerRequest sr;
  const kt2n_sampler& sm = p->sampler;
  if (sm.present & KT2N_SAMPLER_HAS_MODE) sr.mode = sm.mode == KT2N_SAMPLER_STABLE ? "stable" : "expressive";
  if (sm.present & KT2N_SAMPLER_HAS_TEMPERATURE) sr.temperature = sm.temperature;
  if (sm.present & KT2N_SAMPLER_HAS_TOP_P) sr.top_p = sm.top_p;
  if (sm.present & KT2N_SAMPLER_HAS_TOP_K) sr.top_k = double(sm.top_k);
  if (sm.present & KT2N_SAMPLER_HAS_MIN_P) sr.min_p = sm.min_p;
  if (sm.present & KT2N_SAMPLER_HAS_MAX_NEW_TOKENS) sr.max_new_tokens = double(sm.max_new_tokens);
  kt2n::lm::SamplerSettings settings;
  std::string param;
  const std::string msg = kt2n::lm::resolve_sampler(sr, kt2n::lm::SamplerMode::kExpressive, &settings, &param);
  if (!msg.empty()) {
    set_err(err, KT2N_ERR_INVALID_REQUEST, msg, param);
    return KT2N_ERR_INVALID_REQUEST;
  }
  auto s = std::make_unique<Session>();
  s->normalize = (p->flags & KT2N_REQ_NORMALIZE_OFF) == 0;
  s->user_data = p->user_data;
  s->flags = p->flags;
  s->voice = v;
  s->request_id = p->request_id ? p->request_id : "";
  s->seeded = (p->flags & KT2N_REQ_SEED) != 0;
  s->salt = s->seeded ? p->seed : policy::request_seed_salt(s->request_id);
  s->sampler = settings;
  s->speed = p->speed;
  s->format = p->format;
  s->deadline_ns = p->deadline_ns;
  s->opened_s = mono_s();
  s->peak.emplace(false, v->peak_gain_db, 0.999, 5.0);
  {
    std::lock_guard<std::mutex> slk(e->sess_mu);
    std::lock_guard<std::mutex> lk(e->mu);
    if (e->backend_failed.load()) {
      set_err(err, KT2N_ERR_SERVICE_UNAVAILABLE, "engine backend failed");
      return KT2N_ERR_SERVICE_UNAVAILABLE;
    }
    s->id = e->next_id++;
    *out = s->id;
    e->sessions.emplace(s->id, std::move(s));
  }
  e->sess_cv.notify_all();
  return KT2N_OK;
}

// A refused client event ends the session as the reference socket handler does:
// input after done -> input_already_closed (invalid_request_error, close 1008); an oversized message or
// session -> ValueError -> speech_generation_failed (server_error, close 1011). sess_mu held.
static int32_t gate_error(kt2n_engine* e, Session* s, kt2n_error* err, kt2n::textplan::WsInputGate::Verdict v) {
  using V = kt2n::textplan::WsInputGate::Verdict;
  int32_t st = KT2N_OK;
  switch (v) {
    case V::kInputClosed:
      set_err(err, KT2N_ERR_INPUT_ALREADY_CLOSED, "Realtime TTS input is already closed");
      st = KT2N_ERR_INPUT_ALREADY_CLOSED;
      break;
    case V::kMessageTooLarge:
    case V::kInputTooLarge:
      set_err(err, KT2N_ERR_SPEECH_GENERATION_FAILED, "Speech generation failed");
      st = KT2N_ERR_SPEECH_GENERATION_FAILED;
      break;
    default:
      return KT2N_OK;
  }
  if (!s->cancelled) {
    s->cancelled = true;
    s->cancel_status = st;
    std::lock_guard<std::mutex> lk(e->mu);
    for (uint64_t pid : s->phrase_ids) {
      auto it = e->requests.find(pid);
      if (it != e->requests.end() && !it->second->cancelled.load()) {
        it->second->cancel_status = KT2N_ERR_CANCELLED;
        it->second->cancelled = true;
      }
    }
  }
  e->cv.notify_all();
  e->sess_cv.notify_all();
  return st;
}

KT2N_API int32_t kt2n_session_text_append(kt2n_engine* e, kt2n_session_id id, const char* text, size_t len,
                                          kt2n_error* err) {
  if (!e) return KT2N_ERR_INVALID_ARGUMENT;
  const auto u32 = kt2n::textplan::from_utf8(std::string_view(text ? text : "", text ? len : 0));
  std::lock_guard<std::mutex> lk(e->sess_mu);
  Session* s = session_of(e, id);
  if (!s) return KT2N_ERR_INVALID_HANDLE;
  const auto v = s->gate.append(u32.size());
  if (v == kt2n::textplan::WsInputGate::Verdict::kSkip) return KT2N_OK;
  if (v != kt2n::textplan::WsInputGate::Verdict::kForward) return gate_error(e, s, err, v);
  s->planner.feed(kt2n::textplan::WsPlanner::EventType::kText, u32);
  s->dirty = true;
  e->sess_cv.notify_all();
  return KT2N_OK;
}

KT2N_API int32_t kt2n_session_commit(kt2n_engine* e, kt2n_session_id id, kt2n_error* err) {
  if (!e) return KT2N_ERR_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> lk(e->sess_mu);
  Session* s = session_of(e, id);
  if (!s) return KT2N_ERR_INVALID_HANDLE;
  const auto v = s->gate.commit();
  if (v != kt2n::textplan::WsInputGate::Verdict::kForward) return gate_error(e, s, err, v);
  s->planner.feed(kt2n::textplan::WsPlanner::EventType::kCommit);
  s->dirty = true;
  e->sess_cv.notify_all();
  return KT2N_OK;
}

KT2N_API int32_t kt2n_session_done(kt2n_engine* e, kt2n_session_id id, kt2n_error* err) {
  if (!e) return KT2N_ERR_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> lk(e->sess_mu);
  Session* s = session_of(e, id);
  if (!s) return KT2N_ERR_INVALID_HANDLE;
  const auto v = s->gate.done();
  if (v != kt2n::textplan::WsInputGate::Verdict::kForward) return gate_error(e, s, err, v);
  s->planner.feed(kt2n::textplan::WsPlanner::EventType::kDone);
  s->dirty = true;
  e->sess_cv.notify_all();
  return KT2N_OK;
}

KT2N_API int kt2n_event_fd(kt2n_engine* e) { return e ? e->efd : -1; }

KT2N_API int32_t kt2n_poll(kt2n_engine* e, kt2n_event* out, int32_t max, int32_t timeout_ms, int32_t* n) {
  if (!e || !out || !n || max <= 0) return KT2N_ERR_INVALID_ARGUMENT;
  std::unique_lock<std::mutex> lk(e->ev_mu);
  ++e->pollers;
  if (e->events.empty() && timeout_ms != 0) {
    auto pred = [&] { return !e->events.empty() || e->stop.load(); };
    if (timeout_ms < 0)
      e->ev_cv.wait(lk, pred);
    else
      cv_wait_for(e->ev_cv, lk, std::chrono::milliseconds(timeout_ms), pred);
  }
  // Shutdown returns no borrowed buffers: callers already waiting may finish safely while destroy
  // drains them. Callers must still stop new calls and release any earlier borrowed events first.
  if (e->stop.load()) {
    *n = 0;
    if (--e->pollers == 0) e->ev_drained.notify_all();
    return KT2N_OK;
  }
  int32_t k = 0;
  while (k < max && !e->events.empty()) {
    EventRec* r = e->events.front();
    e->events.pop_front();
    r->ev.data = r->data.empty() ? nullptr : reinterpret_cast<const uint8_t*>(r->data.data());
    r->ev.data_len = r->data.size();
    r->ev.name = r->name.empty() ? nullptr : r->name.c_str();
    r->ev.fields_json = r->fields.empty() ? nullptr : r->fields.c_str();
    out[k++] = r->ev;
    e->outstanding[{r->ev.id, r->ev.seq}].reset(r);
  }
  *n = k;
  if (e->events.empty() && e->efd >= 0) {
    uint64_t v;
    (void)!read(e->efd, &v, sizeof v);
  }
  if (--e->pollers == 0) e->ev_drained.notify_all();
  return KT2N_OK;
}

KT2N_API void kt2n_events_release(kt2n_engine* e, const kt2n_event* events, int32_t n) {
  if (!e || !events) return;
  std::lock_guard<std::mutex> lk(e->ev_mu);
  for (int32_t i = 0; i < n; ++i) e->outstanding.erase({events[i].id, events[i].seq});
}

KT2N_API int32_t kt2n_engine_low_memory_limits(kt2n_engine* e, kt2n_low_memory_limits* out) {
  if (!e || !out || out->struct_size < sizeof(uint32_t)) return KT2N_ERR_INVALID_ARGUMENT;
  kt2n_low_memory_limits v = e->fit;
  v.struct_size = out->struct_size < sizeof(v) ? out->struct_size : uint32_t(sizeof(v));
  std::memcpy(out, &v, v.struct_size);
  return KT2N_OK;
}

KT2N_API int32_t kt2n_stats_get(kt2n_engine* e, kt2n_stats* out) {
  if (!e || !out) return KT2N_ERR_INVALID_ARGUMENT;
  kt2n_stats s{};
  s.struct_size = sizeof s;
  s.device_arena_bytes = e->device_reserved_bytes;
  s.device_arena_used = e->device_reserved_bytes;
  s.active_requests = e->active_requests.load();
  s.lm_rows = e->lm_rows.load();
  s.render_inflight = e->render_inflight.load();
  if (e->vcache) {
    s.voices_loaded = int32_t(e->vcache->resident_count());
  } else {
    std::lock_guard<std::mutex> lk(e->voices_mu);
    s.voices_loaded = int32_t(e->voices.size());
  }
  s.requests_done = e->done.load();
  s.requests_failed = e->failed.load();
  s.requests_cancelled = e->cancelled.load();
  s.lm_steps = e->lm_steps.load();
  s.renders = e->renders.load();
  s.render_rows = e->render_rows.load();
  std::memcpy(out, &s, std::min<size_t>(out->struct_size ? out->struct_size : sizeof s, sizeof s));
  return KT2N_OK;
}

KT2N_API int32_t kt2n_stats_json(kt2n_engine* e, char* buf, size_t cap, size_t* len) {
  if (!e || !len) return KT2N_ERR_INVALID_ARGUMENT;
  size_t held;
  {
    std::lock_guard<std::mutex> lk(e->mu);
    held = e->requests.size();
  }
  std::ostringstream o;
  // "render":"stub" whenever the audio is placeholder (CPU stub or the decoder's stand-in stages): a caller's stub
  // detection can read it; render_impl names the backend.
  const std::string rimpl = e->render->name();
  const bool placeholder = rimpl == "stub" || rimpl == "l7-standin";
  o << "{\"lm\":\"" << e->lm->name() << "\",\"render\":\"" << (placeholder ? "stub" : rimpl)
    << "\",\"render_impl\":\"" << rimpl
    << "\",\"normalizer\":\"" << (e->text->real_normalizer() ? "native" : "passthrough") << "\",\"tokenizer\":\""
    << (e->text->real_tokenizer() ? "native" : "codepoints") << "\",\"active_requests\":" << e->active_requests.load()
    << ",\"lm_rows\":" << e->lm_rows.load() << ",\"lm_steps\":" << e->lm_steps.load()
    << ",\"renders\":" << e->renders.load() << ",\"render_rows\":" << e->render_rows.load()
    << ",\"requests_done\":" << e->done.load() << ",\"requests_failed\":" << e->failed.load()
    << ",\"requests_cancelled\":" << e->cancelled.load()
    << ",\"requests_held\":" << held << ",\"startup_sharing\":" << e->sharing.load()
    << ",\"steady_requests\":" << e->steady.load() << ",\"previews_due\":" << e->previews_due.load()
    << ",\"lm_backpressure\":" << e->lm_backpressure.load()
    << ",\"render_backpressure\":" << e->render_backpressure.load()
    << (e->fit.fitted ? ",\"fit_split_chunks\":" + std::to_string(e->fit_split_chunks.load()) : std::string())
    << ",\"poll\":" << (e->poll ? "true" : "false")
    << ",\"poll_idle\":" << e->poll_idle.load() << ",\"render_state\":" << e->render->stats_json()
    // The yield keys appear only when KT2N_LM_RENDER_YIELD is on, so the flag-off stats text stays the
    // earlier builds' byte for byte (a caller may read stats into a fixed 4,096-byte buffer).
    << [&] {
         if (!e->lm_render_yield) return std::string();
         return ",\"lm_render_yield\":true,\"lm_render_yield_episodes\":" +
                std::to_string(e->lm_render_yield_episodes.load()) +
                ",\"lm_render_yield_switches\":" + std::to_string(e->lm_render_yield_switches.load());
       }()
    << ",\"render_voice_errors\":" << e->render_voice_errors << ",\"voice_store_full\":" << e->voice_store_full.load()
    << ",\"lm_voices_released\":" << e->lm_voices_released.load()
    << ",\"render_voices_released\":" << e->render_voices_released.load()
    // Voice cache (LRU / NONE only, so the ALL-mode text is unchanged byte for byte).
    << (e->vcache ? ",\"voice_cache\":" + e->vcache->stats_json() : std::string())
    << ",\"turns\":" << e->turns.load() << ",\"turn_us_mean\":"
    << (e->turns.load() ? double(e->turn_ns.load()) / 1000.0 / double(e->turns.load()) : 0.0)
    << ",\"turn_us_max\":" << double(e->turn_ns_max.load()) / 1000.0
    << ",\"turns_over_1ms\":" << e->turns_over_1ms.load() << ",\"turns_over_5ms\":" << e->turns_over_5ms.load()
    << ",\"turn_max_decide_us\":" << double(e->turn_max_decide_ns.load()) / 1000.0
    << ",\"turn_max_rows\":" << e->turn_max_rows.load()
    << ",\"calls_per_turn\":{\"commit_turns\":" << e->commit_turns.load() << ",\"commit_groups\":" << e->commit_groups.load()
    << ",\"commit_same_stream_extra\":" << e->commit_same_stream_extra.load()
    << ",\"commit_extra_rows\":" << e->commit_extra_rows.load() << ",\"commit_mixed_groups\":" << e->commit_mixed_groups.load()
    << ",\"commit_mixed_later_rows\":" << e->commit_mixed_later_rows.load() << ",\"final_turns\":" << e->final_turns.load()
    << ",\"final_calls\":" << e->final_calls.load() << ",\"final_extra_calls\":" << e->final_extra_calls.load() << "}"
    << ",\"split_urgent\":" << (e->split_urgent ? "true" : "false") << ",\"split_urgent_calls\":" << e->split_urgent_calls.load()
    << ",\"hold_urgent\":" << (e->hold_urgent > 0 ? "true" : "false") << ",\"hold_urgent_level\":" << e->hold_urgent
    << ",\"hold_urgent_calls\":" << e->hold_urgent_calls.load() << ",\"hold_urgent_finals\":" << e->hold_urgent_finals.load()
    << ",\"hold_urgent_rows\":" << e->hold_urgent_rows.load() << ",\"hold_urgent_splits\":" << e->hold_urgent_splits.load()
    << ",\"buffer_urgent_ms\":" << e->buffer_urgent_ms << ",\"buffer_urgent_rows\":" << e->buffer_urgent_rows.load()
    << ",\"buffer_urgent_finals\":" << e->buffer_urgent_finals.load()
    << ",\"buffer_role\":\"" << (e->buffer_to_aux ? "aux" : "urgent") << "\",\"buffer_mid_calls\":" << e->buffer_mid_calls.load()
    << ",\"lm_step_ms_by_rows\":{" << [&] {
         // "21+" (all steps above 20 rows, as before) plus its parts 21-32 / 33-48 / 49+ for the capacity runs.
         static const char* names[9] = {"1-4", "5-8", "9-12", "13-16", "17-20", "21+", "21-32", "33-48", "49+"};
         std::string by_rows;  // not "o": that name is the enclosing stats stream (-Werror=shadow)
         for (int b = 0; b < 9; ++b) {
           uint64_t n = 0, ns = 0;
           for (int k = (b < 5 ? b : b == 5 ? 5 : b - 1); k <= (b < 5 ? b : b == 5 ? 7 : b - 1); ++k) {
             n += e->lm_iv_steps[k].load();
             ns += e->lm_iv_ns[k].load();
           }
           if (!n) continue;
           if (!by_rows.empty()) by_rows += ",";
           by_rows += std::string("\"") + names[b] + "\":{\"steps\":" + std::to_string(n) + ",\"ms\":" +
                      jnum(double(ns) / 1e6 / double(n)) + "}";
         }
         return by_rows;
       }() << "}"
    << ",\"x_sched\":" << ((e->flags & KT2N_CFG_X_SCHED) ? "true" : "false")
    << ",\"dw_runway_ms\":" << e->lv.dw_runway_ms << ",\"max_streams\":" << e->max_streams
    << ",\"engine_rows\":" << e->row_cap << ",\"engine_rows_why\":\"" << e->row_cap_why << "\""
    << ",\"capacity_order\":" << (e->capacity_order && e->row_cap < e->max_streams ? "true" : "false")
    // KT2N_MEMORY_PROFILE_AUTO only (other engines' stats text is unchanged byte for byte)
    << (e->auto_profile.empty() ? std::string() : ",\"memory_profile_auto\":\"" + e->auto_profile + "\"") << "}";
  const std::string s = o.str();
  *len = s.size() + 1;
  if (!buf || cap < s.size() + 1) return KT2N_ERR_TOO_SMALL;
  std::memcpy(buf, s.c_str(), s.size() + 1);
  return KT2N_OK;
}

}  // extern "C"
