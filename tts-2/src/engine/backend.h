// The two execution backends the conductor drives. The conductor makes the reference
// decisions (policy.h); a backend only executes them.
//   LmBackend      the torch-free LM (LM kt2n::lm::Engine), or a CPU stub for tests.
//   RenderBackend  the token -> PCM renderer (render state + encoder/estimator/HiFT op library through
//                  include/kt2n/render/ops.h), or a CPU stub.
// Both are called only from the conductor thread (except stats()).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kt2n/lm.h"
#include "kt2n/render/ops.h"

namespace kt2n::engine {

// ------------------------------------------------------------------------------------- LM
class LmBackend {
 public:
  virtual ~LmBackend() = default;
  // Per-voice LM prefix (voices.kt2n lm_prefix + projected speaker). Returns the LM voice id.
  virtual int add_voice(const std::vector<float>& projected_speaker, const std::vector<int32_t>& lm_prefix,
                        std::string* err) = 0;
  virtual void drop_voice(int lm_voice) = 0;
  // Gives the voice's store space back (LM release_voice): no new rows; rows still reading it keep it until their
  // slots are released. Conductor thread. 0, or -1 for an unknown / already released id.
  virtual int release_voice(int lm_voice) = 0;
  // Voice cache: add_voice without waiting for the GPU (LM build_voice_async). The id is usable once voice_ready()
  // returns 1. -2 = no room in the voice store now (retry after releasing a voice), -1 = failure. Conductor thread.
  virtual int add_voice_async(const std::vector<float>& projected_speaker, const std::vector<int32_t>& lm_prefix,
                              std::string* err) {
    return add_voice(projected_speaker, lm_prefix, err);
  }
  // 1 = ready, 0 = still building, -1 = unknown / released / failed.
  virtual int voice_ready(int lm_voice) { return lm_voice >= 0 ? 1 : -1; }
  virtual int submit(const lm::RowRequest& row, std::string* err) = 0;
  virtual void cancel(uint64_t tag) = 0;
  // Appends the tokens of every completed step. block: wait for at least one event when rows are live.
  virtual int pump(std::vector<lm::TokenEvent>* events, bool block, std::string* err) = 0;
  // Execution-only experimental option. The LM driver orders a priority switch after earlier decode work.
  virtual int set_decode_priority(int level) {
    (void)level;
    return -1;
  }
  virtual int live_rows() const = 0;
  virtual uint64_t steps() const = 0;
  virtual const char* name() const = 0;
};

// The real LM (LM). Only with CUDA.
std::unique_ptr<LmBackend> make_l3_lm(const lm::Config& cfg, std::string* err);

// CPU stub: deterministic audio tokens from the seed (a short opening of sub-gate ids, then
// "speech" ids), EOS after ~3.4 tokens per suffix id or at the budget; step_us of simulated
// decode time per step. For runtime tests and plumbing only; never in a release build.
struct StubLmConfig {
  int step_us = 500;
  double tokens_per_suffix_id = 3.4;
  int opening_quiet_tokens = 3;
  int voice_build_us = 0;          // add_voice_async: the voice is ready this long after the call
  int voice_build_fail_first = 0;  // add_voice_async: the first n calls fail
  int fail_after_steps = 0;        // tests: pump fails (an engine failure) once this many steps ran (0 = never)
  bool voice_required = false;     // voice cache on: a row without a voice (a given-back id is -1) is a violation
};
std::unique_ptr<LmBackend> make_stub_lm(const StubLmConfig& cfg);

// ------------------------------------------------------------------------------------- render
struct RenderRowReq {
  uint64_t row = 0;           // conductor row key (owns a render-state slot from open_row to close_row)
  int voice = 0;              // engine voice index
  render::Phase phase = render::Phase::kSteadyCommit;
  int commit_index = 0;
  int committed_tokens = 0;   // tokens published before this call
  int commit_tokens = 0;      // the commit boundary after this call (next_commit_tokens); finals: all tokens
  bool finalize = false;
  std::vector<int32_t> tokens;  // commit: ids[:next+3]; final: ids + [4299] x 3; preview: ids[:8]
  uint64_t seq = 0;           // per-row publication order
};
struct RenderCall {
  uint64_t call_id = 0;
  render::StreamRole stream = render::StreamRole::kSUrgent;
  bool async = true;          // the reference launch class (async launch rule); native executes both overlapped
  std::vector<RenderRowReq> rows;
};
struct RenderOut {
  uint64_t call_id = 0;
  int status = 0;                       // 0 ok, else kt2n_status
  std::vector<std::vector<float>> pcm;  // per row of the call, the samples it publishes (float, [-1, 1])
  double gpu_ms = 0;
  double launch_ms = 0;
  std::string error;  // a failed call's reason (internal; logs / lab)
  int graph_nodes = 0;          // decoder: nodes of the call's captured graph (flow kernels)
  bool graph_captured = false;  // decoder: this call captured its bucket's graph (capture time is in its wall)
};

// open_row: the renderer cannot take the row yet (no render-state slot, or its pool pages cannot be reserved).
// The conductor puts the job back and retries on a later turn (admission back-pressure, never a failure).
inline constexpr int kRenderBackpressure = 1;

class RenderBackend {
 public:
  virtual ~RenderBackend() = default;
  virtual int add_voice(const std::vector<int32_t>& prompt_tokens, const std::vector<float>& prompt_feat,
                        const std::vector<float>& embedding, std::string* err) = 0;
  // Admission: a render-state slot + the row's noise tables (seed, budget). 0 ok, kRenderBackpressure = retry
  // later, < 0 = failure.
  virtual int open_row(uint64_t row, int voice, uint64_t seed, int budget, bool demo, std::string* err) = 0;
  virtual void close_row(uint64_t row) = 0;
  // Give an evicted voice's table row and prefix pages back (the decoder's RenderService::remove_voice). 0 ok, -2 busy (rows or
  // a prefix build of it still in flight: call again later), -1 failure / not supported (never retried).
  virtual int remove_voice(int voice, std::string* err) {
    (void)voice;
    if (err) *err = "remove_voice: not supported";
    return -1;
  }
  // Voice cache (engine-managed prefixes): build the voice's shared prefix now instead of at its first open_row. It is
  // queued on the GPU and returns without waiting; rows of the voice are ordered after it. 0 = built (or nothing to
  // build), kRenderBackpressure = the pool cannot hold it now (retry later), -1 = failure.
  virtual int prepare_prefix(int voice, std::string* err) {
    (void)voice;
    (void)err;
    return 0;
  }
  // Voice cache: 1 = the voice's prefix (prepare_prefix) is complete on the GPU, 0 = still building, -1 = unknown voice.
  virtual int prefix_ready(int voice) {
    (void)voice;
    return 1;
  }
  // Voice cache: capture the graphs rows of this prompt geometry can hit, with a temporary voice that is removed again
  // before returning (create time only, no rows open). 0 ok, -1 failure (the engine keeps going: graphs missing from
  // the prewarm are captured on first use).
  virtual int prewarm_geometry(const std::vector<int32_t>& prompt_tokens, const std::vector<float>& prompt_feat,
                               const std::vector<float>& embedding, std::string* err) {
    (void)prompt_tokens;
    (void)prompt_feat;
    (void)embedding;
    (void)err;
    return 0;
  }
  // The 5,000,000 + seed noise extension: called once when a row's LM steps reach its
  // budget without finishing, so frames past the budget get their own stream (decoder RenderService::extend_noise).
  virtual int extend_noise(uint64_t row, std::string* err) {
    (void)row;
    (void)err;
    return 0;
  }
  // Never waits for the GPU (bounded queue: at most kMaxQueuedBatchesPerStream per stream).
  virtual int launch(RenderCall&& call, std::string* err) = 0;
  // Finished calls. wait: block until at least one finishes (or none is in flight).
  virtual void collect(std::vector<RenderOut>* out, bool wait) = 0;
  virtual int inflight() const = 0;
  virtual const char* name() const = 0;
  // Backend facts for kt2n_stats_json (a JSON object; "{}" when none).
  virtual std::string stats_json() const { return "{}"; }
};

// CPU stub: a quiet token-keyed tone with the reference sample accounting (960 samples per
// committed token; finals publish the rest); simulated GPU time a_us + b_us x tokens per call on
// one worker per stream role. Plumbing and timing-port tests only: NOT the model's audio.
struct StubRenderConfig {
  int call_us = 2000;
  double token_us = 5.0;
  float amplitude = 0.25f;
  int fail_every = 0;  // tests: every n-th call returns a failure (0 = never)
  int max_open_rows = 0;  // tests: open_row back-pressures beyond this many open rows (0 = unlimited)
  bool prefix_waits_for_rows = false;  // tests: prepare_prefix back-pressures while any row is open
  bool voice_required = false;  // voice cache on: a row or call without a voice (a given-back id is -1) is a violation
};
std::unique_ptr<RenderBackend> make_stub_render(const StubRenderConfig& cfg);

// The decoder's RenderService with its stand-in stages (noise gather + stub PCM): the real render state and call
// path with placeholder audio (a test configuration; built only with the decoder's CUDA target).
std::unique_ptr<RenderBackend> make_l7_standin_render(int device, int slots, int pool_pages, int voices,
                                                      std::string* err);

// The real renderer (the decoder's Renderer over s3gen.kt2n: encoder + estimator + HiFT). nullptr + err when it
// cannot be built (or the library was built without the decoder's flow target).
struct L7RenderOptions {
  std::string s3gen_path;
  int device = 0;
  int slots = 16;             // render-state slots (rows open at once)
  int pool_pages = 256;       // 64-frame K/V pages, 16.5 MB each
  int voices = 64;            // voice table rows (catalogue + loaded / built voices)
  int max_rows = 16;          // largest rows bucket a call may use
  int workspace_mb = 1024;    // per stream role (urgent, bulk)
  int hift_max_frame_rows = 0;  // 0 = the decoder's default (2048; ~600 KB of HiFT workspace per frame-row)
  int hift_max_t = 0;           // S_urgent HiFT handle's widest row (frames); 0 = the renderer's default (1024)
  int hift_bulk_max_t = 0;           // S_bulk HiFT handle (finals + whole renders); 0 = the decoder's default (4096 frames,
  int hift_bulk_max_frame_rows = 0;  // ~3.1 GB, one 2,045-token stream:false row per launch). Lab: lower it.
  // Graph prewarm (decoder RenderService::prewarm_voice): capture every flow graph a voice's rows can hit, up to
  // prewarm_rows rows per call and prewarm_tokens tokens per row, at create, so no live call captures a graph on
  // the conductor thread (2.7-5.8 ms each). With lazy prefixes only the first voice is built + prewarmed at
  // create (voices of the same prompt geometry share the keys); with all prefixes every voice is.
  bool prewarm = true;
  int prewarm_rows = 0;      // 0 = min(max_rows, 20)
  int prewarm_tokens = 320;
  bool lazy_prefix = true;    // build a voice's prefix on its first use (as the reference does) instead of at create
  bool reserve_pages = true;  // admission reserves a row's pool pages (back-pressure instead of no_pages)
  bool render_graphs = true;  // false: the decoder launches its kernels eagerly (no CUDA graph memory; same results)
  int bulk_ws_row_tokens = 0;   // > 0: S_bulk's workspace bound (render::RendererConfig::bulk_ws_row_tokens)
  int bulk_ws_demo_tokens = 0;
  bool timing = true;         // per-call GPU timing events (s3gen_gpu_ms)
  // Low-memory sizing (memory_profile LOW in engine create; all off by default; none changes what a call renders):
  int urgent_ws_prompt_tokens = 0;  // > 0: S_urgent's workspace sized for voices of up to this many prompt tokens
                                    // (render::RendererConfig); add_voice refuses a longer prompt
  bool urgent_ws_long_rows = false;  // ... and for mixed commit groups of long rows (render::RendererConfig)
  bool hift_share = false;          // one HiFT handle for S_urgent and S_bulk (render::RendererConfig::hift_share)
  bool free_pe_tables = false;      // free the encoder's load-time sinusoid tables
  bool pos_tables_from_pool = false;  // encoder position tables for the longest row the pool holds
                                      // (render::RendererConfig::pos_tables_from_pool)
  bool alias_workspaces = false;      // one buffer for both flow workspaces and the shared HiFT handle's scratch
                                      // (render::RendererConfig::alias_workspaces)
  bool pcm_from_pool = false;         // size the per-call PCM staging for the frames the pool can hold
  int hift_max_b = 0;                 // > 0: HiFT rows per launch (render::RendererConfig::hift_max_b); 0 = default
  // A complete-file row (which renders from position 0 and never reads a voice prefix) does not build its voice's
  // prefix, and unbuilds it when it needs those pool pages; the voice's next streaming row rebuilds it (the same pages'
  // contents). The low-memory fit then sizes the pool for max(prefix + streaming row, complete-file row).
  bool demo_gives_back_prefix = false;
  bool fitted = false;  // the low-memory fit shrank the sizes: the stats JSON reports its counters (else unchanged)
  // Evict idle prefixes for a first-use prefix or a row's reservation even while other rows hold reservations (idle
  // prefix pages are never reserved pages: avail_pages subtracts both). The low-memory profile, whose pool holds one
  // prefix plus the rows' pages; otherwise only when nothing is reserved.
  bool evict_while_reserved = false;
  // > 0: voice prefixes hold at most this many pool pages, so the rest stays the rows' share (default profile, cache
  // mode ALL: the pool minus 40 pages per stream). A prefix past it is left to the voice's first use at registration,
  // and a first use evicts idle prefixes (least recently used) to stay under it, also while rows hold reservations.
  int prefix_cap_pages = 0;
  bool prewarm_geometries = false;  // prewarm the graphs of every voice prompt geometry, not only the first voice's:
                                    // at add_voice with the prefix built (and given back) when nothing is in flight,
                                    // else when the prefix is first built (never with it unbuilt)
  bool sync_prefix_build = false;   // a first-use prefix build finishes on the GPU before open_row returns, so the
                                    // row's LM work starts as for a voice whose prefix was built at create
  bool staging_from_rows = false;   // size the per-call token / page-table / mel staging for max_rows rows of at most
                                    // render::kMaxRowTokens tokens each instead of 64 rows (never above the defaults)
  // The engine's voice cache decides which voices keep prefix pages: no prefix is built or prewarmed in add_voice,
  // open_row never drops idle prefixes to make room, and the engine builds prefixes with prepare_prefix().
  bool engine_managed_prefixes = false;
};
std::unique_ptr<RenderBackend> make_l7_render(const L7RenderOptions& opts, std::string* err);

}  // namespace kt2n::engine
