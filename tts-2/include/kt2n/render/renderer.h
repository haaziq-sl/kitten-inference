// Decoder Renderer: the whole token -> PCM renderer from one file (s3gen.kt2n), for the conductor (engine).
//   Renderer::create(s3gen.kt2n) loads the encoder (encoder layout, fp32), the estimator (estimator loader) and HiFT (HiFT
//   319-tensor list), and builds a RenderService whose calls run: [encoder + decoder inputs + estimator] as one
//   captured graph per bucket, then HiFT per equal-width group, then one PCM copy.
//   add_voice(prompt tokens, prompt mel, raw x-vector) computes the speaker conditioning on the GPU with the encoder's
//   spk_affine (spk_embed_affine_layer(l2normalize(x))) and builds the voice prefix (shared pages) once.
// All other calls are RenderService's (include/kt2n/render/service.h) = engine's RenderBackend.
// CUDA builds only (src/render/renderer.cu).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "kt2n/render/hift_stage.h"
#include "kt2n/render/program.h"
#include "kt2n/render/service.h"

namespace kt2n::render {

struct RendererConfig {
  RenderServiceConfig service;
  FlowProgramConfig flow;
  bool hift = true;
  int32_t hift_max_b = 32;
  int32_t hift_max_t = 1024;            // S_urgent HiFT handle: frames of the widest row (commits 0-1 need <= 34)
  int32_t hift_max_frame_rows = 2048;   // S_urgent: B x T bound per launch (HiFT: ~750 KB of workspace per frame-row)
  // S_bulk HiFT handle (finals + whole renders): one stream:false row must fit in one launch. 4096 frames covers a
  // 2,045-token row (DEMO budget 1,000 x 2 for the noise extension + 3 + 3 silence = 2,006 tokens = 4,012 frames);
  // ~3.1 GB. Longer whole renders fail their call with an error (never a crash).
  int32_t hift_bulk_max_t = 4096;
  int32_t hift_bulk_max_frame_rows = 4096;
  // Work policy for the S_bulk handle: convs from 16 x 1024
  // outputs on the tensor cores instead of from 128 x 1024. Bulk commits (18-58 frames, B1-B2, what N=20 runs) take
  // 18-36 % less GPU when calls share the GPU and are as fast or faster alone; correctness within the reference own
  // TF32 error (oracle 52-65 dB per call). The urgent handles keep HiFT's latency default (a first commit is 0.55 ms
  // instead of 0.78 alone). OPT-IN (an engine A/B at N=10/20 decides; it changes bulk PCM within the reference
  // TF32 error, so the quality gates see it too): -1 = HiFT's default on S_bulk (the earlier behaviour); KT2N_L7_HIFT_BULK_TC_MIN=16.
  int32_t hift_bulk_tc_min = -1;
  // S_aux HiFT handle (only with service.caller.aux_max_rows > 0 / KT2N_RENDER_AUX_ROWS): commits only (a 50-token
  // commit with its 8 cached frames is 108 frames wide); ~190 MB at 256 frame-rows.
  int32_t hift_aux_max_t = 128;
  int32_t hift_aux_max_frame_rows = 256;
  bool hift_warm = true;                // run HiFT once per shape class at create (lazy module loading off the live path)
  int32_t hift_prewarm_rows = 20;       // HiFT graphs captured at create (hift_stage.h prewarm); 0 = none
  bool build_prefixes = true;
  // With build_prefixes: add_voice also captures the voice's flow graphs ahead (RenderService::prewarm_voice)
  // for rows buckets <= prewarm_rows and rows up to prewarm_tokens tokens. 0 = off.
  int32_t prewarm_rows = 20;
  int32_t prewarm_tokens = 320;           // voice prefix at add_voice (reference default: built on first use, per-voice prompt noise)
  // Low-memory sizing (the engine's memory_profile LOW; all off by default). None of them changes what a call renders.
  //   urgent_ws_prompt_tokens > 0: S_urgent's workspace is the largest need of any call S_urgent can get (commits 0-1,
  //     previews and voice-prefix builds of up to this many rows, for voices of up to this many prompt tokens) instead
  //     of service.caller.workspace_bytes. Finals and whole renders run on S_bulk, which keeps its full workspace (its
  //     size decides how a stream:false batch is split, so it must not change).
  //   hift_share: S_urgent and S_bulk use ONE HiFT handle (the larger of the two shapes), their HiFT runs ordered one
  //     after the other on the GPU. Rows are still grouped per role exactly as with two handles. Only when no
  //     experiment needs separate handles (bulk SM partition, pipelined post stage, a bulk tensor-core policy).
  //   free_pe_tables: free the encoder's sinusoid tables once the position tables are built (load-time input only).
  //   urgent_ws_long_rows (with urgent_ws_prompt_tokens): a commit group that mixes a commit 0-1 row with rows deep
  //     into a long segment (several rows decoding at once) also goes to S_urgent, so the bound also covers commit
  //     calls whose keys reach the longest row (kMaxRowTokens past the prompt).
  int32_t urgent_ws_prompt_tokens = 0;
  int32_t urgent_ws_rows = 0;
  //   bulk_ws_row_tokens > 0 (with urgent_ws_prompt_tokens: the longest prompt P): S_bulk's workspace is the largest
  //     need of the calls S_bulk gets on a fitted low-memory engine instead of service.caller.workspace_bytes:
  //     one-row whole renders of up to P + bulk_ws_demo_tokens + 6 tokens, and steady commits / finals of
  //     commit-sized suffixes with keys up to P + bulk_ws_row_tokens + look-ahead + 80 tokens. A multi-row whole
  //     render above it is split by rows (execution only); a call above it fails cleanly.
  int32_t bulk_ws_row_tokens = 0;
  int32_t bulk_ws_demo_tokens = 0;
  bool urgent_ws_long_rows = false;
  bool hift_share = false;
  bool free_pe_tables = false;
  //   pos_tables_from_pool: the encoder's relative-position tables cover rows of up to kPageTokens x pool_pages tokens
  //     instead of kMaxRowTokens. Every key of a row lives in pool pages (its own, a voice prefix's shared ones, or a
  //     preview's scratch pages), so no row is longer than that, and a row's distances stay inside the table. A
  //     table row depends on its distance only (host_pe_table, and the position-table GEMM computes each output row
  //     on its own), so every row read is the same as in the full table.
  bool pos_tables_from_pool = false;
  //   alias_workspaces (with hift_share, eager stages): S_urgent's and S_bulk's flow workspaces and the shared HiFT
  //     handle's run scratch are one device buffer of the largest of the three (RenderCallerConfig::
  //     shared_workspace_bytes): calls of both roles run one after the other on the GPU, each flow on a zeroed
  //     workspace. Every stage keeps its own sizes and layout, so the same bytes are rendered.
  bool alias_workspaces = false;
};

class Renderer {
 public:
  static std::unique_ptr<Renderer> create(const std::string& s3gen_path, const RendererConfig& cfg, std::string* err);
  ~Renderer();
  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;

  // prompt_feat [frames][80] (frames = 2 x tokens), embedding = the raw speaker x-vector [192].
  int32_t add_voice(const std::vector<int32_t>& prompt_tokens, const std::vector<float>& prompt_feat,
                    const std::vector<float>& embedding, std::string* err);
  // RenderService::remove_voice (the renderer keeps no other per-voice state).
  int32_t remove_voice(int32_t voice, std::string* err);
  RenderService& service();
  // HiFT graph-mode counters (hift_stage.h); zeros without HiFT.
  HiftStage::GraphStats hift_graph_stats() const;
  // Totals of the add_voice prewarms (keys seen, graphs captured, host ms).
  RenderService::PrewarmStats prewarm_stats() const;

  struct Impl;

 private:
  Renderer();
  std::unique_ptr<Impl> impl_;
};

}  // namespace kt2n::render
