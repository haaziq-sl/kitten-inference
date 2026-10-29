// Decoder Renderer (include/kt2n/render/renderer.h).
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>

#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.h"
#include "kt2n/platform.h"
#include "kt2n/render/hift_stage.h"
#include "kt2n/render/renderer.h"

namespace kt2n::render {
namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct F32 {
  std::vector<int64_t> shape;
  const float* data = nullptr;
  size_t numel = 0;
};

// The encoder group of s3gen.kt2n (fp32 copies under the reference names), mmapped.
struct EncFile {
  const uint8_t* map = nullptr;
  uint64_t size = 0;
  std::map<std::string, F32> t;
  explicit EncFile(const std::string& path) {
    const kt2n::platform::MapResult r = kt2n::platform::map_file(path, &map, &size);
    if (r == kt2n::platform::MapResult::kOpenFailed) throw std::runtime_error("cannot open " + path);
    if (r != kt2n::platform::MapResult::kOk || size == 0) throw std::runtime_error("mmap failed");  // as mmap(0)
    const auto* base = map;
    if (size < 64 || std::memcmp(base, "KT2NS3G1", 8) != 0) throw std::runtime_error("not an s3gen.kt2n file");
    uint64_t meta_off, meta_len, data_off;
    std::memcpy(&meta_off, base + 16, 8);
    std::memcpy(&meta_len, base + 24, 8);
    std::memcpy(&data_off, base + 32, 8);
    const json::Value meta = json::parse(std::string_view(reinterpret_cast<const char*>(base + meta_off), meta_len));
    for (const auto& e : meta.get("tensors")->items()) {
      if (e.get("group")->as_string() != "encoder") continue;
      if (e.get("dtype")->as_string() != "f32") throw std::runtime_error("encoder tensor not f32");
      const json::Value* d = e.get("data");
      F32 x;
      x.numel = 1;
      for (const auto& s : d->get("shape")->items()) {
        x.shape.push_back(s.as_int());
        x.numel *= size_t(s.as_int());
      }
      const uint64_t off = uint64_t(d->get("off")->as_int()), len = uint64_t(d->get("len")->as_int());
      if (len != x.numel * 4 || data_off + off + len > size) throw std::runtime_error("bad encoder segment");
      x.data = reinterpret_cast<const float*>(base + data_off + off);
      t[e.get("name")->as_string()] = x;
    }
  }
  ~EncFile() {
    kt2n::platform::unmap_file(map, size);
  }
  const F32& get(const std::string& n) const {
    auto it = t.find(n);
    if (it == t.end()) throw std::runtime_error("missing encoder tensor " + n);
    return it->second;
  }
};

}  // namespace

struct Renderer::Impl {
  std::vector<void*> allocs;
  enc::Weights enc_w{};
  std::unique_ptr<est::Estimator> est;
  std::unique_ptr<FlowProgram> program;
  std::unique_ptr<HiftStage> hift;
  RenderService::PrewarmStats prewarm;
  std::unique_ptr<RenderService> service;
  RendererConfig cfg;
  // add_voice's speaker affine: buffers allocated once at create and a private non-blocking stream, so a voice
  // registered while serving (clone, saved voice) waits only for its own tiny kernel (cudaMalloc / cudaFree /
  // cudaDeviceSynchronize / legacy-stream copies would wait for every queued LM step and render).
  float* d_emb = nullptr;
  float* d_spk = nullptr;
  cudaStream_t voice_stream = nullptr;

  float* up(const float* p, size_t n) {
    float* d = nullptr;
    check(cudaMalloc(&d, n * sizeof(float)), "encoder weight");
    check(cudaMemcpy(d, p, n * sizeof(float), cudaMemcpyHostToDevice), "encoder weight");
    allocs.push_back(d);
    return d;
  }
  float* up(const F32& x) { return up(x.data, x.numel); }
  float* up_conv(const F32& x) {  // [O][C][K] -> tap-major [O][K][C] (encoder layout)
    const int64_t O = x.shape[0], C = x.shape[1], K = x.shape[2];
    std::vector<float> r(x.numel);
    for (int64_t o = 0; o < O; ++o)
      for (int64_t c = 0; c < C; ++c)
        for (int64_t k = 0; k < K; ++k) r[size_t((o * K + k) * C + c)] = x.data[size_t((o * C + c) * K + k)];
    return up(r.data(), r.size());
  }

  // encoder's weight layout, from s3gen.kt2n instead of safetensors.
  void load_encoder(const EncFile& f, int maxd_tok, int maxd_fr, bool free_pe) {
    cudaStream_t s = nullptr;
    const std::string P = "flow.encoder.";
    enc::Weights& w = enc_w;
    w.emb_table = up(f.get("flow.input_embedding.weight"));
    w.embed_w = up(f.get(P + "embed.out.0.weight"));
    w.embed_b = up(f.get(P + "embed.out.0.bias"));
    w.embed_ln_g = up(f.get(P + "embed.out.1.weight"));
    w.embed_ln_b = up(f.get(P + "embed.out.1.bias"));
    w.pre1_w = up_conv(f.get(P + "pre_lookahead_layer.conv1.weight"));
    w.pre1_b = up(f.get(P + "pre_lookahead_layer.conv1.bias"));
    w.pre2_w = up_conv(f.get(P + "pre_lookahead_layer.conv2.weight"));
    w.pre2_b = up(f.get(P + "pre_lookahead_layer.conv2.bias"));
    w.up_w = up_conv(f.get(P + "up_layer.conv.weight"));
    w.up_b = up(f.get(P + "up_layer.conv.bias"));
    w.up_embed_w = up(f.get(P + "up_embed.out.0.weight"));
    w.up_embed_b = up(f.get(P + "up_embed.out.0.bias"));
    w.up_ln_g = up(f.get(P + "up_embed.out.1.weight"));
    w.up_ln_b = up(f.get(P + "up_embed.out.1.bias"));
    w.after_g = up(f.get(P + "after_norm.weight"));
    w.after_b = up(f.get(P + "after_norm.bias"));
    w.proj_w = up(f.get("flow.encoder_proj.weight"));
    w.proj_b = up(f.get("flow.encoder_proj.bias"));
    w.spk_w = up(f.get("flow.spk_embed_affine_layer.weight"));
    w.spk_b = up(f.get("flow.spk_embed_affine_layer.bias"));
    w.maxd_tok = maxd_tok;
    w.maxd_fr = maxd_fr;
    const std::vector<float> pt = enc::host_pe_table(maxd_tok), pf = enc::host_pe_table(maxd_fr);
    float* pe_tok = up(pt.data(), pt.size());
    float* pe_fr = up(pf.data(), pf.size());
    auto layer = [&](const std::string& pre, enc::LayerW& L, int maxd, const float* pe) {
      L.ln_mha_g = up(f.get(pre + "norm_mha.weight"));
      L.ln_mha_b = up(f.get(pre + "norm_mha.bias"));
      std::vector<float> wqkv, bqkv;
      for (const char* n : {"linear_q", "linear_k", "linear_v"}) {
        const F32& wt = f.get(pre + "self_attn." + n + ".weight");
        const F32& bt = f.get(pre + "self_attn." + n + ".bias");
        wqkv.insert(wqkv.end(), wt.data, wt.data + wt.numel);
        bqkv.insert(bqkv.end(), bt.data, bt.data + bt.numel);
      }
      L.w_qkv = up(wqkv.data(), wqkv.size());
      L.b_qkv = up(bqkv.data(), bqkv.size());
      L.pos_u = up(f.get(pre + "self_attn.pos_bias_u"));
      L.pos_v = up(f.get(pre + "self_attn.pos_bias_v"));
      L.w_out = up(f.get(pre + "self_attn.linear_out.weight"));
      L.b_out = up(f.get(pre + "self_attn.linear_out.bias"));
      L.ln_ff_g = up(f.get(pre + "norm_ff.weight"));
      L.ln_ff_b = up(f.get(pre + "norm_ff.bias"));
      L.w1 = up(f.get(pre + "feed_forward.w_1.weight"));
      L.b1 = up(f.get(pre + "feed_forward.w_1.bias"));
      L.w2 = up(f.get(pre + "feed_forward.w_2.weight"));
      L.b2 = up(f.get(pre + "feed_forward.w_2.bias"));
      float* lp = up(f.get(pre + "self_attn.linear_pos.weight"));
      float* tab = nullptr;
      check(cudaMalloc(&tab, size_t(2 * maxd - 1) * kEncDim * sizeof(float)), "pos table");
      allocs.push_back(tab);
      if (enc::build_pos_table(pe, maxd, lp, tab, s) != 0) throw std::runtime_error("encoder pos table");
      L.ptab = tab;
    };
    for (int l = 0; l < kEnc1Layers; ++l) layer(P + "encoders." + std::to_string(l) + ".", w.enc1[l], maxd_tok, pe_tok);
    for (int l = 0; l < kEnc2Layers; ++l) layer(P + "up_encoders." + std::to_string(l) + ".", w.enc2[l], maxd_fr, pe_fr);
    check(cudaDeviceSynchronize(), "encoder load");
    if (free_pe) {
      // The sinusoid tables are only the position tables' input (nothing reads them after the builds above).
      for (float* p : {pe_tok, pe_fr}) {
        allocs.erase(std::find(allocs.begin(), allocs.end(), static_cast<void*>(p)));
        check(cudaFree(p), "encoder pe table");
      }
    }
  }
};

Renderer::Renderer() : impl_(new Impl) {}
Renderer::~Renderer() {
  if (impl_->voice_stream != nullptr) cudaStreamDestroy(impl_->voice_stream);
  impl_->service.reset();
  impl_->hift.reset();
  impl_->program.reset();
  for (void* p : impl_->allocs) cudaFree(p);
}

std::unique_ptr<Renderer> Renderer::create(const std::string& path, const RendererConfig& cfg, std::string* err) {
  std::unique_ptr<Renderer> r(new Renderer());
  Impl& m = *r->impl_;
  m.cfg = cfg;
  try {
    {
      EncFile f(path);
      // Low-memory profile (pos_tables_from_pool): tables for the longest row the pool can hold (renderer.h).
      const int32_t maxd_tok = cfg.pos_tables_from_pool
                                   ? std::max(kPageTokens, std::min(kMaxRowTokens, kPageTokens * cfg.service.arena.pool_pages))
                                   : kMaxRowTokens;
      if (cfg.pos_tables_from_pool && std::getenv("KT2N_DEBUG_MEM"))
        std::fprintf(stderr, "[mem] encoder position tables: rows of up to %d tokens (default %d)\n", maxd_tok,
                     kMaxRowTokens);
      m.load_encoder(f, maxd_tok, 2 * maxd_tok, cfg.free_pe_tables);
    }
    m.est = est::Estimator::load(path);
    FlowProgramConfig fpc = cfg.flow;
    // Diagnosis switch: KT2N_FLOW_PDL=0 captures the flow graphs without programmatic dependent launch.
    if (const char* pd = std::getenv("KT2N_FLOW_PDL")) fpc.pdl = pd[0] == '1';
    m.program = std::make_unique<FlowProgram>(m.enc_w, *m.est, fpc);
    check(cudaMalloc(&m.d_emb, kSpkDim * sizeof(float)), "spk");
    m.allocs.push_back(m.d_emb);
    check(cudaMalloc(&m.d_spk, kMel * sizeof(float)), "spk");
    m.allocs.push_back(m.d_spk);
    check(cudaStreamCreateWithFlags(&m.voice_stream, cudaStreamNonBlocking), "voice stream");
    RenderServiceConfig sc = cfg.service;
    sc.arena.est_tail_width = est::Estimator::tail_width();
    // A/B switch without a rebuild: KT2N_RENDER_PIPELINE=1 / 0 overrides RenderCallerConfig::pipeline_post.
    if (const char* pp = std::getenv("KT2N_RENDER_PIPELINE")) sc.caller.pipeline_post = pp[0] == '1';
    // Experiment: S_bulk in its own SM partition (RenderCallerConfig::bulk_sms); KT2N_RENDER_BULK_SMS=<n>.
    if (const char* bs = std::getenv("KT2N_RENDER_BULK_SMS")) sc.caller.bulk_sms = std::atoi(bs);
    // Experiment: a second urgent-priority render role (RenderCallerConfig::aux_max_rows); KT2N_RENDER_AUX_ROWS=<n>.
    if (const char* ax = std::getenv("KT2N_RENDER_AUX_ROWS")) sc.caller.aux_max_rows = std::atoi(ax);
    if (const char* aa = std::getenv("KT2N_RENDER_AUX_AUTO")) sc.aux_auto = aa[0] == '1';  // auto dispatch (default on)
    // Experiment: call merging (RenderServiceConfig::merge_*); rows cap per merged call on S_bulk / S_urgent, depth.
    if (const char* mb = std::getenv("KT2N_RENDER_MERGE_BULK")) sc.merge_bulk_rows = std::atoi(mb);
    if (const char* mu = std::getenv("KT2N_RENDER_MERGE_URGENT")) sc.merge_urgent_rows = std::atoi(mu);
    if (const char* md = std::getenv("KT2N_RENDER_MERGE_DEPTH")) sc.merge_depth = std::atoi(md);
    // Engine option: the conductor's KT2N_HOLD_URGENT sends commits a startup hold waits for to S_urgent; prewarm their twins.
    if (const char* hu = std::getenv("KT2N_HOLD_URGENT")) sc.hold_urgent_twins = std::atoi(hu) != 0;
    // Engine option (two-level priority): S_aux at a middle priority for low-buffer commits (KT2N_BUFFER_ROLE=aux).
    if (const char* ap = std::getenv("KT2N_RENDER_AUX_PRIO_LEVEL")) sc.caller.aux_priority_level = std::atoi(ap);
    if (const char* aw = std::getenv("KT2N_RENDER_AUX_WS_MB")) sc.caller.aux_workspace_bytes = size_t(std::max(16, std::atoi(aw))) << 20;
    if (const char* br = std::getenv("KT2N_BUFFER_ROLE")) sc.aux_steady_twins = std::string(br) == "aux";
    if (sc.caller.aux_max_rows > 0) {
      // Measured: with the default 8 hardware work queues the process's ~60
      // streams share queues in order, S_aux's kernels waited behind bulk work and two first commits under load took
      // 2-3x LONGER; with 32 the same arm halves the slowest one. The variable is read when CUDA initialises, so the
      // engine must set it before its first CUDA call (as with CUDA_MODULE_LOADING).
      const char* mc = std::getenv("CUDA_DEVICE_MAX_CONNECTIONS");
      if (mc == nullptr || std::atoi(mc) < 32)
        std::fprintf(stderr, "[decoder] S_aux is on but CUDA_DEVICE_MAX_CONNECTIONS is %s (< 32): first commits on "
                             "S_aux can queue behind bulk work; set it to 32 before the first CUDA call\n",
                     mc ? mc : "unset (8)");
    }
    // Low-memory profile: S_urgent's workspace = the largest need of any call it can get. S_urgent takes previews,
    // first commits, steady commits 0-1 (stream_for) and voice-prefix builds; finals and whole renders go to S_bulk.
    // Bounds per row, for voices of up to P prompt tokens: a prefix build is a row from position 0 over at most
    // P + 3 tokens; a preview P + 8 tokens from position 0; a commit 0 / 1 row the prompt's tail after its prefix plus
    // the commit's tokens and look-ahead (<= P + 12). Every rows bucket up to the call cap, every suffix bucket up to
    // P + 128 tokens and every keys bucket up to 2 x (P + 256) frames is taken (the layout is a pure function of
    // these buckets), so the bound has margin for commit-schedule levers. Conductor experiment switches that send
    // other commits or finals to S_urgent keep the full workspace.
    const bool urgent_routing = std::getenv("KT2N_HOLD_URGENT") != nullptr || std::getenv("KT2N_BUFFER_URGENT_MS") != nullptr ||
                             std::getenv("KT2N_BUFFER_ROLE") != nullptr;
    if (cfg.urgent_ws_prompt_tokens > 0 && !urgent_routing) {
      const RenderStage st = m.program->stage();
      const int32_t P = cfg.urgent_ws_prompt_tokens;
      const int32_t rows = std::max(1, std::min(cfg.urgent_ws_rows > 0 ? cfg.urgent_ws_rows : sc.caller.max_rows,
                                                sc.caller.max_rows));
      const int32_t rb_max = bucket_of(kRowBuckets, rows);
      std::vector<int32_t> sbs(std::begin(kFirstSuffixBuckets), std::end(kFirstSuffixBuckets));
      sbs.insert(sbs.end(), std::begin(kSteadySuffixBuckets), std::end(kSteadySuffixBuckets));
      for (int32_t t = kWholeBucketTokens; t <= whole_bucket(P + 128); t += kWholeBucketTokens) sbs.push_back(whole_bucket(t));
      const int32_t kb_max = keys_bucket_for(kMelPerToken * (P + 256));
      // A mixed commit group (urgent_ws_long_rows): commit-phase suffix buckets with keys up to the longest row.
      const int32_t kb_long = cfg.urgent_ws_long_rows ? keys_bucket_for(kMelPerToken * (P + kMaxRowTokens + kLookahead)) : 0;
      uint64_t need = 0;
      for (int32_t rb : kRowBuckets) {
        if (rb > rb_max) break;
        for (int32_t sb : sbs)
          for (int32_t kb = kKeysBucketFrames; kb <= kb_max; kb += kKeysBucketFrames)
            for (Phase ph : {Phase::kSteadyCommit, Phase::kPreview}) {
              CallDesc h{};
              h.phase = uint16_t(ph);
              h.rows_bucket = rb;
              h.suffix_bucket = sb;
              h.reserved[0] = kb;
              need = std::max<uint64_t>(need, (st.workspace_bytes(st.ctx, h) + 255) / 256 * 256);
            }
        for (int32_t sb : sbs) {
          const bool commit_sb = std::find(std::begin(kFirstSuffixBuckets), std::end(kFirstSuffixBuckets), sb) !=
                                     std::end(kFirstSuffixBuckets) ||
                                 std::find(std::begin(kSteadySuffixBuckets), std::end(kSteadySuffixBuckets), sb) !=
                                     std::end(kSteadySuffixBuckets);
          if (!commit_sb) continue;
          for (int32_t kb = kb_max + kKeysBucketFrames; kb <= kb_long; kb += kKeysBucketFrames) {
            CallDesc h{};
            h.phase = uint16_t(Phase::kSteadyCommit);
            h.rows_bucket = rb;
            h.suffix_bucket = sb;
            h.reserved[0] = kb;
            need = std::max<uint64_t>(need, (st.workspace_bytes(st.ctx, h) + 255) / 256 * 256);
          }
        }
      }
      if (std::getenv("KT2N_DEBUG_MEM"))
        std::fprintf(stderr, "[mem] S_urgent workspace bound: P %d rows %d long_rows %d -> %llu MiB (default %zu MiB)\n", P,
                     rows, cfg.urgent_ws_long_rows ? 1 : 0, (unsigned long long)((need + (1u << 20) - 1) >> 20),
                     sc.caller.workspace_bytes >> 20);
      const size_t ws = size_t((need + (1u << 20) - 1) >> 20) << 20;
      if (ws > 0 && ws < sc.caller.workspace_bytes) sc.caller.urgent_workspace_bytes = ws;
    }
    if (cfg.bulk_ws_row_tokens > 0 && cfg.urgent_ws_prompt_tokens > 0 && !urgent_routing) {
      const RenderStage st = m.program->stage();
      const int32_t P = cfg.urgent_ws_prompt_tokens;
      const int32_t rb_max = bucket_of(kRowBuckets, std::max(1, sc.caller.max_rows));
      uint64_t need = 0;
      auto take = [&](Phase ph, int32_t rb, int32_t sb, int32_t kb) {
        CallDesc h{};
        h.phase = uint16_t(ph);
        h.rows_bucket = rb;
        h.suffix_bucket = sb;
        h.reserved[0] = kb;
        need = std::max<uint64_t>(need, (st.workspace_bytes(st.ctx, h) + 255) / 256 * 256);
      };
      // Complete-file rows: one row from position 0 over the prompt, its DEMO budget, 3 look-ahead and 3 silence.
      const int32_t whole = whole_bucket(P + cfg.bulk_ws_demo_tokens + 6);
      for (int32_t sb = kWholeBucketTokens; sb <= whole; sb += kWholeBucketTokens)
        take(Phase::kWholeRender, 1, sb, keys_bucket_for(kMelPerToken * sb));
      // Streaming commits and finals on S_bulk: commit-sized suffixes, keys up to the longest fitted row.
      const int32_t kb_max = keys_bucket_for(kMelPerToken * (P + cfg.bulk_ws_row_tokens + kLookahead + 80));
      for (int32_t rb : kRowBuckets) {
        if (rb > rb_max) break;
        for (int32_t sb : kSteadySuffixBuckets)
          for (int32_t kb = kKeysBucketFrames; kb <= kb_max; kb += kKeysBucketFrames) {
            take(Phase::kSteadyCommit, rb, sb, kb);
            take(Phase::kFinalFlush, rb, sb, kb);
          }
      }
      const size_t ws = size_t((need + (1u << 20) - 1) >> 20) << 20;
      if (std::getenv("KT2N_DEBUG_MEM"))
        std::fprintf(stderr, "[mem] S_bulk workspace bound: P %d demo %d row %d rows %d -> %zu MiB (configured %zu MiB)\n", P,
                     cfg.bulk_ws_demo_tokens, cfg.bulk_ws_row_tokens, rb_max, ws >> 20, sc.caller.workspace_bytes >> 20);
      if (ws > 0) sc.caller.workspace_bytes = ws;
    }
    int32_t bulk_tc = cfg.hift_bulk_tc_min;
    if (const char* bt = std::getenv("KT2N_L7_HIFT_BULK_TC_MIN")) bulk_tc = std::atoi(bt);
    // One handle for S_urgent and S_bulk (low-memory profile) only when both would get the same policy and
    // context and the post stage is not pipelined (hift_stage.h).
    const bool hift_shared =
        cfg.hift && cfg.hift_share && bulk_tc == -1 && sc.caller.bulk_sms <= 0 && !sc.caller.pipeline_post;
    // alias_workspaces (renderer.h): S_urgent's and S_bulk's flow workspaces and the shared HiFT handle's scratch are
    // one buffer, used by one call at a time.
    if (cfg.alias_workspaces && hift_shared && !sc.caller.use_graphs && sc.caller.aux_max_rows <= 0) {
      const size_t urgent = sc.caller.urgent_workspace_bytes > 0 ? sc.caller.urgent_workspace_bytes : sc.caller.workspace_bytes;
      const size_t hb = HiftStage::shared_scratch_bytes(cfg.hift_max_b, HiftRoleSize{cfg.hift_max_t, cfg.hift_max_frame_rows},
                                                        HiftRoleSize{cfg.hift_bulk_max_t, cfg.hift_bulk_max_frame_rows});
      sc.caller.shared_workspace_bytes = std::max({urgent, sc.caller.workspace_bytes, hb});
      if (std::getenv("KT2N_DEBUG_MEM"))
        std::fprintf(stderr, "[mem] shared workspace: S_urgent %zu MiB, S_bulk %zu MiB, HiFT scratch %zu MiB -> %zu MiB\n",
                     urgent >> 20, sc.caller.workspace_bytes >> 20, hb >> 20, sc.caller.shared_workspace_bytes >> 20);
    }
    m.service = std::make_unique<RenderService>(sc, std::vector<RenderStage>{m.program->stage()});
    if (cfg.hift) {
      std::string e;
      m.hift = HiftStage::load(path, sc.arena.slots, cfg.hift_max_b, HiftRoleSize{cfg.hift_max_t, cfg.hift_max_frame_rows},
                               HiftRoleSize{cfg.hift_bulk_max_t, cfg.hift_bulk_max_frame_rows, bulk_tc}, &e,
                               m.service->caller().role_context(StreamRole::kSBulk),
                               sc.caller.aux_max_rows > 0 ? HiftRoleSize{cfg.hift_aux_max_t, cfg.hift_aux_max_frame_rows}
                                                          : HiftRoleSize{0, 0},
                               hift_shared, m.service->caller().shared_workspace(),
                               m.service->caller().shared_workspace_bytes());
      if (!m.hift) throw std::runtime_error("HiFT: " + e);
      m.service->caller().set_post_stage(m.hift->post_stage());
      if (cfg.hift_prewarm_rows > 0) m.hift->prewarm(m.service->caller(), cfg.hift_prewarm_rows);
      if (cfg.hift_warm) m.hift->warm(m.service->caller());
    }
    {
      // One line with the execution settings that came from the environment, so a server log shows which render
      // configuration actually ran (the release-candidate settings are execution-only: a missing variable would otherwise be silent).
      const char* mc = std::getenv("CUDA_DEVICE_MAX_CONNECTIONS");
      const char* bt = std::getenv("KT2N_L7_HIFT_BULK_TC_MIN");
      std::fprintf(stderr,
                   "[decoder] config: CUDA_DEVICE_MAX_CONNECTIONS=%s aux_rows=%d aux_auto=%d merge_bulk=%d "
                   "merge_urgent=%d merge_depth=%d bulk_sms=%d pipeline_post=%d hift_bulk_tc_min=%d workspace_mb=%zu "
                   "urgent_workspace_mb=%zu hift_shared=%d%s\n",
                   mc ? mc : "unset(8)", sc.caller.aux_max_rows, int(sc.aux_auto), sc.merge_bulk_rows, sc.merge_urgent_rows,
                   sc.merge_depth, sc.caller.bulk_sms, int(sc.caller.pipeline_post),
                   bt ? std::atoi(bt) : cfg.hift_bulk_tc_min, size_t(sc.caller.workspace_bytes >> 20),
                   size_t((sc.caller.urgent_workspace_bytes > 0 ? sc.caller.urgent_workspace_bytes : sc.caller.workspace_bytes) >> 20),
                   int(hift_shared),
                   sc.caller.shared_workspace_bytes > 0
                       ? (" shared_workspace_mb=" + std::to_string(sc.caller.shared_workspace_bytes >> 20)).c_str()
                       : "");
    }
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return nullptr;
  }
  return r;
}

int32_t Renderer::add_voice(const std::vector<int32_t>& prompt_tokens, const std::vector<float>& prompt_feat,
                            const std::vector<float>& embedding, std::string* err) {
  Impl& m = *impl_;
  if (embedding.size() != size_t(kSpkDim)) {
    if (err) *err = "add_voice: embedding must have 192 values";
    return -1;
  }
  std::vector<float> spk(kMel);
  try {
    // One voice at a time (the conductor's admin job); the buffers are reused.
    check(cudaMemcpyAsync(m.d_emb, embedding.data(), kSpkDim * sizeof(float), cudaMemcpyHostToDevice, m.voice_stream),
          "spk");
    const int rc = enc::spk_affine(m.enc_w, m.d_emb, 1, m.d_spk, m.voice_stream);
    if (rc != 0) throw std::runtime_error("spk_affine launch");
    check(cudaMemcpyAsync(spk.data(), m.d_spk, kMel * sizeof(float), cudaMemcpyDeviceToHost, m.voice_stream), "spk");
    check(cudaStreamSynchronize(m.voice_stream), "spk_affine");
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return -1;
  }
  const int32_t v = m.service->add_voice(prompt_tokens, prompt_feat, spk, err);
  if (v >= 0 && m.cfg.build_prefixes) {
    if (m.service->build_voice_prefix(v, err) != 0) return -1;
    if (m.cfg.prewarm_rows > 0) {
      RenderService::PrewarmStats ps;
      if (m.service->prewarm_voice(v, m.cfg.prewarm_rows, m.cfg.prewarm_tokens, &ps, err) < 0) return -1;
      m.prewarm.keys += ps.keys;
      m.prewarm.captured += ps.captured;
      m.prewarm.ms += ps.ms;
      m.prewarm.skipped += ps.skipped;
    }
  }
  return v;
}

int32_t Renderer::remove_voice(int32_t voice, std::string* err) { return impl_->service->remove_voice(voice, err); }
RenderService& Renderer::service() { return *impl_->service; }
RenderService::PrewarmStats Renderer::prewarm_stats() const { return impl_->prewarm; }
HiftStage::GraphStats Renderer::hift_graph_stats() const {
  return impl_->hift ? impl_->hift->graph_stats() : HiftStage::GraphStats{};
}

}  // namespace kt2n::render
