// The renderer behind the conductor's RenderBackend (kt2n::render::RenderService).
// Two arms:
//   "cuda"       the real renderer: kt2n::render::Renderer::create(s3gen.kt2n) = encoder + estimator (one
//                captured graph per bucket) + HiFT, add_voice(prompt tokens, prompt mel, raw x-vector) with
//                the encoder's GPU spk_affine. Voice prefixes are built on a voice's first use (the reference builds its
//                chunk-cache prefix on first use too) or all at create (KT2N_L7_PREFIXES=all).
//   "l7-standin" (lab) the decoder's stand-in stages (noise gather + stub PCM): the REAL render state with placeholder
//                audio. It tests the conductor <-> renderer path on the GPU; it is never a release configuration.
// Page reservation (admission back-pressure): a row reserves the pool pages its token budget can need before it
// is admitted (open_row returns kRenderBackpressure when the pool cannot cover it, and the conductor retries the
// job on a later turn), so a launch within the budget never finds the pool empty mid-request. A row that runs
// past its budget (the 5,000,000 + seed noise extension, up to 2 x budget + 3) reserves the extra pages when the
// extension starts; if the pool cannot cover them it continues on the pool's real free pages (counted as
// `extension_overcommit`). Reservations are released at close_row.
// Voice handles: the engine holds a stable handle per voice; the backend maps it to the decoder's table index.
// When a first-use prefix or a row's reservation can never fit (nothing reserved, so nothing will free pages),
// the backend evicts IDLE voice prefixes, least recently used first, instead of failing the request: the decoder
// remove_voice (refused with -2 while a row, call or prefix build still reads the voice) then add_voice again
// with the same prompt data, so the handle stays valid and its prefix is rebuilt on its next use (the reference
// builds its chunk-cache prefix on first use too). Before this, a pool smaller than the voices in use failed
// every later request (lab slot knobs: 61 of 72 requests at N=4 with 38 voices on a 112-page pool).
// The page counts and these rules live in RenderPageBook (render_pages.h, CPU tests). Default profile, cache mode ALL:
// prefixes are capped at the pool minus the rows' share (L7RenderOptions::prefix_cap_pages), an extension takes idle
// prefixes' pages before it overcommits, and a launch that finds no free page gives idle prefixes' pages back and
// retries before the request fails (a stress run with 130 registered clones at 8 streams: prefixes held 1,018 of 1,030
// pages, requests ran one at a time and an extending row failed with no_pages).
#include "build_flags.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <deque>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend.h"
#include "kt2n/render/service.h"
#include "render_pages.h"
#if KT2N_ENGINE_WITH_RENDERER
#include "kt2n/render/renderer.h"
#endif

namespace kt2n::engine {
namespace {

class L7Render final : public RenderBackend {
 public:
  // Stand-in arm: the service owns its stage list.
  L7Render(const render::RenderServiceConfig& cfg, std::vector<render::RenderStage> program, const char* name)
      : owned_(std::make_unique<render::RenderService>(cfg, std::move(program))), svc_(owned_.get()), name_(name),
        book_(svc_->pool().pages_total(), 0, false, false, nullptr) {}
#if KT2N_ENGINE_WITH_RENDERER
  // Real arm.
  L7Render(std::unique_ptr<render::Renderer> r, const L7RenderOptions& o)
      : renderer_(std::move(r)), svc_(&renderer_->service()), name_("cuda"), lazy_prefix_(o.lazy_prefix),
        reserve_(o.reserve_pages), prewarm_(o.prewarm),
        prewarm_rows_(o.prewarm_rows > 0 ? o.prewarm_rows : std::min(20, std::max(1, o.max_rows))),
        prewarm_tokens_(std::max(16, o.prewarm_tokens)), prewarm_geometries_(o.prewarm_geometries),
        max_prompt_tokens_(o.urgent_ws_prompt_tokens), managed_(o.engine_managed_prefixes),
        evict_while_reserved_(o.evict_while_reserved), demo_gives_back_prefix_(o.demo_gives_back_prefix), fitted_(o.fitted),
        book_(svc_->pool().pages_total(), o.prefix_cap_pages, o.evict_while_reserved, o.engine_managed_prefixes,
              [this](int voice) {
                const int rc = unbuild_prefix(voice);
                if (rc == 0) note_pool();
                return rc;
              }) {}
#endif

  int add_voice(const std::vector<int32_t>& prompt_tokens, const std::vector<float>& prompt_feat,
                const std::vector<float>& embedding, std::string* err) override {
    int l7v;
#if KT2N_ENGINE_WITH_RENDERER
    if (renderer_) {
      if (max_prompt_tokens_ > 0 && int(prompt_tokens.size()) > max_prompt_tokens_) {
        // The low-memory profile sized S_urgent's workspace for prompts up to this length.
        if (err)
          *err = "voice prompt has " + std::to_string(prompt_tokens.size()) + " tokens; the low-memory profile serves up to " +
                 std::to_string(max_prompt_tokens_);
        return -1;
      }
      // Raw 192-d x-vector: the renderer applies the encoder's spk_affine on the GPU.
      l7v = renderer_->add_voice(prompt_tokens, prompt_feat, embedding, err);
      if (l7v >= 0) {
        const int v = next_handle_++;
        l7_of_[v] = l7v;
        prompt_toks_[v] = int(prompt_tokens.size());
        vdata_[v] = VoiceData{prompt_tokens, prompt_feat, embedding};  // for a re-add after prefix eviction
        // Prefix (+ graph prewarm) at create: every voice with KT2N_L7_PREFIXES=all; else only the first voice,
        // whose prewarm captures the keys voices of the same prompt geometry share (engine runs still saw a few
        // live captures at N>=4 on the prewarmed voice itself).
        // Never prewarm on a lazy first use (that request would pay the captures).
        // Engine-managed prefixes (voice cache): the engine builds the prefix (prepare_prefix) when it loads the voice
        // and prewarms each prompt geometry at create (prewarm_geometry).
        bool now = !managed_ && (!lazy_prefix_ || (prewarm_ && !prewarm_tried_));
        // A pool too small for every prefix at create (lab knobs with KT2N_L7_PREFIXES=all): leave the rest lazy
        // (built on first use, with LRU eviction) instead of failing the voice; without this, requests failed on
        // voices whose create-time build did not fit.
        if (now && reserve_ && !book_.build_at_add(prefix_pages_bound(v))) {
          if (create_lazy_fallback_++ == 0)
            std::fprintf(stderr, "[decoder] render pool too small to build every voice prefix at create; the "
                                 "remaining voices build on first use\n");
          now = false;
        }
        if (now) {
          prewarm_tried_ = true;  // lazy mode: only the first voice pays at create, even when its prewarm fails
          if (build_prefix(v, err) != 0) return -1;
          // Prewarm is an optimisation: a failure (e.g. a key whose capture does not fit a lab-sized pool or
          // workspace) must not lose the voice. The voice stays usable; missing graphs are captured live.
          if (prewarm_) {
            std::string perr;
            if (prewarm(l7v, &perr) != 0) {
              if (prewarm_errors_++ == 0) std::fprintf(stderr, "[decoder] graph prewarm failed (voice kept): %s\n", perr.c_str());
              prewarm_error_ = perr;
            }
            prewarmed_geoms_.insert(int(prompt_tokens.size()));
          }
        } else if (prewarm_ && prewarm_geometries_ && !managed_ && prewarmed_geoms_.insert(int(prompt_tokens.size())).second) {
          // A prompt length no earlier voice had (low-memory profile; prefixes are built on first use): capture its
          // keys now (an unbuilt voice is enumerated with its prefix geometry), so no serving call of it captures a
          // graph (or allocates for one) while serving. Engine-managed prefixes: captured once the prefix is built
          // (prefix_ready), never with it unbuilt.
          std::string perr;
          if (prewarm(l7v, &perr) != 0) {
            if (prewarm_errors_++ == 0) std::fprintf(stderr, "[decoder] graph prewarm failed (voice kept): %s\n", perr.c_str());
            prewarm_error_ = perr;
          }
          // The prefix build call has a graph of its own, keyed by the prompt length: build the prefix once now to
          // capture it, then give the pages back (rebuilt on first use, as for every voice but the first). Only with
          // no row reserved and no render call in flight (at create), so the wait below collects nothing but the build.
          if (book_.reserved() == 0 && call_rows_.empty() && (!reserve_ || prefix_pages_bound(v) <= book_.avail())) {
            if (build_prefix(v, &perr) != 0) {
              if (prewarm_errors_++ == 0) std::fprintf(stderr, "[decoder] prefix build prewarm failed (voice kept): %s\n", perr.c_str());
              prewarm_error_ = perr;
            }
            if (book_.built(v)) give_back_prefix_now(v);
          }
        }
        return v;
      }
      return l7v;
    }
#endif
    // The stand-in program reads no speaker conditioning.
    l7v = svc_->add_voice(prompt_tokens, prompt_feat, embedding.size() == 80 ? embedding : std::vector<float>{}, err);
    if (l7v < 0) return l7v;
    const int v = next_handle_++;
    l7_of_[v] = l7v;
    return v;
  }
  int open_row(uint64_t row, int voice, uint64_t seed, int budget, bool demo, std::string* err) override {
    int need = 0;
#if KT2N_ENGINE_WITH_RENDERER
    if (renderer_) {
      if (l7(voice) < 0) {
        if (err)
          *err = voice < 0 ? std::string("voice has no render state (its render add_voice failed; see render_voice_errors)")
                           : "render voice handle " + std::to_string(voice) + " has no table row (lost in a prefix eviction re-add)";
        return -1;
      }
      book_.touch(voice);
      // demo_gives_back_prefix: a complete-file row never reads its voice's prefix (it renders from position 0), so it
      // neither builds one nor keeps one it needs the pages of (below); the voice's next streaming row rebuilds it.
      const bool skip_prefix = demo && demo_gives_back_prefix_;
      if (prompt_toks_.count(voice) && !book_.built(voice) && !skip_prefix) {
        // A first-use prefix takes pool pages for good: with reservations on it must not take pages that open
        // rows have reserved (they could then meet an empty pool mid-request). Wait while rows hold reservations;
        // when nothing is reserved, evict idle prefixes (LRU); refuse only if it still cannot fit.
        if (reserve_) {
          const int pneed = prefix_pages_bound(voice);
          // Evicting under reservations (low-memory profile): build only when this row fits right after, else the new
          // prefix would sit idle while the row waits and the next first use would evict it again (rebuild churn).
          // With the prefix unbuilt the row's page count shares nothing: an upper bound of the count once built.
          const bool under = book_.reserved() > 0 && evict_while_reserved_ && !managed_;
          const auto a = book_.admit_prefix(voice, pneed, under ? svc_->pages_needed(l7(voice), budget, demo) : 0);
          if (a != RenderPageBook::Admit::kOk) {
            if (a == RenderPageBook::Admit::kNever) {
              if (err) *err = "voice prefix needs up to " + std::to_string(pneed) + " pool pages, the pool has " +
                              std::to_string(book_.avail());
              return -1;
            }
            backpressure_++;
            return kRenderBackpressure;
          }
        }
        if (build_prefix(voice, err, true) != 0) return -1;
        prewarm_live(voice);
      }
      if (reserve_) {
        need = svc_->pages_needed(l7(voice), budget, demo);  // the decoder's exact count (gtest RowPagesNeededIsExact)
        auto a = book_.admit_row(voice, need);
        if (a != RenderPageBook::Admit::kOk && skip_prefix && book_.built(voice) && need > book_.avail()) {
          // The row's own voice prefix gives its pages back (unbuilt; the same pages' contents when rebuilt). Busy
          // (a row or call of the voice still reads it): wait, as for any reservation.
          if (unbuild_prefix(voice) != 0) {
            backpressure_++;
            return kRenderBackpressure;
          }
          note_pool();
          demo_prefix_givebacks_++;
          if (l7(voice) < 0) {
            if (err) *err = "render voice handle " + std::to_string(voice) + " has no table row (lost in a prefix re-add)";
            return -1;
          }
          a = book_.admit_row(voice, need);
        }
        if (a != RenderPageBook::Admit::kOk) {
          if (a == RenderPageBook::Admit::kNever) {  // nothing to wait for: this row can never fit
            if (err) *err = "open_row: budget needs " + std::to_string(need) + " pool pages, the pool has " +
                            std::to_string(book_.avail());
            return -1;
          }
          backpressure_++;
          return kRenderBackpressure;
        }
      }
    }
#endif
    const int rc = svc_->open_row(row, l7(voice), seed, budget, demo, err);
    if (rc == -2) return kRenderBackpressure;  // no render-state slot: wait, as for pages
    if (rc < 0) return -1;
    note_pool();
    if (need > 0) {
      book_.reserve(need);
      row_need_[row] = need;
      row_voice_budget_[row] = {voice, budget, demo};
    }
    return 0;
  }
  // The decoder frees a closed row's pages once that row's in-flight calls are collected (release_closed), so its
  // reservation is held until the row's own last call is collected. Keyed per row: collect() returns finished calls
  // in launch order per row only, not globally (an earlier count-based release assumed global order and could free a reservation a few ms early).
  // The decoder's RenderService::remove_voice: the voice's prefix pages leave the prefix count, and
  // its table index may be reused by a later add_voice, so the index's lazy-prefix state is reset.
  int remove_voice(int voice, std::string* err) override {
    const int l7v = l7(voice);
    if (l7v >= 0) {
      const int rc = svc_->remove_voice(l7v, err);
      if (rc != 0) return rc == -2 ? -2 : -1;
    } else if (!l7_of_.count(voice)) {
      if (err) *err = "unknown render voice handle";
      return -1;
    }
    book_.erase_voice(voice);
    l7_of_.erase(voice);
    prompt_toks_.erase(voice);
    vdata_.erase(voice);
    note_pool();
    voices_removed_++;
    return 0;
  }
  int prepare_prefix(int voice, std::string* err) override {
#if KT2N_ENGINE_WITH_RENDERER
    if (!renderer_) return 0;
    if (l7(voice) < 0) {
      if (err) *err = "render voice handle " + std::to_string(voice) + " has no table row";
      return -1;
    }
    if (!prompt_toks_.count(voice) || book_.built(voice)) return 0;
    // The prefix takes pool pages for good: never pages that open rows have reserved. The engine's cache sizes the
    // pool so its voices' prefixes and the rows' reservations both fit; when rows hold more than their share now,
    // the load waits for them.
    if (reserve_ && prefix_pages_bound(voice) > book_.avail()) {
      backpressure_++;
      return kRenderBackpressure;
    }
    return build_prefix(voice, err, true) == 0 ? 0 : -1;
#else
    (void)voice;
    (void)err;
    return 0;
#endif
  }
  int prefix_ready(int voice) override {
    const int l7v = l7(voice);
    if (l7v < 0) return -1;
    if (svc_->prefix_build_pending(l7v)) return 0;
#if KT2N_ENGINE_WITH_RENDERER
    if (renderer_ && managed_ && book_.built(voice)) prewarm_live(voice);
#endif
    return 1;
  }
  int prewarm_geometry(const std::vector<int32_t>& prompt_tokens, const std::vector<float>& prompt_feat,
                       const std::vector<float>& embedding, std::string* err) override {
#if KT2N_ENGINE_WITH_RENDERER
    if (!renderer_ || !prewarm_) return 0;
    const int l7v = renderer_->add_voice(prompt_tokens, prompt_feat, embedding, err);
    if (l7v < 0) return -1;
    int rc = 0;
    // The graph keys include the prefix geometry, so build the prefix first (outside the engine's page accounting:
    // the pages go back with the voice below), capture, then wait for the build and remove the voice.
    if (svc_->build_voice_prefix(l7v, err) != 0) {
      rc = -1;
    } else {
      std::string perr;
      if (prewarm(l7v, &perr) != 0) {
        if (prewarm_errors_++ == 0) std::fprintf(stderr, "[decoder] graph prewarm failed: %s\n", perr.c_str());
        prewarm_error_ = perr;
      }
      // This prompt length is captured: a voice of it added later (engine-managed prefixes) does not prewarm again.
      prewarmed_geoms_.insert(int(prompt_tokens.size()));
    }
    std::vector<render::ServiceOut> so;
    for (int i = 0; i < 100000; ++i) {  // busy (-2) only while its prefix build runs
      std::string rerr;
      const int r = svc_->remove_voice(l7v, &rerr);
      if (r != -2) {
        if (r != 0 && rc == 0) {
          if (err) *err = rerr;
          rc = -1;
        }
        break;
      }
      svc_->collect(&so, true);
    }
    note_pool();
    return rc;
#else
    (void)prompt_tokens;
    (void)prompt_feat;
    (void)embedding;
    (void)err;
    return 0;
#endif
  }
  void close_row(uint64_t row) override {
    svc_->close_row(row);
    note_pool();
    auto it = row_need_.find(row);
    if (it != row_need_.end()) {
      auto c = row_calls_.find(row);
      if (c == row_calls_.end() || c->second == 0)
        book_.release(it->second);
      else
        closed_pending_[row] += it->second;
      row_need_.erase(it);
    }
    row_voice_budget_.erase(row);
  }
  int extend_noise(uint64_t row, std::string* err) override {
#if KT2N_ENGINE_WITH_RENDERER
    auto it = row_need_.find(row);
    auto vb = row_voice_budget_.find(row);
    if (it != row_need_.end() && vb != row_voice_budget_.end()) {
      const int full = svc_->pages_needed(l7(vb->second.voice), 2 * vb->second.budget + 3, vb->second.demo);
      const int extra = std::max(0, full - it->second);
      book_.extend(vb->second.voice, extra);  // idle prefixes' pages first, else counted as an overcommit
      it->second += extra;
    }
#endif
    return svc_->extend_noise(row, err);
  }
  int launch(RenderCall&& call, std::string* err) override {
    render::ServiceCall sc;
    sc.call_id = call.call_id;
    sc.stream = call.stream;
    sc.rows.reserve(call.rows.size());
    for (auto& r : call.rows) {
      render::ServiceRowReq q;
      q.row = r.row;
      q.voice = l7(r.voice);
      q.phase = r.phase;
      q.commit_index = r.commit_index;
      q.committed_tokens = r.committed_tokens;
      q.commit_tokens = r.commit_tokens;
      q.finalize = r.finalize;
      q.tokens = std::move(r.tokens);
      q.seq = r.seq;
      sc.rows.push_back(std::move(q));
    }
    std::vector<uint64_t> rows;
    rows.reserve(sc.rows.size());
    for (const auto& q : sc.rows) rows.push_back(q.row);
    const uint64_t id = sc.call_id;
    // -2: no free page for the call; nothing was allocated and `sc` is intact (launch_impl reads it only), so idle
    // prefixes give their pages back, one at a time (least recently used), and the call is retried before the request
    // fails.
    int rc = svc_->launch(std::move(sc), err);
    while (rc == -2 && book_.evict_one_idle()) {
      launch_page_retries_++;
      note_pool();
      rc = svc_->launch(std::move(sc), err);
    }
    if (rc < 0) return -1;  // refused before it was queued: never collected
    for (uint64_t r : rows) row_calls_[r]++;
    call_rows_[id] = std::move(rows);
    return 0;
  }
  void collect(std::vector<RenderOut>* out, bool wait) override {
    std::vector<render::ServiceOut> so;
    svc_->collect(&so, wait);
#if KT2N_ENGINE_WITH_RENDERER
    if (fitted_ && renderer_ && !so.empty()) {  // conductor thread: mirror the vocoder's widest calls for the stats thread
      const auto g = renderer_->hift_graph_stats();
      vocoder_widest_urgent_ = g.widest[0];
      vocoder_widest_bulk_ = g.widest[1];
    }
#endif
    if (!so.empty()) {
      for (const auto& o : so) {
        auto cr = call_rows_.find(o.call_id);
        if (cr == call_rows_.end()) continue;
        for (uint64_t r : cr->second) {
          auto c = row_calls_.find(r);
          if (c == row_calls_.end() || --c->second > 0) continue;
          row_calls_.erase(c);
          auto pend = closed_pending_.find(r);
          if (pend != closed_pending_.end()) {
            book_.release(pend->second);
            closed_pending_.erase(pend);
          }
        }
        call_rows_.erase(cr);
      }
      note_pool();
    }
    for (auto& o : so) {
      if (o.graph_captured) live_captures_++;
      if (o.status != 0) failed_calls_++;
      RenderOut r;
      r.call_id = o.call_id;
      r.status = o.status;
      r.pcm = std::move(o.pcm);
      r.gpu_ms = o.gpu_ms;
      r.launch_ms = o.launch_ms;
      r.graph_nodes = o.graph_nodes;
      r.graph_captured = o.graph_captured;
      out->push_back(std::move(r));
    }
  }
  int inflight() const override { return svc_->inflight(); }
  const char* name() const override { return name_; }
  // Called from the stats thread: only atomics (the pool itself is the conductor's).
  std::string stats_json() const override {
    return "{\"pool_pages\":" + std::to_string(pool_total_.load()) +
           ",\"pool_free\":" + std::to_string(pool_free_.load()) + ",\"pool_free_min\":" + std::to_string(pool_free_min_.load()) +
           ",\"prefix_pages\":" + std::to_string(book_.prefix_pages()) + ",\"prefix_cap\":" +
           std::to_string(book_.cap()) + ",\"prefixes_built\":" +
           std::to_string(prefixes_built_.load()) + ",\"prefix_build_ms\":" + std::to_string(prefix_build_ms_.load()) +
           ",\"voices_removed\":" + std::to_string(voices_removed_.load()) +
           ",\"prefixes_evicted\":" + std::to_string(prefixes_evicted_.load()) +
           // The low-memory fit's counters only when it is active, so other engines' stats text is unchanged.
           (fitted_ ? ",\"demo_prefix_givebacks\":" + std::to_string(demo_prefix_givebacks_.load()) +
                          ",\"vocoder_widest_urgent\":" + std::to_string(vocoder_widest_urgent_.load()) +
                          ",\"vocoder_widest_bulk\":" + std::to_string(vocoder_widest_bulk_.load())
                    : std::string()) +
           ",\"readd_failures\":" + std::to_string(readd_failures_.load()) +
           ",\"create_lazy_fallback\":" + std::to_string(create_lazy_fallback_.load()) +
           ",\"reserved_pages\":" + std::to_string(book_.reserved()) +
           ",\"page_backpressure\":" + std::to_string(backpressure_.load()) +
           ",\"extension_overcommit\":" + std::to_string(book_.overcommits()) +
           ",\"launch_page_retries\":" + std::to_string(launch_page_retries_.load()) +
           ",\"prewarm\":{\"voices\":" + std::to_string(prewarmed_voices_.load()) + ",\"errors\":" +
           std::to_string(prewarm_errors_.load()) + ",\"keys\":" +
           std::to_string(prewarm_keys_.load()) + ",\"captured\":" + std::to_string(prewarm_captured_.load()) +
           ",\"ms\":" + std::to_string(prewarm_ms_.load()) + "}" + ",\"live_captures\":" +
           std::to_string(live_captures_.load()) + ",\"failed_calls\":" + std::to_string(failed_calls_.load()) + "}";
  }

 private:
#if KT2N_ENGINE_WITH_RENDERER
  // Upper bound on the pages a voice's prefix keeps (its complete 64-frame pages of 2 frames per prompt token).
  int prefix_pages_bound(int voice) const {
    auto it = prompt_toks_.find(voice);
    const int p = it != prompt_toks_.end() ? it->second : 0;
    return (2 * p + 63) / 64 + 1;
  }
  // Unbuild one idle voice's prefix: the decoder's remove_voice (refused while anything reads the voice), then add_voice with
  // the same prompt data under the same handle. 0 = its prefix pages are back in the pool.
  int unbuild_prefix(int voice, bool count = true) {
    auto d = vdata_.find(voice);
    const int l7v = l7(voice);
    if (d == vdata_.end() || l7v < 0) return -1;
    std::string err;
    if (svc_->remove_voice(l7v, &err) != 0) return -1;  // -2: a row, call or prefix build still reads it
    book_.forget(voice);
    const int nv = renderer_->add_voice(d->second.toks, d->second.feat, d->second.emb, &err);
    l7_of_[voice] = nv;  // -1: the handle is dead; its next open_row fails with a clear error
    if (nv < 0) {
      readd_failures_++;
      std::fprintf(stderr, "[decoder] re-add after prefix eviction failed (voice %d): %s\n", voice, err.c_str());
    }
    if (count) prefixes_evicted_++;
    return 0;
  }
  // Create time, nothing else in flight: wait for the voice's prefix build, then unbuild it (pages back to the pool).
  void give_back_prefix_now(int voice) {
    std::vector<render::ServiceOut> so;
    for (int i = 0; i < 100000 && l7(voice) >= 0 && svc_->prefix_build_pending(l7(voice)); ++i) svc_->collect(&so, true);
    unbuild_prefix(voice, false);
    note_pool();
  }
  // A prefix built while serving whose prompt length was never prewarmed (a clone added while rows were open, or an
  // engine-managed voice of a new prompt length): capture its keys now, before its rows, and count them as live.
  void prewarm_live(int voice) {
    auto it = prompt_toks_.find(voice);
    if (!prewarm_ || !prewarm_geometries_ || it == prompt_toks_.end() || !prewarmed_geoms_.insert(it->second).second)
      return;
    const long c0 = prewarm_captured_.load();
    std::string perr;
    if (prewarm(l7(voice), &perr) != 0) {
      if (prewarm_errors_++ == 0) std::fprintf(stderr, "[decoder] graph prewarm failed (voice kept): %s\n", perr.c_str());
      prewarm_error_ = perr;
    }
    live_captures_ += prewarm_captured_.load() - c0;
  }
  int prewarm(int voice, std::string* err) {
    render::RenderService::PrewarmStats st;
    if (svc_->prewarm_voice(voice, prewarm_rows_, prewarm_tokens_, &st, err) < 0) return -1;
    prewarmed_voices_++;
    prewarm_keys_ += st.keys;
    prewarm_captured_ += st.captured;
    prewarm_ms_ = prewarm_ms_.load() + st.ms;
    return 0;
  }
  // `live`: built while serving, so a graph its build call captures counts as a live capture.
  int build_prefix(int voice, std::string* err, bool live = false) {
    const int free0 = svc_->pool().pages_free();
    const auto t0 = std::chrono::steady_clock::now();
    const long cap0 = svc_->prefix_build_captures();
    if (svc_->build_voice_prefix(l7(voice), err) != 0) return -1;
    if (live) live_captures_ += svc_->prefix_build_captures() - cap0;
    prefix_build_ms_ = prefix_build_ms_.load() + std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const int took = std::max(0, free0 - svc_->pool().pages_free());
    book_.on_built(voice, took);
    note_pool();
    prefixes_built_++;
    return 0;
  }
  std::unique_ptr<render::Renderer> renderer_;
#endif
  std::unique_ptr<render::RenderService> owned_;
  render::RenderService* svc_ = nullptr;
  const char* name_;
  bool lazy_prefix_ = true;
  bool reserve_ = false;
  bool prewarm_ = false;
  int prewarm_rows_ = 20, prewarm_tokens_ = 320;
  bool prewarm_geometries_ = false;
  int max_prompt_tokens_ = 0;          // > 0: add_voice refuses longer prompts (low-memory S_urgent workspace)
  std::set<int> prewarmed_geoms_;      // prompt lengths whose graph keys were prewarmed
  bool managed_ = false;  // engine_managed_prefixes (the engine's voice cache owns prefix residency)
  bool evict_while_reserved_ = false;
  bool demo_gives_back_prefix_ = false;  // L7RenderOptions::demo_gives_back_prefix
  bool fitted_ = false;                  // L7RenderOptions::fitted
  RenderPageBook book_;  // pool pages: prefixes, reservations and the eviction rules (after svc_: it reads the pool)
  std::atomic<long> launch_page_retries_{0};  // launches retried after idle prefixes gave their pages back
  std::atomic<int> prewarmed_voices_{0};
  std::atomic<long> prewarm_errors_{0};
  bool prewarm_tried_ = false;
  std::string prewarm_error_;  // conductor / create thread only (stats print the count)
  std::atomic<long> prewarm_keys_{0}, prewarm_captured_{0};
  std::atomic<double> prewarm_ms_{0};
  std::atomic<long> live_captures_{0}, failed_calls_{0};
  // Voice handles (engine side) -> the decoder's table index; -1 = lost in a failed re-add.
  int l7(int voice) const {
    auto it = l7_of_.find(voice);
    return it != l7_of_.end() ? it->second : -1;
  }
  struct VoiceData {
    std::vector<int32_t> toks;
    std::vector<float> feat, emb;
  };
  int next_handle_ = 0;
  std::unordered_map<int, int> l7_of_;
  std::unordered_map<int, VoiceData> vdata_;   // real renderer only: the add_voice inputs, for a re-add
  std::atomic<long> prefixes_evicted_{0}, readd_failures_{0};
  std::atomic<long> demo_prefix_givebacks_{0};  // complete-file rows that unbuilt their voice's prefix for its pages
  std::atomic<int> vocoder_widest_urgent_{0}, vocoder_widest_bulk_{0};  // widest HiFT input (frames) served per role
  std::atomic<int> create_lazy_fallback_{0};  // voices left lazy at create because the pool could not hold their prefix
  std::unordered_map<int, int> prompt_toks_;
  std::unordered_map<uint64_t, int> row_need_;
  struct RowBudget {
    int voice = -1, budget = 0;
    bool demo = false;
  };
  std::unordered_map<uint64_t, RowBudget> row_voice_budget_;
  std::unordered_map<uint64_t, std::vector<uint64_t>> call_rows_;  // call id -> its rows (in flight)
  std::unordered_map<uint64_t, int> row_calls_;                    // row -> its calls in flight
  std::unordered_map<uint64_t, int> closed_pending_;               // closed row -> pages to release at its last call
  std::atomic<int> prefixes_built_{0};
  std::atomic<long> voices_removed_{0};
  std::atomic<double> prefix_build_ms_{0};
  std::atomic<long> backpressure_{0};
  std::atomic<int> pool_total_{0}, pool_free_{0}, pool_free_min_{1 << 30};
  void note_pool() {
    const int f = svc_->pool().pages_free();
    pool_total_ = svc_->pool().pages_total();
    pool_free_ = f;
    if (f < pool_free_min_.load()) pool_free_min_ = f;
  }
};

render::RenderServiceConfig service_config(const L7RenderOptions& o) {
  render::RenderServiceConfig cfg;
  cfg.arena.device = o.device;
  cfg.arena.slots = o.slots;
  cfg.arena.pool_pages = o.pool_pages;
  cfg.arena.voices = o.voices;
  cfg.caller.device = o.device;
  cfg.caller.max_rows = std::max(1, std::min(64, o.max_rows));
  cfg.caller.workspace_bytes = size_t(std::max(64, o.workspace_mb)) << 20;
  if (o.staging_from_rows) {
    // A call carries at most max_rows rows (its rows bucket, for the estimator's padded mel output) and a row at most
    // kMaxRowTokens tokens (build_call refuses longer rows), so these bounds refuse no call the defaults accept. The
    // PCM staging keeps its default (it is the smaller bound).
    const int32_t rows = render::bucket_of(render::kRowBuckets, cfg.caller.max_rows);
    const int32_t row_pages = (render::kMelPerToken * render::kMaxRowTokens + render::kPageFrames - 1) / render::kPageFrames + 1;
    cfg.caller.max_call_tokens = std::min(cfg.caller.max_call_tokens, rows * render::kMaxRowTokens);
    cfg.caller.max_call_pages = std::min(cfg.caller.max_call_pages, rows * row_pages);
  }
  if (o.pcm_from_pool) {
    // A call's PCM is at most its rows' mel cache (8 frames each) plus their new frames, and every new frame has
    // its K/V in a pool page (64 frames each), so (64 x pool + 8 x rows) frames of 480 samples bound every call.
    const int32_t rows = render::bucket_of(render::kRowBuckets, cfg.caller.max_rows);
    const int64_t frames = int64_t(render::kPageFrames) * o.pool_pages + int64_t(render::kMelCacheFrames) * rows;
    cfg.caller.max_call_pcm = int32_t(std::min<int64_t>(cfg.caller.max_call_pcm, frames * render::kSamplesPerFrame));
  }
  cfg.timing = o.timing;
  if (!o.render_graphs) cfg.caller.use_graphs = false;
  if (o.sync_prefix_build) cfg.async_prefix_build = false;
  return cfg;
}

}  // namespace

std::unique_ptr<RenderBackend> make_l7_standin_render(int device, int slots, int pool_pages, int voices,
                                                      std::string* err) {
  try {
    L7RenderOptions o;
    o.device = device;
    o.slots = slots;
    o.pool_pages = pool_pages;
    o.voices = voices;
    o.workspace_mb = 64;
    return std::make_unique<L7Render>(service_config(o), std::vector<render::RenderStage>{render::noise_gather_stage(),
                                                                                          render::stub_pcm_stage()},
                                      "l7-standin");
  } catch (const std::exception& ex) {
    if (err) *err = ex.what();
    return nullptr;
  }
}

std::unique_ptr<RenderBackend> make_l7_render(const L7RenderOptions& o, std::string* err) {
#if KT2N_ENGINE_WITH_RENDERER
  try {
    render::RendererConfig rc;
    rc.service = service_config(o);
    rc.build_prefixes = false;  // the adapter builds them (at create, or on first use)
    if (o.hift_max_frame_rows > 0) rc.hift_max_frame_rows = o.hift_max_frame_rows;
    if (o.hift_max_t > 0) rc.hift_max_t = o.hift_max_t;
    if (o.hift_bulk_max_t > 0) rc.hift_bulk_max_t = o.hift_bulk_max_t;
    if (o.hift_bulk_max_frame_rows > 0) rc.hift_bulk_max_frame_rows = o.hift_bulk_max_frame_rows;
    rc.hift_prewarm_rows = std::min(rc.hift_prewarm_rows, std::max(1, o.max_rows));
    rc.urgent_ws_prompt_tokens = o.urgent_ws_prompt_tokens;
    rc.urgent_ws_rows = o.max_rows;
    rc.urgent_ws_long_rows = o.urgent_ws_long_rows;
    rc.bulk_ws_row_tokens = o.bulk_ws_row_tokens;
    rc.bulk_ws_demo_tokens = o.bulk_ws_demo_tokens;
    rc.hift_share = o.hift_share;
    rc.free_pe_tables = o.free_pe_tables;
    rc.pos_tables_from_pool = o.pos_tables_from_pool;
    rc.alias_workspaces = o.alias_workspaces;
    if (o.hift_max_b > 0) rc.hift_max_b = o.hift_max_b;
    auto r = render::Renderer::create(o.s3gen_path, rc, err);
    if (!r) return nullptr;
    return std::make_unique<L7Render>(std::move(r), o);
  } catch (const std::exception& ex) {
    if (err) *err = ex.what();
    return nullptr;
  }
#else
  (void)o;
  if (err) *err = "built without the decoder (kt2n_render_flow)";
  return nullptr;
#endif
}

}  // namespace kt2n::engine
