// LM / render backends: the CUDA LM adapter and the CPU stubs (tests and plumbing only).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "backend.h"
#include "kt2n/bf16.h"
#include "policy.h"

namespace kt2n::engine {

// ------------------------------------------------------------------------------------- CUDA LM
namespace {
class L3Lm final : public LmBackend {
 public:
  explicit L3Lm(std::unique_ptr<lm::Engine> e) : e_(std::move(e)) {}
  int add_voice(const std::vector<float>& projected, const std::vector<int32_t>& prefix, std::string* err) override {
    std::vector<uint16_t> spk(projected.size());
    for (size_t i = 0; i < projected.size(); ++i) spk[i] = f32_to_bf16_rne(projected[i]);
    std::vector<int64_t> ids(prefix.begin(), prefix.end());
    return e_->build_voice(spk, ids, err);
  }
  int add_voice_async(const std::vector<float>& projected, const std::vector<int32_t>& prefix,
                      std::string* err) override {
    std::vector<uint16_t> spk(projected.size());
    for (size_t i = 0; i < projected.size(); ++i) spk[i] = f32_to_bf16_rne(projected[i]);
    std::vector<int64_t> ids(prefix.begin(), prefix.end());
    return e_->build_voice_async(spk, ids, err);
  }
  int voice_ready(int v) override { return e_->voice_ready(v); }
  void drop_voice(int v) override { e_->drop_voice(v); }
  int release_voice(int v) override { return e_->release_voice(v); }
  int submit(const lm::RowRequest& row, std::string* err) override { return e_->submit(row, err); }
  void cancel(uint64_t tag) override { e_->cancel(tag); }
  int pump(std::vector<lm::TokenEvent>* ev, bool block, std::string* err) override { return e_->pump(ev, block, err); }
  int set_decode_priority(int level) override { return e_->set_decode_priority(level); }
  int live_rows() const override { return e_->live_rows(); }
  uint64_t steps() const override { return e_->stats().steps; }
  const char* name() const override { return "cuda"; }

 private:
  std::unique_ptr<lm::Engine> e_;
};
}  // namespace

std::unique_ptr<LmBackend> make_l3_lm(const lm::Config& cfg, std::string* err) {
  auto e = lm::Engine::create(cfg, err);
  if (!e) return nullptr;
  return std::make_unique<L3Lm>(std::move(e));
}

// ------------------------------------------------------------------------------------- stub LM
namespace {
// Tests: uses of a voice's GPU state after it was given back, or give-backs while rows still read it (both stubs
// count here; the stub renderer's stats_json reports it as "violations").
std::atomic<long> g_stub_violations{0};

class StubLm final : public LmBackend {
 public:
  explicit StubLm(const StubLmConfig& c) : cfg_(c) {}
  int add_voice(const std::vector<float>&, const std::vector<int32_t>& prefix, std::string*) override {
    prefix_len_.push_back(int(prefix.size()));
    // The voice's tokens depend on its prefix (stable across cache modes: the record, not the index), so a row
    // decoded on another voice's prefix gives other tokens.
    uint64_t h = 1469598103934665603ull;
    for (int32_t t : prefix) h = (h ^ uint64_t(uint32_t(t))) * 1099511628211ull;
    prefix_sig_.push_back(h);
    return int(prefix_len_.size()) - 1;
  }
  // Tests (voice cache): a simulated asynchronous build of build_us, and failures of the first n builds.
  int add_voice_async(const std::vector<float>& projected, const std::vector<int32_t>& prefix,
                      std::string* err) override {
    if (cfg_.voice_build_fail_first > 0) {
      --cfg_.voice_build_fail_first;
      if (err) *err = "stub LM: injected voice build failure";
      return -1;
    }
    const int v = add_voice(projected, prefix, err);
    ready_at_.resize(prefix_len_.size());
    ready_at_[size_t(v)] = std::chrono::steady_clock::now() + std::chrono::microseconds(cfg_.voice_build_us);
    return v;
  }
  int voice_ready(int v) override {
    if (v < 0 || v >= int(prefix_len_.size()) || prefix_len_[size_t(v)] < 0) return -1;
    if (size_t(v) < ready_at_.size() && std::chrono::steady_clock::now() < ready_at_[size_t(v)]) return 0;
    return 1;
  }
  void drop_voice(int) override {}
  int release_voice(int v) override {
    if (v < 0 || v >= int(prefix_len_.size()) || prefix_len_[size_t(v)] < 0) {
      ++g_stub_violations;  // released twice, or never added
      return -1;
    }
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (const auto* rs : {&rows_, &pending_})
        for (const Row& r : *rs)
          if (r.req.voice == v) {
            ++g_stub_violations;  // a submitted row still reads it (the cache drops only voices no row uses)
            break;
          }
    }
    prefix_len_[size_t(v)] = -1;
    return 0;
  }
  int submit(const lm::RowRequest& row, std::string* err) override {
    if (cfg_.voice_required && row.voice < 0) ++g_stub_violations;  // a dropped voice's id (-1) reached admission
    if (row.voice >= 0 && (row.voice >= int(prefix_len_.size()) || prefix_len_[size_t(row.voice)] < 0 ||
                           voice_ready(row.voice) != 1)) {
      if (err) *err = "stub LM: voice not ready";
      return -1;
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (rows_.size() >= 64) {
      if (err) *err = "stub LM full";
      return -1;
    }
    Row r;
    r.req = row;
    r.target = std::max(8, int(std::lround(double(row.suffix.size()) * cfg_.tokens_per_suffix_id)));
    r.rng = (row.seed ^ (row.voice >= 0 ? prefix_sig_[size_t(row.voice)] : 0)) * 6364136223846793005ull +
            1442695040888963407ull;
    pending_.push_back(r);
    return 0;
  }
  void cancel(uint64_t tag) override {
    std::lock_guard<std::mutex> lk(mu_);
    cancels_.push_back(tag);
  }
  int pump(std::vector<lm::TokenEvent>* ev, bool block, std::string* err) override {
    std::unique_lock<std::mutex> lk(mu_);
    if (cfg_.fail_after_steps > 0 && steps_ >= uint64_t(cfg_.fail_after_steps)) {
      if (err) *err = "stub LM: injected failure";
      return -1;
    }
    for (auto& r : pending_) rows_.push_back(r);
    pending_.clear();
    const size_t before = ev->size();
    for (uint64_t tag : cancels_) {
      for (auto it = rows_.begin(); it != rows_.end(); ++it)
        if (it->req.tag == tag) {
          ev->push_back({tag, 0, it->n, lm::Finish::kCancelled});
          rows_.erase(it);
          break;
        }
    }
    cancels_.clear();
    if (rows_.empty()) return int(ev->size() - before);
    if (!block && std::chrono::steady_clock::now() < next_step_) return int(ev->size() - before);
    lk.unlock();
    std::this_thread::sleep_until(next_step_);
    lk.lock();
    next_step_ = std::chrono::steady_clock::now() + std::chrono::microseconds(cfg_.step_us);
    steps_ += 1;
    for (auto it = rows_.begin(); it != rows_.end();) {
      Row& r = *it;
      r.n += 1;
      r.rng = r.rng * 6364136223846793005ull + 1442695040888963407ull;
      int s3;
      if (r.n <= cfg_.opening_quiet_tokens)
        s3 = 4299;  // sub-gate silence id: exercises the opening skip
      else
        s3 = int((r.rng >> 33) % 6561u);
      int32_t tok = lm::kAudioIdBase + s3;
      lm::Finish f = lm::Finish::kNone;
      if (r.n >= r.target) {
        tok = lm::kSpeechEnd;
        f = lm::Finish::kEos;
      } else if (r.req.demo ? r.n >= r.req.budget : r.n >= r.req.budget) {
        f = lm::Finish::kBudget;
      } else if (r.req.sampler.max_new_tokens > 0 && r.n >= r.req.sampler.max_new_tokens) {
        f = lm::Finish::kMaxNewTokens;
      }
      ev->push_back({r.req.tag, tok, r.n, f});
      if (f != lm::Finish::kNone)
        it = rows_.erase(it);
      else
        ++it;
    }
    return int(ev->size() - before);
  }
  int live_rows() const override {
    std::lock_guard<std::mutex> lk(mu_);
    return int(rows_.size() + pending_.size());
  }
  uint64_t steps() const override { return steps_; }
  const char* name() const override { return "stub"; }

 private:
  struct Row {
    lm::RowRequest req;
    int target = 0;
    int n = 0;
    uint64_t rng = 0;
  };
  StubLmConfig cfg_;
  std::vector<std::chrono::steady_clock::time_point> ready_at_;  // add_voice_async: usable from
  mutable std::mutex mu_;
  std::vector<Row> rows_, pending_;
  std::vector<uint64_t> cancels_;
  std::vector<int> prefix_len_;
  std::vector<uint64_t> prefix_sig_;
  std::chrono::steady_clock::time_point next_step_{};
  uint64_t steps_ = 0;
};
}  // namespace

std::unique_ptr<LmBackend> make_stub_lm(const StubLmConfig& cfg) { return std::make_unique<StubLm>(cfg); }

// ------------------------------------------------------------------------------------- stub render
namespace {
class StubRender final : public RenderBackend {
 public:
  explicit StubRender(const StubRenderConfig& c) : cfg_(c) {
    for (int s = 0; s < 2; ++s) workers_[s] = std::thread([this, s] { run(s); });
  }
  ~StubRender() override {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
  }
  // Each voice gets a small signature of its prompt tokens that its audio depends on, so a row rendered with another
  // voice's state gives other bytes (stable across voice cache modes: it depends on the record, not the index).
  int add_voice(const std::vector<int32_t>& prompt_tokens, const std::vector<float>&, const std::vector<float>&,
                std::string*) override {
    std::lock_guard<std::mutex> lk(mu_);
    uint64_t h = 1469598103934665603ull;
    for (int32_t t : prompt_tokens) h = (h ^ uint64_t(uint32_t(t))) * 1099511628211ull;
    sig_.push_back(int(h % 31));
    removed_.push_back(false);
    return voices_++;
  }
  int open_row(uint64_t row, int voice, uint64_t seed, int, bool, std::string*) override {
    std::lock_guard<std::mutex> lk(mu_);
    if (voice >= 0 ? voice >= voices_ || removed_[size_t(voice)] : cfg_.voice_required) ++g_stub_violations;
    // Tests: a renderer that can hold only max_open_rows rows (pool pages / slots): back-pressure, not failure.
    if (cfg_.max_open_rows > 0 && int(rows_.size()) >= cfg_.max_open_rows) {
      ++backpressure_;
      return kRenderBackpressure;
    }
    rows_[row] = RowState{seed, 0.0, voice};
    max_open_ = std::max(max_open_, int(rows_.size()));
    return 0;
  }
  // Tests (KT2N_TEST_STUB_RENDER_PREFIX_WAIT): a voice prefix build waits while any row is open (its pages are held
  // by the rows' reservations), as the real renderer's back-pressure does with a tight pool.
  int prepare_prefix(int, std::string*) override {
    std::lock_guard<std::mutex> lk(mu_);
    return cfg_.prefix_waits_for_rows && !rows_.empty() ? kRenderBackpressure : 0;
  }
  std::string stats_json() const override {
    std::lock_guard<std::mutex> lk(mu_);
    return "{\"open_rows_max\":" + std::to_string(max_open_) + ",\"page_backpressure\":" +
           std::to_string(backpressure_) + ",\"remove_busy\":" + std::to_string(remove_busy_) +
           ",\"violations\":" + std::to_string(g_stub_violations.load()) + "}";
  }
  void close_row(uint64_t row) override {
    std::lock_guard<std::mutex> lk(mu_);
    rows_.erase(row);
  }
  // Busy (-2) while an open row reads the voice, as the real renderer; the cache drops only voices no row uses, so a
  // busy answer is also counted as a violation.
  int remove_voice(int voice, std::string* err) override {
    std::lock_guard<std::mutex> lk(mu_);
    if (voice < 0 || voice >= voices_ || removed_[size_t(voice)]) {
      ++g_stub_violations;
      if (err) *err = "stub render: no such voice";
      return -1;
    }
    for (const auto& [id, r] : rows_)
      if (r.voice == voice) {
        ++remove_busy_;
        ++g_stub_violations;
        return -2;
      }
    removed_[size_t(voice)] = true;
    return 0;
  }
  int launch(RenderCall&& call, std::string*) override {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& r : call.rows) {
      if (r.voice >= 0 ? r.voice >= voices_ || removed_[size_t(r.voice)] : cfg_.voice_required) ++g_stub_violations;
      auto it = rows_.find(r.row);
      if (it != rows_.end() && it->second.voice != r.voice) ++g_stub_violations;  // not the voice the row opened with
    }
    const int s = call.stream == render::StreamRole::kSUrgent ? 0 : 1;
    q_[s].push_back(std::move(call));
    ++inflight_;
    cv_.notify_all();
    return 0;
  }
  void collect(std::vector<RenderOut>* out, bool wait) override {
    std::unique_lock<std::mutex> lk(mu_);
    if (wait && done_.empty() && inflight_ > 0) done_cv_.wait(lk, [&] { return !done_.empty() || inflight_ == 0; });
    while (!done_.empty()) {
      out->push_back(std::move(done_.front()));
      done_.pop_front();
    }
  }
  int inflight() const override {
    std::lock_guard<std::mutex> lk(mu_);
    return inflight_;
  }
  const char* name() const override { return "stub"; }

 private:
  struct RowState {
    uint64_t seed;
    double phase;
    int voice;
  };
  void run(int s) {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
      cv_.wait(lk, [&] { return stop_ || !q_[s].empty(); });
      if (stop_) return;
      RenderCall call = std::move(q_[s].front());
      q_[s].pop_front();
      size_t tokens = 0;
      for (auto& r : call.rows) tokens += r.tokens.size();
      std::vector<int> sig;
      for (auto& r : call.rows) sig.push_back(r.voice >= 0 && r.voice < voices_ ? sig_[size_t(r.voice)] : 0);
      lk.unlock();
      const auto t0 = std::chrono::steady_clock::now();
      std::this_thread::sleep_for(std::chrono::microseconds(cfg_.call_us + int(cfg_.token_us * double(tokens))));
      RenderOut out;
      out.call_id = call.call_id;
      for (size_t ri = 0; ri < call.rows.size(); ++ri) {
        const RenderRowReq& r = call.rows[ri];
        const int end_tok = r.finalize ? int(r.tokens.size()) - policy::kLookahead : r.commit_tokens;
        const int n_tok = std::max(0, end_tok - r.committed_tokens);
        std::vector<float> pcm(size_t(n_tok) * policy::kSamplesPerToken);
        for (int t = 0; t < n_tok; ++t) {
          const int id = r.tokens[size_t(r.committed_tokens + t)];
          const double f = 110.0 + double(id % 64) * 6.0 + double(sig[ri]);
          for (int k = 0; k < policy::kSamplesPerToken; ++k) {
            const size_t i = size_t(t) * policy::kSamplesPerToken + size_t(k);
            const double tt = double(r.committed_tokens * policy::kSamplesPerToken + int(i)) / 24000.0;
            pcm[i] = float(cfg_.amplitude * std::sin(2.0 * M_PI * f * tt));
          }
        }
        out.pcm.push_back(std::move(pcm));
      }
      out.gpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (cfg_.fail_every > 0 && ++calls_ % uint64_t(cfg_.fail_every) == 0) out.status = 303;  // KT2N_ERR_INTERNAL
      lk.lock();
      done_.push_back(std::move(out));
      --inflight_;
      done_cv_.notify_all();
    }
  }
  StubRenderConfig cfg_;
  mutable std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  std::deque<RenderCall> q_[2];
  std::deque<RenderOut> done_;
  std::unordered_map<uint64_t, RowState> rows_;
  std::thread workers_[2];
  int inflight_ = 0;
  int voices_ = 0;
  std::vector<int> sig_;       // per voice: its audio signature
  std::vector<bool> removed_;  // per voice: given back (remove_voice)
  long remove_busy_ = 0;
  int max_open_ = 0;
  long backpressure_ = 0;
  std::atomic<uint64_t> calls_{0};  // both workers count calls
  bool stop_ = false;
};
}  // namespace

std::unique_ptr<RenderBackend> make_stub_render(const StubRenderConfig& cfg) {
  return std::make_unique<StubRender>(cfg);
}

}  // namespace kt2n::engine
