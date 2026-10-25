// Text normaliser: the reference SpeechTextNormalizer glue (reference behaviour),
// native, with these defaults:
//   emo_tags=1 emo_request_level=1 demo_formatting=1
//   fast_path=1 language_mode=text unit_words=1
//   version_words=1 cache_entries=4096 (a process pool / the lookahead tagger are
//   output-neutral: the grammar instances here ARE the pool).
//
// prepare(): tag/emphasis protection, demo pre-formatting, bracket flattening,
// UI glyphs, dash rules, the language branch, the straight-quote pair rewrite.
// normalize_prepared(): unit + version words, the fast path that skips the
// WFST when it cannot change the text, the grammar call (2 attempts, then fail
// open), capitalise/terminate, tag restore. normalize() = both.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nemo.h"

namespace kt2n::text {

struct NormalizerConfig {
  std::string grammar_dir;              // tagger.fst, verbalizer.fst, post_processor.fst,
                                        // whitelist_words.txt, data_words.txt
  bool enabled = true;
  bool multilingual = false;            // frozen voice-selected structured values
  bool emo_tags = true;
  bool emo_request_level = true;
  bool demo_formatting = true;
  bool fast_path = true;
  std::string language_mode = "text";   // voice|text|all
  bool unit_words = true;
  bool version_words = true;
  int cache_entries = 4096;
  int grammar_instances = 2;            // 2 text threads
  // What create() does when the grammar files cannot be loaded (missing,
  // unreadable or corrupt tagger/verbalizer/post_processor .fst, or an empty /
  // missing whitelist_words.txt or data_words.txt).
  //   false (default): create() returns nullptr and *error says which file and
  //     why, so an engine with a broken text_data_dir refuses to start (the engine's
  //     TextFront::create then fails the engine create).
  //   true: the reference behaviour (its warm() fails open): a
  //     normaliser that passes every text through un-normalised, with
  //     status() = "unavailable:<why>". Only for callers that report status().
  bool allow_unavailable_grammar = false;
  // TEST ONLY: the next N grammar calls "raise" (fail-open equivalence checks).
  long test_fail_grammar_calls = 0;
};

struct PreparedText {
  std::string text;
  std::string raw;
  bool wfst = false;
  std::vector<std::string> emphasis;  // the request's own (((emphasis))) spans
  std::string voice_language;
  bool normalize = true;
  bool voice_masked_values = false;
};

struct NormalizeOptions {
  std::string voice_language;
  bool normalize = true;
  bool voice_masked_values = false;
  bool wfst = true;
  int fast_path = -1;  // -1 = the normaliser's setting
  bool capitalize = true;
  bool terminate = true;
  const std::string* fallback = nullptr;
  const std::vector<std::string>* emphasis = nullptr;
  int priority = 1;  // grammar priority: 0 = text the first byte waits on
};

struct NormalizerCounters {
  std::atomic<uint64_t> fast_path_skips{0}, fast_path_runs{0}, grammar_calls{0}, retries{0}, fail_open{0},
      cache_hits{0}, cache_misses{0};
};

// Persistent helper threads for the parallel split's pieces (the reference keeps a
// ThreadPoolExecutor for the same job). Creating a thread per piece cost
// ~0.1-0.7 ms per request on a busy CPU set (measured).
class HelperPool {
 public:
  explicit HelperPool(int threads);
  ~HelperPool();
  HelperPool(const HelperPool&) = delete;
  HelperPool& operator=(const HelperPool&) = delete;
  // tasks[0] runs on the calling thread, the rest on the helpers; returns when
  // all are done (the first exception, if any, is rethrown here).
  void run(std::vector<std::function<void()>>& tasks);
  int threads() const { return static_cast<int>(threads_.size()); }

 private:
  void loop();
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::function<void()>*> queue_;
  bool stop_ = false;
  std::vector<std::thread> threads_;
};

class SpeechTextNormalizer {
 public:
  static std::unique_ptr<SpeechTextNormalizer> create(const NormalizerConfig& cfg, std::string* error);

  PreparedText prepare(std::string_view text, std::string_view language = "en", bool normalize = true) const;
  std::string normalize_prepared(std::string_view prepared, const NormalizeOptions& opts);
  std::string normalize(std::string_view text, std::string_view language = "en");
  // Would normalize_prepared(prepared, opts) call the grammar (not the fast
  // path / no-WFST shortcut)? Same preprocessing, no grammar call.
  bool needs_grammar(std::string_view prepared, const NormalizeOptions& opts) const;
  // The parallel split's fan-out: tasks[0] on the caller, the rest on the helper threads.
  void run_parallel(std::vector<std::function<void()>>& tasks);

  // Helpers the incremental path and the planner use.
  bool nemo_can_change(std::string_view text) const;
  bool ends_on_abbreviation(std::string_view text) const;
  std::vector<size_t> grammar_cut_points(std::string_view text) const;  // code-point indices
  std::vector<std::string> split_for_grammar(std::string_view text, int target, int max_pieces,
                                             int minimum = 12) const;
  // grammar_pieces: sentence-aligned pieces of at most `limit` characters.
  std::vector<std::string> grammar_pieces(std::string_view text, int limit,
                                          const std::vector<std::string>& emphasis) const;
  // The raw grammar answer (cache, then a grammar instance); empty optional = raised.
  NemoResult grammar(std::string_view protected_text, int priority = 1);
  // One grammar call, no cache (benchmarks / the C ABI's kt2n_text_nemo).
  NemoResult grammar_uncached(std::string_view text) {
    return pool_ ? pool_->normalize(text) : NemoResult{false, "", "unavailable"};
  }

  // control_spans: [start, end) byte offsets of every tag and
  // counted (((emphasis))) span (emphasis nullptr = every span counts).
  std::vector<std::pair<size_t, size_t>> control_spans(std::string_view text,
                                                       const std::vector<std::string>* emphasis) const;
  bool emo_enabled() const { return cfg_.emo_tags || cfg_.emo_request_level; }
  // Parallel split budget (the reference split budget / segment-0 guard).
  int split_budget() const {
    if (!pool_) return 1;
    if (segment_zero_in_flight_.load() > 1) return 1;
    return std::max(1, pool_->idle_count());
  }
  // One request's segment 0 counts once. A guard nested on the same thread for
  // the same normaliser (e.g. the caller holds one and then calls the serving
  // form of normalize_planned_segment, which takes its own) does not count
  // again; counting it twice made split_budget() return 1, so the parallel split never
  // ran (found by the engine's text_front.cc).
  struct SegmentZeroGuard {
    explicit SegmentZeroGuard(SpeechTextNormalizer& n) : n_(n), prev_(holder()) {
      if (prev_ == &n_) return;  // nested: already counted
      n_.segment_zero_in_flight_++;
      holder() = &n_;
    }
    ~SegmentZeroGuard() {
      if (prev_ == &n_) return;
      n_.segment_zero_in_flight_--;
      holder() = prev_;
    }
    SegmentZeroGuard(const SegmentZeroGuard&) = delete;
    SegmentZeroGuard& operator=(const SegmentZeroGuard&) = delete;
    int in_flight() const { return n_.segment_zero_in_flight_.load(); }

   private:
    static const SpeechTextNormalizer*& holder() {
      thread_local const SpeechTextNormalizer* h = nullptr;
      return h;
    }
    SpeechTextNormalizer& n_;
    const SpeechTextNormalizer* prev_;
  };
  // TEST ONLY: make the next `n` grammar calls raise.
  void test_fail_next(long n) { test_failures_left_ = n; }
  const NormalizerConfig& config() const { return cfg_; }
  const NormalizerCounters& counters() const { return counters_; }
  std::string status() const;

 private:
  explicit SpeechTextNormalizer(const NormalizerConfig& cfg) : cfg_(cfg) {}

  // LRU of the grammar's raw answers, keyed by the exact input (<= 2048 chars).
  struct Cache {
    size_t entries = 0;
    std::mutex mu;
    std::list<std::pair<std::string, std::string>> order;  // front = newest
    std::unordered_map<std::string, std::list<std::pair<std::string, std::string>>::iterator> index;
    bool get(const std::string& key, std::string* value);
    void put(const std::string& key, const std::string& value);
  };

  NormalizerConfig cfg_;
  std::unique_ptr<NemoPool> pool_;
  std::unique_ptr<HelperPool> helpers_;
  std::unordered_set<std::string> whitelist_;   // whitelist_words.txt
  std::unordered_set<std::string> data_words_;  // data_words.txt
  Cache cache_;
  mutable NormalizerCounters counters_;
  std::atomic<long> test_failures_left_{0};
  std::atomic<int> segment_zero_in_flight_{0};
  std::string status_ = "not_loaded";
};

// text_is_english (pure function of the text).
bool text_is_english(std::string_view text);
// normalize_dashes.
std::string normalize_dashes(std::string_view text);
// text_has_tag / leading_expression_tag helpers for the planner.
bool text_has_expression_tag(std::string_view text);

}  // namespace kt2n::text
