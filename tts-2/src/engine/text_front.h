// The text front end in the reference order:
//   decide_request -> normaliser prepare -> text plan plan_segments -> per planned segment normaliser
//   normalize_planned_segment -> text plan resplit_segment -> per LM chunk target_suffix + budget.
// The normaliser (NeMo glue on OpenFst) is compiled in only with KT2N_BUILD_TEXT; without it the front end
// runs a PASSTHROUGH normaliser (text unchanged) and says so (real_normalizer() == false): that
// mode exists for plumbing tests and is never used in a release build.
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "kt2n/textplan.h"

namespace kt2n::engine {

struct PlannedRequest {
  textplan::RequestText decision;
  std::string language;
  std::shared_ptr<void> prepared;          // the normaliser's PreparedText (opaque here)
  std::vector<std::string> raw_segments;   // planned segments (prepared text), UTF-8
};

class TextFront {
 public:
  // text_data_dir: NeMo grammar dir (tagger.fst, verbalizer.fst, post_processor.fst, *_words.txt) plus
  // nemo_whitelist.txt (abbreviations). NULL = passthrough. tokenizer_json NULL = code points as ids
  // (stub LM only).
  // How the normaliser's grammars (most of the text front end's start time) load. The tokenizer and the abbreviation
  // list always load in create. kBackground: on a background thread started by create; kDeferred: on a background
  // thread started by start_loading() (or synchronously by wait_ready() if never started). Either way wait_ready()
  // joins it and must be called before plan(), normalize_segment() or real_normalizer(); a grammar load error is then
  // reported by wait_ready().
  enum class Load { kNow, kBackground, kDeferred };
  static std::unique_ptr<TextFront> create(const char* text_data_dir, const char* tokenizer_json, int instances,
                                           std::string* err, Load load = Load::kNow);
  virtual ~TextFront() = default;
  // Starts a kDeferred load (no-op otherwise).
  virtual void start_loading() = 0;
  // Joins a background load (no-op otherwise). False, with *err set, when the grammars failed to load.
  virtual bool wait_ready(std::string* err) = 0;
  // emotion_preset: a WS phrase's turn stamp (EmotionTurn::stamp), which wins over the phrase text.
  virtual PlannedRequest plan(const std::string& raw, const std::string& language,
                              std::optional<bool> emotion_preset = std::nullopt, bool normalize = true) = 0;
  // Planned segment `index` -> its LM chunks (normalized, re-split).
  virtual std::vector<std::u32string> normalize_segment(const PlannedRequest& p, size_t index) = 0;
  // One of those LM chunks re-split by the same rule into pieces of at most max_chars characters (+1 for terminal
  // punctuation; a single word longer than that stays whole): the low-memory fit's streaming limit on small GPUs.
  virtual std::vector<std::u32string> split_chunk(const PlannedRequest& p, const std::u32string& chunk,
                                                  int max_chars) = 0;
  virtual std::vector<int32_t> suffix(const std::u32string& chunk, bool emotion) const = 0;
  // tokenizer.encode(text, add_special_tokens=False) (clone transcripts).
  virtual std::vector<int32_t> encode(const std::string& utf8) const = 0;
  virtual bool real_normalizer() const = 0;
  virtual bool real_tokenizer() const = 0;
};

}  // namespace kt2n::engine
