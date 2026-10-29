// kt2n text facade for the engine: one request's text
// from entry to LM jobs, in the reference order.
//
// Reference behaviour (default settings), in order:
//   entry              text = input.strip(); blank -> rejected (400)
//                      (a phrase of incremental input reaches the engine as cut, unstripped)
//   step 1             request emotion decision        -> RequestPlan()
//   step 1             demo chunking mode (tagged / plain)
//   steps 2-3          normalisation plan: prepare() (normaliser) + plan
//   segment 0          normalize_planned_segment (normaliser grammar
//                      + re-split)                       -> add_normalized(0)
//   chunk count        demo chunk count = len(raw_segments) - 1 + len(seg-0 chunks)
//   submit             submit the segment-0 chunks: prompt + budget
//   segments 1..       on the request's normaliser thread,
//                      extend the request (returns before the extend when
//                      a segment yields no chunks)
//                                                    -> add_normalized(1, 2, ...)
//
// What the facade does NOT do (and who does):
//   * prepare() and the grammar (normalize_planned_segment) are the normaliser's
//     (kt2n::text, OpenFst).  The caller runs them and hands the results in,
//     so this library keeps no third-party dependency.  Pass the grammar the
//     PLANNED segments (segments() as UTF-8) and the PreparedText it came
//     from; the normaliser's n_split path is the caller's choice (latency only: the text
//     was identical on every corpus text).
//   * When segment k+1 may be normalised (first-audio wait, look-ahead 1) is
//     the normaliser's IncrementalPolicy + the engine's thread.  The facade only checks ORDER.
//   * The voice prefix comes from the voice; suffix ids here are what follows it.
//
// Not thread safe: one request's plan is used by one thread at a time (the engine
// already serialises segment 0 on the request's text task and the rest on its
// normaliser thread).
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "kt2n/textplan.h"

namespace kt2n::textplan {

// The normaliser's prepare() result (kt2n::text::PreparedText), as code points.
struct Prepared {
  U32 text;
  bool wfst = false;   // the grammar applies (abbreviation merge on)
  Emphasis emphasis;   // the request's (((emphasis))) spans
};
// The passthrough prepare (no normaliser compiled in; plumbing tests only, never
// a release build): text unchanged, no grammar, the request's own emphasis spans.
Prepared passthrough_prepare(std::u32string_view request_text);

// stream:true (streaming) or stream:false (complete generation).
enum class BudgetRule { kStream, kDemo };

struct LmChunk {
  U32 text;                      // what the LM is asked to say
  std::vector<int32_t> suffix;   // ids after the voice prefix (empty when no tokenizer)
  int budget = 0;                // max new LM tokens (reference behaviour)
  int ordinal = 0;               // request-wide job index (seed = salt + ordinal)
  int segment = 0;               // planned segment it came from
};

class RequestPlan {
 public:
  // Steps 1-3.  ``request_text``: as the reference streaming entry point receives it
  // (a whole request: already input.strip()-ed and non-blank; a phrase of incremental input: as cut).
  // ``prepared``: the normaliser's prepare(request_text, language), or passthrough_prepare.
  // ``emotion_preset``: a phrase's turn stamp (EmotionTurn::stamp), which
  // wins over the phrase's own text.
  RequestPlan(std::u32string_view request_text, Prepared prepared, const Abbreviations& abbreviations,
              BudgetRule rule, std::optional<bool> emotion_preset = std::nullopt,
              const Settings& settings = {});

  const RequestText& decision() const { return decision_; }
  bool emotion() const { return decision_.emotion; }  // {emo: 1} on every prompt
  int dc_mode() const { return decision_.dc_mode; }   // demo chunking mode (1 plain, 0 tagged)
  const Prepared& prepared() const { return prepared_; }
  // The planned (pre-grammar) segments; the grammar's input.
  const std::vector<U32>& segments() const { return segments_; }

  // Steps 4-5 for planned segment next_segment(): its grammar output (the normaliser's
  // SegmentResult::text) -> demo re-split -> LM chunks with prompt suffix and
  // budget.  Throws std::logic_error when called out of order or after the
  // last segment.  ``tokenizer`` nullptr leaves suffix empty (stub LM).
  std::vector<LmChunk> add_normalized(std::size_t index, std::u32string_view normalized,
                                      const Tokenizer* tokenizer);
  std::size_t next_segment() const { return next_; }
  bool sealed() const { return next_ >= segments_.size(); }  // every segment added
  // the request's demo chunk count (known once segment 0 is added; -1 before).
  int total_chunks() const { return total_chunks_; }
  int chunks_created() const { return chunks_; }

 private:
  Settings settings_;
  BudgetRule rule_;
  RequestText decision_;
  Prepared prepared_;
  std::vector<U32> segments_;
  std::size_t next_ = 0;
  int total_chunks_ = -1;
  int chunks_ = 0;
};

}  // namespace kt2n::textplan
