// Text normaliser: incremental normalisation (reference behaviour), native.
//
// The text planner cuts PREPARED text into planned segments; segment 0 is
// normalised before submission and the rest on the request's normaliser thread
// while segment 0 is on the GPU. This file holds the normaliser's side:
//   * merge_abbreviation_boundaries  -- never end a planned segment on "Nov."
//                                       (moves whole sentences, merge limit 3)
//   * normalize_planned_segment      -- grammar pieces of <= 200 characters,
//                                       segment 0 split for the grammar at
//                                       token-separating spaces (n_split_characters 40)
//                                       when instances are idle, capitalise
//                                       only the first piece of the request,
//                                       terminate only its last
//   * IncrementalPolicy              -- the default timing rules the engine's conductor
//                                       applies to the normaliser thread
// The post-grammar re-split (resplit_segment / demo_split_text, text planner) is a hook: the
// caller passes the returned text to its splitter.
#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "normalizer.h"

namespace kt2n::text {

struct IncrementalPolicy {
  int piece_characters = 200;
  int lookahead_segments = 1;
  int merge_limit = 3;
  double start_audio_seconds = 1.5;
  double first_audio_wait_seconds = 20;
  int n_split_characters = 40;
  int n_pool_workers = 4;                // split fan-out cap
  int chunk_characters = 380;
  double poll_seconds = 0.05;
  int sample_rate = 24000;

  // First-audio wait: may the normaliser thread start segment 1?
  // (segment 0 done, or start_audio_seconds of its audio published; the caller
  // gives up waiting after first_audio_wait_seconds or on cancel).
  bool first_audio_ready(bool first_done, int64_t published_samples, bool preview_ready) const {
    if (first_done) return true;
    const auto start_samples = static_cast<int64_t>(sample_rate * (start_audio_seconds > 0 ? start_audio_seconds : 0));
    if (start_samples <= 0) return preview_ready;
    return published_samples >= start_samples;
  }
  // Normalise-slot wait: may it normalise one more segment now?
  bool slot_ready(size_t jobs_submitted, size_t next_index) const {
    return lookahead_segments <= 0 || jobs_submitted - next_index <= static_cast<size_t>(lookahead_segments);
  }
};

std::vector<std::string> merge_abbreviation_boundaries(const SpeechTextNormalizer& n,
                                                       const std::vector<std::string>& segments,
                                                       const IncrementalPolicy& policy,
                                                       const std::vector<std::string>& emphasis);

struct SegmentResult {
  std::string text;      // joined normalised text, before the planner's re-split
  int grammar_pieces = 1;
  bool n_split = false;  // segment 0 went through the parallel grammar split
};

// `segment_zero_in_flight`: how many requests are normalising their segment 0
// right now INCLUDING this one (the split budget rule). `cancelled` (pieces
// only) stops between two pieces.
SegmentResult normalize_planned_segment(SpeechTextNormalizer& n, const PreparedText& prepared,
                                        const std::vector<std::string>& raw_segments, size_t index,
                                        const IncrementalPolicy& policy, int idle_instances,
                                        int segment_zero_in_flight, const std::atomic<bool>* cancelled = nullptr);
// The serving form: the split budget comes from the normaliser itself
// (segment-0 guard + free grammar copies), as split_budget() computes it.
SegmentResult normalize_planned_segment(SpeechTextNormalizer& n, const PreparedText& prepared,
                                        const std::vector<std::string>& raw_segments, size_t index,
                                        const IncrementalPolicy& policy,
                                        const std::atomic<bool>* cancelled = nullptr);

}  // namespace kt2n::text
