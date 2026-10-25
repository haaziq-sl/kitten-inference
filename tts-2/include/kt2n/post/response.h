// One streamed response's host-audio path, in the reference's exact order
// (the reference streaming generator + the incremental
// branch of its streaming path, default settings):
//
//   per engine chunk of segment i:
//     seam trimmer (if seam_trim_applies) -> kept-token withhold + onset trim
//     -> (initial reservoir: 0 by default) -> atempo stretch (speed != 1) -> publish
//   end of segment i:
//     seam.finish(trim_tail = i > 0 or has_more) -> trim -> stretch -> publish
//     last segment: onset trimmer flush -> stretch -> publish
//     has_more: 0.16 s of zeros -> onset trimmer -> stretch -> publish
//     last segment: stretcher flush -> publish
//
// Then (not in this class, as in the reference): the startup hold (StartupHold) ->
// normalize_audio -> PeakNorm.process -> PCM16 / WAV / encoder. The outer
// publish-time trim gate is a pass-through in this branch (the trimmer has
// already disarmed when anything reaches it) and the internal-silence limiter
// and the shaper are off by default, so neither appears here.
#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "kt2n/post/atempo.h"
#include "kt2n/post/post.h"

namespace kt2n::post {

struct ResponsePostConfig {
  double speed = 1.0;
  bool onset_trim = true;   // on by default, and not for a complete (stream:false) request
  bool withhold = true;     // applied at speed 1 only
  OnsetPolicy onset{};
  double withhold_floor_ms = 20.0;
  double gap_seconds = 0.16;
  int sample_rate = kSampleRate;
};

class ResponsePost {
 public:
  explicit ResponsePost(const ResponsePostConfig& cfg = ResponsePostConfig{});
  // The opening-silence skip's kept token count, known
  // before any audio exists. Ignored when the withhold is not active.
  void set_kept_opening_tokens(int kept);
  // Segment `index` starts; `seam` describes the request/lane now.
  void begin_segment(int index, const SeamContext& seam);
  // One engine chunk of the current segment -> audio to publish now (maybe empty).
  Pcm push(Samples chunk);
  // The current segment's audio is complete; `has_more`: another segment follows.
  // Returns the publications in the reference order (each one is a separate
  // yield there, and the startup hold evaluates its target once per publication):
  // seam tail, onset-trimmer flush (last segment), gap, stretcher flush (last).
  // Empty publications are omitted.
  std::vector<Pcm> end_segment(bool has_more);
  // After the last segment (the reference's belt-and-braces stretcher flush).
  Pcm finish();

  // Trimmed ms so far for the startup hold (nullopt: no onset trimmer).
  std::optional<double> trimmed_ms() const;
  int64_t dropped_samples() const;
  bool stretching() const noexcept { return stretcher_ != nullptr; }
  // The current segment's seam trimmer (nullptr: none for this segment). Read after end_segment() for
  // the reference seam_trimmed note (the seam's stats after seam.finish). Used by the engine.
  const SeamEdgeTrimmer* seam() const noexcept { return seam_.get(); }

 private:
  Pcm trim_and_stretch(Pcm chunk);
  Pcm stretch(Pcm chunk);

  ResponsePostConfig cfg_;
  std::unique_ptr<LeadingSilenceTrimmer> trimmer_;
  std::unique_ptr<KeptTokenWithholder> withholder_;
  std::unique_ptr<SpeedStretcher> stretcher_;
  std::unique_ptr<SeamEdgeTrimmer> seam_;
  int index_ = -1;
};

}  // namespace kt2n::post
