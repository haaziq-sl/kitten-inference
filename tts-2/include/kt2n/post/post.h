// kt2n post-processing: C++ ports of the reference host-audio path
// (reference behaviour). Host float32 only; no CUDA, no allocation on a
// device, no threads. One object per response (not thread safe), exactly like
// the Python classes they port.
//
// Ported reference components:
//   onset trim                    LeadingSilenceTrimmer, SeamEdgeTrimmer
//   streaming runtime             KeptTokenWithholder, StartupHold (startup hold +
//                                 its target function), seam_trim_applies,
//                                 seam_trim_head, trim_and_fade, chunk gap
//   peak normaliser               PeakNorm, demo_peak, demo_int16_with_peak
//   audio output                  normalize_audio, PCM16 / WAV writers,
//                                 whole-file levelling
//
// Numerics: every quantity that decides a branch (frame RMS against a gate) is
// computed in the order numpy 1.26.4 (the reference pin) computes it, so the
// decisions and the samples are bit-identical to the Python reference. The one
// known exception is PeakNorm's fall curve, which uses glibc log/exp where numpy
// may use its own SIMD log/exp, so small numerical differences can remain.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace kt2n::post {

inline constexpr int kSampleRate = 24000;
using Pcm = std::vector<float>;
using Samples = std::span<const float>;

// ---------------------------------------------------------------- numpy-exact helpers
// numpy's pairwise summation (npy pairwise_sum, 8 accumulators, block 128).
double pairwise_sum(const double* a, std::ptrdiff_t n);
// np.mean(np.square(x.astype(float64))) over one frame: (0.0 + pairwise(x^2)) / n.
double frame_mean_square(const float* x, std::ptrdiff_t n);
// np.linspace(start, stop, num, dtype=np.float32) (numpy 1.26 algorithm).
std::vector<float> linspace_f32(double start, double stop, std::ptrdiff_t num);
// Python round(): half to even.
double py_round(double v);
// float(10.0 ** (dbfs / 20.0)).
double dbfs_to_amplitude(double dbfs);

// ---------------------------------------------------------------- onset trim
// Leading-silence trimming defaults for inference.
struct OnsetPolicy {
  double hard_gate_dbfs = -49.0;
  double soft_gate_dbfs = -60.0;
  double keep_ms = 0.0;
  double lookback_ms = 45.0;
  double max_trim_ms = 2000.0;
  double fade_ms = 5.0;
  int sample_rate = kSampleRate;
  double frame_ms = 10.0;
};

class LeadingSilenceTrimmer {
 public:
  // Throws std::invalid_argument on a policy the Python constructor rejects.
  explicit LeadingSilenceTrimmer(const OnsetPolicy& policy = OnsetPolicy{});

  // The samples of `chunk` to publish now (possibly none). Never reorders,
  // never returns a sample twice.
  Pcm feed(Samples chunk);
  // Everything still held back. Call once when the response ends.
  Pcm flush();

  bool armed() const noexcept { return armed_; }
  bool detected() const noexcept { return detected_; }
  int64_t dropped_samples() const noexcept { return dropped_; }
  int64_t held_samples() const noexcept { return held_; }
  double trimmed_ms() const noexcept;

  int frame() const noexcept { return frame_; }
  int keep_frames() const noexcept { return keep_frames_; }
  int lookback_frames() const noexcept { return lookback_frames_; }
  int64_t max_trim_samples() const noexcept { return max_trim_samples_; }
  int fade_samples() const noexcept { return fade_samples_; }

 private:
  Pcm release(Pcm audio, int64_t fade_limit);

  int frame_;
  double hard_gate_;
  double soft_gate_;
  int keep_frames_;
  int lookback_frames_;
  int64_t max_trim_samples_;
  int fade_samples_;
  int sample_rate_;
  bool armed_ = true;
  bool detected_ = false;
  int64_t dropped_ = 0;
  int64_t held_ = 0;
  Pcm pending_;
};

// Reference kept-token withholder (on by default,
// floor 20 ms): cuts the first kept*960 - floor samples before the
// onset trimmer. `kept_tokens` is read on the first non-empty chunk (the decode
// loop records it when the opening closes, before any audio exists); pass a
// pointer the conductor updates, or set it with set_kept_tokens() before audio.
// The reference builds it only at speed 1.0.
class KeptTokenWithholder {
 public:
  KeptTokenWithholder(LeadingSilenceTrimmer trimmer, double floor_ms = 20.0,
                      int sample_rate = kSampleRate);
  void set_kept_tokens(int kept) noexcept { kept_tokens_ = kept; }
  Pcm feed(Samples chunk);
  Pcm flush() { return trimmer_.flush(); }
  int64_t withheld_samples() const noexcept { return withheld_; }
  int64_t dropped_samples() const noexcept { return withheld_ + trimmer_.dropped_samples(); }
  double trimmed_ms() const noexcept;
  const LeadingSilenceTrimmer& trimmer() const noexcept { return trimmer_; }
  bool armed() const noexcept { return trimmer_.armed(); }

 private:
  int64_t budget() const noexcept;
  LeadingSilenceTrimmer trimmer_;
  int floor_;
  int sample_rate_;
  int kept_tokens_ = 0;
  std::optional<int64_t> remaining_;
  int64_t withheld_ = 0;
};

// ---------------------------------------------------------------- seam trim
// SeamEdgeTrimmer: the reference chunk-join edge trim of ONE segment,
// streamed (-45 dBFS gate, keep 20 ms, 8 ms fades, 10 ms frames).
struct SeamPolicy {
  double gate_dbfs = -45.0;
  double keep_ms = 20.0;
  double fade_ms = 8.0;
  int sample_rate = kSampleRate;
  double frame_ms = 10.0;
};

class SeamEdgeTrimmer {
 public:
  explicit SeamEdgeTrimmer(bool trim_head, const SeamPolicy& policy = SeamPolicy{});
  // Throws std::logic_error after finish() (the Python raises RuntimeError).
  Pcm feed(Samples chunk);
  // End of the segment: the held tail, trimmed to keep + fade or whole.
  Pcm finish(bool trim_tail);
  // Releases held whole frames, up to `limit` samples in all (off by default).
  Pcm release(int64_t limit);
  // One whole segment (tests and offline use).
  Pcm trim(Samples audio, bool trim_tail = true);

  bool finished() const noexcept { return finished_; }
  bool loud_seen() const noexcept { return loud_seen_; }
  int64_t head_trimmed_samples() const noexcept { return head_trimmed_; }
  int64_t tail_trimmed_samples() const noexcept { return tail_trimmed_; }
  int64_t stretch_released() const noexcept { return stretch_released_; }

 private:
  int frame_;
  double gate_;
  int keep_;
  int fade_;
  bool trim_head_;
  bool loud_seen_ = false;
  bool finished_ = false;
  int64_t head_trimmed_ = 0;
  int64_t tail_trimmed_ = 0;
  int64_t stretch_released_ = 0;
  int64_t released_ = 0;
  Pcm carry_;
  std::vector<Pcm> held_;  // whole frames, in order
};

// Reference seam-trim choice: does segment `index` get a seam
// trimmer, and does it trim its head? Inputs mirror the request control/lane.
enum class SeamScope { kOff, kAll, kTagged };  // default: tagged
struct SeamContext {
  SeamScope scope = SeamScope::kTagged;
  bool demo_request = false;   // speech mode DEMO
  bool demo_chunked = false;   // demo-chunked request (planned by demo_split_text)
  int dc_mode = 0;             // request dc mode (0 = not a plain demo-chunked request)
  int dc_release_chunks = 0;   // leading chunks without a seam trim (mode 2 only)
  bool lane_known = true;      // lane is not None
  bool lane_sealed = false;
  int lane_jobs = 0;
  bool followup = false;       // dc-guard follow-up job (off by default)
};
bool seam_trim_applies(const SeamContext& ctx, int index);
bool seam_trim_head(const SeamContext& ctx, int index);
// finish(trim_tail=...) in the stream loop: index > 0 or has_more().
inline bool seam_trim_tail(int index, bool has_more) { return index > 0 || has_more; }

// Reference trim-and-fade: the whole-segment (non-incremental) edge trim.
Pcm trim_and_fade(Samples audio, int sample_rate = kSampleRate);

// ---------------------------------------------------------------- gaps
// int(sample_rate * gap_seconds) - credit, never negative (0.16 s by default).
int64_t chunk_gap_samples(double gap_seconds = 0.16, int64_t credit = 0,
                          int sample_rate = kSampleRate);

// ---------------------------------------------------------------- startup hold
// Reference startup hold policy + target function + adaptive_hold_seconds.
// Defaults are the reference default values (mode adaptive, cap 2.0,
// base 0.06, per listener 0.006, alone 0.1,
// alone_first 0.06, first_max_trim_ms 100, min_sharing 0,
// join_hold_ms 240, skip_per_listener 0.0125, skip_min_tokens 8).
struct HoldPolicy {
  bool adaptive = true;          // mode "adaptive" (false = "fixed")
  double cap = 2.0;
  double base = 0.06;
  double per_listener = 0.006;
  std::optional<double> skip_per_listener = 0.0125;
  int skip_min_tokens = 8;
  std::optional<double> alone = 0.1;
  std::optional<double> alone_first = 0.06;
  double first_max_trim_ms = 100.0;
  int min_sharing = 0;
  double join_hold_ms = 240.0;
  // A per-request pinned hold in seconds (the realtime socket pins 0 per phrase).
  std::optional<double> pinned;
  int sample_rate = kSampleRate;
};

// min(cap, base + per_listener * (sharing - 1)), never below 0.
double adaptive_hold_seconds(int sharing, double base, double per_listener, double cap);

// What the target function reads on each held chunk.
struct HoldObservation {
  int listeners = 1;                       // requests holding a lane on this GPU
  bool steady_joiner = false;              // a joiner: someone else is playing
  std::optional<double> trimmed_ms_so_far; // onset trimmer's trimmed_ms (nullopt = no trimmer)
  int opening_skipped_tokens = 0;          // tokens the opening-silence skip dropped
  bool steady_available = true;            // the joiner check is available
};

// Startup hold: the first target samples are released as ONE publication.
class StartupHold {
 public:
  explicit StartupHold(const HoldPolicy& policy = HoldPolicy{});
  // True when a target function exists (adaptive / alone / join hold); false =
  // the fixed hold of fixed_target_samples().
  bool has_target_fn() const noexcept { return has_fn_; }
  int64_t fixed_target_samples() const noexcept { return fixed_target_; }
  // The target function: (target samples, listeners). `call` counts from 1.
  int64_t target(const HoldObservation& obs, int call) const;

  // Feed one published chunk with the observation at that moment. Returns the
  // publication to send now (empty while holding). After release, chunks pass
  // through unchanged.
  Pcm feed(Samples chunk, const HoldObservation& obs);
  // End of the response: whatever is still held (one publication).
  Pcm finish();

  bool released() const noexcept { return released_; }
  bool holding() const noexcept { return !released_ && !held_.empty(); }
  int64_t target_samples() const noexcept { return target_; }
  int64_t held_samples() const noexcept { return static_cast<int64_t>(held_.size()); }
  int peak_sharing() const noexcept { return peak_sharing_; }

 private:
  HoldPolicy policy_;
  bool has_fn_ = false;
  bool passthrough_ = false;
  int64_t fixed_target_ = 0;
  int64_t target_ = 0;
  int calls_ = 0;
  int peak_sharing_ = 0;
  bool released_ = false;
  bool finished_ = false;
  Pcm held_;
};

// ---------------------------------------------------------------- normalise / PCM / WAV
// normalize_audio: false on empty or non-finite input (the reference raises
// RuntimeError -> request error); otherwise `out` = clip(x, -1, 1).
enum class NormalizeStatus { kOk, kEmpty, kNonFinite };
NormalizeStatus normalize_audio(Samples in, Pcm& out);

// Streaming PCM16: (audio * 32767.0) as float32, truncated to int16 (numpy astype).
void pcm16_from_float(Samples in, std::vector<int16_t>& out);
void pcm16_bytes_from_float(Samples in, std::string& out);  // little-endian bytes appended

inline constexpr uint32_t kWavUnknownSize = 0xFFFFFFFFu;
inline constexpr int kWavHeaderBytes = 44;
// RIFF/WAVE PCM16 mono 24 kHz header.
std::string wav_header(uint32_t riff_size, uint32_t data_size, int sample_rate = kSampleRate);
std::string streaming_wav_header(int sample_rate = kSampleRate);  // both sizes 0xFFFFFFFF
// Throws std::invalid_argument when data_bytes would not fit RIFF (as the reference).
std::string complete_wav_header(uint64_t data_bytes, int sample_rate = kSampleRate);
// Reference finalized_wav_header: both size fields set for a total_bytes file.
// A header of any other shape is returned untouched (as the reference); throws
// std::overflow_error when a size does not fit 32 bits (int.to_bytes raises).
std::string finalized_wav_header(std::string_view header, uint64_t total_bytes);

// ---------------------------------------------------------------- peak normalisation
inline constexpr int kDemoFullScale = 32767;
// np.abs(data).max() in float32 (0 when empty).
float demo_peak(Samples audio);
// x / peak (float32) * 32767 (float32) -> int16 (truncation); zeros when peak <= 0.
void demo_int16_with_peak(Samples audio, float peak, std::vector<int16_t>& out);

// Lossy formats (mp3/aac/opus) use the lossy ceiling.
bool is_lossy_format(std::string_view response_format);

struct PeakGainTable {
  std::unordered_map<std::string, double> voices;   // voice id -> gain_db
  std::unordered_map<std::string, std::string> aliases;
  double default_gain_db = 0.0;
  // Loads kittentts-peak-norm-gains.v1 JSON (validates as the reference).
  static PeakGainTable load(const std::string& path);
  static PeakGainTable parse(std::string_view json_text);
};

enum class PeakStreamMode { kVoice, kUniform, kOff };  // default: voice

// voice_gain_db: the starting gain for `voice` (nullopt = custom /
// shared / cloned voice). custom: "median" | "3" | "6" | "12" (default: median).
double voice_gain_db(const PeakGainTable& table, const std::optional<std::string>& voice,
                     PeakStreamMode stream = PeakStreamMode::kVoice,
                     std::string_view custom = "median", std::string* source = nullptr);

class PeakNorm {
 public:
  PeakNorm(bool whole, double start_gain_db, double ceiling, double ms_per_db,
           int sample_rate = kSampleRate);
  // PeakNorm.for_request (defaults: ceiling 999, lossy 891, 5 ms/dB, stream voice,
  // custom median). nullopt when nothing applies (streaming with stream=off).
  static std::optional<PeakNorm> for_request(const PeakGainTable& table,
                                             const std::optional<std::string>& voice, bool whole,
                                             std::string_view response_format,
                                             PeakStreamMode stream = PeakStreamMode::kVoice,
                                             std::string_view custom = "median",
                                             double ms_per_db = 5.0, int ceiling_permille = 999,
                                             int lossy_ceiling_permille = 891,
                                             int sample_rate = kSampleRate);

  // Scale one published chunk (float32 in [-1, 1]).
  Pcm process(Samples audio);

  bool whole() const noexcept { return whole_; }
  double gain() const noexcept { return gain_; }
  double ceiling() const noexcept { return ceiling_; }
  double out_peak() const noexcept { return out_peak_; }
  double peak() const noexcept { return peak_; }
  int chunks() const noexcept { return chunks_; }
  int falls() const noexcept { return falls_; }
  int forced() const noexcept { return forced_; }
  double max_fall_per_sample() const noexcept { return max_fall_per_sample_; }

 private:
  // Returns the per-sample gain curve and writes the slope used.
  std::vector<double> fall_curve(const std::vector<double>& magnitude, double& slope) const;

  bool whole_;
  double start_gain_db_;
  double gain_;
  double ceiling_;
  double ms_per_db_;
  int sample_rate_;
  double max_fall_per_sample_;
  double peak_ = 0.0;
  double out_peak_ = 0.0;
  int chunks_ = 0;
  int falls_ = 0;
  int forced_ = 0;
  double max_forced_db_per_ms_ = 0.0;
};

// The reference per-publication step for a streamed response:
// normalize_audio (clip; an empty / non-finite publication is a request error) then,
// when a streaming PeakNorm state exists, PeakNorm::process. `out` is what goes to the
// PCM16 / WAV writer or to the encoder.
NormalizeStatus level_publication(Samples publication, PeakNorm* peak, Pcm& out);

// The complete-file ("whole") levelling of the reference whole-file /
// whole-WAV peak norm, applied to an already-held utterance chunk with the
// whole utterance's complete-mode peak:
//   pcm / wav : complete-mode int16 (wav bytes = these int16)
//   flac      : complete-mode int16 / 32768 as float32
//   mp3/aac/opus: (x * (ceiling / peak)) in float64 -> float32, clipped (zeros if peak 0)
void whole_file_lossy(Samples chunk, float peak, double ceiling, Pcm& out);
void whole_file_flac(Samples chunk, float peak, Pcm& out);

}  // namespace kt2n::post
