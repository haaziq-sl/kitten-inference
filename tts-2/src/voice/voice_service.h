// Voice cloning: the clone build service behind engine.h kt2n_voice_build, in the reference order per path
// (reference behaviour):
//   inline (P1-P3):
//     transcript rules (validate_transcript)                           -> 215 invalid_reference_transcript
//     the active-safe LRU (32); the clone slot is taken only on a miss (the build holds the slot)
//     admission: 4 running + 16 waiting, 120 s bound, deadline, cancel  -> 212 custom_voice_creation_busy / 5
//     [P3] the stored consent's text re-checked (consent_transcript)   -> 207 / 208
//   saved create (P4/P5/P10): admission FIRST (the management slot wraps the method), then
//     transcript rules -> 215, [P5] consent text re-check -> 207 / 208
//   consent create (P6): admission, consent transcript rules -> 216
//     invalid_consent_transcript, language -> 208 unsupported_consent_language, phrase -> 207
//     consent_phrase_mismatch, then decode + speaker embedding only (no GPU build)
//   then, every path: decode in the caller's thread (in-process FFmpeg) or decoded 24 kHz PCM -> [gpu_turn]
//     -> the GPU build, one at a time per VoiceBuilder (the reference: one blocking GPU build) -> [gpu_done]
// The transcript tokenizer (Qwen BPE, from the text plan) is injected; the projected speaker comes from voices.kt2n[nearest].
#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "clone_policy.h"
#include "voice_blob.h"
#include "voice_build.h"

namespace kt2n::voice {

enum VoiceBuildFlags : uint32_t {  // engine.h kt2n_voice_build_flags
  kVbInline = 1u << 0,
  kVbSaved = 1u << 1,
  kVbDecodedPcm = 1u << 2,
  kVbConsent = 1u << 3,
  kVbVerifySpeaker = 1u << 4,
};

struct VoiceBuildRequest {
  uint32_t flags = 0;
  const uint8_t* audio = nullptr;  // encoded upload (<= 10 MiB decoded is checked by the API layer)
  size_t audio_len = 0;
  const float* pcm = nullptr;      // kVbDecodedPcm: mono float32 at 24 kHz
  size_t pcm_samples = 0;
  int pcm_rate = 24000;
  const float* consent_speaker = nullptr;  // kVbVerifySpeaker: 512 floats
  std::string reference_transcript;  // kVbConsent: the consent transcript
  std::string language;              // kVbConsent: the consent language (BCP 47; the reference: English only)
  // kVbVerifySpeaker (P3/P5): the stored consent's transcript and language, re-checked as the reference does inside
  // the slot. Empty = the caller already checked (engine.h has no field for them yet).
  std::string consent_transcript;
  std::string consent_language;
  std::string identity_key;  // kVbInline: the inline LRU key (the reference lease identity). "" = never cached
  std::string voice_id;      // kVbSaved: the id written into the one-voice artifact ("" = "clone-<audio sha[:16]>")
  std::optional<Clock::time_point> deadline;
  std::function<bool()> cancelled;
  // The conductor's GPU grant: called after admission and decode, with this
  // builder's GPU mutex held, right before the GPU work. Return false to abandon (-> KT2N_ERR_CANCELLED).
  // gpu_done runs right after the GPU work (before the export), also under the mutex, and only when gpu_turn
  // granted the turn (a refused turn did no GPU work). Both optional.
  std::function<bool()> gpu_turn;
  std::function<void()> gpu_done;
};

struct VoiceBuildOutcome {
  int32_t status = 0;  // engine.h kt2n_status (0 ok)
  std::string code;    // the reference error.code
  std::string param;   // the reference error.param
  std::string message; // the reference error.message (its wording, incl. the clip length for 201)
  std::shared_ptr<const CloneResult> result;
  bool cache_hit = false;
  double admission_wait_ms = 0.0;
  double decode_ms = 0.0;
  double gpu_wait_ms = 0.0;  // waiting for the builder's GPU mutex + gpu_turn
  double build_ms = 0.0;     // the GPU build itself (or decode + embedding for consent)
  // With a catalogue (set_catalogue), every successful clone build also returns:
  const float* projected_speaker = nullptr;  // the nearest catalogue voice's projected row (2048 floats)
  std::string nearest_voice_id;
  json::Value identity;           // the reference identity fields
  std::vector<uint8_t> artifact;  // kVbSaved: the one-voice voices.kt2n blob (kt2n_voice_export)
  double export_ms = 0.0;
};

class VoiceService {
 public:
  using Tokenize = std::function<std::vector<int32_t>(const std::string&)>;
  VoiceService(VoiceBuilder& builder, Tokenize tokenize, int slots = 4, int depth = 16, double wait_s = 120.0,
               int inline_lru = 32);

  // Blocking; call from any engine thread. Thread-safe.
  VoiceBuildOutcome build(const VoiceBuildRequest& req);

  // the reference transcript rules: strip; 1..4096 chars and <= 16 KiB UTF-8; no control chars except \n \r \t.
  // Returns "" when valid, else the error message (code invalid_<field>; field = reference_transcript or
  // consent_transcript).
  static std::string validate_transcript(const std::string& raw, std::string* stripped,
                                         const char* field = "reference_transcript");

  // The built-in catalogue (voices.kt2n): nearest built-in voice bank + projected rows; also replaces the builder's bank.
  // Call once before serving.
  void set_catalogue(const Catalogue* catalogue);

  CloneAdmission& admission() { return admission_; }
  int inline_entries() const { return lru_.entries(); }

 private:
  // inline: text_ids / transcript already checked by build(); saved / consent: checked here after admission.
  VoiceBuildOutcome build_uncached(const VoiceBuildRequest& req, const std::vector<int32_t>* text_ids,
                                   std::string transcript);
  void fill_projection(VoiceBuildOutcome& out) const;
  VoiceBuilder& builder_;
  Tokenize tokenize_;
  const Catalogue* catalogue_ = nullptr;
  CloneAdmission admission_;
  double wait_s_;
  ActiveSafeLru<std::string, std::shared_ptr<const CloneResult>> lru_;
  std::mutex gpu_mu_;  // one GPU build at a time per builder
};

}  // namespace kt2n::voice
