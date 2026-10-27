// kt2n native text plan: segmentation, emotion tags, token
// budgets, the Qwen2 byte-level BPE tokenizer, the LM prompt builder and the
// phrase planner for text that arrives in pieces.
//
// Every function is a port of the reference implementation (reference
// behaviour) with these default settings:
//   emotion tags on, decided per request, tagged requests demo-chunked,
//   plain requests demo-chunked too (every request is planned with the demo splitter),
//   demo chunks of 380 / 130 characters, tagged budget 4.4 tokens per character,
//   no tagged budget cap in seconds, demo budget 1000 tokens,
//   sentence batching of streamed input, phrases of at most 380 characters,
//   0.3 s phrase idle time, turn batcher on, commit memory on,
//   30 ms cut grace, one phrase prefetched.
// The Python functions index strings by code point, so the port works on
// UTF-32 (std::u32string); "length" below always means code points.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace kt2n::textplan {

using U32 = std::u32string;

// ---- UTF-8 / code points --------------------------------------------------
// Invalid UTF-8 decodes to U+FFFD per bad byte (the API only hands over valid
// JSON strings, so this does not happen in normal use).
U32 from_utf8(std::string_view utf8);
std::string to_utf8(std::u32string_view text);

// CPython 3.11 str predicates (Unicode 14.0.0, the reference interpreter).
bool py_isspace(char32_t c);
bool py_isalnum(char32_t c);
bool py_isalpha(char32_t c);
U32 py_strip(std::u32string_view s);
U32 py_lstrip(std::u32string_view s);
U32 py_rstrip(std::u32string_view s);
U32 py_rstrip_chars(std::u32string_view s, std::u32string_view chars);
U32 py_casefold(std::u32string_view s);
// str.isalpha() on a whole string (non-empty, every code point alphabetic).
bool py_isalpha_str(std::u32string_view s);

// ---- Default settings (frozen; see the file comment) ----------------------
struct Settings {
  int demo_chunk_characters = 380;
  int demo_chunk_min_characters = 130;
  int normalize_merge_limit = 3;
  double tagged_budget_per_char = 4.4;
  double tagged_budget_max_seconds = 0; // 0 = longest demo chunk
  double max_seconds_per_segment = 30;  // engine max_seconds_per_segment
  int demo_max_new_tokens = 1000;
  double demo_chars_per_sec = 20;
  // Realtime planner.
  int ws_max_phrase_characters = 380;
  double ws_phrase_idle_ms = 300;
  double ws_cut_grace_ms = 30;
  int ws_turn_min_sentence_characters = 50;
  int ws_min_sentence_characters = 24;
  std::size_t ws_max_message_characters = 8192;   // largest message, in characters
  std::size_t ws_max_input_characters = 262144;   // largest total input, in characters
};

// ---- Emotion controls: [tag], <event>, (((emphasis))) ----------------------
struct Span {
  std::size_t start = 0;
  std::size_t end = 0;
  bool operator==(const Span&) const = default;
};
// ``emphasis``: the request's own (((emphasis))) span strings.  nullptr means
// "this IS the request text" (every emphasis match counts), as in Python.
using Emphasis = std::vector<U32>;

Emphasis emphasis_spans(std::u32string_view request_text);
std::vector<Span> control_spans(std::u32string_view text, const Emphasis* emphasis);
bool text_has_tag(std::u32string_view text, const Emphasis* emphasis);
bool text_has_expression_tag(std::u32string_view text);  // the release rule (tags only)
std::optional<U32> leading_expression_tag(std::u32string_view text);
bool speakable_outside_spans(std::u32string_view text, const Emphasis* emphasis);
std::optional<Span> cut_inside_control_span(std::u32string_view text, std::size_t cut,
                                            const Emphasis* emphasis);

struct Masked {
  U32 text;
  std::vector<std::pair<U32, U32>> restore;  // (masked run, original span)
};
Masked mask_control_spans(std::u32string_view text, const Emphasis& emphasis,
                          std::size_t first = 0);
U32 unmask_control_spans(U32 text, const std::vector<std::pair<U32, U32>>& restore);

// ---- Abbreviations (ends_on_abbreviation) --------------------------------
class Abbreviations {
 public:
  // ``whitelist``: NeMo's English whitelist left column, casefolded, one per
  // line (written by tools/kt2n_assets/prepare.py as nemo_whitelist.txt and
  // shipped next to the engine).  An empty table is the reference fallback.
  static Abbreviations from_lines(std::string_view whitelist);
  static Abbreviations load(const std::string& path);
  bool ends_on_abbreviation(std::u32string_view text) const;
  std::size_t whitelist_size() const;

 private:
  struct Impl;
  std::shared_ptr<const Impl> impl_;
};

// ---- Segmentation (reference behaviour) -----------------------------------
U32 ensure_terminal_punctuation(std::u32string_view chunk);
// demo_split_text(text, emphasis, chars): 0 = the default 380 / 130.
std::vector<U32> demo_split_text(std::u32string_view text, const Emphasis& emphasis,
                                 int chars = 0, const Settings& settings = {});
std::vector<U32> merge_abbreviation_boundaries(const std::vector<U32>& segments,
                                               const Abbreviations& abbreviations,
                                               int max_characters, const Emphasis& emphasis,
                                               int limit = 3);
std::vector<U32> emo_segments(const std::vector<U32>& segments, const Emphasis& emphasis);

// The default request plan (normalization planning with demo chunks):
// ``prepared_text`` / ``wfst`` come from the normalizer's prepare().
struct RequestText {
  bool emotion = false;     // request-level {emo: 1} decision
  bool demo_chunks = true;  // always true at the default settings
  int dc_mode = 0;          // 1 = plain request, 0 = tagged
  Emphasis emphasis;        // the request's own spans
  std::optional<U32> leading_tag;
};
// ``preset``: a phrase's turn decision stamped by the caller
// (EmotionTurn::stamp -> control.emotion); it wins over the phrase's own text.
RequestText decide_request(std::u32string_view request_text,
                           std::optional<bool> preset = std::nullopt);
std::vector<U32> plan_segments(std::u32string_view prepared_text, bool wfst,
                               const Emphasis& emphasis, const Abbreviations& abbreviations,
                               const Settings& settings = {});
// The re-split of one normalized planned segment (the demo split).
std::vector<U32> resplit_segment(std::u32string_view normalized, const Emphasis& emphasis,
                                 const Settings& settings = {});
// The prompt's emotion flag at the default settings: the request-level decision.
inline bool prompt_emotion(const RequestText& request) { return request.emotion; }

// ---- LM token budgets ------------------------------------------------------
// Streaming (demo-chunked) segment: chars x 4.4, floor 200, cap 2341.
int tagged_segment_budget(std::size_t chars, const Settings& settings = {});
// stream:false (speech mode DEMO): a single-chunk request gets 1000, a chunk
// of a longer one max(200, min(1000, int(chars / 20 * 25 * 1.8))).
// ``total_chunks``: the request's demo chunk count (len(raw_segments) - 1 +
// len(segment-0 chunks)); ``batch``: the chunks submitted together (segment
// 0's, then each later segment's), ``start_index`` their offset.
std::vector<int> demo_budgets(const std::vector<std::size_t>& batch_chars, int total_chunks,
                              int start_index, const Settings& settings = {});

// ---- Tokenizer (Qwen2 byte-level BPE, tokenizer.json) ---------------------
class Tokenizer {
 public:
  static Tokenizer load(const std::string& tokenizer_json_path);
  static Tokenizer from_json(std::string_view tokenizer_json);
  // tokenizer.encode(text, add_special_tokens=False)
  std::vector<int32_t> encode(std::string_view utf8) const;
  std::vector<int32_t> encode(std::u32string_view text) const;
  std::size_t size() const;  // base vocab + added tokens (158237 for the released checkpoint)

 private:
  struct Impl;
  std::shared_ptr<const Impl> impl_;
};

// ---- Prompt (reference_prompt / target_prompt) ---------------------------
inline constexpr int32_t kStart = 151673;
inline constexpr int32_t kTextStart = 151671;
inline constexpr int32_t kSpeechStart = 151669;
inline constexpr int32_t kSpeechEnd = 151670;
inline constexpr int32_t kFinalSeg = 158236;
inline constexpr int32_t kRefSpeechStart = 158237;
inline constexpr int32_t kRefSpeechEnd = 158238;
inline constexpr int32_t kRefTextStart = 158239;
inline constexpr int32_t kRefTextEnd = 158240;
inline constexpr int32_t kAudioIdBase = 151675;
inline constexpr std::string_view kEmotionControl = "{emo: 1}";

std::vector<int32_t> reference_prompt(const std::vector<int32_t>& transcript_ids,
                                      const std::vector<int32_t>& s3_tokens);
std::vector<int32_t> target_prompt(const Tokenizer& tokenizer, std::u32string_view text,
                                   const std::vector<int32_t>& reference_prefix, bool use_icl,
                                   bool emotion);
// Only the part after the voice prefix ({emo: 1} ids, TEXT_START, text ids,
// FINAL_SEG, SPEECH_START): what a cached-prefix prefill feeds.
std::vector<int32_t> target_suffix(const Tokenizer& tokenizer, std::u32string_view text,
                                   bool emotion);

// ---- Phrase planner for text that arrives in pieces (reference behaviour) --
std::size_t realtime_phrase_cut(std::u32string_view text, bool final, int max_characters = 380,
                                int min_sentence_characters = 24);
std::pair<std::size_t, bool> r_turn_cut(std::u32string_view pending, std::size_t committed_upto,
                                        bool input_done, const Settings& settings = {});
std::size_t r_word_safe_cut(std::u32string_view text, std::size_t cut);
// The emotion-safe cut (0 = wait for more input).
std::size_t emotion_safe_cut(std::u32string_view text, std::size_t cut, bool final,
                             std::optional<int> max_characters);

// The emotion decision of a realtime turn (request level).
class EmotionTurn {
 public:
  explicit EmotionTurn(int max_characters = 380) : max_characters_(max_characters) {}
  std::size_t safe_cut(std::u32string_view text, std::size_t cut, bool final) const;
  void commit(std::u32string_view pending);
  std::size_t turn_cut(std::u32string_view pending, std::size_t cut) const;
  void consumed(std::size_t count);
  struct Stamp {
    bool emotion = false;
    std::optional<U32> carry_tag;  // control.emotion_carry_tag (read only with tag carry, off by default)
  };
  Stamp stamp(std::u32string_view text_batch, std::u32string_view pending_after);

 private:
  void reset();
  bool decision(std::u32string_view pending) const;
  int max_characters_;
  bool applied_ = false;
  std::optional<U32> lead_tag_;
  int phrases_ = 0;
  std::vector<std::size_t> turn_ends_;
};

// The sentence batcher (its default turn-batching branch)
// as an explicit state machine.  The host drives it:
//   1. feed() every input event as it arrives (text / commit / done);
//   2. call step(now_ms, slot_free) whenever an event arrived, a wait expired
//      or the generation slot was freed;
//   3. act on the result: a Phrase (start its generation; the slot is busy
//      until it finishes), Wait (call again at ``until_ms`` or on the next
//      event / slot release, whichever first), or Done.
// The socket reader's rules for each client event, BEFORE the planner sees it
// (reference behaviour).  Lengths are code points, as
// Python's len().  Any verdict other than kForward / kSkip ends the session
// with an error event (the reference raises; the session closes with 1011).
//   append: after done -> kInputClosed; "" -> kSkip (no-op); > 8,192 code
//           points -> kMessageTooLarge; the session's appended total (counted
//           BEFORE the check, so the crossing message fails) > 262,144 ->
//           kInputTooLarge; otherwise kForward (whitespace-only is forwarded).
//   commit: after done -> kInputClosed, else kForward.
//   done:   after done -> kInputClosed, else kForward (and the input closes).
class WsInputGate {
 public:
  enum class Verdict { kForward, kSkip, kMessageTooLarge, kInputTooLarge, kInputClosed };
  explicit WsInputGate(const Settings& settings = {}) : settings_(settings) {}
  Verdict append(std::size_t code_points);
  Verdict commit() const { return closed_ ? Verdict::kInputClosed : Verdict::kForward; }
  Verdict done();
  std::size_t input_characters() const { return input_characters_; }

 private:
  Settings settings_;
  bool closed_ = false;
  std::size_t input_characters_ = 0;
};

class WsPlanner {
 public:
  enum class EventType { kText, kCommit, kDone };
  struct Phrase {
    U32 text;
    bool emotion = false;
    std::optional<U32> carry_tag;
    bool continues_turn = false;
  };
  struct Action {
    enum class Kind { kPhrase, kWait, kDone } kind = Kind::kWait;
    Phrase phrase;
    double until_ms = 0;          // kWait: wake-up time (infinity = only on an event)
    bool needs_slot = false;      // kWait: waiting for the generation slot
  };
  explicit WsPlanner(const Settings& settings = {});
  void feed(EventType type, std::u32string_view value = {});
  Action step(double now_ms, bool slot_free);
  const U32& pending() const { return pending_; }

 private:
  struct Queued {
    EventType type;
    U32 value;
  };
  bool apply(const Queued& event);
  Settings settings_;
  EmotionTurn emo_turn_;
  std::vector<Queued> queue_;
  U32 pending_;
  bool input_done_ = false;
  std::size_t committed_upto_ = 0;
  int phrase_index_ = 0;
  bool slot_held_ = false;
  int turn_phrases_ = 0;
  int idle_timeouts_ = 0;
  bool done_ = false;
  // Wait state machine (where the Python coroutine is suspended).
  enum class Mode { kTop, kGrace, kIdle } mode_ = Mode::kTop;
  double deadline_ms_ = 0;
  bool flush_ = false;
};

}  // namespace kt2n::textplan
