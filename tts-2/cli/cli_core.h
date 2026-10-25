// kitten-tts-native command line: argument parsing, bundle validation and the
// JSON run report. CPU only (no CUDA, no engine calls), so it is unit tested
// on any machine (tests/cli/cli_core_test.cc).
#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace kt2n::cli {

// A command-line mistake: printed with a hint to run --help, exit status 2.
class UsageError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

enum class Command {
  kHelp,         // --help, or no arguments
  kVersion,      // --version
  kCheckAssets,  // --check-assets: validate the bundle and exit
  kListVoices,   // --list-voices: print the built-in voices and exit
  kSpeak         // synthesize (and/or build, save or load a cloned voice)
};

// Where voices are kept on the GPU (--voice-cache).
enum class VoiceCacheMode {
  kAll,   // every built-in voice is prepared at startup and stays (default)
  kLru,   // prepared on first use, least recently used unused voice dropped when full
  kNone   // prepared for the requests that use it, dropped when none does
};

// Working-buffer sizing (--memory-profile).
enum class MemoryProfile { kAuto, kDefault, kLow };

// Parsed and validated command line. Optional fields are unset when the option
// was not given; the engine then applies its own defaults.
struct Options {
  Command command = Command::kHelp;

  std::string assets;                  // --assets DIR
  std::optional<std::string> text;     // --text TEXT
  std::optional<std::string> text_file;  // --text-file FILE|-
  std::optional<std::string> voice;    // --voice NAME
  std::string output;                  // --output FILE|- (default output.wav / output.pcm)
  std::string format = "wav";          // --format wav|pcm
  bool stream = true;                  // --stream true|false
  int device = 0;                      // --device N
  int max_streams = 1;                 // --max-streams N
  int repeat = 1;                      // --repeat N (1..100)
  std::optional<int> threads;          // --threads N (text threads)
  std::optional<std::string> data;     // --data DIR (text data override)
  std::optional<std::string> report;   // --report FILE

  // Sampling.
  std::optional<uint64_t> seed;        // --seed N
  std::optional<std::string> preset;   // --preset stable|expressive
  std::optional<double> temperature;   // --temperature F
  std::optional<double> top_p;         // --top-p F
  std::optional<double> min_p;         // --min-p F
  std::optional<int64_t> top_k;        // --top-k N
  std::optional<int64_t> max_tokens;   // --max-tokens N
  bool no_normalize = false;           // --no-normalize

  // Voice cloning and saved voices.
  std::optional<std::string> reference_audio;       // --reference-audio FILE
  std::optional<std::string> reference_transcript;  // --reference-transcript TEXT
  std::optional<std::string> save_voice;            // --save-voice FILE
  std::optional<std::string> load_voice;            // --load-voice FILE

  // Voice cache and memory (kt2n_engine_config voice_cache_mode, voice_cache_max_voices,
  // voice_cache_max_bytes and memory_profile). The defaults are the engine's defaults.
  VoiceCacheMode voice_cache = VoiceCacheMode::kAll;  // --voice-cache all|lru|none
  bool voice_cache_given = false;                     // --voice-cache was on the command line
  std::optional<int> voice_cache_voices;              // --voice-cache-voices N (lru / none; 1..4096)
  std::optional<uint64_t> voice_cache_mib;            // --voice-cache-mib N (lru / none; MiB)
  MemoryProfile memory_profile = MemoryProfile::kAuto;  // --memory-profile auto|default|low

  bool cloning() const { return reference_audio.has_value(); }
  bool has_text() const { return text.has_value() || text_file.has_value(); }
};

// Parses argv[1..] (program name excluded). Throws UsageError for unknown
// options, missing or malformed values, repeated options and conflicting
// combinations. Does not touch the file system.
Options parse_args(const std::vector<std::string>& args);

// The --help text.
std::string help_text();

// ---------------------------------------------------------------------------
// Bundle (a directory prepared by tools/prepare_assets.py: config.json plus
// the files its "native" manifest lists with their sizes and SHA-256).
// ---------------------------------------------------------------------------
enum class CloningCheck {
  kSkip,       // do not look at the cloning files
  kIfPresent,  // validate them when the manifest lists them
  kRequire     // the manifest must list them (voice cloning)
};

struct BundleCheck {
  bool engine_files = true;  // language model, decoder, tokenizer, text data
  bool voices = true;        // voices.kt2n
  CloningCheck cloning = CloningCheck::kSkip;
  bool verify_sha256 = true;
  // With verify_sha256: false = hash every file while it is checked (load_bundle throws on a mismatch); true = check
  // sizes only and list the files with their expected SHA-256 in Bundle::pending_hashes for verify_bundle_hashes().
  bool defer_sha256 = false;
};

struct BundleFile {
  std::string path;  // absolute or relative to the working directory, as opened
  uint64_t size = 0;
};

struct PendingHash {
  std::string path;
  std::string expected;  // 64 lowercase hex characters
  uint64_t size = 0;
};

struct Bundle {
  std::string root;
  std::string default_voice = "Bruno";  // config.json "default_voice"
  std::string model, decoder, voices, tokenizer;
  std::string text_data;        // directory with the normalisation data
  std::string voice_build_dir;  // cloning networks ("" = not checked or not prepared)
  bool has_cloning = false;     // the manifest lists cloning files
  std::vector<BundleFile> checked;  // every file whose size (and hash) was checked
  std::vector<PendingHash> pending_hashes;  // BundleCheck::defer_sha256: hashes not checked yet, in manifest order
};

// Reads DIR/config.json and checks the listed files. Throws std::runtime_error
// naming the offending file on any missing file, size or SHA-256 mismatch, path
// that leaves its directory, or incomplete cloning file set.
Bundle load_bundle(const std::string& dir, const BundleCheck& check);

// Checks the SHA-256 of every file in bundle.pending_hashes, several files at once in up to `threads` threads.
// Throws the error load_bundle would have thrown ("asset SHA256 mismatch: FILE", "cannot read asset: FILE") for the
// first failing file in manifest order.
void verify_bundle_hashes(const Bundle& bundle, int threads);

// Throws std::runtime_error unless `path` is a relative path of plain names
// (letters, digits, '-', '_', '.', '/'), with no "." or ".." component, no
// leading or trailing '/' and no empty component.
void check_relative_path(const std::string& path);

// ---------------------------------------------------------------------------
// Voices (read from voices.kt2n without starting the engine).
// ---------------------------------------------------------------------------
struct VoiceEntry {
  std::string name;                 // the id accepted by --voice
  std::vector<std::string> aliases; // other names accepted by --voice
  std::string language;
};

// The public built-in voices in catalogue order (hidden entries skipped).
std::vector<VoiceEntry> list_voices(const std::string& voices_file);

// ---------------------------------------------------------------------------
// Run report (one JSON object per synthesized output).
// ---------------------------------------------------------------------------
struct EngineCounters {
  bool present = false;
  std::string output;        // "stream" or "complete", from the engine's response headers ("" = none arrived)
  int64_t segments = 0;      // text segments generated (end-of-request metrics)
  int64_t audio_tokens = 0;
  int64_t lm_steps = 0;
  uint64_t renders = 0;      // decoder render calls during this request
  uint64_t render_rows = 0;  // segment rows over those calls
};

struct RunReport {
  int device = 0;
  std::string voice;
  std::string format;
  bool stream = true;
  uint64_t bytes = 0;
  double first_audio_ms = -1;  // -1 = no audio
  double elapsed_seconds = 0;
  int iteration = 1;
  int64_t samples = 0;         // 24 kHz samples delivered
  std::optional<uint64_t> seed;
  EngineCounters engine;
};

// Compact JSON. Keys, in order: backend, device, voice, sample_rate, format,
// stream, bytes, first_audio_ms, elapsed_seconds, iteration, audio_seconds,
// rtf (elapsed / audio seconds; null without audio), seed (when set) and
// engine {output, segments, audio_tokens, lm_steps, renders, render_rows}; output is null when no
// response headers arrived.
std::string build_report(const RunReport& report);

// Reads a KT2N_EVENT_HEADERS fields_json object. The engine adds complete_bytes only to complete
// (KT2N_OUTPUT_COMPLETE) responses, so `counters.output` becomes "complete" when it is present and
// "stream" otherwise. Malformed input is ignored.
void read_headers(const std::string& fields_json, EngineCounters& counters);

// Reads the integer fields of a KT2N_EVENT_METRICS fields_json object into
// `counters` (segments, audio_tokens, lm_steps). Malformed input is ignored.
void read_metrics(const std::string& fields_json, EngineCounters& counters);

// The canonical 44-byte PCM16 mono 24 kHz WAV header for `data_bytes` bytes.
std::string wav_header(uint32_t data_bytes);

}  // namespace kt2n::cli
