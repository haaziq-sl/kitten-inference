// kitten-tts-native core: argument parsing, bundle checks, report and WAV header.
#include "cli_core.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "json.h"
#include "kt2n/sha256.h"

#if defined(_WIN32)
#include "kt2n/posix_compat.h"
#else
#include <unistd.h>
#endif

namespace {

namespace fs = std::filesystem;
using kt2n::cli::Command;
using kt2n::cli::Options;
using kt2n::cli::parse_args;
using kt2n::cli::UsageError;
using Args = std::vector<std::string>;

Args with_base(Args extra) {
  Args args = {"--assets", "models/kitten", "--text", "Hello there."};
  args.insert(args.end(), extra.begin(), extra.end());
  return args;
}

// The UsageError message, or "" if the arguments parse.
std::string usage_error(const Args& args) {
  try {
    parse_args(args);
  } catch (const UsageError& e) {
    return e.what();
  }
  return "";
}

#define EXPECT_USAGE(args, fragment)                                                  \
  do {                                                                                \
    const std::string message_ = usage_error(args);                                   \
    EXPECT_NE(message_.find(fragment), std::string::npos) << "message: " << message_; \
  } while (0)

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------
TEST(CliArgs, HelpAndVersion) {
  EXPECT_EQ(parse_args({}).command, Command::kHelp);
  EXPECT_EQ(parse_args({"--help"}).command, Command::kHelp);
  EXPECT_EQ(parse_args({"-h"}).command, Command::kHelp);
  EXPECT_EQ(parse_args(with_base({"--help"})).command, Command::kHelp);
  EXPECT_EQ(parse_args({"--version"}).command, Command::kVersion);
  EXPECT_NE(kt2n::cli::help_text().find("--list-voices"), std::string::npos);
}

TEST(CliArgs, SpeakDefaults) {
  const Options o = parse_args(with_base({}));
  EXPECT_EQ(o.command, Command::kSpeak);
  EXPECT_EQ(o.assets, "models/kitten");
  EXPECT_EQ(o.text.value_or(""), "Hello there.");
  EXPECT_FALSE(o.voice);
  EXPECT_EQ(o.format, "wav");
  EXPECT_EQ(o.output, "output.wav");
  EXPECT_TRUE(o.stream);
  EXPECT_EQ(o.device, 0);
  EXPECT_EQ(o.max_streams, 1);
  EXPECT_EQ(o.repeat, 1);
  EXPECT_FALSE(o.seed);
  EXPECT_FALSE(o.cloning());
  EXPECT_EQ(parse_args(with_base({"--format", "pcm"})).output, "output.pcm");
}

TEST(CliArgs, CheckAssetsAndListVoices) {
  EXPECT_EQ(parse_args({"--assets", "m", "--check-assets"}).command, Command::kCheckAssets);
  EXPECT_EQ(parse_args({"--assets", "m", "--list-voices"}).command, Command::kListVoices);
  EXPECT_USAGE(Args({"--check-assets"}), "--assets DIR is required");
  EXPECT_USAGE(Args({"--assets", "m", "--check-assets", "--list-voices"}), "cannot be combined");
  EXPECT_USAGE(Args({"--assets", "m", "--list-voices", "--text", "x"}), "--list-voices cannot be combined with --text");
  EXPECT_USAGE(Args({"--assets", "m", "--list-voices", "--load-voice", "v"}), "--list-voices cannot be combined");
  // The reference options are still checked with --check-assets.
  EXPECT_USAGE(Args({"--assets", "m", "--check-assets", "--reference-audio", "r.wav"}), "--reference-transcript is required");
  const Options o = parse_args({"--assets", "m", "--check-assets", "--reference-audio", "r.wav", "--reference-transcript", "Hi."});
  EXPECT_EQ(o.command, Command::kCheckAssets);
  EXPECT_TRUE(o.cloning());
}

// ---------------------------------------------------------------------------
// Option syntax
// ---------------------------------------------------------------------------
TEST(CliArgs, UnknownOptions) {
  EXPECT_USAGE(with_base({"--reference-language", "en"}), "unknown option --reference-language");
  for (const char* dropped : {"--backend", "--repo", "--revision", "--cache-dir", "--offline", "--download-only", "--decoder"})
    EXPECT_USAGE(with_base({dropped, "x"}), std::string("unknown option ") + dropped);
  EXPECT_USAGE(with_base({"stray"}), "unexpected argument 'stray'");
  EXPECT_USAGE(with_base({"-v"}), "unexpected argument '-v'");
}

TEST(CliArgs, ValuesAndRepeats) {
  EXPECT_USAGE(Args({"--assets"}), "--assets needs a value");
  EXPECT_USAGE(with_base({"--voice"}), "--voice needs a value");
  EXPECT_USAGE(with_base({"--voice", "Bruno", "--voice", "Luna"}), "--voice was given more than once");
  EXPECT_USAGE(with_base({"--no-normalize=1"}), "--no-normalize does not take a value");
  const Options o = parse_args({"--assets=m", "--text=a=b", "--voice=Luna", "--no-normalize"});
  EXPECT_EQ(o.assets, "m");
  EXPECT_EQ(o.text.value_or(""), "a=b");
  EXPECT_EQ(o.voice.value_or(""), "Luna");
  EXPECT_TRUE(o.no_normalize);
  EXPECT_USAGE(with_base({"--voice", ""}), "--voice needs a voice name");
}

TEST(CliArgs, TextSources) {
  EXPECT_USAGE(Args({"--assets", "m"}), "--text or --text-file is required");
  EXPECT_USAGE(Args({"--assets", "m", "--voice", "Bruno"}), "--text or --text-file is required");
  EXPECT_USAGE(with_base({"--text-file", "t.txt"}), "--text cannot be combined with --text-file");
  EXPECT_USAGE(Args({"--assets", "m", "--text-file", ""}), "--text-file needs a file name");
  EXPECT_EQ(parse_args({"--assets", "m", "--text-file", "-"}).text_file.value_or(""), "-");
  EXPECT_TRUE(parse_args({"--assets", "m", "--text-file", "t.txt"}).has_text());
}

// ---------------------------------------------------------------------------
// Output options
// ---------------------------------------------------------------------------
TEST(CliArgs, StreamValues) {
  EXPECT_TRUE(parse_args(with_base({"--stream", "true"})).stream);
  EXPECT_FALSE(parse_args(with_base({"--stream", "false"})).stream);
  for (const char* value : {"0", "1", "False", "TRUE", "no", "yes", ""})
    EXPECT_USAGE(with_base({"--stream", value}), "--stream must be true or false");
}

TEST(CliArgs, FormatOutputRepeat) {
  EXPECT_USAGE(with_base({"--format", "mp3"}), "--format must be wav or pcm");
  EXPECT_USAGE(with_base({"--output", "-", "--repeat", "2"}), "--repeat cannot be combined with --output -");
  EXPECT_EQ(parse_args(with_base({"--output", "-"})).output, "-");
  EXPECT_EQ(parse_args(with_base({"--repeat", "100"})).repeat, 100);
  for (const char* value : {"0", "101", "-1", "2x", "", " 3"})
    EXPECT_USAGE(with_base({"--repeat", value}), "invalid value for --repeat");
  EXPECT_USAGE(with_base({"--output", ""}), "--output needs a file name");
}

TEST(CliArgs, Numbers) {
  const Options o = parse_args(with_base({"--device", "1", "--max-streams", "4", "--threads", "2", "--seed",
                                          "18446744073709551615", "--temperature", "0.7", "--top-p", "0.9",
                                          "--min-p", "0.05", "--top-k", "0", "--max-tokens", "1500", "--preset",
                                          "expressive"}));
  EXPECT_EQ(o.device, 1);
  EXPECT_EQ(o.max_streams, 4);
  EXPECT_EQ(o.threads.value_or(0), 2);
  EXPECT_EQ(o.seed.value_or(0), 18446744073709551615ull);
  EXPECT_DOUBLE_EQ(o.temperature.value_or(0), 0.7);
  EXPECT_DOUBLE_EQ(o.top_p.value_or(0), 0.9);
  EXPECT_DOUBLE_EQ(o.min_p.value_or(0), 0.05);
  EXPECT_EQ(o.top_k.value_or(-1), 0);
  EXPECT_EQ(o.max_tokens.value_or(0), 1500);
  EXPECT_EQ(o.preset.value_or(""), "expressive");
  EXPECT_USAGE(with_base({"--seed", "-1"}), "invalid value for --seed");
  EXPECT_USAGE(with_base({"--seed", "18446744073709551616"}), "invalid value for --seed");
  EXPECT_USAGE(with_base({"--device", "-1"}), "invalid value for --device");
  EXPECT_USAGE(with_base({"--max-streams", "0"}), "invalid value for --max-streams");
  EXPECT_USAGE(with_base({"--threads", "0"}), "invalid value for --threads");
  EXPECT_USAGE(with_base({"--top-k", "-1"}), "invalid value for --top-k");
  EXPECT_USAGE(with_base({"--max-tokens", "0"}), "invalid value for --max-tokens");
  EXPECT_USAGE(with_base({"--temperature", "nan"}), "invalid value for --temperature");
  EXPECT_USAGE(with_base({"--temperature", "inf"}), "invalid value for --temperature");
  EXPECT_USAGE(with_base({"--top-p", "0.9x"}), "invalid value for --top-p");
  EXPECT_USAGE(with_base({"--preset", "fast"}), "--preset must be stable or expressive");
  // One leading '+' is accepted, as before.
  const Options plus = parse_args(with_base({"--temperature", "+0.8", "--max-tokens", "+600", "--seed", "+5"}));
  EXPECT_DOUBLE_EQ(plus.temperature.value_or(0), 0.8);
  EXPECT_EQ(plus.max_tokens.value_or(0), 600);
  EXPECT_EQ(plus.seed.value_or(0), 5u);
  for (const char* value : {"+", "++5", "+-5", "+ 5"})
    EXPECT_USAGE(with_base({"--seed", value}), "invalid value for --seed");
  EXPECT_USAGE(with_base({"--temperature", "+-0.8"}), "invalid value for --temperature");
}

// ---------------------------------------------------------------------------
// Cloning and saved voices
// ---------------------------------------------------------------------------
TEST(CliArgs, ReferenceCombinations) {
  const Args reference = {"--reference-audio", "reference.wav", "--reference-transcript", "A reference recording."};
  const Options o = parse_args(with_base(reference));
  EXPECT_TRUE(o.cloning());
  EXPECT_EQ(o.reference_transcript.value_or(""), "A reference recording.");
  Args with_voice = reference;
  with_voice.insert(with_voice.end(), {"--voice", "Bruno"});
  EXPECT_USAGE(with_base(with_voice), "--reference-audio cannot be combined with --voice");
  EXPECT_USAGE(with_base({"--reference-audio", "reference.wav"}), "--reference-transcript is required");
  EXPECT_USAGE(with_base({"--reference-audio", "reference.wav", "--reference-transcript", ""}),
               "--reference-transcript is required");
  EXPECT_USAGE(with_base({"--reference-transcript", "Reference."}), "--reference-transcript requires --reference-audio");
  EXPECT_USAGE(with_base({"--reference-audio", "", "--reference-transcript", "x"}), "--reference-audio needs a file name");
}

TEST(CliArgs, SaveVoice) {
  EXPECT_USAGE(with_base({"--save-voice", "me.voice"}), "--save-voice requires --reference-audio");
  EXPECT_USAGE(Args({"--assets", "m", "--save-voice", "me.voice"}), "--save-voice requires --reference-audio");
  // Building and saving a voice needs no text.
  const Options o = parse_args({"--assets", "m", "--reference-audio", "r.wav", "--reference-transcript", "Hi.",
                                "--save-voice", "me.voice"});
  EXPECT_EQ(o.command, Command::kSpeak);
  EXPECT_FALSE(o.has_text());
  EXPECT_EQ(o.save_voice.value_or(""), "me.voice");
  EXPECT_USAGE(Args({"--assets", "m", "--reference-audio", "r.wav", "--reference-transcript", "Hi.", "--save-voice", ""}),
               "--save-voice needs a file name");
}

TEST(CliArgs, LoadVoice) {
  const Options o = parse_args(with_base({"--load-voice", "me.voice"}));
  EXPECT_EQ(o.load_voice.value_or(""), "me.voice");
  EXPECT_FALSE(o.cloning());
  EXPECT_USAGE(with_base({"--load-voice", "me.voice", "--voice", "Bruno"}), "--load-voice cannot be combined with --voice");
  EXPECT_USAGE(with_base({"--load-voice", "me.voice", "--reference-audio", "r.wav", "--reference-transcript", "Hi."}),
               "--load-voice cannot be combined with --reference-audio");
  EXPECT_USAGE(Args({"--assets", "m", "--load-voice", "me.voice"}), "--text or --text-file is required");
  EXPECT_USAGE(with_base({"--load-voice", ""}), "--load-voice needs a file name");
  EXPECT_USAGE(with_base({"--load-voice", "a.voice", "--save-voice", "b.voice"}), "--save-voice requires --reference-audio");
}

// ---------------------------------------------------------------------------
// Voice cache and memory
// ---------------------------------------------------------------------------
using kt2n::cli::MemoryProfile;
using kt2n::cli::VoiceCacheMode;

TEST(CliArgs, VoiceCacheAndMemoryDefaults) {
  // Without the options the engine keeps its defaults: every voice resident, default buffers, no limits.
  const Options o = parse_args(with_base({}));
  EXPECT_EQ(o.voice_cache, VoiceCacheMode::kAll);
  EXPECT_FALSE(o.voice_cache_given);  // kitten_tts_native.cc then uses lru with one voice per stream for a built-in voice
  EXPECT_FALSE(o.voice_cache_voices);
  EXPECT_FALSE(o.voice_cache_mib);
  EXPECT_EQ(o.memory_profile, MemoryProfile::kAuto);  // the default profile when it fits, otherwise low
  const std::string help = kt2n::cli::help_text();
  for (const char* option : {"--voice-cache all|lru|none", "--voice-cache-voices N", "--voice-cache-mib N",
                             "--memory-profile auto|default|low"})
    EXPECT_NE(help.find(option), std::string::npos) << option;
}

TEST(CliArgs, VoiceCacheModes) {
  EXPECT_EQ(parse_args(with_base({"--voice-cache", "all"})).voice_cache, VoiceCacheMode::kAll);
  EXPECT_TRUE(parse_args(with_base({"--voice-cache", "all"})).voice_cache_given);
  EXPECT_EQ(parse_args(with_base({"--voice-cache", "lru"})).voice_cache, VoiceCacheMode::kLru);
  EXPECT_EQ(parse_args(with_base({"--voice-cache=none"})).voice_cache, VoiceCacheMode::kNone);
  for (const char* value : {"", "LRU", "All", "off", "0", "lru "})
    EXPECT_USAGE(with_base({"--voice-cache", value}), "--voice-cache must be all, lru or none");
  EXPECT_USAGE(with_base({"--voice-cache"}), "--voice-cache needs a value");
  EXPECT_USAGE(with_base({"--voice-cache", "lru", "--voice-cache", "none"}), "--voice-cache was given more than once");
}

TEST(CliArgs, VoiceCacheLimits) {
  const Options both = parse_args(with_base({"--voice-cache", "lru", "--voice-cache-voices", "4", "--voice-cache-mib", "900"}));
  EXPECT_EQ(both.voice_cache_voices.value_or(0), 4);
  EXPECT_EQ(both.voice_cache_mib.value_or(0), 900u);
  const Options none = parse_args(with_base({"--voice-cache", "none", "--voice-cache-voices", "1"}));
  EXPECT_EQ(none.voice_cache, VoiceCacheMode::kNone);
  EXPECT_EQ(none.voice_cache_voices.value_or(0), 1);
  EXPECT_FALSE(none.voice_cache_mib);
  const Options mib_only = parse_args(with_base({"--voice-cache", "lru", "--voice-cache-mib", "+512"}));
  EXPECT_FALSE(mib_only.voice_cache_voices);
  EXPECT_EQ(mib_only.voice_cache_mib.value_or(0), 512u);
  EXPECT_EQ(parse_args(with_base({"--voice-cache", "lru", "--voice-cache-voices", "4096"})).voice_cache_voices.value_or(0),
            4096);
  // The largest MiB value whose byte count fits in 64 bits.
  EXPECT_EQ(parse_args(with_base({"--voice-cache", "lru", "--voice-cache-mib", "17592186044415"})).voice_cache_mib.value_or(0),
            17592186044415ull);
  for (const char* value : {"0", "4097", "-1", "", "2x", " 3"})
    EXPECT_USAGE(with_base({"--voice-cache", "lru", "--voice-cache-voices", value}), "invalid value for --voice-cache-voices");
  for (const char* value : {"0", "-1", "", "1.5", "17592186044416"})
    EXPECT_USAGE(with_base({"--voice-cache", "lru", "--voice-cache-mib", value}), "invalid value for --voice-cache-mib");
  // The limits apply only to the lru and none caches; all (the default) keeps every voice.
  EXPECT_USAGE(with_base({"--voice-cache-voices", "4"}), "--voice-cache-voices requires --voice-cache lru or none");
  EXPECT_USAGE(with_base({"--voice-cache", "all", "--voice-cache-mib", "900"}),
               "--voice-cache-mib requires --voice-cache lru or none");
}

TEST(CliArgs, MemoryProfile) {
  EXPECT_EQ(parse_args(with_base({"--memory-profile", "auto"})).memory_profile, MemoryProfile::kAuto);
  EXPECT_EQ(parse_args(with_base({"--memory-profile", "default"})).memory_profile, MemoryProfile::kDefault);
  EXPECT_EQ(parse_args(with_base({"--memory-profile", "low"})).memory_profile, MemoryProfile::kLow);
  for (const char* value : {"", "LOW", "small", "1", "Auto"})
    EXPECT_USAGE(with_base({"--memory-profile", value}), "--memory-profile must be auto, default or low");
  // Independent of the voice cache.
  const Options o = parse_args(with_base({"--memory-profile", "low", "--voice-cache", "none"}));
  EXPECT_EQ(o.memory_profile, MemoryProfile::kLow);
  EXPECT_EQ(o.voice_cache, VoiceCacheMode::kNone);
}

// ---------------------------------------------------------------------------
// Bundle checks
// ---------------------------------------------------------------------------
class CliBundle : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    root_ = fs::temp_directory_path() / ("kt2n-cli-test-" + std::to_string(::getpid()) + "-" + info->name());
    fs::remove_all(root_);
    fs::create_directories(root_);
  }
  void TearDown() override { fs::remove_all(root_); }

  std::string entry(const std::string& name, const std::string& data = "test") {
    const fs::path path = root_ / name;
    fs::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary) << data;
    return R"({"file":")" + name + R"(","size":)" + std::to_string(data.size()) + R"(,"sha256":")" +
           kt2n::sha256_hex(data.data(), data.size()) + R"("})";
  }

  std::string manifest(const std::string& lm, const std::string& text_dir = "native/text",
                       const std::string& cloning = "") {
    return R"({"type":"KITTEN2","default_voice":"Luna","native":{"version":1,"decoder":"default","lm":)" + lm +
           R"(,"acoustic":)" + entry("native/acoustic") + R"(,"voices":)" + entry("native/voices") +
           R"(,"tokenizer":)" + entry("native/tokenizer") + R"(,"text":{"directory":")" + text_dir +
           R"(","files":[)" + entry("native/text/rules") + "]}" + cloning + "}}";
  }

  std::string cloning(const std::string& directory, size_t files = 3) {
    const char* names[] = {"s3gen.safetensors", "voicebuild_consts.safetensors", "voicebuild_pyannote.safetensors"};
    std::string list;
    for (size_t i = 0; i < files; ++i) list += (i ? "," : "") + entry("native/cloning/" + std::string(names[i]));
    return R"(,"cloning":{"version":1,"directory":")" + directory + R"(","files":[)" + list + "]}";
  }

  void write_config(const std::string& text) { std::ofstream(root_ / "config.json") << text; }

  std::string error(const kt2n::cli::BundleCheck& check = {}) {
    try {
      kt2n::cli::load_bundle(root_.string(), check);
    } catch (const std::exception& e) {
      return e.what();
    }
    return "";
  }

  fs::path root_;
};

TEST_F(CliBundle, ValidBundle) {
  write_config(manifest(entry("native/lm")));
  const auto b = kt2n::cli::load_bundle(root_.string(), {});
  EXPECT_EQ(b.model, (root_ / "native/lm").string());
  EXPECT_EQ(b.decoder, (root_ / "native/acoustic").string());
  EXPECT_EQ(b.voices, (root_ / "native/voices").string());
  EXPECT_EQ(b.tokenizer, (root_ / "native/tokenizer").string());
  EXPECT_EQ(b.text_data, (root_ / "native/text").string());
  EXPECT_EQ(b.default_voice, "Luna");
  EXPECT_FALSE(b.has_cloning);
  EXPECT_EQ(b.checked.size(), 5u);
}

TEST_F(CliBundle, SizeAndShaMismatch) {
  write_config(manifest(entry("native/lm")));
  std::ofstream(root_ / "native/lm", std::ios::binary) << "evil";
  EXPECT_NE(error().find("SHA256 mismatch"), std::string::npos);
  kt2n::cli::BundleCheck no_hash;
  no_hash.verify_sha256 = false;
  EXPECT_EQ(error(no_hash), "");
  std::ofstream(root_ / "native/lm", std::ios::binary) << "longer";
  EXPECT_NE(error().find("size mismatch"), std::string::npos);
  fs::remove(root_ / "native/lm");
  EXPECT_NE(error().find("missing asset"), std::string::npos);
}

// defer_sha256: load_bundle checks sizes only and lists the hashes; verify_bundle_hashes reports the first mismatch in
// manifest order with load_bundle's message, from any number of threads.
TEST_F(CliBundle, DeferredShaCheck) {
  write_config(manifest(entry("native/lm")));
  kt2n::cli::BundleCheck deferred;
  deferred.defer_sha256 = true;
  const auto good = kt2n::cli::load_bundle(root_.string(), deferred);
  EXPECT_EQ(good.pending_hashes.size(), 5u);
  for (int threads : {1, 2, 8}) EXPECT_NO_THROW(kt2n::cli::verify_bundle_hashes(good, threads));
  std::ofstream(root_ / "native/voices", std::ios::binary) << "evil";
  std::ofstream(root_ / "native/acoustic", std::ios::binary) << "evil";
  EXPECT_EQ(error(deferred), "");  // same sizes: nothing hashed yet
  const auto bad = kt2n::cli::load_bundle(root_.string(), deferred);
  for (int threads : {1, 2, 8}) {
    try {
      kt2n::cli::verify_bundle_hashes(bad, threads);
      ADD_FAILURE() << "no mismatch reported";
    } catch (const std::exception& e) {
      EXPECT_EQ(std::string(e.what()), "asset SHA256 mismatch: " + (root_ / "native/acoustic").string());
    }
  }
  EXPECT_EQ(error(), "asset SHA256 mismatch: " + (root_ / "native/acoustic").string());  // the same as undeferred
  std::ofstream(root_ / "native/lm", std::ios::binary) << "longer";
  EXPECT_NE(error(deferred).find("size mismatch"), std::string::npos);
}

TEST_F(CliBundle, PathEscapes) {
  entry("native/lm");
  write_config(manifest(R"({"file":"../escape","size":4,"sha256":")" + kt2n::sha256_hex("test", 4) + R"("})"));
  EXPECT_NE(error().find("invalid path"), std::string::npos);
  write_config(manifest(entry("native/lm"), "elsewhere"));
  EXPECT_NE(error().find("outside its directory"), std::string::npos);
  for (const char* bad : {"/abs", "a/", "a//b", "a/./b", "a/../b", "..", ".", "a b", "a\\b"}) {
    EXPECT_THROW(kt2n::cli::check_relative_path(bad), std::runtime_error) << bad;
  }
  EXPECT_NO_THROW(kt2n::cli::check_relative_path("native/text/data_words.txt"));
}

TEST_F(CliBundle, Cloning) {
  kt2n::cli::BundleCheck require;
  require.cloning = kt2n::cli::CloningCheck::kRequire;
  write_config(manifest(entry("native/lm")));
  EXPECT_NE(error(require).find("no voice cloning files"), std::string::npos);
  write_config(manifest(entry("native/lm"), "native/text", cloning("native/cloning")));
  const auto b = kt2n::cli::load_bundle(root_.string(), require);
  EXPECT_TRUE(b.has_cloning);
  EXPECT_EQ(b.voice_build_dir, (root_ / "native/cloning").string());
  write_config(manifest(entry("native/lm"), "native/text", cloning("native/cloning", 1)));
  EXPECT_NE(error(require).find("missing cloning asset: voicebuild_consts.safetensors"), std::string::npos);
  EXPECT_EQ(error(), "");  // not looked at without cloning
  kt2n::cli::BundleCheck if_present;
  if_present.cloning = kt2n::cli::CloningCheck::kIfPresent;
  EXPECT_NE(error(if_present).find("missing cloning asset"), std::string::npos);
  write_config(manifest(entry("native/lm"), "native/text", cloning("elsewhere")));
  EXPECT_NE(error(require).find("outside its directory"), std::string::npos);
}

TEST_F(CliBundle, ConfigErrors) {
  EXPECT_NE(error().find("missing asset"), std::string::npos);
  write_config("{");
  EXPECT_NE(error().find("not valid JSON"), std::string::npos);
  write_config(R"({"type":"KITTEN1"})");
  EXPECT_NE(error().find("KITTEN2"), std::string::npos);
  write_config(R"({"type":"KITTEN2"})");
  EXPECT_NE(error().find("prepare_assets.py"), std::string::npos);
  write_config(R"({"type":"KITTEN2","native":{"version":2}})");
  EXPECT_NE(error().find("unsupported asset manifest"), std::string::npos);
  write_config(R"({"type":"KITTEN2","native":{"version":1,"decoder":"student_w4"}})");
  EXPECT_NE(error().find("unsupported asset manifest"), std::string::npos);
  write_config(R"({"type":"KITTEN2","native":{"version":1}})");
  EXPECT_NE(error().find("has no \"lm\""), std::string::npos);
  try {
    kt2n::cli::load_bundle((root_ / "missing").string(), {});
    ADD_FAILURE() << "a missing directory was accepted";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("assets directory not found"), std::string::npos);
  }
}

// ---------------------------------------------------------------------------
// Report and WAV header
// ---------------------------------------------------------------------------
TEST(CliReport, KeysAndValues) {
  kt2n::cli::RunReport r;
  r.device = 0;
  r.voice = "Bruno";
  r.format = "wav";
  r.stream = false;
  r.bytes = 96044;
  r.first_audio_ms = 812.5;
  r.elapsed_seconds = 1.0;
  r.iteration = 2;
  r.samples = 48000;
  r.seed = 7;
  r.engine.present = true;
  r.engine.output = "complete";
  r.engine.segments = 2;
  r.engine.renders = 1;
  r.engine.render_rows = 2;
  const auto v = kt2n::json::parse(kt2n::cli::build_report(r));
  std::vector<std::string> keys;
  for (const auto& member : v.members()) keys.push_back(member.first);
  const std::vector<std::string> expected = {"backend", "device", "voice", "sample_rate", "format", "stream", "bytes",
                                             "first_audio_ms", "elapsed_seconds", "iteration", "audio_seconds", "rtf",
                                             "seed", "engine"};
  EXPECT_EQ(keys, expected);
#if defined(KT2N_HIP)
  EXPECT_EQ(v.get("backend")->as_string(), "hip");
#else
  EXPECT_EQ(v.get("backend")->as_string(), "cuda");
#endif
  EXPECT_EQ(v.get("sample_rate")->as_int(), 24000);
  EXPECT_FALSE(v.get("stream")->as_bool());
  EXPECT_EQ(v.get("bytes")->as_int(), 96044);
  EXPECT_DOUBLE_EQ(v.get("audio_seconds")->as_double(), 2.0);
  EXPECT_DOUBLE_EQ(v.get("rtf")->as_double(), 0.5);
  EXPECT_EQ(v.get("seed")->as_int(), 7);
  EXPECT_EQ(v.at_pointer("/engine/output")->as_string(), "complete");
  EXPECT_EQ(v.at_pointer("/engine/segments")->as_int(), 2);
  EXPECT_EQ(v.at_pointer("/engine/render_rows")->as_int(), 2);
}

TEST(CliReport, NoAudio) {
  kt2n::cli::RunReport r;
  r.voice = "Bruno";
  r.format = "pcm";
  const auto v = kt2n::json::parse(kt2n::cli::build_report(r));
  EXPECT_TRUE(v.get("rtf")->is_null());
  EXPECT_EQ(v.get("first_audio_ms")->as_double(), -1.0);
  EXPECT_EQ(v.get("seed"), nullptr);
  EXPECT_EQ(v.get("engine"), nullptr);
  r.elapsed_seconds = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(kt2n::json::parse(kt2n::cli::build_report(r)).get("elapsed_seconds")->is_null());
  r.seed = 18446744073709551615ull;
  EXPECT_EQ(kt2n::json::parse(kt2n::cli::build_report(r)).get("seed")->as_string(), "18446744073709551615");
}

TEST(CliReport, Metrics) {
  kt2n::cli::EngineCounters c;
  kt2n::cli::read_metrics("not json", c);
  EXPECT_FALSE(c.present);
  kt2n::cli::read_metrics(R"({"producer_seconds":1.5,"audio_tokens":120,"lm_steps":130,"segments":3})", c);
  EXPECT_TRUE(c.present);
  EXPECT_EQ(c.segments, 3);
  EXPECT_EQ(c.audio_tokens, 120);
  EXPECT_EQ(c.lm_steps, 130);
}

TEST(CliReport, Headers) {
  kt2n::cli::EngineCounters c;
  kt2n::cli::read_headers("not json", c);
  EXPECT_EQ(c.output, "");
  kt2n::cli::read_headers(R"({"sample_rate":24000,"format":"pcm","speech_mode":null})", c);
  EXPECT_EQ(c.output, "stream");
  kt2n::cli::read_headers(R"({"sample_rate":24000,"format":"wav","speech_mode":"demo","complete_bytes":96044})", c);
  EXPECT_EQ(c.output, "complete");
  kt2n::cli::RunReport r;
  r.engine.present = true;  // metrics but no headers: output is null
  EXPECT_TRUE(kt2n::json::parse(kt2n::cli::build_report(r)).at_pointer("/engine/output")->is_null());
}

TEST(CliReport, WavHeader) {
  const std::string h = kt2n::cli::wav_header(4800);
  ASSERT_EQ(h.size(), 44u);
  const unsigned char expected[44] = {'R', 'I', 'F', 'F', 0xe4, 0x12, 0, 0, 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ',
                                      16,  0,   0,   0,   1,    0,    1, 0, 0xc0, 0x5d, 0, 0, 0x80, 0xbb, 0, 0,
                                      2,   0,   16,  0,   'd',  'a',  't', 'a', 0xc0, 0x12, 0, 0};
  EXPECT_EQ(h, std::string(reinterpret_cast<const char*>(expected), 44));
}

}  // namespace
