#include "cli_core.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

#include "json.h"
#include "kt2n/sha256.h"
#include "voice_blob.h"

namespace kt2n::cli {
namespace {

namespace fs = std::filesystem;
using json::Value;

// ---------------------------------------------------------------------------
// Option table
// ---------------------------------------------------------------------------
struct OptionSpec {
  const char* name;
  bool takes_value;
};

// Every option the command accepts.
constexpr OptionSpec kOptionTable[] = {
    {"--help", false},
    {"--version", false},
    {"--check-assets", false},
    {"--list-voices", false},
    {"--no-normalize", false},
    {"--assets", true},
    {"--text", true},
    {"--text-file", true},
    {"--voice", true},
    {"--output", true},
    {"--format", true},
    {"--stream", true},
    {"--device", true},
    {"--max-streams", true},
    {"--repeat", true},
    {"--threads", true},
    {"--data", true},
    {"--report", true},
    {"--seed", true},
    {"--preset", true},
    {"--temperature", true},
    {"--top-p", true},
    {"--min-p", true},
    {"--top-k", true},
    {"--max-tokens", true},
    {"--reference-audio", true},
    {"--reference-transcript", true},
    {"--save-voice", true},
    {"--load-voice", true},
    {"--voice-cache", true},
    {"--voice-cache-voices", true},
    {"--voice-cache-mib", true},
    {"--memory-profile", true},
};

const OptionSpec* find_option(const std::string& name) {
  for (const auto& spec : kOptionTable)
    if (name == spec.name) return &spec;
  return nullptr;
}

// One leading '+' is accepted, as std::stoi and std::stod accept it ("+0.8", "+600").
const char* skip_plus(const std::string& text) {
  return text.size() > 1 && text[0] == '+' && text[1] != '+' && text[1] != '-' ? text.data() + 1 : text.data();
}

template <class Int>
Int parse_int(const std::string& text, const char* option, Int minimum, Int maximum) {
  Int value{};
  const char* first = skip_plus(text);
  const char* last = text.data() + text.size();
  const auto [end, ec] = std::from_chars(first, last, value);
  if (text.empty() || ec != std::errc() || end != last || value < minimum || value > maximum)
    throw UsageError(std::string("invalid value for ") + option + ": '" + text + "'");
  return value;
}

double parse_real(const std::string& text, const char* option) {
  double value = 0;
  const char* first = skip_plus(text);
  const char* last = text.data() + text.size();
  const auto [end, ec] = std::from_chars(first, last, value);
  if (text.empty() || ec != std::errc() || end != last || !std::isfinite(value))
    throw UsageError(std::string("invalid value for ") + option + ": '" + text + "'");
  return value;
}

// ---------------------------------------------------------------------------
// Manifest helpers
// ---------------------------------------------------------------------------
[[noreturn]] void bundle_error(const std::string& what) { throw std::runtime_error(what); }

// `where` is taken by value so the returned references (into `object`) are not tied to a temporary.
const Value& member(const Value& object, const char* key, std::string_view where) {
  const Value* v = object.is_object() ? object.get(key) : nullptr;
  if (!v) bundle_error("config.json: " + std::string(where) + " has no \"" + key + "\"");
  return *v;
}

const std::string& string_member(const Value& object, const char* key, std::string_view where) {
  const Value& v = member(object, key, where);
  if (!v.is_string()) bundle_error("config.json: " + std::string(where) + "." + key + " must be a string");
  return v.as_string();
}

bool is_version_one(const Value& object) {
  const Value* v = object.get("version");
  return v && v->is_int() && v->as_int() == 1;
}

std::string read_file(const fs::path& path, const char* what) {
  std::ifstream in(path, std::ios::binary);
  if (!in) bundle_error(std::string("cannot read ") + what + ": " + path.string());
  std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (in.bad()) bundle_error(std::string("cannot read ") + what + ": " + path.string());
  return data;
}

std::string file_sha256(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) bundle_error("cannot read asset: " + path.string());
  Sha256 hash;
  std::vector<char> buffer(1 << 20);
  while (in) {
    in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    hash.update(buffer.data(), static_cast<size_t>(in.gcount()));
  }
  if (!in.eof()) bundle_error("cannot read asset: " + path.string());
  return hash.hexdigest();
}

class ManifestChecker {
 public:
  // pending != nullptr: sizes only; the SHA-256 checks are listed there (BundleCheck::defer_sha256).
  ManifestChecker(fs::path root, bool verify_sha256, std::vector<BundleFile>& checked,
                  std::vector<PendingHash>* pending = nullptr)
      : root_(std::move(root)), verify_(verify_sha256), checked_(checked), pending_(pending) {}

  // One {"file", "size", "sha256"} entry: returns the file's path.
  std::string asset(const Value& entry, const std::string& where) {
    const std::string& file = string_member(entry, "file", where);
    check_relative_path(file);
    const fs::path path = root_ / file;
    std::error_code ec;
    if (!fs::is_regular_file(path, ec)) bundle_error("missing asset: " + path.string());
    const Value& size = member(entry, "size", where);
    if (!size.is_int() || size.as_int() < 0) bundle_error("config.json: " + where + ".size must be a non-negative integer");
    const uint64_t actual = fs::file_size(path, ec);
    if (ec) bundle_error("cannot read asset: " + path.string());
    if (actual != static_cast<uint64_t>(size.as_int())) bundle_error("asset size mismatch: " + path.string());
    const std::string& expected = string_member(entry, "sha256", where);
    if (!is_sha256_hex(expected)) bundle_error("invalid SHA256 in config.json for " + path.string());
    if (verify_ && pending_ != nullptr)
      pending_->push_back({path.string(), expected, actual});
    else if (verify_ && file_sha256(path) != expected)
      bundle_error("asset SHA256 mismatch: " + path.string());
    checked_.push_back({path.string(), actual});
    return path.string();
  }

  // A {"directory", "files": [...]} group whose files must all lie inside the directory.
  std::vector<std::string> group(const Value& block, const std::string& where, const char* kind) {
    const std::string& directory = string_member(block, "directory", where);
    check_relative_path(directory);
    const Value& files = member(block, "files", where);
    if (!files.is_array()) bundle_error("config.json: " + where + ".files must be an array");
    if (files.items().empty()) bundle_error(std::string(kind) + " asset list is empty");
    std::vector<std::string> names;
    for (const Value& entry : files.items()) {
      const std::string& file = string_member(entry, "file", where + ".files[]");
      if (file.rfind(directory + "/", 0) != 0) bundle_error(std::string(kind) + " asset is outside its directory: " + file);
      asset(entry, where + ".files[]");
      names.push_back(file.substr(directory.size() + 1));
    }
    return names;
  }

  std::string directory(const Value& block, const std::string& where) {
    return (root_ / string_member(block, "directory", where)).string();
  }

 private:
  fs::path root_;
  bool verify_;
  std::vector<BundleFile>& checked_;
  std::vector<PendingHash>* pending_;
};

Value number_or_null(double x) { return std::isfinite(x) ? Value::make_double(x) : Value::make_null(); }

}  // namespace

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------
Options parse_args(const std::vector<std::string>& args) {
  std::map<std::string, std::string> values;
  for (size_t i = 0; i < args.size(); ++i) {
    std::string key = args[i];
    std::optional<std::string> inline_value;
    if (key.rfind("--", 0) == 0) {
      if (const auto eq = key.find('='); eq != std::string::npos) {
        inline_value = key.substr(eq + 1);
        key.resize(eq);
      }
    } else if (key == "-h") {
      key = "--help";
    } else {
      throw UsageError("unexpected argument '" + key + "' (options start with --)");
    }
    const OptionSpec* spec = find_option(key);
    if (!spec) throw UsageError("unknown option " + key);
    if (values.count(key)) throw UsageError(key + " was given more than once");
    if (!spec->takes_value) {
      if (inline_value) throw UsageError(key + " does not take a value");
      values[key] = "1";
    } else if (inline_value) {
      values[key] = *inline_value;
    } else {
      if (i + 1 >= args.size()) throw UsageError(key + " needs a value");
      values[key] = args[++i];
    }
  }
  auto has = [&](const char* key) { return values.count(key) != 0; };
  auto get = [&](const char* key) -> std::optional<std::string> {
    auto it = values.find(key);
    return it == values.end() ? std::nullopt : std::optional<std::string>(it->second);
  };

  Options o;
  if (args.empty() || has("--help")) {
    o.command = Command::kHelp;
    return o;
  }
  if (has("--version")) {
    o.command = Command::kVersion;
    return o;
  }
  if (has("--check-assets") && has("--list-voices")) throw UsageError("--check-assets cannot be combined with --list-voices");
  o.command = has("--check-assets") ? Command::kCheckAssets : has("--list-voices") ? Command::kListVoices : Command::kSpeak;

  if (auto v = get("--assets")) o.assets = *v;
  if (o.assets.empty()) throw UsageError("--assets DIR is required (a directory prepared by tools/prepare_assets.py)");

  o.text = get("--text");
  o.text_file = get("--text-file");
  o.voice = get("--voice");
  o.data = get("--data");
  o.report = get("--report");
  o.preset = get("--preset");
  o.reference_audio = get("--reference-audio");
  o.reference_transcript = get("--reference-transcript");
  o.save_voice = get("--save-voice");
  o.load_voice = get("--load-voice");
  o.no_normalize = has("--no-normalize");

  if (o.text && o.text_file) throw UsageError("--text cannot be combined with --text-file");
  if (o.text_file && o.text_file->empty()) throw UsageError("--text-file needs a file name or -");
  if (o.voice && o.voice->empty()) throw UsageError("--voice needs a voice name (see --list-voices)");

  // Voice cloning and saved voices.
  if (o.reference_audio && o.reference_audio->empty()) throw UsageError("--reference-audio needs a file name");
  if (o.cloning() && o.voice) throw UsageError("--reference-audio cannot be combined with --voice");
  if (o.cloning() && (!o.reference_transcript || o.reference_transcript->empty()))
    throw UsageError("--reference-transcript is required with --reference-audio");
  if (!o.cloning() && o.reference_transcript) throw UsageError("--reference-transcript requires --reference-audio");
  if (o.save_voice && !o.cloning()) throw UsageError("--save-voice requires --reference-audio and --reference-transcript");
  if (o.save_voice && o.save_voice->empty()) throw UsageError("--save-voice needs a file name");
  if (o.load_voice) {
    if (o.load_voice->empty()) throw UsageError("--load-voice needs a file name");
    if (o.voice) throw UsageError("--load-voice cannot be combined with --voice");
    if (o.cloning()) throw UsageError("--load-voice cannot be combined with --reference-audio");
  }

  if (o.command == Command::kListVoices) {
    for (const char* key : {"--text", "--text-file", "--reference-audio", "--save-voice", "--load-voice"})
      if (has(key)) throw UsageError(std::string("--list-voices cannot be combined with ") + key);
  }

  // Output.
  if (auto v = get("--format")) o.format = *v;
  if (o.format != "wav" && o.format != "pcm") throw UsageError("--format must be wav or pcm");
  if (auto v = get("--stream")) {
    if (*v != "true" && *v != "false") throw UsageError("--stream must be true or false");
    o.stream = *v == "true";
  }
  if (auto v = get("--repeat")) o.repeat = parse_int<int>(*v, "--repeat", 1, 100);
  o.output = get("--output").value_or(o.format == "wav" ? "output.wav" : "output.pcm");
  if (o.output.empty()) throw UsageError("--output needs a file name or -");
  if (o.output == "-" && o.repeat != 1) throw UsageError("--repeat cannot be combined with --output -");

  // Engine.
  if (auto v = get("--device")) o.device = parse_int<int>(*v, "--device", 0, 1 << 16);
  if (auto v = get("--max-streams")) o.max_streams = parse_int<int>(*v, "--max-streams", 1, 1 << 16);
  if (auto v = get("--threads")) o.threads = parse_int<int>(*v, "--threads", 1, 1 << 12);

  // Voice cache and memory.
  if (auto v = get("--voice-cache")) {
    o.voice_cache_given = true;
    if (*v == "all") o.voice_cache = VoiceCacheMode::kAll;
    else if (*v == "lru") o.voice_cache = VoiceCacheMode::kLru;
    else if (*v == "none") o.voice_cache = VoiceCacheMode::kNone;
    else throw UsageError("--voice-cache must be all, lru or none");
  }
  if (auto v = get("--voice-cache-voices")) o.voice_cache_voices = parse_int<int>(*v, "--voice-cache-voices", 1, 4096);
  if (auto v = get("--voice-cache-mib"))
    o.voice_cache_mib = parse_int<uint64_t>(*v, "--voice-cache-mib", 1, UINT64_MAX >> 20);
  if (o.voice_cache == VoiceCacheMode::kAll) {
    for (const char* key : {"--voice-cache-voices", "--voice-cache-mib"})
      if (has(key)) throw UsageError(std::string(key) + " requires --voice-cache lru or none");
  }
  if (auto v = get("--memory-profile")) {
    if (*v == "auto") o.memory_profile = MemoryProfile::kAuto;
    else if (*v == "default") o.memory_profile = MemoryProfile::kDefault;
    else if (*v == "low") o.memory_profile = MemoryProfile::kLow;
    else throw UsageError("--memory-profile must be auto, default or low");
  }

  // Sampling (the engine checks the ranges and reports the field).
  if (auto v = get("--seed")) o.seed = parse_int<uint64_t>(*v, "--seed", 0, UINT64_MAX);
  if (o.preset && *o.preset != "stable" && *o.preset != "expressive")
    throw UsageError("--preset must be stable or expressive");
  if (auto v = get("--temperature")) o.temperature = parse_real(*v, "--temperature");
  if (auto v = get("--top-p")) o.top_p = parse_real(*v, "--top-p");
  if (auto v = get("--min-p")) o.min_p = parse_real(*v, "--min-p");
  if (auto v = get("--top-k")) o.top_k = parse_int<int64_t>(*v, "--top-k", 0, INT64_MAX);
  if (auto v = get("--max-tokens")) o.max_tokens = parse_int<int64_t>(*v, "--max-tokens", 1, INT64_MAX);

  // Text is required to speak, except when only building and saving a voice.
  if (o.command == Command::kSpeak && !o.has_text() && !o.save_voice)
    throw UsageError("--text or --text-file is required");
  return o;
}

std::string help_text() {
  return "Kitten TTS 2 speech synthesis on NVIDIA GPUs\n"
         "\n"
         "Usage:\n"
         "  kitten-tts-native --assets DIR --text TEXT [--voice NAME] [--output FILE|-]\n"
         "  kitten-tts-native --assets DIR --list-voices\n"
         "  kitten-tts-native --assets DIR --check-assets\n"
         "  kitten-tts-native --version\n"
         "\n"
         "Input and output:\n"
         "  --assets DIR            model directory prepared by tools/prepare_assets.py (required)\n"
         "  --text TEXT             text to speak\n"
         "  --text-file FILE|-      read the text from FILE, or from standard input with -\n"
         "  --voice NAME            built-in voice (default: the bundle's default voice; see --list-voices)\n"
         "  --output FILE|-         output file, or - for standard output (default output.wav / output.pcm)\n"
         "  --format wav|pcm        WAV file or raw 16-bit little-endian 24 kHz mono PCM (default wav)\n"
         "  --stream true|false     true (default): stream audio as it is generated; false: split the text\n"
         "                          into sentences, render each one whole and write one complete file\n"
         "  --report FILE           also write the JSON run report to FILE\n"
         "\n"
         "Sampling:\n"
         "  --seed N                fixed sampling seed; repeated --stream false runs give the same audio\n"
         "  --preset stable|expressive\n"
         "  --temperature F --top-p F --min-p F --top-k N --max-tokens N\n"
         "  --no-normalize          speak the text as given (skip number, date and symbol expansion)\n"
         "\n"
         "Voice cloning (needs a bundle prepared with cloning files):\n"
         "  --reference-audio FILE  2-30 s recording of the voice to clone (at most 10 MiB)\n"
         "  --reference-transcript TEXT\n"
         "                          exact words spoken in the recording (required with --reference-audio)\n"
         "  --save-voice FILE       save the cloned voice to FILE (--text is then optional)\n"
         "  --load-voice FILE       speak with a voice saved by --save-voice\n"
         "\n"
         "Engine:\n"
         "  --device N              CUDA device (default 0)\n"
         "  --max-streams N         concurrent streams to reserve (default 1)\n"
         "  --threads N             text processing threads\n"
         "  --data DIR              use another prepared text-normalisation data directory\n"
         "  --repeat N              synthesize N times (1-100) and report each run\n"
         "\n"
         "Voice cache and memory:\n"
         "  --voice-cache all|lru|none\n"
         "                          all: prepare every built-in voice on the GPU at startup;\n"
         "                          lru: prepare a voice on first use and keep it until room is needed;\n"
         "                          none: keep a voice on the GPU only while a request uses it.\n"
         "                          Default: lru with one voice per stream for a built-in voice (this tool\n"
         "                          speaks with one voice), all for a cloned or loaded voice\n"
         "  --voice-cache-voices N  lru / none: at most N voices on the GPU at once (1-4096)\n"
         "  --voice-cache-mib N     lru / none: at most N MiB of GPU memory for voices\n"
         "  --memory-profile auto|default|low\n"
         "                          auto (default): the default profile when it fits this GPU, otherwise low;\n"
         "                          low: size working buffers for the requests the engine can serve\n"
         "                          instead of the largest ones (for GPUs with less memory)\n"
         "\n"
         "Other:\n"
         "  --list-voices           print the built-in voices and exit\n"
         "  --check-assets          verify every file of the bundle (sizes and SHA-256) and exit\n"
         "  --version               print the library build information and exit\n"
         "  --help                  print this help\n"
         "\n"
         "Ctrl-C stops the current synthesis. A JSON report of each run is printed to standard error.\n"
         "\n"
         "Exit status: 0 success, 1 error, 2 invalid command line, 130 cancelled (Ctrl-C or SIGTERM once\n"
         "synthesis or voice cloning has started; earlier, while the model is checked and loaded, the signal\n"
         "ends the program at once).\n";
}

// ---------------------------------------------------------------------------
// Bundle
// ---------------------------------------------------------------------------
void check_relative_path(const std::string& value) {
  auto invalid = [&] { bundle_error("invalid path in config.json: " + value); };
  if (value.empty() || value.front() == '/' || value.back() == '/') invalid();
  for (unsigned char c : value)
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' ||
          c == '.' || c == '/'))
      invalid();
  if (value.find("//") != std::string::npos) invalid();
  for (const auto& part : fs::path(value))
    if (part == "." || part == "..") invalid();
}

Bundle load_bundle(const std::string& dir, const BundleCheck& check) {
  Bundle b;
  const fs::path root(dir);
  b.root = root.string();
  std::error_code ec;
  if (!fs::is_directory(root, ec)) bundle_error("assets directory not found: " + root.string());
  const fs::path config_path = root / "config.json";
  if (!fs::is_regular_file(config_path, ec)) bundle_error("missing asset: " + config_path.string());
  Value config;
  try {
    config = json::parse(read_file(config_path, "config.json"));
  } catch (const json::ParseError& e) {
    bundle_error("config.json is not valid JSON: " + std::string(e.what()));
  }
  if (!config.is_object()) bundle_error("config.json must hold a JSON object");
  const Value* type = config.get("type");
  if (!type || !type->is_string() || type->as_string() != "KITTEN2")
    bundle_error("config.json is not a Kitten TTS 2 bundle (\"type\" must be \"KITTEN2\")");
  if (const Value* v = config.get("default_voice"); v && v->is_string() && !v->as_string().empty())
    b.default_voice = v->as_string();
  const Value* manifest = config.get("native");
  if (!manifest || !manifest->is_object())
    bundle_error("config.json has no prepared GPU files; prepare the directory with tools/prepare_assets.py");
  const Value* decoder = manifest->get("decoder");
  if (!is_version_one(*manifest) || (decoder && !(decoder->is_string() && decoder->as_string() == "default")))
    bundle_error("unsupported asset manifest in config.json (version 1 with the default decoder is required)");

  ManifestChecker files(root, check.verify_sha256, b.checked, check.defer_sha256 ? &b.pending_hashes : nullptr);
  if (check.engine_files) {
    b.model = files.asset(member(*manifest, "lm", "native"), "native.lm");
    b.decoder = files.asset(member(*manifest, "acoustic", "native"), "native.acoustic");
  }
  if (check.voices) b.voices = files.asset(member(*manifest, "voices", "native"), "native.voices");
  if (check.engine_files) {
    b.tokenizer = files.asset(member(*manifest, "tokenizer", "native"), "native.tokenizer");
    const Value& text = member(*manifest, "text", "native");
    files.group(text, "native.text", "text");
    b.text_data = files.directory(text, "native.text");
  }
  const Value* cloning = manifest->get("cloning");
  b.has_cloning = cloning != nullptr;
  if (check.cloning == CloningCheck::kRequire && !cloning)
    bundle_error("this bundle has no voice cloning files; prepare it again with cloning enabled");
  if (cloning && check.cloning != CloningCheck::kSkip) {
    if (!cloning->is_object() || !is_version_one(*cloning)) bundle_error("unsupported cloning manifest in config.json");
    const auto names = files.group(*cloning, "native.cloning", "cloning");
    for (const char* required : {"s3gen.safetensors", "voicebuild_consts.safetensors", "voicebuild_pyannote.safetensors"})
      if (std::find(names.begin(), names.end(), required) == names.end())
        bundle_error(std::string("missing cloning asset: ") + required);
    b.voice_build_dir = files.directory(*cloning, "native.cloning");
  }
  return b;
}

void verify_bundle_hashes(const Bundle& bundle, int threads) {
  const auto& files = bundle.pending_hashes;
  if (files.empty()) return;
  // Largest files first, so the two big engine files (about 1 GB each) hash in parallel.
  std::vector<size_t> order(files.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return files[a].size > files[b].size; });
  std::vector<std::string> error(files.size());  // "" = the hash matched
  std::atomic<size_t> next{0};
  auto worker = [&] {
    for (size_t k; (k = next.fetch_add(1)) < order.size();) {
      const PendingHash& f = files[order[k]];
      try {
        if (file_sha256(f.path) != f.expected) error[order[k]] = "asset SHA256 mismatch: " + f.path;
      } catch (const std::exception& e) {
        error[order[k]] = e.what();
      }
    }
  };
  const int n = std::max(1, std::min<int>(threads, int(files.size())));
  std::vector<std::thread> pool;
  for (int i = 1; i < n; ++i) pool.emplace_back(worker);
  worker();
  for (auto& t : pool) t.join();
  for (const std::string& e : error)  // the first failure in manifest order, as load_bundle reports it
    if (!e.empty()) bundle_error(e);
}

// ---------------------------------------------------------------------------
// Voices
// ---------------------------------------------------------------------------
std::vector<VoiceEntry> list_voices(const std::string& voices_file) {
  const std::string data = read_file(voices_file, "voices file");
  voice::VoicesBlob blob;
  try {
    blob = voice::parse_voices_blob(data.data(), data.size(), /*verify_sha256=*/false);
  } catch (const voice::VoiceBlobError& e) {
    throw std::runtime_error(std::string("cannot read the voices file: ") + e.what());
  }
  std::vector<VoiceEntry> out;
  for (const auto& rec : blob.voices) {
    if (const Value* h = rec.meta.get("hidden"); h && h->is_bool() && h->as_bool()) continue;
    VoiceEntry v;
    v.name = rec.id();
    if (const Value* names = rec.meta.get("public_names"); names && names->is_array())
      for (const Value& n : names->items())
        if (n.is_string() && n.as_string() != v.name &&
            std::find(v.aliases.begin(), v.aliases.end(), n.as_string()) == v.aliases.end())
          v.aliases.push_back(n.as_string());
    if (const Value* l = rec.meta.get("language"); l && l->is_string()) v.language = l->as_string();
    out.push_back(std::move(v));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------
std::string build_report(const RunReport& r) {
  const double audio_seconds = double(r.samples) / 24000.0;
  std::vector<std::pair<std::string, Value>> m;
#if defined(KT2N_HIP)
  m.emplace_back("backend", Value::make_string("hip"));  // AMD Radeon (ROCm) build
#else
  m.emplace_back("backend", Value::make_string("cuda"));
#endif
  m.emplace_back("device", Value::make_int(r.device));
  m.emplace_back("voice", Value::make_string(r.voice));
  m.emplace_back("sample_rate", Value::make_int(24000));
  m.emplace_back("format", Value::make_string(r.format));
  m.emplace_back("stream", Value::make_bool(r.stream));
  m.emplace_back("bytes", Value::make_int(static_cast<int64_t>(r.bytes)));
  m.emplace_back("first_audio_ms", number_or_null(r.first_audio_ms));
  m.emplace_back("elapsed_seconds", number_or_null(r.elapsed_seconds));
  m.emplace_back("iteration", Value::make_int(r.iteration));
  m.emplace_back("audio_seconds", number_or_null(audio_seconds));
  m.emplace_back("rtf", r.samples > 0 ? number_or_null(r.elapsed_seconds / audio_seconds) : Value::make_null());
  if (r.seed) {
    // Seeds above INT64_MAX are kept exact as a string.
    m.emplace_back("seed", *r.seed <= uint64_t(INT64_MAX) ? Value::make_int(static_cast<int64_t>(*r.seed))
                                                          : Value::make_string(std::to_string(*r.seed)));
  }
  if (r.engine.present) {
    std::vector<std::pair<std::string, Value>> e;
    e.emplace_back("output", r.engine.output.empty() ? Value::make_null() : Value::make_string(r.engine.output));
    e.emplace_back("segments", Value::make_int(r.engine.segments));
    e.emplace_back("audio_tokens", Value::make_int(r.engine.audio_tokens));
    e.emplace_back("lm_steps", Value::make_int(r.engine.lm_steps));
    e.emplace_back("renders", Value::make_int(static_cast<int64_t>(r.engine.renders)));
    e.emplace_back("render_rows", Value::make_int(static_cast<int64_t>(r.engine.render_rows)));
    m.emplace_back("engine", Value::make_object(std::move(e)));
  }
  return Value::make_object(std::move(m)).dump();
}

void read_headers(const std::string& fields_json, EngineCounters& counters) {
  Value v;
  try {
    v = json::parse(fields_json);
  } catch (const json::ParseError&) {
    return;
  }
  if (!v.is_object()) return;
  counters.output = v.get("complete_bytes") ? "complete" : "stream";
}

void read_metrics(const std::string& fields_json, EngineCounters& counters) {
  Value v;
  try {
    v = json::parse(fields_json);
  } catch (const json::ParseError&) {
    return;
  }
  if (!v.is_object()) return;
  auto take = [&](const char* key, int64_t& out) {
    if (const Value* x = v.get(key); x && x->is_int()) out = x->as_int();
  };
  take("segments", counters.segments);
  take("audio_tokens", counters.audio_tokens);
  take("lm_steps", counters.lm_steps);
  counters.present = true;
}

std::string wav_header(uint32_t data_bytes) {
  std::string out;
  out.reserve(44);
  auto u16 = [&](uint16_t n) {
    out.push_back(static_cast<char>(n & 255));
    out.push_back(static_cast<char>(n >> 8));
  };
  auto u32 = [&](uint32_t n) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<char>((n >> (8 * i)) & 255));
  };
  out += "RIFF";
  u32(data_bytes + 36);
  out += "WAVEfmt ";
  u32(16);
  u16(1);       // PCM
  u16(1);       // mono
  u32(24000);   // sample rate
  u32(48000);   // byte rate
  u16(2);       // block align
  u16(16);      // bits per sample
  out += "data";
  u32(data_bytes);
  return out;
}

}  // namespace kt2n::cli
