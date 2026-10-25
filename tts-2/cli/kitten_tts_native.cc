// kitten-tts-native: speak text with the Kitten TTS 2 engine (libkt2n_engine).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <clocale>
#include <cstring>
#include <ctime>
#include <csignal>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <locale>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>  // _O_BINARY
#include <windows.h>
#include "kt2n/platform.h"
#include "kt2n/posix_compat.h"
#else
#include <unistd.h>
#endif

#include "cli_core.h"
#include "ffmpeg_decode.h"
#include "json.h"
#include "kt2n/engine.h"
#include "stats_reader.h"

namespace {

namespace fs = std::filesystem;
using kt2n::cli::Options;

volatile std::sig_atomic_t interrupted = 0;
#if defined(_WIN32)
// The Windows C runtime resets a handler to the default before calling it: re-arm it, so a second Ctrl-C does not end
// the program before it has finished the WAV header.
void on_signal(int sig) {
  interrupted = 1;
  std::signal(sig, on_signal);
}
#else
void on_signal(int) { interrupted = 1; }
#endif

#if !defined(_WIN32)
// A model file of the right size but damaged can crash the engine's file readers during create, before the background
// SHA-256 check (speak) has reported the mismatch. While create runs, a crash signal (SIGSEGV, SIGBUS, SIGFPE, SIGILL,
// SIGABRT) waits for that check: on a mismatch it prints the check's message, as without the crash, and exits with
// status 1; otherwise (or after 300 s) it ends the process with the signal as before. Async-signal-safe calls only;
// the check runs on its own threads, which the crashed thread does not block.
std::atomic<int> g_hash_state{0};  // 0 running, 1 every file matched, 2 mismatch (g_hash_message set first)
char g_hash_message[1024];
void on_create_crash(int sig) {
  for (int i = 0; i < 15000 && g_hash_state.load() == 0; ++i) {
    timespec ts{0, 20000000};
    nanosleep(&ts, nullptr);
  }
  if (g_hash_state.load() == 2) {
    const char prefix[] = "kitten-tts-native: ";
    if (write(2, prefix, sizeof prefix - 1) >= 0 && write(2, g_hash_message, std::strlen(g_hash_message)) >= 0)
      (void)!write(2, "\n", 1);
    _exit(1);
  }
  std::signal(sig, SIG_DFL);
  std::raise(sig);
}
// Installs the handlers for the engine start and restores the previous ones afterwards.
class CreateCrashReport {
 public:
  CreateCrashReport() {
    struct sigaction sa {};
    sa.sa_handler = on_create_crash;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESETHAND | SA_NODEFER;
    for (size_t i = 0; i < std::size(kSignals); ++i) sigaction(kSignals[i], &sa, &old_[i]);
  }
  ~CreateCrashReport() {
    for (size_t i = 0; i < std::size(kSignals); ++i) sigaction(kSignals[i], &old_[i], nullptr);
  }
  CreateCrashReport(const CreateCrashReport&) = delete;
  CreateCrashReport& operator=(const CreateCrashReport&) = delete;

 private:
  static constexpr int kSignals[] = {SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT};
  struct sigaction old_[std::size(kSignals)] {};
};
#endif

// Ctrl-C / SIGTERM stopped the work: exit status 130.
class Interrupted : public std::runtime_error {
 public:
  Interrupted() : std::runtime_error("cancelled") {}
};

void check(int status, const kt2n_error& error = {}) {
  if (status == KT2N_OK) return;
  std::string message = error.message[0] ? error.message : kt2n_error_lookup(status)->code;
  if (status == KT2N_ERR_INVALID_VOICE) message += " (see --list-voices)";
  throw std::runtime_error(message + " (status " + std::to_string(status) + ")");
}

bool has_capability(const char* capability) {
  std::istringstream info(kt2n_build_info());
  std::string token;
  while (info >> token)
    if (token == capability) return true;
  return false;
}

bool can_clone() { return has_capability(KT2N_CAP_VOICE_CLONING); }

#if defined(_WIN32)
// Windows has no SIGPIPE (a closed pipe fails the write instead).
struct SignalHandlers {
  using handler = void (*)(int);
  handler old_int, old_term;
  SignalHandlers() : old_int(std::signal(SIGINT, on_signal)), old_term(std::signal(SIGTERM, on_signal)) {
    interrupted = 0;
  }
  ~SignalHandlers() {
    std::signal(SIGINT, old_int);
    std::signal(SIGTERM, old_term);
  }
};
#else
struct SignalHandlers {
  using handler = void (*)(int);
  handler old_int, old_term, old_pipe;
  SignalHandlers()
      : old_int(std::signal(SIGINT, on_signal)), old_term(std::signal(SIGTERM, on_signal)),
        old_pipe(std::signal(SIGPIPE, SIG_IGN)) {
    interrupted = 0;
  }
  ~SignalHandlers() {
    std::signal(SIGINT, old_int);
    std::signal(SIGTERM, old_term);
    std::signal(SIGPIPE, old_pipe);
  }
};
#endif

std::vector<char> read_limited(const std::string& path, size_t limit, const char* what) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error(std::string("cannot read ") + what + ": " + path);
  std::vector<char> data(limit + 1);
  file.read(data.data(), static_cast<std::streamsize>(data.size()));
  data.resize(static_cast<size_t>(file.gcount()));
  if (file.bad() || data.empty()) throw std::runtime_error(std::string(what) + " is empty or unreadable: " + path);
  if (data.size() > limit)
    throw std::runtime_error(std::string(what) + " must be at most " + std::to_string(limit >> 20) + " MiB: " + path);
  return data;
}

#if defined(_WIN32)
// Windows editors and PowerShell often write text with a UTF-8 byte order mark, or as UTF-16: drop the mark, refuse
// UTF-16 (the engine reads UTF-8).
std::string utf8_text(std::string text, const std::string& where) {
  if (text.size() >= 2 && ((uint8_t(text[0]) == 0xFF && uint8_t(text[1]) == 0xFE) ||
                           (uint8_t(text[0]) == 0xFE && uint8_t(text[1]) == 0xFF)))
    throw std::runtime_error(where + " is UTF-16 text; save it as UTF-8");
  if (text.size() >= 3 && text.compare(0, 3, "\xEF\xBB\xBF") == 0) text.erase(0, 3);
  return text;
}
#endif

std::string read_text(const Options& o) {
  if (o.text) return *o.text;
  if (!o.text_file) return {};
#if defined(_WIN32)
  if (*o.text_file == "-") {
    (void)_setmode(_fileno(stdin), _O_BINARY);  // no CR LF translation, no end at a 0x1A byte
    return utf8_text(std::string((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>()),
                     "standard input");
  }
#else
  if (*o.text_file == "-") return std::string((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
#endif
  std::ifstream file(*o.text_file, std::ios::binary);
  if (!file) throw std::runtime_error("cannot read text file: " + *o.text_file);
  std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  if (file.bad()) throw std::runtime_error("cannot read text file: " + *o.text_file);
#if defined(_WIN32)
  return utf8_text(std::move(text), *o.text_file);
#else
  return text;
#endif
}

// `pcm` (mono float32 at 24 kHz, decoded by ffmpeg) when the library takes decoded audio only, else the file's bytes.
kt2n_voice_id clone_voice(kt2n_engine* engine, const std::vector<char>& audio, const std::vector<float>& pcm,
                          const std::string& transcript) {
  kt2n_voice_build_params params{};
  params.struct_size = sizeof(params);
  params.flags = KT2N_VB_SAVED;
  if (!pcm.empty()) {
    params.flags |= KT2N_VB_DECODED_PCM;
    params.pcm = pcm.data();
    params.pcm_samples = pcm.size();
    params.pcm_rate = 24000;
  } else {
    params.audio = audio.data();
    params.audio_len = audio.size();
  }
  params.reference_transcript = transcript.c_str();
  params.language = "en";
  kt2n_job_id job = 0;
  kt2n_error error{};
  check(kt2n_voice_build(engine, &params, &job, &error), error);
  bool cancelled = false;
  for (;;) {
    if (interrupted && !cancelled) {
      check(kt2n_cancel(engine, job, KT2N_CANCEL_REQUEST));
      cancelled = true;
    }
    kt2n_event events[16]{};
    int32_t count = 0;
    check(kt2n_poll(engine, events, 16, 100, &count));
    bool done = false;
    int status = KT2N_OK;
    kt2n_voice_id voice = 0;
    for (int32_t i = 0; i < count; ++i) {
      const auto& event = events[i];
      if (event.id == job && (event.kind == KT2N_EVENT_VOICE_BUILT || event.kind == KT2N_EVENT_DONE)) {
        done = true;
        status = event.status;
        voice = event.voice;
        if (event.error) error = *event.error;
      }
    }
    kt2n_events_release(engine, events, count);
    if (!done) continue;
    check(kt2n_release(engine, job));
    if (cancelled || interrupted) throw Interrupted();
    check(status, error);
    if (!voice) throw std::runtime_error("the voice build returned no voice");
    return voice;
  }
}

// Writes the cloned voice next to `path` and renames it into place.
void save_voice(kt2n_engine* engine, kt2n_voice_id voice, const std::string& path) {
  size_t needed = 0;
  int status = kt2n_voice_export(engine, voice, nullptr, 0, &needed);
  if (status != KT2N_ERR_TOO_SMALL && status != KT2N_OK) check(status);
  std::vector<char> data(needed);
  check(kt2n_voice_export(engine, voice, data.data(), data.size(), &needed));
  data.resize(needed);
  const std::string temporary = path + ".tmp." + std::to_string(::getpid());
  {
    std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
    file.write(data.data(), static_cast<std::streamsize>(data.size()));
    file.flush();
    if (!file) {
      std::error_code ignored;
      fs::remove(temporary, ignored);
      throw std::runtime_error("cannot write voice file: " + path);
    }
  }
  std::error_code ec;
  fs::rename(temporary, path, ec);
  if (ec) {
    std::error_code ignored;
    fs::remove(temporary, ignored);
    throw std::runtime_error("cannot write voice file: " + path + " (" + ec.message() + ")");
  }
}

// Engine counters for the report. A failure only loses those counters, so it is a warning: the
// audio may already be written.
std::optional<kt2n_stats> stats_now(kt2n_engine* engine) {
  kt2n_stats s{};
  s.struct_size = sizeof(s);
  const int32_t status = kt2n_stats_get(engine, &s);
  if (status == KT2N_OK) return s;
  std::cerr << "kitten-tts-native: warning: cannot read the engine statistics (status " << status << ")\n";
  return std::nullopt;
}

int check_assets(const Options& o) {
  kt2n::cli::BundleCheck needs;
  needs.cloning = o.cloning() ? kt2n::cli::CloningCheck::kRequire : kt2n::cli::CloningCheck::kIfPresent;
  const auto bundle = kt2n::cli::load_bundle(o.assets, needs);
  uint64_t total = 0;
  for (const auto& f : bundle.checked) total += f.size;
  std::cout << "All " << bundle.checked.size() << " files listed in " << (fs::path(bundle.root) / "config.json").string()
            << " are present with the expected size and SHA-256 (" << (total + (1 << 19)) / (1 << 20) << " MiB).\n"
            << "Voice cloning: " << (bundle.has_cloning ? "available" : "not prepared") << "\n";
  return 0;
}

int list_voices(const Options& o) {
  kt2n::cli::BundleCheck needs;
  needs.engine_files = false;
  const auto bundle = kt2n::cli::load_bundle(o.assets, needs);
  const auto voices = kt2n::cli::list_voices(bundle.voices);
  for (const auto& v : voices) {
    std::cout << v.name << '\t' << (v.language.empty() ? "-" : v.language);
    for (size_t i = 0; i < v.aliases.size(); ++i) std::cout << (i ? "," : "\t") << v.aliases[i];
    std::cout << '\n';
  }
  std::cerr << voices.size() << " voices (default: " << bundle.default_voice << ")\n";
  return 0;
}

int speak(const Options& o) {
  const bool cloning = o.cloning();
  const bool loading = o.load_voice.has_value();
  if (cloning && !can_clone())
    throw std::runtime_error(
        "voice cloning is not available in this library build; rebuild with the default preset "
        "(or -DKT2N_BUILD_CLONING=ON -DKT2N_VOICE_AV=ON)");
  kt2n::cli::BundleCheck needs;
  needs.cloning = cloning ? kt2n::cli::CloningCheck::kRequire : kt2n::cli::CloningCheck::kSkip;
  needs.defer_sha256 = true;  // sizes now; the SHA-256 checks run beside the engine start (below)
  const auto bundle = kt2n::cli::load_bundle(o.assets, needs);

  const std::string text = read_text(o);
  std::vector<char> reference, saved;
  std::vector<float> reference_pcm;
  if (cloning) {
    reference = read_limited(*o.reference_audio, size_t(10) << 20, "reference audio");
    // A library without the in-process decode (the Python wheels) takes decoded audio: decode the file with the
    // system's ffmpeg program, as the reference pipeline does, before the engine starts.
    if (has_capability(KT2N_CAP_VOICE_PCM_ONLY)) {
      reference_pcm = kt2n::cli::ffmpeg_decode(*o.reference_audio);
      reference.clear();
    }
  }
  if (loading) saved = read_limited(*o.load_voice, size_t(64) << 20, "voice file");
  const std::string text_data = o.data.value_or(bundle.text_data);

  kt2n_engine_config config{};
  config.struct_size = sizeof(config);
  config.abi_version = KT2N_ABI_VERSION;
  config.flags = (cloning || loading) ? 0 : KT2N_CFG_INFERENCE_ONLY;
  if (cloning) {
    config.voice_build_dir = bundle.voice_build_dir.c_str();
    config.inline_voice_lru = 1;
    config.clone_slots = 1;
    config.max_custom_voices = 1;
  }
  if (loading) config.max_custom_voices = 1;
  config.device = o.device;
  config.max_streams = o.max_streams;
  config.lm_file = bundle.model.c_str();
  config.s3gen_file = bundle.decoder.c_str();
  config.voices_file = bundle.voices.c_str();
  config.tokenizer_json = bundle.tokenizer.c_str();
  config.text_data_dir = text_data.c_str();
  if (o.threads) config.threads_text = *o.threads;
  using kt2n::cli::MemoryProfile;
  using kt2n::cli::VoiceCacheMode;
  config.voice_cache_mode = o.voice_cache == VoiceCacheMode::kLru    ? KT2N_VOICE_CACHE_LRU
                            : o.voice_cache == VoiceCacheMode::kNone ? KT2N_VOICE_CACHE_NONE
                                                                     : KT2N_VOICE_CACHE_ALL;
  config.voice_cache_max_voices = o.voice_cache_voices.value_or(0);
  config.voice_cache_max_bytes = o.voice_cache_mib.value_or(0) << 20;
  if (!o.voice_cache_given && !cloning && !loading) {
    // This tool speaks with one built-in voice, so by default it prepares only that voice (right after create, see
    // kt2n_voice_prepare below) instead of every built-in voice during create: with 47 voices, 2.0 s less start on an
    // RTX 4060 Laptop, 1.6 GB less GPU memory there, and the same audio (seeded --stream false output is identical in every cache mode).
    config.voice_cache_mode = KT2N_VOICE_CACHE_LRU;
    config.voice_cache_max_voices = std::clamp(o.max_streams, 1, 4096);  // the cache's limit; max_streams is checked by create
  }
  config.memory_profile = o.memory_profile == MemoryProfile::kLow       ? KT2N_MEMORY_PROFILE_LOW
                          : o.memory_profile == MemoryProfile::kDefault ? KT2N_MEMORY_PROFILE_DEFAULT
                                                                        : KT2N_MEMORY_PROFILE_AUTO;
  // A saved voice: with the low-memory profile the engine sizes its context for this voice's prompt, not for the
  // longest one a saved voice can have (more room for long segments on small GPUs). Other profiles ignore it, so it
  // is set whenever a voice is loaded (also when a profile is chosen at create).
  if (loading) {
    const int32_t ids = kt2n_voice_artifact_lm_prefix_ids(saved.data(), saved.size());
    if (ids > 0) config.custom_voice_max_prefix_ids = ids;
  }

  // The SHA-256 checks of the model files (about 2 GB: ~1.1-1.7 s on one core with SHA instructions, several times
  // that without) run in background threads while the engine starts, instead of before it. Nothing is synthesised
  // or written until they pass, and a mismatch is reported with the same message once create returns (it wins over
  // an engine start error, which a damaged file can cause). File sizes are still checked before create; a file of
  // the right size damaged in a way the loaders do not detect can crash create: on Linux the mismatch is then still
  // reported (CreateCrashReport).
  std::exception_ptr hash_error;
#if !defined(_WIN32)
  std::optional<CreateCrashReport> crash_report(std::in_place);
#endif
  std::thread hasher([&bundle, &hash_error] {
    try {
      kt2n::cli::verify_bundle_hashes(bundle, 2);
#if !defined(_WIN32)
      g_hash_state.store(1);
#endif
    } catch (const std::exception& e) {
#if !defined(_WIN32)
      std::snprintf(g_hash_message, sizeof g_hash_message, "%s", e.what());
      g_hash_state.store(2);
#endif
      hash_error = std::current_exception();
    } catch (...) {
#if !defined(_WIN32)
      g_hash_state.store(1);  // not a mismatch message: a crash during create ends with its signal
#endif
      hash_error = std::current_exception();
    }
  });
  kt2n_engine* raw = nullptr;
  kt2n_error error{};
  const int created = kt2n_engine_create(&config, &raw, &error);
  std::unique_ptr<kt2n_engine, decltype(&kt2n_engine_destroy)> engine(raw, kt2n_engine_destroy);
  hasher.join();
#if !defined(_WIN32)
  crash_report.reset();
#endif
  if (hash_error) std::rethrow_exception(hash_error);
  check(created, error);
  {
    const auto stats = kt2n::json::parse(kt2n::cli::detail::read_native_runtime_stats(engine.get()));
    auto field = [&](const char* key) {
      const auto* v = stats.get(key);
      return v && v->is_string() ? v->as_string() : std::string();
    };
    // Runtime identity values of the complete engine (see kt2n_stats_json).
    if (field("lm") != "cuda" || field("render") != "cuda" || field("normalizer") != "native" || field("tokenizer") != "native")
      throw std::runtime_error(
          "this engine library is incomplete (it needs the CUDA language model and decoder plus the text normaliser "
          "and tokenizer); rebuild with -DKT2N_CUDA=ON -DKT2N_BUILD_TEXT=ON");
  }

  SignalHandlers handlers;
  kt2n_voice_id voice = 0;
  std::string voice_name;
  if (cloning) {
    voice_name = "reference";
    voice = clone_voice(engine.get(), reference, reference_pcm, *o.reference_transcript);
    if (o.save_voice) {
      save_voice(engine.get(), voice, *o.save_voice);
      std::cerr << "Saved the cloned voice to " << *o.save_voice << "\n";
    }
  } else if (loading) {
    check(kt2n_voice_load(engine.get(), saved.data(), saved.size(), &voice, &error), error);
    kt2n_voice_info info{};
    info.struct_size = sizeof(info);
    check(kt2n_voice_info_get(engine.get(), voice, &info));
    voice_name = info.name;
  } else {
    voice_name = o.voice.value_or(bundle.default_voice);
    check(kt2n_voice_lookup(engine.get(), voice_name.c_str(), &voice, &error), error);
  }
  if (!o.has_text()) return 0;  // --save-voice without text: the voice is saved
  // With the one-voice cache above, prepare the voice now rather than inside the first request, so its first audio
  // comes as soon as with every voice prepared at create (about 35 ms of GPU work on an RTX 4060 Laptop). With voice
  // cache all or none this returns at once. Best effort: if it cannot be prepared now, the request prepares it and
  // reports any error.
  (void)kt2n_voice_prepare(engine.get(), voice, nullptr);

  for (int repeat = 0; repeat < o.repeat; ++repeat) {
    const bool to_stdout = o.output == "-";
    // Streaming to a WAV file: the engine sends PCM and the header is patched at the end.
    const bool file_wav = o.stream && !to_stdout && o.format == "wav";

    kt2n_request_params request{};
    request.struct_size = sizeof(request);
    request.voice = voice;
    request.speed = 1.0;
    request.text = text.c_str();
    request.text_len = text.size();
    request.format = o.format == "wav" && (!o.stream || to_stdout) ? KT2N_FORMAT_WAV : KT2N_FORMAT_PCM;
    request.output = o.stream ? KT2N_OUTPUT_STREAM : KT2N_OUTPUT_COMPLETE;
    if (o.no_normalize) request.flags |= KT2N_REQ_NORMALIZE_OFF;
    if (o.seed) {
      request.seed = *o.seed;
      request.flags |= KT2N_REQ_SEED;
    }
    if (o.preset) {
      request.sampler.present |= KT2N_SAMPLER_HAS_MODE;
      request.sampler.mode = *o.preset == "stable" ? KT2N_SAMPLER_STABLE : KT2N_SAMPLER_EXPRESSIVE;
    }
    auto real = [&](const std::optional<double>& value, uint32_t bit, double& field) {
      if (!value) return;
      field = *value;
      request.sampler.present |= bit;
    };
    real(o.temperature, KT2N_SAMPLER_HAS_TEMPERATURE, request.sampler.temperature);
    real(o.top_p, KT2N_SAMPLER_HAS_TOP_P, request.sampler.top_p);
    real(o.min_p, KT2N_SAMPLER_HAS_MIN_P, request.sampler.min_p);
    if (o.top_k) {
      request.sampler.top_k = *o.top_k;
      request.sampler.present |= KT2N_SAMPLER_HAS_TOP_K;
    }
    if (o.max_tokens) {
      request.sampler.max_new_tokens = *o.max_tokens;
      request.sampler.present |= KT2N_SAMPLER_HAS_MAX_NEW_TOKENS;
    }

    kt2n::cli::RunReport report;
    report.device = o.device;
    report.voice = voice_name;
    report.format = o.format;
    report.stream = o.stream;
    report.iteration = repeat + 1;
    report.seed = o.seed;
    const auto before = stats_now(engine.get());
    const auto start = std::chrono::steady_clock::now();
    bool done = false, cancelled = false;
    kt2n_request_id id = 0;
    check(kt2n_submit(engine.get(), &request, &id, &error), error);
    // The output is opened only once the engine has accepted the request, so a refused request leaves an existing
    // file as it was.
    std::ofstream file;
    if (!to_stdout) {
      file.open(o.output, std::ios::binary | std::ios::trunc);
      if (!file) {
        kt2n_cancel(engine.get(), id, KT2N_CANCEL_REQUEST);
        throw std::runtime_error("cannot write " + o.output);
      }
    }
#if defined(_WIN32)
    if (to_stdout) {  // audio bytes: no "\n" -> "\r\n" translation
      std::cout.flush();
      (void)_setmode(_fileno(stdout), _O_BINARY);
    }
#endif
    std::ostream& out = to_stdout ? std::cout : file;
    if (file_wav) out << kt2n::cli::wav_header(0);
    int terminal_status = KT2N_OK;
    kt2n_error terminal_error{};
    while (!done) {
      if (interrupted && !cancelled) {
        check(kt2n_cancel(engine.get(), id, KT2N_CANCEL_REQUEST));
        cancelled = true;
      }
      kt2n_event events[16]{};
      int32_t count = 0;
      check(kt2n_poll(engine.get(), events, 16, 100, &count));
      for (int32_t i = 0; i < count; ++i) {
        const auto& event = events[i];
        if (event.kind == KT2N_EVENT_AUDIO && !cancelled) {
          if (report.first_audio_ms < 0 && event.data_len)
            report.first_audio_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
          out.write(reinterpret_cast<const char*>(event.data), static_cast<std::streamsize>(event.data_len));
          out.flush();
          report.bytes += event.data_len;
          report.samples += event.samples;
        }
        if (event.kind == KT2N_EVENT_HEADERS && event.fields_json)
          kt2n::cli::read_headers(event.fields_json, report.engine);
        if (event.kind == KT2N_EVENT_METRICS && event.fields_json)
          kt2n::cli::read_metrics(event.fields_json, report.engine);
        if (event.kind == KT2N_EVENT_DONE) {
          done = true;
          terminal_status = event.status;
          if (event.error) terminal_error = *event.error;
        }
      }
      kt2n_events_release(engine.get(), events, count);
      if (done) check(kt2n_release(engine.get(), id));
      if (!out) {
        if (!done) kt2n_cancel(engine.get(), id, KT2N_CANCEL_REQUEST);
        throw std::runtime_error("audio output failed");
      }
    }
    report.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (file_wav) {  // also after a failed stream, so the header gives the length of the audio written
      if (report.bytes > std::numeric_limits<uint32_t>::max() - 36) throw std::runtime_error("audio exceeds the WAV size limit");
      file.seekp(0);
      file << kt2n::cli::wav_header(static_cast<uint32_t>(report.bytes));
      file.flush();
      if (!file) throw std::runtime_error("cannot finalize the WAV header");
    }
    if (!cancelled) check(terminal_status, terminal_error);
    if (cancelled) throw Interrupted();
    if (const auto after = stats_now(engine.get()); before && after) {
      report.engine.renders = after->renders - before->renders;
      report.engine.render_rows = after->render_rows - before->render_rows;
    }
    const std::string line = kt2n::cli::build_report(report);
    if (o.report) {
      std::ofstream report_file(*o.report, std::ios::trunc);
      report_file << line << '\n';
      if (!report_file) throw std::runtime_error("cannot write report: " + *o.report);
    }
    std::cerr << line << '\n';
  }
  return 0;
}

}  // namespace

#if defined(_WIN32)
namespace {
// Windows: the body of main(); wmain() at the end of the file passes it the arguments as UTF-8.
int run_main(int argc, char** argv) {
#else
int main(int argc, char** argv) {
#endif
#if defined(_WIN32)
  // argv is UTF-8 here (wmain below converts the wide arguments), but file names still reach the C runtime's narrow
  // file functions, which use the process code page: the manifest makes that UTF-8 on Windows 10 1903 and newer.
  // Older Windows would mangle non-ASCII file names, so refuse non-ASCII arguments there.
  if (!kt2n::platform::narrow_paths_are_utf8()) {
    for (int i = 1; i < argc; ++i)
      for (const char* c = argv[i]; *c; ++c)
        if (static_cast<unsigned char>(*c) >= 0x80) {
          std::cerr << "kitten-tts-native: non-ASCII arguments need Windows 10 version 1903 or newer; use --text-file "
                       "with a UTF-8 file and ASCII file names\n";
          return 2;
        }
  }
#endif
  std::setlocale(LC_ALL, "C.UTF-8");
  try {
    std::locale::global(std::locale("C.UTF-8"));
  } catch (const std::runtime_error&) {
  }
  Options options;
  try {
    options = kt2n::cli::parse_args(std::vector<std::string>(argv + 1, argv + argc));
  } catch (const kt2n::cli::UsageError& e) {
    std::cerr << "kitten-tts-native: " << e.what() << "\nRun 'kitten-tts-native --help' for usage.\n";
    return 2;
  }
  try {
    switch (options.command) {
      case kt2n::cli::Command::kHelp:
        std::cout << kt2n::cli::help_text();
        return 0;
      case kt2n::cli::Command::kVersion:
        std::cout << "kitten-tts-native " << KT2N_VERSION_STRING << "\n" << kt2n_build_info() << "\n";
        return 0;
      case kt2n::cli::Command::kCheckAssets:
        return check_assets(options);
      case kt2n::cli::Command::kListVoices:
        return list_voices(options);
      case kt2n::cli::Command::kSpeak:
        return speak(options);
    }
  } catch (const Interrupted&) {
    std::cerr << "kitten-tts-native: cancelled\n";
    return 130;
  } catch (const std::exception& e) {
    std::cerr << "kitten-tts-native: " << e.what() << "\n";
    return 1;
  }
  return 1;
}
#if defined(_WIN32)
}  // namespace

// The C runtime splits the command line into wide arguments with the same rules as for main(); converting them to
// UTF-8 here keeps every character, also on Windows versions that ignore the manifest's UTF-8 code page (there the
// narrow argv would hold best-fit replacements, which the check in run_main could not see).
// The engine DLL is delay-loaded (cli/CMakeLists.txt): load it here, before its first use, from next to the program
// (a build tree, an install) or from ..\lib (the Python wheel: bin\kitten-tts-native.exe, lib\kt2n_engine.dll), else
// from the usual DLL search path, so that a missing DLL is a clear error instead of a crash at the first call.
bool load_engine_dll() {
  std::vector<wchar_t> exe(32768);
  const DWORD n = GetModuleFileNameW(nullptr, exe.data(), static_cast<DWORD>(exe.size()));
  if (n > 0 && n < exe.size()) {
    const fs::path dir = fs::path(std::wstring(exe.data(), n)).parent_path();
    for (const fs::path& candidate : {dir / L"kt2n_engine.dll", dir.parent_path() / L"lib" / L"kt2n_engine.dll"}) {
      std::error_code ec;
      if (fs::is_regular_file(candidate, ec) &&
          LoadLibraryExW(candidate.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH) != nullptr)
        return true;
    }
  }
  if (LoadLibraryW(L"kt2n_engine.dll") != nullptr) return true;
  std::cerr << "kitten-tts-native: cannot load kt2n_engine.dll (Windows error " << GetLastError()
            << "); it belongs next to this program or in ..\\lib\n";
  return false;
}

int wmain(int argc, wchar_t** wargv) {
  if (!load_engine_dll()) return 1;
  std::vector<std::string> args;
  args.reserve(size_t(argc));
  for (int i = 0; i < argc; ++i) args.push_back(kt2n::platform::utf8_from_wide(wargv[i]));
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());
  argv.push_back(nullptr);
  return run_main(argc, argv.data());
}
#endif
