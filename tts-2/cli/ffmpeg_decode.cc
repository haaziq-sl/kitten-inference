// kitten-tts-native: reference audio decode with the system's ffmpeg program (see ffmpeg_decode.h).
#include "ffmpeg_decode.h"

#include <cctype>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace kt2n::cli {
namespace {

// More than 30.1 s of mono float32 at any sensible rate; anything bigger is not what we asked ffmpeg for.
constexpr size_t kMaxOutputBytes = size_t(64) << 20;

struct Result {
  bool exited_ok = false;  // exit status 0
  std::string status;      // how it ended, for messages ("exit status 1", "signal 9")
  std::vector<char> out;   // standard output
};

#if defined(_WIN32)
std::wstring widen(const std::string& s) {
  if (s.empty()) return {};
  const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()), nullptr, 0);
  if (n <= 0) throw std::runtime_error("the reference audio file name is not valid UTF-8");
  std::wstring w(static_cast<size_t>(n), L'\0');
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()), w.data(), n);
  return w;
}

// The program file for `name`, from the PATH folders that are absolute paths only (a name with a folder part is taken
// as it is): CreateProcessW(NULL, ...) would look in the current folder before PATH, where a downloaded or planted
// ffmpeg.exe would then run with the user's recording. Only an .exe is taken for a bare name (a .bat or .cmd would run
// through cmd.exe). Empty when there is none.
std::wstring find_program(const std::wstring& name) {
  if (name.find_first_of(L"\\/:") != std::wstring::npos) return name;
  const std::wstring file = name.find(L'.') == std::wstring::npos ? name + L".exe" : name;
  const DWORD size = GetEnvironmentVariableW(L"PATH", nullptr, 0);
  if (size == 0) return {};
  std::wstring path(size, L'\0');
  path.resize(GetEnvironmentVariableW(L"PATH", path.data(), size));
  for (size_t start = 0; start <= path.size();) {
    size_t end = path.find(L';', start);
    if (end == std::wstring::npos) end = path.size();
    std::wstring dir = path.substr(start, end - start);
    start = end + 1;
    dir.erase(std::remove(dir.begin(), dir.end(), L'"'), dir.end());
    while (!dir.empty() && dir.back() == L' ') dir.pop_back();
    while (!dir.empty() && dir.front() == L' ') dir.erase(dir.begin());
    const bool drive = dir.size() >= 3 && ((dir[0] >= L'A' && dir[0] <= L'Z') || (dir[0] >= L'a' && dir[0] <= L'z')) &&
                       dir[1] == L':' && (dir[2] == L'\\' || dir[2] == L'/');
    const bool unc = dir.size() > 2 && dir[0] == L'\\' && dir[1] == L'\\';
    if (!drive && !unc) continue;  // empty or relative: never the current folder
    std::wstring candidate = dir;
    if (candidate.back() != L'\\' && candidate.back() != L'/') candidate += L'\\';
    candidate += file;
    const DWORD attributes = GetFileAttributesW(candidate.c_str());
    if (attributes != INVALID_FILE_ATTRIBUTES && !(attributes & FILE_ATTRIBUTE_DIRECTORY)) return candidate;
  }
  return {};
}

// Quotes one argument so that the C runtime's command-line parser (CommandLineToArgvW rules) gives it back unchanged.
void append_quoted(std::wstring& cmd, const std::wstring& arg) {
  if (!arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
    cmd += arg;
    return;
  }
  cmd += L'"';
  for (auto it = arg.begin();; ++it) {
    size_t backslashes = 0;
    while (it != arg.end() && *it == L'\\') {
      ++it;
      ++backslashes;
    }
    if (it == arg.end()) {
      cmd.append(backslashes * 2, L'\\');
      break;
    }
    if (*it == L'"') {
      cmd.append(backslashes * 2 + 1, L'\\');
    } else {
      cmd.append(backslashes, L'\\');
    }
    cmd += *it;
  }
  cmd += L'"';
}

// Runs program with args; `input` (when not null) is written to its standard input; its standard error goes to ours,
// or nowhere with discard_stderr.
Result run(const std::string& program, const std::vector<std::string>& args, const std::vector<char>* input,
           bool discard_stderr) {
  const std::wstring exe = find_program(widen(program));
  if (exe.empty()) throw FfmpegMissing(ffmpeg_missing_message());
  std::wstring cmd;
  append_quoted(cmd, exe);
  for (const auto& a : args) {
    cmd += L' ';
    append_quoted(cmd, widen(a));
  }
  SECURITY_ATTRIBUTES sa{};
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = TRUE;
  HANDLE out_rd = nullptr, out_wr = nullptr, in_rd = nullptr, in_wr = nullptr;
  if (!CreatePipe(&out_rd, &out_wr, &sa, 0)) throw std::runtime_error("cannot create a pipe for ffmpeg");
  SetHandleInformation(out_rd, HANDLE_FLAG_INHERIT, 0);
  if (input) {
    if (!CreatePipe(&in_rd, &in_wr, &sa, 0)) throw std::runtime_error("cannot create a pipe for ffmpeg");
    SetHandleInformation(in_wr, HANDLE_FLAG_INHERIT, 0);
  }
  HANDLE nul = CreateFileW(L"NUL", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING,
                           0, nullptr);
  HANDLE err = nullptr;
  HANDLE own_err = GetStdHandle(STD_ERROR_HANDLE);
  if (!discard_stderr && own_err && own_err != INVALID_HANDLE_VALUE)
    DuplicateHandle(GetCurrentProcess(), own_err, GetCurrentProcess(), &err, 0, TRUE, DUPLICATE_SAME_ACCESS);
  STARTUPINFOW si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput = input ? in_rd : nul;
  si.hStdOutput = out_wr;
  si.hStdError = err ? err : nul;
  PROCESS_INFORMATION pi{};
  const BOOL started = CreateProcessW(exe.c_str(), cmd.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi);
  const DWORD start_error = started ? 0 : GetLastError();
  CloseHandle(out_wr);
  if (in_rd) CloseHandle(in_rd);
  if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
  if (err) CloseHandle(err);
  if (!started) {
    CloseHandle(out_rd);
    if (in_wr) CloseHandle(in_wr);
    if (start_error == ERROR_FILE_NOT_FOUND || start_error == ERROR_PATH_NOT_FOUND) throw FfmpegMissing(ffmpeg_missing_message());
    throw FfmpegMissing("cannot start ffmpeg (Windows error " + std::to_string(start_error) + "); " +
                        ffmpeg_missing_message());
  }
  CloseHandle(pi.hThread);
  std::thread writer;
  if (input) {
    writer = std::thread([in_wr, input] {
      size_t done = 0;
      while (done < input->size()) {
        DWORD put = 0;
        const DWORD chunk = static_cast<DWORD>(std::min<size_t>(input->size() - done, size_t(1) << 20));
        if (!WriteFile(in_wr, input->data() + done, chunk, &put, nullptr) || put == 0) break;  // ffmpeg stopped reading
        done += put;
      }
      CloseHandle(in_wr);
    });
  }
  Result r;
  char buf[65536];
  bool too_big = false;
  for (;;) {
    DWORD got = 0;
    if (!ReadFile(out_rd, buf, sizeof(buf), &got, nullptr) || got == 0) break;
    if (r.out.size() + got > kMaxOutputBytes) {
      too_big = true;
      TerminateProcess(pi.hProcess, 1);
      break;
    }
    r.out.insert(r.out.end(), buf, buf + got);
  }
  CloseHandle(out_rd);
  WaitForSingleObject(pi.hProcess, INFINITE);
  if (writer.joinable()) writer.join();
  DWORD code = 1;
  GetExitCodeProcess(pi.hProcess, &code);
  CloseHandle(pi.hProcess);
  if (too_big) throw std::runtime_error("ffmpeg returned more audio than expected");
  r.exited_ok = code == 0;
  r.status = "exit status " + std::to_string(code);
  return r;
}
#else
// The program file for `name`, from the PATH folders that are absolute paths only (a name with a '/' is taken as it
// is): posix_spawnp would read an empty PATH entry as the current folder, where a downloaded or planted ffmpeg would
// then run with the user's recording. Empty when there is none.
std::string find_program(const std::string& name) {
  if (name.find('/') != std::string::npos) return name;
  const char* env = std::getenv("PATH");
  if (env == nullptr) return {};
  const std::string path = env;
  for (size_t start = 0; start <= path.size();) {
    size_t end = path.find(':', start);
    if (end == std::string::npos) end = path.size();
    const std::string dir = path.substr(start, end - start);
    start = end + 1;
    if (dir.empty() || dir[0] != '/') continue;  // empty or relative: never the current folder
    const std::string candidate = dir + (dir.back() == '/' ? "" : "/") + name;
    struct stat st {};
    if (stat(candidate.c_str(), &st) == 0 && S_ISREG(st.st_mode) && access(candidate.c_str(), X_OK) == 0)
      return candidate;
  }
  return {};
}

Result run(const std::string& program, const std::vector<std::string>& args, const std::vector<char>* input,
           bool discard_stderr) {
  const std::string exe = find_program(program);
  if (exe.empty()) throw FfmpegMissing(ffmpeg_missing_message());
  std::vector<std::string> argv_s;
  argv_s.reserve(args.size() + 1);
  argv_s.push_back(exe);
  argv_s.insert(argv_s.end(), args.begin(), args.end());
  std::vector<char*> argv;
  for (auto& a : argv_s) argv.push_back(a.data());
  argv.push_back(nullptr);

  int out[2], in[2] = {-1, -1};
  if (pipe(out) != 0 || (input && pipe(in) != 0))
    throw std::runtime_error(std::string("cannot create a pipe for ffmpeg: ") + std::strerror(errno));
  for (int fd : {out[0], out[1], in[0], in[1]})
    if (fd >= 0) fcntl(fd, F_SETFD, FD_CLOEXEC);
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  if (input)
    posix_spawn_file_actions_adddup2(&actions, in[0], 0);
  else
    posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0);
  posix_spawn_file_actions_adddup2(&actions, out[1], 1);
  if (discard_stderr) posix_spawn_file_actions_addopen(&actions, 2, "/dev/null", O_WRONLY, 0);
  pid_t pid = 0;
  const int rc = posix_spawn(&pid, exe.c_str(), &actions, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  close(out[1]);
  if (input) close(in[0]);
  if (rc != 0) {
    close(out[0]);
    if (input) close(in[1]);
    if (rc == ENOENT) throw FfmpegMissing(ffmpeg_missing_message());
    throw FfmpegMissing(std::string("cannot start ffmpeg (") + std::strerror(rc) + "); " + ffmpeg_missing_message());
  }
  std::thread writer;
  if (input) {
    const int fd = in[1];
    writer = std::thread([fd, input] {
      size_t done = 0;
      while (done < input->size()) {
        const ssize_t put = write(fd, input->data() + done, input->size() - done);
        if (put < 0 && errno == EINTR) continue;
        if (put <= 0) break;  // ffmpeg stopped reading (EPIPE: SIGPIPE is ignored meanwhile, below)
        done += static_cast<size_t>(put);
      }
      close(fd);
    });
  }
  Result r;
  char buf[65536];
  bool too_big = false;
  for (;;) {
    const ssize_t got = read(out[0], buf, sizeof(buf));
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0) break;
    if (r.out.size() + static_cast<size_t>(got) > kMaxOutputBytes) {
      too_big = true;
      kill(pid, SIGKILL);
      break;
    }
    r.out.insert(r.out.end(), buf, buf + got);
  }
  close(out[0]);
  int status = 0;
  while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
  if (writer.joinable()) writer.join();
  if (too_big) throw std::runtime_error("ffmpeg returned more audio than expected");
  r.exited_ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
  r.status = WIFSIGNALED(status) ? "signal " + std::to_string(WTERMSIG(status))
                                 : "exit status " + std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1);
  return r;
}
#endif

}  // namespace

std::vector<std::string> ffmpeg_decode_args(const std::string& input, int rate, double max_seconds) {
  char seconds[32];
  std::snprintf(seconds, sizeof(seconds), "%.3f", max_seconds);
  return {"-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1", "-i", input, "-map_metadata", "-1",
          "-vn", "-sn", "-dn", "-ac", "1", "-ar", std::to_string(rate), "-t", seconds, "-f", "f32le", "pipe:1"};
}

bool is_playlist(const std::vector<char>& data) {
  // As FFmpeg's probes find them: HLS and ffconcat lists at the start, DASH and IMF anywhere in what it probes (up
  // to the whole file here), SDP by a "c=IN IP" line. A UTF-8 byte order mark and blank space at the start are
  // skipped, which refuses a little more than FFmpeg would read as such.
  const std::string all(data.begin(), data.end());
  size_t i = all.compare(0, 3, "\xEF\xBB\xBF") == 0 ? 3 : 0;
  while (i < all.size() && i < 4096 && (all[i] == ' ' || all[i] == '\t' || all[i] == '\r' || all[i] == '\n')) ++i;
  if (all.compare(i, 7, "#EXTM3U") == 0 || all.compare(i, 8, "ffconcat") == 0) return true;
  std::string low(all);
  for (char& c : low) c = (char)std::tolower((unsigned char)c);
  if (low.find("<mpd") != std::string::npos && low.find("dash:profile") != std::string::npos) return true;
  if (low.find("<compositionplaylist") != std::string::npos) return true;
  return all.compare(0, 7, "c=IN IP") == 0 || all.find("\nc=IN IP") != std::string::npos ||
         all.find("\n\rc=IN IP") != std::string::npos;
}

std::string ffmpeg_missing_message() {
#if defined(_WIN32)
  return "ffmpeg was not found: voice cloning uses it to read the reference audio. Install it with: "
         "winget install --id Gyan.FFmpeg -e (then open a new terminal)";
#else
  return "ffmpeg was not found: voice cloning uses it to read the reference audio. Install it with: "
         "sudo apt-get install -y ffmpeg";
#endif
}

std::vector<float> ffmpeg_decode(const std::string& path, const std::vector<char>& data, int rate, double max_seconds,
                                 const std::string& program) {
#if !defined(_WIN32)
  // A write to ffmpeg's standard input after it stopped reading must fail with EPIPE, not end this program.
  struct IgnorePipe {
    void (*old)(int) = std::signal(SIGPIPE, SIG_IGN);
    ~IgnorePipe() { std::signal(SIGPIPE, old); }
  } ignore_pipe;
#endif
  if (is_playlist(data))
    throw std::runtime_error("the reference audio is a playlist or stream manifest (HLS, DASH or similar), not a "
                             "recording: " + path);
  // From standard input first, as the reference pipeline and the in-process decode read it (same samples); its
  // messages are not shown, since a file that needs seeking is then read by name.
  // By name with the "file:" protocol prefix: ffmpeg reads "-i 2026-10-25T10:30.m4a" as protocol "2026-10-25T10"
  // (everything before a ':' that has only URL scheme characters), as its documentation says for names with a ':'.
  Result r = run(program, ffmpeg_decode_args("pipe:0", rate, max_seconds), &data, true);
  if (!r.exited_ok || r.out.size() < sizeof(float))
    r = run(program, ffmpeg_decode_args("file:" + path, rate, max_seconds), nullptr, false);
  if (!r.exited_ok) throw std::runtime_error("ffmpeg could not decode the reference audio (" + r.status + "): " + path);
  const size_t n = r.out.size() / sizeof(float);
  if (n == 0) throw std::runtime_error("ffmpeg found no audio in the reference audio file: " + path);
  std::vector<float> samples(n);
  std::memcpy(samples.data(), r.out.data(), n * sizeof(float));
  return samples;
}

}  // namespace kt2n::cli
