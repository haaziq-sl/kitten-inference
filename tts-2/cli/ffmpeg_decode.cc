// kitten-tts-native: reference audio decode with the system's ffmpeg program (see ffmpeg_decode.h).
#include "ffmpeg_decode.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace kt2n::cli {
namespace {

// More than 30.1 s of mono float32 at any sensible rate; anything bigger is not what we asked ffmpeg for.
constexpr size_t kMaxOutputBytes = size_t(64) << 20;

std::vector<float> to_samples(const std::vector<char>& bytes, const std::string& path) {
  const size_t n = bytes.size() / sizeof(float);
  if (n == 0) throw std::runtime_error("ffmpeg found no audio in the reference audio file: " + path);
  std::vector<float> out(n);
  std::memcpy(out.data(), bytes.data(), n * sizeof(float));
  return out;
}

#if defined(_WIN32)
std::wstring widen(const std::string& s) {
  if (s.empty()) return {};
  const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()), nullptr, 0);
  if (n <= 0) throw std::runtime_error("the reference audio file name is not valid UTF-8");
  std::wstring w(static_cast<size_t>(n), L'\0');
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()), w.data(), n);
  return w;
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

std::vector<char> run(const std::string& program, const std::vector<std::string>& args, const std::string& path,
                      DWORD* exit_code) {
  std::wstring cmd;
  append_quoted(cmd, widen(program));
  for (const auto& a : args) {
    cmd += L' ';
    append_quoted(cmd, widen(a));
  }
  SECURITY_ATTRIBUTES sa{};
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = TRUE;
  HANDLE rd = nullptr, wr = nullptr;
  if (!CreatePipe(&rd, &wr, &sa, 0)) throw std::runtime_error("cannot create a pipe for ffmpeg");
  SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
  HANDLE nul = CreateFileW(L"NUL", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING,
                           0, nullptr);
  HANDLE err = nullptr;
  HANDLE own_err = GetStdHandle(STD_ERROR_HANDLE);
  if (own_err && own_err != INVALID_HANDLE_VALUE)
    DuplicateHandle(GetCurrentProcess(), own_err, GetCurrentProcess(), &err, 0, TRUE, DUPLICATE_SAME_ACCESS);
  STARTUPINFOW si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput = nul;
  si.hStdOutput = wr;
  si.hStdError = err ? err : nul;
  PROCESS_INFORMATION pi{};
  const BOOL started = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi);
  const DWORD start_error = started ? 0 : GetLastError();
  CloseHandle(wr);
  if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
  if (err) CloseHandle(err);
  if (!started) {
    CloseHandle(rd);
    if (start_error == ERROR_FILE_NOT_FOUND || start_error == ERROR_PATH_NOT_FOUND) throw FfmpegMissing(ffmpeg_missing_message());
    throw FfmpegMissing("cannot start ffmpeg (Windows error " + std::to_string(start_error) + "); " +
                        ffmpeg_missing_message());
  }
  CloseHandle(pi.hThread);
  std::vector<char> bytes;
  char buf[65536];
  bool too_big = false;
  for (;;) {
    DWORD got = 0;
    if (!ReadFile(rd, buf, sizeof(buf), &got, nullptr) || got == 0) break;
    if (bytes.size() + got > kMaxOutputBytes) {
      too_big = true;
      TerminateProcess(pi.hProcess, 1);
      break;
    }
    bytes.insert(bytes.end(), buf, buf + got);
  }
  CloseHandle(rd);
  WaitForSingleObject(pi.hProcess, INFINITE);
  GetExitCodeProcess(pi.hProcess, exit_code);
  CloseHandle(pi.hProcess);
  if (too_big) throw std::runtime_error("ffmpeg returned more audio than expected for: " + path);
  return bytes;
}
#endif

}  // namespace

std::vector<std::string> ffmpeg_decode_args(const std::string& path, int rate, double max_seconds) {
  char seconds[32];
  std::snprintf(seconds, sizeof(seconds), "%.3f", max_seconds);
  return {"-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1", "-i", path, "-map_metadata", "-1",
          "-vn", "-sn", "-dn", "-ac", "1", "-ar", std::to_string(rate), "-t", seconds, "-f", "f32le", "pipe:1"};
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

std::vector<float> ffmpeg_decode(const std::string& path, int rate, double max_seconds, const std::string& program) {
  const std::vector<std::string> args = ffmpeg_decode_args(path, rate, max_seconds);
#if defined(_WIN32)
  DWORD code = 1;
  const std::vector<char> bytes = run(program, args, path, &code);
  if (code != 0)
    throw std::runtime_error("ffmpeg could not decode the reference audio (exit status " + std::to_string(code) +
                             "): " + path);
  return to_samples(bytes, path);
#else
  std::vector<std::string> argv_s;
  argv_s.reserve(args.size() + 1);
  argv_s.push_back(program);
  argv_s.insert(argv_s.end(), args.begin(), args.end());
  std::vector<char*> argv;
  for (auto& a : argv_s) argv.push_back(a.data());
  argv.push_back(nullptr);

  int fds[2];
  if (pipe(fds) != 0) throw std::runtime_error(std::string("cannot create a pipe for ffmpeg: ") + std::strerror(errno));
  fcntl(fds[0], F_SETFD, FD_CLOEXEC);
  fcntl(fds[1], F_SETFD, FD_CLOEXEC);
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0);
  posix_spawn_file_actions_adddup2(&actions, fds[1], 1);
  pid_t pid = 0;
  const int rc = posix_spawnp(&pid, program.c_str(), &actions, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  close(fds[1]);
  if (rc != 0) {
    close(fds[0]);
    if (rc == ENOENT) throw FfmpegMissing(ffmpeg_missing_message());
    throw FfmpegMissing(std::string("cannot start ffmpeg (") + std::strerror(rc) + "); " + ffmpeg_missing_message());
  }
  std::vector<char> bytes;
  char buf[65536];
  bool too_big = false;
  for (;;) {
    const ssize_t got = read(fds[0], buf, sizeof(buf));
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0) break;
    if (bytes.size() + static_cast<size_t>(got) > kMaxOutputBytes) {
      too_big = true;
      kill(pid, SIGKILL);
      break;
    }
    bytes.insert(bytes.end(), buf, buf + got);
  }
  close(fds[0]);
  int status = 0;
  while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
  if (too_big) throw std::runtime_error("ffmpeg returned more audio than expected for: " + path);
  if (WIFSIGNALED(status))
    throw std::runtime_error("ffmpeg stopped with signal " + std::to_string(WTERMSIG(status)) +
                             " while decoding the reference audio: " + path);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    throw std::runtime_error("ffmpeg could not decode the reference audio (exit status " +
                             std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1) + "): " + path);
  return to_samples(bytes, path);
#endif
}

}  // namespace kt2n::cli
