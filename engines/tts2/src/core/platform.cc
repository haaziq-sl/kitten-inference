// Platform layer (include/kt2n/platform.h): POSIX and Win32 implementations.
#include "kt2n/platform.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <timeapi.h>

#include "kt2n/posix_compat.h"

#include <cstdlib>
#include <vector>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#endif

namespace kt2n::platform {

#if defined(_WIN32)

namespace {

std::wstring widen(const std::string& s) {
  if (s.empty()) return {};
  const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), int(s.size()), nullptr, 0);
  if (n <= 0) return {};
  std::wstring w(size_t(n), L'\0');
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), int(s.size()), w.data(), n);
  return w;
}

// The POSIX strerror text for the common cases (messages and tests read the same on both systems), else the
// system's own message.
std::string error_text(DWORD code) {
  switch (code) {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
    case ERROR_INVALID_NAME:
      return "No such file or directory";
    case ERROR_ACCESS_DENIED:
    case ERROR_SHARING_VIOLATION:
      return "Permission denied";
    case ERROR_NOT_ENOUGH_MEMORY:
    case ERROR_OUTOFMEMORY:
    case ERROR_COMMITMENT_LIMIT:
      return "Cannot allocate memory";
    default:
      break;
  }
  char* buf = nullptr;
  const DWORD n = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 nullptr, code, 0, reinterpret_cast<char*>(&buf), 0, nullptr);
  std::string out = n && buf ? std::string(buf, n) : "Windows error " + std::to_string(code);
  if (buf) LocalFree(buf);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ' || out.back() == '.')) out.pop_back();
  return out;
}

}  // namespace

MapResult map_file(const std::string& path, const uint8_t** data, uint64_t* size, std::string* reason) {
  const std::wstring w = widen(path);
  if (w.empty()) {
    if (reason) *reason = "No such file or directory";
    return MapResult::kOpenFailed;
  }
  const DWORD attrs = GetFileAttributesW(w.c_str());
  if (attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY)) return MapResult::kNotRegular;
  // FILE_SHARE_DELETE: like POSIX, the file may be renamed or replaced while it is mapped.
  HANDLE file = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    if (reason) *reason = error_text(GetLastError());
    return MapResult::kOpenFailed;
  }
  if (GetFileType(file) != FILE_TYPE_DISK) {
    CloseHandle(file);
    return MapResult::kNotRegular;
  }
  LARGE_INTEGER li{};
  if (!GetFileSizeEx(file, &li)) {
    if (reason) *reason = error_text(GetLastError());
    CloseHandle(file);
    return MapResult::kStatFailed;
  }
  const uint64_t n = uint64_t(li.QuadPart);
  if (n == 0) {  // CreateFileMapping refuses empty files; POSIX callers saw size 0 before mapping
    CloseHandle(file);
    *data = nullptr;
    *size = 0;
    return MapResult::kOk;
  }
  HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (mapping == nullptr) {
    if (reason) *reason = error_text(GetLastError());
    CloseHandle(file);
    return MapResult::kMapFailed;
  }
  void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  const DWORD view_error = view ? 0 : GetLastError();
  CloseHandle(mapping);
  CloseHandle(file);
  if (view == nullptr) {
    if (reason) *reason = error_text(view_error);
    return MapResult::kMapFailed;
  }
  *data = static_cast<const uint8_t*>(view);
  *size = n;
  return MapResult::kOk;
}

void unmap_file(const void* data, uint64_t /*size*/) {
  if (data != nullptr) UnmapViewOfFile(data);
}

size_t page_size() {
  SYSTEM_INFO si{};
  GetSystemInfo(&si);
  return si.dwPageSize > 0 ? size_t(si.dwPageSize) : 4096;
}

void advise_willneed(const void* addr, size_t len) {
  if (addr == nullptr || len == 0) return;
  WIN32_MEMORY_RANGE_ENTRY range{const_cast<void*>(addr), len};
  (void)PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
}

void advise_dontneed(const void* addr, size_t len) {
  if (addr == nullptr || len == 0) return;
  // VirtualUnlock on a range that is not locked removes its pages from the working set (documented behaviour);
  // it then reports ERROR_NOT_LOCKED, which is expected here.
  (void)VirtualUnlock(const_cast<void*>(addr), len);
}

void sleep_us(int64_t us) {
  if (us <= 0) return;
  static const int64_t freq = [] {
    LARGE_INTEGER f{};
    QueryPerformanceFrequency(&f);
    return int64_t(f.QuadPart);
  }();
  LARGE_INTEGER t0{}, t{};
  QueryPerformanceCounter(&t0);
  const int64_t ticks = us * freq / 1000000;
  do {
    if (!SwitchToThread()) YieldProcessor();
    QueryPerformanceCounter(&t);
  } while (int64_t(t.QuadPart - t0.QuadPart) < ticks);
}

bool narrow_paths_are_utf8() { return GetACP() == CP_UTF8; }

std::string utf8_from_wide(const wchar_t* s) {
  if (s == nullptr || *s == L'\0') return {};
  const int n = WideCharToMultiByte(CP_UTF8, 0, s, -1, nullptr, 0, nullptr, nullptr);
  if (n <= 1) return {};
  std::string out(size_t(n), '\0');
  WideCharToMultiByte(CP_UTF8, 0, s, -1, out.data(), n, nullptr, nullptr);
  out.pop_back();  // the terminating NUL
  return out;
}

TimerResolution::TimerResolution() { held_ = timeBeginPeriod(1) == TIMERR_NOERROR; }
TimerResolution::~TimerResolution() {
  if (held_) timeEndPeriod(1);
}

#else  // POSIX

MapResult map_file(const std::string& path, const uint8_t** data, uint64_t* size, std::string* reason) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    if (reason) *reason = std::strerror(errno);
    return MapResult::kOpenFailed;
  }
  struct stat st {};
  if (fstat(fd, &st) != 0) {
    if (reason) *reason = std::strerror(errno);
    ::close(fd);
    return MapResult::kStatFailed;
  }
  if (!S_ISREG(st.st_mode)) {
    ::close(fd);
    return MapResult::kNotRegular;
  }
  const uint64_t n = static_cast<uint64_t>(st.st_size);
  if (n == 0) {
    ::close(fd);
    *data = nullptr;
    *size = 0;
    return MapResult::kOk;
  }
  void* m = mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0);
  const int map_errno = errno;
  ::close(fd);
  if (m == MAP_FAILED) {
    if (reason) *reason = std::strerror(map_errno);
    return MapResult::kMapFailed;
  }
  *data = static_cast<const uint8_t*>(m);
  *size = n;
  return MapResult::kOk;
}

void unmap_file(const void* data, uint64_t size) {
  if (data != nullptr) munmap(const_cast<void*>(data), size);
}

size_t page_size() {
  const long page = sysconf(_SC_PAGESIZE);
  return page > 0 ? size_t(page) : 4096;
}

void advise_willneed(const void* addr, size_t len) { (void)madvise(const_cast<void*>(addr), len, MADV_WILLNEED); }
void advise_dontneed(const void* addr, size_t len) { (void)madvise(const_cast<void*>(addr), len, MADV_DONTNEED); }

void sleep_us(int64_t us) {
  const timespec ts{time_t(us / 1000000), long((us % 1000000) * 1000)};
  nanosleep(&ts, nullptr);
}

bool narrow_paths_are_utf8() { return true; }

TimerResolution::TimerResolution() = default;
TimerResolution::~TimerResolution() = default;

#endif

}  // namespace kt2n::platform

#if defined(_WIN32)
// include/kt2n/posix_compat.h
int setenv(const char* name, const char* value, int overwrite) {
  if (!overwrite) {
    size_t len = 0;
    if (getenv_s(&len, nullptr, 0, name) == 0 && len > 0) return 0;
  }
  return _putenv_s(name, value) == 0 ? 0 : -1;
}

int unsetenv(const char* name) { return _putenv_s(name, "") == 0 ? 0 : -1; }

int clock_gettime(int /*clock_id*/, struct timespec* ts) {
  static const int64_t freq = [] {
    LARGE_INTEGER f{};
    QueryPerformanceFrequency(&f);
    return int64_t(f.QuadPart);
  }();
  LARGE_INTEGER t{};
  QueryPerformanceCounter(&t);
  const int64_t c = int64_t(t.QuadPart);
  ts->tv_sec = time_t(c / freq);
  ts->tv_nsec = long((c % freq) * 1000000000 / freq);
  return 0;
}

int nanosleep(const struct timespec* req, struct timespec* rem) {
  kt2n::platform::sleep_us(int64_t(req->tv_sec) * 1000000 + (int64_t(req->tv_nsec) + 999) / 1000);
  if (rem) *rem = timespec{};
  return 0;
}

int pthread_setaffinity_np(void* thread, size_t /*size*/, const cpu_set_t* set) {
  // Like Linux, pin to the requested CPUs that this process may use (SetThreadAffinityMask refuses the whole mask
  // when it names a CPU outside the process mask); none of them: an error.
  DWORD_PTR process = 0, system = 0;
  DWORD_PTR mask = DWORD_PTR(set->mask);
  if (GetProcessAffinityMask(GetCurrentProcess(), &process, &system)) mask &= process;
  if (mask == 0) return -1;
  return SetThreadAffinityMask(static_cast<HANDLE>(thread), mask) != 0 ? 0 : -1;
}
#endif
