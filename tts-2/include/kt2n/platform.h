// The few operating-system services the engine uses that differ between POSIX (Linux) and Windows.
// POSIX builds call exactly what the engine called before (open + fstat + mmap, madvise, nanosleep, setenv);
// Windows builds use the Win32 equivalents. Implementation: src/core/platform.cc (kt2n_core).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace kt2n::platform {

enum class MapResult { kOk, kOpenFailed, kStatFailed, kNotRegular, kMapFailed };

// Maps a whole regular file read-only. POSIX: open(O_RDONLY | O_CLOEXEC) + fstat + mmap(PROT_READ, MAP_PRIVATE),
// then the descriptor is closed (the mapping stays valid). Windows: CreateFileW (the UTF-8 path converted to
// UTF-16) + CreateFileMappingW(PAGE_READONLY) + MapViewOfFile(FILE_MAP_READ), then both handles are closed (the
// view keeps the file open). An empty file gives kOk with *data = nullptr and *size = 0. On failure *data and *size
// are left as they were and *reason (optional) gets the operating system's error text.
MapResult map_file(const std::string& path, const uint8_t** data, uint64_t* size, std::string* reason = nullptr);
// Unmaps a mapping from map_file (nullptr: no-op).
void unmap_file(const void* data, uint64_t size);

// Virtual memory page size (sysconf(_SC_PAGESIZE); GetSystemInfo on Windows), 4096 if unknown.
size_t page_size();
// Paging hints for a page-aligned range of a mapping (best effort, results never depend on them).
// POSIX: madvise(MADV_WILLNEED / MADV_DONTNEED). Windows: PrefetchVirtualMemory / VirtualUnlock (drops the pages
// from the working set; they stay in the file cache).
void advise_willneed(const void* addr, size_t len);
void advise_dontneed(const void* addr, size_t len);

// A short sleep for poll loops. POSIX: nanosleep(us). Windows sleeps have a 1-15.6 ms granularity, so this waits
// on the performance counter instead, yielding the core to other ready threads (SwitchToThread) while it waits.
void sleep_us(int64_t us);

// True when narrow (char) file names reach the file system as UTF-8: always on POSIX; on Windows only when the
// process code page is UTF-8 (the CLI's manifest asks for it; Windows 10 1903 and newer honour that).
bool narrow_paths_are_utf8();

#if defined(_WIN32)
// UTF-16 (wchar_t) to UTF-8; unpaired surrogates become U+FFFD. For command-line arguments (wmain).
std::string utf8_from_wide(const wchar_t* s);
#endif

// Holds a 1 ms system timer resolution while it lives (Windows timeBeginPeriod(1)/timeEndPeriod(1), so short
// condition-variable timeouts are not rounded up to 15.6 ms). No-op on POSIX.
class TimerResolution {
 public:
  TimerResolution();
  ~TimerResolution();
  TimerResolution(const TimerResolution&) = delete;
  TimerResolution& operator=(const TimerResolution&) = delete;

 private:
  bool held_ = false;
};

}  // namespace kt2n::platform
