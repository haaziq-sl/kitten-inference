// Windows only: the few simple POSIX calls the engine, the command line tool and the tests make, implemented over
// Win32 (src/core/platform.cc), so that code reads the same on both systems. POSIX builds include the system
// headers instead and never see this file's declarations.
#pragma once

#if defined(_WIN32)

#include <io.h>       // read / write / close (UCRT POSIX names)
#include <process.h>  // getpid (UCRT POSIX name of _getpid)
#include <stddef.h>
#include <stdint.h>
#include <time.h>  // struct timespec (C11)

// Environment (UCRT _putenv_s; it also updates the process environment block that the CUDA driver reads).
int setenv(const char* name, const char* value, int overwrite);
int unsetenv(const char* name);

// Clocks and sleeps. CLOCK_MONOTONIC is the performance counter (what std::chrono::steady_clock reads on MSVC).
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
int clock_gettime(int clock_id, struct timespec* ts);
// Waits on the performance counter, yielding the core (kt2n::platform::sleep_us): Windows sleeps would round a
// 20 us poll pause up to 1-15.6 ms.
int nanosleep(const struct timespec* req, struct timespec* rem);

// Thread pinning: an affinity mask over logical CPUs 0-63 (SetThreadAffinityMask).
typedef struct {
  unsigned long long mask;
} cpu_set_t;
#define CPU_SETSIZE 64
#define CPU_ZERO(set) ((set)->mask = 0ull)
#define CPU_SET(cpu, set) ((set)->mask |= 1ull << (cpu))
// `thread` is std::thread::native_handle() (a HANDLE on MSVC). Returns 0 on success.
int pthread_setaffinity_np(void* thread, size_t size, const cpu_set_t* set);

#endif  // _WIN32
