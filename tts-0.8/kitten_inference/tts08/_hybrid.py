"""Intel CPUs with performance and efficiency cores ("hybrid" CPUs: 12th-generation Core and newer).

The engine runs each CPU operation on a team of OpenMP threads and normally gives every thread an equal share of the
work, with one thread per logical CPU. On a hybrid CPU every operation then waits for the threads on the slower
efficiency cores (E-cores), and with that many threads the GNU OpenMP runtime in the Linux wheels (from the
manylinux_2_28 build image) stalled calls for seconds on a Core i9-14900K. So on these CPUs the package:
- runs one thread per logical CPU of the performance cores (P-cores), unless OMP_NUM_THREADS or set_num_threads()
  chooses the count;
- has the engine cut its matrix multiplications into DYNAMIC_SPLIT pieces per thread, which the threads take as they
  finish, so the faster cores do more of them. Each piece is computed whole by one thread: the output is the same as
  with an equal split at the same thread count.
On other CPUs nothing changes (there the dynamic split was up to 21% slower for the int8 models).

The detection only reads what the operating system reports:
- Linux: the kernel's list of P-core CPUs, /sys/devices/cpu_core/cpus (Intel hybrid CPUs, Linux 5.13 and newer),
  compared with the CPUs this process may use;
- Windows: GetSystemCpuSetInformation, whose EfficiencyClass is highest on the P-cores (Intel CPUs only).
"""
import os
import platform
import struct
import sys

DYNAMIC_SPLIT = 4  # pieces per thread on hybrid CPUs

# SYSTEM_CPU_SET_INFORMATION (winnt.h): DWORD Size, CPU_SET_INFORMATION_TYPE Type, then for Type 0 (CpuSetInformation)
# DWORD Id, WORD Group, BYTE LogicalProcessorIndex, CoreIndex, LastLevelCacheIndex, NumaNodeIndex, EfficiencyClass,
# AllFlags, ... (32 bytes in all on current Windows). Each record gives its own Size, so later, longer versions
# still parse.
_CPU_SET_HEADER = struct.Struct("<II")
_CPU_SET_FIELDS = struct.Struct("<IHBBBBBB")
_CPU_SET_INFORMATION = 0


def parse_cpulist(text):
    """The CPUs in a Linux CPU list such as "0-15,20,22-23", as a frozenset; None if it is not one."""
    cpus = set()
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        first, sep, last = part.partition("-")
        try:
            lo = int(first)
            hi = int(last) if sep else lo
        except ValueError:
            return None
        if lo < 0 or hi < lo:
            return None
        cpus.update(range(lo, hi + 1))
    return frozenset(cpus) or None


def linux_performance_cpus(sysfs="/sys"):
    """The logical CPUs of the P-cores on an Intel hybrid CPU (the kernel's cpu_core PMU), or None."""
    try:
        with open(os.path.join(sysfs, "devices", "cpu_core", "cpus"), encoding="ascii") as f:
            return parse_cpulist(f.read())
    except (OSError, UnicodeDecodeError):
        return None


def linux_performance_threads(sysfs="/sys", allowed=None):
    """The number of P-core logical CPUs this process may use, when it may also use other cores of an Intel hybrid
    CPU; else None."""
    perf = linux_performance_cpus(sysfs)
    if not perf:
        return None
    if allowed is None:
        try:
            allowed = os.sched_getaffinity(0)
        except (AttributeError, OSError):
            return None
    allowed = frozenset(allowed)
    if not (perf & allowed) or not (allowed - perf):  # only P-cores, or only E-cores: nothing to balance
        return None
    return len(perf & allowed)


def parse_windows_cpu_sets(buf):
    """(Group, LogicalProcessorIndex, CoreIndex, EfficiencyClass) of each CPU set in a GetSystemCpuSetInformation
    buffer; None if the buffer is malformed."""
    sets = []
    offset = 0
    while offset < len(buf):
        if offset + _CPU_SET_HEADER.size > len(buf):
            return None
        size, kind = _CPU_SET_HEADER.unpack_from(buf, offset)
        if size < _CPU_SET_HEADER.size or offset + size > len(buf):
            return None
        if kind == _CPU_SET_INFORMATION:
            if size < _CPU_SET_HEADER.size + _CPU_SET_FIELDS.size:
                return None
            _id, group, logical, core, _llc, _numa, efficiency, _flags = _CPU_SET_FIELDS.unpack_from(
                buf, offset + _CPU_SET_HEADER.size)
            sets.append((group, logical, core, efficiency))
        offset += size
    return sets


def windows_performance_threads(sets):
    """The number of logical CPUs with the highest EfficiencyClass (the P-cores), when the CPU sets have more than one
    class; else None."""
    if not sets:
        return None
    classes = {s[3] for s in sets}
    if len(classes) < 2:
        return None
    best = max(classes)
    return sum(1 for s in sets if s[3] == best)


def _windows_cpu_sets():
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    query = getattr(kernel32, "GetSystemCpuSetInformation", None)  # Windows 10 and newer
    if query is None:
        return None
    query.argtypes = [ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG), wintypes.HANDLE, wintypes.ULONG]
    query.restype = wintypes.BOOL
    needed = wintypes.ULONG(0)
    query(None, 0, ctypes.byref(needed), None, 0)  # fails with ERROR_INSUFFICIENT_BUFFER and gives the size
    if not needed.value:
        return None
    buf = ctypes.create_string_buffer(needed.value)
    if not query(buf, needed, ctypes.byref(needed), None, 0):
        return None
    return parse_windows_cpu_sets(buf.raw[:needed.value])


def _is_intel():
    # e.g. "Intel64 Family 6 Model 151 Stepping 2, GenuineIntel" (from PROCESSOR_IDENTIFIER)
    return "GenuineIntel" in (platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""))


def performance_threads():
    """The number of P-core logical CPUs on an Intel CPU with performance and efficiency cores (that this process may
    use); None when it is not one or this cannot be told."""
    try:
        if sys.platform.startswith("linux"):
            return linux_performance_threads()
        if sys.platform == "win32" and _is_intel():
            return windows_performance_threads(_windows_cpu_sets())
    except Exception:  # never fail a model load over this
        return None
    return None


def default_threads(perf_threads, environ=None):
    """The engine's default thread count for a hybrid CPU with perf_threads P-core logical CPUs: None (OpenMP's
    choice) when OMP_NUM_THREADS is set."""
    environ = os.environ if environ is None else environ
    if not perf_threads or environ.get("OMP_NUM_THREADS", "").strip():
        return None
    return perf_threads
