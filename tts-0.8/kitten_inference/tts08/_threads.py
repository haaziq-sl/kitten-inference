"""The engine's default CPU thread count on Intel CPUs with performance and efficiency cores.

OpenMP's own default is one thread per logical CPU. The engine splits every operation into equal parts, one per
thread, so on a CPU with performance cores (P-cores) and slower efficiency cores (E-cores) each operation waits
for the E-core threads. On such a CPU the engine therefore defaults to one thread per logical CPU of the P-cores
(16 on a Core i9-12900, which has 8 P-cores with two threads each and 8 E-cores).

Explicit choices win: OMP_NUM_THREADS, or set_num_threads() in the calling thread. When this module cannot tell,
OpenMP's default stays. The detection only reads what the operating system reports (it never pins threads):
- Linux: the kernel's list of P-core CPUs, /sys/devices/cpu_core/cpus (Intel hybrid CPUs, Linux 5.13 and newer),
  within the CPUs this process may use;
- Windows: GetSystemCpuSetInformation, whose EfficiencyClass is higher on the P-cores (Intel CPUs only).
"""
import os
import platform
import struct
import sys

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


def _read(path):
    try:
        with open(path, encoding="ascii") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def linux_performance_cpus(sysfs="/sys"):
    """The logical CPUs of the P-cores on an Intel hybrid CPU (the kernel's cpu_core PMU), or None."""
    text = _read(os.path.join(sysfs, "devices", "cpu_core", "cpus"))
    return parse_cpulist(text) if text is not None else None


def linux_core_key(cpu, sysfs="/sys"):
    """An id shared by the logical CPUs of one physical core (its list of sibling CPUs), or None."""
    topo = os.path.join(sysfs, "devices", "system", "cpu", f"cpu{cpu}", "topology")
    for name in ("core_cpus_list", "thread_siblings_list"):
        text = _read(os.path.join(topo, name))
        if text is not None:
            return parse_cpulist(text)
    return None


def linux_threads(sysfs="/sys", allowed=None, per_core=False):
    """The default thread count on Linux, or None to keep OpenMP's (one thread per CPU this process may use).
    per_core: one thread per physical P-core instead of one per P-core logical CPU."""
    perf = linux_performance_cpus(sysfs)
    if not perf:
        return None
    if allowed is None:
        try:
            allowed = os.sched_getaffinity(0)
        except (AttributeError, OSError):
            return None
    allowed = frozenset(allowed)
    perf = perf & allowed
    # No P-core allowed (only E-cores), or only P-cores allowed: OpenMP's default is already right.
    if not perf or perf == allowed:
        return None
    if not per_core:
        return len(perf)
    cores = {linux_core_key(cpu, sysfs) for cpu in perf}
    return None if None in cores else len(cores)


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


def windows_threads_from_cpu_sets(sets, per_core=False):
    """The default thread count for these CPU sets: the logical CPUs (per_core: the physical cores) with the highest
    EfficiencyClass when there is more than one class (a hybrid CPU), else None."""
    if not sets:
        return None
    classes = {s[3] for s in sets}
    if len(classes) < 2:
        return None
    best = max(classes)
    perf = [s for s in sets if s[3] == best]
    return len({(s[0], s[2]) for s in perf}) if per_core else len(perf)


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
    query(None, 0, ctypes.byref(needed), None, 0)
    if not needed.value:
        return None
    buf = ctypes.create_string_buffer(needed.value)
    if not query(buf, needed, ctypes.byref(needed), None, 0):
        return None
    return parse_windows_cpu_sets(buf.raw[:needed.value])


def _is_intel():
    # e.g. "Intel64 Family 6 Model 151 Stepping 2, GenuineIntel" (from PROCESSOR_IDENTIFIER)
    return "GenuineIntel" in (platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""))


PER_CORE = False  # EXPERIMENT switch: True = one thread per physical P-core


def default_threads(environ=None, per_core=None):
    """The engine's default CPU thread count on this machine, or None to keep OpenMP's default."""
    environ = os.environ if environ is None else environ
    per_core = PER_CORE if per_core is None else per_core
    if environ.get("OMP_NUM_THREADS", "").strip():
        return None
    try:
        if sys.platform.startswith("linux"):
            return linux_threads(per_core=per_core)
        if sys.platform == "win32" and _is_intel():
            return windows_threads_from_cpu_sets(_windows_cpu_sets(), per_core=per_core)
    except Exception:  # never fail a model load over this
        return None
    return None
