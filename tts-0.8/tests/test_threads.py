"""The default thread count on CPUs with performance and efficiency cores (kitten_inference/tts08/_threads.py).
No downloads: the Linux and Windows detection is fed with made-up system data; the real calls run where they can."""
import os
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from kitten_inference.tts08 import _threads


def cpu_set(cpu_id, group, logical, core, efficiency, size=32, kind=0):
    """One SYSTEM_CPU_SET_INFORMATION record as GetSystemCpuSetInformation writes it (32 bytes on current Windows)."""
    body = struct.pack("<IHBBBBBB", cpu_id, group, logical, core, 0, 0, efficiency, 0)
    record = struct.pack("<II", size, kind) + body
    return record + bytes(size - len(record))


def i9_12900_sets():
    # 8 P-cores with two logical CPUs each (EfficiencyClass 1), then 8 E-cores with one (EfficiencyClass 0).
    sets = [cpu_set(0x100 + i, 0, i, i - i % 2, 1) for i in range(16)]
    sets += [cpu_set(0x100 + i, 0, i, i, 0) for i in range(16, 24)]
    return b"".join(sets)


class CpuListTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(_threads.parse_cpulist("0-15\n"), frozenset(range(16)))
        self.assertEqual(_threads.parse_cpulist("0-3,8,10-11"), frozenset({0, 1, 2, 3, 8, 10, 11}))
        for bad in ("", "\n", "x", "3-1", "1-", "-2", "0-3,a"):
            self.assertIsNone(_threads.parse_cpulist(bad), bad)


class LinuxTests(unittest.TestCase):
    def sysfs(self, perf, cores):
        """A made-up /sys: perf = the cpu_core list (None: not a hybrid CPU), cores = lists of sibling CPUs."""
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        if perf is not None:
            (root / "devices" / "cpu_core").mkdir(parents=True)
            (root / "devices" / "cpu_core" / "cpus").write_text(perf + "\n")
        for siblings in cores:
            for cpu in siblings:
                topo = root / "devices" / "system" / "cpu" / f"cpu{cpu}" / "topology"
                topo.mkdir(parents=True)
                (topo / "core_cpus_list").write_text(",".join(map(str, siblings)) + "\n")
        return str(root)

    def test_hybrid_cpu_uses_the_performance_cores(self):
        # Core i9-12900: CPUs 0-15 are the 8 P-cores (pairs 0-1, 2-3, ...), 16-23 the 8 E-cores.
        root = self.sysfs("0-15", [[i, i + 1] for i in range(0, 16, 2)] + [[i] for i in range(16, 24)])
        everything = range(24)
        self.assertEqual(_threads.linux_threads(root, allowed=everything), 16)
        self.assertEqual(_threads.linux_threads(root, allowed=everything, per_core=True), 8)
        # Only the CPUs this process may use count.
        self.assertEqual(_threads.linux_threads(root, allowed=[0, 1, 2, 16, 17]), 3)
        self.assertEqual(_threads.linux_threads(root, allowed=[0, 1, 2, 16, 17], per_core=True), 2)
        # Only E-cores, or only P-cores, allowed: OpenMP's default (one thread per allowed CPU) stays.
        self.assertIsNone(_threads.linux_threads(root, allowed=range(16, 24)))
        self.assertIsNone(_threads.linux_threads(root, allowed=range(16)))

    def test_other_cpus_keep_the_openmp_default(self):
        root = self.sysfs(None, [[i, i + 12] for i in range(12)])
        self.assertIsNone(_threads.linux_threads(root, allowed=range(24)))
        self.assertIsNone(_threads.linux_threads(self.sysfs("garbage", []), allowed=range(24)))

    def test_missing_topology_with_per_core(self):
        self.assertIsNone(_threads.linux_threads(self.sysfs("0-3", []), allowed=range(8), per_core=True))

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux only")
    def test_real_system(self):
        perf = _threads.linux_performance_cpus()
        threads = _threads.default_threads(environ={})
        if perf is None:
            self.assertIsNone(threads)
        else:
            self.assertTrue(threads is None or 1 <= threads <= len(perf))


class WindowsTests(unittest.TestCase):
    def test_hybrid_cpu_sets(self):
        sets = _threads.parse_windows_cpu_sets(i9_12900_sets())
        self.assertEqual(len(sets), 24)
        self.assertEqual(sets[0], (0, 0, 0, 1))
        self.assertEqual(sets[23], (0, 23, 23, 0))
        self.assertEqual(_threads.windows_threads_from_cpu_sets(sets), 16)
        self.assertEqual(_threads.windows_threads_from_cpu_sets(sets, per_core=True), 8)

    def test_three_efficiency_classes(self):
        # Core Ultra (Meteor Lake) style: P-cores 2, E-cores 1, low-power E-cores 0. Only the P-cores count.
        buf = b"".join([cpu_set(i, 0, i, i - i % 2, 2) for i in range(4)] +
                       [cpu_set(i, 0, i, i, 1) for i in range(4, 12)] +
                       [cpu_set(i, 0, i, i, 0) for i in range(12, 14)])
        sets = _threads.parse_windows_cpu_sets(buf)
        self.assertEqual(_threads.windows_threads_from_cpu_sets(sets), 4)
        self.assertEqual(_threads.windows_threads_from_cpu_sets(sets, per_core=True), 2)

    def test_one_efficiency_class_keeps_the_default(self):
        buf = b"".join(cpu_set(i, 0, i, i - i % 2, 0) for i in range(16))
        self.assertIsNone(_threads.windows_threads_from_cpu_sets(_threads.parse_windows_cpu_sets(buf)))
        self.assertIsNone(_threads.windows_threads_from_cpu_sets([]))
        self.assertIsNone(_threads.windows_threads_from_cpu_sets(None))

    def test_records_are_walked_by_their_own_size(self):
        # A longer record from a newer Windows, and a record of another type, between ordinary ones.
        buf = (cpu_set(0, 0, 0, 0, 1) + cpu_set(1, 0, 1, 0, 1, size=48) + struct.pack("<II", 16, 7) + bytes(8) +
               cpu_set(2, 0, 2, 2, 0))
        self.assertEqual(_threads.parse_windows_cpu_sets(buf), [(0, 0, 0, 1), (0, 1, 0, 1), (0, 2, 2, 0)])
        self.assertEqual(_threads.windows_threads_from_cpu_sets(_threads.parse_windows_cpu_sets(buf)), 2)

    def test_processor_groups(self):
        # Two groups, each with logical processor and core numbers of its own: cores are told apart by group.
        buf = b"".join([cpu_set(g * 64 + i, g, i, i - i % 2, 1) for g in (0, 1) for i in range(4)] +
                       [cpu_set(200 + i, 1, 4 + i, 4 + i, 0) for i in range(4)])
        sets = _threads.parse_windows_cpu_sets(buf)
        self.assertEqual(_threads.windows_threads_from_cpu_sets(sets), 8)
        self.assertEqual(_threads.windows_threads_from_cpu_sets(sets, per_core=True), 4)

    def test_malformed_buffers(self):
        good = i9_12900_sets()
        for bad in (good[:-1], good + b"\x01", struct.pack("<II", 4, 0), struct.pack("<II", 1000, 0) + bytes(24),
                    struct.pack("<II", 12, 0) + bytes(4)):
            self.assertIsNone(_threads.parse_windows_cpu_sets(bad))
        self.assertEqual(_threads.parse_windows_cpu_sets(b""), [])

    def test_only_intel_cpus_on_windows(self):
        with mock.patch.object(_threads.sys, "platform", "win32"), \
                mock.patch.object(_threads, "_windows_cpu_sets", return_value=_threads.parse_windows_cpu_sets(i9_12900_sets())):
            with mock.patch.object(_threads, "_is_intel", return_value=True):
                self.assertEqual(_threads.default_threads(environ={}, per_core=False), 16)
                self.assertEqual(_threads.default_threads(environ={}, per_core=True), 8)
            with mock.patch.object(_threads, "_is_intel", return_value=False):
                self.assertIsNone(_threads.default_threads(environ={}))

    @unittest.skipUnless(sys.platform == "win32", "Windows only")
    def test_real_system(self):
        sets = _threads._windows_cpu_sets()
        self.assertTrue(sets)
        self.assertEqual(len(sets), os.cpu_count())
        threads = _threads.default_threads(environ={})
        self.assertTrue(threads is None or 1 <= threads <= len(sets))


class PrecedenceTests(unittest.TestCase):
    def test_omp_num_threads_wins(self):
        with mock.patch.object(_threads, "linux_threads", return_value=16), \
                mock.patch.object(_threads.sys, "platform", "linux"):
            self.assertEqual(_threads.default_threads(environ={}), 16)
            self.assertEqual(_threads.default_threads(environ={"OMP_NUM_THREADS": " "}), 16)
            self.assertIsNone(_threads.default_threads(environ={"OMP_NUM_THREADS": "4"}))

    def test_detection_errors_keep_the_default(self):
        with mock.patch.object(_threads, "linux_threads", side_effect=OSError("boom")), \
                mock.patch.object(_threads.sys, "platform", "linux"):
            self.assertIsNone(_threads.default_threads(environ={}))


class EngineTests(unittest.TestCase):
    """The engine's side: the default applies to threads that did not call set_num_threads()."""

    def test_default_and_explicit_threads(self):
        from kitten_inference.tts08 import _engine

        before = _engine.default_threads()
        self.addCleanup(_engine.set_default_threads, before)
        with self.assertRaises(ValueError):
            _engine.set_default_threads(-1)
        seen = {}

        def probe():
            _engine.set_num_threads(2)
            seen["openmp"] = _engine.max_threads() == 2

        t = threading.Thread(target=probe)
        t.start()
        t.join()
        if not seen["openmp"]:
            self.skipTest("this build has no OpenMP (one thread)")

        def worker(explicit):
            seen[explicit, "before"] = _engine.max_threads()
            if explicit:
                _engine.set_num_threads(2)
            seen[explicit, "after"] = _engine.max_threads()

        _engine.set_default_threads(3)
        self.assertEqual(_engine.default_threads(), 3)
        for explicit in (False, True):
            t = threading.Thread(target=worker, args=(explicit,))
            t.start()
            t.join()
        self.assertEqual(seen[False, "before"], 3)
        self.assertEqual(seen[False, "after"], 3)
        self.assertEqual(seen[True, "after"], 2)  # set_num_threads() wins in its own thread
        _engine.set_default_threads(0)
        t = threading.Thread(target=worker, args=(False,))
        t.start()
        t.join()
        self.assertGreaterEqual(seen[False, "before"], 1)


if __name__ == "__main__":
    unittest.main()
