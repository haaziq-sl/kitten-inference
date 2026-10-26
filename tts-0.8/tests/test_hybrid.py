"""Intel CPUs with performance and efficiency cores (kitten_inference/tts08/_hybrid.py). No downloads: the Linux and
Windows detection is fed with made-up system data; the real system calls run where they can."""
import os
import shutil
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from kitten_inference.tts08 import _hybrid


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
        self.assertEqual(_hybrid.parse_cpulist("0-15\n"), frozenset(range(16)))
        self.assertEqual(_hybrid.parse_cpulist("0-3,8,10-11"), frozenset({0, 1, 2, 3, 8, 10, 11}))
        for bad in ("", "\n", "x", "3-1", "1-", "-2", "0-3,a"):
            self.assertIsNone(_hybrid.parse_cpulist(bad), bad)


class LinuxTests(unittest.TestCase):
    def sysfs(self, perf):
        """A made-up /sys with the kernel's cpu_core list (None: not an Intel hybrid CPU)."""
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        if perf is not None:
            (root / "devices" / "cpu_core").mkdir(parents=True)
            (root / "devices" / "cpu_core" / "cpus").write_text(perf + "\n")
        return str(root)

    def test_hybrid_cpu(self):
        # Core i9-12900: CPUs 0-15 are the 8 P-cores (two threads each), 16-23 the 8 E-cores.
        root = self.sysfs("0-15")
        self.assertEqual(_hybrid.linux_performance_cpus(root), frozenset(range(16)))
        self.assertEqual(_hybrid.linux_performance_threads(root, allowed=range(24)), 16)
        # Only the CPUs this process may use count.
        self.assertEqual(_hybrid.linux_performance_threads(root, allowed=[0, 1, 2, 16, 17]), 3)
        # P-cores only, or E-cores only: nothing to balance, OpenMP's default stays.
        self.assertIsNone(_hybrid.linux_performance_threads(root, allowed=range(16)))
        self.assertIsNone(_hybrid.linux_performance_threads(root, allowed=range(16, 24)))

    def test_other_cpus(self):
        self.assertIsNone(_hybrid.linux_performance_cpus(self.sysfs(None)))
        for perf in (None, "garbage", ""):
            self.assertIsNone(_hybrid.linux_performance_threads(self.sysfs(perf), allowed=range(24)), perf)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux only")
    def test_real_system(self):
        threads = _hybrid.performance_threads()
        if _hybrid.linux_performance_cpus() is None:
            self.assertIsNone(threads)
        else:
            self.assertTrue(threads is None or 1 <= threads < os.cpu_count())


class WindowsTests(unittest.TestCase):
    def test_hybrid_cpu_sets(self):
        sets = _hybrid.parse_windows_cpu_sets(i9_12900_sets())
        self.assertEqual(len(sets), 24)
        self.assertEqual(sets[0], (0, 0, 0, 1))
        self.assertEqual(sets[23], (0, 23, 23, 0))
        self.assertEqual(_hybrid.windows_performance_threads(sets), 16)

    def test_three_efficiency_classes(self):
        # Core Ultra (Meteor Lake) style: P-cores 2, E-cores 1, low-power E-cores 0. Only the P-cores count.
        buf = b"".join([cpu_set(i, 0, i, i - i % 2, 2) for i in range(4)] +
                       [cpu_set(i, 0, i, i, 1) for i in range(4, 12)] +
                       [cpu_set(i, 0, i, i, 0) for i in range(12, 14)])
        self.assertEqual(_hybrid.windows_performance_threads(_hybrid.parse_windows_cpu_sets(buf)), 4)

    def test_one_efficiency_class(self):
        buf = b"".join(cpu_set(i, 0, i, i - i % 2, 0) for i in range(16))
        self.assertIsNone(_hybrid.windows_performance_threads(_hybrid.parse_windows_cpu_sets(buf)))
        self.assertIsNone(_hybrid.windows_performance_threads([]))
        self.assertIsNone(_hybrid.windows_performance_threads(None))

    def test_records_are_walked_by_their_own_size(self):
        # A longer record from a newer Windows, and a record of another type, between ordinary ones.
        buf = (cpu_set(0, 0, 0, 0, 1) + cpu_set(1, 0, 1, 0, 1, size=48) + struct.pack("<II", 16, 7) + bytes(8) +
               cpu_set(2, 0, 2, 2, 0))
        sets = _hybrid.parse_windows_cpu_sets(buf)
        self.assertEqual(sets, [(0, 0, 0, 1), (0, 1, 0, 1), (0, 2, 2, 0)])
        self.assertEqual(_hybrid.windows_performance_threads(sets), 2)

    def test_processor_groups(self):
        buf = b"".join([cpu_set(g * 64 + i, g, i, i - i % 2, 1) for g in (0, 1) for i in range(4)] +
                       [cpu_set(200 + i, 1, 4 + i, 4 + i, 0) for i in range(4)])
        sets = _hybrid.parse_windows_cpu_sets(buf)
        self.assertEqual([s[0] for s in sets], [0] * 4 + [1] * 8)
        self.assertEqual(_hybrid.windows_performance_threads(sets), 8)

    def test_malformed_buffers(self):
        good = i9_12900_sets()
        for bad in (good[:-1], good + b"\x01", struct.pack("<II", 4, 0), struct.pack("<II", 1000, 0) + bytes(24),
                    struct.pack("<II", 12, 0) + bytes(4)):
            self.assertIsNone(_hybrid.parse_windows_cpu_sets(bad))
        self.assertEqual(_hybrid.parse_windows_cpu_sets(b""), [])

    def test_only_intel_cpus(self):
        sets = _hybrid.parse_windows_cpu_sets(i9_12900_sets())
        with mock.patch.object(_hybrid.sys, "platform", "win32"), \
                mock.patch.object(_hybrid, "_windows_cpu_sets", return_value=sets):
            with mock.patch.object(_hybrid, "_is_intel", return_value=True):
                self.assertEqual(_hybrid.performance_threads(), 16)
            with mock.patch.object(_hybrid, "_is_intel", return_value=False):
                self.assertIsNone(_hybrid.performance_threads())

    def test_detection_errors(self):
        with mock.patch.object(_hybrid.sys, "platform", "win32"), \
                mock.patch.object(_hybrid, "_is_intel", return_value=True), \
                mock.patch.object(_hybrid, "_windows_cpu_sets", side_effect=OSError("no such function")):
            self.assertIsNone(_hybrid.performance_threads())

    @unittest.skipUnless(sys.platform == "win32", "Windows only")
    def test_real_system(self):
        # The real GetSystemCpuSetInformation call: one CPU set per logical processor.
        sets = _hybrid._windows_cpu_sets()
        self.assertTrue(sets)
        self.assertEqual(len(sets), os.cpu_count())
        threads = _hybrid.performance_threads()
        self.assertTrue(threads is None or 1 <= threads < os.cpu_count())


class DefaultThreadsTests(unittest.TestCase):
    def test_omp_num_threads_wins(self):
        self.assertEqual(_hybrid.default_threads(16, environ={}), 16)
        self.assertEqual(_hybrid.default_threads(16, environ={"OMP_NUM_THREADS": " "}), 16)
        self.assertIsNone(_hybrid.default_threads(16, environ={"OMP_NUM_THREADS": "4"}))
        self.assertIsNone(_hybrid.default_threads(None, environ={}))


class EngineTests(unittest.TestCase):
    def test_dynamic_split_setting(self):
        from kitten_inference.tts08 import _engine

        self.addCleanup(_engine.set_dynamic_split, _engine.dynamic_split())
        with self.assertRaises(ValueError):
            _engine.set_dynamic_split(-1)
        _engine.set_dynamic_split(4)
        self.assertEqual(_engine.dynamic_split(), 4)
        _engine.set_dynamic_split(0)
        self.assertEqual(_engine.dynamic_split(), 0)

    def test_default_threads_apply_to_threads_without_their_own(self):
        from kitten_inference.tts08 import _engine

        self.addCleanup(_engine.set_default_threads, _engine.default_threads())
        with self.assertRaises(ValueError):
            _engine.set_default_threads(-1)
        seen = {}

        def worker(name, explicit=None):
            if explicit:
                _engine.set_num_threads(explicit)
            seen[name] = _engine.max_threads()

        def run(*args):
            t = threading.Thread(target=worker, args=args)
            t.start()
            t.join()

        run("openmp", 2)
        if seen["openmp"] != 2:
            self.skipTest("this build has no OpenMP (one thread)")
        _engine.set_default_threads(3)
        self.assertEqual(_engine.default_threads(), 3)
        run("default")
        run("explicit", 2)
        self.assertEqual(seen["default"], 3)
        self.assertEqual(seen["explicit"], 2)  # set_num_threads() wins in its own thread
        _engine.set_default_threads(0)
        run("none")
        self.assertGreaterEqual(seen["none"], 1)


class PackageTests(unittest.TestCase):
    """What the package sets in the engine once per process (cpp_model._set_up_cpu)."""

    class FakeEngine:
        def __init__(self):
            self.calls = []

        def set_dynamic_split(self, n):
            self.calls.append(("split", n))

        def set_default_threads(self, n):
            self.calls.append(("threads", n))

    def run_setup(self, perf_threads, environ):
        from kitten_inference.tts08 import cpp_model

        engine = self.FakeEngine()
        with mock.patch.object(cpp_model, "_cpu_setup_done", False), \
                mock.patch.object(_hybrid, "performance_threads", return_value=perf_threads), \
                mock.patch.dict(os.environ, environ):
            if "OMP_NUM_THREADS" not in environ:
                os.environ.pop("OMP_NUM_THREADS", None)
            cpp_model._set_up_cpu(engine)
            cpp_model._set_up_cpu(engine)  # once per process
        return engine.calls

    def test_hybrid_cpu(self):
        self.assertEqual(self.run_setup(16, {}), [("split", _hybrid.DYNAMIC_SPLIT), ("threads", 16)])

    def test_hybrid_cpu_with_omp_num_threads(self):
        self.assertEqual(self.run_setup(16, {"OMP_NUM_THREADS": "4"}), [("split", _hybrid.DYNAMIC_SPLIT)])

    def test_other_cpus(self):
        self.assertEqual(self.run_setup(None, {}), [])


if __name__ == "__main__":
    unittest.main()
