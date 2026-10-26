"""Intel CPUs with performance and efficiency cores (kitten_inference/tts08/_hybrid.py). No downloads: the Linux and
Windows detection is fed with made-up system data; the real system calls run where they can."""
import os
import shutil
import struct
import sys
import tempfile
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
        self.assertTrue(_hybrid.linux_is_hybrid(root, allowed=range(24)))
        self.assertTrue(_hybrid.linux_is_hybrid(root, allowed=[0, 16]))
        # Only the CPUs this process may use count: P-cores only, or E-cores only, is not hybrid.
        self.assertFalse(_hybrid.linux_is_hybrid(root, allowed=range(16)))
        self.assertFalse(_hybrid.linux_is_hybrid(root, allowed=range(16, 24)))

    def test_other_cpus(self):
        self.assertIsNone(_hybrid.linux_performance_cpus(self.sysfs(None)))
        self.assertFalse(_hybrid.linux_is_hybrid(self.sysfs(None), allowed=range(24)))
        self.assertFalse(_hybrid.linux_is_hybrid(self.sysfs("garbage"), allowed=range(24)))
        self.assertFalse(_hybrid.linux_is_hybrid(self.sysfs(""), allowed=range(24)))

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux only")
    def test_real_system(self):
        perf = _hybrid.linux_performance_cpus()
        self.assertIs(_hybrid.hybrid_cpu(), bool(perf) and _hybrid.linux_is_hybrid())


class WindowsTests(unittest.TestCase):
    def test_hybrid_cpu_sets(self):
        sets = _hybrid.parse_windows_cpu_sets(i9_12900_sets())
        self.assertEqual(len(sets), 24)
        self.assertEqual(sets[0], (0, 0, 0, 1))
        self.assertEqual(sets[23], (0, 23, 23, 0))
        self.assertTrue(_hybrid.windows_is_hybrid(sets))

    def test_three_efficiency_classes(self):
        # Core Ultra (Meteor Lake) style: P-cores, E-cores and low-power E-cores.
        buf = b"".join([cpu_set(i, 0, i, i - i % 2, 2) for i in range(4)] +
                       [cpu_set(i, 0, i, i, 1) for i in range(4, 12)] +
                       [cpu_set(i, 0, i, i, 0) for i in range(12, 14)])
        self.assertTrue(_hybrid.windows_is_hybrid(_hybrid.parse_windows_cpu_sets(buf)))

    def test_one_efficiency_class(self):
        buf = b"".join(cpu_set(i, 0, i, i - i % 2, 0) for i in range(16))
        self.assertFalse(_hybrid.windows_is_hybrid(_hybrid.parse_windows_cpu_sets(buf)))
        self.assertFalse(_hybrid.windows_is_hybrid([]))
        self.assertFalse(_hybrid.windows_is_hybrid(None))

    def test_records_are_walked_by_their_own_size(self):
        # A longer record from a newer Windows, and a record of another type, between ordinary ones.
        buf = (cpu_set(0, 0, 0, 0, 1) + cpu_set(1, 0, 1, 0, 1, size=48) + struct.pack("<II", 16, 7) + bytes(8) +
               cpu_set(2, 0, 2, 2, 0))
        self.assertEqual(_hybrid.parse_windows_cpu_sets(buf), [(0, 0, 0, 1), (0, 1, 0, 1), (0, 2, 2, 0)])

    def test_processor_groups(self):
        buf = b"".join([cpu_set(g * 64 + i, g, i, i - i % 2, 1) for g in (0, 1) for i in range(4)] +
                       [cpu_set(200 + i, 1, 4 + i, 4 + i, 0) for i in range(4)])
        sets = _hybrid.parse_windows_cpu_sets(buf)
        self.assertEqual([s[0] for s in sets], [0] * 4 + [1] * 8)
        self.assertTrue(_hybrid.windows_is_hybrid(sets))

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
                self.assertTrue(_hybrid.hybrid_cpu())
            with mock.patch.object(_hybrid, "_is_intel", return_value=False):
                self.assertFalse(_hybrid.hybrid_cpu())

    def test_detection_errors(self):
        with mock.patch.object(_hybrid.sys, "platform", "win32"), \
                mock.patch.object(_hybrid, "_is_intel", return_value=True), \
                mock.patch.object(_hybrid, "_windows_cpu_sets", side_effect=OSError("no such function")):
            self.assertFalse(_hybrid.hybrid_cpu())

    @unittest.skipUnless(sys.platform == "win32", "Windows only")
    def test_real_system(self):
        # The real GetSystemCpuSetInformation call: one CPU set per logical processor.
        sets = _hybrid._windows_cpu_sets()
        self.assertTrue(sets)
        self.assertEqual(len(sets), os.cpu_count())
        self.assertIsInstance(_hybrid.hybrid_cpu(), bool)


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

    def test_package_turns_it_on_only_for_hybrid_cpus(self):
        from kitten_inference.tts08 import cpp_model

        class FakeEngine:
            def __init__(self):
                self.calls = []

            def set_dynamic_split(self, n):
                self.calls.append(n)

        for hybrid, want in ((True, [_hybrid.DYNAMIC_SPLIT]), (False, [])):
            engine = FakeEngine()
            with mock.patch.object(cpp_model, "_cpu_setup_done", False), \
                    mock.patch.object(_hybrid, "hybrid_cpu", return_value=hybrid):
                cpp_model._set_up_cpu(engine)
                cpp_model._set_up_cpu(engine)  # once per process
            self.assertEqual(engine.calls, want)


if __name__ == "__main__":
    unittest.main()
