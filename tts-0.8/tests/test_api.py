"""Fast checks of the installed package: engine extension, public API, packaging. No downloads."""
import importlib
import importlib.metadata as metadata
import os
import re
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from pathlib import Path

import kitten_inference.tts08 as tts08

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


class EngineTests(unittest.TestCase):
    def test_extension_loads_and_selects_cpu(self):
        from kitten_inference.tts08 import _engine

        self.assertTrue(_engine.set_backend("cpu"))
        self.assertEqual(_engine.active_backend(), "cpu")
        self.assertFalse(_engine.set_backend("no-such-backend"))
        for name in ("forward", "set_speed", "speed"):
            self.assertTrue(hasattr(_engine.InferenceModel, name), name)


class ApiTests(unittest.TestCase):
    def test_public_names(self):
        from kitten_inference.tts08 import KittenTTS, get_model, normalize_text

        self.assertTrue(callable(get_model))
        self.assertIs(tts08.get_model, get_model)
        for name in ("generate", "generate_stream", "generate_to_file", "available_voices"):
            self.assertTrue(hasattr(KittenTTS, name), name)
        self.assertEqual(normalize_text("Fig. 2"), "Figure two")

    def test_builtin_models_pin_the_weights(self):
        gm = importlib.import_module("kitten_inference.tts08.get_model")
        self.assertEqual(gm.NATIVE_WEIGHTS_REPO, "KittenML/meownn-models")
        self.assertEqual(gm.NATIVE_WEIGHTS_REVISION, "5e9bca0cc3a4e2a617f604b1333d5f34536c1eed")
        for size in ("nano", "micro", "mini"):
            for suffix in ("", "-int8"):
                self.assertIn(f"kittenml/kitten-tts-{size}-0.8{suffix}", gm.BUILTIN_NATIVE_MODELS)
        for spec in gm.BUILTIN_NATIVE_MODELS.values():
            self.assertTrue(Path(gm._builtin_arch_path(spec["arch_file"])).is_file(), spec["arch_file"])

    def test_follows_the_reference_onnx_path(self):
        # kittenml 0.9.6 (KittenTTS_1_Onnx): nano 0.8 speed priors, "…" end token, last 5000 samples dropped.
        import phonemizer

        gm = importlib.import_module("kitten_inference.tts08.get_model")
        cm = importlib.import_module("kitten_inference.tts08.cpp_model")
        for name in ("nano-0.8", "nano-0.8-fp32", "nano-0.8-int8"):
            priors = gm.BUILTIN_NATIVE_MODELS[f"kittenml/kitten-tts-{name}"]["speed_priors"]
            self.assertEqual(priors["expr-voice-4-m"], 0.9)
            self.assertEqual(priors["expr-voice-2-f"], 0.8)
        for name in ("micro-0.8", "mini-0.8"):
            self.assertNotIn("speed_priors", gm.BUILTIN_NATIVE_MODELS[f"kittenml/kitten-tts-{name}"])
        # The default model is kittenml's default, nano 0.8 with its priors; the nano-0.1 name loads the same
        # weights, so it gets the same priors.
        self.assertEqual(gm.DEFAULT_MODEL, "KittenML/kitten-tts-nano-0.8")
        self.assertIs(gm.BUILTIN_NATIVE_MODELS["kittenml/kitten-tts-nano-0.1"]["speed_priors"], gm.NANO_08_SPEED_PRIORS)
        self.assertEqual((cm.END_TOKEN, cm.TAIL_TRIM_SAMPLES), (10, 5000))
        stub = type("Stub", (), {})()
        stub.phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )
        stub.text_cleaner = cm.TextCleaner()
        ids = cm.KittenTTS_1_Cpp._phoneme_ids(stub, "Hello there.")
        self.assertEqual(float(ids[0]), 0.0)
        self.assertEqual([float(x) for x in ids[-2:]], [10.0, 0.0])

    def test_version_matches_pyproject(self):
        if not PYPROJECT.is_file():
            self.skipTest("pyproject.toml not available")
        match = re.search(r'^version = "([^"]+)"', PYPROJECT.read_text(encoding="utf-8"), re.M)
        self.assertEqual(tts08.__version__, match.group(1))


class FakeEngine:
    def __init__(self):
        self.calls = []

    def set_backend(self, name):
        self.calls.append(name)
        return name in ("cpu", "metal")


class EngineGateTests(unittest.TestCase):
    """The guard around the engine's process-wide backend (cpp_model._EngineGate), with a stand-in engine."""

    def setUp(self):
        self.gate = importlib.import_module("kitten_inference.tts08.cpp_model")._EngineGate()
        self.engine = FakeEngine()

    def run_in_thread(self, backend, inside=None, release=None):
        def call():
            with self.gate.use(self.engine, backend):
                if inside:
                    inside.set()
                if release:
                    release.wait(10)

        t = threading.Thread(target=call, daemon=True)
        t.start()
        return t

    def test_each_call_sets_its_backend(self):
        for backend in ("cpu", "metal", "cpu"):
            with self.gate.use(self.engine, backend):
                pass
        self.assertEqual(self.engine.calls, ["cpu", "metal", "cpu"])

    def test_cpu_calls_overlap(self):
        both_inside = threading.Barrier(3, timeout=10)
        threads = [threading.Thread(target=self._cpu_at_barrier, args=(both_inside,), daemon=True) for _ in range(2)]
        for t in threads:
            t.start()
        both_inside.wait()  # raises BrokenBarrierError if the two CPU calls could not run together
        for t in threads:
            t.join(10)
        self.assertEqual(self.engine.calls, ["cpu"])

    def _cpu_at_barrier(self, barrier):
        with self.gate.use(self.engine, "cpu"):
            barrier.wait()

    def test_a_metal_call_runs_alone(self):
        inside, release, cpu_inside = threading.Event(), threading.Event(), threading.Event()
        metal = self.run_in_thread("metal", inside, release)
        self.assertTrue(inside.wait(10))
        cpu = self.run_in_thread("cpu", cpu_inside)
        self.assertFalse(cpu_inside.wait(0.3), "a CPU call ran while a Metal call was running")
        release.set()
        self.assertTrue(cpu_inside.wait(10))
        metal.join(10)
        cpu.join(10)
        self.assertEqual(self.engine.calls, ["metal", "cpu"])

    def test_a_metal_call_waits_for_running_cpu_calls(self):
        inside, release, metal_inside = threading.Event(), threading.Event(), threading.Event()
        cpu = self.run_in_thread("cpu", inside, release)
        self.assertTrue(inside.wait(10))
        metal = self.run_in_thread("metal", metal_inside)
        self.assertFalse(metal_inside.wait(0.3), "a Metal call ran while a CPU call was running")
        release.set()
        self.assertTrue(metal_inside.wait(10))
        cpu.join(10)
        metal.join(10)
        self.assertEqual(self.engine.calls, ["cpu", "metal"])

    def test_cpu_calls_get_in_while_two_threads_keep_calling_metal(self):
        # Two threads generating on Metal always have one Metal call waiting; a CPU call must still get its turn.
        stop = threading.Event()

        def metal_loop():
            while not stop.is_set():
                with self.gate.use(self.engine, "metal"):
                    time.sleep(0.005)

        metals = [threading.Thread(target=metal_loop, daemon=True) for _ in range(2)]
        for t in metals:
            t.start()
        time.sleep(0.1)
        cpu_inside = threading.Event()
        cpu = self.run_in_thread("cpu", cpu_inside)
        try:
            self.assertTrue(cpu_inside.wait(5), "a CPU call waited while two threads kept calling Metal")
        finally:
            stop.set()
            for t in metals:
                t.join(10)
            cpu.join(10)

    def test_a_waiting_metal_call_goes_before_later_cpu_calls(self):
        inside, release = threading.Event(), threading.Event()
        cpu1 = self.run_in_thread("cpu", inside, release)
        self.assertTrue(inside.wait(10))
        metal_inside, metal_release = threading.Event(), threading.Event()
        metal = self.run_in_thread("metal", metal_inside, metal_release)
        time.sleep(0.2)  # the Metal call is waiting for the running CPU call
        cpu2_inside = threading.Event()
        cpu2 = self.run_in_thread("cpu", cpu2_inside)
        self.assertFalse(cpu2_inside.wait(0.3), "a later CPU call passed the waiting Metal call")
        release.set()
        self.assertTrue(metal_inside.wait(10))
        self.assertFalse(cpu2_inside.is_set())
        metal_release.set()
        self.assertTrue(cpu2_inside.wait(10))
        for t in (cpu1, metal, cpu2):
            t.join(10)
        self.assertEqual(self.engine.calls, ["cpu", "metal", "cpu"])

    def test_an_interrupted_wait_gives_up_its_place(self):
        inside, release = threading.Event(), threading.Event()
        cpu = self.run_in_thread("cpu", inside, release)
        self.assertTrue(inside.wait(10))

        def interrupted(*args, **kwargs):
            raise KeyboardInterrupt

        real_wait, self.gate._cond.wait = self.gate._cond.wait, interrupted
        with self.assertRaises(KeyboardInterrupt):
            with self.gate.use(self.engine, "metal"):
                pass
        self.gate._cond.wait = real_wait
        release.set()
        cpu.join(10)
        done = threading.Event()
        later = threading.Thread(target=lambda: self._use_and_set("metal", done), daemon=True)
        later.start()
        self.assertTrue(done.wait(5), "the interrupted call's place blocked the calls after it")
        later.join(10)

    def _use_and_set(self, backend, event):
        with self.gate.use(self.engine, backend):
            event.set()

    def test_an_unavailable_backend_raises_and_frees_the_gate(self):
        with self.assertRaises(ValueError):
            with self.gate.use(self.engine, "cuda"):
                pass
        with self.gate.use(self.engine, "cpu"):
            pass
        self.assertEqual(self.engine.calls, ["cuda", "cpu"])


class DefaultBackendTests(unittest.TestCase):
    """backend=None: Metal on Apple Silicon for fp32 models, the CPU for int8 models and everywhere else."""

    def setUp(self):
        self.cm = importlib.import_module("kitten_inference.tts08.cpp_model")
        gm = importlib.import_module("kitten_inference.tts08.get_model")
        self.arch = {size: {q: gm._builtin_arch_path(f"kitten_{q}_{size}_arch.json") for q in ("fp32", "int8")}
                     for size in ("15m", "40m", "80m")}

    def test_int8_models_are_recognised(self):
        for size, paths in self.arch.items():
            self.assertFalse(self.cm._has_int8_layers(paths["fp32"]), size)
            self.assertTrue(self.cm._has_int8_layers(paths["int8"]), size)

    def choose(self, system, machine, arch_path):
        with mock.patch.object(self.cm.sys, "platform", system), \
                mock.patch.object(self.cm.platform, "machine", return_value=machine):
            return self.cm._default_backend(FakeEngine(), arch_path)

    def test_choice(self):
        for size, paths in self.arch.items():
            self.assertEqual(self.choose("darwin", "arm64", paths["fp32"]), "metal", size)
            self.assertEqual(self.choose("darwin", "arm64", paths["int8"]), "cpu", size)
            for system, machine in (("darwin", "x86_64"), ("linux", "aarch64"), ("win32", "AMD64")):
                for q in ("fp32", "int8"):
                    self.assertEqual(self.choose(system, machine, paths[q]), "cpu", (size, system, q))


def write_w1x1(path, tensors, nbytes_extra=None):
    """A W1X1 weights file: name -> (shape, raw bytes). nbytes_extra adds to one entry's recorded size."""
    import struct

    header = 12 + 80 * len(tensors)
    blobs, entries, offset = [], [], header
    for name, (shape, data) in tensors.items():
        blob = struct.pack("<BBH", 1, 0, len(shape)) + struct.pack(f"<{len(shape)}i", *shape) + data
        extra = (nbytes_extra or {}).get(name, 0)
        entries.append(name.encode().ljust(64, b"\0") + struct.pack("<QII", offset, len(blob) + extra, 0))
        blobs.append(blob)
        offset += len(blob)
    Path(path).write_bytes(b"W1X1" + struct.pack("<II", 1, len(tensors)) + b"".join(entries) + b"".join(blobs))


class WeightFileTests(unittest.TestCase):
    """Model files can come from any repository: damaged or mismatched ones are refused with an error, never read or
    written out of bounds."""

    ARCH = {"model": "tiny", "inputs": [{"name": "x", "shape": [1, 0, 1, 4], "dtype": "float32"}],
            "outputs": [{"name": "y"}],
            "nodes": [{"name": "fc", "op": "seqgemm_fp32", "inputs": ["x"], "outputs": ["y"], "C_in": 4, "C_out": 3}]}

    def setUp(self):
        import json
        import struct

        from kitten_inference.tts08 import _engine

        self.engine = _engine
        self.engine.set_backend("cpu")
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.arch = os.path.join(self.dir.name, "arch.json")
        Path(self.arch).write_text(json.dumps(self.ARCH), encoding="utf-8")
        self.weights = os.path.join(self.dir.name, "w.bin")
        self.f32 = lambda *v: struct.pack(f"<{len(v)}f", *v)

    def load(self, tensors, **kw):
        write_w1x1(self.weights, tensors, **kw)
        return self.engine.InferenceModel(self.arch, self.weights)

    def test_a_matching_file_loads_and_runs(self):
        import numpy as np

        model = self.load({"fc.weight": ([3, 4], self.f32(*range(12))), "fc.bias": ([3], self.f32(1, 2, 3))})
        out = np.asarray(model.forward(np.ones(4, dtype=np.float32)))
        self.assertEqual(out.tolist(), [7.0, 24.0, 41.0])  # rows 0..3, 4..7, 8..11 summed, plus the bias

    def test_damaged_or_mismatched_tensors_are_refused(self):
        cases = {
            "not a whole number of 4-byte values": {"fc.weight": ([51], bytes(51))},
            "fc.weight is missing": {"fc.bias": ([3], self.f32(1, 2, 3))},
            "fc.weight has 8 values; the layer needs 12": {"fc.weight": ([2, 4], self.f32(*range(8)))},
            "fc.bias has 2 values; the layer needs 3": {"fc.weight": ([3, 4], self.f32(*range(12))),
                                                        "fc.bias": ([2], self.f32(1, 2))},
            "bytes for 12 values": {"fc.weight": ([3, 4], bytes(47))},
        }
        for message, tensors in cases.items():
            with self.subTest(message), self.assertRaisesRegex(RuntimeError, message):
                self.load(tensors)
        with self.assertRaisesRegex(RuntimeError, "outside the file"):
            self.load({"fc.weight": ([3, 4], self.f32(*range(12)))}, nbytes_extra={"fc.weight": 1 << 20})

    def test_a_deeply_nested_architecture_is_refused(self):
        Path(self.arch).write_text("[" * 100000 + "]" * 100000, encoding="utf-8")
        write_w1x1(self.weights, {})
        with self.assertRaisesRegex(RuntimeError, "nested too deeply"):
            self.engine.InferenceModel(self.arch, self.weights)


class CpuCheckTests(unittest.TestCase):
    """The engine is compiled for AVX2/FMA (x86-64) or the ARMv8.2 dot-product instructions (Linux ARM64): an older
    CPU gets an error before the engine loads, not "Illegal instruction"."""

    def setUp(self):
        self.cm = importlib.import_module("kitten_inference.tts08.cpp_model")
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(self.cm.SKIP_CPU_CHECK, None)

    def test_a_cpu_without_the_instructions_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, r"does not have AVX2 and FMA, .*Illegal instruction.*"
                                                  + self.cm.SKIP_CPU_CHECK):
            self.cm._check_cpu(["avx2", "fma"], {"sse4_2", "avx"})
        with self.assertRaisesRegex(RuntimeError, r"dot-product instructions \(asimddp\).*Raspberry Pi 4"):
            self.cm._check_cpu(["asimddp"], {"fp", "asimd"})

    def test_cases_that_are_not_refused(self):
        self.cm._check_cpu(["avx2", "fma"], {"avx2", "fma", "sse4_2"})
        self.cm._check_cpu(["native"], set())  # tuned for the build machine: cannot be checked
        self.cm._check_cpu([], set())
        with mock.patch.object(self.cm, "_cpu_features", return_value=None):  # this system cannot tell
            self.cm._check_cpu(["avx2", "fma"])
        os.environ[self.cm.SKIP_CPU_CHECK] = "1"
        self.cm._check_cpu(["avx2", "fma"], set())

    def test_cpuinfo(self):
        x86 = "processor\t: 0\nflags\t\t: fpu sse4_2 avx avx2 fma\n\nprocessor\t: 1\nflags\t\t: fpu\n"
        self.assertEqual(self.cm._cpuinfo_features(x86, "x86_64"), {"fpu", "sse4_2", "avx", "avx2", "fma"})
        arm = "processor\t: 0\nBogoMIPS\t: 108.00\nFeatures\t: fp asimd evtstrm crc32 cpuid\n"
        self.assertEqual(self.cm._cpuinfo_features(arm, "aarch64"), {"fp", "asimd", "evtstrm", "crc32", "cpuid"})
        self.assertIsNone(self.cm._cpuinfo_features("processor\t: 0\n", "x86_64"))

    def test_this_machine_passes(self):
        path = Path(self.cm.__file__).parent / "_cpu_requirements.txt"
        if not path.is_file():
            self.skipTest("not running from an installed wheel")
        print(f"build needs {self.cm._cpu_requirements() or 'nothing'}; this CPU: "
              f"{sorted((self.cm._cpu_features() or set()) & {'avx2', 'fma', 'asimddp'})}")
        self.cm._check_cpu()


class WindowsRuntimeTests(unittest.TestCase):
    """espeak-ng.dll needs MSVCP140.dll by that name: the wheel's copy is loaded only when Windows has none."""

    def setUp(self):
        self.cm = importlib.import_module("kitten_inference.tts08.cpp_model")
        self.addCleanup(self.cm._msvc_loaded.clear)

    def test_the_package_copy_is_loaded_only_for_what_windows_cannot_load(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in self.cm.MSVC_RUNTIME:
                Path(folder, name).write_bytes(b"")
            calls = []

            def windll(name, installed=("vcruntime140.dll",)):
                calls.append(name)
                if name == os.path.basename(name) and name not in installed:
                    raise OSError(f"{name} not found")
                return name

            self.assertEqual(self.cm._load_msvc_runtime(folder, windll), ["vcruntime140_1.dll", "msvcp140.dll"])
            self.assertEqual(calls, ["vcruntime140.dll", "vcruntime140_1.dll", os.path.join(folder, "vcruntime140_1.dll"),
                                     "msvcp140.dll", os.path.join(folder, "msvcp140.dll")])
            self.assertEqual(self.cm._load_msvc_runtime(folder, lambda name: name), [])  # an installed runtime
        self.assertEqual(self.cm._load_msvc_runtime(os.path.join(folder, "missing"), windll), [])

    @unittest.skipUnless(sys.platform == "win32", "Windows only")
    def test_the_wheel_carries_the_runtime_and_uses_the_installed_one(self):
        import ctypes
        from ctypes import wintypes

        folder = Path(self.cm.__file__).parent / "_msvc"
        if not (Path(self.cm.__file__).parent / "_cpu_requirements.txt").is_file():
            self.skipTest("not running from an installed wheel")
        self.assertTrue((folder / "msvcp140.dll").is_file())
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        handle = kernel32.GetModuleHandleW("msvcp140.dll")
        self.assertTrue(handle, "MSVCP140.dll is not loaded")
        buf = ctypes.create_unicode_buffer(32768)
        kernel32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
        kernel32.GetModuleFileNameW(handle, buf, len(buf))
        print(f"MSVCP140.dll loaded from {buf.value}")
        # GitHub's Windows runners have the Visual C++ redistributable installed: its copy is the one in use.
        self.assertNotEqual(Path(buf.value).parent.resolve(), folder.resolve())


class PackagingTests(unittest.TestCase):
    def test_wheel_does_not_ship_the_umbrella_init(self):
        owners = []
        for dist in metadata.distributions():
            names = {str(f).replace("\\", "/") for f in (dist.files or [])}
            if any(n.startswith("kitten_inference/tts08/_engine") for n in names):
                owners.append(dist.metadata["Name"])
                self.assertNotIn("kitten_inference/__init__.py", names)
                self.assertIn("kitten_inference/tts08/model_defs/kitten_fp32_80m_arch.json", names)
        if not owners:
            self.skipTest("not running from an installed wheel")
        self.assertEqual(len(owners), 1, owners)


if __name__ == "__main__":
    unittest.main()
