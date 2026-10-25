"""Fast checks of the installed package: engine extension, public API, packaging. No downloads."""
import importlib
import importlib.metadata as metadata
import re
import threading
import unittest
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
        for name in ("micro-0.8", "mini-0.8", "nano-0.1"):
            self.assertNotIn("speed_priors", gm.BUILTIN_NATIVE_MODELS[f"kittenml/kitten-tts-{name}"])
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

    def test_an_unavailable_backend_raises_and_frees_the_gate(self):
        with self.assertRaises(ValueError):
            with self.gate.use(self.engine, "cuda"):
                pass
        with self.gate.use(self.engine, "cpu"):
            pass
        self.assertEqual(self.engine.calls, ["cuda", "cpu"])


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
