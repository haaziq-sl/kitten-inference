"""Fast checks of the installed package: engine extension, public API, packaging. No downloads."""
import importlib
import importlib.metadata as metadata
import re
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

    def test_version_matches_pyproject(self):
        if not PYPROJECT.is_file():
            self.skipTest("pyproject.toml not available")
        match = re.search(r'^version = "([^"]+)"', PYPROJECT.read_text(encoding="utf-8"), re.M)
        self.assertEqual(tts08.__version__, match.group(1))


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
