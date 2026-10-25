import os
import sys
import types
from pathlib import Path

import pytest

GPU_DIR = Path(__file__).parent / "gpu"
# Unit tests (not tests/gpu) that were skipped: KT2N_STRICT_SKIPS=1 (CI) makes them a failure, except tests for
# another operating system (skip reason "platform: ...").
_skipped = []
_gpu_tests = set()  # node ids of the tests in tests/gpu (filled at collection)


def pytest_runtest_logreport(report):
    if not report.skipped or report.nodeid in _gpu_tests:
        return
    reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
    if not str(reason).replace("Skipped: ", "").startswith("platform:"):
        _skipped.append(f"{report.nodeid} ({reason})")


def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("KT2N_STRICT_SKIPS") == "1" and _skipped:
        print(f"\nKT2N_STRICT_SKIPS: {len(_skipped)} unit test(s) skipped: {', '.join(_skipped)}", file=sys.stderr)
        session.exitstatus = 1


def pytest_collection_modifyitems(config, items):
    _gpu_tests.update(item.nodeid for item in items if GPU_DIR in Path(str(item.fspath)).parents)
    # tests/gpu runs only when asked for (python -m pytest tests/gpu); a plain run is the unit tests.
    args = [str(a) for a in config.args]
    if any("gpu" in Path(a).parts for a in args):
        return
    skip = pytest.mark.skip(reason="GPU test: run python -m pytest tests/gpu")
    for item in items:
        if GPU_DIR in Path(str(item.fspath)).parents:
            item.add_marker(skip)


@pytest.fixture
def fake_backend(tmp_path, monkeypatch):
    """Installs a fake kitten_inference._tts2_cuda module (no engine library) and returns it."""
    def install(name="cuda", abi=2, caps=("voice_cloning",)):
        module = types.ModuleType(f"kitten_inference._tts2_{name}")
        bin_dir = tmp_path / name / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        module.BACKEND = name
        module.ABI = abi
        module.VERSION = "0.2.0"
        module.LIB = tmp_path / name / "lib" / "libkt2n_engine.so"
        module.CLI = bin_dir / "kitten-tts-native"
        module.TEXT_DATA_TOOL = bin_dir / "kt2n_prepare_text_data"
        module.BUILD_INFO = "fake"
        module.CAPS = tuple(caps)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        return module
    monkeypatch.delenv("KITTEN_TTS2_BACKEND", raising=False)
    return install


@pytest.fixture
def no_backends(monkeypatch):
    """Makes both engine modules unimportable, whatever is installed."""
    from kitten_inference.tts2 import _backend
    monkeypatch.setattr(_backend, "_import", lambda name: sys.modules.get(_backend.MODULES[name]))
    for module in _backend.MODULES.values():
        monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.delenv("KITTEN_TTS2_BACKEND", raising=False)
