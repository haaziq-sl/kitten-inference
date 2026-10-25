"""Engine discovery: the installed backend package, the backend= argument and $KITTEN_TTS2_BACKEND."""
import pytest

from kitten_inference.tts2 import _backend

MISSING = 'pip install "kitten-inference[tts-2]" (NVIDIA) or "kitten-inference[tts-2-rocm]" (AMD)'


def test_missing_backend_message(no_backends):
    with pytest.raises(_backend.EngineNotInstalled) as e:
        _backend.find()
    assert MISSING in str(e.value)
    assert isinstance(e.value, ImportError)


def test_missing_requested_backend(no_backends, fake_backend):
    fake_backend("cuda")
    with pytest.raises(_backend.EngineNotInstalled) as e:
        _backend.find("rocm")
    assert 'pip install "kitten-inference[tts-2-rocm]"' in str(e.value)


def test_kittentts2_reports_missing_backend(no_backends):
    from kitten_inference.tts2 import KittenTTS2
    with pytest.raises(ImportError) as e:
        KittenTTS2()
    assert MISSING in str(e.value)


def test_found_backend(no_backends, fake_backend):
    module = fake_backend("cuda")
    b = _backend.find()
    assert (b.name, b.distribution, b.abi, b.version) == ("cuda", "kitten-inference-tts2-cu12", 2, "0.2.0")
    assert b.cli == module.CLI and b.text_data_tool == module.TEXT_DATA_TOOL and b.lib == module.LIB
    assert b.can_clone


def test_rocm_backend_cannot_clone(no_backends, fake_backend):
    fake_backend("rocm", caps=())
    b = _backend.find()
    assert b.name == "rocm" and b.distribution == "kitten-inference-tts2-rocm" and not b.can_clone


def test_env_selects_backend(no_backends, fake_backend, monkeypatch):
    fake_backend("cuda")
    fake_backend("rocm", caps=())
    monkeypatch.setenv("KITTEN_TTS2_BACKEND", "rocm")
    assert _backend.find().name == "rocm"
    monkeypatch.setenv("KITTEN_TTS2_BACKEND", "CUDA")
    assert _backend.find().name == "cuda"
    assert _backend.find("rocm").name == "rocm"  # the argument wins over the environment


def test_both_installed_prefers_present_driver(no_backends, fake_backend, monkeypatch):
    fake_backend("cuda")
    fake_backend("rocm", caps=())
    monkeypatch.setattr(_backend, "_driver_present", lambda name: name == "rocm")
    assert _backend.find().name == "rocm"
    monkeypatch.setattr(_backend, "_driver_present", lambda name: False)
    assert _backend.find().name == "cuda"


def test_unknown_backend_name(no_backends):
    with pytest.raises(ValueError, match="'cuda'.*'rocm'"):
        _backend.find("metal")


def test_abi_mismatch(no_backends, fake_backend):
    fake_backend("cuda", abi=3)
    with pytest.raises(_backend.EngineMismatch) as e:
        _backend.find()
    assert "ABI 3" in str(e.value) and "needs ABI 2" in str(e.value)
    assert 'pip install -U "kitten-inference[tts-2]"' in str(e.value)


def test_library_abi_check(tmp_path):
    from kitten_inference.tts2 import _abi
    with pytest.raises(_abi.LibraryError, match="not found"):
        _abi.load(tmp_path / "libkt2n_engine.so")
    bogus = tmp_path / "bogus.so"
    bogus.write_bytes(b"not a library")
    with pytest.raises(_abi.LibraryError, match="cannot load"):
        _abi.load(bogus)
