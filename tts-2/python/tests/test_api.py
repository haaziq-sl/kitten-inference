"""KittenTTS2 without a GPU: the cloning files are added to the model already in use (never a newer main), and a
saved voice of another prepared model is refused before the engine is touched, naming the revision it needs."""
import hashlib
import json
import struct
import types
from pathlib import Path

import pytest

from kitten_inference.tts2 import _abi, _api, _bundle, _prepare
from kitten_inference.tts2._engine import EngineError

A = "a" * 40
B = "b" * 40


def _blob(meta: dict) -> bytes:
    data = json.dumps(meta).encode()
    header = struct.pack("<8sIIQQQQQ8s", b"KT2NVOX1", 1, 0, 64, len(data), 4096, 0, 4096, b"\0" * 8)
    return header + data + bytes(max(0, 4096 - 64 - len(data)))


def _record(root: Path, rel: str, data: bytes) -> dict:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"file": rel, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _model(cache: Path, sha: str, checkpoint: str, s3gen: str, cloning: bool = False) -> Path:
    root = cache / sha
    voices = _blob({"format": "kt2n", "checkpoint": checkpoint, "s3gen_sha256": s3gen,
                    "voices": [{"id": "Bruno", "public_names": ["Bruno"], "language": "en"}]})
    native = {"version": 1, "decoder": "default",
              "lm": _record(root, "native/model.kt2n", b"lm" + sha.encode()),
              "acoustic": _record(root, "native/s3gen.kt2n", b"s3"),
              "voices": _record(root, "native/voices.kt2n", voices),
              "tokenizer": _record(root, "native/tokenizer.json", b"{}"),
              "text": {"directory": "native/text", "files": [_record(root, "native/text/t.fst", b"t")]},
              "source": {"revision": sha}}
    if cloning:
        native["cloning"] = {"directory": "native/cloning", "files": [
            _record(root, f"native/cloning/{n}", b"c") for n in
            ("s3gen.safetensors", "voicebuild_consts.safetensors", "voicebuild_pyannote.safetensors")]}
    (root / "config.json").write_text(json.dumps({"type": "KITTEN2", "native": native}))
    return root


def _tts(model_dir: Path) -> _api.KittenTTS2:
    t = _api.KittenTTS2.__new__(_api.KittenTTS2)
    t.model_dir = model_dir
    t._bundle = _bundle.load(model_dir)
    t._identity = None
    t._revision = None  # the caller asked for main
    t._cache_dir = model_dir.parent
    t._lock = __import__("threading").RLock()
    t.backend = types.SimpleNamespace(text_data_tool=Path("tool"), name="cuda")
    t._engine = None
    t._lib = types.SimpleNamespace(kt2n_voice_artifact_lm_prefix_ids=lambda data, n: 10)
    t._device, t._memory_profile, t._voice_cache, t._max_prefix_ids = 0, _api.MEMORY_PROFILES["auto"], None, 0
    t._mode, t._voice_ids, t._custom_ids, t._custom_users, t._last_voice = _api.BUILTIN, {}, {}, {}, 0
    return t


def test_revision_is_the_commit_in_use(tmp_path):
    assert _tts(_model(tmp_path, A, "c1", "s1")).revision == A


def test_first_clone_adds_cloning_to_the_running_revision(tmp_path, monkeypatch):
    t = _tts(_model(tmp_path, A, "c1", "s1"))
    asked = []

    def ensure(**kw):
        asked.append(kw)
        return _model(tmp_path, A, "c1", "s1", cloning=True)

    monkeypatch.setattr(_prepare, "ensure", ensure)
    monkeypatch.setattr(_api.KittenTTS2, "_create", lambda self, config, bundle: types.SimpleNamespace())
    t._start(_api.CLONING)
    assert asked[0]["revision"] == A and asked[0]["with_cloning"] is True  # not None (= main, maybe moved)
    assert t.model_dir == tmp_path / A and t._bundle.has_cloning


def test_voice_of_another_prepared_revision_names_it(tmp_path):
    _model(tmp_path, A, "c-old", "s1")
    t = _tts(_model(tmp_path, B, "c-new", "s1"))
    path = tmp_path / "me.voice"
    path.write_bytes(_blob({"format": "kt2n", "checkpoint": "c-old", "s3gen_sha256": "s1", "voices": []}))
    with pytest.raises(ValueError) as e:
        t.load_voice(path)
    assert f'KittenTTS2(revision="{A}")' in str(e.value) and B in str(e.value)


def test_voice_of_an_unknown_model(tmp_path):
    t = _tts(_model(tmp_path, B, "c-new", "s1"))
    path = tmp_path / "me.voice"
    path.write_bytes(_blob({"format": "kt2n", "checkpoint": "c-gone", "s3gen_sha256": "s1", "voices": []}))
    with pytest.raises(ValueError, match=r"works only with the model revision it was made with: pass that "
                                         r"revision as KittenTTS2\(revision=\.\.\.\)"):
        t.load_voice(path)


def test_voice_of_this_model_goes_to_the_engine(tmp_path, monkeypatch):
    t = _tts(_model(tmp_path, A, "c1", "s1"))
    path = tmp_path / "me.voice"
    path.write_bytes(_blob({"format": "kt2n", "checkpoint": "c1", "s3gen_sha256": "s1", "voices": []}))
    loaded = []
    monkeypatch.setattr(_api.KittenTTS2, "_voice_id", lambda self, voice: loaded.append(voice) or 5)
    v = t.load_voice(path)
    assert loaded == [v] and v.name == "me"


def test_blob_identity():
    meta = _bundle.blob_meta(_blob({"checkpoint": "c", "s3gen_sha256": "s", "voices": []}))
    assert _bundle.identity(meta) == ("c", "s")
    with pytest.raises(_bundle.BundleError):
        _bundle.blob_meta(b"RIFF....")


@pytest.mark.parametrize("engine_message", [
    # this engine, and the 0.2.0 engine (which named only the C call)
    "voice cloning is off on this GPU: the low-memory profile uses the memory the cloning model needs (about 4.4 GB "
    "more); build and save the voice on a larger GPU, then load it here (kitten-tts-native --load-voice FILE, Python "
    "load_voice(), C API kt2n_voice_load)",
    "voice cloning is off on this GPU: the low-memory profile uses the memory the cloning model needs (about 4.4 GB "
    "more); build and save the voice on a larger GPU, then load it with kt2n_voice_load",
])
def test_cloning_off_names_the_python_call(engine_message):
    err = _api._clone_error(EngineError(_abi.ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE, engine_message))
    text = str(err)
    assert isinstance(err, RuntimeError) and "about 4.4 GB more" in text, text
    assert "kt2n_voice_load" not in text and "--load-voice" not in text, text
    assert text.endswith("then load it here with load_voice()"), text
