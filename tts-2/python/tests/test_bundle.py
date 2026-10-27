"""Reading a prepared model directory: manifest paths, sizes, SHA-256 checks and the voice list."""
import hashlib
import json
import struct

import pytest

from kitten_inference.tts2 import _bundle


def _record(root, rel, data):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"file": rel, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _voices_file(voices):
    meta = json.dumps({"format": "kt2n", "voices": voices}).encode()
    header = struct.pack("<8sIIQQQQQ8s", b"KT2NVOX1", 1, 0, 64, len(meta), 4096, 0, 4096, b"\0" * 8)
    return header + meta + bytes(4096 - 64 - len(meta))


@pytest.fixture
def model(tmp_path):
    voices = _voices_file([
        {"id": "Bruno", "public_names": ["Bruno"], "language": "en", "hidden": False},
        {"id": "Hidden", "public_names": ["Hidden"], "language": "en", "hidden": True},
        {"id": "Luna", "public_names": ["Luna", "luna-old"], "language": "fr", "hidden": False},
    ])
    native = {"version": 1, "decoder": "default",
              "lm": _record(tmp_path, "native/model.kt2n", b"lm" * 10),
              "acoustic": _record(tmp_path, "native/s3gen.kt2n", b"s3"),
              "voices": _record(tmp_path, "native/voices.kt2n", voices),
              "tokenizer": _record(tmp_path, "native/tokenizer.json", b"{}"),
              "text": {"directory": "native/text", "files": [_record(tmp_path, "native/text/tagger.fst", b"t")]},
              "source": {"revision": "c93cb7c271baa51d9ae8204d5277988906f235a2"}}
    (tmp_path / "config.json").write_text(json.dumps({"type": "KITTEN2", "native": native}))
    return tmp_path


def test_load(model):
    b = _bundle.load(model)
    assert b.lm == model / "native/model.kt2n" and b.text_data == model / "native/text"
    assert b.model_sha.startswith("c93cb7c") and b.default_voice == "Bruno" and not b.has_cloning
    assert len(b.files) == 5
    _bundle.Verifier(b).result()
    with pytest.raises(_bundle.BundleError, match="no voice cloning"):
        _bundle.load(model, need_cloning=True)


def test_voice_list(model):
    b = _bundle.load(model)
    assert _bundle.public_voices(b.voices) == [("Bruno", [], "en"), ("Luna", ["luna-old"], "fr")]


def test_size_and_hash_errors(model):
    (model / "native/s3gen.kt2n").write_bytes(b"s4")  # same size, other bytes
    b = _bundle.load(model)
    with pytest.raises(_bundle.BundleError, match=r"SHA-256 mismatch\): .*s3gen.kt2n"):
        _bundle.Verifier(b).result()
    (model / "native/s3gen.kt2n").write_bytes(b"s")
    with pytest.raises(_bundle.BundleError, match="wrong size"):
        _bundle.load(model)
    (model / "native/s3gen.kt2n").unlink()
    with pytest.raises(_bundle.BundleError, match="missing model file"):
        _bundle.load(model)


def test_unsafe_paths(model):
    config = json.loads((model / "config.json").read_text())
    config["native"]["lm"]["file"] = "../outside.kt2n"
    (model / "config.json").write_text(json.dumps(config))
    with pytest.raises(_bundle.BundleError, match="invalid path"):
        _bundle.load(model)


def test_not_prepared(tmp_path):
    with pytest.raises(_bundle.BundleError, match="not prepared"):
        _bundle.load(tmp_path)
