"""Cloning preparation integrity and separation from ordinary inference assets."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from kt2n_assets import clone_assets, convert_s3gen
from kt2n_assets.prepare import main
from kt2n_assets.safetensors_raw import SafetensorsFile, write_safetensors


@pytest.fixture
def exports(tmp_path, monkeypatch):
    data = b"\x00\x00\x80\x3f"
    digest = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(clone_assets, "CONTRACTS", {
        "consts": {"window": ([1], digest)}, "pyannote": {"weight": ([1], digest)}})
    consts, pyannote, acoustic = (tmp_path / n for n in ("consts", "pyannote", "acoustic"))
    write_safetensors(consts, {"window": ("F32", (1,), data)}, {"source": "private-source"})
    write_safetensors(pyannote, {"weight": ("F32", (1,), data),
        "bank.embeddings": ("F32", (1,), data)}, {"voice_ids": "private-bank"})
    acoustic.write_bytes(b"pinned acoustic")
    monkeypatch.setattr(convert_s3gen, "PINNED_SHA256", hashlib.sha256(acoustic.read_bytes()).hexdigest())
    return acoustic, consts, pyannote


def test_clone_exports_keep_exact_network_bytes_and_strip_bank(exports, tmp_path):
    destination = tmp_path / "out"
    provenance = clone_assets.prepare_clone_files(*exports, destination)
    assert provenance["legacy_speaker_bank"] == "removed"
    assert (destination / "s3gen.safetensors").read_bytes() == exports[0].read_bytes()
    with SafetensorsFile(destination / "voicebuild_pyannote.safetensors") as sf:
        assert set(sf.by_name) == {"weight"}
        assert bytes(sf.raw("weight")) == b"\x00\x00\x80\x3f"
        assert "voice_ids" not in sf.metadata
    with SafetensorsFile(destination / "voicebuild_consts.safetensors") as sf:
        assert sf.metadata["source"] == "native-cloning-constants-v1"


@pytest.mark.parametrize("dtype,shape,data", [
    ("F32", (1,), b"\x00\x00\x00\x40"),
    ("I32", (1,), b"\x00\x00\x80\x3f"),
    ("F32", (1, 1), b"\x00\x00\x80\x3f")])
def test_changed_weight_dtype_or_shape_rejected(exports, tmp_path, dtype, shape, data):
    write_safetensors(exports[2], {"weight": (dtype, shape, data)})
    with pytest.raises(ValueError, match="checksum/shape"):
        clone_assets.prepare_clone_files(*exports, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_missing_network_tensor_rejected(exports, tmp_path):
    write_safetensors(exports[2], {})
    with pytest.raises(ValueError, match="tensor names"):
        clone_assets.prepare_clone_files(*exports, tmp_path / "out")


def test_acoustic_mismatch_rejected(exports, tmp_path):
    exports[0].write_bytes(b"modified")
    with pytest.raises(ValueError, match="acoustic source checksum"):
        clone_assets.prepare_clone_files(*exports, tmp_path / "out")


def test_existing_clone_assets_not_overwritten(exports, tmp_path):
    destination = tmp_path / "out"
    destination.mkdir()
    (destination / "keep").write_bytes(b"existing")
    with pytest.raises(ValueError, match="already exists"):
        clone_assets.prepare_clone_files(*exports, destination)
    assert (destination / "keep").read_bytes() == b"existing"


def test_cloning_rejects_incomplete_explicit_sources(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--out", str(tmp_path), "--with-cloning", "--clone-consts", "constants"])
    assert exc.value.code == 2


def test_original_clone_tensor_contract_is_complete():
    assert len(clone_assets.CONTRACTS["consts"]) == 9
    assert len(clone_assets.CONTRACTS["pyannote"]) == 49
    assert all(not key.startswith("bank.") for key in clone_assets.CONTRACTS["pyannote"])


def _bundle(tmp_path, monkeypatch):
    from kt2n_assets import prepare
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "voices.kt2n").write_bytes(b"catalogue")
    monkeypatch.setattr(prepare.voices, "read", lambda _: SimpleNamespace(
        voices=[{"id": "Bruno", "kind": "builtin"}],
        arrays={"Bruno": {"raw_speaker": np.ones(512, dtype="f4")}}))
    manifest = {"voices": {"file": "voices.kt2n"},
                "source": {"model_repo": "KittenML/kitten-tts-2", "revision": "c" * 40}}
    return bundle, manifest


def test_default_cloning_uses_the_model_repository_weights_without_login(exports, tmp_path, monkeypatch):
    import huggingface_hub
    import shutil
    from kt2n_assets import prepare
    bundle, manifest = _bundle(tmp_path, monkeypatch)
    snapshot = tmp_path / "snapshot"
    (snapshot / "speaker").mkdir(parents=True)
    (snapshot / "speaker" / "model.safetensors").write_bytes(b"public weights")
    calls = []

    def snapshot_download(repo, **kwargs):
        calls.append((repo, kwargs["revision"], kwargs["token"], kwargs["allow_patterns"]))
        return str(snapshot)

    def public_exports(speaker, acoustic, destination):
        assert speaker == snapshot / "speaker" / "model.safetensors" and acoustic == exports[0]
        destination.mkdir()
        shutil.copyfile(exports[1], destination / "voicebuild_consts.safetensors")
        shutil.copyfile(exports[2], destination / "voicebuild_pyannote.safetensors")
        return {"pyannote_weights_sha256": "w", "pyannote_access": "none"}

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: pytest.fail("no gated download"))
    monkeypatch.setattr(clone_assets, "public_clone_exports", public_exports)
    monkeypatch.setitem(sys.modules, "torch", None)
    args = SimpleNamespace(out=bundle, clone_consts=None, clone_pyannote=None, pyannote_model=None,
                           cache_dir=tmp_path / "cache", offline=False)
    result = prepare.prepare_cloning(args, exports[0], manifest)
    assert calls == [("KittenML/kitten-tts-2", "c" * 40, False, ["speaker/model.safetensors"])]
    assert result["version"] == 1 and result["speaker_bank_voices"] == 1 and len(result["files"]) == 3
    assert result["source"]["pyannote_access"] == "none"
    assert result["source"]["pyannote_source"] == f"KittenML/kitten-tts-2@{'c' * 40}/speaker/model.safetensors"


def test_model_dir_cloning_reads_the_local_speaker_file(exports, tmp_path, monkeypatch):
    import shutil
    from kt2n_assets import prepare
    bundle, manifest = _bundle(tmp_path, monkeypatch)
    model = tmp_path / "model"
    (model / "speaker").mkdir(parents=True)
    (model / "speaker" / "model.safetensors").write_bytes(b"public weights")
    seen = []

    def public_exports(speaker, acoustic, destination):
        seen.append(speaker)
        destination.mkdir()
        shutil.copyfile(exports[1], destination / "voicebuild_consts.safetensors")
        shutil.copyfile(exports[2], destination / "voicebuild_pyannote.safetensors")
        return {}

    monkeypatch.setattr(clone_assets, "public_clone_exports", public_exports)
    args = SimpleNamespace(out=bundle, clone_consts=None, clone_pyannote=None, pyannote_model=None,
                           cache_dir=None, offline=True)
    prepare.prepare_cloning(args, exports[0], manifest, model)
    assert seen == [model / "speaker" / "model.safetensors"]


def test_pyannote_model_override_uses_the_torch_exporter(exports, tmp_path, monkeypatch):
    import shutil
    from kt2n_assets import export_clone, prepare
    bundle, manifest = _bundle(tmp_path, monkeypatch)
    checkpoint_dir = tmp_path / "authorized-model"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "pytorch_model.bin"
    checkpoint.write_bytes(b"approved pinned model fixture")
    calls = []

    def export_model(source, acoustic, destination):
        assert source == checkpoint and acoustic == exports[0]
        calls.append("export")
        destination.mkdir()
        shutil.copyfile(exports[1], destination / "voicebuild_consts.safetensors")
        shutil.copyfile(exports[2], destination / "voicebuild_pyannote.safetensors")
        return {}

    monkeypatch.setattr(export_clone, "export", export_model)
    args = SimpleNamespace(out=bundle, clone_consts=None, clone_pyannote=None, pyannote_model=checkpoint_dir,
                           cache_dir=tmp_path / "cache", offline=True)
    result = prepare.prepare_cloning(args, exports[0], manifest)
    assert calls == ["export"] and len(result["files"]) == 3


def _speaker_file(path, tensors):
    write_safetensors(path, {name: ("F32", value.shape, value.astype("<f4").tobytes()) for name, value in tensors.items()}
                      | {"layer.num_batches_tracked": ("I64", (), b"\0" * 8)})


@pytest.fixture
def speaker_pins(monkeypatch):
    values = {"layer.weight": np.arange(6, dtype="<f4").reshape(2, 3), "layer.bias": np.ones(2, "<f4"),
              "derived.filters": np.full(4, 0.5, "<f4")}
    pins = {name: (list(v.shape), hashlib.sha256(v.tobytes()).hexdigest()) for name, v in values.items()}
    monkeypatch.setattr(clone_assets, "CONTRACTS", {"consts": {}, "pyannote": pins})
    monkeypatch.setattr(clone_assets, "DERIVED_PYANNOTE", ("derived.filters",))
    monkeypatch.setattr(clone_assets, "reference_sincnet", lambda: {"derived.filters": values["derived.filters"]})
    return values


def test_public_speaker_weights_are_checked_and_completed(speaker_pins, tmp_path):
    path = tmp_path / "model.safetensors"
    _speaker_file(path, {k: v for k, v in speaker_pins.items() if k != "derived.filters"})
    out = clone_assets.public_speaker_tensors(path)
    assert list(out) == list(clone_assets.CONTRACTS["pyannote"])  # pinned order; batch counter dropped
    assert all(np.array_equal(out[k], v) for k, v in speaker_pins.items())


@pytest.mark.parametrize("change", ["value", "missing", "extra"])
def test_public_speaker_weights_must_match_the_pins(speaker_pins, tmp_path, change):
    tensors = {k: v.copy() for k, v in speaker_pins.items() if k != "derived.filters"}
    if change == "value":
        tensors["layer.bias"][0] = 2
    elif change == "missing":
        del tensors["layer.bias"]
    else:
        tensors["other.weight"] = np.zeros(1, "<f4")
    path = tmp_path / "model.safetensors"
    _speaker_file(path, tensors)
    with pytest.raises(ValueError, match="pinned pyannote/embedding"):
        clone_assets.public_speaker_tensors(path)


def test_shipped_sincnet_reference_matches_its_pins():
    values = clone_assets.reference_sincnet()
    assert list(values) == list(clone_assets.DERIVED_PYANNOTE)
    assert [list(v.shape) for v in values.values()] == [[1, 125], [125], [80, 1, 251]]


def test_public_frontend_constants_check_the_acoustic_checksum_without_torch(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", None)
    acoustic = tmp_path / "s3gen.safetensors"
    acoustic.write_bytes(b"not the pinned decoder")
    with pytest.raises(ValueError, match="acoustic checkpoint checksum"):
        clone_assets.public_frontend_tensors(acoustic)
