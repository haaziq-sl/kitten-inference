"""Portable clone preparation: trusted input gate and exact tensor export."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from kt2n_assets import export_clone
from kt2n_assets.clone_assets import CONTRACTS, checked_tensors
from kt2n_assets.safetensors_raw import SafetensorsFile


def test_unrecognized_checkpoint_is_rejected_before_loading_pickle(tmp_path, monkeypatch):
    checkpoint = tmp_path / "pytorch_model.bin"
    checkpoint.write_bytes(b"untrusted checkpoint")
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="pyannote checkpoint checksum"):
        export_clone.checkpoint_state(checkpoint)


def test_unrecognized_acoustic_is_rejected_before_importing_preparation_stack(tmp_path, monkeypatch):
    checkpoint = tmp_path / "s3gen.safetensors"
    checkpoint.write_bytes(b"changed checkpoint")
    monkeypatch.setitem(sys.modules, "torchaudio", None)
    with pytest.raises(ValueError, match="acoustic checkpoint checksum"):
        export_clone.frontend_tensors(checkpoint)


def test_tensor_mismatch_leaves_no_export_directory(tmp_path, monkeypatch):
    def wrong(kind):
        return {name: np.zeros(shape, dtype=np.float32)
                for name, (shape, _) in CONTRACTS[kind].items()}
    monkeypatch.setattr(export_clone, "speaker_tensors", lambda _: wrong("pyannote"))
    monkeypatch.setattr(export_clone, "frontend_tensors", lambda _: wrong("consts"))
    destination = tmp_path / "export"
    with pytest.raises(ValueError, match="checksum/shape mismatch"):
        export_clone.export(tmp_path / "model", tmp_path / "acoustic", destination)
    assert not destination.exists()
    assert not list(tmp_path.iterdir())


def test_local_offline_inputs_do_not_import_download_client(tmp_path, monkeypatch, capsys):
    model = tmp_path / "model"
    model.mkdir()
    seen = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setattr(export_clone, "export", lambda *args: seen.append(args) or {"ok": True})
    assert export_clone.main(["--pyannote-model", str(model), "--acoustic-source", str(tmp_path / "s3"),
                              "--offline", "--out", str(tmp_path / "exports")]) == 0
    assert seen == [(model / "pytorch_model.bin", tmp_path / "s3", tmp_path / "exports")]
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_existing_destination_is_never_replaced(tmp_path):
    target = tmp_path / "exports"
    target.mkdir()
    (target / "keep").write_bytes(b"existing content")
    with pytest.raises(ValueError, match="already exists"):
        export_clone.export(tmp_path / "absent", tmp_path / "absent", target)
    assert (target / "keep").read_bytes() == b"existing content"


@pytest.mark.skipif(not (os.environ.get("KT2N_CLONE_PYANNOTE_MODEL") and
                        os.environ.get("KT2N_CLONE_ACOUSTIC_SOURCE")),
                    reason="set the two upstream checkpoint paths to verify reproducible exports")
def test_pinned_upstream_checkpoints_reproduce_all_58_tensors(tmp_path):
    model = Path(os.environ["KT2N_CLONE_PYANNOTE_MODEL"])
    if model.is_dir():
        model /= "pytorch_model.bin"
    acoustic = Path(os.environ["KT2N_CLONE_ACOUSTIC_SOURCE"])
    out = tmp_path / "exports"
    report = export_clone.export(model, acoustic, out)
    assert sum(row["tensors"] for row in report["files"]) == 58
    for filename, kind in (("voicebuild_consts.safetensors", "consts"),
                           ("voicebuild_pyannote.safetensors", "pyannote")):
        assert set(checked_tensors(out / filename, kind)) == set(CONTRACTS[kind])
        with SafetensorsFile(out / filename) as source:
            assert not any(name.startswith("bank.") for name in source.by_name)
            assert "voice_ids" not in (source.metadata or {})


def _pin_matches(name, value):
    import hashlib
    shape, digest = CONTRACTS["consts"][name]
    raw = np.ascontiguousarray(value, dtype="<f4").tobytes()
    return list(value.shape) == shape and hashlib.sha256(raw).hexdigest() == digest


def test_shipped_reference_constants_match_their_pins():
    reference = export_clone.reference_constants()
    assert set(reference) == set(export_clone.COMPUTED_CONSTS)
    assert all(_pin_matches(name, value) for name, value in reference.items())


def test_exact_constants_are_written_unchanged(capsys):
    reference = export_clone.reference_constants()
    out = export_clone.canonical_constants({k: v.copy() for k, v in reference.items()}, reference)
    assert all(_pin_matches(name, value) for name, value in out.items())
    assert capsys.readouterr().err == ""


def test_constants_within_tolerance_are_replaced_by_the_reference_values(capsys):
    # The AMD case: 12 of the 20,560 kaldi_mel_banks values off by up to 7.06e-6.
    reference = export_clone.reference_constants()
    computed = {k: v.copy() for k, v in reference.items()}
    flat = computed["kaldi_mel_banks"].reshape(-1)
    nonzero = np.flatnonzero(flat)[:12]
    flat[nonzero] += np.float32(7.06e-6)
    assert not _pin_matches("kaldi_mel_banks", computed["kaldi_mel_banks"])
    out = export_clone.canonical_constants(computed, reference)
    assert all(_pin_matches(name, value) for name, value in out.items())
    assert "kaldi_mel_banks: 12 of 20560 values differ" in capsys.readouterr().err


@pytest.mark.parametrize("change", ["value", "nan", "shape"])
def test_constants_outside_tolerance_fail(change):
    reference = export_clone.reference_constants()
    computed = {k: v.copy() for k, v in reference.items()}
    if change == "value":
        computed["mel24_basis"].reshape(-1)[5] += np.float32(2e-4)
        match = "mel24_basis computed on this machine differs"
    elif change == "nan":
        computed["mel24_window"][3] = np.nan
        match = "mel24_window computed on this machine differs"
    else:
        computed["kaldi_mel_banks"] = computed["kaldi_mel_banks"][:, :256]
        match = "kaldi_mel_banks has shape"
    with pytest.raises(ValueError, match=match):
        export_clone.canonical_constants(computed, reference)


def test_damaged_reference_file_is_refused(tmp_path):
    reference = export_clone.reference_constants()
    reference["poly_2_3_taps"] = reference["poly_2_3_taps"].copy()
    reference["poly_2_3_taps"][0] += np.float32(1e-7)
    damaged = tmp_path / "reference.json"
    export_clone.write_reference_constants(reference, damaged)
    with pytest.raises(ValueError, match="do not match the pin: poly_2_3_taps"):
        export_clone.reference_constants(damaged)


def test_this_machine_computes_the_reference_constants_within_tolerance():
    pytest.importorskip("torchaudio")
    pytest.importorskip("librosa")
    pytest.importorskip("scipy")
    out = export_clone.canonical_constants(export_clone.computed_constants(), export_clone.reference_constants())
    assert all(_pin_matches(name, value) for name, value in out.items())
