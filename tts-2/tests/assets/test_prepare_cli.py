"""prepare_assets: pinned grammar download, text converter discovery, weight-norm device check, summary."""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import pytest

from kt2n_assets import prepare


def fake_grammar(monkeypatch):
    files = {name: f"contents of {name}\n".encode() for name in prepare.GRAMMAR_FILES}
    monkeypatch.setattr(prepare, "GRAMMAR_FILES", {n: hashlib.sha256(d).hexdigest() for n, d in files.items()})
    calls = []

    def fetch(url):
        calls.append(url)
        return files[url.rsplit("/", 1)[1]]

    return files, calls, fetch


def test_pins_are_the_tested_public_revisions():
    assert prepare.MODEL_REPO == "KittenML/kitten-tts-2"
    assert prepare.GRAMMAR_REPO == "KittenML/kitten-text-processing"
    assert prepare.GRAMMAR_COMMIT.startswith("1988d61") and len(prepare.GRAMMAR_COMMIT) == 40
    assert set(prepare.GRAMMAR_FILES) == {"metadata.json", "tagger.fst.gz", "verbalizer.fst.gz", "post.fst.gz"}
    assert all(len(d) == 64 and set(d) <= set("0123456789abcdef") for d in prepare.GRAMMAR_FILES.values())


def test_pinned_grammar_downloads_once_and_checks_sha256(tmp_path, monkeypatch):
    files, calls, fetch = fake_grammar(monkeypatch)
    out = prepare.pinned_grammar(tmp_path, offline=False, fetch=fetch)
    assert out == tmp_path / f"kitten-text-processing-{prepare.GRAMMAR_COMMIT}" / "en"
    assert {p.name: p.read_bytes() for p in out.iterdir()} == files
    assert sorted(calls) == sorted(
        f"https://raw.githubusercontent.com/KittenML/kitten-text-processing/{prepare.GRAMMAR_COMMIT}/"
        f"kitten_text_processing/data/en/{name}" for name in files)
    calls.clear()
    assert prepare.pinned_grammar(tmp_path, offline=True, fetch=fetch) == out
    assert calls == []  # served from the cache


def test_pinned_grammar_replaces_a_corrupt_cached_file(tmp_path, monkeypatch):
    files, calls, fetch = fake_grammar(monkeypatch)
    out = prepare.pinned_grammar(tmp_path, offline=False, fetch=fetch)
    (out / "post.fst.gz").write_bytes(b"truncated")
    calls.clear()
    prepare.pinned_grammar(tmp_path, offline=False, fetch=fetch)
    assert [c.rsplit("/", 1)[1] for c in calls] == ["post.fst.gz"]
    assert (out / "post.fst.gz").read_bytes() == files["post.fst.gz"]


def test_pinned_grammar_rejects_a_bad_download(tmp_path, monkeypatch):
    _, _, fetch = fake_grammar(monkeypatch)
    with pytest.raises(ValueError, match="checksum mismatch: metadata.json"):
        prepare.pinned_grammar(tmp_path, offline=False, fetch=lambda url: b"tampered" + fetch(url))
    assert not any(tmp_path.rglob("metadata.json"))


def test_pinned_grammar_offline_needs_cache_or_grammar_dir(tmp_path, monkeypatch):
    _, calls, fetch = fake_grammar(monkeypatch)
    with pytest.raises(ValueError, match="offline mode needs --grammar-dir"):
        prepare.pinned_grammar(tmp_path, offline=True, fetch=fetch)
    assert calls == []


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8", newline="\n")
    path.chmod(0o755)
    return path


def test_text_converter_discovery_order(tmp_path, monkeypatch):
    source, cwd, bindir = tmp_path / "src", tmp_path / "cwd", tmp_path / "bin"
    for d in (source, cwd, bindir):
        d.mkdir()
    monkeypatch.setattr(prepare, "ROOT", source)
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("PATH", str(bindir))
    assert prepare.find_text_converter(None) is None
    on_path = _executable(bindir / prepare.TEXT_CONVERTER)
    assert prepare.find_text_converter(None) == on_path
    in_cwd = _executable(cwd / "build/bin" / prepare.TEXT_CONVERTER)
    assert prepare.find_text_converter(None) == in_cwd
    other_preset = _executable(source / "build/dev/bin" / prepare.TEXT_CONVERTER)
    assert prepare.find_text_converter(None) == other_preset
    default_preset = _executable(source / "build/default/bin" / prepare.TEXT_CONVERTER)
    assert prepare.find_text_converter(None) == default_preset
    in_source = _executable(source / "build/bin" / prepare.TEXT_CONVERTER)
    assert prepare.find_text_converter(None) == in_source
    explicit = _executable(tmp_path / "elsewhere" / "converter")
    assert prepare.find_text_converter(explicit) == explicit
    assert prepare.find_text_converter(tmp_path / "missing") is None


def test_missing_text_converter_is_a_clear_usage_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    with pytest.raises(SystemExit) as exc:
        prepare.main(["--out", str(tmp_path / "bundle")])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "cannot find kt2n_prepare_text_data" in err and "build/default/bin/" in err and "--text-converter" in err


def _fake_torch(cuda: bool, cuda_version, hip=None):
    return types.SimpleNamespace(__version__="2.7.1+test", version=types.SimpleNamespace(cuda=cuda_version, hip=hip),
                                 cuda=types.SimpleNamespace(is_available=lambda: cuda))


def test_weight_norm_device_without_cuda_explains_the_fix(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False, None))
    with pytest.raises(SystemExit) as exc:
        prepare.check_weight_norm_device("cuda")
    message = str(exc.value.code)
    assert "cannot use CUDA" in message and "CUDA build of torch 2.7.1" in message
    assert "leave out --weight-norm-device" in message
    prepare.check_weight_norm_device("cpu")  # a CPU fold only needs torch itself
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, "12.8"))
    prepare.check_weight_norm_device("cuda")


def test_weight_norm_device_refuses_rocm_torch(monkeypatch):
    # ROCm torch answers torch.cuda.is_available() too, but its fold differs from the tested bundle.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, None, hip="7.1"))
    with pytest.raises(SystemExit) as exc:
        prepare.check_weight_norm_device("cuda")
    message = str(exc.value.code)
    assert "ROCm (AMD) build" in message and "leave out --weight-norm-device" in message.lower()
    prepare.check_weight_norm_device("cpu")  # a CPU fold is still allowed


def test_weight_norm_device_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(SystemExit) as exc:
        prepare.check_weight_norm_device("cpu")
    message = str(exc.value.code)
    assert "torch 2.7.1" in message and "default numpy fold needs no torch" in message


def test_default_numpy_fold_needs_no_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch would raise
    prepare.check_weight_norm_device("numpy")


def test_cli_defaults_and_contract_flags(tmp_path, monkeypatch):
    seen = {}

    def stop(device):
        seen["device"] = device
        raise SystemExit(3)

    monkeypatch.setattr(prepare, "find_text_converter", lambda explicit: seen.setdefault("tool", explicit))
    monkeypatch.setattr(prepare, "check_weight_norm_device", stop)
    tool = _executable(tmp_path / "bin" / prepare.TEXT_CONVERTER)
    with pytest.raises(SystemExit) as exc:
        prepare.main(["--out", str(tmp_path / "bundle"), "--text-data-tool", str(tool)])
    assert exc.value.code == 3 and seen == {"tool": tool, "device": "numpy"}


def test_installed_package_has_no_source_tree_root(tmp_path, monkeypatch):
    # In the wheel the package lives at kitten_inference/tts2/_assets: no tools/ or CMakeLists.txt above it.
    monkeypatch.setattr(prepare, "ROOT", None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    assert prepare.find_text_converter(None) is None
    in_cwd = _executable(tmp_path / "build/default/bin" / prepare.TEXT_CONVERTER)
    assert prepare.find_text_converter(None) == in_cwd


def test_summary_prints_size_and_next_command(tmp_path, capsys):
    (tmp_path / "native").mkdir()
    (tmp_path / "native" / "model.kt2n").write_bytes(b"\0" * 1500)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8", newline="\n")
    prepare.print_summary(tmp_path, tmp_path / "config.json", cloning=False)
    out = capsys.readouterr().out.splitlines()
    assert out[0] == str(tmp_path / "config.json")
    assert out[1] == f"Bundle: {tmp_path} (2 files, 1.50 kB)"
    assert out[2].startswith(f"Next: kitten-tts-native --assets {tmp_path} --text ")
    assert "--voice Bruno" in out[2] and len(out) == 3
    prepare.print_summary(tmp_path, tmp_path / "config.json", cloning=True)
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 4 and "--reference-audio" in out[3] and "--reference-transcript" in out[3]


def test_human_size():
    assert prepare.human_size(999) == "999 bytes"
    assert prepare.human_size(2_000_000) == "2.00 MB"
    assert prepare.human_size(1_969_000_000) == "1.97 GB"


def test_pinned_wheel_checks_before_caching_and_replaces_a_bad_cached_copy(tmp_path, monkeypatch):
    wheel = b"pinned wheel bytes"
    monkeypatch.setattr(prepare, "NEMO_WHEEL_SHA256", hashlib.sha256(wheel).hexdigest())
    cached = tmp_path / "nemo_text_processing-1.2.0.whl"
    with pytest.raises(ValueError, match="downloaded NeMo data wheel checksum mismatch"):
        prepare.pinned_wheel(tmp_path, offline=False, fetch=lambda url: b"<html>captive portal</html>")
    assert not cached.exists()
    cached.write_bytes(b"<html>saved by an older version</html>")
    with pytest.raises(ValueError, match="cached NeMo data wheel .*nemo_text_processing-1.2.0.whl"):
        prepare.pinned_wheel(tmp_path, offline=True, fetch=lambda url: wheel)
    calls = []
    assert prepare.pinned_wheel(tmp_path, offline=False, fetch=lambda url: calls.append(url) or wheel) == wheel
    assert cached.read_bytes() == wheel and len(calls) == 1
    assert prepare.pinned_wheel(tmp_path, offline=True, fetch=lambda url: calls.append(url) or b"") == wheel
    assert len(calls) == 1


def test_imports_as_a_renamed_package_without_torch(tmp_path, monkeypatch):
    # The wheel copies tools/kt2n_assets to kitten_inference/tts2/_assets: relative imports only, no torch at import.
    import importlib
    import shutil
    root = tmp_path / "site"
    shutil.copytree(Path(prepare.__file__).parent, root / "wheelpkg" / "_assets",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (root / "wheelpkg" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.setitem(sys.modules, "torch", None)
    for name in ("prepare", "convert_s3gen", "clone_assets", "weight_norm", "voices", "convert"):
        module = importlib.import_module(f"wheelpkg._assets.{name}")
        assert Path(module.__file__).is_relative_to(root)
    renamed = importlib.import_module("wheelpkg._assets.prepare")
    assert renamed.ROOT is None
    assert "speaker/model.safetensors" in renamed.CLONING_ALLOW_PATTERNS
    assert "lm/model.safetensors" in renamed.MODEL_ALLOW_PATTERNS
    assert importlib.import_module("wheelpkg._assets.clone_assets").reference_sincnet()
