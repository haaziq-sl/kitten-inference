"""kitten-tts2: wrapper options, --assets injection, --prepare-only, and pass-through to kitten-tts-native."""
import pytest

from kitten_inference.tts2 import _cli, _prepare


def test_split_args_keeps_native_values():
    own, native = _cli.split_args(["--revision", "abc", "--text", "--revision", "--voice=Bruno", "--cache-dir=/c",
                                   "--prepare-only", "--stream", "false"])
    assert own == {"--revision": "abc", "--cache-dir": "/c", "--prepare-only": True}
    assert native == ["--text", "--revision", "--voice=Bruno", "--stream", "false"]


def test_split_args_errors():
    with pytest.raises(_cli.UsageError):
        _cli.split_args(["--revision"])
    with pytest.raises(_cli.UsageError):
        _cli.split_args(["--prepare-only=yes"])
    with pytest.raises(_cli.UsageError):
        _cli.split_args(["--backend="])


def test_native_options():
    assert _cli.native_options(["--text", "--assets", "--assets=/x", "-h"]) == {"--text", "--assets", "--help"}
    assert "--assets" not in _cli.native_options(["--text", "--assets"])  # a value, not the option


@pytest.fixture
def harness(no_backends, fake_backend, monkeypatch, tmp_path):
    fake_backend("cuda")
    calls = {"prepare": [], "run": []}

    def ensure(**kw):
        calls["prepare"].append(kw)
        return tmp_path / "model"

    monkeypatch.setattr(_prepare, "ensure", ensure)
    monkeypatch.setattr(_cli, "_run", lambda cli, args: calls["run"].append((cli, args)) or 0)
    return calls


def test_injects_assets(harness):
    assert _cli.main(["--text", "Hello", "--voice", "Bruno", "--revision", "abc"]) == 0
    (kw,) = harness["prepare"]
    assert kw["revision"] == "abc" and kw["with_cloning"] is False and kw["cache_dir"] is None
    cli, args = harness["run"][0]
    assert cli.endswith("kitten-tts-native")
    assert args[:2] == ["--assets", str(kw["text_data_tool"].parents[2] / "model")]
    assert args[2:] == ["--text", "Hello", "--voice", "Bruno"]


def test_reference_audio_prepares_cloning(harness):
    _cli.main(["--text", "x", "--reference-audio", "ref.wav", "--reference-transcript", "words"])
    assert harness["prepare"][0]["with_cloning"] is True


def test_explicit_assets_is_kept(harness):
    assert _cli.main(["--assets", "/mine", "--list-voices"]) == 0
    assert harness["prepare"] == [] and harness["run"][0][1] == ["--assets", "/mine", "--list-voices"]
    assert _cli.main(["--assets", "/mine", "--revision", "x", "--list-voices"]) == 2


def test_prepare_only_prints_the_directory(harness, capsys):
    assert _cli.main(["--prepare-only", "--with-cloning", "--cache-dir", "/c"]) == 0
    assert capsys.readouterr().out.strip().endswith("model")
    assert harness["prepare"][0]["with_cloning"] is True and harness["prepare"][0]["cache_dir"] == "/c"
    assert harness["run"] == []
    assert _cli.main(["--prepare-only", "--text", "x"]) == 2
    assert _cli.main(["--with-cloning", "--text", "x"]) == 2


def test_version_needs_no_model(harness, capsys):
    assert _cli.main(["--version"]) == 0
    assert harness["prepare"] == [] and harness["run"][0][1] == ["--version"]
    assert "kitten-inference-tts2-cu12 0.2.0" in capsys.readouterr().out


def test_help(harness, capsys):
    assert _cli.main(["--help"]) == 0
    assert "--prepare-only" in capsys.readouterr().out and harness["run"][0][1] == ["--help"]
    assert harness["prepare"] == []


def test_missing_backend(no_backends, capsys):
    assert _cli.main(["--text", "x"]) == 1
    assert 'pip install "kitten-inference[tts-2]" (NVIDIA)' in capsys.readouterr().err


def test_prepare_failure_exit_status(harness, monkeypatch, capsys):
    def boom(**kw):
        raise _prepare.PrepareError("no network")
    monkeypatch.setattr(_prepare, "ensure", boom)
    assert _cli.main(["--text", "x"]) == 1
    assert "no network" in capsys.readouterr().err
