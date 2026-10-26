"""kitten-tts2: wrapper options, --assets injection, --prepare-only, and pass-through to kitten-tts-native."""
import types

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


def test_cloning_on_an_engine_without_it_stops_before_preparing(no_backends, fake_backend, monkeypatch, capsys):
    # The ROCm engine has no voice cloning: no 1.1 GB of cloning files, and the advice is one a pip user can follow.
    fake_backend("rocm", caps=())
    prepared, ran = [], []
    monkeypatch.setattr(_prepare, "ensure", lambda **kw: prepared.append(kw))
    monkeypatch.setattr(_cli, "_run", lambda cli, args: ran.append(args) or 0)
    for argv in (["--text", "x", "--reference-audio", "ref.wav", "--reference-transcript", "words"],
                 ["--prepare-only", "--with-cloning"]):
        assert _cli.main(argv) == 2
        err = capsys.readouterr().err
        assert "voice cloning needs the NVIDIA (CUDA) engine" in err and "--load-voice FILE" in err
        assert "rebuild" not in err
    assert prepared == [] and ran == []
    assert _cli.main(["--text", "x", "--load-voice", "me.voice"]) == 0 and len(prepared) == 1


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


def test_gpu_check_before_the_first_download(harness, monkeypatch):
    from kitten_inference.tts2 import _gpu
    checked = []
    monkeypatch.setattr(_gpu, "check", lambda backend, device=0, log=None: checked.append((backend, device)))
    assert _cli.main(["--text", "x", "--device", "1"]) == 0
    hook = harness["prepare"][-1]["before_download"]
    hook()  # ensure() calls it only when it has to download and prepare
    assert checked == [("cuda", 1)]
    assert _cli.main(["--prepare-only"]) == 0
    assert harness["prepare"][-1]["before_download"] is None  # preparing alone needs no GPU


def test_gpu_check_failure_stops_before_downloading(harness, monkeypatch, capsys):
    from kitten_inference.tts2 import _gpu

    def ensure(**kw):
        kw["before_download"]()
        raise AssertionError("downloaded")

    def no_gpu(backend, device=0, log=None):
        raise _gpu.GPUError("no NVIDIA driver found on this machine")

    monkeypatch.setattr(_prepare, "ensure", ensure)
    monkeypatch.setattr(_gpu, "check", no_gpu)
    assert _cli.main(["--text", "x"]) == 1
    assert "no NVIDIA driver found" in capsys.readouterr().err


def test_device_value():
    assert _cli._device(["--text", "x", "--device", "2"]) == 2
    assert _cli._device(["--device=3"]) == 3 and _cli._device(["--text", "x"]) == 0
    assert _cli._device(["--device", "gpu"]) == 0


# A saved voice works only with the prepared model it was made with. kitten-tts-native would only say
# "voice_artifact_version_mismatch ... (status 210)"; kitten-tts2 names the revision the voice needs.
OLD, NEW = "a" * 40, "b" * 40


@pytest.fixture
def voices(no_backends, fake_backend, monkeypatch, tmp_path):
    from test_api import _blob, _model
    fake_backend("cuda")
    cache = tmp_path / "cache"
    _model(cache, OLD, "c-old", "s1")
    voice = tmp_path / "me.voice"
    voice.write_bytes(_blob({"format": "kt2n", "checkpoint": "c-old", "s3gen_sha256": "s1", "voices": []}))
    ran, prepared = [], []
    monkeypatch.setattr(_cli, "_run", lambda cli, args: ran.append(args) or 0)
    return types.SimpleNamespace(cache=cache, voice=voice, ran=ran, prepared=prepared, model=_model)


def test_a_voice_of_a_prepared_older_revision_stops_before_the_download(voices, monkeypatch, capsys):
    from kitten_inference.tts2 import _gpu
    monkeypatch.setattr(_gpu, "check", lambda *a, **k: None)

    def ensure(**kw):  # main moved to NEW, which is not prepared yet
        kw["before_download"]()
        raise AssertionError("downloaded")

    monkeypatch.setattr(_prepare, "ensure", ensure)
    argv = ["--cache-dir", str(voices.cache), "--load-voice", str(voices.voice), "--text", "hi", "--output", "o.wav"]
    assert _cli.main(argv) == 1
    err = capsys.readouterr().err
    assert f"made with the model revision {OLD}" in err and f"--revision {OLD}" in err, err
    assert voices.ran == []


def test_a_voice_of_another_model_is_named_before_the_engine_runs(voices, monkeypatch, capsys):
    monkeypatch.setattr(_prepare, "ensure", lambda **kw: voices.model(voices.cache, NEW, "c-new", "s1"))
    argv = ["--cache-dir", str(voices.cache), "--load-voice", str(voices.voice), "--text", "hi"]
    assert _cli.main(argv) == 1
    err = capsys.readouterr().err
    assert f"revision {OLD}, which is still prepared" in err and f"this run uses {NEW}" in err, err
    assert f"add --revision {OLD}" in err and voices.ran == []
    (voices.cache / OLD / "config.json").unlink()  # the old model is gone: say what to pass
    assert _cli.main(argv) == 1
    assert "--save-voice prints it" in capsys.readouterr().err


def test_a_voice_of_this_model_runs(voices, monkeypatch, capsys):
    monkeypatch.setattr(_prepare, "ensure", lambda **kw: voices.cache / OLD)
    assert _cli.main(["--cache-dir", str(voices.cache), f"--load-voice={voices.voice}", "--text", "hi"]) == 0
    assert voices.ran[0][:2] == ["--assets", str(voices.cache / OLD)]


def test_save_voice_names_the_revision(voices, monkeypatch, capsys):
    monkeypatch.setattr(_prepare, "ensure", lambda **kw: voices.cache / OLD)
    argv = ["--cache-dir", str(voices.cache), "--reference-audio", "r.wav", "--reference-transcript", "w",
            "--save-voice", "me2.voice", "--text", "hi"]
    assert _cli.main(argv) == 0
    assert f"model revision {OLD}; a voice saved now loads only with this revision" in capsys.readouterr().err
