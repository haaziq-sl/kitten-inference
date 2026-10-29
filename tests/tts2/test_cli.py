"""kitten-tts: wrapper options, --assets injection, --prepare-only, pass-through to kitten-tts-native, and what it
says when this Python, install or machine cannot run Kitten TTS 2."""
import os
import platform
import re
import sys
import types

import pytest

import kitten_inference
import kitten_inference.tts2 as tts2
from kitten_inference.tts2 import _backend, _cli, _gpu, _prepare


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
    # The installed kitten-inference's version (a post release changes only that one) and the engine's.
    version = getattr(kitten_inference, "__version__", tts2.__version__)
    assert re.sub(r"\.post\d+$", "", version) == tts2.__version__
    assert capsys.readouterr().out.strip() == f"kitten-tts {version} (NVIDIA (CUDA) engine 0.2.0)"


def test_help(harness, capsys):
    assert _cli.main(["--help"]) == 0
    assert "--prepare-only" in capsys.readouterr().out and harness["run"][0][1] == ["--help"]
    assert harness["prepare"] == []


def test_missing_backend(no_backends, capsys):
    assert _cli.main(["--text", "x"]) == 1
    assert capsys.readouterr().err == f"kitten-tts: {_backend.no_engine_message()}\n"


def test_version_on_a_platform_without_an_engine(no_backends, monkeypatch, capsys):
    # macOS, Linux aarch64, Windows ARM64, Android: Kitten TTS 0.8 runs there, and --version still answers.
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    version = getattr(kitten_inference, "__version__", tts2.__version__)
    assert _cli.main(["--version"]) == 0
    out, err = capsys.readouterr()
    assert out == f"kitten-tts {version} (no Kitten TTS 2 engine on this platform)\n"
    assert err == ("kitten-tts: Kitten TTS 2 runs on Linux x86-64 (NVIDIA or AMD Radeon GPU) and Windows x64 "
                   "(NVIDIA GPU); this is macOS arm64. Kitten TTS 0.8 (kitten_inference.tts08) runs here.\n")
    assert _cli.main(["--text", "x"]) == 1 and capsys.readouterr().out == ""
    # Linux x86-64 and Windows x64 have an engine: an install without its files is an error, --version included.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert _cli.main(["--version"]) == 1
    out, err = capsys.readouterr()
    assert out == "" and err == f"kitten-tts: {_backend.no_engine_message()}\n" and "engine files" in err


def test_missing_libraries_are_one_line_before_preparing(harness, hide_modules, capsys):
    hide_modules("numpy", "tokenizers")
    for argv in (["--text", "x"], ["--prepare-only"]):
        assert _cli.main(argv) == 1, argv
        err = capsys.readouterr().err
        assert err == ('kitten-tts: Kitten TTS 2 needs more Python packages (not installed: numpy, tokenizers). '
                       'Install them: pip install "kitten-inference"\n'), err
    assert harness["prepare"] == [] and harness["run"] == []
    # --help, --version and a model given with --assets need none of them
    assert _cli.main(["--help"]) == 0 and _cli.main(["--version"]) == 0
    assert _cli.main(["--assets", "m", "--text", "x"]) == 0 and len(harness["run"]) == 3


def test_python_38_is_one_line(harness, python38, capsys):
    for argv in (["--help"], ["--version"], ["--text", "x"]):
        assert _cli.main(argv) == 1, argv
        assert capsys.readouterr().err == ("kitten-tts: Kitten TTS 2 needs Python 3.9 or newer (this is Python 3.8). "
                                           "Kitten TTS 0.8 (kitten_inference.tts08) works on Python 3.8.\n")
    assert harness["run"] == [] and harness["prepare"] == []


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
    from tests.tts2.test_api import _blob, _model
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


def _no_runtime(monkeypatch, *missing):
    def load(name):
        if name in missing or os.path.basename(name) in missing:
            raise OSError(f"{name}: cannot open shared object file: No such file or directory")
        return object()
    monkeypatch.setattr(_gpu, "_cdll", load)


def test_missing_rocm_runtime_is_one_line_before_any_download(no_backends, fake_backend, monkeypatch, capsys):
    # The ROCm tool is linked against libamdhip64.so.7: without ROCm the loader would only print its raw error.
    fake_backend("rocm", caps=())
    prepared, ran = [], []
    monkeypatch.setattr(_prepare, "ensure", lambda **kw: prepared.append(kw) or "model")
    monkeypatch.setattr(_cli, "_run", lambda cli, args: ran.append(args) or 0)
    _no_runtime(monkeypatch, "libamdhip64.so.7")
    for argv in (["--text", "x"], ["--version"], ["--assets", "m", "--text", "x"]):
        assert _cli.main(argv) == 1, argv
        err = capsys.readouterr().err
        assert "AMD ROCm 7.x runtime (libamdhip64.so.7) was not found" in err and "amd-radeon.md" in err, err
        assert len(err.strip().splitlines()) == 1, err
    assert prepared == [] and ran == []
    assert _cli.main(["--help"]) == 0 and ran == []  # the usage is still printed
    assert "ROCm 7.x runtime" in capsys.readouterr().err
    assert _cli.main(["--prepare-only"]) == 0 and len(prepared) == 1  # preparing needs no ROCm
    _no_runtime(monkeypatch, None)
    assert _cli.main(["--text", "x"]) == 0 and len(ran) == 1


def test_missing_nvidia_driver_is_one_line_before_any_download(no_backends, fake_backend, monkeypatch, capsys):
    fake_backend("cuda")
    prepared, ran = [], []
    monkeypatch.setattr(_prepare, "ensure", lambda **kw: prepared.append(kw) or "model")
    monkeypatch.setattr(_cli, "_run", lambda cli, args: ran.append(args) or 0)
    from kitten_inference.tts2 import _gpu
    _no_runtime(monkeypatch, _gpu.CUDA_DRIVER)
    assert _cli.main(["--text", "x"]) == 1 and prepared == [] and ran == []
    assert "no NVIDIA driver found" in capsys.readouterr().err
    # --help and --version run the tool without the driver
    assert _cli.main(["--version"]) == 0 and _cli.main(["--help"]) == 0 and _cli.main([]) == 0 and len(ran) == 3


def test_no_gpu_runtime_names_both_vendors(no_backends, fake_backend, monkeypatch, capsys):
    # Linux x86-64 has both engines; with neither runtime the NVIDIA engine is picked and the message names both.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(_gpu, "CUDA_DRIVER", "libcuda.so.1")
    fake_backend("cuda")
    fake_backend("rocm", caps=())
    monkeypatch.setattr(_prepare, "ensure", lambda **kw: pytest.fail("prepared"))
    monkeypatch.setattr(_cli, "_run", lambda cli, args: pytest.fail("ran"))
    _no_runtime(monkeypatch, "libcuda.so.1", "libamdhip64.so.7")
    assert _cli.main(["--text", "x"]) == 1
    assert capsys.readouterr().err == (
        "kitten-tts: no NVIDIA driver found on this machine (libcuda.so.1 is missing). Kitten TTS 2 needs an NVIDIA "
        "GPU with the NVIDIA driver R570 or newer, or an AMD Radeon RX 7000 or RX 9000 with ROCm 7 (libamdhip64.so.7 "
        "is missing too; setup: https://github.com/KittenML/kitten-inference/blob/main/engines/tts2/docs/amd-radeon.md). In "
        "a container, attach the GPU (docker run --gpus all, or --device /dev/kfd --device /dev/dri for AMD). With "
        "both GPUs, --backend rocm or KITTEN_TTS2_BACKEND=rocm picks the AMD one.\n")


@pytest.mark.skipif(sys.platform == "win32", reason="platform: the AMD engine is Linux only")
def test_nvidia_that_cannot_start_hands_over_to_amd(no_backends, fake_backend, monkeypatch, capsys):
    fake_backend("cuda")
    fake_backend("rocm", caps=())
    ran = []
    monkeypatch.setattr(_cli, "_run", lambda cli, args: ran.append(cli) or 0)

    def no_gpu(device, log=None):
        raise _gpu.DriverStartError("the NVIDIA driver could not start (cuInit error 999)")
    monkeypatch.setattr(_gpu, "check_cuda", no_gpu)
    monkeypatch.setattr(_gpu, "check_rocm", lambda device: 1)
    assert _cli.main(["--assets", "m", "--text", "x"]) == 0
    assert os.path.join("rocm", "bin", "kitten-tts-native") in ran[0]
    assert capsys.readouterr().err == ("kitten-tts: the NVIDIA driver could not start (cuInit error 999); using the "
                                       "AMD (ROCm) engine instead\n")
    # Asked for: no check and no hand-over (the tool reports for itself).
    assert _cli.main(["--backend", "cuda", "--assets", "m", "--text", "x"]) == 0
    assert os.path.join("cuda", "bin", "kitten-tts-native") in ran[1]


@pytest.mark.skipif(sys.platform == "win32", reason="platform: the AMD engine is Linux only")
def test_rocm_found_in_its_own_folder_reaches_the_tool(no_backends, fake_backend, monkeypatch, tmp_path):
    # ROCm without AMD's ld.so.conf step: the tool's loader looks in that folder only through LD_LIBRARY_PATH.
    fake_backend("rocm", caps=())
    ran = []
    monkeypatch.setattr(_cli, "_run", lambda cli, args: ran.append(dict(os.environ)) or 0)
    root = tmp_path / "rocm-7.2.0"
    monkeypatch.setenv("ROCM_PATH", str(root))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/x")
    wanted = os.path.join(str(root), "lib", "libamdhip64.so.7")

    def load(name):
        if name != wanted:
            raise OSError(f"{name}: cannot open shared object file: No such file or directory")
        return object()
    monkeypatch.setattr(_gpu, "_cdll", load)
    assert _cli.main(["--assets", "m", "--text", "x"]) == 0
    assert ran[0]["LD_LIBRARY_PATH"] == os.pathsep.join([os.path.dirname(wanted), "/x"])
