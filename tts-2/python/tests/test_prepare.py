"""First-use preparation: revision resolution, lock, atomic rename, resume after an interrupted run, progress."""
import json
import os
import threading
import types
from pathlib import Path

import pytest

from kitten_inference.tts2 import _prepare

SHA = "c93cb7c271baa51d9ae8204d5277988906f235a2"
OTHER = "0" * 39 + "1"


class FakeHub:
    def __init__(self, monkeypatch, tmp_path, sha=SHA):
        import huggingface_hub
        self.sha = sha
        self.calls = []
        self.online = True
        hub = self

        class Api:
            def model_info(self, repo, revision=None, token=None):
                hub.calls.append(("model_info", repo, revision, token))
                if not hub.online:
                    raise ConnectionError("no network")
                return types.SimpleNamespace(sha=hub.sha)

        def snapshot_download(repo, revision=None, allow_patterns=None, cache_dir=None, token=None, **kw):
            hub.calls.append(("snapshot", repo, revision, tuple(allow_patterns), cache_dir, token))
            path = tmp_path / "hf" / "snapshots" / revision
            path.mkdir(parents=True, exist_ok=True)
            return str(path)

        def hf_hub_download(repo, name, revision=None, cache_dir=None, token=None, **kw):
            hub.calls.append(("file", repo, name, revision, cache_dir, token))
            path = tmp_path / "hf" / "files" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"decoder")
            return str(path)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
        monkeypatch.setattr(huggingface_hub, "HfApi", Api)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)


class FakePrepare:
    """Stands in for kitten_inference.tts2._assets.prepare and its child process (the real ones run end to end in
    tests/gpu)."""
    MODEL_ALLOW_PATTERNS = ["config.json", "lm/model.safetensors", "voices/*.npz"]
    CLONING_ALLOW_PATTERNS = ["speaker/model.safetensors"]
    ACOUSTIC_REPO, ACOUSTIC_FILE = "ResembleAI/chatterbox-turbo", "s3gen_meanflow.safetensors"
    convert_s3gen = types.SimpleNamespace(PINNED_REVISION="1e4698ca7cbb41ff030c4185f0927a3b42d76924")

    def __init__(self):
        self.runs = []
        self.fail_with = None

    def run(self, argv, log_path):  # replaces _prepare._run_prepare
        log_path.write_text("Next: kitten-tts-native --assets ...\n")
        try:
            self.main(argv)
        except (ValueError, OSError) as exc:
            raise _prepare.PrepareError(f"preparing the model failed: {exc} (full log: {log_path})")

    def main(self, argv):
        self.runs.append(list(argv))
        out = Path(argv[argv.index("--out") + 1])
        if self.fail_with is not None:
            (out / "native").mkdir(parents=True, exist_ok=True)
            (out / "native" / "model.kt2n").write_bytes(b"partial")
            raise self.fail_with
        if "--cloning-only" in argv:
            config = json.loads((out / "config.json").read_text())
            (out / "native" / "cloning").mkdir(parents=True, exist_ok=True)
            config["native"]["cloning"] = {"version": 1, "directory": "native/cloning", "files": []}
        else:
            out.mkdir(parents=True, exist_ok=True)
            config = {"type": "KITTEN2", "native": {"version": 1}}
            if "--with-cloning" in argv:
                config["native"]["cloning"] = {"version": 1}
        (out / "config.json").write_text(json.dumps(config))
        return 0


@pytest.fixture
def env(monkeypatch, tmp_path):
    hub = FakeHub(monkeypatch, tmp_path)
    prep = FakePrepare()
    monkeypatch.setattr(_prepare, "_assets", lambda: prep)
    monkeypatch.setattr(_prepare, "_run_prepare", prep.run)
    lines = []
    run = lambda **kw: _prepare.ensure(text_data_tool=tmp_path / "tool", cache_dir=tmp_path / "cache",
                                       log=lines.append, **kw)
    return types.SimpleNamespace(hub=hub, prep=prep, lines=lines, run=run, cache=tmp_path / "cache")


def test_first_use_prepares_once(env, capsys):
    d = env.run()
    assert d == env.cache / SHA and _prepare.is_prepared(d)
    assert [l[:5] for l in env.lines] == ["[1/3]", "[2/3]", "[3/3]"]  # one line per stage
    argv = env.prep.runs[0]
    assert argv[argv.index("--model-dir") + 1].endswith(SHA)
    assert argv[argv.index("--text-data-tool") + 1].endswith("tool")
    assert argv[argv.index("--revision") + 1] == SHA
    assert argv[argv.index("--weight-norm-device") + 1] == "numpy"
    assert argv[argv.index("--cache-dir") + 1] == str(env.cache / "downloads")
    assert "--with-cloning" not in argv
    assert "Next:" not in capsys.readouterr().out
    assert (env.cache / f"prepare-{SHA[:12]}.log").read_text().startswith("Next:")
    snapshot = [c for c in env.hub.calls if c[0] == "snapshot"][0]
    assert snapshot[2] == SHA and snapshot[3] == tuple(FakePrepare.MODEL_ALLOW_PATTERNS)
    assert snapshot[5] is False  # no token is ever sent
    env.lines.clear()
    assert env.run() == d and len(env.prep.runs) == 1 and env.lines == []


def test_pinned_revision_needs_no_network(env):
    env.run(revision=SHA)
    env.hub.online = False
    env.hub.calls.clear()
    assert env.run(revision=SHA) == env.cache / SHA
    assert env.hub.calls == []


def test_offline_falls_back_to_last_prepared_main(env):
    env.run()
    env.hub.online = False
    assert env.run() == env.cache / SHA
    assert any("cannot reach Hugging Face" in l for l in env.lines)


def test_offline_without_prepared_model(env):
    env.hub.online = False
    with pytest.raises(_prepare.PrepareError, match="cannot find KittenML/kitten-tts-2 revision 'main'"):
        env.run()


def test_new_main_revision_is_prepared_next_to_the_old(env):
    env.run()
    env.hub.sha = OTHER
    assert env.run() == env.cache / OTHER
    assert _prepare.is_prepared(env.cache / SHA) and len(env.prep.runs) == 2


def test_interrupted_run_leaves_nothing_and_resumes(env):
    env.prep.fail_with = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        env.run()
    assert not (env.cache / SHA).exists()
    assert (env.cache / f".partial-{SHA}").exists()  # cleaned at the next run
    env.prep.fail_with = None
    d = env.run()
    assert _prepare.is_prepared(d) and not (env.cache / f".partial-{SHA}").exists()
    assert not (d / "native" / "model.kt2n").exists()  # nothing of the interrupted attempt survives


def test_failure_message(env):
    env.prep.fail_with = ValueError("full decoder source checksum mismatch")
    with pytest.raises(_prepare.PrepareError, match="checksum mismatch.*full log:"):
        env.run()
    assert not (env.cache / SHA).exists()


def test_cloning_added_to_a_prepared_model(env):
    d = env.run()
    (d / "native" / "cloning").mkdir(parents=True)
    (d / "native" / "cloning" / "stale.safetensors").write_bytes(b"x")  # a stopped earlier attempt
    env.lines.clear()
    assert env.run(with_cloning=True) == d
    argv = env.prep.runs[-1]
    assert "--cloning-only" in argv and argv[argv.index("--out") + 1] == str(d)
    assert _prepare.is_prepared(d, with_cloning=True)
    assert not (d / "native" / "cloning" / "stale.safetensors").exists()
    assert "speaker/model.safetensors" in [c for c in env.hub.calls if c[0] == "snapshot"][-1][3]
    assert [l[:5] for l in env.lines] == ["[1/3]", "[2/3]", "[3/3]"]


def test_first_use_with_cloning(env):
    d = env.run(with_cloning=True)
    assert "--with-cloning" in env.prep.runs[0] and _prepare.is_prepared(d, with_cloning=True)


def test_concurrent_processes_prepare_once(env):
    errors = []

    def worker():
        try:
            env.run()
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(env.prep.runs) == 1


def test_default_cache_dir():
    assert _prepare.cache_root() == Path.home() / ".cache" / "kitten-inference" / "tts2"


def test_decoder_weights_downloaded_in_stage_one(env):
    env.run()
    argv = env.prep.runs[0]
    assert argv[argv.index("--acoustic-source") + 1].endswith("s3gen_meanflow.safetensors")
    (call,) = [c for c in env.hub.calls if c[0] == "file"]
    assert call[1:4] == ("ResembleAI/chatterbox-turbo", "s3gen_meanflow.safetensors",
                         "1e4698ca7cbb41ff030c4185f0927a3b42d76924") and call[5] is False


def test_child_process_failure_shows_the_log_tail(tmp_path, monkeypatch):
    import subprocess
    log = tmp_path / "prepare.log"

    def fake_run(cmd, stdin=None, stdout=None, stderr=None, env=None):
        assert cmd[1] == "-c" and "kitten_inference.tts2._assets.prepare" in cmd[2] and cmd[3:] == ["--out", "x"]
        assert env["PYTHONPATH"].split(os.pathsep)[0].endswith(("site-packages", "python"))
        stdout.write(b"line 1\nerror: full decoder source checksum mismatch\n")
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(_prepare.PrepareError, match="checksum mismatch .full log: .*prepare.log"):
        _prepare._run_prepare(["--out", "x"], log)


def test_gpu_check_runs_only_before_a_download(env):
    checks = []
    env.run(before_download=lambda: checks.append(1))
    assert checks == [1]
    env.run(before_download=lambda: checks.append(2))  # prepared: nothing to download, no check
    assert checks == [1]


def test_gpu_check_failure_downloads_nothing(env):
    def no_gpu():
        raise RuntimeError("no NVIDIA driver found")
    with pytest.raises(RuntimeError, match="no NVIDIA driver"):
        env.run(before_download=no_gpu)
    assert not [c for c in env.hub.calls if c[0] in ("snapshot", "file")] and env.prep.runs == []


def test_child_writes_utf8(tmp_path):
    e = _prepare._child_env(tmp_path)
    assert e["PYTHONIOENCODING"] == "utf-8" and e["PYTHONUTF8"] == "1"


def test_child_prints_a_non_ascii_path_to_its_log(tmp_path, monkeypatch):
    # The bundled prepare prints the bundle path at the end. On Windows (Python < 3.15) output to a file used the
    # ANSI code page, and a path outside it stopped the child after the conversion.
    out = tmp_path / "Łukasz 田中" / "bundle"
    out.mkdir(parents=True)
    (out / "native").mkdir()
    (out / "native" / "model.kt2n").write_bytes(b"x")
    child = ("import sys; from pathlib import Path; from kitten_inference.tts2._assets.prepare import print_summary; "
             "out = Path(sys.argv[1]); print_summary(out, out / 'config.json', cloning=True)")
    monkeypatch.setattr(_prepare, "_CHILD", child)
    log = tmp_path / "prepare.log"
    _prepare._run_prepare([str(out)], log)
    assert str(out) in log.read_text(encoding="utf-8")


class _Windows:
    """cache_root() on Windows with a non-UTF-8 code page, with the short-path answer chosen by the test."""

    def __init__(self, monkeypatch, tmp_path, short):
        monkeypatch.setattr(_prepare, "_narrow_paths_are_utf8", lambda: False)
        monkeypatch.setattr(_prepare, "_short_path", lambda path: short(path))
        monkeypatch.setattr(_prepare, "DEFAULT_CACHE", tmp_path / "José" / ".cache" / "kitten-inference" / "tts2")
        monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "ProgramData"))


def test_non_ascii_cache_uses_the_short_path(monkeypatch, tmp_path):
    _Windows(monkeypatch, tmp_path, lambda path: tmp_path / "JOSE~1" / "tts2")
    assert _prepare.cache_root() == tmp_path / "JOSE~1" / "tts2"
    assert _prepare.cache_root(tmp_path / "Łukasz") == tmp_path / "JOSE~1" / "tts2"


def test_non_ascii_default_cache_without_a_short_name_moves(monkeypatch, tmp_path):
    _Windows(monkeypatch, tmp_path, lambda path: path)  # "José" is a valid 8.3 name: no ASCII short form
    notes = []
    root = _prepare.cache_root(log=notes.append)
    assert root.parent == tmp_path / "ProgramData" / "kitten-inference" and root.name.startswith("tts2-")
    assert root.is_dir() and len(notes) == 1 and str(root) in notes[0]
    assert _prepare.cache_root(log=notes.append) == root and len(notes) == 1  # said once


def test_non_ascii_cache_dir_is_refused_before_any_download(monkeypatch, tmp_path, env):
    _Windows(monkeypatch, tmp_path, lambda path: None)
    with pytest.raises(_prepare.PrepareError, match='ASCII folder: KittenTTS2.cache_dir="C:/kitten-tts2"'):
        _prepare.ensure(text_data_tool=tmp_path / "tool", cache_dir=tmp_path / "Łukasz")
    assert env.hub.calls == []


def test_child_temp_folder_is_ascii_where_the_engine_needs_it(monkeypatch, tmp_path):
    monkeypatch.setattr(_prepare, "_narrow_paths_are_utf8", lambda: False)
    monkeypatch.setenv("TEMP", str(tmp_path / "José" / "Temp"))
    e = _prepare._child_env(tmp_path / "cache")
    assert e["TEMP"] == e["TMP"] == str(tmp_path / "cache" / "tmp")
    monkeypatch.setenv("TEMP", str(tmp_path / "Temp"))
    assert _prepare._child_env(tmp_path / "cache")["TEMP"] == str(tmp_path / "Temp")


@pytest.mark.skipif(os.name != "nt", reason="platform: Windows only")
def test_windows_real_non_ascii_folders(tmp_path, monkeypatch):
    """On a real Windows file system: a folder name that needs a short name (Ł is not in the OEM code page) and one
    that may not get one (é is): the engine gets an ASCII path to the same folder, or the default moves."""
    if _prepare._narrow_paths_are_utf8():
        pytest.skip("platform: this Windows runs with the UTF-8 code page")
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "ProgramData"))
    for name in ("Łukasz Wiśniewski", "José"):
        home = tmp_path / name
        monkeypatch.setattr(_prepare, "DEFAULT_CACHE", home / ".cache" / "kitten-inference" / "tts2")
        root = _prepare.cache_root()
        assert all(ord(c) < 128 for c in str(root)), root
        root.mkdir(parents=True, exist_ok=True)
        moved = root.parent.name == "kitten-inference" and root.name.startswith("tts2-")
        assert moved or os.path.samefile(root, home / ".cache" / "kitten-inference" / "tts2"), root
        print(f"{name!r}: {root} ({'moved' if moved else 'short path'})")


def test_a_concurrent_ref_update_is_not_an_error(env, monkeypatch):
    # Windows refuses os.replace while another process replaces the same file (CI saw PermissionError with 4 threads).
    env.run()
    ref = _prepare._ref_file(env.cache, "KittenML/kitten-tts-2", "main")
    ref.write_text("0" * 40 + "\n")  # stale: the next run rewrites it

    def busy(src, dst):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(_prepare.os, "replace", busy)
    assert env.run() == env.cache / SHA
    assert not list(ref.parent.glob("*.tmp"))
