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
        self.timeouts = []
        self.online = True
        hub = self

        class Api:
            def model_info(self, repo, revision=None, token=None, timeout=None):
                hub.calls.append(("model_info", repo, revision, token))
                hub.timeouts.append(timeout)
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
        self.pinned = []          # ("wheel"/"grammar", cache) in the order fetched
        self.pinned_fail = None   # when set, pinned_wheel raises it (a blocked PyPI/GitHub)

    def pinned_wheel(self, cache, offline, fetch=None):
        if self.pinned_fail is not None:
            raise ValueError(self.pinned_fail)
        self.pinned.append(("wheel", Path(cache)))
        return b"wheel"

    def pinned_grammar(self, cache, offline, fetch=None):
        if self.pinned_fail is not None:
            raise ValueError(self.pinned_fail)
        self.pinned.append(("grammar", Path(cache)))
        d = Path(cache) / "grammar"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def nemo_wheel_path(self, cache):
        return Path(cache) / "nemo_text_processing-1.2.0.whl"

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
            if "--cloning-only" in argv:  # what a stopped cloning preparation leaves
                (out / "native" / "cloning").mkdir(parents=True, exist_ok=True)
                (out / "native" / "cloning" / "s3gen.safetensors").write_bytes(b"partial")
                (out / "native" / ".clone-prepare-1234").mkdir(exist_ok=True)
            else:
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
    # the small pinned inputs were fetched and handed to the child, so it does not download them again
    assert [kind for kind, _ in env.prep.pinned] == ["wheel", "grammar"]
    assert argv[argv.index("--grammar-dir") + 1].endswith("grammar")
    assert argv[argv.index("--nemo-wheel") + 1].endswith("nemo_text_processing-1.2.0.whl")
    env.lines.clear()
    assert env.run() == d and len(env.prep.runs) == 1 and env.lines == []


def test_pinned_inputs_are_fetched_before_the_big_download(env):
    # A network that blocks PyPI or GitHub must fail before the 4.6 GB Hugging Face snapshot, not after it.
    env.prep.pinned_fail = "the grammar host is unreachable"
    with pytest.raises(_prepare.PrepareError, match="pinned text-processing files failed.*GitHub"):
        env.run()
    assert [c for c in env.hub.calls if c[0] == "snapshot"] == []  # the big download never started
    assert env.prep.runs == []


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
    assert not (env.cache / f".partial-{SHA}").exists()  # removed at once, not at the next start
    env.prep.fail_with = None
    d = env.run()
    assert _prepare.is_prepared(d) and not (env.cache / f".partial-{SHA}").exists()
    assert not (d / "native" / "model.kt2n").exists()  # nothing of the interrupted attempt survives


def test_stopped_preparation_of_an_older_revision_is_removed(env):
    old = env.cache / f".partial-{OTHER}"  # a stopped attempt at a revision that is no longer main
    (old / "native").mkdir(parents=True)
    (old / "native" / "model.kt2n").write_bytes(b"partial")
    busy = env.cache / f".partial-{'2' * 40}"  # another process is preparing that one: left alone
    busy.mkdir()
    from filelock import FileLock
    with FileLock(str(env.cache / f".{'2' * 40}.lock")):
        env.run()
    assert not old.exists() and busy.exists()


def test_a_new_main_that_cannot_be_prepared_falls_back(env):
    env.run()
    env.hub.sha = OTHER
    env.prep.fail_with = OSError(28, "No space left on device")
    env.lines.clear()
    assert env.run() == env.cache / SHA
    assert any(l.startswith("warning: the new KittenML/kitten-tts-2 revision 000000000000 could not be prepared")
               and "No space left on device" in l and f'revision="{SHA}"' in l for l in env.lines), env.lines
    # The stopped conversion's files are gone at once: after a full disk they would keep it full until the next start.
    assert not (env.cache / f".partial-{OTHER}").exists() and not (env.cache / OTHER).exists()
    assert _prepare._ref_file(env.cache, "KittenML/kitten-tts-2", "main").read_text().strip() == SHA
    with pytest.raises(_prepare.PrepareError, match="No space left"):  # an explicit revision: no fallback
        env.run(revision=OTHER)
    with pytest.raises(_prepare.PrepareError, match=f'prepared without voice cloning: revision="{SHA}"'):
        env.run(with_cloning=True)
    env.prep.fail_with = None
    assert env.run() == env.cache / OTHER  # tried again at the next start


def test_asking_for_the_latest_revision_has_a_time_limit(env):
    # Every start without a pinned revision asks Hugging Face which commit main is; a network that drops packets
    # must not hold the start for minutes before the prepared model is used.
    env.run()
    assert env.hub.timeouts and all(t is not None and 0 < t <= 10 for t in env.hub.timeouts), env.hub.timeouts
    env.hub.online = False
    env.lines.clear()
    assert env.run() == env.cache / SHA
    assert any("HF_HUB_OFFLINE=1" in l for l in env.lines), env.lines


def test_failed_cloning_files_are_removed(env):
    env.run()
    env.prep.fail_with = OSError(28, "No space left on device")
    with pytest.raises(_prepare.PrepareError, match="No space left"):
        env.run(with_cloning=True)
    native = env.cache / SHA / "native"
    assert not (native / "cloning").exists() and not list(native.glob(".clone-prepare-*"))
    assert _prepare.is_prepared(env.cache / SHA) and not _prepare.is_prepared(env.cache / SHA, True)


def test_no_fallback_without_an_earlier_revision(env):
    env.prep.fail_with = ValueError("token map changed")
    with pytest.raises(_prepare.PrepareError, match="token map changed"):
        env.run()


def test_relative_cache_dir_is_made_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    root = _prepare.cache_root("models")
    assert root == tmp_path / "models" and root.is_absolute()
    assert _prepare._download_cache("models") == tmp_path / "models" / "downloads"


def test_failure_message(env):
    env.prep.fail_with = ValueError("full decoder source checksum mismatch")
    with pytest.raises(_prepare.PrepareError, match="checksum mismatch.*full log:"):
        env.run()
    assert not (env.cache / SHA).exists()


def test_cloning_added_to_a_prepared_model(env):
    d = env.run()
    (d / "native" / "cloning").mkdir(parents=True)
    (d / "native" / "cloning" / "stale.safetensors").write_bytes(b"x")  # a stopped earlier attempt
    staging = d / "native" / ".clone-prepare-k1ll3d" / "cloning"  # the staging folder of a killed attempt
    staging.mkdir(parents=True)
    (staging / "s3gen.safetensors").write_bytes(b"x" * 1000)
    env.lines.clear()
    assert env.run(with_cloning=True) == d
    argv = env.prep.runs[-1]
    assert "--cloning-only" in argv and argv[argv.index("--out") + 1] == str(d)
    assert _prepare.is_prepared(d, with_cloning=True)
    assert not (d / "native" / "cloning" / "stale.safetensors").exists()
    assert not list((d / "native").glob(".clone-prepare-*"))
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


def test_fallback_folder_is_private_and_must_be_this_users(monkeypatch, tmp_path):
    # The folder's name can be worked out by any user of the machine: it is made for this user only, and one that
    # someone else made first is refused.
    _Windows(monkeypatch, tmp_path, lambda path: path)
    root = _prepare.cache_root(log=lambda m: None)
    if os.name != "nt":
        assert root.stat().st_mode & 0o777 == 0o700
    monkeypatch.setattr(_prepare, "_owned_by_me", lambda path: False)
    with pytest.raises(_prepare.PrepareError, match="belongs to another user or can be changed by other users; pass "
                                                     "a new ASCII folder as cache_dir"):
        _prepare.cache_root(log=lambda m: None)


class _Acl:
    """cache_root() on Windows with the owner and the access list chosen by the test."""

    def __init__(self, monkeypatch, owner_is_me=True, others=("S-1-5-11",)):
        self.others = list(others)
        self.made_private = []
        monkeypatch.setattr(_prepare, "_windows", lambda: True)
        monkeypatch.setattr(_prepare, "_owned_by_me", lambda path: owner_is_me)
        monkeypatch.setattr(_prepare, "_other_writers", lambda path: list(self.others))

        def set_private(path):
            self.made_private.append(path)
            self.others = []

        monkeypatch.setattr(_prepare, "_set_private", set_private)


def test_an_existing_cache_dir_others_can_change_is_refused(monkeypatch, tmp_path):
    # A folder made under C:\ or on another drive lets every signed-in user add files: a model planted there would
    # be used without a warning.
    folder = tmp_path / "kitten-tts2"
    (folder / SHA).mkdir(parents=True)
    acl = _Acl(monkeypatch)
    with pytest.raises(_prepare.PrepareError, match="can be changed by other users of this computer.*Pass a new "
                                                     "folder as cache_dir"):
        _prepare.cache_root(folder)
    assert acl.made_private == []
    acl = _Acl(monkeypatch, owner_is_me=False, others=())
    with pytest.raises(_prepare.PrepareError, match="belongs to another user"):
        _prepare.cache_root(folder)


def test_an_empty_cache_dir_of_this_user_is_made_private(monkeypatch, tmp_path):
    folder = tmp_path / "kitten-tts2"
    folder.mkdir()
    acl = _Acl(monkeypatch)
    assert _prepare.cache_root(folder) == folder and acl.made_private == [folder]
    acl = _Acl(monkeypatch, others=())  # already private: used as it is
    (folder / SHA).mkdir()
    assert _prepare.cache_root(folder) == folder and acl.made_private == []
    acl = _Acl(monkeypatch)  # a new folder: not checked here, _make_root makes it private
    assert _prepare.cache_root(tmp_path / "new") == tmp_path / "new" and acl.made_private == []


def test_default_cache_is_not_made_private_on_windows(monkeypatch, tmp_path):
    # Under the user profile Windows already keeps the folder private; giving it the OWNER-RIGHTS descriptor would
    # lock the same user out of it after a run in an elevated window. A folder outside the profile is still private.
    monkeypatch.setattr(_prepare.sys, "platform", "win32")
    # Path.home() reads USERPROFILE on Windows and HOME elsewhere. On a Windows runner tmp_path itself lies inside the
    # real profile (AppData\Local\Temp), so the profile is pinned here.
    home = tmp_path / "home"
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    assert Path.home() == home
    default = home / ".cache" / "kitten-inference" / "tts2"
    monkeypatch.setattr(_prepare, "DEFAULT_CACHE", default)
    private = []
    monkeypatch.setattr(_prepare, "_private_mkdir", lambda p: private.append(p))
    _prepare._make_root(default)
    assert default.is_dir() and private == []
    inside = home / "models" / "kitten-tts2"
    _prepare._make_root(inside)
    assert inside.is_dir() and private == []  # inside the profile: the profile's access list covers it
    custom = tmp_path / "C" / "kitten-tts2"
    _prepare._make_root(custom)
    assert private == [custom]  # outside the profile: made private


def test_a_reparse_point_cache_dir_is_refused(monkeypatch, tmp_path):
    # A junction another user owns that points at the victim's folder would pass the owner/ACL checks (they follow
    # the link) and could be re-pointed later; a model folder must be a real directory.
    folder = tmp_path / "kitten-tts2"
    folder.mkdir()
    monkeypatch.setattr(_prepare, "_windows", lambda: True)
    monkeypatch.setattr(_prepare, "_is_reparse_point", lambda p: p == folder)
    with pytest.raises(_prepare.PrepareError, match="reparse point"):
        _prepare._check_folder_is_yours(folder)


def test_the_default_folder_is_not_checked(monkeypatch, tmp_path):
    # It is in the user's own profile, which Windows keeps private.
    _Acl(monkeypatch, owner_is_me=False)
    monkeypatch.setattr(_prepare, "DEFAULT_CACHE", tmp_path / "home" / ".cache" / "kitten-inference" / "tts2")
    assert _prepare.cache_root() == _prepare.DEFAULT_CACHE


@pytest.mark.skipif(os.name != "nt", reason="platform: Windows only")
def test_windows_cache_dir_access_list(tmp_path):
    """On a real Windows file system: a folder that every signed-in user may change is refused while it has files,
    made private when it is empty, and then used."""
    import subprocess
    folder = tmp_path / "shared"
    folder.mkdir()
    assert _prepare._other_writers(folder) == [], subprocess.run(["icacls", str(folder)], capture_output=True,
                                                                   text=True).stdout
    subprocess.run(["icacls", str(folder), "/grant", "*S-1-5-11:(OI)(CI)M"], check=True, capture_output=True)
    assert _prepare._other_writers(folder) == ["S-1-5-11"]
    (folder / "planted.txt").write_text("x")
    with pytest.raises(_prepare.PrepareError, match="can be changed by other users"):
        _prepare.cache_root(folder)
    (folder / "planted.txt").unlink()
    assert _prepare.cache_root(folder) == folder
    acl = subprocess.run(["icacls", str(folder)], capture_output=True, text=True, check=True).stdout
    print(acl)
    assert "Authenticated Users" not in acl and _prepare._other_writers(folder) == []
    (folder / "model").mkdir()
    assert _prepare._other_writers(folder / "model") == []  # inherited by what is made inside


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


@pytest.mark.skipif(os.name != "nt", reason="platform: Windows only")
def test_windows_private_folder(tmp_path):
    """The fallback folder's access list on a real Windows file system: no access for other users, inherited by
    what is made inside, and owned by this user."""
    import subprocess
    folder = tmp_path / "private"
    _prepare._private_mkdir(folder)
    (folder / "sub").mkdir()
    (folder / "sub" / "file.bin").write_bytes(b"x")
    assert _prepare._owned_by_me(folder) and _prepare._owned_by_me(folder / "sub" / "file.bin")
    for path in (folder, folder / "sub", folder / "sub" / "file.bin"):
        acl = subprocess.run(["icacls", str(path)], capture_output=True, text=True, check=True).stdout
        print(acl)
        for other in ("BUILTIN\\Users", "Authenticated Users", "Everyone", "CREATOR OWNER"):
            assert other not in acl, (path, acl)
    with pytest.raises(FileExistsError):
        _prepare._private_mkdir(folder)
    assert not _prepare._owned_by_me(Path(os.environ.get("SystemRoot", "C:\\Windows")))  # owned by TrustedInstaller


def test_child_ignores_modules_in_the_current_folder(tmp_path, monkeypatch):
    # python -c puts the current folder first on sys.path: a json.py or kitten_inference/ there must not be imported.
    here = tmp_path / "downloads"
    (here / "kitten_inference").mkdir(parents=True)
    (here / "kitten_inference" / "__init__.py").write_text("raise SystemExit('planted package imported')\n")
    for name in ("json", "argparse", "hashlib", "numpy"):
        (here / f"{name}.py").write_text(f"raise SystemExit('planted {name}.py imported')\n")
    monkeypatch.chdir(here)
    log = tmp_path / "prepare.log"
    _prepare._run_prepare(["--help"], log)
    text = log.read_text(encoding="utf-8")
    assert "planted" not in text and "usage" in text.lower(), text


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
