"""First-use preparation: revision resolution, lock, atomic rename, resume after an interrupted run, progress."""
import json
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
    """Stands in for kitten_inference.tts2._assets.prepare (the real one is tested end to end on a GPU box)."""
    __file__ = __file__  # _prepare_supports() reads the source: this test file mentions both new flags

    def __init__(self):
        self.runs = []
        self.fail_with = None

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
        print("Next: kitten-tts-native --assets ...")  # prepare's own summary goes to the log, not the terminal
        (out / "config.json").write_text(json.dumps(config))
        return 0


# The flags _prepare_supports() looks for in the real prepare.py: "--text-data-tool" "numpy"


@pytest.fixture
def env(monkeypatch, tmp_path):
    hub = FakeHub(monkeypatch, tmp_path)
    prep = FakePrepare()
    monkeypatch.setattr(_prepare, "_assets", lambda: prep)
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
    assert snapshot[2] == SHA and "lm/model.safetensors" in snapshot[3] and "speaker/*" not in snapshot[3]
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
    with pytest.raises(_prepare.PrepareError, match="checksum mismatch.*log:"):
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
    assert "speaker/*" in [c for c in env.hub.calls if c[0] == "snapshot"][-1][3]
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


def test_old_prepare_flags(env, monkeypatch, tmp_path):
    old = tmp_path / "old_prepare.py"
    old.write_text('p.add_argument("--text-converter")\n')
    env.prep.__file__ = str(old)
    env.run()
    argv = env.prep.runs[0]
    assert "--text-converter" in argv and "--text-data-tool" not in argv and "--weight-norm-device" not in argv


def test_default_cache_dir():
    assert _prepare.cache_root() == Path.home() / ".cache" / "kitten-inference" / "tts2"


def test_decoder_weights_downloaded_in_stage_one(env):
    env.prep.ACOUSTIC_REPO, env.prep.ACOUSTIC_FILE = "ResembleAI/chatterbox-turbo", "s3gen_meanflow.safetensors"
    env.prep.convert_s3gen = types.SimpleNamespace(PINNED_REVISION="1e4698ca7cbb41ff030c4185f0927a3b42d76924")
    env.run()
    argv = env.prep.runs[0]
    assert argv[argv.index("--acoustic-source") + 1].endswith("s3gen_meanflow.safetensors")
    (call,) = [c for c in env.hub.calls if c[0] == "file"]
    assert call[1:4] == ("ResembleAI/chatterbox-turbo", "s3gen_meanflow.safetensors",
                         "1e4698ca7cbb41ff030c4185f0927a3b42d76924") and call[5] is False
