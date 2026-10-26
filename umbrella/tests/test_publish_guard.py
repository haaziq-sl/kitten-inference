"""The release check that keeps the umbrella off PyPI until every engine its extras name is there
(.github/scripts/pypi_ready.py), against fake PyPI JSON answers."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "pypi_ready.py"
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


@pytest.fixture(scope="module")
def guard():
    if not SCRIPT.is_file():
        pytest.skip("not a source checkout (.github/scripts is not here)")
    spec = importlib.util.spec_from_file_location("pypi_ready", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wheel(name, version, plat="any", yanked=False):
    return {"filename": f"{name.replace('-', '_')}-{version}-py3-none-{plat}.whl", "packagetype": "bdist_wheel",
            "yanked": yanked}


def _pypi(releases):
    return lambda index, name: None if name not in releases else {"releases": releases[name]}


FULL = {
    "kitten-inference-tts08": {"0.2.0": [_wheel("kitten-inference-tts08", "0.2.0", "manylinux_2_17_x86_64"),
                                         _wheel("kitten-inference-tts08", "0.2.0", "macosx_11_0_arm64"),
                                         _wheel("kitten-inference-tts08", "0.2.0", "win_amd64")]},
    "kitten-inference-tts2": {"0.2.0": [_wheel("kitten-inference-tts2", "0.2.0")]},
    "kitten-inference-tts2-cu12": {"0.2.0": [_wheel("kitten-inference-tts2-cu12", "0.2.0", "manylinux_2_28_x86_64"),
                                             _wheel("kitten-inference-tts2-cu12", "0.2.0", "win_amd64")]},
    "kitten-inference-tts2-rocm": {"0.2.0": [_wheel("kitten-inference-tts2-rocm", "0.2.0", "manylinux_2_28_x86_64")]},
}


def test_reads_the_umbrella_extras(guard):
    reqs = guard.requirements(PYPROJECT)
    assert reqs == {name: ">=0.2.0,<0.3" for name in FULL}


def test_ready_when_every_engine_is_there(guard):
    assert guard.missing("idx", guard.requirements(PYPROJECT), get=_pypi(FULL)) == []


def test_not_ready_before_the_engines(guard):
    partial = {k: v for k, v in FULL.items() if k in ("kitten-inference-tts2", "kitten-inference-tts08")}
    problems = guard.missing("idx", guard.requirements(PYPROJECT), get=_pypi(partial))
    assert problems == ["kitten-inference-tts2-cu12: not on idx", "kitten-inference-tts2-rocm: not on idx"]


def test_missing_platform_and_wrong_versions(guard):
    releases = dict(FULL)
    releases["kitten-inference-tts2-cu12"] = {"0.2.0": [_wheel("kitten-inference-tts2-cu12", "0.2.0",
                                                               "manylinux_2_28_x86_64")]}
    releases["kitten-inference-tts2-rocm"] = {"0.3.0": [_wheel("kitten-inference-tts2-rocm", "0.3.0",
                                                               "manylinux_2_28_x86_64")],
                                              "0.2.0": [_wheel("kitten-inference-tts2-rocm", "0.2.0",
                                                               "manylinux_2_28_x86_64", yanked=True)]}
    problems = guard.missing("idx", guard.requirements(PYPROJECT), get=_pypi(releases))
    assert problems == ["kitten-inference-tts2-cu12 0.2.0: no wheel for win_amd64",
                        "kitten-inference-tts2-rocm: no release >=0.2.0,<0.3 with wheels"]


def test_tts08_needs_its_windows_wheel(guard):
    # tts-0.8 uploads its ~65 wheels in one sorted run, Windows last. An upload that stopped before the Windows
    # wheel (its macOS and Linux files up) must not count as ready, or the umbrella would publish with no tts08
    # Windows wheel behind the [tts-0.8] extra.
    releases = dict(FULL)
    releases["kitten-inference-tts08"] = {"0.2.0": [_wheel("kitten-inference-tts08", "0.2.0", "manylinux_2_17_x86_64"),
                                                    _wheel("kitten-inference-tts08", "0.2.0", "macosx_11_0_arm64")]}
    problems = guard.missing("idx", guard.requirements(PYPROJECT), get=_pypi(releases))
    assert problems == ["kitten-inference-tts08 0.2.0: no wheel for win_amd64"]


def test_versions(guard):
    ok = [("0.2.0", True), ("0.2.0.post1", True), ("0.2.5", True), ("0.3.0", False), ("0.1.1", False),
          ("0.2.1rc1", False)]
    for version, want in ok:
        assert guard.satisfies(version, ">=0.2.0,<0.3") is want, version


def test_warns_while_0_1_is_not_yanked(guard):
    old = {"kitten-inference": {"0.1.0": [_wheel("kitten-inference", "0.1.0", yanked=True)],
                                "0.1.1": [_wheel("kitten-inference", "0.1.1")]}}
    assert guard.earlier_not_yanked("idx", get=_pypi(old)) == ["0.1.1"]


def test_only_the_engines(guard):
    reqs = guard.only(guard.requirements(PYPROJECT), ["kitten_inference_tts2_cu12", "kitten-inference-tts2-rocm"])
    assert reqs == {"kitten-inference-tts2-cu12": ">=0.2.0,<0.3", "kitten-inference-tts2-rocm": ">=0.2.0,<0.3"}
    engines = {k: v for k, v in FULL.items() if k.startswith("kitten-inference-tts2-")}
    assert guard.missing("idx", reqs, get=_pypi(engines)) == []
    assert guard.missing("idx", reqs, get=_pypi({})) == ["kitten-inference-tts2-cu12: not on idx",
                                                         "kitten-inference-tts2-rocm: not on idx"]
    assert guard.only(reqs, []) == reqs
    with pytest.raises(SystemExit, match="do not name kitten-inference-tts3"):
        guard.only(reqs, ["kitten-inference-tts3"])


def test_wait_retries_a_failed_request(guard, capsys):
    import urllib.error
    answers = [urllib.error.HTTPError("u", 503, "Service Unavailable", {}, None), TimeoutError("timed out"),
               ConnectionResetError(104, "reset")]
    reqs = guard.only(guard.requirements(PYPROJECT), ["kitten-inference-tts2-cu12"])

    def flaky(index, name):
        if answers:
            raise answers.pop(0)
        return _pypi(FULL)(index, name)

    slept = []
    assert guard.wait("idx", reqs, "wait", 3600, 60, get=flaky, sleep=slept.append) == []
    assert len(slept) == 3 and "503" in capsys.readouterr().out
    # check mode (and the end of the wait) still reports the failure instead of passing
    answers.append(urllib.error.URLError("no route"))
    assert guard.wait("idx", reqs, "check", 3600, 60, get=flaky, sleep=slept.append)[0].startswith("PyPI request failed")
    clock = iter([0, 7201])
    answers.append(urllib.error.URLError("no route"))
    assert guard.wait("idx", reqs, "wait", 7200, 60, get=flaky, sleep=slept.append, clock=lambda: next(clock))


def _provenance(guard, repository="KittenML/kitten-inference", asked=None):
    def provenance(index, name, version, filename):
        if asked is not None:
            asked.append(filename)
        if repository is None:
            return None  # no attestation (an upload with an API token, or another publisher's tool)
        return {"version": 1, "attestation_bundles": [{"publisher": {
            "kind": "GitHub", "repository": repository, "workflow": guard.WORKFLOWS[name], "environment": "pypi"},
            "attestations": []}]}
    return provenance


def test_only_files_this_repository_published_count(guard):
    # The four new project names are public before the release. Someone who registers one and uploads 0.2.0 wheels
    # must not make the umbrella (or kitten-inference-tts2) publish, pointing its extras at their package.
    reqs = guard.requirements(PYPROJECT)
    asked = []
    assert guard.missing("idx", reqs, get=_pypi(FULL), provenance=_provenance(guard, asked=asked)) == []
    assert "kitten_inference_tts2_cu12-0.2.0-py3-none-win_amd64.whl" in asked  # every required platform
    for repository in ("someone/kitten-inference", None):
        problems = guard.missing("idx", reqs, get=_pypi(FULL), provenance=_provenance(guard, repository))
        assert len(problems) == 4 and all("was not published by KittenML/kitten-inference" in p for p in problems)
    wrong_workflow = {"attestation_bundles": [{"publisher": {"kind": "GitHub", "repository": "KittenML/kitten-inference",
                                                             "workflow": "other.yml"}}]}
    assert not guard.published_by(wrong_workflow, "KittenML/kitten-inference", "tts-2-engine.yml")
    assert guard.published_by({"attestation_bundles": [{"publisher": {
        "kind": "GitHub", "repository": "kittenml/Kitten-Inference", "workflow": ".github/workflows/tts-2-engine.yml"}}]},
        "KittenML/kitten-inference", "tts-2-engine.yml")
    assert set(guard.WORKFLOWS) >= set(reqs) | {"kitten-inference"}
