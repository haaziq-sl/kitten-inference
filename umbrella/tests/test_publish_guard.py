"""The release guards: the wheel sets of a release and the check that PyPI has all of them after the upload
(.github/scripts/pypi_ready.py, against fake PyPI answers), and the shape of the workflows: only packages.yml can
upload, and only from a v* tag in KittenML/kitten-inference."""
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github" / "scripts" / "pypi_ready.py"
WORKFLOWS = ROOT / ".github" / "workflows"
LOCK = ROOT / ".github" / "release-inputs.json"
PUBLISH_ACTION = "pypa/gh-action-pypi-publish@7245906fb61fa33a13554f9d5fb856fe61758086"
FAMILIES = ["linux-x86_64", "linux-aarch64", "macos-arm64", "macos-x86_64", "windows-x86_64", "windows-arm64",
            "android-arm64"]
# The artifacts a release takes its inputs from, by the workflow that uploads them (the manylinux2014 x86-64 TTS 0.8
# wheels are built and tested there but not used).
INPUT_ARTIFACTS = {
    "tts-0.8.yml": ["tts08-wheels-" + f for f in (
        "windows-arm64", "windows-x86_64", "macos-x86_64", "macos-arm64", "linux-x86_64-manylinux-2-28",
        "linux-x86_64-manylinux2014", "linux-aarch64-manylinux2014", "linux-aarch64-manylinux-2-28",
        "android-arm64")],
    "tts-2-engine.yml": ["tts2-cu12-linux", "tts2-cu12-windows"],
    "tts-2-rocm.yml": ["kitten-inference-tts2-rocm-manylinux_2_28_x86_64"],
}
RUN_WORKFLOWS = {"tts08": "tts-0.8.yml", "cuda": "tts-2-engine.yml", "rocm": "tts-2-rocm.yml"}


@pytest.fixture(scope="module")
def guard():
    if not SCRIPT.is_file():
        pytest.skip("not a source checkout (.github/scripts is not here)")
    spec = importlib.util.spec_from_file_location("pypi_ready", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tts08_wheels(version="0.2.0"):
    """The 62 TTS 0.8 wheels as tts-0.8.yml builds them (the file names of the tested 0.2.0 run), sorted by name."""
    plats = ["macosx_11_0_arm64", "macosx_11_0_x86_64", "manylinux2014_aarch64.manylinux_2_17_aarch64",
             "manylinux2014_x86_64.manylinux_2_17_x86_64", "manylinux_2_27_aarch64.manylinux_2_28_aarch64",
             "manylinux_2_27_x86_64.manylinux_2_28_x86_64", "win_amd64"]
    names = []
    for py in ("cp38", "cp39", "cp310", "cp311", "cp312", "cp313", "cp314", "cp315"):
        extra = (["win_arm64"] if py not in ("cp38", "cp39", "cp310") else []) + \
                (["android_24_arm64_v8a"] if py == "cp313" else [])
        names += [f"kitten_inference_tts08-{version}-{py}-{py}-{plat}.whl" for plat in plats + extra]
    return sorted(names)


def _release_wheels(version="0.2.0"):
    """The 54 kitten-inference wheels assembled from them: no manylinux2014 x86-64 wheel, and the Linux x86-64 wheels
    with engines tagged manylinux_2_28 only (cp38, which has no engine, keeps the TTS 0.8 wheel's tags)."""
    out = []
    for name in _tts08_wheels(version):
        _, _, py, abi, plat = name[:-4].split("-")
        if plat.startswith("manylinux2014_x86_64"):
            continue
        if plat.endswith("manylinux_2_28_x86_64") and py != "cp38":
            plat = "manylinux_2_28_x86_64"
        out.append(f"kitten_inference-{version}-{py}-{abi}-{plat}.whl")
    return sorted(out)


def _digest(name):
    return hashlib.sha256(name.encode()).hexdigest()


def _release(files, yanked=(), digests=None):
    """A fake PyPI JSON answer for one release with these files."""
    digests = digests or {}
    return {"urls": [{"filename": n, "packagetype": "bdist_wheel", "yanked": n in yanked,
                      "digests": {"sha256": digests.get(n, _digest(n))}} for n in files]}


def _pypi(releases):
    return lambda index, name, version: releases.get((name, version))


def _provenance(repository="KittenML/kitten-inference", workflow="packages.yml", asked=None):
    def provenance(index, name, version, filename):
        if asked is not None:
            asked.append(filename)
        if repository is None:
            return None  # no attestation (an upload with an API token, or another publisher's tool)
        return {"version": 1, "attestation_bundles": [{"publisher": {
            "kind": "GitHub", "repository": repository, "workflow": workflow, "environment": "pypi"},
            "attestations": []}]}
    return provenance


# The wheel sets


def test_the_release_is_54_wheels_assembled_from_62(guard):
    inputs, release = _tts08_wheels(), _release_wheels()
    assert len(inputs) == 62 and len(release) == 54
    assert len(guard.COMPLETE["kitten-inference-tts08"]) == 62 and len(guard.COMPLETE["kitten-inference"]) == 54
    assert guard.check_built("kitten-inference-tts08", inputs) == []
    assert guard.check_built("kitten-inference", release) == []
    per_platform = {}
    for py, plat in guard.COMPLETE["kitten-inference"]:
        per_platform[plat] = per_platform.get(plat, 0) + 1
    assert per_platform == {"manylinux_2_28_x86_64": 8, "manylinux_2_17_aarch64": 8, "manylinux_2_28_aarch64": 8,
                            "macosx_11_0_x86_64": 8, "macosx_11_0_arm64": 8, "win_amd64": 8, "win_arm64": 5,
                            "android_24_arm64_v8a": 1}


def test_complete_reports_missing_extra_and_double_wheels(guard):
    release = _release_wheels()
    assert guard.check_built("kitten-inference", release[1:]) == [f"not built: {guard.short(release[0])}"]
    extra = "kitten_inference-0.2.0-cp316-cp316-win_amd64.whl"
    assert guard.check_built("kitten-inference", release + [extra]) == [
        f"built but not in pypi_ready.COMPLETE: {extra}"]
    # A release has no manylinux2014 x86-64 wheel: on Linux x86-64 every wheel is the manylinux_2_28 build.
    old = "kitten_inference-0.2.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
    assert guard.check_built("kitten-inference", release + [old]) == [f"built but not in pypi_ready.COMPLETE: {old}"]
    twice = "kitten_inference-0.2.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    problems = guard.check_built("kitten-inference", release + [twice])
    assert len(problems) == 1 and problems[0].startswith("two wheels for cp312-manylinux_2_28_x86_64"), problems
    assert guard.wheel_tags("kitten_inference-0.2.0-cp310-cp310-manylinux2014_aarch64.manylinux_2_17_aarch64.whl") \
        == {("cp310", "manylinux2014_aarch64"), ("cp310", "manylinux_2_17_aarch64")}


def test_complete_command_line(guard, tmp_path):
    # As packages.yml and tts-0.8.yml run it: pypi_ready.py complete --dist NAME FILES
    def run(dist, names):
        folder = tmp_path / dist
        folder.mkdir(exist_ok=True)
        for n in names:
            (folder / n).write_bytes(b"")
        paths = sorted(str(folder / n) for n in names)
        return subprocess.run([sys.executable, str(SCRIPT), "complete", "--dist", dist, *paths],
                              capture_output=True, text=True)

    for dist, names, count in (("kitten-inference", _release_wheels(), 54),
                               ("kitten-inference-tts08", _tts08_wheels(), 62)):
        ok = run(dist, names)
        assert ok.returncode == 0 and f"the {count} built wheels are the complete set" in ok.stdout, ok.stdout
    bad = run("kitten-inference-bad", _release_wheels()[1:])
    assert bad.returncode == 1 and "has no COMPLETE entry" in bad.stdout, bad.stdout + bad.stderr
    bad = subprocess.run([sys.executable, str(SCRIPT), "complete", "--dist", "kitten-inference",
                          *sorted(str(tmp_path / "kitten-inference" / n) for n in _release_wheels()[1:])],
                         capture_output=True, text=True)
    assert bad.returncode == 1 and "not built: cp310-macosx_11_0_arm64" in bad.stdout, bad.stdout + bad.stderr


# After the upload: verify


def test_verify_counts_a_partial_upload(guard):
    # twine uploads the files in name order, one at a time. A stop at any point must be reported with how many are
    # missing, and verify must pass only once all 54 are there.
    release = _release_wheels()
    local = {n: _digest(n) for n in release}
    assert guard.on_pypi("idx", "0.2.0", local, get=_pypi({}), provenance=None) == [
        "kitten-inference 0.2.0: not on idx yet (0 of 54 wheels)"]
    for stopped in range(1, len(release)):
        pypi = _pypi({("kitten-inference", "0.2.0"): _release(release[:stopped])})
        problems = guard.on_pypi("idx", "0.2.0", local, get=pypi, provenance=None)
        assert len(problems) == 1, (stopped, problems)
        assert f": {54 - stopped} of 54 wheels not on idx yet" in problems[0], (stopped, problems)
    pypi = _pypi({("kitten-inference", "0.2.0"): _release(release[:20])})
    problem = guard.on_pypi("idx", "0.2.0", local, get=pypi, provenance=None)[0]
    assert problem.startswith("kitten-inference 0.2.0: 34 of 54 wheels not on idx yet (") and problem.endswith(
        " and 28 more)"), problem
    pypi = _pypi({("kitten-inference", "0.2.0"): _release(release)})
    assert guard.on_pypi("idx", "0.2.0", local, get=pypi, provenance=_provenance()) == []


def test_verify_wants_the_same_bytes_unyanked_and_nothing_else(guard):
    release = _release_wheels()
    local = {n: _digest(n) for n in release}
    first, last = release[0], release[-1]
    pypi = _pypi({("kitten-inference", "0.2.0"): _release(release, yanked={last}, digests={first: "0" * 64})})
    problems = guard.on_pypi("idx", "0.2.0", local, get=pypi, provenance=None)
    assert problems == [f"{first}: sha256 on idx is {'0' * 64}, this file's is {_digest(first)}",
                        f"{last}: yanked on idx"]
    sdist = "kitten_inference-0.2.0.tar.gz"
    pypi = _pypi({("kitten-inference", "0.2.0"): _release(release + [sdist])})
    assert guard.on_pypi("idx", "0.2.0", local, get=pypi, provenance=None) == [
        f"kitten-inference 0.2.0 on idx has files this release does not: {sdist}"]


def test_verify_wants_this_repositorys_packages_yml(guard):
    # A file uploaded some other way (an API token, another repository's workflow, another workflow here) fails.
    release = _release_wheels()
    local = {n: _digest(n) for n in release}
    pypi = _pypi({("kitten-inference", "0.2.0"): _release(release)})
    asked = []
    assert guard.on_pypi("idx", "0.2.0", local, get=pypi, provenance=_provenance(asked=asked)) == []
    assert sorted(asked) == release  # every file's provenance
    for provenance in (_provenance(None), _provenance("someone/kitten-inference"),
                       _provenance(workflow="tts-0.8.yml")):
        problems = guard.on_pypi("idx", "0.2.0", local, get=pypi, provenance=provenance)
        assert len(problems) == 54 and all("not published by KittenML/kitten-inference (packages.yml)" in p
                                           for p in problems)
    assert guard.published_by({"attestation_bundles": [{"publisher": {
        "kind": "GitHub", "repository": "kittenml/Kitten-Inference", "workflow": ".github/workflows/packages.yml"}}]},
        "KittenML/kitten-inference", "packages.yml")
    # A file found good is not asked about again on the next try.
    checked, asked = set(), []
    guard.on_pypi("idx", "0.2.0", local, get=pypi, provenance=_provenance(asked=asked), checked=checked)
    guard.on_pypi("idx", "0.2.0", local, get=pypi, provenance=_provenance(asked=asked), checked=checked)
    assert len(asked) == 54 and checked == set(release)


def test_verify_retries_until_the_timeout(guard, capsys):
    import urllib.error
    release = _release_wheels()
    local = {n: _digest(n) for n in release}
    answers = [urllib.error.HTTPError("u", 503, "Service Unavailable", {}, None), TimeoutError("timed out"),
               ConnectionResetError(104, "reset"), {"urls": []}]

    def flaky(index, name, version):
        if answers:
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        return _release(release)

    slept = []
    assert guard.verify("idx", "0.2.0", local, 3600, 60, get=flaky, provenance=None, sleep=slept.append) == []
    out = capsys.readouterr().out
    assert len(slept) == 4 and "503" in out and "54 of 54 wheels not on idx yet" in out
    # At the timeout it reports what is still missing instead of passing.
    clock = iter([0, 7201])
    problems = guard.verify("idx", "0.2.0", local, 7200, 60, get=_pypi({}), provenance=None,
                            sleep=slept.append, clock=lambda: next(clock))
    assert problems == ["kitten-inference 0.2.0: not on idx yet (0 of 54 wheels)"]


def test_verify_command_line_refuses_a_wrong_or_partial_set(guard, tmp_path):
    def run(version, names):
        folder = tmp_path / version
        folder.mkdir(exist_ok=True)
        for n in names:
            (folder / n).write_bytes(b"")
        return subprocess.run([sys.executable, str(SCRIPT), "verify", "--version", version, "--index",
                               "http://127.0.0.1:9", "--timeout-minutes", "0", *sorted(str(folder / n) for n in names)],
                              capture_output=True, text=True)

    # Before asking PyPI: every file must be the version's, and the set complete.
    other = run("0.2.0.post1", _release_wheels("0.2.0"))
    assert other.returncode == 1 and "is not version 0.2.0.post1" in other.stdout, other.stdout
    partial = run("0.2.0", _release_wheels()[1:])
    assert partial.returncode == 1 and "not built: cp310-macosx_11_0_arm64" in partial.stdout, partial.stdout
    assert guard.file_version("kitten_inference-0.2.0.post1-cp312-cp312-win_amd64.whl") == "0.2.0.post1"


def test_warns_while_0_1_is_not_yanked(guard):
    def wheel(version, yanked=False):
        return {"filename": f"kitten_inference-{version}-py3-none-any.whl", "packagetype": "bdist_wheel",
                "yanked": yanked}
    old = {"releases": {"0.1.0": [wheel("0.1.0", yanked=True)], "0.1.1": [wheel("0.1.1")]}}
    assert guard.earlier_not_yanked("idx", get=lambda index, name: old) == ["0.1.1"]
    assert guard.earlier_not_yanked("idx", get=lambda index, name: None) == []


# The workflows (read as text: no YAML parser needed)


def _workflows():
    if not WORKFLOWS.is_dir():
        pytest.skip("not a source checkout (.github/workflows is not here)")
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(WORKFLOWS.glob("*.yml"))}


def _jobs(text):
    """The `jobs:` section of a workflow."""
    return text.split("\njobs:\n", 1)[1]


def _job(text, name):
    """The text of one job of a workflow (from `  name:` to the next job or the end)."""
    m = re.search(rf"^  {re.escape(name)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)", _jobs(text), re.S | re.M)
    assert m, f"no job {name}"
    return m.group(1)


def _steps_using(text, action):
    """The text of every step that uses ``action`` (its `uses:` line and the lines indented under it)."""
    lines = text.splitlines()
    steps = []
    for i, line in enumerate(lines):
        if f"uses: {action}@" not in line:
            continue
        indent = line.index("uses:")
        block = [line]
        for nxt in lines[i + 1:]:
            stripped = nxt.lstrip(" ")
            if stripped and (len(nxt) - len(stripped) < indent or stripped.startswith("- ")):
                break
            block.append(nxt)
        steps.append("\n".join(block))
    return steps


def _uploaded(text):
    """The artifact names a workflow uploads, with ${{ matrix.artifact }} expanded over its matrix values."""
    names = set()
    values = re.findall(r"^\s+artifact: ([A-Za-z0-9_.-]+)\s*$", text, re.M)
    for step in _steps_using(text, "actions/upload-artifact"):
        m = re.search(r"^\s+name: (.+?)\s*$", step, re.M)
        if not m:
            continue
        name = m.group(1)
        if "${{ matrix.artifact }}" in name:
            names |= {name.replace("${{ matrix.artifact }}", v) for v in values}
        else:
            names.add(name)
    return names


def test_only_packages_yml_can_upload_to_pypi():
    flows = _workflows()
    for name, text in flows.items():
        if name == "packages.yml":
            continue
        assert "id-token" not in text, f"{name} must not ask for an OIDC token"
        assert "gh-action-pypi-publish" not in text and "environment: pypi" not in text, name
        assert not re.search(r"^\s+tags:", text, re.M), f"{name} must not run on tags"
    text = flows["packages.yml"]
    publish = _job(text, "publish")
    assert text.count("id-token: write") == 1 and "id-token: write" in publish
    uses = re.findall(r"uses: (pypa/gh-action-pypi-publish@\S+)", text)
    assert uses == [PUBLISH_ACTION] and PUBLISH_ACTION in publish


def test_the_publish_job_runs_only_for_a_release_tag_after_every_test():
    text = _workflows()["packages.yml"]
    assert re.search(r"^    tags: \[\"v\*\"\]$", text, re.M)
    publish = _job(text, "publish")
    assert "environment: pypi" in publish
    condition = re.search(r"^    if: (.*)$", publish, re.M).group(1)
    for part in ("github.event_name == 'push'", "github.repository == 'KittenML/kitten-inference'",
                 "startsWith(github.ref, 'refs/tags/v')"):
        assert part in condition, part
    assert "always()" not in condition and "||" not in condition
    needs = re.search(r"^    needs: \[(.*)\]$", publish, re.M).group(1)
    # source-tests runs only when nothing is assembled, so never before a publish (a skipped need would skip it).
    jobs = set(re.findall(r"^  ([A-Za-z0-9_-]+):\n", _jobs(text), re.M)) - {"publish", "build", "lock",
                                                                            "source-tests"}
    assert {n.strip() for n in needs.split(",")} == jobs
    assert "release_files.py assembled upload" in publish and "pypi_ready.py verify" in publish
    assert "skip-existing: true" in publish


def test_assembled_artifacts_are_one_per_family():
    text = _workflows()["packages.yml"]
    assert _uploaded(_job(text, "assemble")) == {"assembled-" + f for f in FAMILIES} | {"assembly-report"}
    # A dry run's are kept as long as a tag run's: .github/RELEASING.md tests its wheels
    # and compares its sha256 list.
    for step in _steps_using(_job(text, "assemble"), "actions/upload-artifact"):
        assert ("retention-days: ${{ (startsWith(github.ref, 'refs/tags/') || "
                "github.event_name == 'workflow_dispatch') && 90 || 14 }}") in step, step
    assert re.search(r"pattern: assembled-\*$", _job(text, "publish"), re.M)
    for family in FAMILIES:
        if family != "android-arm64":  # no runner: check_assembled.py checks its contents
            assert f"family={family}" in _job(text, "test"), family


def test_the_input_artifacts_are_kept_90_days():
    flows = _workflows()
    for workflow, names in INPUT_ARTIFACTS.items():
        text = flows[workflow]
        assert set(names) <= _uploaded(text), (workflow, set(names) - _uploaded(text))
        for step in _steps_using(text, "actions/upload-artifact"):
            name = re.search(r"^\s+name: (.+?)\s*$", step, re.M).group(1)
            if name.startswith(("tts08-wheels-", "tts2-cu12-", "kitten-inference-tts2-rocm-")):
                assert re.search(r"^\s+retention-days: 90\s*$", step, re.M), (workflow, name)


def test_the_lock_names_artifacts_these_workflows_upload():
    if not LOCK.is_file():
        pytest.skip(".github/release-inputs.json is not here yet")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    flows = _workflows()
    assert set(lock["runs"]) == set(RUN_WORKFLOWS)
    for key, workflow in RUN_WORKFLOWS.items():
        assert lock["runs"][key]["workflow"] == workflow, key
        named = {row["artifact"] for row in lock["inputs"] if row["run"] == key}
        assert named and named <= set(INPUT_ARTIFACTS[workflow]) <= _uploaded(flows[workflow]), (key, named)
    assert "tts08-wheels-linux-x86_64-manylinux2014" not in {row["artifact"] for row in lock["inputs"]}


def test_the_lock_job_checks_what_the_assembly_would_refuse():
    # A push or pull request that changes the engine sources or the versions before new builds are locked skips the
    # assembly instead of failing it; the Python unit tests then run without the assembled wheels.
    text = _workflows()["packages.yml"]
    lock = _job(text, "lock")
    called = set(re.findall(r"\baw\.([A-Za-z_]+)\(", lock))
    assert {"load_lock", "check_sources", "Source"} <= called, called
    assert "umbrella/pyproject.toml" in lock and "tts-2/python/pyproject.toml" in lock
    assert "strict = tag_run or by_hand" in lock
    path = ROOT / ".github" / "scripts" / "assemble_wheels.py"
    if not path.is_file():
        pytest.skip("not a source checkout (.github/scripts is not here)")
    spec = importlib.util.spec_from_file_location("assemble_wheels_of_the_lock_job", path)
    aw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aw)
    for name in sorted(called):
        assert callable(getattr(aw, name, None)), f"the lock job calls assemble_wheels.{name}, which is not there"
    source = _job(text, "source-tests")
    assert re.search(r"^    if: needs\.lock\.outputs\.assemble == 'false'$", source, re.M)
    assert "umbrella/tests" in source and "tts-2/python/tests" in source


def test_releasing_md_names_the_publisher():
    path = ROOT / ".github" / "RELEASING.md"
    if not path.is_file():
        pytest.skip("not a source checkout (.github/RELEASING.md is not here)")
    text = path.read_text(encoding="utf-8")
    assert "packages.yml" in text and "`pypi`" in text and "KittenML" in text
    for name in ("kitten-inference-tts08", "kitten-inference-tts2", "tts-2-rocm]"):
        assert name not in text, name
