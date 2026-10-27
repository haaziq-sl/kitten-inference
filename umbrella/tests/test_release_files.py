"""The release upload picker (.github/scripts/release_files.py): which files of the assembled artifacts a v* tag
uploads. Only kitten-inference wheels of the tag's version, never an sdist, a py3-none-any wheel or the internal
builds the wheels are assembled from."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "release_files.py"
RELEASE = ["kitten_inference-0.2.0-cp312-cp312-manylinux_2_28_x86_64.whl",
           "kitten_inference-0.2.0-cp38-cp38-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
           "kitten_inference-0.2.0-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.whl",
           "kitten_inference-0.2.0-cp312-cp312-win_amd64.whl",
           "kitten_inference-0.2.0-cp313-cp313-android_24_arm64_v8a.whl"]


@pytest.fixture(scope="module")
def pick():
    if not SCRIPT.is_file():
        pytest.skip("not a source checkout (.github/scripts is not here)")
    spec = importlib.util.spec_from_file_location("release_files", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _files(root: Path, *names):
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    return root


def _artifacts(root: Path, names=RELEASE):
    """The wheels as the publish job downloads them: one folder per assembled-<family> artifact (or merged)."""
    return _files(root, *(f"assembled-{i}/{n}" for i, n in enumerate(names)))


def test_file_names(pick):
    assert pick.parse("kitten_inference-0.2.0-cp312-cp312-manylinux_2_28_x86_64.whl") == \
        ("kitten-inference", "0.2.0", "cp312", "manylinux_2_28_x86_64")
    assert pick.parse("kitten_inference_tts08-0.2.0.post1-cp312-cp312-macosx_11_0_arm64.whl") == \
        ("kitten-inference-tts08", "0.2.0.post1", "cp312", "macosx_11_0_arm64")
    assert pick.parse("kitten_inference-0.2.0.tar.gz") == ("kitten-inference", "0.2.0", None, None)
    assert pick.parse("assembly-report.json") is None and pick.parse("SHA256SUMS") is None


def test_the_tag_releases_every_wheel_of_its_version(pick, tmp_path):
    art = _artifacts(tmp_path / "a")
    _files(art, "assembled-0/notes.txt")  # other files are not packages: left alone
    files = pick.select(art, "v0.2.0")
    assert [f.name for f in files] == sorted(RELEASE)
    pick.copy(files, tmp_path / "upload")
    assert sorted(p.name for p in (tmp_path / "upload").iterdir()) == sorted(RELEASE)
    post = _artifacts(tmp_path / "p", [n.replace("-0.2.0-", "-0.2.0.post1-") for n in RELEASE])
    assert len(pick.select(post, "v0.2.0.post1")) == len(RELEASE)


def test_a_tag_of_another_version_is_an_error(pick, tmp_path):
    art = _artifacts(tmp_path / "a")
    for tag, message in (("v0.2.0.post1", "the tag v0.2.0.post1 names 0.2.0.post1"), ("v0.2.1", "names 0.2.1"),
                         ("0.2.0", "a release tag is v<version>"), ("", "no tag: a release tag")):
        with pytest.raises(SystemExit, match=message):
            pick.select(art, tag)


@pytest.mark.parametrize("name", [
    "kitten_inference_tts08-0.2.0-cp312-cp312-win_amd64.whl",
    "kitten_inference_tts2-0.2.0-py3-none-any.whl",
    "kitten_inference_tts2_cu12-0.2.0-py3-none-manylinux_2_28_x86_64.whl",
    "kitten_inference_tts2_rocm-0.2.0-py3-none-manylinux_2_28_x86_64.whl",
    "renamed_kitten_inference-0.2.0-cp312-cp312-win_amd64.whl",
    "kitten_inference-0.2.0-py3-none-any.whl",
    "kitten_inference-0.2.0.tar.gz",
])
def test_nothing_else_ever_reaches_the_upload(pick, tmp_path, name):
    # The internal TTS 0.8 and engine builds, the pure TTS 2 wheel, renamed copies, and a kitten-inference wheel or
    # sdist that would install without any engine: the whole release stops, nothing is copied.
    art = _artifacts(tmp_path / "a", RELEASE + [name])
    with pytest.raises(SystemExit, match="error: "):
        pick.select(art, "v0.2.0")
    assert not (tmp_path / "upload").exists()


def test_broken_sets_fail(pick, tmp_path):
    with pytest.raises(SystemExit, match="no kitten-inference wheel"):
        pick.select(_files(tmp_path / "e", "notes.txt"), "v0.2.0")
    mixed = _artifacts(tmp_path / "m", RELEASE[:2] + [RELEASE[2].replace("-0.2.0-", "-0.2.0.post1-")])
    with pytest.raises(SystemExit, match="several versions"):
        pick.select(mixed, "v0.2.0")
    twice = _files(tmp_path / "t", f"x/{RELEASE[0]}", f"y/{RELEASE[0]}")
    with pytest.raises(SystemExit, match="twice"):
        pick.select(twice, "v0.2.0")
    full = _files(tmp_path / "upload", "stale.whl")
    with pytest.raises(SystemExit, match="not empty"):
        pick.copy(pick.select(_artifacts(tmp_path / "a"), "v0.2.0"), full)


def test_command_line(pick, tmp_path):
    # As the publish job runs it: release_files.py assembled upload (the tag from $GITHUB_REF_NAME, or --tag).
    art = _artifacts(tmp_path / "a")
    run = subprocess.run([sys.executable, str(SCRIPT), str(art), str(tmp_path / "upload"), "--tag", "v0.2.0"],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert f"{len(RELEASE)} kitten-inference 0.2.0 wheels" in run.stdout, run.stdout
    bad = subprocess.run([sys.executable, str(SCRIPT), str(art), str(tmp_path / "u2"), "--tag", "v0.3.0"],
                         capture_output=True, text=True)
    assert bad.returncode == 1 and "names 0.3.0" in bad.stderr and not (tmp_path / "u2").exists()
