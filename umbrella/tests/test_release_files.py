"""The release upload picker (.github/scripts/release_files.py): which files of the CI artifacts a v* tag uploads."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "release_files.py"
CU12 = "kitten-inference-tts2-cu12"
PLATFORMS = ["manylinux_2_28_x86_64", "win_amd64"]


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


def test_file_names(pick):
    assert pick.parse("kitten_inference_tts2_cu12-0.2.0-py3-none-win_amd64.whl") == (CU12, "0.2.0")
    assert pick.parse("kitten_inference_tts08-0.2.0.post1-cp312-cp312-macosx_11_0_arm64.whl") == \
        ("kitten-inference-tts08", "0.2.0.post1")
    assert pick.parse("kitten_inference-0.2.0.tar.gz") == ("kitten-inference", "0.2.0")
    assert pick.parse("auditwheel-show.txt") is None and pick.parse("build-rocm-wheel.txt") is None


def test_engine_wheels_of_the_tag(pick, tmp_path):
    art = _files(tmp_path / "a", "tts2-cu12-linux/kitten_inference_tts2_cu12-0.2.0-py3-none-manylinux_2_28_x86_64.whl",
                 "tts2-cu12-linux/auditwheel-show.txt", "tts2-cu12-windows/dll-imports.txt",
                 "tts2-cu12-windows/kitten_inference_tts2_cu12-0.2.0-py3-none-win_amd64.whl")
    files = pick.select(art, CU12)
    assert [f.name for f in files] == ["kitten_inference_tts2_cu12-0.2.0-py3-none-manylinux_2_28_x86_64.whl",
                                       "kitten_inference_tts2_cu12-0.2.0-py3-none-win_amd64.whl"]
    assert pick.decide(files, "v0.2.0", PLATFORMS) == ("0.2.0", True)
    assert pick.decide(files, "v0.2.0.post1", PLATFORMS) == ("0.2.0", False)  # another distribution's post release
    with pytest.raises(SystemExit, match="no wheel for win_amd64"):
        pick.decide(files[:1], "v0.2.0", PLATFORMS)


def test_umbrella_is_not_its_engines(pick, tmp_path):
    art = _files(tmp_path / "dist", "kitten_inference-0.2.0-py3-none-any.whl", "kitten_inference-0.2.0.tar.gz",
                 "kitten_inference_tts2-0.2.0-py3-none-any.whl", "kitten_inference_tts2-0.2.0.tar.gz")
    assert [f.name for f in pick.select(art, "kitten-inference")] == ["kitten_inference-0.2.0-py3-none-any.whl",
                                                                      "kitten_inference-0.2.0.tar.gz"]
    assert len(pick.select(art, "kitten-inference-tts2")) == 2


def test_broken_sets_fail(pick, tmp_path):
    with pytest.raises(SystemExit, match="no wheel or sdist"):
        pick.decide(pick.select(_files(tmp_path / "e", "notes.txt"), CU12), "v0.2.0", [])
    mixed = _files(tmp_path / "m", "kitten_inference_tts2_cu12-0.2.0-py3-none-win_amd64.whl",
                   "kitten_inference_tts2_cu12-0.2.0.post1-py3-none-manylinux_2_28_x86_64.whl")
    with pytest.raises(SystemExit, match="several versions"):
        pick.decide(pick.select(mixed, CU12), "v0.2.0", [])
    twice = _files(tmp_path / "t", "x/kitten_inference_tts2_cu12-0.2.0-py3-none-win_amd64.whl",
                   "y/kitten_inference_tts2_cu12-0.2.0-py3-none-win_amd64.whl")
    with pytest.raises(SystemExit, match="twice"):
        pick.select(twice, CU12)
