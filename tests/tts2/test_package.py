"""The installed package: import, metadata, the bundled preparation code, the engines and the console script."""
import importlib
import platform
import re
import subprocess
import sys
from pathlib import Path

import pytest

import kitten_inference.tts2 as tts2


def _dist():
    from importlib import metadata
    try:
        return metadata.distribution("kitten-inference")
    except metadata.PackageNotFoundError:
        pytest.skip("kitten-inference is not installed (running from the source tree)")


def _files(dist):
    return {str(f).replace("\\", "/") for f in dist.files or []}


PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_import_and_version():
    # __version__ is a literal (CONTRACT: no metadata lookups), the version of this build input (engines/tts2/python); a
    # post release changes only kitten-inference's own version (umbrella)
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.post\d+)?", tts2.__version__), tts2.__version__
    if PYPROJECT.is_file():
        assert tts2.__version__ == re.search(r'^version = "([^"]+)"', PYPROJECT.read_text(encoding="utf-8"),
                                             re.M).group(1)


def test_metadata():
    from packaging.requirements import Requirement  # a dependency of pytest
    from packaging.utils import canonicalize_name
    dist = _dist()
    meta = dist.metadata
    assert meta["Name"] == "kitten-inference"
    # A post release (0.2.0.post1) changes only the distribution's version; the Python packages keep theirs.
    assert re.sub(r"\.post\d+$", "", dist.version) == tts2.__version__, dist.version
    assert meta["Requires-Python"] == ">=3.8"
    assert not meta.get_all("Provides-Extra")
    requires = [Requirement(r) for r in dist.requires or []]
    assert not any("torch" in r.name for r in requires)
    expected = {"numpy", "huggingface-hub", "soundfile", "phonemizer", "espeakng-loader", "tokenizers", "filelock"}
    assert {canonicalize_name(r.name) for r in requires} == expected
    scripts = {ep.name: ep.value for ep in dist.entry_points if ep.group == "console_scripts"}
    assert scripts == {"kitten-tts": "kitten_inference._cli:main"}


def test_one_package_holds_everything():
    dist = _dist()
    files = _files(dist)
    assert "kitten_inference/__init__.py" in files
    assert "kitten_inference/tts2/__init__.py" in files, "kitten-inference does not hold kitten_inference.tts2"
    assert "kitten_inference/tts2/_assets/prepare.py" in files
    assert "kitten_inference/tts2/NOTICE" in files
    # The engines: NVIDIA in the Linux x86-64 and Windows x64 wheels, AMD in the Linux x86-64 ones (Python 3.9+).
    import os
    if os.environ.get("KITTEN_RELEASE_WHEEL") != "1":
        return  # Source builds compile CPU; release assembly verifies GPU payloads separately.
    machine = platform.machine()
    linux_x64 = sys.platform == "linux" and machine == "x86_64" and sys.version_info >= (3, 9)
    windows_x64 = sys.platform == "win32" and machine == "AMD64" and sys.version_info >= (3, 9)
    assert ("kitten_inference/_tts2_cuda/__init__.py" in files) == (linux_x64 or windows_x64)
    assert ("kitten_inference/_tts2_rocm/__init__.py" in files) == linux_x64


def test_bundled_prepare_imports_without_torch():
    code = ("import sys; import kitten_inference.tts2._assets.prepare as p; "
            "import kitten_inference.tts2._assets.convert_s3gen, kitten_inference.tts2._assets.voices, "
            "kitten_inference.tts2._assets.clone_assets; "
            "assert callable(p.main); print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_bundled_prepare_has_the_contract_flags():
    # kitten_inference.tts2._prepare passes these to prepare.main (CONTRACT: prepare entry point).
    code = "import sys; from kitten_inference.tts2._assets.prepare import main; sys.exit(main(['--help']))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    for flag in ("--model-dir", "--out", "--text-data-tool", "--weight-norm-device", "--with-cloning",
                 "--cloning-only", "--acoustic-source", "--cache-dir", "--revision"):
        assert flag in out.stdout, flag
    assert "numpy" in out.stdout
    from kitten_inference.tts2._assets import prepare
    assert prepare.MODEL_ALLOW_PATTERNS and prepare.CLONING_ALLOW_PATTERNS
    assert prepare.ACOUSTIC_REPO and prepare.ACOUSTIC_FILE and prepare.convert_s3gen.PINNED_REVISION


def test_bundled_prepare_data_files():
    from importlib import resources
    root = resources.files("kitten_inference.tts2._assets")
    for name in ("s3gen_tensors.tsv", "clone_consts_reference.json", "clone_sincnet_reference.json"):
        assert root.joinpath(name).is_file(), name


def test_import_does_not_load_an_engine():
    code = ("import sys, kitten_inference.tts2; "
            "print(any(m.startswith('kitten_inference._tts2_') for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_module_entry_point_help(no_backends):
    out = subprocess.run([sys.executable, "-m", "kitten_inference.tts2", "--help"], capture_output=True, text=True)
    assert out.returncode in (0, 1), out.stderr
    assert "kitten-tts2" in out.stdout
