"""The installed package: import, metadata, the bundled preparation code and the console script."""
import importlib
import subprocess
import sys

import pytest

import kitten_inference.tts2 as tts2


def _dist():
    from importlib import metadata
    try:
        return metadata.distribution("kitten-inference-tts2")
    except metadata.PackageNotFoundError:
        pytest.skip("kitten-inference-tts2 is not installed (running from the source tree)")


def test_import_and_version():
    assert tts2.__version__ == "0.2.0"
    assert callable(tts2.KittenTTS2)
    for name in tts2.__all__:
        assert hasattr(tts2, name), name


def test_metadata():
    dist = _dist()
    meta = dist.metadata
    assert meta["Name"] == "kitten-inference-tts2"
    assert dist.version == "0.2.0"
    assert meta["Requires-Python"] == ">=3.9"
    requires = {r.split(";")[0].replace(" ", "") for r in (dist.requires or [])}
    names = {r.split(">")[0].split("<")[0].split("=")[0].lower() for r in requires}
    assert names == {"numpy", "huggingface-hub", "tokenizers", "filelock"}, requires
    assert not any("torch" in r for r in requires)
    scripts = {ep.name: ep.value for ep in dist.entry_points if ep.group == "console_scripts"}
    assert scripts == {"kitten-tts2": "kitten_inference.tts2._cli:main"}


def test_wheel_does_not_own_the_namespace_init():
    dist = _dist()
    files = {str(f).replace("\\", "/") for f in dist.files or []}
    assert "kitten_inference/__init__.py" not in files  # the umbrella (kitten-inference) owns it
    assert "kitten_inference/tts2/__init__.py" in files
    assert "kitten_inference/tts2/_assets/prepare.py" in files
    assert "kitten_inference/tts2/NOTICE" in files


def test_bundled_prepare_imports_without_torch():
    code = ("import sys; import kitten_inference.tts2._assets.prepare as p; "
            "import kitten_inference.tts2._assets.convert_s3gen, kitten_inference.tts2._assets.voices; "
            "assert callable(p.main); print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_bundled_prepare_data_files():
    from importlib import resources
    root = resources.files("kitten_inference.tts2._assets")
    for name in ("s3gen_tensors.tsv", "clone_consts_reference.json"):
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
