"""The umbrella package: version, namespace extension, no engine imports, and its extras."""
import importlib
import subprocess
import sys
import textwrap

import pytest

EXTRAS = {
    "tts-0-8": {"kitten-inference-tts08>=0.2.0,<0.3"},
    "tts-2": {"kitten-inference-tts2>=0.2.0,<0.3", "kitten-inference-tts2-cu12>=0.2.0,<0.3"},
    "tts-2-rocm": {"kitten-inference-tts2>=0.2.0,<0.3", "kitten-inference-tts2-rocm>=0.2.0,<0.3"},
}


def test_version():
    import kitten_inference
    assert kitten_inference.__version__ == "0.2.0"


def test_import_does_nothing_else():
    code = "import sys, kitten_inference; print(sorted(m for m in sys.modules if m.startswith('kitten_inference')))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "['kitten_inference']"


def test_engines_found_in_another_path_entry(tmp_path):
    # An engine installed in another sys.path entry (no kitten_inference/__init__.py there) is still importable.
    pkg = tmp_path / "kitten_inference" / "fake_engine"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("NAME = 'fake'\n")
    code = textwrap.dedent(f"""
        import sys
        sys.path.append({str(tmp_path)!r})
        import kitten_inference, kitten_inference.fake_engine
        print(kitten_inference.fake_engine.NAME, len(kitten_inference.__path__))
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["fake", "2"]


def test_missing_engine_is_a_plain_import_error():
    code = "import kitten_inference.tts08_not_installed"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 1 and "ModuleNotFoundError" in out.stderr


def test_metadata_extras():
    from importlib import metadata
    try:
        dist = metadata.distribution("kitten-inference")
    except metadata.PackageNotFoundError:
        pytest.skip("kitten-inference is not installed")
    assert dist.version == "0.2.0"
    assert set(dist.metadata.get_all("Provides-Extra")) == set(EXTRAS)
    def canonical(spec):  # "name<0.3,>=0.2.0" -> ("name", {"<0.3", ">=0.2.0"})
        spec = spec.replace(" ", "")
        cut = min(i for i, c in enumerate(spec) if c in "<>=!~")
        return spec[:cut], frozenset(spec[cut:].split(","))
    found = {}
    for req in dist.requires or []:
        spec, _, marker = req.partition(";")
        extra = marker.split("==")[1].strip().strip("'\"")
        found.setdefault(extra, set()).add(canonical(spec))
    assert found == {extra: {canonical(s) for s in specs} for extra, specs in EXTRAS.items()}
    files = {str(f).replace("\\", "/") for f in dist.files or []}
    assert {f for f in files if f.startswith("kitten_inference/")} == {"kitten_inference/__init__.py"}
