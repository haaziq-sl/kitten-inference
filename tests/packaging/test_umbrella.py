"""kitten-inference: version, namespace extension, no imports, the extras, and what the installed wheel holds."""
import ast
import importlib.util
import platform
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
REPO = Path(__file__).resolve().parents[2]

def test_version():
    # __version__ is a literal (CONTRACT: no metadata lookups); a post release must change it with the pyproject
    import kitten_inference
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.post\d+)?", kitten_inference.__version__), kitten_inference.__version__
    if PYPROJECT.is_file():
        assert kitten_inference.__version__ == re.search(r'^version = "([^"]+)"',
                                                         PYPROJECT.read_text(encoding="utf-8"), re.M).group(1)


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


