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

# Where Kitten TTS 2 runs: [tts-2] installs its libraries only there, so elsewhere it installs nothing.
TTS2_HERE = {("linux", "x86_64"), ("win32", "AMD64")}
TTS2_MARKER = ('python_version >= "3.9" and ((sys_platform == "linux" and platform_machine == "x86_64") or '
               '(sys_platform == "win32" and platform_machine == "AMD64"))')
# The extras add only third-party libraries; the engines are in the package itself.
EXTRAS = {
    "tts-0-8": ["espeakng-loader>=0.2.4", "phonemizer>=3.3", "soundfile>=0.12.1", "numpy>=1.23",
                "huggingface-hub>=0.20"],
    "tts-2": [f"{spec}; {TTS2_MARKER}" for spec in ("numpy>=1.24", "huggingface-hub>=0.25,<2",
                                                    "tokenizers>=0.22,<0.23", "filelock>=3.12")],
}
# The import names each engine looks up before it imports any of its libraries (file, variable).
CHECKED = {"tts-0-8": ("tts08/_deps.py", "REQUIRED"), "tts-2": ("tts2/_backend.py", "PYTHON_MODULES")}
# The markers are compared by their value in these environments (sys_platform, platform_machine) x Python.
PLATFORMS = [("linux", "x86_64"), ("linux", "aarch64"), ("darwin", "arm64"), ("darwin", "x86_64"), ("win32", "AMD64"),
             ("win32", "ARM64"), ("android", "aarch64")]
PYTHONS = ["3.8", "3.9", "3.12", "3.15"]

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
REPO = Path(__file__).resolve().parents[2]
SOURCES = {"tts08": REPO / "tts-0.8" / "kitten_inference" / "tts08",
           "tts2": REPO / "tts-2" / "python" / "kitten_inference" / "tts2"}
ENGINE_PYPROJECTS = {"tts-0-8": REPO / "tts-0.8" / "pyproject.toml",
                     "tts-2": REPO / "tts-2" / "python" / "pyproject.toml"}


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


def _flat(extras):
    """Requires-Dist lines for {extra: [requirement]}, in the form hatchling writes."""
    lines = []
    for extra, requirements in extras.items():
        for requirement in requirements:
            spec, _, marker = (part.strip() for part in requirement.partition(";"))
            lines.append(f"{spec}; ({marker}) and extra == '{extra}'" if marker else f"{spec}; extra == '{extra}'")
    return lines


def _parse(lines):
    """[((PEP 503 name, version ranges), marker)]. Parsed, not text: hatchling rewrites quotes, names and the order
    of ranges."""
    pytest.importorskip("packaging.requirements")
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    return [((canonicalize_name(r.name), frozenset(str(s) for s in r.specifier)), r.marker)
            for r in map(Requirement, lines)]


def _ranges(lines):
    return {key for key, _ in _parse(lines)}


def _table(lines):
    """For each extra (none, ours, and the removed tts-2-rocm) and environment: the (name, version ranges) that apply.
    Markers are compared by their value, which also checks the platform limit of [tts-2]."""
    parsed = _parse(lines)
    table = {}
    for extra in ("", *EXTRAS, "tts-2-rocm"):
        for sys_platform, machine in PLATFORMS:
            for python in PYTHONS:
                env = {"sys_platform": sys_platform, "platform_machine": machine, "python_version": python,
                       "python_full_version": f"{python}.0", "extra": extra}
                table[extra, sys_platform, machine, python] = {
                    key for key, marker in parsed if marker is None or marker.evaluate(env)}
    return table


def _toml(path):
    try:
        import tomllib
    except ImportError:  # Python < 3.11: pytest itself depends on tomli there
        import tomli as tomllib
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _names(requirements):
    return {re.sub(r"[-_.]+", "-", re.match(r"[A-Za-z0-9_.-]+", r).group(0)).lower() for r in requirements}


def test_the_table_says_where_each_extra_installs():
    for (extra, sys_platform, machine, python), found in _table(_flat(EXTRAS)).items():
        names = {name for name, _ in found}
        if extra == "tts-0-8":
            assert names == _names(EXTRAS["tts-0-8"])
        elif extra == "tts-2" and (sys_platform, machine) in TTS2_HERE and python != "3.8":
            assert names == {"numpy", "huggingface-hub", "tokenizers", "filelock"}
        else:
            assert not names, (extra, sys_platform, machine, python)


def test_pyproject_extras():
    project = _toml(PYPROJECT)["project"]
    assert project["dependencies"] == []
    extras = {re.sub(r"[-_.]+", "-", k).lower(): v for k, v in project["optional-dependencies"].items()}
    assert _table(_flat(extras)) == _table(_flat(EXTRAS))
    assert project["scripts"] == {"kitten-tts2": "kitten_inference.tts2._cli:main"}
    assert project["requires-python"] == ">=3.8"


def test_extras_match_the_engine_pyprojects():
    # The engines' own build inputs name the same libraries and versions; only the umbrella adds the markers.
    if not all(p.is_file() for p in ENGINE_PYPROJECTS.values()):
        pytest.skip("not run from the repository")
    for extra, path in ENGINE_PYPROJECTS.items():
        assert _ranges(_toml(path)["project"]["dependencies"]) == _ranges(EXTRAS[extra]), path


def _source(relative):
    """An engine file of the installed package, or of this checkout when that engine is not installed."""
    sub, _, rest = relative.partition("/")
    spec = importlib.util.find_spec(f"kitten_inference.{sub}")  # imports kitten_inference only, not the engine
    folder = Path(spec.origin).parent if spec and spec.origin else SOURCES[sub]
    return folder / rest


def _literal(path, name):
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None and any(
                isinstance(t, ast.Name) and t.id == name for t in targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"no {name} in {path}")


@pytest.mark.parametrize("extra", sorted(CHECKED))
def test_engine_checks_name_the_extra_libraries(extra):
    # Pip names to import names: espeakng-loader -> espeakng_loader, huggingface-hub -> huggingface_hub.
    relative, name = CHECKED[extra]
    modules = _literal(_source(relative), name)
    assert sorted(modules) == sorted(n.replace("-", "_") for n in _names(EXTRAS[extra]))


@pytest.fixture(scope="module")
def dist():
    from importlib import metadata
    try:
        return metadata.distribution("kitten-inference")
    except metadata.PackageNotFoundError:
        pytest.skip("kitten-inference is not installed")


def _build_input(dist):
    """The py3-none-any wheel built from umbrella/ alone: an input of the assembled wheels, never uploaded."""
    tags = [line.split(":", 1)[1].strip() for line in (dist.read_text("WHEEL") or "").splitlines()
            if line.startswith("Tag:")]
    return tags == ["py3-none-any"]


def _engines_here():
    """The GPU engines a kitten-inference wheel for this interpreter holds. The cp38 wheels have none: TTS 2 needs
    Python 3.9."""
    machine = platform.machine().lower()
    linux_x64 = sys.platform.startswith("linux") and machine in ("x86_64", "amd64")
    windows_x64 = sys.platform == "win32" and machine in ("amd64", "x86_64")
    if sys.version_info < (3, 9):
        return set()
    return ({"_tts2_cuda"} if linux_x64 or windows_x64 else set()) | ({"_tts2_rocm"} if linux_x64 else set())


def test_metadata(dist):
    import kitten_inference
    assert dist.version == kitten_inference.__version__
    assert dist.metadata["Requires-Python"] == ">=3.8"
    assert set(dist.metadata.get_all("Provides-Extra")) == set(EXTRAS)
    assert _table(dist.requires or []) == _table(_flat(EXTRAS))
    scripts = sorted((ep.name, ep.value) for ep in dist.entry_points if ep.group == "console_scripts")
    assert scripts == [("kitten-tts2", "kitten_inference.tts2._cli:main")]
    if PYPROJECT.is_file():
        license_ = re.search(r'^license = "([^"]+)"', PYPROJECT.read_text(encoding="utf-8"), re.M).group(1)
        assert dist.metadata["License-Expression"] == license_


def test_installed_files(dist):
    info = f"kitten_inference-{dist.version}.dist-info"
    # Paths starting with .. are the kitten-tts2 script.
    files = {str(f).replace("\\", "/") for f in dist.files or []}
    ours = {f for f in files if not f.startswith("../") and "__pycache__" not in f}
    if _build_input(dist):
        assert {f for f in ours if not f.startswith(info + "/")} == {"kitten_inference/__init__.py"}
        return
    subpackages = {"tts08", "tts2"} | _engines_here()
    assert {f.split("/")[1] for f in ours if f.startswith("kitten_inference/") and f.count("/") > 1} == subpackages
    assert {"kitten_inference/__init__.py", *(f"kitten_inference/{s}/__init__.py" for s in subpackages)} <= ours
    assert {f.split("/")[0] for f in ours} <= {"kitten_inference", "kitten_inference_tts08.libs", info}
    # Every wheel carries the same licence files (one METADATA for all), at their paths in the repository.
    licences = set(dist.metadata.get_all("License-File") or [])
    assert {"LICENSE", "NOTICE", "tts-0.8/NOTICE", "tts-2/NOTICE"} <= licences
    assert {f"{info}/licenses/{name}" for name in licences} <= ours


def test_engine_versions_follow_the_distribution(dist):
    # A post release (0.2.0.postN) changes only the kitten-inference version; the engines keep the base version.
    if _build_input(dist):
        pytest.skip("the build-input wheel holds no engine")
    base = re.sub(r"\.post\d+$", "", dist.version)
    for sub in ("tts08", "tts2"):
        text = Path(dist.locate_file(f"kitten_inference/{sub}/__init__.py")).read_text(encoding="utf-8")
        assert re.search(r'^__version__ = "([^"]+)"', text, re.M).group(1) == base, sub
