"""Copies the model preparation code into the package at build time (one source of truth).

tts-2/tools/kt2n_assets -> kitten_inference/tts2/_assets, and tts-2/NOTICE + tts-2/licenses -> kitten_inference/tts2/.
A wheel built from the sdist takes them from the sdist, where this hook already placed them.
"""
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

SOURCES = {
    "../tools/kt2n_assets": "kitten_inference/tts2/_assets",
    "../NOTICE": "kitten_inference/tts2/NOTICE",
    "../licenses": "kitten_inference/tts2/licenses",
}


def _files(source: Path, target: str):
    if source.is_file():
        yield source, target
        return
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        if path.is_file() and "__pycache__" not in rel.parts and path.suffix not in (".pyc", ".pyo"):
            yield path, f"{target}/{rel.as_posix()}"


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        root = Path(self.root)
        for source, target in SOURCES.items():
            src = (root / source).resolve()
            if src.exists():
                for path, dest in _files(src, target):
                    build_data["force_include"][str(path)] = dest
            elif not (root / target).exists():
                raise FileNotFoundError(f"{src} is missing (and {target} is not in this source tree): build from a "
                                        "checkout of the repository or from the sdist")
