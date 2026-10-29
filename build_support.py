"""Build-only inputs for release assembly, derived from the root project."""
import os
import shutil
import tempfile
from pathlib import Path
try:
    import tomllib
except ImportError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent


def hooks(part):
    def build(kind, output, config_settings=None, metadata_directory=None):
        import tomli_w
        output = str(Path(output).resolve())
        config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        with tempfile.TemporaryDirectory(prefix='kitten-build-') as folder:
            stage = Path(folder)
            project = config['project']
            project['license-files'] = ['LICENSE']
            for name in ('LICENSE', 'README.md'):
                shutil.copy2(ROOT / name, stage / name)
            target = stage / 'kitten_inference'
            target.mkdir()
            if part == 'metadata':
                for p in (ROOT / 'src/kitten_inference').glob('*.py'):
                    shutil.copy2(p, target / p.name)
            else:
                project['name'] += '-' + part
                project.pop('scripts', None)
                shutil.copytree(ROOT / 'src/kitten_inference' / part, target / part)
            if part == 'tts08':
                shutil.copy2(ROOT / 'engines/tts08/NOTICE', stage / 'NOTICE')
                shutil.copytree(ROOT / 'engines/tts08/licenses', stage / 'licenses')
                project['license-files'] = ['LICENSE', 'NOTICE', 'licenses/**']
                shutil.copytree(ROOT / 'engines/tts08/engine', stage / 'engine')
                config['tool'] = {'scikit-build': dict(config['tool']['scikit-build'])}
                opts = config['tool']['scikit-build']
                opts['cmake']['source-dir'] = 'engine'
                opts['wheel']['packages'] = {'kitten_inference/tts08': 'kitten_inference/tts08'}
                from scikit_build_core import build as backend
            else:
                if part == 'tts2':
                    shutil.copytree(ROOT / 'engines/tts2/tools/kt2n_assets', target / 'tts2/_assets')
                    shutil.copytree(ROOT / 'engines/tts2/licenses', target / 'tts2/licenses')
                    shutil.copy2(ROOT / 'engines/tts2/NOTICE', target / 'tts2/NOTICE')
                config['build-system'] = {'requires': ['hatchling==1.27.0'], 'build-backend': 'hatchling.build'}
                config['tool'] = {'hatch': {'build': {'targets': {'wheel': {'packages': ['kitten_inference'], 'core-metadata-version': '2.4'}, 'sdist': {'include': ['kitten_inference', 'LICENSE', 'README.md', 'pyproject.toml']}}}}}
                from hatchling import build as backend
            (stage / 'pyproject.toml').write_text(tomli_w.dumps(config))
            previous = Path.cwd()
            try:
                os.chdir(stage)
                return getattr(backend, kind)(output, config_settings)
            finally:
                os.chdir(previous)
    return (lambda wheel_directory, config_settings=None, metadata_directory=None: build('build_wheel', wheel_directory, config_settings),
            lambda sdist_directory, config_settings=None: build('build_sdist', sdist_directory, config_settings))
