# Releasing

A release is one PyPI project, `kitten-inference`: 54 wheels, about 0.64 GB. One job uploads them, `publish` in
`.github/workflows/packages.yml`, when a `v*` tag is pushed to KittenML/kitten-inference. Nothing else is published:
no GitHub release, no sdist, no other project.

The release builds no engine. A rebuild is not byte for byte the same (the Windows CUDA build is not reproducible), so
the wheels are assembled from the tested builds instead:

- `tts-0.8.yml`, `tts-2-engine.yml` (CUDA) and `tts-2-rocm.yml` (ROCm) build the engines on every push that changes
  their sources, or when run by hand, and keep them as GitHub Actions artifacts for 90 days.
- `.github/release-inputs.json` (the lock) names the runs a release uses and the sha256 of every file in them.
- `packages.yml` downloads exactly those artifacts, checks every file against the lock, and assembles each wheel: a
  TTS 0.8 wheel, the TTS 2 Python API and `kitten_inference/__init__.py` of the commit, and on Linux x86-64 and
  Windows x64 the TTS 2 engines. A second script checks the result independently: the engine files are the tested
  ones, byte for byte. Then the wheels are tested on Linux, Windows and macOS.

In a repository other than the one the lock names (a fork, or KittenML before its own lock), `packages.yml` skips the
assembly with a notice, and a tag run there fails. In the lock's repository, a push or pull request also skips it,
with a warning, when the lock's artifacts have expired or the commit's engine sources, TTS 0.8 sources or versions
differ from those of the locked builds (an engine change, or a new version before its lock); a dry run or a tag run
fails instead. Whenever the assembly is skipped, the umbrella and TTS 2 Python unit tests run without it.

## Once

- PyPI, project `kitten-inference`, Manage, Publishing: one trusted publisher (GitHub Actions) with owner `KittenML`,
  repository `kitten-inference`, workflow `packages.yml` and environment `pypi`. Remove any other publisher, and any
  pending publishers made earlier for other project names.
- GitHub, Settings, Environments: create `pypi` and protect it. Without that, anyone who can push a `v*` tag chooses
  the code that runs with upload rights.
  - Required reviewers: the maintainers who release. The upload then waits for one of them to approve it.
  - Deployment branches and tags: "Selected branches and tags", with one rule for the tag pattern `v*`.
  - A tag ruleset (Settings, Rules, Rulesets; target: tags matching `v*`) that lets only those maintainers create,
    update or delete such tags.
- PyPI's default project limit of 10 GB holds about 15 releases: ask PyPI for a larger limit. Every file is under the
  100 MB file limit (the largest, Linux x86-64, is about 53 MB).

## A new version

1. **Versions.** Write the new version in every place, all the same:
   - `umbrella/pyproject.toml` and `__version__` in `umbrella/kitten_inference/__init__.py`;
   - `tts-2/python/pyproject.toml` and `__version__` in `tts-2/python/kitten_inference/tts2/__init__.py`;
   - `tts-0.8/pyproject.toml` and `__version__` in `tts-0.8/kitten_inference/tts08/__init__.py`;
   - the engine version in `tts-2/CMakeLists.txt` (`project(kt2n VERSION ...)`).
2. **Build.** Merge to `main`. CI runs `tts-0.8.yml`, `tts-2-engine.yml` and `tts-2-rocm.yml` on that commit. All three
   must pass; the CUDA jobs print `SASS check: IDENTICAL`. Note the three run ids. Until the new lock is committed
   (step 3), `packages.yml` skips the assembly with a warning: the versions and engine sources no longer match the
   lock.
3. **Lock.** Download the three runs' artifacts into one folder and record them:

   ```
   gh run download <tts-0.8 run> --repo KittenML/kitten-inference --dir inputs
   gh run download <cuda run> --repo KittenML/kitten-inference --dir inputs
   gh run download <rocm run> --repo KittenML/kitten-inference --dir inputs
   python .github/scripts/lock_inputs.py --repo KittenML/kitten-inference --tts08-run <tts-0.8 run> --cuda-run <cuda run> --rocm-run <rocm run> --inputs inputs --out .github/release-inputs.json
   ```

   Commit the lock to `main`. It changes no file that goes into a wheel.
4. **Dry run.** Run `packages.yml` by hand on that commit (Actions, packages, Run workflow). Every job must pass, and
   the summary shows the sha256 of all 54 wheels.
5. **GPU tests**, on the dry run's wheels (they carry the engine files of step 2, byte for byte), in a checkout of
   the same commit, in a fresh virtual environment with Python 3.9 or newer on each machine: an NVIDIA GPU on Linux,
   an AMD Radeon RX 7000 or RX 9000 with ROCm 7 on Linux, and an NVIDIA GPU on Windows. pip picks the wheel for that
   Python from the downloaded ones. On Linux:

   ```
   gh run download <dry run> --repo KittenML/kitten-inference -n assembled-linux-x86_64 -D wheels
   python3 -m venv venv
   . venv/bin/activate
   python -m pip install -U pip
   python -m pip install --no-index --no-deps --find-links wheels kitten-inference
   python -m pip install "kitten-inference[tts-2]" pytest
   KT2N_TEST_CLONING=1 python -m pytest tts-2/python/tests/gpu
   repo=$PWD
   cd /tmp && python "$repo/.github/scripts/engine_smoke.py" --lock "$repo/.github/release-inputs.json"
   ```

   On the AMD machine, leave out `KT2N_TEST_CLONING=1` (the AMD engine has no voice cloning) and add `--backend rocm`
   to `engine_smoke.py`. On Windows, in PowerShell:

   ```
   gh run download <dry run> --repo KittenML/kitten-inference -n assembled-windows-x86_64 -D wheels
   python -m venv venv
   $env:Path = "$PWD\venv\Scripts;$env:Path"
   python -m pip install -U pip
   python -m pip install --no-index --no-deps --find-links wheels kitten-inference
   python -m pip install "kitten-inference[tts-2]" pytest
   $env:KT2N_TEST_CLONING = "1"
   python -m pytest tts-2/python/tests/gpu
   $repo = $PWD.Path
   Set-Location $env:TEMP
   python "$repo\.github\scripts\engine_smoke.py" --lock "$repo\.github\release-inputs.json"
   ```

6. **Record the tested files.** Write the tested commit's packaged files into the lock, and commit it to `main`:

   ```
   python .github/scripts/lock_inputs.py --set-tested-trees <dry-run commit> --out .github/release-inputs.json
   ```

   A tag run assembles only from those files: any later change to them stops it.
7. **Tag** the commit of step 6:

   ```
   git tag -a v0.3.0 -m "kitten-inference 0.3.0"
   git push origin v0.3.0
   ```

8. **Approve.** The tag run assembles and tests the wheels again. Compare its sha256 list with the dry run's:

   ```
   gh run download <dry run> --repo KittenML/kitten-inference -n assembly-report -D dry
   gh run download <tag run> --repo KittenML/kitten-inference -n assembly-report -D tag
   diff dry/SHA256SUMS tag/SHA256SUMS
   ```

   Approve the `pypi` environment once every job is green and `diff` prints nothing. If a line differs, find out why
   first (`assembly-report.json` lists where every file of every wheel came from).
9. **Check PyPI.** The publish job uploads the 54 wheels, then checks that PyPI has every one, byte for byte, published
   by `packages.yml`. If it stops partway, re-run the job: it skips the files already there.
10. **First release only (0.2.0):** yank `kitten-inference` 0.1.0 and 0.1.1 on PyPI. Pinned installs of them keep
    working; without the yank, pip installs 0.1.1 with only a warning wherever no 0.2 wheel matches.

## A post release

A post release, `<version>.postN`, ships the same engine builds (the same lock) with changes only to the files a
release takes from its own commit:

- `umbrella/`, `tts-2/python/` and `tts-2/tools/kt2n_assets/`;
- the four TTS 0.8 Python files in the lock's `tts08_python_overlay`: `_deps.py`, `get_model.py`, `cpp_model.py` and
  `_espeak_path.py` in `tts-0.8/kitten_inference/tts08/`;
- `NOTICE`, `LICENSE`, `tts-0.8/NOTICE`, `tts-0.8/licenses/`, `tts-2/NOTICE` and `tts-2/licenses/`.

Any other change to `tts-0.8/kitten_inference/tts08/` or `tts-0.8/engine/`, or to the TTS 2 engine sources
(`tts-2/src/`, `tts-2/include/`, `tts-2/cli/`, `tts-2/cmake/`, `tts-2/CMakeLists.txt`, `tts-2/CMakePresets.json`),
needs "A new version": the assembly refuses sources other than those the locked builds were made from.

1. Set `<version>.postN` in `umbrella/pyproject.toml` and `umbrella/kitten_inference/__init__.py` only. Every other
   version stays `<version>`, and the lock stays.
2. Do steps 4 to 9: dry run, GPU tests, record the tested files, tag `v<version>.postN`, approve, check PyPI. All 54
   wheels are released again.

If the lock's artifacts have expired (90 days after their runs), the engines must be built and tested again: run
`tts-0.8.yml`, `tts-2-engine.yml` and `tts-2-rocm.yml` by hand on `main` (Actions, the workflow, Run workflow), note
the three run ids, and continue from step 3 of "A new version", keeping the versions of step 1 above.
