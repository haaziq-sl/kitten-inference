# Releasing

One tag, `v<version>`, pushed to KittenML/kitten-inference, releases all five distributions. Four workflows build and
test them again on that tag, then upload them with PyPI trusted publishing (no API token). Forks never upload: every
publish job runs only in KittenML/kitten-inference.

| PyPI project | Workflow | Job | Waits for |
|---|---|---|---|
| `kitten-inference-tts2-cu12` | `tts-2-engine.yml` | `publish` | its Linux and Windows wheel jobs |
| `kitten-inference-tts2-rocm` | `tts-2-rocm.yml` | `publish` | its wheel and install jobs |
| `kitten-inference-tts08` | `tts-0.8.yml` | `publish` | its wheel jobs, then the two engine wheels above on PyPI |
| `kitten-inference-tts2` | `packages.yml` | `publish-tts2` | its tests, then the two engine wheels above on PyPI |
| `kitten-inference` | `packages.yml` | `publish-umbrella` | `publish-tts2`, then every distribution its extras name on PyPI |

So PyPI gets the engine wheels first, then `kitten-inference-tts2` and `kitten-inference-tts08`, then the umbrella. An
umbrella published first would point its extras at nothing. Each wait gives up after 5 hours; once the missing
distributions are on PyPI, re-run the failed job.

## Once: trusted publishers and the `pypi` environment

On PyPI, add one trusted publisher (GitHub Actions) per project, all with owner `KittenML`, repository
`kitten-inference` and environment `pypi`:

| PyPI project | Workflow name |
|---|---|
| `kitten-inference` | `packages.yml` |
| `kitten-inference-tts2` | `packages.yml` |
| `kitten-inference-tts08` | `tts-0.8.yml` |
| `kitten-inference-tts2-cu12` | `tts-2-engine.yml` |
| `kitten-inference-tts2-rocm` | `tts-2-rocm.yml` |

`kitten-inference` already exists on PyPI (0.1.x): add the publisher in its settings (Manage, Publishing). The other
four do not exist yet: add each as a pending publisher (Your account, Publishing, "Add a new pending publisher"),
which creates the project at its first upload.

On GitHub, create the environment `pypi` (Settings, Environments). Adding required reviewers to it makes every upload
wait for a manual approval.

No file-size request is needed: the largest wheels (`kitten-inference-tts2-cu12`) are about 31 MB, under PyPI's
100 MB limit.

## Release

1. Tag the commit whose CI was tested. Every distribution's version must be the release's version, in each place it
   is written:
   - `kitten-inference`: `umbrella/pyproject.toml` and `__version__` in `umbrella/kitten_inference/__init__.py`;
   - `kitten-inference-tts2`: `tts-2/python/pyproject.toml` and `__version__` in
     `tts-2/python/kitten_inference/tts2/__init__.py`;
   - `kitten-inference-tts08`: `tts-0.8/pyproject.toml` and `__version__` in `tts-0.8/kitten_inference/tts08/__init__.py`;
   - the engine wheels: the engine version in `tts-2/CMakeLists.txt` (`project(kt2n VERSION ...)`), which both take
     unless `CU12_VERSION` in `tts-2-engine.yml` or `ROCM_VERSION` in `tts-2-rocm.yml` sets one.

   The unit tests fail when a `__version__` and its pyproject differ, and the engine wheel builds refuse a version
   that is not the engine's (or a post release of it).
2. `git tag -a v0.2.0 -m "kitten-inference 0.2.0"` on that commit, then `git push origin v0.2.0`.
3. Watch the four runs. The CUDA wheel jobs fail unless their GPU code is the reference's; their logs say
   `SASS check: IDENTICAL`. The Windows CUDA wheel takes about 40 minutes (36 and 38 in two runs; at most about 2.5
   hours): its build compiles the CUDA-core decode kernels again until their GPU code is the reference's.
4. After the release, yank `kitten-inference` 0.1.0 and 0.1.1 on PyPI (pinned installs of them keep working). Without
   the yank, pip installs 0.1.1 with only a warning wherever an extra's engine has no wheel. Then remove the lines
   about 0.1.1 from `umbrella/README.md` and `tts-2/python/README.md`.

## Post releases

A fix to one distribution goes out as `<version>.postN` of that distribution only. Change its version in the places
step 1 lists for it (for an engine wheel: set `CU12_VERSION` or `ROCM_VERSION` to `<version>.postN`; the engine version
in `tts-2/CMakeLists.txt` stays), then tag `v<version>.postN`. A publish job uploads only when its
distribution's version equals the tag's, so the other distributions are left out. Their extras ranges
(`>=0.2.0,<0.3`) take the post release.
