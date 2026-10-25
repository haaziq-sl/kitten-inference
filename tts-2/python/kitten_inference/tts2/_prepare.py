"""First-use preparation: download KittenML/kitten-tts-2 and convert it for the engine, once per model revision.

The prepared directory is ``<cache>/<model commit sha>/``. Preparation runs under a cross-process file lock, into
``<cache>/.partial-<sha>/``, which is renamed into place only when it is complete. After Ctrl-C (or any failure) the
next run starts the conversion again from the files already downloaded: the Hugging Face cache keeps finished
downloads and resumes unfinished ones, and the pinned decoder, grammar and data files are cached too.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

MODEL_REPO = "KittenML/kitten-tts-2"
DEFAULT_CACHE = Path.home() / ".cache" / "kitten-inference" / "tts2"
# What prepare reads from the model repository (the same list prepare.py downloads itself without --model-dir).
MODEL_FILES = ["config.json", "lm/config.json", "lm/model.safetensors", "lm/tokenizer.json", "cpp/default/voices.json",
               "voices/voices.json", "voices/*.npz"]
# Voice cloning also reads the speaker-embedding network from the same snapshot.
CLONING_FILES = ["speaker/*"]
_SHA = re.compile(r"[0-9a-f]{40}")


class PrepareError(RuntimeError):
    """The model could not be downloaded or prepared."""


def _log(message: str) -> None:
    print(f"kitten-tts2: {message}", file=sys.stderr, flush=True)


def cache_root(cache_dir: Optional[os.PathLike] = None) -> Path:
    return Path(cache_dir).expanduser() if cache_dir is not None else DEFAULT_CACHE


def _download_cache(cache_dir: Optional[os.PathLike]) -> Optional[Path]:
    """Where downloads go: the standard Hugging Face cache, or <cache_dir>/downloads when cache_dir is given."""
    return None if cache_dir is None else cache_root(cache_dir) / "downloads"


def _assets():
    from ._assets import prepare  # the copy of tts-2/tools/kt2n_assets made at build time
    return prepare


def _model_files(prepare) -> List[str]:
    return list(getattr(prepare, "MODEL_ALLOW_PATTERNS", MODEL_FILES))


def _cloning_files(prepare) -> List[str]:
    return list(getattr(prepare, "CLONING_ALLOW_PATTERNS", CLONING_FILES))


def _prepare_supports(prepare, flag: str) -> bool:
    try:
        source = Path(prepare.__file__).read_text(encoding="utf-8")
    except OSError:
        return False
    return f'"{flag}"' in source


def _ref_file(root: Path, repo: str, revision: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{repo}@{revision}")
    return root / "refs" / safe


def resolve_revision(repo: str, revision: Optional[str], root: Path, log: Callable[[str], None] = None) -> str:
    """The commit sha of ``revision`` (default main). A 40-hex sha needs no network. Otherwise Hugging Face is asked;
    without a connection the sha recorded by the last successful preparation of that revision is used."""
    rev = revision or "main"
    if _SHA.fullmatch(rev):
        return rev
    ref = _ref_file(root, repo, rev)
    try:
        from huggingface_hub import HfApi
        sha = HfApi().model_info(repo, revision=rev, token=False).sha
        if not sha or not _SHA.fullmatch(sha):
            raise PrepareError(f"Hugging Face returned no commit for {repo}@{rev}")
        return sha
    except PrepareError:
        raise
    except Exception as exc:  # no network, offline mode, rate limit, unknown revision
        if ref.is_file():
            sha = ref.read_text(encoding="utf-8").strip()
            if _SHA.fullmatch(sha) and (root / sha / "config.json").is_file():
                (log or _log)(f"cannot reach Hugging Face ({type(exc).__name__}); using the prepared {repo}@{rev} "
                     f"from {root / sha}")
                return sha
        raise PrepareError(f"cannot find {repo} revision {rev!r} on Hugging Face: {exc}") from exc


def _record_ref(root: Path, repo: str, revision: Optional[str], sha: str) -> None:
    rev = revision or "main"
    if _SHA.fullmatch(rev):
        return
    ref = _ref_file(root, repo, rev)
    ref.parent.mkdir(parents=True, exist_ok=True)
    tmp = ref.with_name(f"{ref.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(sha + "\n", encoding="utf-8")
    os.replace(tmp, ref)


def _config(path: Path) -> Optional[dict]:
    try:
        return json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_prepared(path: Path, with_cloning: bool = False) -> bool:
    config = _config(path)
    native = config.get("native") if isinstance(config, dict) else None
    if not isinstance(native, dict):
        return False
    return not with_cloning or isinstance(native.get("cloning"), dict)


@contextlib.contextmanager
def _locked(path: Path):
    from filelock import FileLock, Timeout
    lock = FileLock(str(path))
    try:
        lock.acquire(timeout=1)
    except Timeout:
        _log(f"waiting for another process that is preparing the model (lock {path})")
        lock.acquire()
    try:
        yield
    finally:
        lock.release()


def _run_prepare(prepare, argv: List[str], log_path: Path) -> None:
    """prepare.main(argv) in this process; its own printout goes to log_path (and is shown on failure)."""
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = prepare.main(argv)
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, str):
            out.write(code + "\n")
            code = 1
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        _write(log_path, out.getvalue())
        raise PrepareError(f"preparing the model failed: {exc} (log: {log_path})") from exc
    _write(log_path, out.getvalue())
    if code not in (0, None):
        tail = "\n".join(out.getvalue().strip().splitlines()[-5:])
        raise PrepareError(f"preparing the model failed (exit {code}): {tail} (log: {log_path})")


def _write(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError:
        pass


def _snapshot(repo: str, sha: str, patterns: List[str], cache_dir: Optional[os.PathLike]) -> Path:
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(repo, revision=sha, allow_patterns=patterns, cache_dir=_download_cache(cache_dir),
                                      token=False))
    except Exception as exc:
        raise PrepareError(f"downloading {repo} at {sha[:12]} failed: {exc}") from exc


def _acoustic(prepare, cache_dir: Optional[os.PathLike]) -> Optional[Path]:
    """The pinned decoder weights (ResembleAI/chatterbox-turbo), downloaded here so that every large download shows
    its progress in stage 1; prepare checks their SHA-256. None when this prepare does not name them."""
    repo, name = getattr(prepare, "ACOUSTIC_REPO", None), getattr(prepare, "ACOUSTIC_FILE", None)
    revision = getattr(getattr(prepare, "convert_s3gen", None), "PINNED_REVISION", None)
    if not (repo and name and revision):
        return None
    from huggingface_hub import hf_hub_download
    try:
        return Path(hf_hub_download(repo, name, revision=revision, cache_dir=_download_cache(cache_dir), token=False))
    except Exception as exc:
        raise PrepareError(f"downloading {repo}/{name} failed: {exc}") from exc


def ensure(*, text_data_tool: Path, revision: Optional[str] = None, cache_dir: Optional[os.PathLike] = None,
           with_cloning: bool = False, repo: str = MODEL_REPO, log: Callable[[str], None] = _log) -> Path:
    """The prepared model directory for ``revision`` (default: the latest, main), preparing it first if needed."""
    root = cache_root(cache_dir)
    sha = resolve_revision(repo, revision, root, log)
    final = root / sha
    if is_prepared(final, with_cloning):
        _record_ref(root, repo, revision, sha)
        return final
    root.mkdir(parents=True, exist_ok=True)
    with _locked(root / f".{sha}.lock"):
        if not is_prepared(final, with_cloning):
            prepare = _assets()
            tool = Path(text_data_tool)
            tool_flag = "--text-data-tool" if _prepare_supports(prepare, "--text-data-tool") else "--text-converter"
            common = ["--repo", repo, "--revision", sha]
            if cache_dir is not None:
                common += ["--cache-dir", str(_download_cache(cache_dir))]
            if _prepare_supports(prepare, "numpy"):
                common += ["--weight-norm-device", "numpy"]
            started = time.monotonic()
            if not is_prepared(final):
                patterns = _model_files(prepare) + (_cloning_files(prepare) if with_cloning else [])
                log(f"[1/3] downloading {repo} at {sha[:12]} and the decoder weights (first use only)")
                snapshot = _snapshot(repo, sha, patterns, cache_dir)
                acoustic = _acoustic(prepare, cache_dir)
                if acoustic is not None:
                    common += ["--acoustic-source", str(acoustic)]
                partial = root / f".partial-{sha}"
                shutil.rmtree(partial, ignore_errors=True)
                log(f"[2/3] preparing the model for this machine{' with voice cloning' if with_cloning else ''} "
                    "(about a minute)")
                argv = ["--model-dir", str(snapshot), "--out", str(partial), tool_flag, str(tool), *common]
                if with_cloning:
                    argv.append("--with-cloning")
                _run_prepare(prepare, argv, root / f"prepare-{sha[:12]}.log")
                if not is_prepared(partial, with_cloning):
                    raise PrepareError(f"preparation finished without a complete model in {partial}")
                if final.exists():  # an unprepared leftover (no config.json): replace it
                    shutil.rmtree(final)
                os.replace(partial, final)
            else:
                log(f"[1/3] downloading the voice cloning files of {repo} at {sha[:12]} (first use only)")
                snapshot = _snapshot(repo, sha, _model_files(prepare) + _cloning_files(prepare), cache_dir)
                acoustic = _acoustic(prepare, cache_dir)
                if acoustic is not None:
                    common += ["--acoustic-source", str(acoustic)]
                # config.json is the commit point: without its cloning entry, files in native/cloning are a
                # stopped earlier attempt.
                shutil.rmtree(final / "native" / "cloning", ignore_errors=True)
                log("[2/3] preparing the voice cloning files")
                argv = ["--model-dir", str(snapshot), "--out", str(final), "--cloning-only", *common]
                _run_prepare(prepare, argv, root / f"prepare-cloning-{sha[:12]}.log")
                if not is_prepared(final, True):
                    raise PrepareError(f"preparation finished without the cloning files in {final}")
            log(f"[3/3] model ready in {time.monotonic() - started:.0f} s: {final}")
    _record_ref(root, repo, revision, sha)
    return final
