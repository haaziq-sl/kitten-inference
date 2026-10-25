"""First-use preparation: download KittenML/kitten-tts-2 and convert it for the engine, once per model revision.

The prepared directory is ``<cache>/<model commit sha>/``. Preparation runs under a cross-process file lock, into
``<cache>/.partial-<sha>/``, which is renamed into place only when it is complete. After Ctrl-C (or any failure) the
next run starts the conversion again from the files already downloaded: the Hugging Face cache keeps finished
downloads and resumes unfinished ones, and the pinned decoder, grammar and data files are cached too.
"""
from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

MODEL_REPO = "KittenML/kitten-tts-2"
DEFAULT_CACHE = Path.home() / ".cache" / "kitten-inference" / "tts2"
_SHA = re.compile(r"[0-9a-f]{40}")
# The child process that runs prepare.main(argv) (kitten_inference.tts2._assets.prepare, the bundled copy of
# tts-2/tools/kt2n_assets): its output goes to a log file, and the memory the conversion uses is returned afterwards.
_CHILD = "import sys; from kitten_inference.tts2._assets.prepare import main; sys.exit(main(sys.argv[1:]))"


class PrepareError(RuntimeError):
    """The model could not be downloaded or prepared."""


def _log(message: str) -> None:
    print(f"kitten-tts2: {message}", file=sys.stderr, flush=True)


def _ascii(text: str) -> bool:
    return all(ord(c) < 128 for c in text)


def _narrow_paths_are_utf8() -> bool:
    """Whether the engine can open non-ASCII paths: always, except on Windows when the process code page is not
    UTF-8 (the engine and its text-data tool open files through the C runtime, which reads paths in that code page,
    and the engine refuses non-ASCII paths there). python.exe has no UTF-8 code page manifest, so this is the
    common Windows case."""
    if sys.platform != "win32":
        return True
    try:
        return ctypes.windll.kernel32.GetACP() == 65001
    except (AttributeError, OSError):  # pragma: no cover
        return False


def _short_path(path: Path) -> Optional[Path]:
    """The 8.3 short form of an existing Windows path (None when Windows has none for it)."""
    try:
        buf = ctypes.create_unicode_buffer(32768)
        n = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, len(buf))
    except (AttributeError, OSError):  # pragma: no cover
        return None
    return Path(buf.value) if 0 < n < len(buf) else None


def _fallback_root() -> Path:
    """An ASCII folder for this user's models when the user folder's path is not ASCII (Windows, see cache_root)."""
    base = Path(os.environ.get("PROGRAMDATA") or "C:\\ProgramData")
    user = hashlib.sha256(str(Path.home()).encode("utf-8")).hexdigest()[:12]
    return base / "kitten-inference" / f"tts2-{user}"


def cache_root(cache_dir: Optional[os.PathLike] = None, log: Optional[Callable[[str], None]] = None) -> Path:
    """Where prepared models are kept: ``cache_dir``, or ~/.cache/kitten-inference/tts2.

    On Windows with a non-UTF-8 code page (the usual case), the engine opens only ASCII paths. A path with other
    characters is then replaced by its 8.3 short form when that is ASCII; else the default folder moves to an ASCII
    folder under %PROGRAMDATA% (with a note), and a given cache_dir is refused at once, before any download."""
    root = Path(cache_dir).expanduser() if cache_dir is not None else DEFAULT_CACHE
    if _ascii(str(root)) or _narrow_paths_are_utf8():
        return root
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    short = _short_path(root)
    if short is not None and _ascii(str(short)):
        return short
    if cache_dir is None:
        fallback = _fallback_root()
        if _ascii(str(fallback)):
            try:
                if not fallback.is_dir():  # say it once, when the folder is made
                    fallback.mkdir(parents=True, exist_ok=True)
                    (log or _log)(f"the path of your user folder has characters the engine cannot open on this "
                                  f"Windows setup; models are kept in {fallback} instead (cache_dir= chooses another "
                                  "folder)")
                return fallback
            except OSError:
                pass
    raise PrepareError(f"the model folder {root} has characters outside ASCII, which the engine cannot open on Windows "
                       "with this system's code page. Use an ASCII folder: KittenTTS2(cache_dir=\"C:/kitten-tts2\") or "
                       "kitten-tts2 --cache-dir C:\\kitten-tts2 (or turn on \"Use Unicode UTF-8 for worldwide language "
                       "support\" in the Region settings)")


def _download_cache(cache_dir: Optional[os.PathLike]) -> Optional[Path]:
    """Where downloads go: the standard Hugging Face cache, or <cache_dir>/downloads when cache_dir is given."""
    return None if cache_dir is None else cache_root(cache_dir) / "downloads"


def _assets():
    """The bundled prepare module, for its constants (what to download). Importing it does not import torch."""
    from ._assets import prepare
    return prepare


def _ref_file(root: Path, repo: str, revision: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{repo}@{revision}")
    return root / "refs" / safe


def resolve_revision(repo: str, revision: Optional[str], root: Path,
                     log: Optional[Callable[[str], None]] = None) -> str:
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
    """Remembers which commit ``revision`` resolved to, for use without a network. Best effort: on Windows a replace
    fails while another process replaces the same file (two KittenTTS2() starting together), and losing that race
    only means the other process wrote the same note."""
    rev = revision or "main"
    if _SHA.fullmatch(rev):
        return
    ref = _ref_file(root, repo, rev)
    try:
        if ref.read_text(encoding="utf-8").strip() == sha:
            return
    except OSError:
        pass
    tmp = ref.with_name(f"{ref.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        ref.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(sha + "\n", encoding="utf-8")
        os.replace(tmp, ref)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink()


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


def _child_env(root: Path) -> dict:
    """The environment of the prepare child: this package on its path, and UTF-8 output. On Windows before Python
    3.15 output sent to a file uses the ANSI code page, so printing a path outside it (C:\\Users\\Łukasz under
    cp1252) would stop the child after the conversion. Where the engine opens only ASCII paths, the child's temporary
    folder is also under the (ASCII) model folder, because the text-data tool reads its input from there."""
    import kitten_inference.tts2 as package
    site = str(Path(package.__file__).resolve().parents[2])  # the folder that holds kitten_inference/tts2
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([site, *filter(None, [os.environ.get("PYTHONPATH")])]),
               PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    if not _narrow_paths_are_utf8():
        temp = env.get("TEMP") or env.get("TMP") or ""
        if not _ascii(temp) or not temp:
            tmp = root / "tmp"
            tmp.mkdir(parents=True, exist_ok=True)
            env["TEMP"] = env["TMP"] = str(tmp)
    return env


def _run_prepare(argv: List[str], log_path: Path) -> None:
    """Runs prepare.main(argv) in a child Python process; its output goes to log_path (the tail is shown on failure).
    Ctrl-C stops the child too (same console)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as log:
        code = subprocess.run([sys.executable, "-c", _CHILD, *argv], stdin=subprocess.DEVNULL, stdout=log,
                              stderr=subprocess.STDOUT, env=_child_env(log_path.parent)).returncode
    if code != 0:
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        except OSError:
            lines = []
        tail = "; ".join(line.strip() for line in lines[-3:]) or f"exit status {code}"
        raise PrepareError(f"preparing the model failed: {tail} (full log: {log_path})")


def _snapshot(repo: str, sha: str, patterns: List[str], cache_dir: Optional[os.PathLike]) -> Path:
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(repo, revision=sha, allow_patterns=patterns, cache_dir=_download_cache(cache_dir),
                                      token=False))
    except Exception as exc:
        raise PrepareError(f"downloading {repo} at {sha[:12]} failed: {exc}") from exc


def _acoustic(prepare, cache_dir: Optional[os.PathLike]) -> Path:
    """The pinned decoder weights, downloaded here so that every large download shows its progress in stage 1
    (prepare checks their SHA-256 again)."""
    from huggingface_hub import hf_hub_download
    repo, name, revision = prepare.ACOUSTIC_REPO, prepare.ACOUSTIC_FILE, prepare.convert_s3gen.PINNED_REVISION
    try:
        return Path(hf_hub_download(repo, name, revision=revision, cache_dir=_download_cache(cache_dir), token=False))
    except Exception as exc:
        raise PrepareError(f"downloading {repo}/{name} failed: {exc}") from exc


def ensure(*, text_data_tool: Path, revision: Optional[str] = None, cache_dir: Optional[os.PathLike] = None,
           with_cloning: bool = False, repo: str = MODEL_REPO, log: Callable[[str], None] = _log,
           before_download: Optional[Callable[[], None]] = None) -> Path:
    """The prepared model directory for ``revision`` (default: the latest, main), preparing it first if needed.
    ``before_download`` runs once before anything is downloaded (the GPU check: fail before minutes of work)."""
    root = cache_root(cache_dir, log)
    sha = resolve_revision(repo, revision, root, log)
    final = root / sha
    if is_prepared(final, with_cloning):
        _record_ref(root, repo, revision, sha)
        return final
    if before_download is not None:
        before_download()
    root.mkdir(parents=True, exist_ok=True)
    with _locked(root / f".{sha}.lock"):
        if not is_prepared(final, with_cloning):
            prepare = _assets()
            common = ["--repo", repo, "--revision", sha, "--weight-norm-device", "numpy"]  # numpy: no torch
            if cache_dir is not None:
                common += ["--cache-dir", str(_download_cache(cache_dir))]
            started = time.monotonic()
            if not is_prepared(final):
                patterns = list(prepare.MODEL_ALLOW_PATTERNS) + (list(prepare.CLONING_ALLOW_PATTERNS)
                                                                 if with_cloning else [])
                log(f"[1/3] downloading {repo} at {sha[:12]} and the decoder weights (first use only)")
                snapshot = _snapshot(repo, sha, patterns, cache_dir)
                common += ["--acoustic-source", str(_acoustic(prepare, cache_dir))]
                downloaded = time.monotonic()
                partial = root / f".partial-{sha}"
                shutil.rmtree(partial, ignore_errors=True)
                log(f"[2/3] preparing the model for this machine{' with voice cloning' if with_cloning else ''} "
                    "(a few minutes)")
                argv = ["--model-dir", str(snapshot), "--out", str(partial), "--text-data-tool", str(text_data_tool),
                        *common]
                if with_cloning:
                    argv.append("--with-cloning")
                _run_prepare(argv, root / f"prepare-{sha[:12]}.log")
                if not is_prepared(partial, with_cloning):
                    raise PrepareError(f"preparation finished without a complete model in {partial}")
                if final.exists():  # an unprepared leftover (no config.json): replace it
                    shutil.rmtree(final)
                os.replace(partial, final)
            else:
                log(f"[1/3] downloading the voice cloning files of {repo} at {sha[:12]} (first use only)")
                snapshot = _snapshot(repo, sha, list(prepare.MODEL_ALLOW_PATTERNS)
                                     + list(prepare.CLONING_ALLOW_PATTERNS), cache_dir)
                common += ["--acoustic-source", str(_acoustic(prepare, cache_dir))]
                downloaded = time.monotonic()
                # config.json is the commit point: without its cloning entry, files in native/cloning are a
                # stopped earlier attempt.
                shutil.rmtree(final / "native" / "cloning", ignore_errors=True)
                log("[2/3] preparing the voice cloning files")
                argv = ["--model-dir", str(snapshot), "--out", str(final), "--cloning-only", *common]
                _run_prepare(argv, root / f"prepare-cloning-{sha[:12]}.log")
                if not is_prepared(final, True):
                    raise PrepareError(f"preparation finished without the cloning files in {final}")
            done = time.monotonic()
            log(f"[3/3] model ready in {done - started:.0f} s (downloads {downloaded - started:.0f} s, preparation "
                f"{done - downloaded:.0f} s): {final}")
    _record_ref(root, repo, revision, sha)
    return final
