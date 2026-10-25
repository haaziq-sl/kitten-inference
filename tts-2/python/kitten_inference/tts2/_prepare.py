"""First-use preparation: download KittenML/kitten-tts-2 and convert it for the engine, once per model revision.

The prepared directory is ``<cache>/<model commit sha>/``. Preparation runs under a cross-process file lock, into
``<cache>/.partial-<sha>/``, which is renamed into place only when it is complete. After Ctrl-C (or any failure) the
next run starts the conversion again from the files already downloaded: the Hugging Face cache keeps finished
downloads and resumes unfinished ones, and the pinned decoder, grammar and data files are cached too.
"""
from __future__ import annotations

import contextlib
import ctypes
import errno
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
# `python -c` puts the current folder first on sys.path (only -P, Python 3.11+, turns that off), so a json.py or a
# kitten_inference/ folder where the user runs kitten-tts2 would be imported instead of the real modules: the child
# drops that entry before its first import (os and sys are already loaded at startup).
_CHILD = ("import os, sys; sys.path[:] = [p for i, p in enumerate(sys.path) if i or p not in ('', '.', os.getcwd())]; "
          "from kitten_inference.tts2._assets.prepare import main; sys.exit(main(sys.argv[1:]))")
# Windows: a folder only this user (and SYSTEM and Administrators) can open, inherited by everything inside it. The
# same security descriptor as os.mkdir(path, 0o700) on Python 3.13.
_PRIVATE_SDDL = "D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FA;;;OW)"


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


def _private_mkdir(path: Path) -> None:
    """Creates the folder ``path`` (its parent exists) so that only this user can read or change it: on Windows with a
    protected access list for SYSTEM, Administrators and the folder's owner, which everything inside inherits;
    elsewhere with mode 0o700. FileExistsError when something is already there."""
    if sys.platform != "win32":
        os.mkdir(path, 0o700)
        return
    from ctypes import wintypes
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class SecurityAttributes(ctypes.Structure):
        _fields_ = [("nLength", wintypes.DWORD), ("lpSecurityDescriptor", ctypes.c_void_p),
                    ("bInheritHandle", wintypes.BOOL)]

    convert = advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    convert.restype = wintypes.BOOL
    create = kernel32.CreateDirectoryW
    create.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(SecurityAttributes)]
    create.restype = wintypes.BOOL
    free = kernel32.LocalFree
    free.argtypes = [ctypes.c_void_p]
    free.restype = ctypes.c_void_p
    sd = ctypes.c_void_p()
    if not convert(_PRIVATE_SDDL, 1, ctypes.byref(sd), None):  # SDDL_REVISION_1
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        sa = SecurityAttributes(ctypes.sizeof(SecurityAttributes), sd, False)
        if not create(str(path), ctypes.byref(sa)):
            code = ctypes.get_last_error()
            if code == 183:  # ERROR_ALREADY_EXISTS
                raise FileExistsError(errno.EEXIST, "the folder already exists", str(path))
            raise ctypes.WinError(code)
    finally:
        free(sd)


def _owned_by_me(path: Path) -> bool:
    """Whether this user owns ``path`` (on Windows: its owner is this process's user, or the owner the process gives
    new files, which is Administrators in an elevated window). OSError when it cannot be read."""
    if sys.platform != "win32":
        return path.stat().st_uid == os.getuid()
    from ctypes import wintypes
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    free = kernel32.LocalFree
    free.argtypes = [ctypes.c_void_p]
    free.restype = ctypes.c_void_p
    to_text = advapi32.ConvertSidToStringSidW
    to_text.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    to_text.restype = wintypes.BOOL

    def sid_text(psid) -> str:
        out = ctypes.c_void_p()
        if not to_text(psid, ctypes.byref(out)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return ctypes.wstring_at(out.value)
        finally:
            free(out)

    get = advapi32.GetNamedSecurityInfoW
    get.argtypes = [wintypes.LPCWSTR, ctypes.c_int, wintypes.DWORD] + [ctypes.POINTER(ctypes.c_void_p)] * 5
    get.restype = wintypes.DWORD
    owner, sd = ctypes.c_void_p(), ctypes.c_void_p()
    code = get(str(path), 1, 1, ctypes.byref(owner), None, None, None, ctypes.byref(sd))  # SE_FILE_OBJECT, OWNER_...
    if code != 0:
        raise ctypes.WinError(code)
    try:
        owner_sid = sid_text(owner)
    finally:
        free(sd)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    open_token = advapi32.OpenProcessToken
    open_token.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    open_token.restype = wintypes.BOOL
    info = advapi32.GetTokenInformation
    info.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    info.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    token = wintypes.HANDLE()
    if not open_token(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)):  # TOKEN_QUERY
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        mine = set()
        for token_class in (1, 4):  # TokenUser, TokenOwner: both start with the SID
            size = wintypes.DWORD(0)
            info(token, token_class, None, 0, ctypes.byref(size))
            buf = ctypes.create_string_buffer(max(size.value, 64))
            if not info(token, token_class, buf, len(buf), ctypes.byref(size)):
                raise ctypes.WinError(ctypes.get_last_error())
            mine.add(sid_text(ctypes.c_void_p.from_buffer(buf).value))
    finally:
        kernel32.CloseHandle(token)
    return owner_sid in mine


def _make_root(root: Path) -> None:
    """Creates the model folder if it is missing. On Windows only this user can change it (a folder made under C:\\
    would otherwise let every signed-in user replace the model files)."""
    if root.is_dir():
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        with contextlib.suppress(FileExistsError):
            _private_mkdir(root)
    else:
        root.mkdir(exist_ok=True)


def cache_root(cache_dir: Optional[os.PathLike] = None, log: Optional[Callable[[str], None]] = None) -> Path:
    """Where prepared models are kept: ``cache_dir``, or ~/.cache/kitten-inference/tts2.

    On Windows with a non-UTF-8 code page (the usual case), the engine opens only ASCII paths. A path with other
    characters is then replaced by its 8.3 short form when that is ASCII; else the default folder moves to an ASCII
    folder under %PROGRAMDATA% (with a note), and a given cache_dir is refused at once, before any download."""
    # Absolute once, here: the engine is started again later (load_voice, clone_voice) with these paths, also after
    # the program changed its working folder.
    root = Path(cache_dir).expanduser().absolute() if cache_dir is not None else DEFAULT_CACHE
    if _ascii(str(root)) or _narrow_paths_are_utf8():
        return root
    try:
        _make_root(root)
    except OSError:
        pass
    short = _short_path(root)
    if short is not None and _ascii(str(short)):
        return short
    if cache_dir is None:
        fallback = _fallback_root()
        if _ascii(str(fallback)):
            try:
                made = False
                if not fallback.is_dir():
                    # %PROGRAMDATA% lets every user add folders, and this one's name can be worked out from the user
                    # folder's path: make it private, and use an existing one only when this user owns it.
                    fallback.parent.mkdir(parents=True, exist_ok=True)
                    with contextlib.suppress(FileExistsError):
                        _private_mkdir(fallback)
                        made = True
                if not _owned_by_me(fallback):
                    raise PrepareError(f"{fallback}, the model folder for your user, belongs to another user; pass a "
                                       "new ASCII folder as cache_dir (KittenTTS2(cache_dir=\"C:/kitten-tts2\") or "
                                       "kitten-tts2 --cache-dir C:\\kitten-tts2), which is then made for you only")
                if made:  # say it once, when the folder is made
                    (log or _log)(f"the path of your user folder has characters the engine cannot open on this "
                                  f"Windows setup; models are kept in {fallback} instead (cache_dir= chooses another "
                                  "folder)")
                return fallback
            except OSError:
                pass
    raise PrepareError(f"the model folder {root} has characters outside ASCII, which the engine cannot open on Windows "
                       "with this system's code page. Use an ASCII folder: KittenTTS2(cache_dir=\"C:/kitten-tts2\") or "
                       "kitten-tts2 --cache-dir C:\\kitten-tts2, a new folder, which is then made so that only you can "
                       "change it (or turn on \"Use Unicode UTF-8 for worldwide language support\" in the Region "
                       "settings)")


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


def _remove_stale_partials(root: Path, keep: str) -> None:
    """Removes the .partial-<sha> folders of other revisions that no process is preparing (their lock is free): a
    stopped preparation of a revision that is no longer the latest would otherwise stay (up to about 2 GB)."""
    from filelock import FileLock, Timeout
    for partial in root.glob(".partial-*"):
        other = partial.name[len(".partial-"):]
        if other == keep or not _SHA.fullmatch(other) or not partial.is_dir():
            continue
        lock = FileLock(str(root / f".{other}.lock"))
        try:
            lock.acquire(timeout=0)
        except Timeout:
            continue
        try:
            shutil.rmtree(partial, ignore_errors=True)
        finally:
            lock.release()


def _previous(root: Path, repo: str, sha: str) -> Optional[str]:
    """The commit that main resolved to at the last successful preparation, when it is not ``sha`` and is prepared."""
    try:
        old = _ref_file(root, repo, "main").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return old if _SHA.fullmatch(old) and old != sha and is_prepared(root / old) else None


def ensure(*, text_data_tool: Path, revision: Optional[str] = None, cache_dir: Optional[os.PathLike] = None,
           with_cloning: bool = False, repo: str = MODEL_REPO, log: Callable[[str], None] = _log,
           before_download: Optional[Callable[[], None]] = None) -> Path:
    """The prepared model directory for ``revision`` (default: the latest, main), preparing it first if needed.
    ``before_download`` runs once before anything is downloaded (the GPU check: fail before minutes of work).

    Without ``revision``, when a newer main cannot be downloaded or prepared (no disk space, a download error, a
    model this version cannot convert), the revision prepared last time is used instead, with a warning."""
    root = cache_root(cache_dir, log)
    sha = resolve_revision(repo, revision, root, log)
    final = root / sha
    if is_prepared(final, with_cloning):
        _record_ref(root, repo, revision, sha)
        return final
    if before_download is not None:
        before_download()
    try:
        _prepare(root, final, sha, text_data_tool=text_data_tool, cache_dir=cache_dir, with_cloning=with_cloning,
                 repo=repo, log=log)
    except (PrepareError, OSError) as exc:
        previous = None if revision is not None else _previous(root, repo, sha)
        if previous is None:
            raise
        if not is_prepared(root / previous, with_cloning):
            raise PrepareError(f"{exc} (the earlier revision {previous} is prepared without voice cloning: "
                               f"revision=\"{previous}\" or kitten-tts2 --revision {previous} uses it)") from exc
        log(f"warning: the new {repo} revision {sha[:12]} could not be prepared ({exc}); using the revision prepared "
            f"before, {previous[:12]}. The next start tries again; revision=\"{previous}\" (kitten-tts2 --revision "
            f"{previous}) keeps this one")
        return root / previous
    _record_ref(root, repo, revision, sha)
    return final


def _prepare(root: Path, final: Path, sha: str, *, text_data_tool: Path, cache_dir: Optional[os.PathLike],
             with_cloning: bool, repo: str, log: Callable[[str], None]) -> None:
    """Downloads and prepares ``sha`` into ``final`` under the revision's lock (see ensure)."""
    _make_root(root)
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
                _remove_stale_partials(root, sha)
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
                # stopped earlier attempt, and so are the staging folders (.clone-prepare-*, up to 1.1 GB each) of
                # an attempt that was killed before it could remove its own.
                shutil.rmtree(final / "native" / "cloning", ignore_errors=True)
                for staging in (final / "native").glob(".clone-prepare-*"):
                    shutil.rmtree(staging, ignore_errors=True)
                log("[2/3] preparing the voice cloning files")
                argv = ["--model-dir", str(snapshot), "--out", str(final), "--cloning-only", *common]
                _run_prepare(argv, root / f"prepare-cloning-{sha[:12]}.log")
                if not is_prepared(final, True):
                    raise PrepareError(f"preparation finished without the cloning files in {final}")
            done = time.monotonic()
            log(f"[3/3] model ready in {done - started:.0f} s (downloads {downloaded - started:.0f} s, preparation "
                f"{done - downloaded:.0f} s): {final}")
