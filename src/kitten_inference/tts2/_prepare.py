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
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

MODEL_REPO = "KittenML/kitten-tts-2"
DEFAULT_CACHE = Path.home() / ".cache" / "kitten-inference" / "tts2"
# Seconds to wait for Hugging Face when a start asks which commit main is (every start without a pinned revision).
# Without a limit, a network that drops packets (a firewall, a captive portal) holds the start for minutes or for
# ever before it falls back to the prepared model. HF_HUB_OFFLINE=1 skips the question.
HF_TIMEOUT = 10
_SHA = re.compile(r"[0-9a-f]{40}")
# The child process that runs prepare.main(argv) (kitten_inference.tts2._assets.prepare, the bundled copy of
# engines/tts2/tools/kt2n_assets): its output goes to a log file, and the memory the conversion uses is returned afterwards.
# `python -c` puts the current folder first on sys.path (only -P, Python 3.11+, turns that off), so a json.py or a
# kitten_inference/ folder where the user runs kitten-tts would be imported instead of the real modules: the child
# drops that entry before its first import (os and sys are already loaded at startup).
# The folder that holds this kitten_inference comes in _PACKAGE_VAR, not PYTHONPATH: PYTHONPATH entries go before the
# standard library, so a stale backport in site-packages with a standard-library name (PyPI's pathlib 1.0.1, enum34)
# would replace that module in the child only. The child adds the folder only when its path lacks it (the usual case:
# same interpreter, same site-packages), and then just before site-packages, where it takes precedence over another
# installed copy as it did in this process.
_PACKAGE_VAR = "_KITTEN_INFERENCE_TTS2_DIR"
_CHILD = ("import os, sys\n"
          "sys.path[:] = [p for i, p in enumerate(sys.path) if i or p not in ('', '.', os.getcwd())]\n"
          f"d = os.environ.pop({_PACKAGE_VAR!r}, '')\n"
          "if d and os.path.realpath(d) not in [os.path.realpath(p) for p in sys.path if p]:\n"
          "    try:\n"
          "        import site\n"
          "        sp = {os.path.realpath(p) for p in [*site.getsitepackages(), site.getusersitepackages()]}\n"
          "    except Exception:\n"
          "        sp = set()\n"
          "    at = [i for i, p in enumerate(sys.path) if os.path.realpath(p) in sp]\n"
          "    sys.path.insert(at[0] if at else len(sys.path), d)\n"
          "from kitten_inference.tts2._assets.prepare import main\n"
          "sys.exit(main(sys.argv[1:]))\n")
# Windows: a folder only this user (and SYSTEM and Administrators) can open, inherited by everything inside it. The
# same security descriptor as os.mkdir(path, 0o700) on Python 3.13.
_PRIVATE_SDDL = "D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FA;;;OW)"
# Windows access rights that let someone change a folder's contents or its access list.
_WRITE_RIGHTS = (0x2 | 0x4 | 0x10 | 0x40 | 0x100 | 0x10000 | 0x40000 | 0x80000  # add file/folder, EA, delete child,
                 | 0x10000000 | 0x40000000)                                      # attributes, delete, DAC, owner; ALL, WRITE
# Who may hold them besides this user: SYSTEM, Administrators, CREATOR OWNER, OWNER RIGHTS, TrustedInstaller.
_TRUSTED_SIDS = {"S-1-5-18", "S-1-5-32-544", "S-1-3-0", "S-1-3-4",
                 "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464"}


class PrepareError(RuntimeError):
    """The model could not be downloaded or prepared."""


def _log(message: str) -> None:
    print(f"kitten-tts: {message}", file=sys.stderr, flush=True)


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


def _windows() -> bool:
    return sys.platform == "win32"


def _under_profile(root: Path) -> bool:
    """Whether ``root`` is the default cache or sits inside the user's home profile, which Windows already keeps
    private to this user. Such a folder must NOT get the OWNER-RIGHTS private descriptor: a run in an elevated
    window owns new objects as Administrators, and OWNER RIGHTS would then lock the same user out in a normal
    window. The profile's own access list keeps the user's full control whoever owns the files."""
    try:
        return root == DEFAULT_CACHE or root.is_relative_to(Path.home())
    except (ValueError, OSError):
        return False


def _is_reparse_point(path: Path) -> bool:
    """Whether ``path`` is a reparse point (a junction or symbolic link). GetNamedSecurityInfoW and the owner/ACL
    checks follow such a link, so another user's junction that points at a folder of the victim's would pass them
    and could be re-pointed later; a model folder must be a real directory."""
    # The real platform, not _windows() (tests patch _windows() to exercise the Windows ACL path on other systems):
    # ctypes.windll exists only on a real Windows.
    if sys.platform != "win32":
        try:
            return path.is_symlink()
        except OSError:
            return False
    attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
    return attrs != 0xFFFFFFFF and bool(attrs & 0x400)  # INVALID_FILE_ATTRIBUTES; FILE_ATTRIBUTE_REPARSE_POINT


def _reject_reparse(path: Path, what: str) -> None:
    """Refuses a model folder that is a reparse point (Windows junction/symlink)."""
    if path.exists() and _is_reparse_point(path):
        raise PrepareError(f"the model {what} {path} is a reparse point (a junction or symbolic link); it could "
                           "point to another user's folder. Delete it, or pass a new folder as cache_dir "
                           "(KittenTTS(\"KittenML/kitten-tts-2\", cache_dir=...) or kitten-tts --model KittenML/kitten-tts-2 --cache-dir)")


def _win32():
    """advapi32, kernel32 and wintypes, with the prototypes used here."""
    from ctypes import wintypes
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    advapi32.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    advapi32.GetNamedSecurityInfoW.argtypes = ([wintypes.LPCWSTR, ctypes.c_int, wintypes.DWORD]
                                               + [ctypes.POINTER(ctypes.c_void_p)] * 5)
    advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD
    return advapi32, kernel32, wintypes


def _sid_text(psid) -> str:
    advapi32, kernel32, _ = _win32()
    out = ctypes.c_void_p()
    if not advapi32.ConvertSidToStringSidW(psid, ctypes.byref(out)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.wstring_at(out.value)
    finally:
        kernel32.LocalFree(out)


def _my_sids() -> set:
    """This process's user and the owner it gives new files (Administrators in an elevated window)."""
    advapi32, kernel32, wintypes = _win32()
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
            mine.add(_sid_text(ctypes.c_void_p.from_buffer(buf).value))
    finally:
        kernel32.CloseHandle(token)
    return mine


def _owned_by_me(path: Path) -> bool:
    """Whether this user owns ``path`` (on Windows: its owner is this process's user, or the owner the process gives
    new files, which is Administrators in an elevated window). OSError when it cannot be read."""
    if not _windows():
        return path.stat().st_uid == os.getuid()
    advapi32, kernel32, _ = _win32()
    owner, sd = ctypes.c_void_p(), ctypes.c_void_p()
    code = advapi32.GetNamedSecurityInfoW(str(path), 1, 1, ctypes.byref(owner), None, None, None,
                                          ctypes.byref(sd))  # SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION
    if code != 0:
        raise ctypes.WinError(code)
    try:
        owner_sid = _sid_text(owner)
    finally:
        kernel32.LocalFree(sd)
    return owner_sid in _my_sids()


def _other_writers(path: Path) -> List[str]:
    """Windows: who besides this user (and SYSTEM, Administrators) may add, change or delete files in ``path``, or
    change its access list, as SID strings. OSError when it cannot be read."""
    advapi32, kernel32, wintypes = _win32()
    dacl, sd = ctypes.c_void_p(), ctypes.c_void_p()
    code = advapi32.GetNamedSecurityInfoW(str(path), 1, 4, None, None, ctypes.byref(dacl), None,
                                          ctypes.byref(sd))  # DACL_SECURITY_INFORMATION
    if code != 0:
        raise ctypes.WinError(code)
    try:
        if not dacl.value:
            return ["everyone (the folder has no access list)"]

        class AclSizeInformation(ctypes.Structure):
            _fields_ = [("AceCount", wintypes.DWORD), ("AclBytesInUse", wintypes.DWORD),
                        ("AclBytesFree", wintypes.DWORD)]

        advapi32.GetAclInformation.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.c_int]
        advapi32.GetAclInformation.restype = wintypes.BOOL
        advapi32.GetAce.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p)]
        advapi32.GetAce.restype = wintypes.BOOL
        size = AclSizeInformation()
        if not advapi32.GetAclInformation(dacl, ctypes.byref(size), ctypes.sizeof(size), 2):  # AclSizeInformation
            raise ctypes.WinError(ctypes.get_last_error())
        trusted = _TRUSTED_SIDS | _my_sids()
        others = []
        for i in range(size.AceCount):
            ace = ctypes.c_void_p()
            if not advapi32.GetAce(dacl, i, ctypes.byref(ace)):
                raise ctypes.WinError(ctypes.get_last_error())
            ace_type, _flags, _size, mask = struct.unpack("<BBHI", ctypes.string_at(ace.value, 8))
            if ace_type in (1, 10):  # access denied (also callback): takes nothing away from this check
                continue
            if ace_type not in (0, 9):  # not a plain (or callback) allow entry: count it, to be safe
                others.append(f"an access entry of type {ace_type}")
                continue
            if mask & _WRITE_RIGHTS:
                sid = _sid_text(ace.value + 8)
                if sid not in trusted:
                    others.append(sid)
        return others
    finally:
        kernel32.LocalFree(sd)


def _set_private(path: Path) -> None:
    """Windows: gives ``path`` the private access list of _private_mkdir (inherited ones from its parent removed)."""
    advapi32, kernel32, wintypes = _win32()
    convert = advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    convert.restype = wintypes.BOOL
    get_dacl = advapi32.GetSecurityDescriptorDacl
    get_dacl.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL), ctypes.POINTER(ctypes.c_void_p),
                         ctypes.POINTER(wintypes.BOOL)]
    get_dacl.restype = wintypes.BOOL
    set_info = advapi32.SetNamedSecurityInfoW
    set_info.argtypes = [wintypes.LPWSTR, ctypes.c_int, wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, ctypes.c_void_p]
    set_info.restype = wintypes.DWORD
    sd = ctypes.c_void_p()
    if not convert(_PRIVATE_SDDL, 1, ctypes.byref(sd), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        present, defaulted, dacl = wintypes.BOOL(), wintypes.BOOL(), ctypes.c_void_p()
        if not get_dacl(sd, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted)):
            raise ctypes.WinError(ctypes.get_last_error())
        code = set_info(str(path), 1, 0x4 | 0x80000000, None, None, dacl, None)  # DACL + PROTECTED_DACL
        if code != 0:
            raise ctypes.WinError(code)
    finally:
        kernel32.LocalFree(sd)


def _refuse_folder(root: Path, mine: bool) -> None:
    why = "belongs to another user" if not mine else "can be changed by other users of this computer"
    raise PrepareError(f"the model folder {root} {why}, so the model in it could be replaced. Pass a new folder "
                       "as cache_dir (KittenTTS(\"KittenML/kitten-tts-2\", cache_dir=...) or kitten-tts --model KittenML/kitten-tts-2 --cache-dir), which is then made "
                       "so that only you can change it, or delete this one")


def _group_is_only_mine(gid: int) -> bool:
    """Whether the group ``gid`` is this user's private group (Ubuntu and Fedora give every user one, named like the
    user, and a umask of 002): its group-write bit lets no one else in."""
    try:
        import grp
        import pwd
        group, me = grp.getgrgid(gid), pwd.getpwuid(os.getuid()).pw_name
    except (ImportError, KeyError, OSError):
        return False
    return gid == os.getgid() and group.gr_name == me and set(group.gr_mem) <= {me}


def _check_posix_folder(root: Path) -> None:
    """Linux and macOS: the cache_dir folder (and a symbolic link to it) must be this user's, and no other user may
    write to it, or a model (or a link that our log file would be written through) planted there by another user of
    the machine would be used. An empty folder of this user's is made private (mode 0o700)."""
    try:
        link, st = os.lstat(root), os.stat(root)
        mine = link.st_uid == os.getuid() and st.st_uid == os.getuid()
        open_to_others = bool(st.st_mode & 0o002) or (bool(st.st_mode & 0o020) and not _group_is_only_mine(st.st_gid))
        if mine and open_to_others and not any(root.iterdir()):
            os.chmod(root, 0o700)
            open_to_others = bool(os.stat(root).st_mode & 0o022) or any(root.iterdir())
    except OSError as exc:
        raise PrepareError(f"cannot check who may change the model folder {root}: {exc}") from None
    if not mine or open_to_others:
        _refuse_folder(root, mine)


def _check_folder_is_yours(root: Path) -> None:
    """A model folder given as cache_dir that already exists must be this user's, and no other user may change it.
    On Windows, folders made under C:\\ or on another drive let every signed-in user add and change files by default,
    so a model there could be replaced with one that says something else; elsewhere see _check_posix_folder. An empty
    folder of this user's is made private; anything else is refused. Folders that do not exist yet are made private
    when they are created."""
    if not root.exists():
        return
    if not _windows():
        _check_posix_folder(root)
        return
    _reject_reparse(root, "folder")
    try:
        mine = _owned_by_me(root)
        others = _other_writers(root) if mine else []
        if mine and others and not any(root.iterdir()):
            _set_private(root)
            others = _other_writers(root) if not any(root.iterdir()) else others
    except OSError as exc:
        raise PrepareError(f"cannot check who may change the model folder {root}: {exc}") from None
    if not mine or others:
        _refuse_folder(root, mine)


def _make_root(root: Path) -> None:
    """Creates the model folder if it is missing, so that only this user can change it (on Windows a folder made under
    C:\\ would otherwise let every signed-in user replace the model files; elsewhere it gets mode 0o700)."""
    if root.is_dir():
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    # Only folders outside the user profile need the private descriptor (one made under C:\ or on another drive
    # otherwise lets every signed-in user replace the model files). The default cache and other folders inside the
    # profile are already private to this user; giving them the OWNER-RIGHTS descriptor would lock the same user out
    # after a run in an elevated window (see _under_profile).
    if sys.platform == "win32" and not _under_profile(root):
        with contextlib.suppress(FileExistsError):
            _private_mkdir(root)
    elif sys.platform == "win32":
        root.mkdir(exist_ok=True)
    else:
        root.mkdir(mode=0o700, exist_ok=True)


def _check_fallback(fallback: Path) -> None:
    """The %PROGRAMDATA% folder for this user's models (see cache_root): any user can make it first, and its name
    can be worked out from the user folder's path, so it must be a real folder (not a junction, which the owner and
    access checks would follow, to a folder of this user's that could be re-pointed later), owned by this user and
    changeable by no one else. OSError when it cannot be checked."""
    _reject_reparse(fallback, "folder")
    if not _owned_by_me(fallback) or (_windows() and _other_writers(fallback)):
        raise PrepareError(f"{fallback}, the model folder for your user, belongs to another user or can be changed by "
                           "other users; pass a new ASCII folder as cache_dir (KittenTTS(\"KittenML/kitten-tts-2\", cache_dir=\"C:/kitten-tts2\") "
                           "or kitten-tts --model KittenML/kitten-tts-2 --cache-dir C:\\kitten-tts2), which is then made for you only")


def _recheck_root(root: Path, cache_dir: Optional[os.PathLike]) -> bool:
    """Checks again a model folder that another user could reach (a given cache_dir, or the %PROGRAMDATA% fallback):
    a folder or junction planted since cache_root() looked (the Hugging Face request can take up to HF_TIMEOUT
    seconds) must not be used. Returns whether the folder needed the check."""
    if cache_dir is not None:
        _check_folder_is_yours(root)
        return True
    if root == DEFAULT_CACHE or root != _fallback_root():
        return False
    try:
        _check_fallback(root)
    except OSError as exc:
        raise PrepareError(f"cannot check who may change the model folder {root}: {exc}") from None
    return True


def cache_root(cache_dir: Optional[os.PathLike] = None, log: Optional[Callable[[str], None]] = None) -> Path:
    """Where prepared models are kept: ``cache_dir``, or ~/.cache/kitten-inference/tts2.

    On Windows with a non-UTF-8 code page (the usual case), the engine opens only ASCII paths. A path with other
    characters is then replaced by its 8.3 short form when that is ASCII; else the default folder moves to an ASCII
    folder under %PROGRAMDATA% (with a note), and a given cache_dir is refused at once, before any download."""
    # Absolute once, here: the engine is started again later (load_voice, clone_voice) with these paths, also after
    # the program changed its working folder.
    root = Path(cache_dir).expanduser().absolute() if cache_dir is not None else DEFAULT_CACHE
    if _ascii(str(root)) or _narrow_paths_are_utf8():
        if cache_dir is not None:
            _check_folder_is_yours(root)
        return root
    if cache_dir is not None:
        _check_folder_is_yours(root)
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
                _check_fallback(fallback)
                if made:  # say it once, when the folder is made
                    (log or _log)(f"the path of your user folder has characters the engine cannot open on this "
                                  f"Windows setup; models are kept in {fallback} instead (cache_dir= chooses another "
                                  "folder)")
                return fallback
            except OSError:
                pass
    raise PrepareError(f"the model folder {root} has characters outside ASCII, which the engine cannot open on Windows "
                       "with this system's code page. Use an ASCII folder: KittenTTS(\"KittenML/kitten-tts-2\", cache_dir=\"C:/kitten-tts2\") or "
                       "kitten-tts --model KittenML/kitten-tts-2 --cache-dir C:\\kitten-tts2, a new folder, which is then made so that only you can "
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
        sha = HfApi().model_info(repo, revision=rev, token=False, timeout=HF_TIMEOUT).sha
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
                              f"from {root / sha} (HF_HUB_OFFLINE=1 skips this check)")
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


def _package_dir() -> str:
    """The folder that holds this kitten_inference/tts2 (site-packages for an installed package)."""
    import kitten_inference.tts2 as package
    return str(Path(package.__file__).resolve().parents[2])


def _child_env(root: Path) -> dict:
    """The environment of the prepare child: where this package is (see _CHILD), and UTF-8 output. On Windows before
    Python 3.15 output sent to a file uses the ANSI code page, so printing a path outside it (C:\\Users\\Łukasz under
    cp1252) would stop the child after the conversion. Where the engine opens only ASCII paths, the child's temporary
    folder is also under the (ASCII) model folder, because the text-data tool reads its input from there."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    env[_PACKAGE_VAR] = _package_dir()
    if not _narrow_paths_are_utf8():
        temp = env.get("TEMP") or env.get("TMP") or ""
        if not _ascii(temp) or not temp:
            tmp = root / "tmp"
            tmp.mkdir(parents=True, exist_ok=True)
            env["TEMP"] = env["TMP"] = str(tmp)
    return env


def _check_child_python() -> None:
    """Preparation runs in a child Python (sys.executable -c ...). In an app frozen with PyInstaller, cx_Freeze or
    similar, sys.executable is the app itself: it ignores -c and starts the app again, which then waits for ever on
    the lock this process holds (or opens a second window). Refused before anything is downloaded."""
    if getattr(sys, "frozen", False) or not sys.executable:
        raise PrepareError("this program is a frozen app (PyInstaller, cx_Freeze or similar), which cannot start the "
                           "Python process that prepares the model. Prepare it once from a normal Python install with "
                           "kitten-tts --model KittenML/kitten-tts-2 --prepare-only (add --revision R and --cache-dir DIR as needed), then pass the "
                           "same revision= and cache_dir= to KittenTTS in the app")


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


def _pinned_inputs(prepare, cache_dir: Optional[os.PathLike]):
    """Downloads the small pinned text-processing inputs (the NeMo data wheel from PyPI, the grammar files from
    GitHub) before the 4.6 GB model snapshot, so a network that blocks either host fails in a few seconds rather
    than after the large download. Returns (grammar_dir, nemo_wheel) to hand the child, which then does not fetch
    them again. The cache path matches the child's (prepare.main: ``<cache-dir or ~/.cache/kitten-tts>``)."""
    conv = (_download_cache(cache_dir) or Path.home() / ".cache" / "kitten-tts") / "native-conversion"
    try:
        prepare.pinned_wheel(conv, False)
        grammar = prepare.pinned_grammar(conv, False)
    except Exception as exc:
        raise PrepareError(f"downloading the pinned text-processing files failed: {exc}. First use needs PyPI "
                           "(files.pythonhosted.org) and GitHub (raw.githubusercontent.com) reachable, besides "
                           "Hugging Face") from exc
    return Path(grammar), prepare.nemo_wheel_path(conv)


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
    # resolve_revision may have waited on Hugging Face; a folder or junction another user planted in that window must
    # not be trusted. Re-check the folder now that it exists, and refuse a <sha> reached through a junction.
    if _recheck_root(root, cache_dir) and _windows():
        _reject_reparse(final, "folder")
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
                               f"revision=\"{previous}\" or kitten-tts --model KittenML/kitten-tts-2 --revision {previous} uses it)") from exc
        log(f"warning: the new {repo} revision {sha[:12]} could not be prepared ({exc}); using the revision prepared "
            f"before, {previous[:12]}. The next start tries again; revision=\"{previous}\" (kitten-tts --model KittenML/kitten-tts-2 --revision "
            f"{previous}) keeps this one")
        return root / previous
    _record_ref(root, repo, revision, sha)
    return final


def _prepare(root: Path, final: Path, sha: str, *, text_data_tool: Path, cache_dir: Optional[os.PathLike],
             with_cloning: bool, repo: str, log: Callable[[str], None]) -> None:
    """Downloads and prepares ``sha`` into ``final`` under the revision's lock (see ensure)."""
    _make_root(root)
    _recheck_root(root, cache_dir)  # a folder planted while the root was being created is caught here
    with _locked(root / f".{sha}.lock"):
        if not is_prepared(final, with_cloning):
            _check_child_python()
            prepare = _assets()
            common = ["--repo", repo, "--revision", sha, "--weight-norm-device", "numpy"]  # numpy: no torch
            if cache_dir is not None:
                common += ["--cache-dir", str(_download_cache(cache_dir))]
            started = time.monotonic()
            if not is_prepared(final):
                patterns = list(prepare.MODEL_ALLOW_PATTERNS) + (list(prepare.CLONING_ALLOW_PATTERNS)
                                                                 if with_cloning else [])
                # Small pinned inputs first (a few MB from PyPI and GitHub), so a network that blocks either host
                # fails here rather than after the 4.6 GB model download (prepare.main's own order).
                grammar_dir, nemo_wheel = _pinned_inputs(prepare, cache_dir)
                log(f"[1/3] downloading {repo} at {sha[:12]} and the decoder weights (first use only)")
                snapshot = _snapshot(repo, sha, patterns, cache_dir)
                common += ["--acoustic-source", str(_acoustic(prepare, cache_dir)),
                           "--grammar-dir", str(grammar_dir), "--nemo-wheel", str(nemo_wheel)]
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
                try:
                    _run_prepare(argv, root / f"prepare-{sha[:12]}.log")
                    if not is_prepared(partial, with_cloning):
                        raise PrepareError(f"preparation finished without a complete model in {partial}")
                    if final.exists():  # an unprepared leftover (no config.json): replace it
                        shutil.rmtree(final)
                    os.replace(partial, final)
                except BaseException:
                    # A failed or stopped conversion leaves up to about 2 GB here: removed now, not at the next
                    # start (after a full disk, the disk would otherwise stay full).
                    shutil.rmtree(partial, ignore_errors=True)
                    raise
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
                try:
                    _run_prepare(argv, root / f"prepare-cloning-{sha[:12]}.log")
                    if not is_prepared(final, True):
                        raise PrepareError(f"preparation finished without the cloning files in {final}")
                except BaseException:
                    if not is_prepared(final, True):  # the files of the stopped attempt (up to about 1.1 GB)
                        shutil.rmtree(final / "native" / "cloning", ignore_errors=True)
                        for staging in (final / "native").glob(".clone-prepare-*"):
                            shutil.rmtree(staging, ignore_errors=True)
                    raise
            done = time.monotonic()
            log(f"[3/3] model ready in {done - started:.0f} s (downloads {downloaded - started:.0f} s, preparation "
                f"{done - downloaded:.0f} s): {final}")
