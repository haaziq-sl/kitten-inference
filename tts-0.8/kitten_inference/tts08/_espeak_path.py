"""A data folder path that espeak-ng can open.

espeak-ng (the library from espeakng_loader that turns text into phonemes) is told where its data folder is through
ESPEAK_DATA_PATH. Two kinds of path make it print "Error processing file '.../phontab'" and end the whole Python
process with exit status 1, with no exception a caller could catch:

- a long path: espeak-ng keeps it in a fixed 160-byte buffer (Linux and macOS; 158 characters work, 160 do not), so a
  virtual environment about 100 characters deep is enough;
- on Windows, a path with a character outside ASCII (a user folder such as C:\\Users\\Jose with an accent, or a
  Polish, Cyrillic or CJK name): espeak-ng reads the variable in the ANSI code page, then switches the C library to
  UTF-8 before it opens its files, so the same bytes no longer name the folder.

usable_data_path() returns a path that avoids both: the data folder itself when its path is short (and ASCII on
Windows); on Windows its 8.3 short form when that is short and ASCII; otherwise a copy of the folder in a short ASCII
folder that only this user can change. Each copy is made once per espeakng_loader version and content (the folder name
holds both), complete before it gets its final name (a rename), so processes that start at the same time share one.
EspeakPathError when no such folder can be made; its message says why and what to do.
"""
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import threading
import time
import uuid

# Paths of this many bytes or more are not given to espeak-ng (its buffer is 160 bytes; files below the folder add to
# the path, so keep a margin).
MAX_PATH_BYTES = 150
MARKER = "kitten-inference-espeak.json"
WINDOWS = sys.platform == "win32"


class EspeakPathError(RuntimeError):
    pass


def fits(path, windows=WINDOWS):
    """Whether espeak-ng can use ``path`` as it is."""
    if len(os.fsencode(path)) >= MAX_PATH_BYTES:
        return False
    return not windows or path.isascii()


def why_not(path, windows=WINDOWS):
    if len(os.fsencode(path)) >= MAX_PATH_BYTES:
        return f"its path is {len(os.fsencode(path))} bytes long and espeak-ng needs fewer than {MAX_PATH_BYTES}"
    if windows and not path.isascii():
        return "its path has characters outside ASCII, which espeak-ng cannot open on Windows"
    return ""


def content_digest(src):
    """sha256 over the names and contents of every file in ``src``."""
    h = hashlib.sha256()
    for root, dirs, files in os.walk(src):
        dirs.sort()
        for name in sorted(files):
            if name == MARKER:
                continue
            path = os.path.join(root, name)
            h.update(os.path.relpath(path, src).replace(os.sep, "/").encode("utf-8") + b"\0")
            with open(path, "rb") as f:
                h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# Windows: 8.3 names, and folders only this user can change (the same checks as kitten_inference.tts2's model folder).

_PRIVATE_SDDL = "D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FA;;;OW)"
_WRITE_RIGHTS = (0x2 | 0x4 | 0x10 | 0x40 | 0x100 | 0x10000 | 0x40000 | 0x80000 | 0x10000000 | 0x40000000)
# SYSTEM, Administrators, CREATOR OWNER, OWNER RIGHTS, TrustedInstaller.
_TRUSTED_SIDS = {"S-1-5-18", "S-1-5-32-544", "S-1-3-0", "S-1-3-4",
                 "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464"}


def short_path(path):
    """The 8.3 short form of an existing Windows path, or None."""
    import ctypes
    try:
        buf = ctypes.create_unicode_buffer(32768)
        n = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, len(buf))
    except (AttributeError, OSError):
        return None
    return buf.value if 0 < n < len(buf) else None


def _win32():
    import ctypes
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
    return ctypes, advapi32, kernel32, wintypes


def _sid_text(psid):
    ctypes, advapi32, kernel32, _ = _win32()
    out = ctypes.c_void_p()
    if not advapi32.ConvertSidToStringSidW(psid, ctypes.byref(out)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.wstring_at(out.value)
    finally:
        kernel32.LocalFree(out)


def _my_sids():
    """This process's user and the owner it gives new files (Administrators in an elevated window)."""
    ctypes, advapi32, kernel32, wintypes = _win32()
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    info = advapi32.GetTokenInformation
    info.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    info.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)):  # TOKEN_QUERY
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        mine = []
        for token_class in (1, 4):  # TokenUser, TokenOwner: both start with the SID
            size = wintypes.DWORD(0)
            info(token, token_class, None, 0, ctypes.byref(size))
            buf = ctypes.create_string_buffer(max(size.value, 64))
            if not info(token, token_class, buf, len(buf), ctypes.byref(size)):
                raise ctypes.WinError(ctypes.get_last_error())
            mine.append(_sid_text(ctypes.c_void_p.from_buffer(buf).value))
    finally:
        kernel32.CloseHandle(token)
    return mine


def _is_reparse_point(path):
    """A junction or symbolic link (owner and access checks follow it, so it is refused)."""
    if not WINDOWS:
        return os.path.islink(path)
    import ctypes
    attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
    return attrs != 0xFFFFFFFF and bool(attrs & 0x400)  # INVALID_FILE_ATTRIBUTES; FILE_ATTRIBUTE_REPARSE_POINT


def _owned_by_me(path):
    ctypes, advapi32, kernel32, _ = _win32()
    owner, sd = ctypes.c_void_p(), ctypes.c_void_p()
    code = advapi32.GetNamedSecurityInfoW(str(path), 1, 1, ctypes.byref(owner), None, None, None, ctypes.byref(sd))
    if code != 0:
        raise ctypes.WinError(code)
    try:
        return _sid_text(owner) in _my_sids()
    finally:
        kernel32.LocalFree(sd)


def _other_writers(path):
    """Who besides this user, SYSTEM and Administrators may change ``path`` (SID strings)."""
    import struct
    ctypes, advapi32, kernel32, wintypes = _win32()
    dacl, sd = ctypes.c_void_p(), ctypes.c_void_p()
    code = advapi32.GetNamedSecurityInfoW(str(path), 1, 4, None, None, ctypes.byref(dacl), None, ctypes.byref(sd))
    if code != 0:
        raise ctypes.WinError(code)
    try:
        if not dacl.value:
            return ["everyone (no access list)"]

        class AclSizeInformation(ctypes.Structure):
            _fields_ = [("AceCount", wintypes.DWORD), ("AclBytesInUse", wintypes.DWORD),
                        ("AclBytesFree", wintypes.DWORD)]

        advapi32.GetAclInformation.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.c_int]
        advapi32.GetAclInformation.restype = wintypes.BOOL
        advapi32.GetAce.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p)]
        advapi32.GetAce.restype = wintypes.BOOL
        size = AclSizeInformation()
        if not advapi32.GetAclInformation(dacl, ctypes.byref(size), ctypes.sizeof(size), 2):
            raise ctypes.WinError(ctypes.get_last_error())
        trusted = _TRUSTED_SIDS | set(_my_sids())
        others = []
        for i in range(size.AceCount):
            ace = ctypes.c_void_p()
            if not advapi32.GetAce(dacl, i, ctypes.byref(ace)):
                raise ctypes.WinError(ctypes.get_last_error())
            ace_type, _flags, _size, mask = struct.unpack("<BBHI", ctypes.string_at(ace.value, 8))
            if ace_type in (1, 10):  # access denied entries take nothing away from this check
                continue
            if ace_type not in (0, 9):
                others.append(f"an access entry of type {ace_type}")
                continue
            if mask & _WRITE_RIGHTS:
                sid = _sid_text(ace.value + 8)
                if sid not in trusted:
                    others.append(sid)
        return others
    finally:
        kernel32.LocalFree(sd)


def _private_mkdir_windows(path):
    """Creates ``path`` with an access list for SYSTEM, Administrators and its owner only, which everything inside
    inherits. FileExistsError when something is already there."""
    ctypes, advapi32, kernel32, wintypes = _win32()

    class SecurityAttributes(ctypes.Structure):
        _fields_ = [("nLength", wintypes.DWORD), ("lpSecurityDescriptor", ctypes.c_void_p),
                    ("bInheritHandle", wintypes.BOOL)]

    convert = advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    convert.restype = wintypes.BOOL
    create = kernel32.CreateDirectoryW
    create.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(SecurityAttributes)]
    create.restype = wintypes.BOOL
    sd = ctypes.c_void_p()
    if not convert(_PRIVATE_SDDL, 1, ctypes.byref(sd), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        sa = SecurityAttributes(ctypes.sizeof(SecurityAttributes), sd, False)
        if not create(str(path), ctypes.byref(sa)):
            code = ctypes.get_last_error()
            if code == 183:  # ERROR_ALREADY_EXISTS
                raise FileExistsError(17, "the folder already exists", str(path))
            raise ctypes.WinError(code)
    finally:
        kernel32.LocalFree(sd)


# ---------------------------------------------------------------------------------------------------------------------

def _private_folder(path):
    """Makes ``path`` (its parent exists) a folder only this user can change, or checks that the existing one is.
    Returns None when it is, else why not."""
    try:
        if WINDOWS:
            try:
                _private_mkdir_windows(path)
            except FileExistsError:
                pass
        else:
            try:
                os.mkdir(path, 0o700)
            except FileExistsError:
                pass
        if _is_reparse_point(path):
            return "it is a link (a junction or symbolic link)"
        if not os.path.isdir(path):
            return "it is not a folder"
        if WINDOWS:
            if not _owned_by_me(path):
                return "it belongs to another user"
            others = _other_writers(path)
            if others:
                return "other users can change it (" + ", ".join(others) + ")"
        else:
            st = os.lstat(path)
            if st.st_uid != os.getuid():
                return "it belongs to another user"
            if st.st_mode & 0o022:
                return "other users can change it"
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _valid_copy(path, digest):
    try:
        if _is_reparse_point(path) or not os.path.isfile(os.path.join(path, "phontab")):
            return False
        with open(os.path.join(path, MARKER), encoding="utf-8") as f:
            return json.load(f).get("sha256") == digest
    except (OSError, ValueError, AttributeError):
        return False


def _copy_into(parent, name, src, digest, version):
    """``parent``/``name``: a complete copy of ``src``, made now (renamed into place when complete) or already there.
    Raises OSError when it cannot be made."""
    final = os.path.join(parent, name)
    if _valid_copy(final, digest):
        return final
    staging = os.path.join(parent, f".{name}.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    os.mkdir(staging, 0o700)  # inside a private parent, so private too
    try:
        shutil.copytree(src, staging, dirs_exist_ok=True)
        if content_digest(staging) != digest:
            raise OSError(f"the copy in {staging} does not match {src}")
        with open(os.path.join(staging, MARKER), "w", encoding="utf-8") as f:
            json.dump({"espeakng_loader": version, "sha256": digest, "source": src}, f)
        for attempt in range(5):
            try:
                os.rename(staging, final)
                break
            except OSError:
                # Another process finished the same copy first (the name is taken), or a broken one is in the way
                # (only this user can write here), or a virus scanner still holds a file (Windows): look, then retry.
                if _valid_copy(final, digest):
                    return final
                if attempt == 4:
                    raise
                if os.path.isdir(final) and not _is_reparse_point(final):
                    shutil.rmtree(final, ignore_errors=True)
                time.sleep(0.1 * (attempt + 1))
    finally:
        if os.path.isdir(staging):
            shutil.rmtree(staging, ignore_errors=True)
    if not _valid_copy(final, digest):
        raise OSError(f"{final} is not a complete copy")
    return final


def _user_tag():
    """A short name for this user (and, on Windows, the owner of new files, which differs in an elevated window)."""
    if WINDOWS:
        try:
            who = "|".join(_my_sids())
        except OSError:
            who = os.path.expanduser("~")
    else:
        who = str(os.getuid())
    return hashlib.sha256(who.encode("utf-8")).hexdigest()[:8]


def _copy_roots(windows=WINDOWS):
    """(parents to create, the private folder for copies) for each place a copy may go, best first."""
    roots = []
    if windows:
        program_data = os.environ.get("PROGRAMDATA") or "C:\\ProgramData"
        roots.append(([os.path.join(program_data, "kitten-inference")],
                      os.path.join(program_data, "kitten-inference", "espeak-" + _user_tag())))
        tmp = tempfile.gettempdir()
        if not fits(tmp, windows):
            tmp = short_path(tmp) or tmp
        roots.append(([], os.path.join(tmp, "kitten-espeak-" + _user_tag())))
    else:
        cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        roots.append(([cache, os.path.join(cache, "kitten-inference")],
                      os.path.join(cache, "kitten-inference", "espeak")))
        roots.append(([], os.path.join(tempfile.gettempdir(), "kitten-espeak-" + _user_tag())))
    return roots


_lock = threading.Lock()
_done = {}
how = {}  # how the last path was chosen ("as-is", "short-name" or "copy"), for tests and bug reports


def usable_data_path(src, version="unknown", roots=None):
    """A path to espeak-ng's data folder ``src`` (or to a copy of it) that espeak-ng can open."""
    with _lock:
        if src in _done:
            return _done[src]
        path = _choose(src, str(version), roots)
        _done[src] = path
        return path


def _choose(src, version, roots):
    if fits(src):
        how.update(how="as-is", path=src)
        return src
    if WINDOWS:
        short = short_path(src)
        if short and fits(short):
            how.update(how="short-name", path=short)
            return short
    digest = content_digest(src)
    name = f"{''.join(c if c.isalnum() or c in '.-' else '_' for c in version)}-{digest[:16]}"
    tried = []
    for parents, folder in (roots if roots is not None else _copy_roots()):
        final = os.path.join(folder, name)
        if not fits(final):
            tried.append(f"{final}: {why_not(final)}")
            continue
        try:
            for parent in parents:
                os.makedirs(parent, exist_ok=True)
            problem = _private_folder(folder)
            if problem:
                tried.append(f"{folder}: {problem}")
                continue
            path = _copy_into(folder, name, src, digest, version)
        except OSError as exc:
            tried.append(f"{final}: {type(exc).__name__}: {exc}")
            continue
        how.update(how="copy", path=path)
        return path
    raise EspeakPathError(
        f"espeak-ng (the text-to-phoneme library of kitten-inference-tts08) cannot use its data folder {src}: "
        f"{why_not(src)}, and it would end the Python process. No short copy could be made either ("
        + "; ".join(tried) + "). Fix: create the virtual environment in a shorter folder"
        + (" whose path has only ASCII characters (for example C:\\venvs\\kitten)" if WINDOWS else
           " (for example ~/venvs/kitten)")
        + (", or set TMP" if WINDOWS else ", or set TMPDIR") + " to such a folder that you can write to."
    )
