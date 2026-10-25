"""Measurement only: does the C runtime's narrow fopen() open a UTF-8 encoded non-ASCII path in this process?"""
import ctypes, locale, os, platform, sys, tempfile

k32 = ctypes.windll.kernel32
ucrt = ctypes.CDLL("ucrtbase")
ucrt.fopen.restype = ctypes.c_void_p
ucrt.fopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
ucrt._wfopen.restype = ctypes.c_void_p
ucrt._wfopen.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
ucrt.fclose.argtypes = [ctypes.c_void_p]
ucrt.setlocale.restype = ctypes.c_char_p
ucrt.setlocale.argtypes = [ctypes.c_int, ctypes.c_char_p]
ucrt.___lc_codepage_func.restype = ctypes.c_uint
LC_CTYPE = 2


def try_open(label, fn, arg, mode):
    f = fn(arg, mode)
    ok = bool(f)
    if ok:
        ucrt.fclose(f)
    print(f"  {label}: {'opened' if ok else 'FAILED'}")
    return ok


def state():
    print(f"  GetACP={k32.GetACP()} GetOEMCP={k32.GetOEMCP()} AreFileApisANSI={k32.AreFileApisANSI()} "
          f"CRT setlocale(LC_CTYPE)={ucrt.setlocale(LC_CTYPE, None)!r} CRT codepage={ucrt.___lc_codepage_func()} "
          f"python locale={locale.setlocale(locale.LC_CTYPE)!r}")


print(sys.version, platform.platform(), platform.version())
root = tempfile.mkdtemp(prefix="kitten-é-Ł-猫-")
path = os.path.join(root, "f.bin")
open(path, "wb").write(b"x")
print("path:", ascii(path))
for setting in (None, b"English_United States.1252", b"C", b".UTF8"):
    if setting is not None:
        ucrt.setlocale(LC_CTYPE, setting)
    print(f"after CRT setlocale(LC_CTYPE, {setting!r}):" if setting else "at startup:")
    state()
    try_open("narrow fopen, UTF-8 bytes", ucrt.fopen, path.encode("utf-8"), b"rb")
    try_open("wide _wfopen", ucrt._wfopen, path, "rb")
