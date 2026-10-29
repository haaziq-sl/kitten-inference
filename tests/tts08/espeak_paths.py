"""CI check: TTS 0.8 in virtual environments whose espeak-ng data folder path is long (all systems) or, on Windows,
not ASCII (folders named "Jos\u00e9" and "\u0141ukasz"). Each case makes a fresh venv, installs the wheel from --wheels
with pip, and runs KittenTTS().generate() in a new Python process.

    espeak_paths.py deep|names --wheels DIR --expect fixed|broken [--spec REQ] [--overlay PKG_DIR] [--out FILE.json]

--spec is what pip installs, from --wheels plus the index for its dependencies: kitten-inference-tts08 (the default,
the TTS 0.8 build in engines/tts08.yml) or "kitten-inference" (the released wheels, packages.yml). The installed
version must be one of the wheels in --wheels.
--expect fixed: every case must speak (where Windows can load the engine at all; past that, the error must say the
path is too long). --expect broken: the cases must fail the way the old wheels do (espeak-ng ends the process after
"Error processing file ...phontab"; on Windows a 260+ character engine path gives "native engine ... is missing").
--overlay copies this checkout's Python files over the installed package (an early check on an older wheel).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

WIN = sys.platform == "win32"

SPEAK = r'''
import json, os
from kitten_inference.tts08 import KittenTTS
audio = KittenTTS().generate("Hello from a folder espeak did not like.", voice="Jasper")
info = {"audio_s": round(len(audio) / 24000, 3), "data_path": os.environ.get("ESPEAK_DATA_PATH")}
try:
    from kitten_inference.tts08 import _espeak_path
    info["how"] = _espeak_path.how.get("how")
except ImportError:
    pass
print("RESULT " + json.dumps(info))
'''

# Windows: what the C runtime and espeak-ng see. espeak_Initialize with espeakINITIALIZE_DONT_EXIT (0x8000) reports
# the error instead of ending the process, so the locale after it can be printed.
PROBE = r'''
import ctypes, os, sys
from kitten_inference.tts08 import cpp_model
import espeakng_loader
if hasattr(cpp_model, "_set_espeak_data_path"):
    cpp_model._set_espeak_data_path()
ucrt = ctypes.CDLL("ucrtbase")
ucrt.setlocale.restype = ctypes.c_char_p
ucrt.setlocale.argtypes = [ctypes.c_int, ctypes.c_char_p]
ucrt.getenv.restype = ctypes.c_char_p
ucrt.getenv.argtypes = [ctypes.c_char_p]
lib = ctypes.CDLL(espeakng_loader.get_library_path())
lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
print("PROBE ANSI code page:", ctypes.windll.kernel32.GetACP())
print("PROBE ESPEAK_DATA_PATH in Python:", ascii(os.environ["ESPEAK_DATA_PATH"]))
print("PROBE ESPEAK_DATA_PATH as the C runtime's getenv gives it (bytes):", ucrt.getenv(b"ESPEAK_DATA_PATH"))
print("PROBE C locale LC_CTYPE before espeak_Initialize:", ucrt.setlocale(2, None))
sys.stdout.flush()
rate = lib.espeak_Initialize(2, 0, None, 0x8000)
sys.stderr.flush()
print("PROBE espeak_Initialize returned:", rate)
print("PROBE C locale LC_CTYPE after espeak_Initialize:", ucrt.setlocale(2, None))
'''


def run(cmd, env=None, timeout=1200):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    text = p.stdout.decode("utf-8", "replace")
    lines = [ln for ln in text.strip().splitlines() if "unauthenticated requests" not in ln]
    return p.returncode, lines


def venv_python(venv):
    return str(venv / ("Scripts" if WIN else "bin") / ("python.exe" if WIN else "python"))


def make_venv(venv, wheels, overlay, env, spec):
    if venv.exists():
        shutil.rmtree(venv)
    venv.parent.mkdir(parents=True, exist_ok=True)
    rc, out = run([sys.executable, "-m", "venv", str(venv)], env=env)
    if rc:
        raise RuntimeError("venv: " + "\n".join(out[-5:]))
    py = venv_python(venv)
    rc, out = run([py, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--find-links", str(wheels),
                   spec], env=env)
    if rc:
        raise RuntimeError("pip: " + "\n".join(out[-8:]))
    # The index may have other versions of the same name (kitten-inference 0.1.x): the test is about these wheels.
    name = re.match(r"[A-Za-z0-9._-]+", spec).group(0)
    rc, out = run([py, "-c", "import sys, importlib.metadata as m; print(m.version(sys.argv[1]))", name], env=env)
    wanted = {f.name.split("-")[1] for f in Path(wheels).glob("*.whl")
              if re.sub(r"[-_.]+", "_", f.name.split("-")[0]).lower() == re.sub(r"[-_.]+", "_", name).lower()}
    got = out[-1] if out and not rc else "?"
    if got not in wanted:
        raise RuntimeError(f"pip installed {name} {got}, not one of the wheels in {wheels} "
                           f"(versions there: {sorted(wanted) or 'none'})")
    if overlay:
        rc, out = run([py, "-c", "import os, kitten_inference.tts08 as m; print(os.path.dirname(m.__file__))"],
                      env=env)
        dest = Path(out[-1])
        for f in Path(overlay).glob("*.py"):
            if f.name != "__init__.py":  # the wheel's own: delvewheel adds the DLL folder there on Windows
                shutil.copy2(f, dest / f.name)
    rc, out = run([py, "-c", "import espeakng_loader; print(espeakng_loader.get_data_path())"], env=env)
    return py, out[-1]


def short_name(path):
    import ctypes
    buf = ctypes.create_unicode_buffer(32768)
    n = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, len(buf))
    return buf.value if 0 < n < len(buf) else str(path)


class NoShortNames(Exception):
    pass


def make_short_named(folder):
    """Creates ``folder`` (a non-ASCII name on C:) with an ASCII 8.3 short name, as Windows 10/11 usually gives
    folders on the system drive. Turns 8.3 names on for C: first (the runner is an administrator; the per-volume
    setting needs the registry in per-volume mode, 2), else sets one. NoShortNames when this runner cannot."""
    for cmd in (["fsutil", "8dot3name", "set", "2"], ["fsutil", "8dot3name", "set", "C:", "0"],
                ["fsutil", "8dot3name", "query", "C:"]):
        rc, out = run(cmd)
        print(" ".join(cmd), "->", rc, " | ".join(out[-3:]), flush=True)
    # A new name each time: NTFS gives a folder made again under a just-deleted name the old short name (tunneling).
    folder = folder.parent / f"{folder.name}-{os.getpid()}"
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir()
    if not short_name(folder).isascii():
        rc, out = run(["fsutil", "file", "setshortname", str(folder), "LUKASZ8"])
        print("fsutil file setshortname ->", rc, " | ".join(out[-3:]), flush=True)
    rc, out = run(["cmd", "/c", "dir", "/x", "/ad", str(folder.parent)])
    print("dir /x:", " | ".join(ln for ln in out if "ukasz" in ln.lower() or "LUKASZ" in ln), flush=True)
    short = short_name(folder)
    if not short.isascii():
        raise NoShortNames(f"this runner gave {folder!r} no ASCII 8.3 name (got {short!r})")
    return folder, short


def judge(expect, kind, rc, lines):
    """(passed, what happened)"""
    text = "\n".join(lines)
    result = next((json.loads(ln[7:]) for ln in lines if ln.startswith("RESULT ")), None)
    if rc == 0 and result and result["audio_s"] > 0:
        got = "speaks"
    elif rc != 0 and "phontab" in text and "Traceback" not in text:
        got = "espeak-ng ended the process (no exception)"
    elif rc != 0 and "too long for Windows" in text:
        got = "clear error: path too long for Windows"
    elif rc != 0 and "is missing or does not match" in text:
        got = "misleading error: native engine missing"
    else:
        got = f"other (exit {rc})"
    want = {
        ("fixed", "speak"): "speaks",
        ("fixed", "engine-too-long"): "clear error: path too long for Windows",
        ("broken", "speak"): "espeak-ng ended the process (no exception)",
        ("broken", "engine-too-long"): "misleading error: native engine missing",
        ("broken", "windows-ok"): "speaks",
        ("fixed", "windows-ok"): "speaks",
        ("fixed", "short-name"): "speaks",
    }[(expect, kind)]
    return got == want, got, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=["deep", "names"])
    ap.add_argument("--wheels", required=True)
    ap.add_argument("--expect", choices=["fixed", "broken"], required=True)
    ap.add_argument("--spec", default="kitten-inference-tts08")
    ap.add_argument("--overlay")
    ap.add_argument("--out")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # folder names on a cp1252 console
    wheels = Path(args.wheels).resolve()
    base = Path(os.path.realpath(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir())) / f"ep-{args.expect}"
    hf_home = base / "hf"  # one model download for every case
    ver = "%d.%d" % sys.version_info[:2]
    tail = (os.path.join("Lib", "site-packages") if WIN else os.path.join("lib", f"python{ver}", "site-packages"))
    tail = os.path.join(tail, "espeakng_loader", "espeak-ng-data")

    cases = []  # (label, venv, kind, extra env)
    if args.case == "deep":
        # Windows espeak-ng takes 200 characters; past about 230 Windows cannot load the engine's files.
        targets = [(200, "windows-ok"), (230, "engine-too-long")] if WIN else [(200, "speak")]
        for target, kind in targets:
            pad = target - len(str(base)) - 2 * len(os.sep) - len(tail) - 1
            cases.append((f"data path {target} characters", base / ("v" + "x" * pad), kind, {}))
    else:
        if not WIN:
            sys.exit("names: Windows only")
        # A non-ASCII user profile: the venv, TEMP/TMP and LOCALAPPDATA all sit in such a folder. One on the
        # runner's work drive and one on C: (8.3 short names may differ per drive).
        for label, root in (("Jos\u00e9", base / "Jos\u00e9"), ("\u0141ukasz", Path("C:\\") / f"ep-{args.expect}" / "\u0141ukasz")):
            local = root / "AppData" / "Local"
            env = {"TEMP": str(local / "Temp"), "TMP": str(local / "Temp"), "LOCALAPPDATA": str(local)}
            cases.append((f"venv under {label}", root / "venv", "speak", env))
        if args.expect == "fixed":
            # The 8.3 route (_espeak_path.short_path): a non-ASCII folder with an ASCII short name. Last, because it
            # turns on 8.3 names for the runner's drives.
            root = Path("C:\\") / "ep-short" / "\u0141ukasz8"
            local = root / "AppData" / "Local"
            env = {"TEMP": str(local / "Temp"), "TMP": str(local / "Temp"), "LOCALAPPDATA": str(local)}
            cases.append(("venv under \u0141ukasz8 (ASCII 8.3 name)", root / "venv", "short-name", env))

    rows, failed = [], 0
    for label, venv, kind, extra in cases:
        row = {"case": label, "kind": kind}
        try:
            if kind == "short-name":  # the folder gets its 8.3 name when it is made, so make it first
                root, row["short_name"] = make_short_named(venv.parent)
                venv = root / "venv"
                local = root / "AppData" / "Local"
                extra = {"TEMP": str(local / "Temp"), "TMP": str(local / "Temp"), "LOCALAPPDATA": str(local)}
            row.update(venv=str(venv), venv_len=len(str(venv)))
            env = dict(os.environ, HF_HOME=str(hf_home), PYTHONIOENCODING="utf-8", **extra)
            for key in ("TEMP", "LOCALAPPDATA"):
                if key in extra:
                    Path(extra[key]).mkdir(parents=True, exist_ok=True)
            py, data_path = make_venv(venv, wheels, args.overlay, env, args.spec)
            row["data_path"], row["data_path_len"] = data_path, len(data_path)
            if WIN and args.case == "names":
                rc, lines = run([py, "-c", PROBE], env=env)
                row["probe"] = {"exit": rc, "lines": lines[-12:]}
            runs = []
            for attempt in range(2 if args.expect == "fixed" else 1):  # the second process reuses the copy
                rc, lines = run([py, "-c", SPEAK], env=env)
                ok, got, result = judge(args.expect, kind, rc, lines)
                runs.append({"exit": rc, "got": got, "ok": ok, "result": result, "tail": lines[-4:]})
            row["runs"] = runs
            row["ok"] = all(r["ok"] for r in runs)
            if row["ok"] and len(runs) == 2 and runs[0]["result"] and runs[1]["result"]:
                row["ok"] = runs[0]["result"]["data_path"] == runs[1]["result"]["data_path"]
            if kind == "short-name":  # espeak-ng must have loaded through the short name, not a copy
                row["ok"] = row["ok"] and all((r["result"] or {}).get("how") == "short-name" for r in runs)
        except NoShortNames as exc:  # a runner limit, not a TTS 0.8 failure: say so loudly, do not fail
            row["error"] = f"not run: {exc}"[:800]
            row["ok"] = True
            print(f"::warning::8.3 short-name case not run: {exc}", flush=True)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:800]
            row["ok"] = False
        failed += not row["ok"]
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True, indent=1), flush=True)
        shutil.rmtree(venv, ignore_errors=True)

    summary = [f"### espeak data paths: {args.case}, expect {args.expect}{' (overlay)' if args.overlay else ''}", "",
               "| case | data path chars | result | as expected |", "|---|---:|---|---|"]
    for r in rows:
        got = "; ".join(x["got"] for x in r.get("runs", [])) or r.get("error", "")
        summary.append(f"| {r['case']} | {r.get('data_path_len', '')} | {got} | {'yes' if r['ok'] else 'NO'} |")
    text = "\n".join(summary) + "\n"
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(text)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=True, indent=1), encoding="utf-8")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
