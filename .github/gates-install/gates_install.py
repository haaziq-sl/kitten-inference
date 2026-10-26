#!/usr/bin/env python3
"""gates-install: install the release files the way a user would, each case in a fresh venv.

Our five distributions come ONLY from --find-links WHEELS (no index serves them; real PyPI serves the
dependencies and kitten-inference 0.1.1 for the upgrade case). Every case records the pip command, its exit code
and warnings, where pip took each kitten-* file from, pip list, what imports and what messages a user sees, and
the files left in site-packages.

    gates_install.py --wheels DIR --out DIR [--python EXE] [--label NAME] [--upgrade-pip] [--cases a,b,...]
                     [--no-generate]

Standard library only (runs with the system Python of each machine). Writes OUT/result.json and OUT/log.txt.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WIN = sys.platform == "win32"
OURS = ("kitten-inference", "kitten-inference-tts08", "kitten-inference-tts2", "kitten-inference-tts2-cu12",
        "kitten-inference-tts2-rocm")

# What the probe checks inside a venv. Prints one JSON object.
PROBE = r'''
import json, os, sys, importlib
r = {}
def t(key, fn):
    try:
        r[key] = {"ok": True, "value": fn()}
    except BaseException as e:  # noqa: B036 - record every failure, SystemExit included
        r[key] = {"ok": False, "error": (type(e).__name__ + ": " + str(e))[:700]}
def umbrella():
    import kitten_inference as k
    return {"version": getattr(k, "__version__", None), "file": getattr(k, "__file__", None),
            "namespace": getattr(k, "__file__", None) is None}
t("kitten_inference", umbrella)
def tts08():
    import kitten_inference.tts08 as m
    from kitten_inference.tts08 import KittenTTS  # noqa: F401  (lazy attribute: loads get_model)
    return {"version": m.__version__}
t("tts08", tts08)
def engine(mod):
    def f():
        m = importlib.import_module(mod)
        return {"BACKEND": m.BACKEND, "ABI": m.ABI, "VERSION": m.VERSION, "CAPS": list(m.CAPS),
                "BUILD_INFO": m.BUILD_INFO, "lib": m.LIB.exists(), "cli": m.CLI.exists(),
                "text_data_tool": m.TEXT_DATA_TOOL.exists()}
    return f
t("tts2_cuda", engine("kitten_inference._tts2_cuda"))
t("tts2_rocm", engine("kitten_inference._tts2_rocm"))
def tts2():
    import kitten_inference.tts2 as m
    from kitten_inference.tts2 import KittenTTS2  # noqa: F401
    return {"version": m.__version__}
t("tts2", tts2)
def backend():
    from kitten_inference.tts2 import _backend
    b = _backend.find()
    return {"name": b.name, "distribution": b.distribution, "version": b.version, "can_clone": b.can_clone}
t("tts2_backend", backend)
def old():
    import model_inference
    return os.path.basename(os.path.dirname(model_inference.__file__)) + "/" + os.path.basename(model_inference.__file__)
t("model_inference", old)
print(json.dumps(r))
'''

# KittenTTS2() as a user calls it. CUDA_VISIBLE_DEVICES="" and HF_HUB_OFFLINE=1 keep it from downloading 4.6 GB on a
# machine with a GPU: the engine is found and loaded, then the GPU check stops it.
PROBE_TTS2 = r'''
import json, sys, tempfile
try:
    from kitten_inference.tts2 import KittenTTS2
    KittenTTS2(cache_dir=tempfile.mkdtemp())
    print(json.dumps({"ok": True}))
except BaseException as e:
    print(json.dumps({"ok": False, "error": (type(e).__name__ + ": " + str(e))[:700]}))
'''

PROBE_GEN = r'''
import json, sys, time
try:
    from kitten_inference.tts08 import KittenTTS
    t0 = time.time()
    tts = KittenTTS("KittenML/kitten-tts-nano-0.8")
    audio = tts.generate("Hello from the install check.", voice="Jasper")
    print(json.dumps({"ok": True, "seconds_of_audio": round(len(audio) / 24000, 3), "dtype": str(audio.dtype),
                      "backend": str(getattr(tts, "backend", "")), "wall_s": round(time.time() - t0, 1)}))
except BaseException as e:
    print(json.dumps({"ok": False, "error": (type(e).__name__ + ": " + str(e))[:700]}))
'''

LOG = None


def log(*parts):
    text = " ".join(str(p) for p in parts)
    LOG.write(text + "\n")
    LOG.flush()


def run(cmd, env=None, timeout=1800, cwd=None):
    t0 = time.time()
    full_env = dict(os.environ)
    full_env.update(env or {})
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=full_env, timeout=timeout,
                           cwd=cwd)
        out, rc = p.stdout.decode("utf-8", "replace"), p.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or b"").decode("utf-8", "replace") + "\n[TIMEOUT]", -999
    log(f"$ {' '.join(map(str, cmd))}  -> rc={rc} ({time.time() - t0:.1f} s)")
    log(out.rstrip())
    return rc, out, round(time.time() - t0, 1)


class Venv:
    def __init__(self, base: Path, python: str, upgrade_pip: bool):
        if base.exists():
            shutil.rmtree(base)
        rc, out, _ = run([python, "-m", "venv", str(base)])
        if rc != 0:
            raise SystemExit(f"venv failed: {out[-500:]}")
        self.dir = base
        self.bin = base / ("Scripts" if WIN else "bin")
        self.py = str(self.bin / ("python.exe" if WIN else "python"))
        self.upgrade = None
        if upgrade_pip:
            rc, out, _ = run([self.py, "-m", "pip", "install", "-U", "pip"])  # the docs' first line
            self.upgrade = {"rc": rc, "tail": out.strip().splitlines()[-1:]}
        self.pip_version = self.out(["-m", "pip", "--version"]).split()[1]
        self.site = Path(self.out(["-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"]).strip())
        self.platsite = Path(self.out(["-c", "import sysconfig; print(sysconfig.get_paths()['platlib'])"]).strip())

    def out(self, args):
        return subprocess.run([self.py, *args], stdout=subprocess.PIPE, check=True).stdout.decode()

    def report_ok(self):
        major, minor = (int(x) for x in self.pip_version.split(".")[:2])
        return (major, minor) >= (22, 2)


def sources(out: str, report_path):
    """Where pip took each kitten-* file from: the --report (pip >= 22.2) or the pip output."""
    found = []
    if report_path and Path(report_path).exists():
        rep = json.loads(Path(report_path).read_text(encoding="utf-8"))
        for item in rep.get("install", []):
            name = item["metadata"]["name"].lower().replace("_", "-")
            if name.startswith("kitten"):
                url = item.get("download_info", {}).get("url", "")
                found.append({"name": name, "version": item["metadata"]["version"],
                              "from": "find-links (local file)" if url.startswith("file:") else url,
                              "file": url.rsplit("/", 1)[-1]})
        return found
    for line in out.splitlines():
        s = line.strip()
        if re.search(r"kitten[-_]inference", s, re.I) and re.match(r"^(Processing|Downloading|Using cached)", s):
            found.append({"line": s[:300]})
    return found


def notes(out: str):
    keep = []
    for line in out.splitlines():
        s = line.strip()
        if re.match(r"^(WARNING|ERROR|DEPRECATION|error:|Successfully installed|Successfully uninstalled|Found existing|"
                    r"Attempting uninstall|Uninstalling )", s, re.I) or "does not provide the extra" in s:
            keep.append(s[:600])
    return keep


def pip_list(v: Venv):
    rows = json.loads(v.out(["-m", "pip", "list", "--format=json", "--disable-pip-version-check"]))
    ours = {r["name"].lower().replace("_", "-"): r["version"] for r in rows if r["name"].lower().startswith("kitten")}
    others = sorted(f"{r['name']}=={r['version']}" for r in rows
                    if not r["name"].lower().startswith("kitten") and r["name"].lower() not in ("pip", "setuptools", "wheel"))
    return {"ours": ours, "others": others}


def leftovers(v: Venv):
    found = {}
    for site in {v.site, v.platsite}:
        if not site.exists():
            continue
        for p in sorted(site.iterdir()):
            n = p.name.lower()
            if n.startswith(("kitten_inference", "model_inference", "kitten-inference")):
                if p.is_dir():
                    files = [str(q.relative_to(site)).replace("\\", "/") for q in p.rglob("*") if q.is_file()]
                    found[p.name + "/"] = {"files": len(files), "sample": sorted(files)[:12]}
                else:
                    found[p.name] = "file"
    scripts = sorted(p.name for p in v.bin.iterdir() if p.name.lower().startswith("kitten"))
    return {"site": found, "scripts": scripts}


def probe(v: Venv, generate: bool, cwd: Path):
    r = {}
    rc, out, _ = run([v.py, "-c", PROBE], cwd=cwd, timeout=600)
    try:
        r["imports"] = json.loads(out.strip().splitlines()[-1])
    except Exception:
        r["imports"] = {"probe_failed": out[-800:]}
    imp = r["imports"]
    if imp.get("tts08", {}).get("ok"):
        # its own process: on a CPU without the engine's instructions this import stops Python (KittenTTS() checks first)
        rc, out, _ = run([v.py, "-c", "import os, kitten_inference.tts08._engine as e; print(os.path.basename(e.__file__))"],
                         cwd=cwd, timeout=300)
        r["tts08 _engine import"] = {"rc": rc, "out": out.strip().splitlines()[-1:]}
    if imp.get("tts2", {}).get("ok"):
        rc, out, _ = run([v.py, "-c", PROBE_TTS2], cwd=cwd, timeout=600,
                         env={"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"})
        try:
            r["KittenTTS2()"] = json.loads(out.strip().splitlines()[-1])
        except Exception:
            r["KittenTTS2()"] = {"probe_failed": out[-800:]}
        exe = v.bin / ("kitten-tts2.exe" if WIN else "kitten-tts2")
        rc, out, _ = run([str(exe), "--version"], cwd=cwd, timeout=300, env={"CUDA_VISIBLE_DEVICES": ""})
        r["kitten-tts2 --version"] = {"rc": rc, "out": out.strip().splitlines()[:3]}
    if generate and imp.get("tts08", {}).get("ok"):
        rc, out, _ = run([v.py, "-c", PROBE_GEN], cwd=cwd, timeout=900)
        try:
            r["tts08 generate"] = json.loads(out.strip().splitlines()[-1])
        except Exception:
            r["tts08 generate"] = {"probe_failed": out[-800:]}
    rc, out, _ = run([v.py, "-m", "pip", "check", "--disable-pip-version-check"])
    r["pip check"] = {"rc": rc, "out": out.strip().splitlines()[:4]}
    return r


def install(v: Venv, spec: str, wheels: Path, work: Path, tag: str, extra=(), find_links=True):
    report = work / f"report-{tag}.json"
    cmd = [v.py, "-m", "pip", "install", *extra]
    if find_links:
        cmd += ["--find-links", str(wheels)]
    if v.report_ok():
        cmd += ["--report", str(report)]
    cmd.append(spec)
    rc, out, secs = run(cmd)
    return {"cmd": "pip install " + " ".join(extra) + (" --find-links WHEELS " if find_links else " ") + spec,
            "rc": rc, "seconds": secs, "notes": notes(out),
            "sources": sources(out, report if v.report_ok() else None)}


def uninstall(v: Venv, dist: str):
    rc, out, secs = run([v.py, "-m", "pip", "uninstall", "-y", dist])
    return {"cmd": f"pip uninstall -y {dist}", "rc": rc, "notes": notes(out)}


def state(v, generate, cwd):
    return {"pip_list": pip_list(v), "probe": probe(v, generate, cwd), "leftovers": leftovers(v)}


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--label", default=platform.platform())
    ap.add_argument("--upgrade-pip", action="store_true")
    ap.add_argument("--cases", default="")
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--rocm", action="store_true", help="also put tts-2-rocm in the uninstall cases (Linux)")
    a = ap.parse_args()
    wheels = Path(a.wheels).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    LOG = open(out / "log.txt", "a", encoding="utf-8")
    work = Path(tempfile.mkdtemp(prefix="gi-"))
    cwd = work / "cwd"
    cwd.mkdir()
    generate = not a.no_generate
    files = sorted(p.name for p in wheels.iterdir())
    res = {"label": a.label, "python": run([a.python, "--version"])[1].strip(), "platform": platform.platform(),
           "machine": platform.machine(), "upgrade_pip": a.upgrade_pip, "wheels_dir_files": len(files),
           "wheels_sha256_of_listing": hashlib.sha256("\n".join(files).encode()).hexdigest(),
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cases": {}}
    both = "kitten-inference[tts-0.8,tts-2]"
    cases = {
        "plain": [("install", "kitten-inference")],
        "tts-0.8": [("install", "kitten-inference[tts-0.8]")],
        "tts-0-8": [("install", "kitten-inference[tts-0-8]")],
        "tts-2": [("install", "kitten-inference[tts-2]")],
        "tts-2-rocm": [("install", "kitten-inference[tts-2-rocm]")],
        "both": [("install", both)],
        "both-0-8": [("install", "kitten-inference[tts-0-8,tts-2]")],
        "cuda+rocm": [("install", "kitten-inference[tts-2,tts-2-rocm]")],
        "upgrade": [("install-pypi", "kitten-inference==0.1.1"), ("install-U", "kitten-inference[tts-0.8]")],
        "upgrade-0-8": [("install-pypi", "kitten-inference==0.1.1"), ("install-U", "kitten-inference[tts-0-8]")],
        "upgrade-plain": [("install-pypi", "kitten-inference==0.1.1"), ("install-U", "kitten-inference")],
        "uninstall-umbrella-first": [("install", both)] + ([("install", "kitten-inference[tts-2-rocm]")] if a.rocm else []) +
            [("uninstall", d) for d in ("kitten-inference", "kitten-inference-tts2-cu12") +
             (("kitten-inference-tts2-rocm",) if a.rocm else ()) + ("kitten-inference-tts08", "kitten-inference-tts2")],
        "uninstall-engines-first": [("install", both)] +
            [("uninstall", d) for d in ("kitten-inference-tts2", "kitten-inference-tts08", "kitten-inference-tts2-cu12",
                                        "kitten-inference")],
    }
    chosen = [c for c in a.cases.split(",") if c] or list(cases)
    for name in chosen:
        steps = cases[name]
        log(f"\n===== case {name} =====")
        v = Venv(work / f"venv-{name}", a.python, a.upgrade_pip)
        case = {"pip": v.pip_version, "pip_upgrade": v.upgrade, "steps": []}
        for i, (kind, arg) in enumerate(steps):
            if kind == "install":
                step = install(v, arg, wheels, work, f"{name}-{i}")
            elif kind == "install-U":
                step = install(v, arg, wheels, work, f"{name}-{i}", extra=("-U",))
            elif kind == "install-pypi":
                step = install(v, arg, wheels, work, f"{name}-{i}", find_links=False)
            else:
                step = uninstall(v, arg)
            step.update(state(v, generate and ((kind != "uninstall" and i == 0) or kind == "install-U"), cwd))
            if kind == "install-pypi":
                # the files 0.1.1 installed, to check after the upgrade that none is left behind
                shown = v.out(["-m", "pip", "show", "-f", "kitten-inference", "--disable-pip-version-check"])
                rel = [ln.strip() for ln in shown.split("Files:", 1)[-1].splitlines() if ln.strip()]
                old_files = [r for r in rel if not r.startswith("..")]
                step["old_files"] = {"count": len(old_files), "sample": old_files[:20]}
            if kind == "install-U" and case["steps"] and "old_files_all" in case:
                left = [r for r in case["old_files_all"] if (v.site / r).exists() or (v.platsite / r).exists()]
                step["old_files_left"] = left[:50]
            if kind == "install-pypi":
                case["old_files_all"] = old_files
            case["steps"].append(step)
        case.pop("old_files_all", None)
        res["cases"][name] = case
        (out / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        shutil.rmtree(v.dir, ignore_errors=True)
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("done", out / "result.json")


if __name__ == "__main__":
    main()
