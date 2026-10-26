#!/usr/bin/env python3
"""Windows tts08 wheels: adds the Microsoft Visual C++ runtime under its own file names (kitten_inference/tts08/_msvc/).

    python .github/scripts/add_msvc_runtime.py WHEEL_DIR    (after delvewheel; KITTEN_MSVC_REDIST names the folders)

delvewheel bundles the runtime that _engine.pyd needs under changed names (msvcp140-<hash>.dll), so it never clashes
with another copy in the process. espeak-ng.dll from espeakng_loader imports MSVCP140.dll (and VCRUNTIME140_1.dll) by
their plain names, which a renamed copy does not satisfy: on a Windows without the Visual C++ redistributable,
phonemizer then reports "espeak not installed". cpp_model.py loads these copies only when Windows cannot load the
runtime by name. They come from the same redistributable folder (VC\\Redist) of the Visual Studio that built the engine.
The wheel's RECORD gets their hashes; every other member is copied unchanged.
"""
import base64
import hashlib
import os
import sys
import zipfile
from pathlib import Path

NAMES = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")
DEST = "kitten_inference/tts08/_msvc/"


def crt_folder() -> Path:
    folders = [Path(p) for p in os.environ.get("KITTEN_MSVC_REDIST", "").split(";") if p.strip()]
    crt = [f for f in folders if f.name.upper().endswith(".CRT")]
    if len(crt) != 1:
        raise SystemExit(f"error: KITTEN_MSVC_REDIST must name one Microsoft.VC*.CRT folder, got {folders}")
    return crt[0]


def record_line(name: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{name},sha256={digest},{len(data)}"


def add(wheel: Path, crt: Path) -> list:
    files = {n.lower(): p for p in crt.iterdir() for n in [p.name] if p.is_file()}
    extra = {DEST + n: files[n].read_bytes() for n in NAMES if n in files}
    if "msvcp140.dll" not in [k.rsplit("/", 1)[1] for k in extra]:
        raise SystemExit(f"error: no msvcp140.dll in {crt}")
    tmp = wheel.with_name(wheel.name + ".tmp")
    with zipfile.ZipFile(wheel) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        names = zin.namelist()
        record = next(n for n in names if n.endswith(".dist-info/RECORD"))
        if any(n in names for n in extra):
            raise SystemExit(f"error: {wheel.name} already has files in {DEST}")
        for info in zin.infolist():
            if info.filename == record:
                continue
            zout.writestr(info, zin.read(info.filename))
        for name, data in extra.items():
            zout.writestr(zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)), data, zipfile.ZIP_DEFLATED)
        lines = [l for l in zin.read(record).decode("utf-8").splitlines() if l.strip()]
        lines = [l for l in lines if not l.startswith(record + ",")]
        lines += [record_line(n, d) for n, d in extra.items()] + [f"{record},,"]
        zout.writestr(zin.getinfo(record), "\n".join(lines) + "\n")
    os.replace(tmp, wheel)
    return sorted(extra)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    crt = crt_folder()
    wheels = sorted(Path(sys.argv[1]).glob("*.whl"))
    if not wheels:
        raise SystemExit(f"error: no wheel in {sys.argv[1]}")
    for wheel in wheels:
        for name in add(wheel, crt):
            print(f"{wheel.name}: added {name} (from {crt})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
