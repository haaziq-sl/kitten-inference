#!/usr/bin/env python3
"""Checks that engines/tts2/packaging/build_wheel_rocm.sh runs on the ROCm wheel it builds.

  check_wheel_rocm.py tag WHEEL TAG     the wheel's file name and its .dist-info/WHEEL file both carry exactly TAG
  check_wheel_rocm.py symbols FILE...   no ELF file needs a glibc, libstdc++ or libgcc symbol version newer than the
                                        manylinux_2_28 policy allows (GLIBC 2.28, GLIBCXX 3.4.24, CXXABI 1.3.11,
                                        GCC 7.0.0); versions of other libraries (the system ROCm's hip_*) are ignored

auditwheel repair --plat enforces the same floor, but auditwheel show cannot confirm it for this wheel: show does not
know that libamdhip64.so.7 is excluded, so it reports linux_x86_64. These checks read the files directly instead.
Each check prints one OK line, or a reason and exits with status 1. Needs readelf (binutils) for "symbols".
"""
import os
import re
import subprocess
import sys
import zipfile

# The newest version of each symbol family in auditwheel's manylinux_2_28 x86_64 policy (auditwheel/policy/
# manylinux-policy.json, "symbol_versions"), and the two non-numeric versions that policy also lists.
MANYLINUX_2_28 = {"GLIBC": (2, 28), "GLIBCXX": (3, 4, 24), "CXXABI": (1, 3, 11), "GCC": (7, 0, 0)}
ALSO_ALLOWED = {"CXXABI_FLOAT128", "CXXABI_TM_1"}


def fail(msg):
    print(f"check_wheel_rocm: {msg}", file=sys.stderr)
    sys.exit(1)


def check_tag(wheel, want):
    name = os.path.basename(wheel)
    if not name.endswith(".whl"):
        fail(f"{name}: not a .whl file")
    parts = name[: -len(".whl")].split("-")
    if len(parts) not in (5, 6):
        fail(f"{name}: not a wheel file name (name-version[-build]-python-abi-platform.whl)")
    got = "-".join(parts[-3:])
    if got != want:
        fail(f"{name}: the file name's tag is {got}, expected {want}")
    with zipfile.ZipFile(wheel) as z:
        infos = [n for n in z.namelist() if re.fullmatch(r"[^/]+\.dist-info/WHEEL", n)]
        if len(infos) != 1:
            fail(f"{name}: {len(infos)} .dist-info/WHEEL files, expected 1")
        lines = z.read(infos[0]).decode("utf-8").splitlines()
    tags = [line.split(":", 1)[1].strip() for line in lines if line.startswith("Tag:")]
    if tags != [want]:
        fail(f"{name}: {infos[0]} has the Tag lines {tags}, expected only {want}")
    print(f"tag OK: {name} and its WHEEL file carry only {want}")


def needed_versions(path):
    """The symbol versions an ELF file needs from other libraries (its .gnu.version_r section)."""
    out = subprocess.run(["readelf", "-V", "-W", path], check=True, capture_output=True, text=True).stdout
    needs = out.split("Version needs section", 1)
    return re.findall(r"Name: (\S+)", needs[1]) if len(needs) == 2 else []


def check_symbols(paths):
    newest = {}
    for path in paths:
        for version in needed_versions(path):
            family, _, number = version.partition("_")
            if family not in MANYLINUX_2_28 or version in ALSO_ALLOWED:
                continue
            if not re.fullmatch(r"[0-9]+(\.[0-9]+)*", number):
                fail(f"{path}: needs {version}, which manylinux_2_28 does not allow")
            key = tuple(int(x) for x in number.split("."))
            if key > MANYLINUX_2_28[family]:
                limit = ".".join(map(str, MANYLINUX_2_28[family]))
                fail(f"{path}: needs {version}, newer than manylinux_2_28 allows ({family}_{limit})")
            if key > newest.get(family, ()):
                newest[family] = key
    found = ", ".join(f"{f}_{'.'.join(map(str, v))}" for f, v in sorted(newest.items()))
    print(f"symbols OK: {len(paths)} files need at most {found or 'nothing versioned'} (manylinux_2_28)")


def main(argv):
    if len(argv) == 3 and argv[0] == "tag":
        check_tag(argv[1], argv[2])
    elif len(argv) >= 2 and argv[0] == "symbols":
        check_symbols(argv[1:])
    else:
        fail("usage: check_wheel_rocm.py tag WHEEL TAG | symbols FILE...")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
