#!/usr/bin/env python3
"""The wheel's GPU code must be the default build's, instruction for instruction (standard library + cuobjdump).

  sass_manifest.py dump LIB --out MANIFEST [--note TEXT]
  sass_manifest.py check LIB --reference MANIFEST [--build-dir DIR --attempts N -- REBUILD COMMAND...]
  sass_manifest.py objects BUILD_DIR [--arch sm_75]

`dump` writes the SASS (the machine code of each GPU architecture) of every function in an engine library as one hash
per function and architecture: packaging/sass-reference.txt holds the default build's, made from a build that passed
the gate (SASS identical to the reference release on every architecture, seeded audio identical). `check` compares a
library with it and exits 0 only when every function on every architecture is the same, with the same set of
functions.

Why the retry: nvcc's device front end (cicc, which turns CUDA C++ into PTX) is not deterministic for a few kernels.
With the same source, flags and toolkit, a few compiles in 44 on a busy Linux build machine (none in 24 on an idle CI
runner) wrote a slightly different PTX for one of the decoder's vocoder kernels (sm_90 conv1d_tc3_k<64>: one 16-byte
shared-memory load split into a 16-byte and an 8-byte load of the same bytes), and on Windows nearly every compile
varies in a few of the CUDA-core decode kernels. ptxas (PTX to SASS) gives the same SASS for the same PTX every time,
and usually also for these variants, but not always. So a build can differ from the reference only where cicc varied, and compiling
those files again gives the reference SASS. With --build-dir and a rebuild command, `check` finds the object files
that hold the differing functions, deletes them, runs the rebuild command (which recompiles just those files and links
the library again) and checks again, up to N attempts. A function that is new, missing or still different after the
attempts is a real change of the GPU code: the check fails, and the reference has to be made again from a gated build.

LIB is an ELF shared library or object (Linux), or a Windows DLL or object file, whose CUDA fatbinary section
(.nv_fatb) is read directly. Windows names are paired with the reference's across the int64 ABI (int64_t / uint64_t
are long / unsigned long on Linux, mangled l / m, and long long / unsigned long long on Windows, mangled x / y).
Function names are compared with the anonymous-namespace prefix that nvcc derives from the file path normalised.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ANON = re.compile(r"_GLOBAL__N__[0-9a-f]+_\d+_[A-Za-z0-9_]+?_[0-9a-f]{8}")
FUNC = re.compile(r"\s*Function : (\S+)")
HEX = 16  # hash digits kept per function and architecture


def norm(s: str) -> str:
    return ANON.sub("_GLOBAL__N_", s)


def cuobjdump() -> str:
    exe = shutil.which("cuobjdump")
    if not exe:
        for root in (os.environ.get("CUDA_PATH", ""), "/usr/local/cuda-12.8", "/usr/local/cuda"):
            cand = Path(root) / "bin" / ("cuobjdump.exe" if os.name == "nt" else "cuobjdump")
            if root and cand.is_file():
                return str(cand)
        sys.exit("sass_manifest.py: cuobjdump (CUDA 12.8) is not on PATH")
    return exe


def fatbin_sections(data: bytes) -> bytes | None:
    """The .nv_fatb section(s) of a Windows PE image (DLL) or COFF object file (also the /bigobj form, which the
    build's CUDA objects use); None for anything else (ELF is read by cuobjdump as it is)."""
    if data[:2] == b"MZ":
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe:pe + 4] != b"PE\0\0":
            return None
        coff = pe + 4
        nsec = struct.unpack_from("<H", data, coff + 2)[0]
        sec = coff + 20 + struct.unpack_from("<H", data, coff + 16)[0]
    elif data[:2] == b"\x64\x86":  # an x64 COFF object file (IMAGE_FILE_MACHINE_AMD64)
        nsec = struct.unpack_from("<H", data, 2)[0]
        sec = 20 + struct.unpack_from("<H", data, 16)[0]
    elif data[:4] == b"\0\0\xff\xff" and struct.unpack_from("<HH", data, 4) == (2, 0x8664):
        # /bigobj (ANON_OBJECT_HEADER_BIGOBJ, 56 bytes): 32-bit section count at offset 44, sections right after
        nsec = struct.unpack_from("<I", data, 44)[0]
        sec = 56
    else:
        return None
    out = b""
    for i in range(nsec):
        name, vsize, _vaddr, rsize, rptr = struct.unpack_from("<8sIIII", data, sec + 40 * i)
        if name.rstrip(b"\0").startswith(b".nv_fatb"):
            size = min(vsize, rsize) if vsize else rsize  # objects have no virtual size
            out += data[rptr:rptr + size]
    return out


def _dump_target(target: str, arch: str | None, strict: bool) -> dict[str, dict[str, list[str]]]:
    cmd = [cuobjdump(), "-sass"] + (["-arch", arch] if arch else []) + [target]
    res: dict[str, dict[str, list[str]]] = {}
    arch_now = cur_arch = name = None
    h = None

    def finish():
        if name is not None:
            res.setdefault(cur_arch, {}).setdefault(name, []).append(h.hexdigest()[:HEX])

    errf = tempfile.TemporaryFile(mode="w+")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf, text=True, bufsize=1 << 20)
        for line in proc.stdout:
            s = line.strip()
            if s.startswith("/*"):  # an instruction line
                if name is not None:
                    h.update((norm(s) if "_GLOBAL__N__" in s else s).encode() + b"\n")
            elif "Function :" in s:
                m = FUNC.match(line)
                if m:
                    finish()
                    name, cur_arch, h = norm(m.group(1)), arch_now, hashlib.sha256()
            elif s.startswith("arch = "):
                finish()
                name = None
                arch_now = s.split("=", 1)[1].strip()
        finish()
        rc = proc.wait()
        errf.seek(0)
        err = errf.read()
        if rc != 0 and strict:
            sys.exit(f"sass_manifest.py: {' '.join(cmd)} failed: {err.strip()[-400:]}")
    finally:
        errf.close()
    return {a: {k: sorted(v) for k, v in fs.items()} for a, fs in res.items()}


def dump(path: Path, arch: str | None = None, strict: bool = True) -> dict[str, dict[str, list[str]]]:
    """{arch: {function: [hash of its instruction lines]}} (one architecture with `arch`; all of them otherwise, one
    cuobjdump process per architecture in parallel). strict=False: a file without GPU code (for that architecture)
    gives {} instead of an error."""
    data = path.read_bytes()
    fat = fatbin_sections(data)
    del data
    tmp = None
    if fat is not None:
        if not fat:
            return {}
        tmp = tempfile.NamedTemporaryFile(suffix=".fatbin", delete=False)
        tmp.write(fat)
        tmp.close()
        target = tmp.name
    else:
        target = str(path)
    try:
        if arch:
            return _dump_target(target, arch, strict)
        listing = subprocess.run([cuobjdump(), "-lelf", target], capture_output=True, text=True)
        archs = sorted(set(re.findall(r"\.(sm_\d+[a-z]?)\.cubin", listing.stdout)))
        if not archs:
            if strict:
                sys.exit(f"sass_manifest.py: {path}: no GPU code (cuobjdump -lelf: {listing.stderr.strip()[-300:]})")
            return {}
        from concurrent.futures import ProcessPoolExecutor
        out: dict[str, dict[str, list[str]]] = {}
        with ProcessPoolExecutor(max_workers=min(len(archs), os.cpu_count() or 2)) as ex:
            for part in ex.map(_dump_target, [target] * len(archs), archs, [strict] * len(archs)):
                out.update(part)
        missing = [x for x in archs if not out.get(x)]
        if missing and strict:  # cuobjdump -sass needs nvdisasm; without it, it prints no instructions
            sys.exit(f"sass_manifest.py: {path}: cuobjdump printed no SASS for {', '.join(missing)} "
                     "(is nvdisasm from CUDA 12.8 on PATH?)")
        return out
    finally:
        if tmp:
            os.unlink(tmp.name)


def write_manifest(path: Path, d: dict, note: str) -> None:
    archs = sorted(d)
    names = sorted({n for fs in d.values() for n in fs})
    lines = [
        "# SASS reference of the Kitten TTS 2 engine library: the default build (cmake --preset default) of a commit that",
        "# passed the gate. Read by tts-2/packaging/sass_manifest.py check; see that file for how it is made and used.",
        *(["# " + x for x in textwrap.wrap(note, 114)] or ["#"]),
        "# One line per GPU function: its name, then per architecture of the archs line the first 16 hex digits of the",
        "# sha256 of its SASS instruction lines ('-': not in that architecture's code).",
        "archs " + " ".join(archs),
    ]
    for n in names:
        lines.append(" ".join(["+".join(d[a][n]) if n in d[a] else "-" for a in archs] + [n]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> dict[str, dict[str, list[str]]]:
    archs: list[str] = []
    d: dict[str, dict[str, list[str]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("archs "):
            archs = line.split()[1:]
            d = {a: {} for a in archs}
            continue
        *hashes, name = line.split(" ")
        if len(hashes) != len(archs):
            sys.exit(f"{path}: malformed line: {line[:120]}")
        for a, h in zip(archs, hashes):
            if h != "-":
                d[a][name] = h.split("+")
    return d


def int64_key(name: str) -> str:
    return name.replace("x", "l").replace("y", "m")


def pair(name: str, ref_names: dict[str, list[str]]) -> str | None:
    """The reference name for `name`: itself, or the one that differs only in the int64 mangling (Windows x / y for
    Linux l / m)."""
    cands = ref_names.get(int64_key(name), [])
    if name in cands:
        return name
    for cand in cands:
        if all(x == y or (x, y) in (("l", "x"), ("m", "y")) for x, y in zip(cand, name)):
            return cand
    return None


def compare(got: dict, ref: dict):
    """(differing [(arch, lib name)], only in the library [(arch, name)], only in the reference [(arch, name)],
    identical count)."""
    differ, only_lib, only_ref, same = [], [], [], 0
    for arch in sorted(set(got) | set(ref)):
        fg, fr = got.get(arch, {}), ref.get(arch, {})
        index: dict[str, list[str]] = {}
        for n in fr:
            index.setdefault(int64_key(n), []).append(n)
        matched = set()
        for n, h in fg.items():
            r = pair(n, index)
            if r is None:
                only_lib.append((arch, n))
                continue
            matched.add(r)
            if h == fr[r]:
                same += 1
            else:
                differ.append((arch, n))
        only_ref += [(arch, n) for n in fr if n not in matched]
    return differ, only_lib, only_ref, same


def report(lib: Path, got: dict, ref: dict) -> tuple[bool, list]:
    differ, only_lib, only_ref, same = compare(got, ref)
    total = sum(len(f) for f in ref.values())
    print(f"{lib.name}: {same} of {total} functions identical to the reference on {len(ref)} architectures; "
          f"differ {len(differ)}, only in the library {len(only_lib)}, only in the reference {len(only_ref)}")
    for label, xs in (("differs", differ), ("only in the library", only_lib), ("only in the reference", only_ref)):
        for arch, n in xs[:20]:
            print(f"  {label}: {arch} {n}")
    return not (differ or only_lib or only_ref), (differ if not (only_lib or only_ref) else None)


def objects_with(build: Path, differ: list) -> list[Path]:
    """The CUDA object files under `build` that hold the differing functions."""
    objs = sorted(p for p in build.rglob("*") if p.is_file() and p.name.endswith((".cu.o", ".cu.obj")))
    want: dict[str, set[str]] = {}
    for arch, n in differ:
        want.setdefault(arch, set()).add(n)
    hits = []
    for o in objs:
        for arch, names in want.items():
            fs = dump(o, arch, strict=False).get(arch, {})
            if any(n in fs for n in names):
                hits.append(o)
                break
    return hits


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("lib", type=Path)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--note", default="")
    c = sub.add_parser("check")
    c.add_argument("lib", type=Path)
    c.add_argument("--reference", type=Path, required=True)
    c.add_argument("--build-dir", type=Path)
    c.add_argument("--attempts", type=int, default=6)
    ob = sub.add_parser("objects", help="list the CUDA object files of a build that hold GPU code (a self-test of the "
                                         "rebuild step: they must be readable on this system)")
    ob.add_argument("build_dir", type=Path)
    ob.add_argument("--arch", default="sm_75")
    argv = sys.argv[1:]
    rebuild = argv[argv.index("--") + 1:] if "--" in argv else []  # the command that rebuilds the library
    a = p.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    if a.cmd == "dump":
        got = dump(a.lib)
        if not got:
            sys.exit(f"{a.lib}: no GPU code found")
        write_manifest(a.out, got, a.note)
        print(f"{a.out}: {sum(len(f) for f in got.values())} functions on {len(got)} architectures")
        return 0
    if a.cmd == "objects":
        objs = sorted(p for p in a.build_dir.rglob("*") if p.is_file() and p.name.endswith((".cu.o", ".cu.obj")))
        counts = {o: len(dump(o, a.arch, strict=False).get(a.arch, {})) for o in objs}
        for o, n in counts.items():
            if n:
                print(f"{n:5d} {a.arch} functions  {o}")
        total = sum(counts.values())
        print(f"{len(objs)} CUDA object files, {sum(1 for n in counts.values() if n)} with {a.arch} code, {total} functions")
        return 0 if total else 1
    ref = read_manifest(a.reference)
    import time
    t0 = time.time()
    for attempt in range(1, a.attempts + 1):
        ok, differ = report(a.lib, dump(a.lib), ref)
        print(f"  (attempt {attempt}, {time.time() - t0:.0f} s so far)", flush=True)
        if ok:
            print(f"SASS check: IDENTICAL to {a.reference.name}" + (f" (after {attempt - 1} rebuilds)" if attempt > 1 else ""))
            return 0
        if differ is None:
            print("SASS check: the set of GPU functions differs from the reference: the GPU code changed. Make the "
                  "reference again from a gated default build (see the top of sass_manifest.py).")
            return 1
        if not (a.build_dir and rebuild) or attempt == a.attempts:
            break
        objs = objects_with(a.build_dir, differ)
        if not objs:
            print("SASS check: no object file in the build holds the differing functions; cannot rebuild them")
            return 1
        for o in objs:
            print(f"  compiling again: {o}")
            o.unlink()
        print(f"SASS check: attempt {attempt + 1}: {' '.join(rebuild)}", flush=True)
        if subprocess.run(rebuild).returncode != 0:
            print("SASS check: the rebuild failed")
            return 1
    print(f"SASS check: DIFFERENT from {a.reference.name}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
