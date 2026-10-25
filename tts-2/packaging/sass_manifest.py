#!/usr/bin/env python3
"""The wheel's GPU code must be the default build's, instruction for instruction (standard library + cuobjdump).

  sass_manifest.py dump LIB --out MANIFEST [--note TEXT]
  sass_manifest.py check LIB --reference MANIFEST [--build-dir DIR --attempts N [--object-attempts M]
                                                    -- REBUILD COMMAND...]
  sass_manifest.py objects BUILD_DIR [--arch sm_75]

`dump` writes the SASS (the machine code of each GPU architecture) of every function in an engine library as one hash
per function and architecture: packaging/sass-reference.txt holds the default build's, made from a build whose GPU code
was checked (packaging/README.md: SASS identical to the previous reference on every architecture, or the changed
functions reviewed, and seeded audio unchanged). `check` compares a library with it and exits 0 only when every
function on every architecture is the same, with the same set of functions.

Why the retry: nvcc's device front end (cicc, which turns CUDA C++ into PTX) is not deterministic for a few kernels.
With the same source, flags and toolkit, a few compiles in 44 on a busy Linux build machine (none in 24 on an idle CI
runner) wrote a slightly different PTX for one of the decoder's vocoder kernels (sm_90 conv1d_tc3_k<64>: one 16-byte
shared-memory load split into a 16-byte and an 8-byte load of the same bytes), and on Windows nearly every compile
varies in a few of the CUDA-core decode kernels. ptxas (PTX to SASS) gives the same SASS for the same PTX every time,
and usually also for these variants, but not always. So a build can differ from the reference only where cicc varied, and compiling
those files again gives the reference SASS. With --build-dir and a rebuild command, `check` finds the object files
that hold the differing functions, deletes them, runs the rebuild command (which recompiles just those files and links
the library again) and checks again, up to N attempts. A function that is new, missing or still different after the
attempts is a real change of the GPU code: the check fails, and the reference has to be made again from a checked
build (packaging/README.md).

--object-attempts M (rebuild commands of the form `cmake --build ...` only): before each relink, compile just those
object files again (the rebuild command with --target OBJECT) and compare their own GPU code with the reference, up to
M times, until it is the reference's; then link and check the whole library. On Windows nearly every compile of the
CUDA-core decode kernels (src/cuda/legacy/tern_dec_cc.cu, 39 kernels for sm_61 and sm_75) varies in a few of them, so
a compile whose 78 kernels all match is the exception; an object compile and check takes about 45 s there, a relink
and full check about 5 minutes. The last line gives the number of object compiles and how many matched.

LIB is an ELF shared library or object (Linux), or a Windows DLL or object file, whose CUDA fatbinary section
(.nv_fatb) is read directly. Windows names are paired with the reference's across the int64 ABI (int64_t / uint64_t
are long / unsigned long on Linux, mangled l / m, and long long / unsigned long long on Windows, mangled x / y).
Function names are compared with the anonymous-namespace prefix that nvcc derives from the file path normalised.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
    """The CUDA fatbinary section(s) of a Windows PE image (DLL: `.nv_fatb`, names cut to 8 characters) or COFF object
    file (`.nv_fatbin`, a long name kept in the string table; also the /bigobj form, which the build's CUDA objects
    use); None for anything else (ELF is read by cuobjdump as it is)."""
    strtab = None  # COFF objects: where the string table starts (long section names are "/<offset>" into it)
    if data[:2] == b"MZ":
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe:pe + 4] != b"PE\0\0":
            return None
        coff = pe + 4
        nsec = struct.unpack_from("<H", data, coff + 2)[0]
        sec = coff + 20 + struct.unpack_from("<H", data, coff + 16)[0]
    elif data[:2] == b"\x64\x86":  # an x64 COFF object file (IMAGE_FILE_MACHINE_AMD64); symbols are 18 bytes
        nsec = struct.unpack_from("<H", data, 2)[0]
        sec = 20 + struct.unpack_from("<H", data, 16)[0]
        symptr, nsym = struct.unpack_from("<II", data, 8)
        strtab = symptr + 18 * nsym
    elif data[:4] == b"\0\0\xff\xff" and struct.unpack_from("<HH", data, 4) == (2, 0x8664):
        # /bigobj (ANON_OBJECT_HEADER_BIGOBJ, 56 bytes): 32-bit section count at offset 44, symbol table pointer and
        # count at 48 and 52 (20-byte symbols), section headers right after the header
        nsec = struct.unpack_from("<I", data, 44)[0]
        sec = 56
        symptr, nsym = struct.unpack_from("<II", data, 48)
        strtab = symptr + 20 * nsym
    else:
        return None
    out = b""
    for i in range(nsec):
        name, vsize, _vaddr, rsize, rptr = struct.unpack_from("<8sIIII", data, sec + 40 * i)
        name = name.rstrip(b"\0")
        if name.startswith(b"/") and strtab is not None and name[1:].isdigit():
            at = strtab + int(name[1:])
            name = data[at:data.index(b"\0", at)]
        if name.startswith(b".nv_fatb"):
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
        "# SASS reference of the Kitten TTS 2 engine library: the default build (cmake --preset default) of a commit whose",
        "# GPU code was checked. Read by tts-2/packaging/sass_manifest.py check; see that file and packaging/README.md.",
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


def function_identifier(name: str) -> str | None:
    """The function's own identifier from an Itanium-mangled name (with the anonymous-namespace prefix normalised):
    _ZN12kt2n_tern_cc47_GLOBAL__N_13dec_cc_kernelILi8E... -> dec_cc_kernel."""
    if not name.startswith("_Z"):
        return None
    i, last = (3 if name.startswith("_ZN") else 2), None
    while i < len(name) and name[i].isdigit():
        j = i
        while j < len(name) and name[j].isdigit():
            j += 1
        if name.startswith("_GLOBAL__N_", j):  # its length is the original (longer) prefix's
            i = j + len("_GLOBAL__N_")
            continue
        n = int(name[i:j])
        last, i = name[j:j + n], j + n
    return last


def objects_of_sources(build: Path, differ: list) -> list[Path]:
    """Fallback: the objects of the .cu sources (compile_commands.json) that mention the differing functions' names."""
    try:
        entries = json.loads((build / "compile_commands.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    idents = {x for x in (function_identifier(n) for _, n in differ) if x}
    hits = []
    for e in entries:
        src = Path(e["file"])
        if src.suffix != ".cu" or not src.is_file():
            continue
        text = src.read_text(encoding="utf-8", errors="replace")
        if not any(re.search(r"\b" + re.escape(x) + r"\b", text) for x in idents):
            continue
        out = e.get("output")
        if not out:
            m = re.search(r'\s-o\s+("[^"]+"|\S+)', e.get("command", ""))
            out = m.group(1).strip('"') if m else None
        if out:  # "output" is relative to the build's top directory, "-o" to the entry's directory
            for obj in ([Path(out)] if Path(out).is_absolute() else [build / out, Path(e["directory"]) / out]):
                if obj.is_file():
                    hits.append(obj)
                    break
    return hits


def objects_with(build: Path, differ: list) -> list[Path]:
    """The CUDA object files under `build` that hold the differing functions (read from their GPU code; if that finds
    none, the objects of the sources that name them)."""
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
    if not hits:
        hits = objects_of_sources(build, differ)
        if hits:
            print("  (objects found from the sources that name the functions)")
    return hits


def is_cmake_build(command: list) -> bool:
    return bool(command) and Path(command[0]).name.lower() in ("cmake", "cmake.exe") and "--build" in command


def recompile_objects(build: Path, rebuild: list, objs: list, ref: dict, tries: int, stats: dict) -> bool:
    """Compiles `objs` again, and only them (`rebuild` with --target), until the GPU code of each is the reference's,
    up to `tries` compiles; the caller then links the library again and checks all of it. False when a compile
    fails."""
    targets = [str(o.relative_to(build)) for o in objs]  # the build tool's (Ninja's) names of the object files
    for k in range(1, tries + 1):
        for o in objs:
            o.unlink(missing_ok=True)
        if subprocess.run(rebuild + ["--target", *targets]).returncode != 0:
            print("SASS check: compiling the object files again failed")
            return False
        differ, only_lib = [], []
        for o in objs:
            d, ol, _, _ = compare(dump(o, strict=False), ref)  # the rest of the reference is in other objects
            differ += d
            only_lib += ol
        stats["compiles"] += 1
        if only_lib:  # not in the reference at all: compiling again cannot help; the full check reports it
            print(f"  object compile {k}: {len(only_lib)} functions are not in the reference")
            return True
        if not differ:
            stats["identical"] += 1
            print(f"  object compile {k}: identical to the reference", flush=True)
            return True
        per_arch = {}
        for arch, _ in differ:
            per_arch[arch] = per_arch.get(arch, 0) + 1
        print(f"  object compile {k}: {len(differ)} functions differ ("
              + ", ".join(f"{arch} {n}" for arch, n in sorted(per_arch.items())) + ")", flush=True)
    return True


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
    c.add_argument("--object-attempts", type=int, default=1,
                   help="compiles of the differing object files, each checked on its own, before each relink")
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
    stats = {"compiles": 0, "identical": 0}

    def compiles() -> str:
        return (f"; object compiles {stats['compiles']}, identical to the reference {stats['identical']}"
                if stats["compiles"] else "")

    for attempt in range(1, a.attempts + 1):
        ok, differ = report(a.lib, dump(a.lib), ref)
        print(f"  (attempt {attempt}, {time.time() - t0:.0f} s so far)", flush=True)
        if ok:
            print(f"SASS check: IDENTICAL to {a.reference.name}" + (f" (after {attempt - 1} rebuilds{compiles()})"
                                                                   if attempt > 1 else ""))
            return 0
        if differ is None:
            print("SASS check: the set of GPU functions differs from the reference: the GPU code changed. Make the "
                  "reference again from a checked default build (packaging/README.md).")
            return 1
        if not (a.build_dir and rebuild) or attempt == a.attempts:
            break
        objs = objects_with(a.build_dir, differ)
        if not objs:
            print("SASS check: no object file in the build holds the differing functions; cannot rebuild them")
            return 1
        for o in objs:
            print(f"  compiling again: {o}")
        if a.object_attempts > 1 and is_cmake_build(rebuild):
            if not recompile_objects(a.build_dir, rebuild, objs, ref, a.object_attempts, stats):
                return 1
        else:
            for o in objs:
                o.unlink()
        print(f"SASS check: attempt {attempt + 1}: {' '.join(rebuild)}", flush=True)
        if subprocess.run(rebuild).returncode != 0:
            print("SASS check: the rebuild failed")
            return 1
    print(f"SASS check: DIFFERENT from {a.reference.name}{compiles()}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
