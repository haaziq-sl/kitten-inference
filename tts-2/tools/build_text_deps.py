#!/usr/bin/env python3
"""Build the pinned native normalization dependencies without root privileges."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

SOURCES = (
    ("openfst-1.8.3", "https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.3.tar.gz?download=1",
     "077714159d5cf3e38a80b6c6656d3ccc2c8b8b6c50bb41bb65c5fec10796bf53"),
    ("pcre2-10.42", "https://github.com/PCRE2Project/pcre2/releases/download/pcre2-10.42/pcre2-10.42.tar.gz",
     "c33b418e3b936ee3153de2c61cc638e7e4fe3156022a5c77d0711bcbb9d64f1f"),
)


# Windows (MSVC): OpenFst 1.8.3 has no CMake build, so this small project builds its library sources the way
# `./configure --enable-static --disable-shared --disable-bin` does on Linux. FST_NO_DYNAMIC_LINKING drops the
# dlopen() of extension FST types (the normaliser reads standard vector FSTs only); the engine's CMake defines it
# for every OpenFst user on Windows.
OPENFST_CMAKE = """cmake_minimum_required(VERSION 3.20)
project(openfst_kt2n CXX)
file(GLOB FST_SOURCES "${FST_SRC}/src/lib/*.cc")
add_library(fst STATIC ${FST_SOURCES})
target_include_directories(fst PRIVATE "${FST_SRC}/src/include")
target_compile_features(fst PRIVATE cxx_std_17)
target_compile_definitions(fst PRIVATE FST_NO_DYNAMIC_LINKING NOMINMAX _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
target_compile_options(fst PRIVATE /utf-8 /bigobj /W0 /fp:precise)
install(TARGETS fst ARCHIVE DESTINATION lib)
install(DIRECTORY "${FST_SRC}/src/include/fst" DESTINATION include)
"""


# MSVC C runtime of the libraries: the DLL one (/MD, the engine's windows preset) or, with --msvc-static-runtime, the
# static one (/MT, the wheel-windows preset).
MSVC_RUNTIME = "MultiThreadedDLL"


def cmake_build(source, build, prefix, args, jobs):
    """Configure, build and install one CMake project with Ninja + the MSVC from the developer prompt."""
    subprocess.run(["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_C_COMPILER=cl", "-DCMAKE_CXX_COMPILER=cl", f"-DCMAKE_INSTALL_PREFIX={prefix}",
                    f"-DCMAKE_MSVC_RUNTIME_LIBRARY={MSVC_RUNTIME}", "-DCMAKE_POLICY_DEFAULT_CMP0091=NEW",
                    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5", *args],
                   check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", str(jobs)], check=True)
    subprocess.run(["cmake", "--install", str(build)], check=True)


def build_windows(name, source, cache, prefix, jobs):
    """MSVC static libraries: lib/fst.lib and lib/pcre2-8.lib (the names cmake/Kt2nText.cmake expects)."""
    if name.startswith("openfst"):
        if (prefix / "lib" / "fst.lib").exists():
            return
        project = cache / "openfst-cmake"
        project.mkdir(parents=True, exist_ok=True)
        (project / "CMakeLists.txt").write_text(OPENFST_CMAKE, encoding="utf-8", newline="\n")
        cmake_build(project, cache / "build-openfst", prefix, [f"-DFST_SRC={source.as_posix()}"], jobs)
    else:
        target = prefix / "lib" / "pcre2-8.lib"
        if target.exists():
            return
        cmake_build(source, cache / "build-pcre2", prefix,
                    ["-DBUILD_SHARED_LIBS=OFF", "-DBUILD_STATIC_LIBS=ON", "-DPCRE2_BUILD_PCRE2_8=ON",
                     "-DPCRE2_SUPPORT_JIT=ON", "-DPCRE2_SUPPORT_UNICODE=ON", "-DPCRE2_NEWLINE=LF",
                     "-DPCRE2_BUILD_PCRE2GREP=OFF", "-DPCRE2_BUILD_TESTS=OFF",
                     "-DPCRE2_SUPPORT_LIBBZ2=OFF", "-DPCRE2_SUPPORT_LIBZ=OFF", "-DPCRE2_SUPPORT_LIBEDIT=OFF",
                     "-DPCRE2_SUPPORT_LIBREADLINE=OFF"], jobs)
        built = sorted((prefix / "lib").glob("pcre2-8*.lib"))
        if not built:
            raise RuntimeError("PCRE2 build installed no pcre2-8 library")
        shutil.copyfile(built[0], target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--msvc-static-runtime", action="store_true",
                        help="Windows: build with the static C runtime (/MT), for the wheel-windows preset")
    args = parser.parse_args()
    if args.msvc_static_runtime:
        global MSVC_RUNTIME
        MSVC_RUNTIME = "MultiThreaded"
    prefix = args.prefix.resolve()
    cache = (args.cache or prefix / "sources").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CFLAGS="-O3 -fPIC", CXXFLAGS="-O3 -fPIC -std=c++17")
    for name, url, digest in SOURCES:
        archive = cache / (name + ".tar.gz")
        if not archive.exists():
            if args.offline:
                raise RuntimeError(f"missing cached source: {archive}")
            temporary = archive.with_suffix(".part")
            request = urllib.request.Request(url, headers={"User-Agent": "KittenTTS-build/1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                temporary.write_bytes(response.read())
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
                temporary.unlink()
                raise RuntimeError(f"source checksum mismatch: {name}")
            temporary.replace(archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"cached source checksum mismatch: {archive}")
        source = cache / name
        if not source.exists():
            with tarfile.open(archive) as tar:
                for member in tar.getmembers():
                    if not (cache / member.name).resolve().is_relative_to(cache):
                        raise RuntimeError("unsafe source archive path")
                tar.extractall(cache)
        if name.startswith("openfst"):
            # OpenFst 1.8.3 copy assignment is diagnosed by GCC 15 before instantiation.
            header = source / "src/include/fst/fst.h"
            text = header.read_text(encoding="utf-8")
            for symbol in ("isymbols_", "osymbols_"):
                old = f"{symbol} = impl.{symbol} ? impl.{symbol}->Copy() : nullptr;"
                new = f"{symbol}.reset(impl.{symbol} ? impl.{symbol}->Copy() : nullptr);"
                text = text.replace(old, new)
            header.write_text(text, encoding="utf-8", newline="\n")
            table = source / "src/include/fst/bi-table.h"
            table.write_text(table.read_text(encoding="utf-8").replace("selector_(table.s_)", "selector_(table.selector_)"), encoding="utf-8", newline="\n")
            # Windows: std::shared_ptr::unique() was removed in C++20, and MSVC's library drops it (the engine
            # includes these headers as C++20); use_count() == 1 is the same test.
            for relative, old, new in (("symbol-table.h", "impl_.unique()", "impl_.use_count() == 1"),
                                       ("edit-fst.h", "data_.unique()", "data_.use_count() == 1"),
                                       ("impl-to-fst.h", "impl_.unique()", "impl_.use_count() == 1")):
                header = source / "src/include/fst" / relative
                if os.name == "nt":
                    header.write_text(header.read_text(encoding="utf-8").replace(old, new), encoding="utf-8", newline="\n")
        if os.name == "nt":
            build_windows(name, source, cache, prefix, args.jobs)
            continue
        library = prefix / "lib" / ("libfst.a" if name.startswith("openfst") else "libpcre2-8.a")
        if library.exists():
            continue
        flags = [f"--prefix={prefix}", "--enable-static", "--disable-shared"]
        if name.startswith("openfst"):
            flags += ["--disable-bin"]
        else:
            flags += ["--enable-jit", "--disable-pcre2grep", "--disable-pcre2test"]
        subprocess.run(["./configure", *flags], cwd=source, env=env, check=True)
        subprocess.run(["make", f"-j{args.jobs}"], cwd=source, env=env, check=True)
        subprocess.run(["make", "install"], cwd=source, env=env, check=True)
    print(f"Configure with -DKT2N_TEXT_DEPS={prefix}")


if __name__ == "__main__":
    main()
