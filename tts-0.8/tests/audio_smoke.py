import argparse
import importlib
import json
import platform
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from kitten_inference.tts08 import KittenTTS, _engine


SAMPLE_RATE = 24000
TEXT = "Kitten TTS is generating a short audio sample for release testing."
QUICK_CASE = {
    "model": "KittenML/kitten-tts-nano-0.8",
    "voice": "Bella",
    "speed": 1.0,
    "filename": "quick-nano-bella.wav",
}
CASES = [
    {
        "model": "KittenML/kitten-tts-nano-0.8",
        "voice": "Bella",
        "speed": 1.0,
        "filename": "nano-bella-normal.wav",
    },
    {
        "model": "KittenML/kitten-tts-nano-0.8",
        "voice": "Jasper",
        "speed": 1.25,
        "filename": "nano-jasper-fast.wav",
    },
    {
        "model": "KittenML/kitten-tts-nano-0.8-int8",
        "voice": "Luna",
        "speed": 0.85,
        "filename": "nano-int8-luna-slow.wav",
    },
]
SPEED_CASES = [
    {
        "model": "KittenML/kitten-tts-nano-0.8",
        "voice": "Kiki",
        "speed": 0.75,
        "filename": "speed-kiki-slow.wav",
    },
    {
        "model": "KittenML/kitten-tts-nano-0.8",
        "voice": "Kiki",
        "speed": 1.25,
        "filename": "speed-kiki-fast.wav",
    },
]
# micro and mini have wider layers than nano (for example a 256-wide LSTM instead of 64): run each once.
# nano int8 is here too, so that every int8 model runs on Metal when the backend is set to metal.
MODEL_CASES = [
    {"model": "KittenML/kitten-tts-nano-0.8-int8", "voice": "Luna", "speed": 1.0, "filename": "nano-int8-luna.wav"},
    {"model": "KittenML/kitten-tts-mini-0.8", "voice": "Jasper", "speed": 1.0, "filename": "mini-jasper.wav"},
    {"model": "KittenML/kitten-tts-mini-0.8-int8", "voice": "Bella", "speed": 1.0, "filename": "mini-int8-bella.wav"},
    {"model": "KittenML/kitten-tts-micro-0.8", "voice": "Hugo", "speed": 1.0, "filename": "micro-hugo.wav"},
    {"model": "KittenML/kitten-tts-micro-0.8-int8", "voice": "Rosie", "speed": 1.0, "filename": "micro-int8-rosie.wav"},
]
NANO = "KittenML/kitten-tts-nano-0.8"
NANO_INT8 = "KittenML/kitten-tts-nano-0.8-int8"
ALL_MODELS = [f"KittenML/kitten-tts-{size}-0.8{q}" for size in ("nano", "micro", "mini") for q in ("", "-int8")]
# One model per thread, all generating at once. None is the default backend (Metal on Apple Silicon).
THREAD_PLAN = [
    (NANO, None, "Bella"),
    (NANO, None, "Jasper"),
    (NANO_INT8, "cpu", "Luna"),
    (NANO, "cpu", "Kiki"),
]
# A Latin-1, a Latin-2 and a CJK letter: no Windows ANSI code page has all three.
UNICODE_DIR_PREFIX = "kitten-é-Ł-猫-"
# Metal output is compared loosely: the same length, the difference at most 1% of the signal (RMS) and no sample off
# by more than 0.05. A race between calls gave a wrong length or differences of 0.9. Within one process Metal output
# now repeats bit for bit (see repeat_check); the loose comparison stays for outputs of separate model instances.
METAL_MAX_DIFF = 0.05
METAL_RMS_RATIO = 0.01
# The repeat check: the medium text of the TTS 0.8 benchmark (three sentences, so three engine passes of different
# lengths per call) on micro, whose wider layers showed the Metal race most often.
REPEAT_MODEL = "KittenML/kitten-tts-micro-0.8"
REPEAT_TEXT = (
    "This small model turns written text into natural speech on an ordinary computer. It runs quickly without a "
    "graphics card, so you can add a voice to small devices, desktop apps and servers. Try a few voices and pick "
    "your favourite."
)


def validate_audio(audio, label):
    if audio.ndim != 1:
        raise AssertionError(f"{label}: expected mono audio, got shape {audio.shape}")
    if len(audio) < SAMPLE_RATE // 10:
        raise AssertionError(f"{label}: audio is unexpectedly short: {len(audio)} samples")
    if not np.isfinite(audio).all():
        raise AssertionError(f"{label}: audio contains non-finite samples")
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-5:
        raise AssertionError(f"{label}: audio is silent")
    return peak


def as_audio(values):
    return np.asarray(values, dtype=np.float32).reshape(-1)


def synthesize(model_cache, out_dir, cache_dir, case, backend=None):
    model_name = case["model"]
    cache_key = (model_name, backend or "default")
    model = model_cache.get(cache_key)
    if model is None:
        model = KittenTTS(model_name, cache_dir=str(cache_dir), backend=backend)
        model_cache[cache_key] = model

    label = f"{model_name} {case['voice']} speed={case['speed']} backend={backend or 'default'}"
    start = time.perf_counter()
    audio = as_audio(model.generate(TEXT, voice=case["voice"], speed=case["speed"]))
    gen_seconds = time.perf_counter() - start
    peak = validate_audio(audio, label)

    path = out_dir / case["filename"]
    sf.write(path, audio, SAMPLE_RATE)
    return {
        "model": model_name,
        "backend": backend or "default",
        "active_backend": _engine.active_backend(),
        "voice": case["voice"],
        "speed": case["speed"],
        "path": str(path),
        "samples": int(len(audio)),
        "seconds": len(audio) / SAMPLE_RATE,
        "peak": peak,
        "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
        "gen_seconds": round(gen_seconds, 4),
    }


def compare(results, other_dir):
    """Same cases on another backend: durations within 5% and loudness within 2x."""
    summary = json.loads((Path(other_dir) / "summary.json").read_text(encoding="utf-8"))
    other = {Path(r["path"]).name: r for r in summary}
    for r in results:
        o = other[Path(r["path"]).name]
        label = f"{Path(r['path']).name}: {r['active_backend']} vs {o['active_backend']}"
        r["samples_vs_other_pct"] = round(100.0 * (r["samples"] - o["samples"]) / o["samples"], 2)
        if abs(r["samples"] - o["samples"]) > 0.05 * o["samples"]:
            raise AssertionError(f"{label}: samples {r['samples']} vs {o['samples']}")
        if not 0.5 <= r["rms"] / o["rms"] <= 2.0:
            raise AssertionError(f"{label}: rms {r['rms']:.4f} vs {o['rms']:.4f}")


def difference(a, b):
    """Largest sample difference, or None when the lengths differ."""
    if a.shape != b.shape:
        return None
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def check_same(label, got, want, exact):
    """CPU output must repeat exactly; Metal output must keep its length and stay close (see METAL_MAX_DIFF)."""
    diff = difference(got, want)
    if diff is not None and not exact and diff > 0.0:
        rms_ratio = float(np.sqrt(np.mean(np.square(got - want, dtype=np.float64))
                                  / np.mean(np.square(want, dtype=np.float64))))
        ok = diff <= METAL_MAX_DIFF and rms_ratio <= METAL_RMS_RATIO
    else:
        ok = diff is not None and diff == 0.0
    if not ok:
        raise AssertionError(
            f"{label}: {len(got)} samples vs {len(want)} expected, largest difference {diff}"
        )
    return diff


def resolved_backend(backend, model_name=NANO):
    """The backend a model gets: the one asked for, else Metal for fp32 models on Apple Silicon, else the CPU."""
    if backend:
        return backend
    apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
    return "metal" if apple_silicon and not model_name.endswith("-int8") else "cpu"


def auto_backend_check(cache_dir, runs=3):
    """backend=None for every model: check which backend it gets, and time warm calls (median of `runs`)."""
    results = []
    for name in ALL_MODELS:
        model = KittenTTS(name, cache_dir=str(cache_dir))
        want = resolved_backend(None, name)
        if model.model.backend != want:
            raise AssertionError(f"{name}: default backend {model.model.backend!r}, expected {want!r}")
        model.generate(TEXT, voice="Jasper")  # warm-up
        times, used = [], set()
        for _ in range(runs):
            start = time.perf_counter()
            audio = as_audio(model.generate(TEXT, voice="Jasper"))
            times.append(time.perf_counter() - start)
            used.add(_engine.active_backend())
        if used != {want}:
            raise AssertionError(f"{name}: ran on {sorted(used)}, expected {want!r}")
        validate_audio(audio, name)
        median = sorted(times)[len(times) // 2]
        results.append({"model": name, "backend": want, "samples": int(len(audio)),
                        "median_seconds": round(median, 4), "rtf": round(median / (len(audio) / SAMPLE_RATE), 4)})
    return results


def engine_load_with_default_c_locale(root, spec):
    """Windows: the engine itself must open non-ASCII paths in the system's code page.

    Through KittenTTS this is hidden: espeak-ng, which phonemizer starts before the engine loads, switches the
    process's C locale to UTF-8, and then even narrow fopen() reads UTF-8 paths. So load the same files with the
    engine directly while the C runtime uses the default locale (for example English_United States.1252).
    """
    import ctypes

    ucrt = ctypes.CDLL("ucrtbase")
    ucrt.setlocale.restype = ctypes.c_char_p
    ucrt.setlocale.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lc_ctype = 2
    after_phonemizer = ucrt.setlocale(lc_ctype, None)
    ucrt.setlocale(lc_ctype, b"")
    try:
        default = ucrt.setlocale(lc_ctype, None)
        _engine.InferenceModel(str(root / spec["arch_file"]), str(root / spec["weights_file"]))
    finally:
        ucrt.setlocale(lc_ctype, after_phonemizer)
    return {
        "windows_ansi_code_page": ctypes.windll.kernel32.GetACP(),
        "c_locale_after_phonemizer": after_phonemizer.decode(),
        "c_locale_for_direct_engine_load": default.decode(),
    }


def unicode_path_check(cache_dir, backend):
    """Load nano from a folder whose path has non-ASCII characters; it must speak like the built-in nano."""
    gm = importlib.import_module("kitten_inference.tts08.get_model")
    spec = gm.BUILTIN_NATIVE_MODELS[NANO.lower()]
    root = Path(tempfile.mkdtemp(prefix=UNICODE_DIR_PREFIX))
    try:
        shutil.copy(gm._builtin_arch_path(spec["arch_file"]), root / spec["arch_file"])
        weights = gm._download_file(
            gm.NATIVE_WEIGHTS_REPO, spec["weights_file"], str(cache_dir), gm.NATIVE_WEIGHTS_REVISION
        )
        shutil.copy(weights, root / spec["weights_file"])
        voice_files = {}
        for voice in gm.VOICE_NAMES:
            src = gm._download_file(
                gm.NATIVE_WEIGHTS_REPO, f"{spec['voice_dir']}/{voice}.bin", str(cache_dir), gm.NATIVE_WEIGHTS_REVISION
            )
            shutil.copy(src, root / f"{voice}.bin")
            voice_files[voice] = f"{voice}.bin"
        config = {
            "type": "CPP1",
            "arch_file": spec["arch_file"],
            "weights_file": spec["weights_file"],
            "voice_files": voice_files,
            "voice_aliases": gm.DEFAULT_VOICE_ALIASES,
            "speed_priors": spec["speed_priors"],
            "style_index": "text_length",
        }
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")

        local = KittenTTS(str(root), backend=backend)
        got = as_audio(local.generate(TEXT, voice="Bella"))
        builtin = KittenTTS(NANO, cache_dir=str(cache_dir), backend=backend)
        want = as_audio(builtin.generate(TEXT, voice="Bella"))
        validate_audio(got, f"nano from {root}")
        diff = check_same(f"nano from {root}", got, want, exact=resolved_backend(backend, NANO) == "cpu")
        result = {"check": "unicode path", "path": str(root), "samples": int(len(got)), "max_diff": diff}
        if sys.platform == "win32":
            result.update(engine_load_with_default_c_locale(root, spec))
        return [result]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def backend_mix_check(cache_dir):
    """Models on two backends in one process: a model made later must not move an earlier one to its backend."""
    results = []
    cpu_model = KittenTTS(NANO_INT8, cache_dir=str(cache_dir), backend="cpu")
    cpu_ref = as_audio(cpu_model.generate(TEXT, voice="Luna"))
    default_model = KittenTTS(NANO, cache_dir=str(cache_dir))
    default_ref = as_audio(default_model.generate(TEXT, voice="Bella"))
    default_exact = resolved_backend(None, NANO) == "cpu"

    again = as_audio(cpu_model.generate(TEXT, voice="Luna"))
    results.append({
        "check": "cpu model after a default-backend model was made",
        "samples": int(len(again)),
        "max_diff": check_same("cpu model after a default-backend model", again, cpu_ref, exact=True),
    })

    KittenTTS(NANO, cache_dir=str(cache_dir), backend="cpu")
    again = as_audio(default_model.generate(TEXT, voice="Bella"))
    results.append({
        "check": f"default-backend ({resolved_backend(None, NANO)}) model after a cpu model was made",
        "samples": int(len(again)),
        "max_diff": check_same("default-backend model after a cpu model", again, default_ref, exact=default_exact),
    })
    return results


def threads_check(cache_dir, rounds=2, timeout=900):
    """One model per thread, all generating at once; each output must match the same model run alone."""
    refs = {}
    for model_name, backend, voice in THREAD_PLAN:
        key = (model_name, backend, voice)
        if key not in refs:
            model = KittenTTS(model_name, cache_dir=str(cache_dir), backend=backend)
            refs[key] = as_audio(model.generate(TEXT, voice=voice))

    start = threading.Barrier(len(THREAD_PLAN))
    outcomes = [None] * len(THREAD_PLAN)

    def worker(i, model_name, backend, voice):
        try:
            model = KittenTTS(model_name, cache_dir=str(cache_dir), backend=backend)
            start.wait(timeout=timeout)
            diffs = []
            for r in range(rounds):
                audio = as_audio(model.generate(TEXT, voice=voice))
                diffs.append(check_same(
                    f"thread {i} ({model_name}, {resolved_backend(backend, model_name)}) round {r}",
                    audio, refs[(model_name, backend, voice)], exact=resolved_backend(backend, model_name) == "cpu",
                ))
            outcomes[i] = {"thread": i, "model": model_name, "backend": resolved_backend(backend, model_name),
                           "voice": voice, "max_diff": max(diffs)}
        except BaseException as exc:  # reported by the main thread
            outcomes[i] = {"thread": i, "error": f"{type(exc).__name__}: {exc}"}
            start.abort()  # the other threads stop waiting at the barrier

    threads = [threading.Thread(target=worker, args=(i,) + plan, daemon=True) for i, plan in enumerate(THREAD_PLAN)]
    began = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
    hung = [i for i, t in enumerate(threads) if t.is_alive()]
    if hung:
        raise AssertionError(f"threads {hung} did not finish within {timeout} s")
    errors = [o["error"] for o in outcomes if "error" in o]
    if errors:
        raise AssertionError("; ".join(errors))
    seconds = round(time.perf_counter() - began, 2)
    return [dict(o, rounds=rounds, wall_seconds=seconds) for o in outcomes]


def repeat_check(cache_dir, backend, repeats=20, model_name=REPEAT_MODEL):
    """The medium text `repeats` times in one process, with a short call before each repeat so that GPU memory is
    reused in between: every output must be bit-identical to the first, on every backend.

    On Metal, four normalisation kernels (layer norm, AdaLayerNorm, AdaIN with and without folded weights) used to
    read the mean from threadgroup memory and then overwrite it with no barrier in between. Now and then one SIMD
    group read the overwritten value and normalised its elements with a wrong mean: on an Apple M4 about one engine
    pass in 30, and the waveform then drifted out of phase for the rest of that sentence (differences up to 1.3)."""
    model = KittenTTS(model_name, cache_dir=str(cache_dir), backend=backend)
    first = as_audio(model.generate(REPEAT_TEXT, voice="Bella"))
    validate_audio(first, f"{model_name} repeat 0")
    diffs, differing = [], []
    for i in range(1, repeats):
        model.generate(TEXT, voice="Jasper")
        audio = as_audio(model.generate(REPEAT_TEXT, voice="Bella"))
        diff = difference(audio, first)
        diffs.append(diff)
        if diff != 0.0:
            differing.append(i)
    result = {"check": "repeat", "model": model_name, "backend": backend or "default",
              "active_backend": _engine.active_backend(), "repeats": repeats, "samples": int(len(first)),
              "differing_repeats": differing, "max_diff": max((d for d in diffs if d is not None), default=0.0)}
    print(json.dumps(result))
    if differing:
        raise AssertionError(f"{model_name} on {result['active_backend']}: repeats {differing} of {repeats - 1} "
                             f"differ from the first output (largest difference {result['max_diff']}; "
                             f"None = a different length)")
    return [result]


def process_memory_mb():
    """This process's memory in MB: the footprint on macOS (it counts GPU memory too), private bytes on Windows,
    resident memory elsewhere."""
    import ctypes
    import os

    if sys.platform == "darwin":
        class RusageInfoV2(ctypes.Structure):  # struct rusage_info_v2 up to ri_phys_footprint, and the rest
            _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"f{i}", ctypes.c_uint64) for i in range(18)]

        info = RusageInfoV2()
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        if libc.proc_pid_rusage(os.getpid(), 2, ctypes.byref(info)) != 0:  # RUSAGE_INFO_V2
            raise OSError("proc_pid_rusage failed")
        return info.f7 / 1e6  # ri_phys_footprint
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS_EX
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                               "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                                               "PrivateUsage")]

        c = Counters()
        c.cb = ctypes.sizeof(c)
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return c.PrivateUsage / 1e6
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6


def memory_check(cache_dir, backend, model_name=NANO, warmup=100, calls=100, limit_mb=8.0):
    """The same short text again and again: once warm, memory must stay level (a leak of a few MB per call adds up
    in a long-running program).

    The macOS runner drops its footprint once, by a fixed ~160 MB, partway through a run (a one-time compaction
    unrelated to the engine, seen in every run). Measuring growth as last-minus-first then reads that drop as
    negative growth and hides a real leak underneath it (a +17 MB-per-100-calls leak still shows as about -147 MB
    and passes). So two things: a long warm-up (100 calls) puts the one-time drop before the measured calls, and
    growth is measured from the LOWEST sampled footprint to the last, sampled finely, so a one-time downward step
    cannot mask a steady upward leak. Without an emptied autorelease pool Metal grew 0.17 MB per call on the CI's
    virtual M1 (6.3 MB per call on an Apple M4); with it, by at most about 0.1 MB over 100 calls."""
    model = KittenTTS(model_name, cache_dir=str(cache_dir), backend=backend)
    for _ in range(warmup):
        model.generate(TEXT, voice="Jasper")
    before = process_memory_mb()
    samples = [before]
    every = max(1, calls // 10)
    for i in range(calls):
        audio = as_audio(model.generate(TEXT, voice="Jasper"))
        if (i + 1) % every == 0:
            samples.append(process_memory_mb())
    validate_audio(audio, f"{model_name} memory check")
    growth = samples[-1] - min(samples)  # rise from the lowest point: immune to a one-time compaction drop
    result = {"model": model_name, "backend": backend or "default", "active_backend": _engine.active_backend(),
              "calls": calls, "warmup": warmup, "memory_mb": [round(m, 1) for m in samples],
              "min_mb": round(min(samples), 1), "last_mb": round(samples[-1], 1), "growth_mb": round(growth, 1),
              "growth_mb_per_call": round(growth / calls, 3), "limit_mb": limit_mb}
    print(json.dumps(result))
    if growth > limit_mb:
        raise AssertionError(f"{model_name} on {result['active_backend']}: memory grew {growth:.0f} MB from its "
                             f"lowest point over {calls} calls after warm-up ({growth / calls:.2f} MB per call; "
                             f"limit {limit_mb:.0f} MB)")
    return [result]


def main():
    parser = argparse.ArgumentParser(description="Generate KittenTTS release smoke audio.")
    parser.add_argument("--out-dir", default="audio-smoke")
    parser.add_argument("--cache-dir", default=".hf-cache")
    parser.add_argument("--backend", default=None, help="Native backend to use, such as cpu or metal.")
    parser.add_argument("--expect-backend", default=None,
                        help="Fail unless the engine ran on this backend; auto: the backend each model should get.")
    parser.add_argument("--compare-to", default=None, help="Output dir of a run on another backend to compare with.")
    parser.add_argument(
        "--mode",
        choices=["quick", "release", "speed", "models", "paths", "mix", "threads", "auto", "memory", "repeat"],
        default="release",
        help="quick: one WAV; release: nano voices, speeds and int8; speed: the speed check only; "
        "models: micro and mini, fp32 and int8; paths: a model folder with non-ASCII characters in its path; "
        "mix: CPU and default-backend models in one process; threads: several models generating at once; "
        "auto: the default backend of every model, with warm timings; memory: memory stays level over 100 calls; "
        "repeat: the same medium text 20 times in one process, every output bit-identical.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_cache = {}
    results = []

    if args.mode == "quick":
        results.append(synthesize(model_cache, out_dir, cache_dir, QUICK_CASE, args.backend))
    elif args.mode == "models":
        results.extend(synthesize(model_cache, out_dir, cache_dir, case, args.backend) for case in MODEL_CASES)
    elif args.mode == "paths":
        results.extend(unicode_path_check(cache_dir, args.backend))
    elif args.mode == "mix":
        results.extend(backend_mix_check(cache_dir))
    elif args.mode == "threads":
        results.extend(threads_check(cache_dir))
    elif args.mode == "auto":
        results.extend(auto_backend_check(cache_dir))
    elif args.mode == "memory":
        results.extend(memory_check(cache_dir, args.backend))
    elif args.mode == "repeat":
        results.extend(repeat_check(cache_dir, args.backend))
    else:
        if args.mode == "release":
            results.extend(
                synthesize(model_cache, out_dir, cache_dir, case, args.backend)
                for case in CASES
            )

        slow = synthesize(model_cache, out_dir, cache_dir, SPEED_CASES[0], args.backend)
        fast = synthesize(model_cache, out_dir, cache_dir, SPEED_CASES[1], args.backend)
        if fast["samples"] >= slow["samples"]:
            raise AssertionError(
                "speed smoke failed: faster audio should have fewer samples "
                f"(slow={slow['samples']}, fast={fast['samples']})"
            )
        results.extend([slow, fast])

    if args.expect_backend == "auto":
        for r in results:
            want = resolved_backend(args.backend, r["model"])
            if r["active_backend"] != want:
                raise AssertionError(f"{r['model']}: expected backend {want!r}, ran on {r['active_backend']!r}")
    elif args.expect_backend:
        used = sorted({r["active_backend"] for r in results})
        if used != [args.expect_backend]:
            raise AssertionError(f"expected backend {args.expect_backend!r}, ran on {used}")
    if args.compare_to:
        compare(results, args.compare_to)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"python {platform.python_version()} {sys.platform} {platform.machine()} mode={args.mode}")
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
