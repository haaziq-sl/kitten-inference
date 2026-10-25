import argparse
import json
import platform
import sys
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


def synthesize(model_cache, out_dir, cache_dir, case, backend=None):
    model_name = case["model"]
    cache_key = (model_name, backend or "default")
    model = model_cache.get(cache_key)
    if model is None:
        model = KittenTTS(model_name, cache_dir=str(cache_dir), backend=backend)
        model_cache[cache_key] = model

    label = f"{model_name} {case['voice']} speed={case['speed']} backend={backend or 'default'}"
    start = time.perf_counter()
    audio = np.asarray(
        model.generate(TEXT, voice=case["voice"], speed=case["speed"]),
        dtype=np.float32,
    ).reshape(-1)
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
        if abs(r["samples"] - o["samples"]) > 0.05 * o["samples"]:
            raise AssertionError(f"{label}: samples {r['samples']} vs {o['samples']}")
        if not 0.5 <= r["rms"] / o["rms"] <= 2.0:
            raise AssertionError(f"{label}: rms {r['rms']:.4f} vs {o['rms']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Generate KittenTTS release smoke audio.")
    parser.add_argument("--out-dir", default="audio-smoke")
    parser.add_argument("--cache-dir", default=".hf-cache")
    parser.add_argument("--backend", default=None, help="Native backend to use, such as cpu or metal.")
    parser.add_argument("--expect-backend", default=None, help="Fail unless the engine ran on this backend.")
    parser.add_argument("--compare-to", default=None, help="Output dir of a run on another backend to compare with.")
    parser.add_argument(
        "--mode",
        choices=["quick", "release", "speed"],
        default="release",
        help="quick generates one WAV; release generates voice/model samples plus speed check.",
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

    if args.expect_backend:
        used = sorted({r["active_backend"] for r in results})
        if used != [args.expect_backend]:
            raise AssertionError(f"expected backend {args.expect_backend!r}, ran on {used}")
    if args.compare_to:
        compare(results, args.compare_to)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"python {platform.python_version()} {sys.platform} {platform.machine()}")
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
