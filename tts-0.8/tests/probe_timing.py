"""Timing probe (not a test): seconds per clip for a few models on one backend, warm, median of N runs."""
import argparse
import json
import platform
import statistics
import sys
import time

import numpy as np

from kitten_inference.tts08 import KittenTTS, _engine

TEXT = "Kitten TTS is generating a short audio sample for release testing."
SR = 24000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--cache-dir", default=".hf-cache")
    ap.add_argument("--label", default="")
    ap.add_argument("models", nargs="+")
    args = ap.parse_args()
    rows = []
    for name in args.models:
        tts = KittenTTS(name, cache_dir=args.cache_dir, backend=args.backend)
        tts.generate(TEXT, voice="Jasper")  # warm-up
        times, n = [], 0
        for _ in range(args.runs):
            t = time.perf_counter()
            audio = np.asarray(tts.generate(TEXT, voice="Jasper"), dtype=np.float32)
            times.append(time.perf_counter() - t)
            n = len(audio)
        med = statistics.median(times)
        rows.append({"label": args.label, "model": name, "backend": _engine.active_backend(),
                     "max_threads": _engine.max_threads(), "samples": n, "median_s": round(med, 4),
                     "rtf": round(med / (n / SR), 4), "runs": [round(x, 4) for x in times]})
    print(f"python {platform.python_version()} {sys.platform} {platform.machine()}")
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
