"""Measurement only (branch ws/integ-metal-probe, never merged): how far Metal output moves from the same model run
alone when two Metal threads and two CPU threads generate at once, per engine gate version.

    python metal_probe.py ARM ITERATION   (ARM: old | new | fixed | new-keep | solo; run with the matching cpp_model.py installed)

new-keep keeps every thread's model alive until all threads end (no model is destroyed while another thread's call
runs). Prints one JSON line per Metal thread and round: max sample difference and RMS ratio vs the reference."""
import json, sys, threading, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tts-0.8" / "tests"))
import audio_smoke as s  # noqa: E402
from kitten_inference.tts08 import KittenTTS  # noqa: E402

arm, it = sys.argv[1], int(sys.argv[2])
cache = Path(".hf-cache")
refs = {}
for name, backend, voice in s.THREAD_PLAN:
    if (name, backend, voice) not in refs:
        refs[(name, backend, voice)] = s.as_audio(KittenTTS(name, cache_dir=str(cache), backend=backend).generate(s.TEXT, voice=voice))
keep, rows = [], []
barrier = threading.Barrier(len(s.THREAD_PLAN))


def worker(i, name, backend, voice):
    model = KittenTTS(name, cache_dir=str(cache), backend=backend)
    if arm == "new-keep":
        keep.append(model)
    barrier.wait(600)
    for r in range(2):
        got = s.as_audio(model.generate(s.TEXT, voice=voice))
        want = refs[(name, backend, voice)]
        same_len = got.shape == want.shape
        diff = float(np.max(np.abs(got - want))) if same_len else None
        rms = (float(np.sqrt(np.mean(np.square(got - want, dtype=np.float64)) / np.mean(np.square(want, dtype=np.float64))))
               if same_len else None)
        rows.append({"arm": arm, "iteration": it, "thread": i, "backend": s.resolved_backend(backend, name), "round": r,
                     "same_length": same_len, "max_diff": diff, "rms_ratio": rms, "t": round(time.perf_counter(), 3)})


t0 = time.perf_counter()
if arm == "solo":  # no threads: each Metal model of the plan alone, the same two generations, one after another
    barrier = threading.Barrier(1)
    for i, p in enumerate(s.THREAD_PLAN):
        if s.resolved_backend(p[1], p[0]) == "metal":
            worker(i, *p)
else:
    threads = [threading.Thread(target=worker, args=(i,) + p) for i, p in enumerate(s.THREAD_PLAN)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
for row in sorted(rows, key=lambda x: (x["thread"], x["round"])):
    print(json.dumps(dict(row, wall=round(time.perf_counter() - t0, 2))))
