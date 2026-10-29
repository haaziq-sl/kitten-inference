"""Helper of test_gpu.py::test_ctrl_c_during_start_frees_the_engine (run in its own process).

    python ctrl_c_start.py REVISION CACHE_DIR_OR_EMPTY OUT.json

It starts and closes a KittenTTS2, then starts another and interrupts it 0.3 s into kt2n_engine_create with
_thread.interrupt_main() (create blocks for seconds and releases the GIL, so Python raises KeyboardInterrupt as the
call returns, as with Ctrl-C at the keyboard), and records the GPU memory in use (nvidia-smi) after each step.
"""
import _thread
import json
import signal
import subprocess
import sys
import threading
import time

from kitten_inference.tts2 import KittenTTS2, _abi

revision, cache, out = sys.argv[1], sys.argv[2] or None, sys.argv[3]
# A process started without a terminal can inherit SIGINT ignored; Python then has no KeyboardInterrupt handler.
signal.signal(signal.SIGINT, signal.default_int_handler)
state = {"arm": False, "fired": None, "returned": None}


def used_mib() -> int:
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    return int(q.split()[0])


class Lib:
    """The engine library, with an interrupt armed for the next kt2n_engine_create."""

    def __init__(self, lib):
        self._lib = lib

    def __getattr__(self, name):
        return getattr(self._lib, name)

    def kt2n_engine_create(self, *args):
        if state["arm"]:
            state["arm"] = False

            def fire():
                state["fired"] = time.monotonic()
                _thread.interrupt_main()
            threading.Timer(0.3, fire).start()
        try:
            return self._lib.kt2n_engine_create(*args)
        finally:
            state["returned"] = time.monotonic()


real_load = _abi.load
_abi.load = lambda path: Lib(real_load(path))
r = {}
tts = KittenTTS2(revision=revision, cache_dir=cache)
tts.generate("Warm up.", voice="Bruno", seed=1)
r["engine_mib"] = used_mib()
tts.close()
time.sleep(1)
r["after_close_mib"] = used_mib()
state["arm"] = True
try:
    KittenTTS2(revision=revision, cache_dir=cache)
    r["interrupted"] = False
except KeyboardInterrupt:
    r["interrupted"] = True
r["interrupt_during_create"] = bool(state["fired"] and state["returned"] and state["fired"] < state["returned"])
time.sleep(1)
r["after_interrupted_start_mib"] = used_mib()
tts = KittenTTS2(revision=revision, cache_dir=cache)
r["next_started"] = tts.generate("Still working.", voice="Bruno", seed=2).size > 0
r["next_engine_mib"] = used_mib()
tts.close()
with open(out, "w") as f:
    json.dump(r, f)
