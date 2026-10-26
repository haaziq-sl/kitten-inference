"""The engine wrapper without a GPU (a fake engine library): start cleanup on Ctrl-C, event hand-off between threads,
abandoned streams, and garbage-collected streams."""
import collections
import ctypes
import gc
import threading
import time

import pytest

from kitten_inference.tts2 import _abi, _engine


class FakeLib:
    """The kt2n_* calls Engine makes, on a queue of fake events (poll waits for them, up to its timeout)."""

    def __init__(self):
        self.next_id = 1
        self.pending = collections.deque()
        self.cond = threading.Condition()
        self.destroyed = []
        self.cancelled = []
        self.released = []
        self.borrowed = 0
        self.polls = 0
        self.create_raises = None
        self._keep = []

    # lifetime
    def kt2n_engine_create(self, config, handle, err):
        handle._obj.value = 0x1234
        if self.create_raises is not None:  # as Ctrl-C raised by Python when the call returns
            raise self.create_raises
        return 0

    def kt2n_engine_destroy(self, handle):
        self.destroyed.append(handle.value)

    # requests
    def kt2n_submit(self, handle, params, out, err):
        out._obj.value = self.next_id
        self.next_id += 1
        return 0

    def kt2n_cancel(self, handle, ident, reason):
        self.cancelled.append(ident)
        return 0

    def kt2n_release(self, handle, ident):
        self.released.append(ident)
        return 0

    def kt2n_error_lookup(self, status):
        return None

    # events
    def push(self, ident, kind=_abi.EVENT_AUDIO, data=b"\1\0"):
        with self.cond:
            self.pending.append((ident, kind, data))
            self.cond.notify_all()

    def kt2n_poll(self, handle, events, max_events, timeout_ms, count):
        self.polls += 1
        with self.cond:
            if not self.pending:
                self.cond.wait(timeout_ms / 1000.0)
            n = 0
            while self.pending and n < max_events:
                ident, kind, data = self.pending.popleft()
                buf = (ctypes.c_uint8 * max(1, len(data))).from_buffer_copy(data or b"\0")
                self._keep.append(buf)
                ev = events[n]
                ev.kind, ev.status, ev.id = kind, 0, ident
                ev.data = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8))
                ev.data_len = len(data)
                ev.at_ns = time.monotonic_ns()
                n += 1
            count._obj.value = n
            self.borrowed += n
        return 0

    def kt2n_events_release(self, handle, events, n):
        self.borrowed -= n


def _engine_with(lib):
    return _engine.Engine(lib, _abi.EngineConfig())


def _submit(engine):
    return engine.submit(_abi.RequestParams())


def test_ctrl_c_as_create_returns_frees_the_new_engine():
    lib = FakeLib()
    lib.create_raises = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        _engine_with(lib)
    assert lib.destroyed == [0x1234]  # the engine that create made is not left behind (GPU memory)


def test_failed_create_without_an_engine():
    lib = FakeLib()
    lib.kt2n_engine_create = lambda config, handle, err: 7  # an error status, no engine
    with pytest.raises(_engine.EngineError):
        _engine_with(lib)
    assert lib.destroyed == []


def test_events_in_order_and_released_at_the_end():
    lib = FakeLib()
    engine = _engine_with(lib)
    ident = _submit(engine)
    for _ in range(3):
        lib.push(ident)
    lib.push(ident, _abi.EVENT_DONE, b"")
    kinds = [ev.kind for ev in engine.events(ident)]
    assert kinds == [_abi.EVENT_AUDIO] * 3 + [_abi.EVENT_DONE]
    assert lib.released == [ident] and lib.borrowed == 0 and not engine.busy()


def test_an_event_polled_by_another_thread_reaches_its_owner_at_once(monkeypatch):
    # Thread B polls (its own request waits behind A's in the engine); A's chunk, routed by B, must reach A without
    # waiting for B's next poll to time out. The poll timeout is 1 s here, so a missed hand-off shows as >= 1 s.
    monkeypatch.setattr(_engine, "POLL_TIMEOUT_MS", 1000)
    lib = FakeLib()
    engine = _engine_with(lib)
    a, b = _submit(engine), _submit(engine)
    got = {}

    def consume(ident, key):
        for ev in engine.events(ident):
            got.setdefault(key, []).append((ev.kind, time.monotonic()))

    tb = threading.Thread(target=consume, args=(b, "b"))
    tb.start()
    deadline = time.monotonic() + 5
    while not engine._pumping and time.monotonic() < deadline:  # B is the polling thread
        time.sleep(0.001)
    ta = threading.Thread(target=consume, args=(a, "a"))
    ta.start()
    time.sleep(0.05)  # A now waits while B polls
    lags = []
    for _ in range(5):
        sent = time.monotonic()
        lib.push(a)
        while len(got.get("a", [])) < len(lags) + 1 and time.monotonic() < sent + 5:
            time.sleep(0.0005)
        lags.append(got["a"][-1][1] - sent)
        time.sleep(0.02)
    lib.push(a, _abi.EVENT_DONE, b"")
    lib.push(b, _abi.EVENT_DONE, b"")
    ta.join(5)
    tb.join(5)
    assert not ta.is_alive() and not tb.is_alive()
    assert max(lags) < 0.25, lags  # milliseconds in practice; the old pump took up to the 1 s timeout
    assert lib.borrowed == 0


def test_abandoned_request_is_cancelled_and_released_on_its_terminal_event():
    lib = FakeLib()
    engine = _engine_with(lib)
    ident = _submit(engine)
    lib.push(ident)
    it = engine.events(ident)
    next(it)
    engine.abandon(ident)
    assert lib.cancelled == [ident] and not engine.busy() and not engine.is_live(ident)
    lib.push(ident)  # more audio after the cancel: dropped
    lib.push(ident, _abi.EVENT_DONE, b"")
    other = _submit(engine)
    lib.push(other, _abi.EVENT_DONE, b"")
    assert [ev.kind for ev in engine.events(other)] == [_abi.EVENT_DONE]  # pumping for another request
    assert ident in lib.released and other in lib.released


def test_abandon_inside_the_engine_lock_does_not_deadlock():
    # The garbage collector can run an abandoned stream's finally while this thread is inside a critical section.
    lib = FakeLib()
    engine = _engine_with(lib)
    ident = _submit(engine)
    done = threading.Event()

    def inside():
        with engine._lock:
            engine.abandon(ident)
        done.set()

    t = threading.Thread(target=inside, daemon=True)
    t.start()
    assert done.wait(5), "abandon() blocked on a lock its own thread held"
    assert not engine.busy() and lib.cancelled == [ident]


def test_garbage_collected_stream_inside_a_critical_section():
    lib = FakeLib()
    engine = _engine_with(lib)
    ident = _submit(engine)
    lib.push(ident)

    def stream():
        try:
            for ev in engine.events(ident):
                yield ev
        finally:
            engine.abandon(ident)

    class Player:  # a reference cycle: only the cyclic garbage collector frees it
        pass

    player = Player()
    player.self = player
    player.it = stream()
    next(player.it)
    del player
    done = threading.Event()

    def collect_inside():
        with engine._lock:
            gc.collect()  # runs the generator's finally on this thread, as an automatic collection would
        done.set()

    t = threading.Thread(target=collect_inside, daemon=True)
    t.start()
    assert done.wait(10), "a garbage-collected stream deadlocked inside Engine._lock"
    assert lib.cancelled == [ident] and not engine.busy()


def test_ctrl_c_as_poll_returns_keeps_the_events(monkeypatch):
    lib = FakeLib()
    engine = _engine_with(lib)
    a = _submit(engine)
    lib.push(a)
    real_poll = lib.kt2n_poll

    def poll_then_interrupt(*args):
        real_poll(*args)
        raise KeyboardInterrupt  # as raised by Python right after kt2n_poll returns

    lib.kt2n_poll = poll_then_interrupt
    with pytest.raises(KeyboardInterrupt):
        engine.pump(10)
    assert lib.borrowed == 0 and len(engine._queues[a]) == 1 and not engine._pumping


def test_destroy_wakes_a_waiting_thread():
    lib = FakeLib()
    engine = _engine_with(lib)
    a, b = _submit(engine), _submit(engine)
    errors = []

    def wait(ident):
        try:
            for _ in engine.events(ident):
                pass
        except RuntimeError as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=wait, args=(i,)) for i in (a, b)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    engine.destroy()
    for t in threads:
        t.join(5)
    assert not any(t.is_alive() for t in threads)
    assert lib.destroyed == [0x1234] and sorted(lib.cancelled) == [a, b]
    assert errors and all("closed" in e for e in errors)


class NoisyLib(FakeLib):
    """An engine that writes its developer line and a line for users to file descriptor 2 while it starts."""

    def __init__(self, status=0):
        super().__init__()
        self.status = status

    def kt2n_engine_create(self, config, handle, err):
        import os
        os.write(2, b"[decoder] config: CUDA_DEVICE_MAX_CONNECTIONS=unset(8) aux_rows=0 aux_auto=1\n"
                    b"kt2n: a line for users\n")
        handle._obj.value = 0x1234 if self.status == 0 else 0
        return self.status


@pytest.mark.skipif(__import__("sys").platform == "win32",
                    reason="platform: on Windows the engine's C runtime keeps its own stderr")
def test_engine_start_is_quiet(capfd, monkeypatch):
    # The Python API shows no developer output by default; KITTEN_TTS2_VERBOSE=1 shows it, and a failed start shows
    # everything the engine said.
    monkeypatch.delenv("KITTEN_TTS2_VERBOSE", raising=False)
    _engine_with(NoisyLib())
    err = capfd.readouterr().err
    assert "[decoder] config" not in err and "kt2n: a line for users" in err, err
    monkeypatch.setenv("KITTEN_TTS2_VERBOSE", "1")
    _engine_with(NoisyLib())
    assert "[decoder] config: CUDA_DEVICE_MAX_CONNECTIONS" in capfd.readouterr().err
    monkeypatch.delenv("KITTEN_TTS2_VERBOSE")
    with pytest.raises(_engine.EngineError):
        _engine_with(NoisyLib(status=7))
    err = capfd.readouterr().err
    assert "[decoder] config" in err and "kt2n: a line for users" in err, err
    import os
    os.write(2, b"after\n")  # file descriptor 2 is back where it was
    assert capfd.readouterr().err == "after\n"
