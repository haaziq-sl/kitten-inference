"""A thin, thread-safe owner of one ``kt2n_engine``: create/destroy, submit, voice calls and an event pump.

Events are copied out of engine memory and returned to the engine at once (``kt2n_events_release``), then routed
to the queue of the request or build they belong to. Any thread waiting for its own events may pump; only one
pumps at a time, so several requests (or a suspended stream and a new request) never steal each other's events.
"""
from __future__ import annotations

import collections
import ctypes
import threading
from dataclasses import dataclass
from typing import Deque, Dict, Iterator, Optional, Set

from . import _abi

POLL_BATCH = 16
POLL_TIMEOUT_MS = 100  # the longest a pump waits, so Ctrl-C and cancellation stay responsive


class EngineError(RuntimeError):
    """A failed engine call: ``status`` (kt2n_status), ``code`` (the public error code), ``param`` and ``message``."""

    def __init__(self, status: int, message: str = "", code: str = "", param: str = "", http_status: int = 0):
        self.status = int(status)
        self.code = code
        self.param = param
        self.http_status = http_status
        self.message = message or code or "engine error"
        super().__init__(f"{self.message} (status {self.status})")


def error_from(lib, status: int, err: Optional[_abi.Error] = None) -> EngineError:
    message = code = param = ""
    http = 0
    if err is not None and int(err.status) == status:
        message, code, param, http = _abi.text(err.message), _abi.text(err.code), _abi.text(err.param), err.http_status
    if not code:
        info = lib.kt2n_error_lookup(int(status))
        if info:
            code = _abi.text(info.contents.code)
            http = http or int(info.contents.http_status)
    if not message and err is not None and err.message:
        message = _abi.text(err.message)
    return EngineError(status, message or code, code, param, http)


@dataclass
class Event:
    kind: int
    status: int
    id: int
    data: bytes = b""
    samples: int = 0
    is_final_chunk: bool = False
    fields_json: Optional[str] = None
    error: Optional[EngineError] = None
    voice: int = 0


class Engine:
    def __init__(self, lib, config: _abi.EngineConfig):
        self.lib = lib
        handle = ctypes.c_void_p()
        err = _abi.Error()
        status = lib.kt2n_engine_create(ctypes.byref(config), ctypes.byref(handle), ctypes.byref(err))
        if status != _abi.OK:
            if handle.value:
                lib.kt2n_engine_destroy(handle)
            raise error_from(lib, status, err)
        self._handle = handle
        self._queues: Dict[int, Deque[Event]] = {}
        self._abandoned: Set[int] = set()
        self._lock = threading.Lock()        # queues, abandoned set
        self._pump_lock = threading.Lock()   # one kt2n_poll at a time; destroy waits for it
        self._call_lock = threading.RLock()  # every other engine call; destroy waits for it
        self._events = (_abi.Event * POLL_BATCH)()
        self.closing = False  # destroy() has begun: requests end as cancelled

    # -- lifetime ---------------------------------------------------------------------------------------------
    @property
    def handle(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise RuntimeError("the engine is closed")
        return self._handle

    def busy(self) -> bool:
        with self._lock:
            return any(i not in self._abandoned for i in self._queues)

    def destroy(self) -> None:
        """Cancels what is running and frees the engine. A thread still waiting for its events gets RuntimeError
        ("the engine is closed"); no engine call can overlap kt2n_engine_destroy (engine.h's lifetime rule)."""
        with self._lock:
            if self._handle is None:
                return
            self.closing = True
            running = list(self._queues)
        for ident in running:
            self.cancel(ident)
        with self._pump_lock, self._call_lock:
            handle, self._handle = self._handle, None
            if handle is not None:
                self.lib.kt2n_engine_destroy(handle)

    # -- calls ------------------------------------------------------------------------------------------------
    def check(self, status: int, err: Optional[_abi.Error] = None) -> None:
        if status != _abi.OK:
            raise error_from(self.lib, status, err)

    def _call(self, name: str, *args) -> int:
        """lib.<name>(engine, *args), never concurrently with destroy."""
        with self._call_lock:
            return getattr(self.lib, name)(self.handle, *args)

    def stats_json(self) -> str:
        need = ctypes.c_size_t(0)
        status = self._call("kt2n_stats_json", None, 0, ctypes.byref(need))
        if status not in (_abi.OK, _abi.ERR_TOO_SMALL):
            self.check(status)
        buf = ctypes.create_string_buffer(max(int(need.value), 1) + 1)
        self.check(self._call("kt2n_stats_json", buf, len(buf), ctypes.byref(need)))
        return buf.value.decode("utf-8", "replace")

    def low_memory_limits(self) -> _abi.LowMemoryLimits:
        out = _abi.LowMemoryLimits(struct_size=ctypes.sizeof(_abi.LowMemoryLimits))
        self.check(self._call("kt2n_engine_low_memory_limits", ctypes.byref(out)))
        return out

    def voice_lookup(self, name: str) -> int:
        out = ctypes.c_uint64(0)
        err = _abi.Error()
        self.check(self._call("kt2n_voice_lookup", name.encode("utf-8"), ctypes.byref(out), ctypes.byref(err)),
                   err)
        return int(out.value)

    def voice_info(self, voice: int) -> _abi.VoiceInfo:
        out = _abi.VoiceInfo(struct_size=ctypes.sizeof(_abi.VoiceInfo))
        self.check(self._call("kt2n_voice_info_get", voice, ctypes.byref(out)))
        return out

    def voice_prepare(self, voice: int) -> int:
        """Best effort, like kitten-tts-native: a failure here is reported by the request that uses the voice."""
        return int(self._call("kt2n_voice_prepare", voice, None))

    def voice_load(self, artifact: bytes) -> int:
        out = ctypes.c_uint64(0)
        err = _abi.Error()
        self.check(self._call("kt2n_voice_load", artifact, len(artifact), ctypes.byref(out), ctypes.byref(err)), err)
        return int(out.value)

    def voice_export(self, voice: int) -> bytes:
        need = ctypes.c_size_t(0)
        status = self._call("kt2n_voice_export", voice, None, 0, ctypes.byref(need))
        if status not in (_abi.OK, _abi.ERR_TOO_SMALL):
            self.check(status)
        buf = ctypes.create_string_buffer(int(need.value))
        self.check(self._call("kt2n_voice_export", voice, buf, len(buf), ctypes.byref(need)))
        return buf.raw[: int(need.value)]

    def voice_evict(self, voice: int) -> None:
        self.check(self._call("kt2n_voice_evict", voice))

    def submit(self, params: _abi.RequestParams) -> int:
        out = ctypes.c_uint64(0)
        err = _abi.Error()
        with self._lock:  # the queue exists before the first event can be routed
            self.check(self._call("kt2n_submit", ctypes.byref(params), ctypes.byref(out), ctypes.byref(err)),
                       err)
            self._queues[int(out.value)] = collections.deque()
        return int(out.value)

    def voice_build(self, params: _abi.VoiceBuildParams) -> int:
        out = ctypes.c_uint64(0)
        err = _abi.Error()
        with self._lock:
            self.check(self._call("kt2n_voice_build", ctypes.byref(params), ctypes.byref(out), ctypes.byref(err)), err)
            self._queues[int(out.value)] = collections.deque()
        return int(out.value)

    def cancel(self, ident: int) -> None:
        with self._call_lock:
            if self._handle is not None:
                self.lib.kt2n_cancel(self._handle, ident, _abi.CANCEL_REQUEST)

    def abandon(self, ident: int) -> None:
        """Cancels ``ident`` and stops waiting for it: its events are dropped and its id released on its terminal
        event (by whichever thread pumps next). Safe to call from a generator's ``finally``."""
        with self._lock:
            if ident not in self._queues:
                return
            terminal = any(e.kind in (_abi.EVENT_DONE, _abi.EVENT_VOICE_BUILT) for e in self._queues[ident])
            if terminal:
                del self._queues[ident]
            else:
                self._abandoned.add(ident)
        if terminal:
            self._release(ident)
        else:
            self.cancel(ident)

    def _release(self, ident: int) -> None:
        with self._call_lock:
            if self._handle is not None:
                self.lib.kt2n_release(self._handle, ident)

    # -- events -----------------------------------------------------------------------------------------------
    def _copy(self, ev: _abi.Event) -> Event:
        data = ctypes.string_at(ev.data, ev.data_len) if ev.data and ev.data_len else b""
        error = None
        if ev.error and ev.status != _abi.OK:
            e = ev.error.contents
            error = error_from(self.lib, ev.status, e)
        fields = ev.fields_json.decode("utf-8", "replace") if ev.fields_json else None
        return Event(kind=ev.kind, status=ev.status, id=ev.id, data=data, samples=ev.samples,
                     is_final_chunk=bool(ev.is_final_chunk), fields_json=fields, error=error, voice=ev.voice)

    def pump(self, timeout_ms: int = POLL_TIMEOUT_MS) -> None:
        """Polls once (waiting at most timeout_ms) and routes what arrived."""
        with self._pump_lock:
            handle = self.handle  # RuntimeError once destroyed
            count = ctypes.c_int32(0)
            self.check(self.lib.kt2n_poll(handle, self._events, POLL_BATCH, timeout_ms, ctypes.byref(count)))
            n = int(count.value)
            if n <= 0:
                return
            try:
                copied = [self._copy(self._events[i]) for i in range(n)]
            finally:
                self.lib.kt2n_events_release(handle, self._events, n)
        release = []
        with self._lock:
            for ev in copied:
                terminal = ev.kind in (_abi.EVENT_DONE, _abi.EVENT_VOICE_BUILT)
                if ev.id in self._abandoned:
                    if terminal:
                        self._abandoned.discard(ev.id)
                        self._queues.pop(ev.id, None)
                        release.append(ev.id)
                    continue
                queue = self._queues.get(ev.id)
                if queue is not None:
                    queue.append(ev)
        for ident in release:
            self._release(ident)

    def events(self, ident: int) -> Iterator[Event]:
        """Yields the events of ``ident`` in order until its terminal event (included), then releases the id."""
        while True:
            with self._lock:
                queue = self._queues[ident]
                ev = queue.popleft() if queue else None
                if ev is not None and ev.kind in (_abi.EVENT_DONE, _abi.EVENT_VOICE_BUILT):
                    del self._queues[ident]
            if ev is None:
                self.pump()
                continue
            if ev.kind in (_abi.EVENT_DONE, _abi.EVENT_VOICE_BUILT):
                self._release(ident)
                yield ev
                return
            yield ev
