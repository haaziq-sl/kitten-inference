"""ctypes declarations of the engine C ABI, version 2 (``tts-2/include/kt2n/engine.h``).

Only declarations live here: structs, constants and function prototypes. The struct layouts are checked against
the header by ``tests/test_abi_layout.py`` (it compiles a small C program that prints every size and offset).
ctypes releases the GIL around every foreign call, so the calls that can wait (create, destroy, poll with a
timeout, voice prepare) never block other Python threads.
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import (POINTER, Structure, c_char, c_char_p, c_double, c_float, c_int, c_int32, c_int64, c_size_t,
                    c_uint32, c_uint64, c_uint8, c_void_p)
from pathlib import Path

ABI_VERSION = 2
INVALID_ID = 0

# kt2n_status (the subset this package names; any other value is reported with its number)
OK = 0
ERR_INVALID_ARGUMENT = 1
ERR_INVALID_HANDLE = 2
ERR_ABI_MISMATCH = 3
ERR_AGAIN = 4
ERR_CANCELLED = 5
ERR_NOT_READY = 6
ERR_UNSUPPORTED = 7
ERR_TOO_SMALL = 8
ERR_INVALID_REQUEST = 100
ERR_INVALID_INPUT = 101
ERR_INVALID_VOICE = 102
ERR_REQUEST_TOO_LARGE = 105
ERR_VOICE_NOT_FOUND = 110
ERR_GPU_MEMORY_LIMIT = 111
ERR_INVALID_AUDIO = 200
ERR_INVALID_AUDIO_DURATION = 201
ERR_INVALID_AUDIO_SIGNAL = 202
ERR_VOICE_ARTIFACT_VERSION_MISMATCH = 210
ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE = 214
ERR_INVALID_REFERENCE_TRANSCRIPT = 215
ERR_DEADLINE_EXCEEDED = 301
ERR_INTERNAL = 303

# kt2n_config_flags
CFG_INFERENCE_ONLY = 1 << 4

# kt2n_voice_cache_mode
VOICE_CACHE_ALL = 0
VOICE_CACHE_LRU = 1
VOICE_CACHE_NONE = 2

# kt2n_memory_profile
MEMORY_PROFILE_DEFAULT = 0
MEMORY_PROFILE_LOW = 1
MEMORY_PROFILE_AUTO = 2

# kt2n_voice_kind
VOICE_BUILTIN = 0
VOICE_INLINE = 1
VOICE_SAVED = 2

# kt2n_voice_build_flags
VB_INLINE = 1 << 0
VB_SAVED = 1 << 1
VB_DECODED_PCM = 1 << 2

# kt2n_audio_format
FORMAT_PCM = 0
FORMAT_WAV = 1
FORMAT_F32 = 6

# kt2n_output_mode
OUTPUT_STREAM = 0
OUTPUT_STREAM_FILE = 1
OUTPUT_COMPLETE = 2

# kt2n_speech_mode
SPEECH_MODE_DEFAULT = 0

# kt2n_sampler_mode
SAMPLER_UNSET = 0
SAMPLER_STABLE = 1
SAMPLER_EXPRESSIVE = 2

# kt2n_sampler_present
SAMPLER_HAS_MODE = 1 << 0
SAMPLER_HAS_TEMPERATURE = 1 << 1
SAMPLER_HAS_TOP_P = 1 << 2
SAMPLER_HAS_TOP_K = 1 << 3
SAMPLER_HAS_MIN_P = 1 << 4
SAMPLER_HAS_MAX_NEW_TOKENS = 1 << 5

# kt2n_request_flags
REQ_SEED = 1 << 1
REQ_NORMALIZE_OFF = 1 << 4

# kt2n_cancel_reason
CANCEL_REQUEST = 0

# kt2n_event_kind
EVENT_AUDIO = 1
EVENT_TIMELINE = 2
EVENT_METRICS = 4
EVENT_PHRASE_DONE = 5
EVENT_DONE = 6
EVENT_VOICE_BUILT = 7
EVENT_HEADERS = 8

CAP_VOICE_CLONING = "cap:voice.cloning"


class ErrorInfo(Structure):
    _c_name_ = "kt2n_error_info"
    _fields_ = [("status", c_int32), ("http_status", c_int32), ("code", c_char_p), ("type", c_char_p)]


class Error(Structure):
    _c_name_ = "kt2n_error"
    _fields_ = [("status", c_int32), ("http_status", c_int32), ("code", c_char * 48), ("param", c_char * 48),
                ("message", c_char * 400)]


class EngineConfig(Structure):
    _c_name_ = "kt2n_engine_config"
    _fields_ = [
        ("struct_size", c_uint32), ("abi_version", c_uint32), ("flags", c_uint32), ("device", c_int32),
        ("lm_file", c_char_p), ("s3gen_file", c_char_p), ("voices_file", c_char_p), ("tokenizer_json", c_char_p),
        ("text_data_dir", c_char_p), ("reserved_ptr0", c_char_p), ("reserved_ptr1", c_char_p),
        ("max_streams", c_int32), ("max_lm_rows", c_int32), ("render_pages", c_int32),
        ("inline_voice_lru", c_int32), ("clone_slots", c_int32), ("clone_queue", c_int32),
        ("clone_timeout_ms", c_int32), ("device_arena_bytes", c_uint64), ("pinned_arena_bytes", c_uint64),
        ("threads_text", c_int32), ("threads_post", c_int32), ("threads_voice", c_int32), ("cpuset", c_char_p),
        ("voice_build_dir", c_char_p), ("max_custom_voices", c_int32),
        ("voice_cache_mode", c_int32), ("voice_cache_max_voices", c_int32), ("voice_cache_max_bytes", c_uint64),
        ("memory_profile", c_int32), ("min_free_device_bytes", c_uint64),
        ("custom_voice_max_prefix_ids", c_int32),
    ]


class LowMemoryLimits(Structure):
    _c_name_ = "kt2n_low_memory_limits"
    _fields_ = [("struct_size", c_uint32), ("fitted", c_int32), ("lm_context_tokens", c_int32),
                ("decoder_pages", c_int32), ("vocoder_frames", c_int32), ("render_graphs", c_int32),
                ("free_after_context", c_uint64), ("min_free_bytes", c_uint64), ("streaming_budget_tokens", c_int32)]


class VoiceInfo(Structure):
    _c_name_ = "kt2n_voice_info"
    _fields_ = [("struct_size", c_uint32), ("kind", c_int32), ("name", c_char * 96), ("language", c_char * 16),
                ("prompt_tokens", c_int32), ("prompt_frames", c_int32), ("lm_prefix_ids", c_int32),
                ("hidden", c_int32), ("peak_gain_db", c_float), ("refs", c_int32)]


class VoiceBuildParams(Structure):
    _c_name_ = "kt2n_voice_build_params"
    _fields_ = [("struct_size", c_uint32), ("flags", c_uint32), ("audio", c_void_p), ("audio_len", c_size_t),
                ("audio_format", c_char_p), ("pcm", POINTER(c_float)), ("pcm_samples", c_size_t),
                ("pcm_rate", c_int32), ("consent_speaker_dim", c_int32), ("consent_speaker", POINTER(c_float)),
                ("reference_transcript", c_char_p), ("language", c_char_p), ("identity_key", c_char_p),
                ("deadline_ns", c_int64), ("user_data", c_uint64)]


class Sampler(Structure):
    _c_name_ = "kt2n_sampler"
    _fields_ = [("present", c_uint32), ("mode", c_int32), ("temperature", c_double), ("top_p", c_double),
                ("min_p", c_double), ("top_k", c_int64), ("max_new_tokens", c_int64)]


class RequestParams(Structure):
    _c_name_ = "kt2n_request_params"
    _fields_ = [("struct_size", c_uint32), ("flags", c_uint32), ("text", c_char_p), ("text_len", c_size_t),
                ("voice", c_uint64), ("request_id", c_char_p), ("seed", c_uint64), ("sampler", Sampler),
                ("speed", c_double), ("format", c_int32), ("output", c_int32), ("speech_mode", c_int32),
                ("reserved0", c_int32), ("deadline_ns", c_int64), ("user_data", c_uint64),
                ("control_json", c_char_p)]


class SessionParams(Structure):
    _c_name_ = "kt2n_session_params"
    _fields_ = [("struct_size", c_uint32), ("flags", c_uint32), ("voice", c_uint64), ("request_id", c_char_p),
                ("seed", c_uint64), ("sampler", Sampler), ("speed", c_double), ("format", c_int32),
                ("speech_mode", c_int32), ("deadline_ns", c_int64), ("user_data", c_uint64)]


class Event(Structure):
    _c_name_ = "kt2n_event"
    _fields_ = [("kind", c_int32), ("status", c_int32), ("id", c_uint64), ("user_data", c_uint64),
                ("seq", c_uint64), ("at_ns", c_int64), ("segment_index", c_int32), ("phrase_index", c_int32),
                ("data", POINTER(c_uint8)), ("data_len", c_size_t), ("sample_begin", c_int64), ("samples", c_int32),
                ("engine_chunk_index", c_int32), ("is_final_chunk", c_int32), ("opening_skipped_tokens", c_int32),
                ("name", c_char_p), ("fields_json", c_char_p), ("error", POINTER(Error)), ("voice", c_uint64)]


class Stats(Structure):
    _c_name_ = "kt2n_stats"
    _fields_ = [("struct_size", c_uint32), ("active_requests", c_int32), ("queued_requests", c_int32),
                ("active_sessions", c_int32), ("lm_rows", c_int32), ("render_queue_depth", c_int32),
                ("render_inflight", c_int32), ("pages_total", c_int32), ("pages_free", c_int32),
                ("voices_loaded", c_int32), ("voice_builds_active", c_int32), ("voice_builds_queued", c_int32),
                ("device_arena_bytes", c_uint64), ("device_arena_used", c_uint64), ("pinned_arena_bytes", c_uint64),
                ("pinned_arena_used", c_uint64), ("requests_done", c_uint64), ("requests_failed", c_uint64),
                ("requests_cancelled", c_uint64), ("lm_steps", c_uint64), ("renders", c_uint64),
                ("render_rows", c_uint64), ("underrun_risks", c_uint64)]


STRUCTS = (ErrorInfo, Error, EngineConfig, LowMemoryLimits, VoiceInfo, VoiceBuildParams, Sampler, RequestParams,
           SessionParams, Event, Stats)

_engine_p = c_void_p
_u64_p = POINTER(c_uint64)

# name -> (restype, argtypes). Every function of engine.h; kt2n_voice_build_cancel_queued is optional (older
# libraries may lack it).
PROTOTYPES = {
    "kt2n_error_lookup": (POINTER(ErrorInfo), [c_int32]),
    "kt2n_engine_low_memory_limits": (c_int32, [_engine_p, POINTER(LowMemoryLimits)]),
    "kt2n_abi_version": (c_uint32, []),
    "kt2n_build_info": (c_char_p, []),
    "kt2n_engine_create": (c_int32, [POINTER(EngineConfig), POINTER(_engine_p), POINTER(Error)]),
    "kt2n_engine_destroy": (None, [_engine_p]),
    "kt2n_voice_lookup": (c_int32, [_engine_p, c_char_p, _u64_p, POINTER(Error)]),
    "kt2n_voice_info_get": (c_int32, [_engine_p, c_uint64, POINTER(VoiceInfo)]),
    "kt2n_voice_speaker": (c_int32, [_engine_p, c_uint64, POINTER(c_float)]),
    "kt2n_voice_build": (c_int32, [_engine_p, POINTER(VoiceBuildParams), _u64_p, POINTER(Error)]),
    "kt2n_voice_build_cancel_queued": (c_int32, [_engine_p, c_uint64, POINTER(c_int32)]),
    "kt2n_voice_artifact_lm_prefix_ids": (c_int32, [c_void_p, c_size_t]),
    "kt2n_voice_load": (c_int32, [_engine_p, c_void_p, c_size_t, _u64_p, POINTER(Error)]),
    "kt2n_voice_export": (c_int32, [_engine_p, c_uint64, c_void_p, c_size_t, POINTER(c_size_t)]),
    "kt2n_voice_evict": (c_int32, [_engine_p, c_uint64]),
    "kt2n_voice_prepare": (c_int32, [_engine_p, c_uint64, POINTER(Error)]),
    "kt2n_submit": (c_int32, [_engine_p, POINTER(RequestParams), _u64_p, POINTER(Error)]),
    "kt2n_cancel": (c_int32, [_engine_p, c_uint64, c_int32]),
    "kt2n_release": (c_int32, [_engine_p, c_uint64]),
    "kt2n_session_open": (c_int32, [_engine_p, POINTER(SessionParams), _u64_p, POINTER(Error)]),
    "kt2n_session_text_append": (c_int32, [_engine_p, c_uint64, c_char_p, c_size_t, POINTER(Error)]),
    "kt2n_session_commit": (c_int32, [_engine_p, c_uint64, POINTER(Error)]),
    "kt2n_session_done": (c_int32, [_engine_p, c_uint64, POINTER(Error)]),
    "kt2n_event_fd": (c_int, [_engine_p]),
    "kt2n_poll": (c_int32, [_engine_p, POINTER(Event), c_int32, c_int32, POINTER(c_int32)]),
    "kt2n_events_release": (None, [_engine_p, POINTER(Event), c_int32]),
    "kt2n_stats_get": (c_int32, [_engine_p, POINTER(Stats)]),
    "kt2n_stats_json": (c_int32, [_engine_p, c_char_p, c_size_t, POINTER(c_size_t)]),
}
OPTIONAL = {"kt2n_voice_build_cancel_queued"}


class LibraryError(OSError):
    """The engine library could not be loaded, or it is not ABI 2."""


_dll_dirs = []  # Windows: keep the add_dll_directory handles alive while the library is loaded


def load(path: os.PathLike | str) -> ctypes.CDLL:
    """Loads the engine shared library, declares every prototype and checks that it implements ABI 2."""
    path = Path(path)
    if not path.is_file():
        raise LibraryError(f"engine library not found: {path}")
    try:
        if sys.platform == "win32":
            # The default search (the DLL's own folder, System32 and added folders) finds its dependencies.
            _dll_dirs.append(os.add_dll_directory(str(path.parent)))
        lib = ctypes.CDLL(str(path))
    except OSError as exc:
        raise LibraryError(f"cannot load the engine library {path}: {exc}") from exc
    for name, (restype, argtypes) in PROTOTYPES.items():
        try:
            fn = getattr(lib, name)
        except AttributeError:
            if name in OPTIONAL:
                continue
            raise LibraryError(f"{path} has no {name}: it is not a Kitten TTS 2 engine library of ABI "
                               f"{ABI_VERSION}") from None
        fn.restype = restype
        fn.argtypes = argtypes
    found = int(lib.kt2n_abi_version())
    if found != ABI_VERSION:
        raise LibraryError(f"{path} implements engine ABI {found}, but this kitten-inference-tts2 needs ABI "
                           f"{ABI_VERSION}; install matching versions of the two packages")
    return lib


def has(lib: ctypes.CDLL, name: str) -> bool:
    """True when the loaded library exports the optional function ``name``."""
    try:
        getattr(lib, name)
    except AttributeError:
        return False
    return True


def text(buf) -> str:
    """A NUL-terminated char array or char* as text (invalid UTF-8 replaced)."""
    if buf is None:
        return ""
    raw = buf if isinstance(buf, bytes) else bytes(buf)
    return raw.split(b"\0", 1)[0].decode("utf-8", "replace")
