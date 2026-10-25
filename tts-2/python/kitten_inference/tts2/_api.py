"""``KittenTTS2``: Kitten TTS 2 speech synthesis on the GPU, through the engine's C API (ctypes)."""
from __future__ import annotations

import ctypes
import hashlib
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import numpy as np

from . import _abi, _audio, _backend, _bundle, _prepare
from ._engine import Engine, EngineError

SAMPLE_RATE = _audio.SAMPLE_RATE
MEMORY_PROFILES = {"auto": _abi.MEMORY_PROFILE_AUTO, "default": _abi.MEMORY_PROFILE_DEFAULT,
                   "low": _abi.MEMORY_PROFILE_LOW}
VOICE_CACHES = {"all": _abi.VOICE_CACHE_ALL, "lru": _abi.VOICE_CACHE_LRU, "none": _abi.VOICE_CACHE_NONE}
PRESETS = {"stable": _abi.SAMPLER_STABLE, "expressive": _abi.SAMPLER_EXPRESSIVE}
# generate()/stream() sampling keywords -> (sampler field, present bit, type). max_tokens is kitten-tts-native's name.
SAMPLING = {
    "temperature": ("temperature", _abi.SAMPLER_HAS_TEMPERATURE, float),
    "top_p": ("top_p", _abi.SAMPLER_HAS_TOP_P, float),
    "min_p": ("min_p", _abi.SAMPLER_HAS_MIN_P, float),
    "top_k": ("top_k", _abi.SAMPLER_HAS_TOP_K, int),
    "max_new_tokens": ("max_new_tokens", _abi.SAMPLER_HAS_MAX_NEW_TOKENS, int),
    "max_tokens": ("max_new_tokens", _abi.SAMPLER_HAS_MAX_NEW_TOKENS, int),
}
OTHER_OPTIONS = ("speed", "normalize")
VOICE_FILE_LIMIT = 64 << 20  # as kitten-tts-native --load-voice

# Engine modes, in increasing order of what they allow (and of the GPU memory they use).
BUILTIN, CUSTOM, CLONING = 0, 1, 2
_MODE_NAMES = {BUILTIN: "built-in voices", CUSTOM: "saved voices", CLONING: "voice cloning"}


class Voice:
    """A cloned or loaded voice. It holds the saved-voice artifact, which works only with the model it was made
    with (the same prepared model revision)."""

    def __init__(self, artifact: bytes, name: str = ""):
        self.artifact = bytes(artifact)
        self.name = name
        self.key = hashlib.sha256(self.artifact).hexdigest()

    def __repr__(self) -> str:
        return f"Voice(name={self.name!r}, bytes={len(self.artifact)})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Voice) and other.key == self.key

    def __hash__(self) -> int:
        return hash(self.key)


def _enum(value, table: dict, what: str) -> int:
    if not isinstance(value, str) or value.lower() not in table:
        raise ValueError(f"{what} must be one of {', '.join(table)}; got {value!r}")
    return table[value.lower()]


def request_params(text: str, voice_id: int, *, stream: bool, seed: Optional[int] = None,
                   preset: Optional[str] = None, **options) -> _abi.RequestParams:
    """The kt2n_request_params of one generate()/stream() call (kept alive by the returned struct's _keep)."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, not {type(text).__name__}")
    unknown = sorted(set(options) - set(SAMPLING) - set(OTHER_OPTIONS))
    if unknown:
        raise TypeError(f"unknown option(s) {', '.join(unknown)}; the options are seed, preset, "
                        f"{', '.join(SAMPLING)}, {', '.join(OTHER_OPTIONS)}")
    if "max_tokens" in options and "max_new_tokens" in options:
        raise TypeError("pass max_tokens or max_new_tokens, not both")
    encoded = text.encode("utf-8")
    p = _abi.RequestParams()
    p.struct_size = ctypes.sizeof(_abi.RequestParams)
    p.text = encoded
    p.text_len = len(encoded)
    p.voice = voice_id
    p.speed = float(options.get("speed", 1.0))
    p.format = _abi.FORMAT_PCM  # the same 16-bit samples kitten-tts-native writes (levelled and peak-normalised)
    p.output = _abi.OUTPUT_STREAM if stream else _abi.OUTPUT_COMPLETE
    p.speech_mode = _abi.SPEECH_MODE_DEFAULT
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 1 << 64:
            raise ValueError(f"seed must be an integer from 0 to 2**64-1; got {seed!r}")
        p.flags |= _abi.REQ_SEED
        p.seed = seed
    if options.get("normalize") is False:  # None (like the sampling options) means the default: normalise
        p.flags |= _abi.REQ_NORMALIZE_OFF
    if preset is not None:
        p.sampler.mode = _enum(preset, PRESETS, "preset")
        p.sampler.present |= _abi.SAMPLER_HAS_MODE
    for key, (name, bit, kind) in SAMPLING.items():
        if key not in options or options[key] is None:
            continue
        value = options[key]
        if kind is int and (isinstance(value, bool) or not isinstance(value, (int, np.integer))):
            raise TypeError(f"{key} must be an integer; got {value!r}")
        setattr(p.sampler, name, kind(value))
        p.sampler.present |= bit
    p._keep = encoded  # the struct points into this buffer
    return p


class KittenTTS2:
    """Kitten TTS 2 on this machine's GPU.

    On first use it downloads KittenML/kitten-tts-2 (the latest revision, or ``revision``) and prepares it into
    ``<cache_dir or ~/.cache/kitten-inference/tts2>/<model sha>/``; later uses reuse that directory. It then starts
    the engine on GPU ``device`` with the same settings as the ``kitten-tts2`` command: the automatic memory profile
    and a voice cache that keeps the voice in use, so small GPUs work.

    ``memory_profile``: "auto" (default), "default" or "low". ``voice_cache``: None (as kitten-tts2: only the voices
    in use), "all", "lru" or "none". ``backend``: "cuda" or "rocm" (default: the installed engine, or
    $KITTEN_TTS2_BACKEND).
    """

    sample_rate = SAMPLE_RATE

    def __init__(self, revision: Optional[str] = None, device: int = 0, memory_profile: str = "auto",
                 voice_cache: Optional[str] = None, cache_dir: Optional[Union[str, os.PathLike]] = None,
                 backend: Optional[str] = None):
        if isinstance(device, bool) or not isinstance(device, int) or device < 0:
            raise ValueError(f"device must be a GPU number (0, 1, ...); got {device!r}")
        self._memory_profile = _enum(memory_profile, MEMORY_PROFILES, "memory_profile")
        self._voice_cache = None if voice_cache is None else _enum(voice_cache, VOICE_CACHES, "voice_cache")
        self._device = device
        self._revision = revision
        self._cache_dir = cache_dir
        self._lock = threading.RLock()
        self._engine: Optional[Engine] = None
        self._mode = BUILTIN
        self._voice_ids: Dict[str, int] = {}     # built-in name -> id, for the current engine
        self._custom_ids: Dict[str, int] = {}    # Voice.key -> id, for the current engine
        self._custom_users: Dict[int, str] = {}  # request id -> Voice.key of the requests that used a custom voice
        self._last_voice = 0
        self._max_prefix_ids = 0
        self._verified = set()  # model files whose SHA-256 matched in this process
        self.backend = _backend.find(backend)
        try:
            self._lib = _abi.load(self.backend.lib)
        except _abi.LibraryError as exc:
            if self.backend.name == "rocm" and "libamdhip64" in str(exc):
                raise _abi.LibraryError(f"{exc}. The AMD engine needs the ROCm 7 runtime (libamdhip64.so.7) "
                                        "installed on the system") from None
            raise
        self.model_dir = _prepare.ensure(text_data_tool=self.backend.text_data_tool, revision=revision,
                                         cache_dir=cache_dir)
        self._bundle = _bundle.load(self.model_dir)
        self._start(BUILTIN)

    # -- engine lifetime ---------------------------------------------------------------------------------------
    def _config(self, mode: int, bundle: _bundle.Bundle) -> _abi.EngineConfig:
        c = _abi.EngineConfig()
        c.struct_size = ctypes.sizeof(_abi.EngineConfig)
        c.abi_version = _abi.ABI_VERSION
        c.device = self._device
        c.max_streams = 1
        keep = [str(p).encode("utf-8") for p in (bundle.lm, bundle.decoder, bundle.voices, bundle.tokenizer,
                                                  bundle.text_data)]
        c.lm_file, c.s3gen_file, c.voices_file, c.tokenizer_json, c.text_data_dir = keep
        c.memory_profile = self._memory_profile
        if mode == BUILTIN:
            c.flags = _abi.CFG_INFERENCE_ONLY
        else:
            c.max_custom_voices = 1
        if mode == CUSTOM and self._max_prefix_ids > 0:
            # as kitten-tts-native --load-voice: the low-memory fit plans context for this voice's prompt
            c.custom_voice_max_prefix_ids = self._max_prefix_ids
        if mode == CLONING:
            if bundle.cloning_dir is None:
                raise RuntimeError(f"{bundle.root} has no voice cloning files")
            cloning = str(bundle.cloning_dir).encode("utf-8")
            keep.append(cloning)
            c.voice_build_dir = cloning
            c.inline_voice_lru = 1
            c.clone_slots = 1
        if self._voice_cache is None:
            # kitten-tts-native's defaults: one voice prepared on first use; NONE (only voices in use) with
            # custom voices, which also lets a saved voice load on 4 GB cards.
            if mode == BUILTIN:
                c.voice_cache_mode, c.voice_cache_max_voices = _abi.VOICE_CACHE_LRU, 1
            else:
                c.voice_cache_mode = _abi.VOICE_CACHE_NONE
        else:
            c.voice_cache_mode = self._voice_cache
        c._keep = keep
        return c

    def _start(self, mode: int) -> None:
        """(Re)creates the engine for ``mode``; the SHA-256 checks of the model files run beside the start."""
        bundle = self._bundle
        if mode == CLONING and not bundle.has_cloning:
            self.model_dir = _prepare.ensure(text_data_tool=self.backend.text_data_tool, revision=self._revision,
                                             cache_dir=self._cache_dir, with_cloning=True)
            bundle = self._bundle = _bundle.load(self.model_dir, need_cloning=True)
        elif mode == CLONING:
            bundle = self._bundle = _bundle.load(self.model_dir, need_cloning=True)
        config = self._config(mode, bundle)
        previous = None
        if self._engine is not None:
            if self._engine.busy():
                raise RuntimeError("finish or close the running stream() first: switching to "
                                   f"{_MODE_NAMES[mode]} restarts the engine")
            print(f"kitten-tts2: restarting the engine for {_MODE_NAMES[mode]}", file=sys.stderr, flush=True)
            previous = self._mode
            self._engine.destroy()
            self._engine = None
        self._voice_ids.clear()
        self._custom_ids.clear()
        self._custom_users.clear()
        self._last_voice = 0
        try:
            self._engine = self._create(config, bundle)
            self._mode = mode
        except BaseException as exc:
            if previous is None or not isinstance(exc, Exception):  # Ctrl-C: stop here, the object is closed
                raise
            try:  # keep a working engine: start the previous one again, then report
                self._engine = self._create(self._config(previous, self._bundle), self._bundle)
                self._mode = previous
            except Exception as again:
                raise RuntimeError(f"{exc}; the engine for {_MODE_NAMES[previous]} could not be started again "
                                   f"either ({again}), so this KittenTTS2 is closed: create a new one") from exc
            raise

    def _create(self, config: _abi.EngineConfig, bundle: _bundle.Bundle) -> Engine:
        """Creates the engine while the files' SHA-256 checks run beside it (each file is checked once)."""
        todo = [f for f in bundle.files if f[0] not in self._verified]
        verifier = _bundle.Verifier(bundle, files=todo)
        engine = None
        try:
            engine = Engine(self._lib, config)
            verifier.result()
        except _bundle.BundleError as exc:
            if engine is not None:
                engine.destroy()
            raise _bundle.BundleError(f"{exc}. Delete {self.model_dir} and run again to prepare it anew") from None
        except BaseException:
            if engine is not None:  # e.g. Ctrl-C while the checks finish: do not keep an engine nobody owns
                engine.destroy()
            try:
                verifier.result()
            except _bundle.BundleError as exc:  # a damaged file explains a failed start best
                raise _bundle.BundleError(f"{exc}. Delete {self.model_dir} and run again to prepare it anew") \
                    from None
            raise
        self._verified.update(f[0] for f in todo)
        return engine

    def _ensure_mode(self, mode: int) -> Engine:
        if self._engine is None:
            raise RuntimeError("this KittenTTS2 is closed")
        if mode > self._mode:
            self._start(mode)
        return self._engine

    def close(self) -> None:
        """Stops the engine and frees its GPU memory, except the CUDA context, which stays until the process exits
        (2.1 GB on an RTX 3060; a new KittenTTS2 reuses it). The object cannot be used afterwards."""
        with self._lock:
            engine, self._engine = self._engine, None
            if engine is not None:
                engine.destroy()

    def __enter__(self) -> "KittenTTS2":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -- information -------------------------------------------------------------------------------------------
    def list_voices(self) -> List[str]:
        """The built-in voice names, in catalogue order (pass one as ``voice=``)."""
        return [vid for vid, _aliases, _language in _bundle.public_voices(self._bundle.voices)]

    @property
    def default_voice(self) -> str:
        return self._bundle.default_voice

    @property
    def can_clone(self) -> bool:
        """Whether this engine build includes voice cloning (CUDA engine). The GPU also needs about 11 GB free
        (measured: a 12 GB RTX 3060 clones; smaller cards load saved voices only)."""
        return self.backend.can_clone and _abi.CAP_VOICE_CLONING in _abi.text(self._lib.kt2n_build_info()).split()

    def build_info(self) -> str:
        return _abi.text(self._lib.kt2n_build_info())

    def stats(self) -> str:
        """The engine's health snapshot (JSON text)."""
        with self._lock:
            return self._ensure_mode(BUILTIN).stats_json()

    # -- voices ------------------------------------------------------------------------------------------------
    def _voice_id(self, voice) -> int:
        """Resolves ``voice`` (a built-in name or a Voice) to an id of the current engine (lock held)."""
        if isinstance(voice, Voice):
            engine = self._ensure_mode(CUSTOM)
            vid = self._custom_ids.get(voice.key)
            if vid is None:
                self._drop_custom_voices(engine)  # room for one custom voice: drop the previous one
                try:
                    vid = engine.voice_load(voice.artifact)
                except EngineError as exc:
                    ids = _prefix_ids(self._lib, voice.artifact)
                    if exc.status != _abi.ERR_GPU_MEMORY_LIMIT or self._mode != CUSTOM or ids <= self._max_prefix_ids:
                        raise _voice_error(exc) from None
                    # the low-memory fit planned for a shorter voice prompt: start again planning for this one
                    self._max_prefix_ids = ids
                    self._start(CUSTOM)
                    engine = self._engine
                    vid = engine.voice_load(voice.artifact)
                self._custom_ids[voice.key] = vid
            return vid
        if not isinstance(voice, str) or not voice:
            raise TypeError(f"voice must be a voice name (see list_voices()) or a Voice; got {voice!r}")
        engine = self._ensure_mode(BUILTIN)
        vid = self._voice_ids.get(voice)
        if vid is None:
            try:
                vid = engine.voice_lookup(voice)
            except EngineError as exc:
                if exc.status == _abi.ERR_INVALID_VOICE:
                    raise ValueError(f"unknown voice {voice!r}; list_voices() gives the names") from None
                raise
            self._voice_ids[voice] = vid
        return vid

    def _drop_custom_voices(self, engine: Engine) -> None:
        """Evicts the loaded custom voice (the engine has room for one), unless a running request still uses it.
        Requests that were stopped (a closed stream) are waited for: they release the voice within milliseconds."""
        for key, vid in list(self._custom_ids.items()):
            if any(k == key and engine.is_live(i) for i, k in list(self._custom_users.items())):
                raise RuntimeError("the previous cloned or loaded voice is still in use by a running stream(); "
                                   "finish or close it before using another one")
            deadline = time.monotonic() + 10
            while engine.voice_info(vid).refs > 0 and time.monotonic() < deadline:
                engine.pump(20)
            engine.voice_evict(vid)
            del self._custom_ids[key]
        self._custom_users = {i: k for i, k in self._custom_users.items() if engine.is_live(i)}

    def load_voice(self, path: Union[str, os.PathLike]) -> Voice:
        """Reads a voice saved by save_voice() (or kitten-tts2 --save-voice) and loads it into the engine."""
        path = Path(path)
        size = path.stat().st_size
        if size > VOICE_FILE_LIMIT:
            raise ValueError(f"voice file must be at most 64 MiB: {path}")
        voice = Voice(path.read_bytes(), name=path.stem)
        ids = _prefix_ids(self._lib, voice.artifact)
        if ids < 0:
            raise ValueError(f"not a Kitten TTS 2 voice file: {path}")
        with self._lock:
            if self._mode == BUILTIN:
                self._max_prefix_ids = max(self._max_prefix_ids, ids)
            self._voice_id(voice)
        return voice

    def save_voice(self, voice: Voice, path: Union[str, os.PathLike]) -> Path:
        """Writes ``voice`` to ``path`` (atomically). It loads again with load_voice() on the same model."""
        if not isinstance(voice, Voice):
            raise TypeError("save_voice() takes a Voice from clone_voice() or load_voice()")
        path = Path(path)
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
        try:
            tmp.write_bytes(voice.artifact)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        return path

    def clone_voice(self, reference_audio_path: Union[str, os.PathLike], transcript: str,
                    language: str = "en") -> Voice:
        """Makes a voice from a 2-30 s recording of one speaker (any format the system ffmpeg decodes, at most
        10 MiB) and the exact words spoken in it. Needs the CUDA engine and about 11 GB of free GPU memory; the first
        call adds the cloning files to the prepared model and restarts the engine with cloning."""
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError("transcript must be the words spoken in the recording")
        if not self.can_clone:
            raise RuntimeError(f"voice cloning needs the NVIDIA (CUDA) engine; the installed {self.backend.distribution} "
                               "has no voice cloning. Clone and save the voice on an NVIDIA GPU with 12 GB or more, "
                               "then load it here with load_voice()")
        pcm = np.ascontiguousarray(_audio.decode_reference(reference_audio_path), dtype=np.float32)
        with self._lock:
            try:
                engine = self._ensure_mode(CLONING)
            except EngineError as exc:  # the engine with the cloning model did not fit (the previous one is back)
                raise _clone_error(exc, starting=True) from None
            self._drop_custom_voices(engine)
            params = _abi.VoiceBuildParams()
            params.struct_size = ctypes.sizeof(_abi.VoiceBuildParams)
            params.flags = _abi.VB_SAVED | _abi.VB_DECODED_PCM
            params.pcm = pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            params.pcm_samples = pcm.size
            params.pcm_rate = SAMPLE_RATE
            encoded_transcript = transcript.encode("utf-8")
            encoded_language = language.encode("utf-8")
            params.reference_transcript = encoded_transcript
            params.language = encoded_language
            try:
                job = engine.voice_build(params)
            except EngineError as exc:
                raise _clone_error(exc) from None
            built = None
            try:
                for ev in engine.events(job):
                    if ev.kind in (_abi.EVENT_VOICE_BUILT, _abi.EVENT_DONE):
                        built = ev
            except BaseException:
                engine.abandon(job)
                raise
            if built is None or built.status != _abi.OK or not built.voice:
                status = built.status if built is not None and built.status != _abi.OK else _abi.ERR_INTERNAL
                raise _clone_error((built.error if built is not None else None)
                                   or EngineError(status, "the voice build returned no voice"))
            artifact = engine.voice_export(built.voice)
            voice = Voice(artifact, name=Path(reference_audio_path).stem)
            self._custom_ids[voice.key] = built.voice
            return voice

    # -- speech ------------------------------------------------------------------------------------------------
    def _submit(self, text: str, voice, stream: bool, seed, preset, options) -> tuple:
        with self._lock:
            vid = self._voice_id(voice)
            engine = self._engine
            params = request_params(text, vid, stream=stream, seed=seed, preset=preset, **options)
            if vid != self._last_voice:
                engine.voice_prepare(vid)  # like kitten-tts-native: first audio without the voice preparation
                self._last_voice = vid
            try:
                ident = engine.submit(params)
            except EngineError as exc:
                raise _request_error(exc) from None
            if isinstance(voice, Voice):
                if len(self._custom_users) > 64:  # keep only the running ones
                    self._custom_users = {i: k for i, k in self._custom_users.items() if engine.is_live(i)}
                self._custom_users[ident] = voice.key
            return engine, ident

    def generate(self, text: str, voice: Union[str, Voice] = "Bruno", stream: bool = False,
                 seed: Optional[int] = None, preset: Optional[str] = None, **sampling) -> np.ndarray:
        """Speaks ``text`` and returns the audio: float32, 24 kHz, mono.

        ``stream=False`` (default) renders each sentence as a whole, as ``kitten-tts2 --stream false``; with a
        ``seed`` the same input then gives the same audio on the same GPU. ``stream=True`` uses the streaming path
        (as stream()) and returns all of it. ``preset``: "stable" or "expressive". ``sampling``: temperature, top_p,
        top_k, min_p, max_new_tokens (also max_tokens), speed (0.25-4.0) and normalize (False speaks the text as
        given, without number, date and symbol expansion).
        """
        engine, ident = self._submit(text, voice, stream, seed, preset, sampling)
        chunks = []
        for data in self._audio(engine, ident):
            chunks.append(data)
        return _audio.to_float(b"".join(chunks))

    def stream(self, text: str, voice: Union[str, Voice] = "Bruno", seed: Optional[int] = None,
               preset: Optional[str] = None, **sampling) -> Iterator[np.ndarray]:
        """Speaks ``text`` and yields the audio as it is generated, in float32 24 kHz mono chunks. Closing the
        iterator early (break) cancels the rest."""
        engine, ident = self._submit(text, voice, True, seed, preset, sampling)
        events = self._audio(engine, ident)
        carry = b""
        try:
            for data in events:
                data = carry + data
                cut = len(data) // 2 * 2
                carry = data[cut:]
                if cut:
                    yield _audio.to_float(data[:cut])
        finally:
            events.close()  # an early close cancels the request

    def _audio(self, engine: Engine, ident: int) -> Iterator[bytes]:
        done = False
        try:
            for ev in engine.events(ident):
                if ev.kind == _abi.EVENT_AUDIO and ev.data:
                    yield ev.data
                elif ev.kind == _abi.EVENT_DONE:
                    done = True
                    if ev.status == _abi.ERR_CANCELLED and engine.closing:
                        raise RuntimeError("this KittenTTS2 was closed while the request was running")
                    if ev.status != _abi.OK:
                        raise _request_error(ev.error or EngineError(ev.status))
        finally:
            if not done:
                engine.abandon(ident)

    def save_wav(self, path: Union[str, os.PathLike], audio) -> Path:
        """Writes ``audio`` (from generate(), or the joined chunks of stream()) as a 16-bit 24 kHz mono WAV file."""
        if isinstance(audio, (list, tuple)):
            audio = np.concatenate([np.asarray(a, dtype=np.float32).reshape(-1) for a in audio]) if audio \
                else np.zeros(0, np.float32)
        return _audio.save_wav(path, audio, SAMPLE_RATE)


def _prefix_ids(lib, artifact: bytes) -> int:
    return int(lib.kt2n_voice_artifact_lm_prefix_ids(artifact, len(artifact)))


def _voice_error(exc: EngineError) -> Exception:
    if exc.status == _abi.ERR_VOICE_ARTIFACT_VERSION_MISMATCH:
        return ValueError(f"this voice was saved with another prepared model: {exc.message}")
    return exc


def _clone_error(exc: EngineError, starting: bool = False) -> Exception:
    if starting or exc.status == _abi.ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE:
        return RuntimeError(f"this GPU cannot clone voices: {exc.message}. Cloning needs about 11 GB of free GPU "
                            "memory (a 12 GB card or larger); clone and save the voice there, then load it here")
    if exc.status in (_abi.ERR_INVALID_AUDIO, _abi.ERR_INVALID_AUDIO_DURATION, _abi.ERR_INVALID_AUDIO_SIGNAL,
                      _abi.ERR_INVALID_REFERENCE_TRANSCRIPT):
        return ValueError(exc.message)
    return exc


def _request_error(exc: EngineError) -> Exception:
    if exc.status in (_abi.ERR_INVALID_REQUEST, _abi.ERR_INVALID_INPUT, _abi.ERR_REQUEST_TOO_LARGE):
        return ValueError(f"{exc.message}" + (f" ({exc.param})" if exc.param else ""))
    return exc
