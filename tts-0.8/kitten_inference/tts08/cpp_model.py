import contextlib
import json
import os
import platform
import struct
import sys
import threading

import espeakng_loader
import numpy as np
import phonemizer
import soundfile as sf
from phonemizer.backend.espeak.wrapper import EspeakWrapper

from .preprocess import TextPreprocessor, chunk_text, normalize_text

_HERE = os.path.dirname(os.path.abspath(__file__))
# Windows: the Microsoft Visual C++ runtime files that espeak-ng.dll (from espeakng_loader) imports by these names.
# Python ships only some of them and a fresh Windows has none; phonemizer then reports "espeak not installed".
MSVC_RUNTIME = ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")  # in load order
_msvc_loaded = []  # handles of the copies loaded from this package (kept for the life of the process)


def _load_msvc_runtime(folder=os.path.join(_HERE, "_msvc"), windll=None):
    """Loads this package's copy of each MSVC_RUNTIME file (the Windows wheels carry one in _msvc/) that Windows
    cannot load by name. An installed runtime is always used first: it can be newer, and other modules in the
    process share it. Returns the names loaded from ``folder``."""
    if windll is None:
        if sys.platform != "win32":
            return []
        import ctypes
        windll = ctypes.WinDLL
    loaded = []
    for name in MSVC_RUNTIME:
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        try:
            windll(name)  # already loaded, or found next to python.exe, in System32 or an added DLL folder
            continue
        except OSError:
            pass
        try:
            _msvc_loaded.append(windll(path))
            loaded.append(name)
        except OSError:
            pass  # espeak's own error follows; the README names the redistributable
    return loaded


_load_msvc_runtime()
EspeakWrapper.set_library(espeakng_loader.get_library_path())
os.environ["ESPEAK_DATA_PATH"] = espeakng_loader.get_data_path()

SAMPLE_RATE = 24000
STYLE_DIM = 256
# The reference ONNX path (kittenml 0.9.6) ends the ids with "…" (10) then the pad (0), and drops the last
# 5000 samples of every chunk. Both are copied here so the two paths give the same audio length.
END_TOKEN = 10
TAIL_TRIM_SAMPLES = 5000
DEFAULT_VOICE = "expr-voice-5-m"
DEFAULT_NATIVE_VOICE = "expr-voice-2-f"

DEFAULT_VOICE_ALIASES = {
    "Bella": "expr-voice-2-f",
    "Jasper": "expr-voice-2-m",
    "Luna": "expr-voice-3-f",
    "Bruno": "expr-voice-3-m",
    "Rosie": "expr-voice-4-f",
    "Hugo": "expr-voice-4-m",
    "Kiki": "expr-voice-5-f",
    "Leo": "expr-voice-5-m",
}


def basic_english_tokenize(text):
    """Basic English tokenizer that splits on whitespace and punctuation."""
    import re

    return re.findall(r"\w+|[^\w\s]", text)


class TextCleaner:
    def __init__(self, dummy=None):
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»"" '
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

        symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
        self.word_index_dictionary = {symbols[i]: i for i in range(len(symbols))}

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                pass
        return indexes


SKIP_CPU_CHECK = "KITTEN_TTS08_SKIP_CPU_CHECK"
_CPU_FEATURES = {
    "avx2": "AVX2",
    "fma": "FMA",
    "asimddp": "the ARMv8.2 dot-product instructions (asimddp)",
}
_CPU_HINT = {
    "avx2": "Intel CPUs have them since 2013, AMD CPUs since 2015",
    "asimddp": "a Raspberry Pi 5 has them, a Raspberry Pi 4 does not",
}


GLIBC_FLOOR = (2, 29)


def _glibc_version():
    """(major, minor) of the GNU C library, or None when this is not glibc or it cannot be told."""
    if not (sys.platform.startswith("linux") or sys.platform == "android"):
        return None
    try:
        name = os.confstr("CS_GNU_LIBC_VERSION")  # e.g. "glibc 2.31"
    except (ValueError, OSError, AttributeError):
        return None
    if not name or not name.startswith("glibc"):
        return None
    try:
        major, minor = name.split()[1].split(".")[:2]
        return (int(major), int(minor))
    except (IndexError, ValueError):
        return None


def _check_glibc(version=None):
    """espeak-ng (from espeakng_loader) needs glibc 2.29 or newer on Linux; an older glibc otherwise fails later
    with a confusing "espeak not installed on your system". Refuse it here with a clear message instead. pip can
    install the wheel on an older glibc because espeakng_loader's x86_64 wheel is tagged manylinux_2_17."""
    v = _glibc_version() if version is None else version
    if v is not None and v < GLIBC_FLOOR:
        raise RuntimeError(
            f"This system's glibc is {v[0]}.{v[1]}, older than the {GLIBC_FLOOR[0]}.{GLIBC_FLOOR[1]} that "
            "kitten-inference-tts08 needs (the espeak-ng library it uses for text). On glibc older than 2.29 "
            "(for example RHEL/Rocky 8, Debian 10, Ubuntu 18.04 or Amazon Linux 2) it would otherwise fail with "
            "\"espeak not installed on your system\". Use a Linux with glibc 2.29 or newer."
        )


def _pos_weight_rows(weights_path, name=b"bert_embed.pos_weight"):
    """The number of position rows in the text encoder's position table (bert_embed.pos_weight), read from the W1X1
    weights file, or None when it cannot be told. That is the most phoneme tokens (ids) one chunk may have; past it
    the engine reports a weights-size mismatch, which misleads users into re-downloading an intact model."""
    try:
        with open(weights_path, "rb") as f:
            if f.read(4) != b"W1X1":
                return None
            _version, count = struct.unpack("<II", f.read(8))
            offset = None
            for _ in range(count):
                entry = f.read(80)
                if len(entry) < 80:
                    return None
                nm = entry[:64].split(b"\0", 1)[0]
                if nm == name or (offset is None and nm.endswith(b".pos_weight")):
                    offset = struct.unpack("<QII", entry[64:80])[0]
                    if nm == name:
                        break
            if offset is None:
                return None
            f.seek(offset)
            head = f.read(4)
            if len(head) < 4:
                return None
            _dtype, _flags, ndim = struct.unpack("<BBH", head)
            if ndim < 1:
                return None
            rows = struct.unpack("<i", f.read(4))[0]
            return rows if rows > 0 else None
    except (OSError, struct.error):
        return None


def _cpu_requirements(path=os.path.join(_HERE, "_cpu_requirements.txt")):
    """The instruction sets the engine was compiled for, as the build wrote them ("native": the build machine's own
    CPU, which cannot be checked here). Empty when the file is missing (a build without it)."""
    try:
        with open(path, encoding="ascii") as f:
            return f.read().split()
    except OSError:
        return []


def _cpuinfo_features(text, machine):
    """The feature names /proc/cpuinfo lists for the first CPU ("flags" on x86, "Features" on Arm)."""
    key = "features" if machine in ("aarch64", "arm64", "armv8l") else "flags"
    for line in text.splitlines():
        name, sep, value = line.partition(":")
        if sep and name.strip().lower() == key:
            return set(value.split())
    return None


def _cpu_features():
    """The features of this CPU that the engine's builds can need, or None when they cannot be told."""
    machine = platform.machine().lower()
    if sys.platform.startswith("linux") or sys.platform == "android":
        if machine in ("aarch64", "arm64"):
            try:  # the kernel's view, also inside containers: getauxval(AT_HWCAP) & HWCAP_ASIMDDP
                import ctypes
                libc = ctypes.CDLL(None)
                libc.getauxval.restype = ctypes.c_ulong
                libc.getauxval.argtypes = [ctypes.c_ulong]
                return {"asimddp"} if libc.getauxval(16) & (1 << 20) else set()
            except (OSError, AttributeError):
                pass
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                return _cpuinfo_features(f.read(), machine)
        except OSError:
            return None
    if sys.platform == "win32" and machine in ("amd64", "x86_64"):
        import ctypes
        present = ctypes.windll.kernel32.IsProcessorFeaturePresent
        if present(40):  # PF_AVX2_INSTRUCTIONS_AVAILABLE (every AVX2 CPU also has FMA)
            return {"avx2", "fma"}
        # Windows versions that do not know these feature numbers answer "no" for all of them; SSE4.2 (38) is on
        # every x86-64 CPU that runs a current Windows, so "no" for it means this Windows cannot tell.
        return set() if present(38) else None
    if sys.platform == "darwin" and machine == "x86_64":
        import ctypes
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        libc.sysctlbyname.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                      ctypes.c_void_p, ctypes.c_size_t]
        have = set()
        for feature, key in (("avx2", b"hw.optional.avx2_0"), ("fma", b"hw.optional.fma")):
            value, size = ctypes.c_int(0), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
            if libc.sysctlbyname(key, ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
                return None
            if value.value:
                have.add(feature)
        return have
    return None


def _check_cpu(requirements=None, features=None):
    """Refuses to load the engine on a CPU without the instructions it was compiled for: it would stop the whole
    Python process with "Illegal instruction" instead of raising an error."""
    need = [f for f in (_cpu_requirements() if requirements is None else requirements) if f != "native"]
    if not need or os.environ.get(SKIP_CPU_CHECK):
        return
    have = _cpu_features() if features is None else features
    if have is None:
        return
    lacking = [f for f in need if f not in have]
    if lacking:
        hint = next((_CPU_HINT[f] for f in lacking if f in _CPU_HINT), "")
        raise RuntimeError(
            f"This CPU does not have {' and '.join(_CPU_FEATURES.get(f, f) for f in lacking)}, which the native "
            f"engine of kitten-inference-tts08 is built for{f' ({hint})' if hint else ''}. Loading it would stop "
            f"Python with \"Illegal instruction\". If this check is wrong, set {SKIP_CPU_CHECK}=1."
        )


def _import_engine():
    _check_cpu()
    try:
        from . import _engine
    except ImportError as exc:
        raise ImportError(
            "The native engine (kitten_inference.tts08._engine) is missing or does not match "
            'this Python and platform. Reinstall it with: pip install "kitten-inference[tts-0.8]"'
        ) from exc
    return _engine


class _EngineGate:
    """Lets engine calls run at the same time only when that is safe.

    The engine has one process-wide backend, which a model's kernels use while it loads and runs, and the
    Metal backend records all GPU work into one shared command buffer. So CPU calls may overlap (one model
    per thread), while a Metal call, or a call on another backend than the running ones, waits until the
    engine is idle. Each call sets its own model's backend first.

    Calls get in in the order they arrive: a call waits while an earlier one is still waiting. So a Metal call
    is not passed by CPU calls that come after it, and CPU calls are not held back for ever by Metal calls that
    keep queuing (two threads generating on Metal always have one call waiting); CPU calls that arrive one
    after another still run together.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._running = 0
        self._exclusive = False
        self._backend = None
        self._next_ticket = 0  # handed to each arriving call
        self._serving = 0  # the earliest ticket still waiting
        self._abandoned = set()  # tickets whose wait was interrupted

    def _must_wait(self, backend, exclusive):
        if self._running == 0:
            return False
        return exclusive or self._exclusive or self._backend != backend

    def _advance(self):
        self._serving += 1
        while self._serving in self._abandoned:
            self._abandoned.discard(self._serving)
            self._serving += 1
        self._cond.notify_all()

    @contextlib.contextmanager
    def use(self, engine, backend):
        exclusive = backend != "cpu"
        with self._cond:
            ticket = self._next_ticket
            self._next_ticket += 1
            try:
                while ticket != self._serving or self._must_wait(backend, exclusive):
                    self._cond.wait()
            except BaseException:  # e.g. KeyboardInterrupt while waiting: give up the place in the queue
                if ticket == self._serving:
                    self._advance()
                else:
                    self._abandoned.add(ticket)
                raise
            try:
                if not self._running:
                    if not engine.set_backend(backend):
                        raise ValueError(
                            f"Backend {backend!r} is not available in this build of kitten-inference-tts08."
                        )
                    self._backend = backend
                    self._exclusive = exclusive
                self._running += 1
            finally:
                self._advance()
        try:
            yield
        finally:
            with self._cond:
                self._running -= 1
                self._cond.notify_all()


_ENGINE_GATE = _EngineGate()


def _has_int8_layers(arch_path):
    # int8 models use *_int8 layers (conv1d_int8, seqgemm_int8, attention_int8).
    with open(arch_path, encoding="utf-8") as f:
        nodes = json.load(f).get("nodes", [])
    return any(str(node.get("op", "")).endswith("_int8") for node in nodes)


def _default_backend(engine, arch_path):
    # Apple Silicon: Metal when a Metal device is available, except for int8 models, which run several times
    # faster on the CPU than on Metal. Everywhere else: the CPU.
    if sys.platform == "darwin" and platform.machine() == "arm64" and not _has_int8_layers(arch_path):
        try:
            with _ENGINE_GATE.use(engine, "metal"):
                return "metal"
        except ValueError:
            pass
    return "cpu"


def _coerce_voice_pack(values):
    pack = np.asarray(values, dtype=np.float32)
    if pack.size % STYLE_DIM != 0:
        raise ValueError(
            f"Voice style data has {pack.size} floats, which is not divisible by {STYLE_DIM}."
        )
    return pack.reshape(-1, STYLE_DIM)


def _load_voice_bin(path):
    return _coerce_voice_pack(np.fromfile(path, dtype=np.float32))


class KittenTTS_1_Cpp:
    def __init__(
        self,
        arch_path,
        weights_path,
        voices_path=None,
        voice_paths=None,
        speed_priors=None,
        voice_aliases=None,
        backend=None,
        style_index="token_count",
        sample_rate=SAMPLE_RATE,
    ):
        """Initialize KittenTTS with the native engine."""
        _check_glibc()  # before espeak-ng is loaded below, so an old glibc fails with a clear message
        self.arch_path = arch_path
        self.weights_path = weights_path
        self.sample_rate = sample_rate
        self.style_index = style_index
        # Most phoneme tokens one chunk may have (the model's position table); past it the engine reports a
        # weights-size mismatch, so check it in Python and say what is really wrong.
        self._max_tokens = _pos_weight_rows(weights_path)

        aliases = dict(DEFAULT_VOICE_ALIASES)
        aliases.update(voice_aliases or {})
        self.voice_aliases = aliases
        self.speed_priors = speed_priors or {}
        self.preprocessor = TextPreprocessor(remove_punctuation=False)
        self.phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )
        self.text_cleaner = TextCleaner()
        # One call at a time per model: the engine model's forward reuses its own buffers and speed setting, and
        # espeak (this model's phonemizer) is not thread-safe. Different models still run at the same time.
        self._lock = threading.Lock()

        self._voices_npz = None
        self._voice_packs = {}
        self._load_voices(voices_path, voice_paths or {})
        if self._voices_npz is None and not self._voice_packs:
            raise ValueError(
                "Native KittenTTS config must provide voices as a .npz file or voice_files."
            )

        self.available_voices = sorted(self._available_voice_names())
        self.all_voice_names = [
            alias for alias, target in self.voice_aliases.items()
            if target in self.available_voices
        ] or self.available_voices

        self._mi = _import_engine()
        if backend == "amd_gpu":
            raise ValueError("The native engine does not support backend='amd_gpu'.")
        self.backend = backend or _default_backend(self._mi, arch_path)
        # The engine allocates and uploads the weights for the backend that is set while the model loads.
        with _ENGINE_GATE.use(self._mi, self.backend):
            self.session = self._mi.InferenceModel(arch_path, weights_path)

    def _load_voices(self, voices_path, voice_paths):
        if voices_path:
            if str(voices_path).endswith(".npz"):
                self._voices_npz = np.load(voices_path)
            else:
                self._voice_packs[DEFAULT_NATIVE_VOICE] = _load_voice_bin(voices_path)

        for voice, path in voice_paths.items():
            self._voice_packs[voice] = _load_voice_bin(path)

    def _available_voice_names(self):
        names = set(self._voice_packs.keys())
        if self._voices_npz is not None:
            names.update(self._voices_npz.files)
        return names

    def _resolve_voice(self, voice):
        voice = self.voice_aliases.get(voice, voice)
        if voice not in self.available_voices:
            choices = sorted(set(self.all_voice_names) | set(self.available_voices))
            raise ValueError(f"Voice {voice!r} not available. Choose from: {choices}")
        return voice

    def _phoneme_ids(self, text):
        phonemes_list = self.phonemizer.phonemize([text])
        phonemes = basic_english_tokenize(phonemes_list[0])
        phonemes = " ".join(phonemes)
        tokens = self.text_cleaner(phonemes)
        tokens.insert(0, 0)
        tokens.append(END_TOKEN)
        tokens.append(0)
        return np.asarray(tokens, dtype=np.float32)

    def _style_row(self, text, n_ids, rows):
        if rows <= 0:
            return 0
        if self.style_index == "text_length":
            return min(len(text), rows - 1)
        if self.style_index == "phoneme_count":
            return min(max(0, n_ids - 2), rows - 1)
        if self.style_index != "token_count":
            raise ValueError(
                "style_index must be one of: 'token_count', 'phoneme_count', 'text_length'."
            )
        return min(max(0, n_ids), rows - 1)

    def _style_for_voice(self, voice, text, n_ids):
        if voice in self._voice_packs:
            pack = self._voice_packs[voice]
        elif self._voices_npz is not None and voice in self._voices_npz:
            pack = _coerce_voice_pack(self._voices_npz[voice])
        else:
            raise ValueError(f"Voice {voice!r} has no style data.")

        row = self._style_row(text, n_ids, pack.shape[0])
        style = pack[row]
        return style[:128].copy(), style[128:].copy()

    def _effective_speed(self, voice, speed):
        return float(speed) * float(self.speed_priors.get(voice, 1.0))

    def _set_native_speed(self, speed):
        speed = float(speed)
        if not np.isfinite(speed) or speed <= 0.0:
            raise ValueError("speed must be a positive finite value")

        set_speed = getattr(self.session, "set_speed", None)
        if set_speed is None:
            if np.isclose(speed, 1.0):
                return
            raise RuntimeError("This build of the native engine does not support speed control.")
        set_speed(speed)

    def normalize_text(self, text, locale="en-US", return_spans=False):
        return normalize_text(text, locale=locale, return_spans=return_spans)

    def generate(self, text, voice=DEFAULT_VOICE, speed=1.0, clean_text=True):
        out_chunks = []
        if clean_text:
            text = self.preprocessor(text)
        for text_chunk in chunk_text(text):
            out_chunks.append(self.generate_single_chunk(text_chunk, voice, speed))
        if not out_chunks:
            return np.asarray([], dtype=np.float32)
        return np.concatenate(out_chunks, axis=-1)

    def generate_stream(self, text, voice=DEFAULT_VOICE, speed=1.0, clean_text=True):
        if clean_text:
            text = self.preprocessor(text)
        for text_chunk in chunk_text(text):
            yield self.generate_single_chunk(text_chunk, voice, speed)

    def generate_single_chunk(self, text, voice=DEFAULT_VOICE, speed=1.0):
        voice = self._resolve_voice(voice)
        effective_speed = self._effective_speed(voice, speed)

        with self._lock:
            ids = self._phoneme_ids(text)
            if self._max_tokens is not None and len(ids) > self._max_tokens:
                raise ValueError(
                    f"this text becomes {len(ids)} phoneme tokens in one chunk, more than the model's limit of "
                    f"{self._max_tokens}. Use shorter sentences (chunks split on sentence breaks), or "
                    "clean_text=True so numbers and symbols are normalized rather than spelled out at length."
                )
            style_dec, style_pred = self._style_for_voice(voice, text, len(ids))
            with _ENGINE_GATE.use(self._mi, self.backend):
                self._set_native_speed(effective_speed)
                outputs = self.session.forward(
                    {
                        "phoneme_ids": ids,
                        "style_dec": style_dec,
                        "style_pred": style_pred,
                    }
                )
        audio = np.asarray(outputs["t_audio"], dtype=np.float32).reshape(-1)
        return audio[:-TAIL_TRIM_SAMPLES]

    def generate_to_file(
        self,
        text,
        output_path,
        voice=DEFAULT_VOICE,
        speed=1.0,
        sample_rate=SAMPLE_RATE,
        clean_text=True,
    ):
        audio = self.generate(text, voice, speed, clean_text=clean_text)
        sf.write(output_path, audio, sample_rate)
        print(f"Audio saved to {output_path}")
