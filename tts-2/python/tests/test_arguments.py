"""generate()/stream() keywords and KittenTTS2() settings map onto the engine's C structs as kitten-tts-native does."""
import ctypes

import numpy as np
import pytest

from kitten_inference.tts2 import _abi, _api


def test_defaults():
    p = _api.request_params("Hello", 7, stream=False)
    assert p.struct_size == ctypes.sizeof(_abi.RequestParams)
    assert ctypes.string_at(p.text, p.text_len) == b"Hello"
    assert (p.voice, p.speed, p.format, p.output, p.flags) == (7, 1.0, _abi.FORMAT_PCM, _abi.OUTPUT_COMPLETE, 0)
    assert p.sampler.present == 0 and p.control_json is None


def test_unseeded_requests_get_their_own_request_id():
    # The engine seeds an unseeded request from sha256(request_id); one shared (empty) id made every unseeded
    # request sound the same.
    ids = {_api.request_params("Hello", 7, stream=False).request_id for _ in range(50)}
    assert len(ids) == 50 and all(i.startswith(b"req_") and len(i) == 36 for i in ids)


def test_stream_and_seed():
    p = _api.request_params("x", 1, stream=True, seed=42)
    assert p.output == _abi.OUTPUT_STREAM
    assert p.flags & _abi.REQ_SEED and p.seed == 42
    assert p.request_id is None  # seeded: the seed alone decides (same audio as kitten-tts2 --seed)
    big = _api.request_params("x", 1, stream=False, seed=2**64 - 1)
    assert big.seed == 2**64 - 1
    for bad in (-1, 2**64, 1.5, True, "7"):
        with pytest.raises(ValueError):
            _api.request_params("x", 1, stream=False, seed=bad)


def test_utf8_text():
    p = _api.request_params("Grüße, 你好", 1, stream=False)
    assert ctypes.string_at(p.text, p.text_len) == "Grüße, 你好".encode("utf-8")
    with pytest.raises(TypeError):
        _api.request_params(b"bytes", 1, stream=False)


@pytest.mark.parametrize("preset, mode", [("stable", _abi.SAMPLER_STABLE), ("expressive", _abi.SAMPLER_EXPRESSIVE),
                                          ("Stable", _abi.SAMPLER_STABLE)])
def test_preset(preset, mode):
    p = _api.request_params("x", 1, stream=False, preset=preset)
    assert p.sampler.present == _abi.SAMPLER_HAS_MODE and p.sampler.mode == mode


def test_bad_preset():
    with pytest.raises(ValueError, match="stable, expressive"):
        _api.request_params("x", 1, stream=False, preset="calm")


def test_sampling_fields():
    p = _api.request_params("x", 1, stream=False, temperature=0.7, top_p=0.85, min_p=0.05, top_k=40,
                            max_new_tokens=900)
    s = p.sampler
    assert s.present == (_abi.SAMPLER_HAS_TEMPERATURE | _abi.SAMPLER_HAS_TOP_P | _abi.SAMPLER_HAS_MIN_P
                         | _abi.SAMPLER_HAS_TOP_K | _abi.SAMPLER_HAS_MAX_NEW_TOKENS)
    assert (s.temperature, s.top_p, s.min_p, s.top_k, s.max_new_tokens) == (0.7, 0.85, 0.05, 40, 900)


def test_cli_spelling_max_tokens():
    p = _api.request_params("x", 1, stream=False, max_tokens=600, top_k=np.int64(3))
    assert p.sampler.max_new_tokens == 600 and p.sampler.top_k == 3
    assert p.sampler.present == _abi.SAMPLER_HAS_MAX_NEW_TOKENS | _abi.SAMPLER_HAS_TOP_K
    with pytest.raises(TypeError, match="not both"):
        _api.request_params("x", 1, stream=False, max_tokens=600, max_new_tokens=700)


def test_none_means_unset():
    p = _api.request_params("x", 1, stream=False, temperature=None, top_k=None)
    assert p.sampler.present == 0


def test_integer_fields_refuse_floats():
    with pytest.raises(TypeError, match="top_k"):
        _api.request_params("x", 1, stream=False, top_k=4.5)


def test_speed_and_normalize():
    p = _api.request_params("x", 1, stream=False, speed=1.25, normalize=False)
    assert p.speed == 1.25 and p.flags & _abi.REQ_NORMALIZE_OFF


def test_unknown_option():
    with pytest.raises(TypeError, match="unknown option.*temprature"):
        _api.request_params("x", 1, stream=False, temprature=0.5)


class _Bundle:
    root = lm = decoder = voices = tokenizer = text_data = "/m"
    cloning_dir = "/m/native/cloning"
    has_cloning = True


def _tts(**kw):
    t = _api.KittenTTS2.__new__(_api.KittenTTS2)
    t._device = kw.get("device", 0)
    t._memory_profile = _api.MEMORY_PROFILES[kw.get("memory_profile", "auto")]
    vc = kw.get("voice_cache")
    t._voice_cache = None if vc is None else _api.VOICE_CACHES[vc]
    t._max_prefix_ids = kw.get("max_prefix_ids", 0)
    return t


def test_engine_config_builtin_defaults_match_the_cli():
    c = _tts()._config(_api.BUILTIN, _Bundle())
    assert c.struct_size == ctypes.sizeof(_abi.EngineConfig) and c.abi_version == 2
    assert c.flags == _abi.CFG_INFERENCE_ONLY and c.max_streams == 1 and c.device == 0
    assert c.memory_profile == _abi.MEMORY_PROFILE_AUTO
    assert (c.voice_cache_mode, c.voice_cache_max_voices) == (_abi.VOICE_CACHE_LRU, 1)
    assert c.max_custom_voices == 0 and c.voice_build_dir is None
    assert c.reserved_ptr0 is None and c.pinned_arena_bytes == 0


def test_engine_config_saved_voice_and_cloning():
    t = _tts(max_prefix_ids=380)
    c = t._config(_api.CUSTOM, _Bundle())
    assert c.flags == 0 and c.max_custom_voices == 1 and c.voice_build_dir is None
    assert c.voice_cache_mode == _abi.VOICE_CACHE_NONE and c.custom_voice_max_prefix_ids == 380
    c = t._config(_api.CLONING, _Bundle())
    assert c.flags == 0 and c.voice_build_dir == b"/m/native/cloning"
    assert (c.clone_slots, c.inline_voice_lru, c.max_custom_voices) == (1, 1, 1)
    assert c.custom_voice_max_prefix_ids == 0  # a new clone may have a longer prompt


@pytest.mark.parametrize("profile, value", [("auto", _abi.MEMORY_PROFILE_AUTO), ("default", 0), ("low", 1)])
def test_memory_profile(profile, value):
    assert _tts(memory_profile=profile)._config(_api.BUILTIN, _Bundle()).memory_profile == value


@pytest.mark.parametrize("cache, value", [("all", 0), ("lru", 1), ("none", 2)])
def test_explicit_voice_cache(cache, value):
    c = _tts(voice_cache=cache)._config(_api.BUILTIN, _Bundle())
    assert c.voice_cache_mode == value and c.voice_cache_max_voices == 0


def test_bad_settings(no_backends):
    with pytest.raises(ValueError, match="memory_profile"):
        _api.KittenTTS2(memory_profile="tiny")
    with pytest.raises(ValueError, match="voice_cache"):
        _api.KittenTTS2(voice_cache="some")
    with pytest.raises(ValueError, match="device"):
        _api.KittenTTS2(device=-1)
