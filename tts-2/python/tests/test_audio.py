"""WAV writing, PCM conversion and the ffmpeg reference decode used by clone_voice()."""
import shutil
import subprocess
import wave

import numpy as np
import pytest

from kitten_inference.tts2 import _audio


def test_pcm16_round_trip(tmp_path):
    pcm = np.array([0, 1, -1, 32767, -32768, 12345], dtype="<i2")
    audio = _audio.to_float(pcm.tobytes())
    assert audio.dtype == np.float32 and audio[3] < 1.0 and audio[4] == -1.0
    assert _audio.to_pcm16(audio) == pcm.tobytes()
    path = _audio.save_wav(tmp_path / "a.wav", audio)
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)
        assert w.readframes(w.getnframes()) == pcm.tobytes()
    assert path.stat().st_size == 44 + pcm.nbytes  # the canonical 44-byte header, as kitten-tts-native writes


def test_to_pcm16_clips_and_checks():
    assert np.frombuffer(_audio.to_pcm16([2.0, -2.0, 1.0]), "<i2").tolist() == [32767, -32768, 32767]
    assert _audio.to_pcm16(np.zeros((5, 1))) == bytes(10)
    with pytest.raises(ValueError, match="one channel"):
        _audio.to_pcm16(np.zeros((5, 2)))
    with pytest.raises(ValueError, match="NaN"):
        _audio.to_pcm16([0.0, float("nan")])


def test_ffmpeg_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    with pytest.raises(_audio.FFmpegNotFound) as e:
        _audio.decode_reference(ref)
    assert "sudo apt install ffmpeg" in str(e.value) and "winget install ffmpeg" in str(e.value)


def test_reference_limits(tmp_path):
    with pytest.raises(_audio.AudioError, match="cannot read"):
        _audio.decode_reference(tmp_path / "missing.wav")
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(_audio.AudioError, match="empty"):
        _audio.decode_reference(empty)
    big = tmp_path / "big.wav"
    with open(big, "wb") as f:
        f.truncate((10 << 20) + 1)
    with pytest.raises(_audio.AudioError, match="10 MiB"):
        _audio.decode_reference(big)


def test_ffmpeg_arguments_match_the_reference_decode():
    a = _audio.FFMPEG_ARGS
    assert a[a.index("-i") + 1] == "pipe:0" and a[-1] == "pipe:1" and "-nostdin" in a
    assert a[a.index("-ac") + 1] == "1" and a[a.index("-ar") + 1] == "24000"
    assert a[a.index("-t") + 1] == "30.100" and a[a.index("-f") + 1] == "f32le"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_ffmpeg_decode(tmp_path):
    # 1 s of a 440 Hz tone, stereo 44.1 kHz -> mono 24 kHz float
    src = tmp_path / "tone.wav"
    t = np.arange(44100) / 44100.0
    tone = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
    with wave.open(str(src), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(np.repeat(tone, 2).tobytes())
    pcm = _audio.decode_reference(src)
    assert pcm.dtype == np.float32 and abs(pcm.size - 24000) <= 64
    assert 0.45 < np.abs(pcm).max() < 0.55
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"this is not audio" * 100)
    with pytest.raises(_audio.AudioError, match="no decodable audio"):
        _audio.decode_reference(bad)
