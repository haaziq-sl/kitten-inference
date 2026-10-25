"""WAV writing, PCM conversion and the ffmpeg reference decode used by clone_voice()."""
import os
import shutil
import subprocess
import wave

import numpy as np
import pytest

from kitten_inference.tts2 import _audio

# CI installs ffmpeg on every OS and sets KT2N_REQUIRE_FFMPEG=1: the decode tests then never skip.
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None and os.environ.get("KT2N_REQUIRE_FFMPEG") != "1",
                                  reason="ffmpeg is not installed")


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
    assert ("winget install --id Gyan.FFmpeg -e" if os.name == "nt" else "sudo apt-get install -y ffmpeg") in str(e.value)


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


def test_ffmpeg_arguments_match_the_reference_decode(tmp_path):
    a = _audio.ffmpeg_args()
    assert a[a.index("-i") + 1] == "pipe:0" and a[-1] == "pipe:1" and "-nostdin" in a
    assert a[a.index("-ac") + 1] == "1" and a[a.index("-ar") + 1] == "24000"
    assert a[a.index("-t") + 1] == "30.100" and a[a.index("-f") + 1] == "f32le"
    b = _audio.ffmpeg_args(tmp_path / "-ref.m4a")  # the by-name fallback
    assert b[b.index("-i") + 1] == os.path.abspath(tmp_path / "-ref.m4a") and b[:6] == a[:6]


@needs_ffmpeg
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
    assert 0.3 < np.abs(pcm).max() < 0.8  # ffmpeg downmixes stereo to mono at -3 dB per channel
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"this is not audio" * 100)
    with pytest.raises(_audio.AudioError, match="no decodable audio"):
        _audio.decode_reference(bad)


@needs_ffmpeg
def test_ffmpeg_decode_falls_back_to_the_file_name(tmp_path):
    # An M4A with its index at the end cannot be read from a pipe; it is then read by name.
    src = tmp_path / "tone.wav"
    t = np.arange(20 * 24000) / 24000.0  # long enough that the index at the end is out of the pipe's reach
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes((0.3 * np.sin(2 * np.pi * 330 * t) * 32767).astype("<i2").tobytes())
    m4a = tmp_path / "tone.m4a"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-c:a", "aac", str(m4a)],
                   check=True)
    piped = subprocess.run(["ffmpeg", *_audio.ffmpeg_args()], input=m4a.read_bytes(), capture_output=True)
    pcm = _audio.decode_reference(m4a)
    assert abs(pcm.size - 20 * 24000) <= 2048 and np.abs(pcm).max() > 0.1
    # Whether this ffmpeg could read that M4A from the pipe (the next test forces the by-name path either way).
    print(f"pipe decode of an M4A with its index at the end: {'ok' if piped.returncode == 0 and piped.stdout else 'no audio'}")


@needs_ffmpeg
def test_by_name_fallback_with_a_non_ascii_name(tmp_path, monkeypatch):
    # The by-name path with a real ffmpeg, whatever this ffmpeg can read from a pipe: the pipe attempt is made to
    # find no audio. The name has spaces and non-ASCII letters (Windows passes it to ffmpeg as a UTF-16 command line).
    folder = tmp_path / "Łukasz 田中 dir"
    folder.mkdir()
    src = folder / "-réf ü.wav"
    t = np.arange(24000) / 24000.0
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes((0.3 * np.sin(2 * np.pi * 330 * t) * 32767).astype("<i2").tobytes())
    real = _audio._ffmpeg
    calls = []

    def ffmpeg(exe, args, data):
        calls.append(data is None)
        if data is not None:  # the pipe attempt: as an M4A with its index at the end
            return subprocess.CompletedProcess(args, 1, b"", b"moov atom not found")
        return real(exe, args, data)

    monkeypatch.setattr(_audio, "_ffmpeg", ffmpeg)
    pcm = _audio.decode_reference(src)
    assert calls == [False, True] and abs(pcm.size - 24000) <= 64 and np.abs(pcm).max() > 0.2
