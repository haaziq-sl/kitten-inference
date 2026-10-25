"""Audio helpers: reference-audio decoding with the system ``ffmpeg`` (voice cloning) and WAV writing."""
from __future__ import annotations

import os
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Optional

import numpy as np

SAMPLE_RATE = 24000
REFERENCE_LIMIT = 10 << 20  # bytes; the engine and kitten-tts-native accept at most 10 MiB of reference audio
# The reference pipeline's upload decode, which the engine's in-process FFmpeg decode mirrors and kitten-tts-native
# runs (tts-2/cli/ffmpeg_decode.h): the file's bytes on standard input, first audio stream, no metadata, mono, 24 kHz,
# at most 30.1 s, float32 little-endian. Same samples as the in-process decode (measured bit for bit on WAV, MP3, FLAC,
# Ogg Vorbis and M4A; reading the file by name gives MP3s 516 fewer samples). A file that cannot be read from a pipe
# (an MP4/M4A whose index is at the end) is then read by name, as kitten-tts-native does.
FFMPEG_ARGS = ["-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1", "-i", "pipe:0",
               "-map_metadata", "-1", "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-t", "30.100",
               "-f", "f32le", "pipe:1"]
FFMPEG_MISSING = ("ffmpeg was not found: voice cloning uses it to read the reference audio. Install it with: " +
                  ("winget install --id Gyan.FFmpeg -e (then open a new terminal)" if os.name == "nt"
                   else "sudo apt-get install -y ffmpeg"))


class AudioError(ValueError):
    """The reference recording could not be read or decoded."""


class FFmpegNotFound(AudioError, FileNotFoundError):
    """The ffmpeg executable is not on PATH."""


def ffmpeg_args(path: Optional[os.PathLike] = None) -> list:
    """The ffmpeg arguments: from standard input, or with ``path`` from that file (made absolute, so a name that starts
    with '-' is never read as an option)."""
    if path is None:
        return list(FFMPEG_ARGS)
    return [os.path.abspath(path) if a == "pipe:0" else a for a in FFMPEG_ARGS]


def find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegNotFound(FFMPEG_MISSING)
    return path


def decode_reference(path: os.PathLike, ffmpeg: Optional[str] = None) -> np.ndarray:
    """The recording at ``path`` as mono float32 PCM at 24 kHz (at most 30.1 s), decoded by ffmpeg (no shell)."""
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise AudioError(f"cannot read the reference audio {path}: {exc.strerror or exc}") from None
    if size == 0:
        raise AudioError(f"the reference audio is empty: {path}")
    if size > REFERENCE_LIMIT:
        raise AudioError(f"the reference audio must be at most 10 MiB: {path} ({size} bytes)")
    exe = ffmpeg or find_ffmpeg()
    data = path.read_bytes()
    run = _ffmpeg(exe, ffmpeg_args(), data)
    if run.returncode != 0 or len(run.stdout) < 4:
        by_name = _ffmpeg(exe, ffmpeg_args(path), None)
        if by_name.returncode == 0 and len(by_name.stdout) >= 4:
            run = by_name
    pcm = np.frombuffer(run.stdout[: len(run.stdout) // 4 * 4], dtype="<f4").astype(np.float32)
    if run.returncode != 0 or pcm.size == 0:
        detail = run.stderr.decode("utf-8", "replace").strip().splitlines()
        reason = detail[-1] if detail else f"ffmpeg exited with status {run.returncode}"
        raise AudioError(f"ffmpeg found no decodable audio in {path}: {reason}")
    return pcm


def _ffmpeg(exe: str, args: list, data: Optional[bytes]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run([exe, *args], input=data, stdin=None if data is not None else subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError as exc:
        raise FFmpegNotFound(f"{FFMPEG_MISSING} ({exc})") from None


def to_float(pcm16: bytes) -> np.ndarray:
    """Little-endian 16-bit PCM bytes as float32 in [-1, 1) (sample / 32768, exact)."""
    return np.frombuffer(pcm16, dtype="<i2").astype(np.float32) * np.float32(1.0 / 32768.0)


def to_pcm16(audio) -> bytes:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 2 and 1 in a.shape:  # (n, 1) or (1, n)
        a = a.reshape(-1)
    if a.ndim != 1:
        raise ValueError("audio must be one channel: a 1-D array of samples")
    if not np.all(np.isfinite(a)):
        raise ValueError("audio contains NaN or infinite samples")
    return np.clip(np.rint(a * 32768.0), -32768, 32767).astype("<i2").tobytes()


def save_wav(path: os.PathLike, audio, sample_rate: int = SAMPLE_RATE) -> Path:
    """Writes mono 16-bit PCM WAV. Audio from generate()/stream() round-trips exactly (it came from 16-bit PCM)."""
    path = Path(path)
    data = to_pcm16(audio)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(data)
    return path
