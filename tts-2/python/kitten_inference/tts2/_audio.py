"""Audio helpers: reference-audio decoding with the system ``ffmpeg`` (voice cloning) and WAV writing."""
from __future__ import annotations

import os
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
# Demuxers the by-name fallback may use: ordinary recording containers only. It leaves out every demuxer that opens
# other files or URLs or runs a script (hls, dash, concat, imf, sdp, rtsp, rtp, image2, avisynth, vapoursynth,
# dvdvideo), the same ones the in-process decode excludes, so a reference file named like a *.avs AviSynth script or
# an ID3-hidden ffconcat list cannot pick one of them. ffmpeg has no demuxer deny-list, so this allow-list names the
# formats a recording can be instead. The read-from-pipe run above is unrestricted (guarded by is_playlist).
BY_NAME_FORMATS = ("wav,w64,aiff,au,caf,mp3,mp2,flac,ogg,oga,opus,aac,amr,mov,mp4,m4a,3gp,3g2,mj2,"
                   "matroska,webm,asf")


class AudioError(ValueError):
    """The reference recording could not be read or decoded."""


class FFmpegNotFound(AudioError, FileNotFoundError):
    """The ffmpeg executable is not on PATH."""


def _id3v2_len(data: bytes) -> int:
    """Bytes of a leading ID3v2 tag, or 0 when there is none. FFmpeg skips such a tag before it probes the format, so
    a playlist can hide behind one; is_playlist skips it the same way (10-byte header plus a syncsafe size, plus 10
    for a footer). A size that runs past the data leaves nothing to match, so the whole buffer is reported."""
    if len(data) < 10 or data[:3] != b"ID3" or data[3] == 0xFF or data[4] == 0xFF or any(b & 0x80 for b in data[6:10]):
        return 0
    size = (data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9]
    total = 10 + size + (10 if data[5] & 0x10 else 0)
    return total if total < len(data) else len(data)


def is_playlist(data: bytes) -> bool:
    """Whether ``data`` is a playlist, manifest or session description (HLS/M3U, DASH, IMF, ffconcat, SDP) rather than
    a recording (kitten-tts-native's check in cli/ffmpeg_decode.cc, except that its DASH/IMF/SDP search also reads
    past the first NUL byte, see below). Such a file makes ffmpeg open the files or
    URLs it names: the DASH demuxer of FFmpeg 6.1 (Ubuntu 24.04's ffmpeg) fetches them even from standard input, and
    the by-name fallback would allow local files. Found as FFmpeg's probes find them, after a leading ID3v2 tag (which
    FFmpeg skips before probing): HLS and ffconcat at the start, DASH and IMF anywhere in the text, SDP by a "c=IN IP"
    line. DASH is matched on "<mpd" alone: FFmpeg's dash_probe needs only "<MPD" plus any of several profile strings,
    one of which ("3GPP:PSS:profile:DASH1") does not contain "dash:profile".

    The DASH, IMF and SDP probes read the probe buffer as a C string (av_stristr, strstr, a loop up to '\\0'), so
    they never look past the first NUL byte. A real recording has one within its first bytes (the RIFF size, the
    MP4 box size, the FLAC and Ogg headers), so only the text before it is searched: the compressed audio after it is
    close to random and contains "<mpd" in some case about once in 500 million bytes, which refused about 0.4% of
    2 MB recordings and 2% of 10 MiB ones although ffmpeg reads them as audio."""
    head = data[_id3v2_len(data):]
    text = head.split(b"\0", 1)[0]  # what FFmpeg's C-string probes see
    head = head[3:] if head.startswith(b"\xef\xbb\xbf") else head
    if head[:4096].lstrip(b" \t\r\n").startswith((b"#EXTM3U", b"ffconcat")):
        return True
    low = text.lower()
    if b"<mpd" in low or b"<compositionplaylist" in low:
        return True
    return text.startswith(b"c=IN IP") or b"\nc=IN IP" in text or b"\n\rc=IN IP" in text


def ffmpeg_args(path: Optional[os.PathLike] = None) -> list:
    """The ffmpeg arguments: from standard input, or with ``path`` from that file (made absolute, so a name that starts
    with '-' is never read as an option). The by-name form (``path`` given) is the fallback for files that cannot be
    read from a pipe (an MP4/M4A whose index is at the end); it holds ffmpeg to ordinary recording demuxers and to
    local files, so a reference file with another extension (for example a .avs read as an AviSynth script, which can
    load and run any DLL) cannot select a demuxer that runs code or opens other inputs."""
    if path is None:
        return list(FFMPEG_ARGS)
    args = list(FFMPEG_ARGS)
    i = args.index("-i")
    args[i + 1] = os.path.abspath(path)
    # Hold ffmpeg to ordinary recording demuxers and to local files, so a reference file with another extension
    # (a .avs read as an AviSynth script, which can load and run any DLL) or an ID3-hidden ffconcat list cannot
    # select a demuxer that runs code or opens other inputs.
    args[i:i] = ["-format_whitelist", BY_NAME_FORMATS, "-protocol_whitelist", "file"]
    return args


def find_ffmpeg() -> str:
    """The ffmpeg program on PATH. Only absolute PATH folders count: shutil.which would also take one from the current
    folder (on Windows before every PATH entry, and on any system for an empty PATH entry), where a downloaded or
    planted ffmpeg would then get the recording. On Windows only ffmpeg.exe is taken: a .bat or .cmd runs through
    cmd.exe, which would read characters such as & in the file name as commands."""
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    for folder in os.environ.get("PATH", "").split(os.pathsep):
        folder = folder.strip().strip('"') if os.name == "nt" else folder
        if not folder or not os.path.isabs(folder) or (os.name == "nt" and not os.path.splitdrive(folder)[0]):
            continue
        path = os.path.join(folder, name)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise FFmpegNotFound(FFMPEG_MISSING)


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
    if is_playlist(data):
        raise AudioError(f"the reference audio is a playlist or stream manifest (HLS, DASH or similar), not a "
                         f"recording: {path}")
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
