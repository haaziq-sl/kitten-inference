"""Audio writing shared by both model families."""
from pathlib import Path
import wave


def save_wav(path, audio, sample_rate):
    import numpy as np
    if isinstance(audio, (list, tuple)):
        audio = np.concatenate([np.asarray(a, dtype=np.float32).reshape(-1) for a in audio]) if audio else np.zeros(0, np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2 and 1 in audio.shape:
        audio = audio.reshape(-1)
    if audio.ndim != 1:
        raise ValueError('audio must be one channel: a 1-D array of samples')
    if not np.all(np.isfinite(audio)):
        raise ValueError('audio contains NaN or infinite samples')
    pcm = np.clip(np.rint(audio * 32768.0), -32768, 32767).astype('<i2').tobytes()
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(int(sample_rate))
        out.writeframes(pcm)
    return Path(path)
