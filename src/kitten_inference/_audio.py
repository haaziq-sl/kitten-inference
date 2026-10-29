"""Audio writing shared by both model families."""
from pathlib import Path


def save_wav(path, audio, sample_rate):
    import numpy as np
    import soundfile as sf
    if isinstance(audio, (list, tuple)):
        audio = np.concatenate([np.asarray(a, dtype=np.float32).reshape(-1) for a in audio]) if audio else np.zeros(0, np.float32)
    sf.write(path, audio, sample_rate, format='WAV', subtype='PCM_16')
    return Path(path)
