"""Audio writing shared by both model families."""
def save_wav(path, audio, sample_rate):
    import numpy as np
    from .tts2._audio import save_wav as write_wav
    if isinstance(audio, (list, tuple)):
        audio = np.concatenate([np.asarray(a, dtype=np.float32).reshape(-1) for a in audio]) if audio else np.zeros(0, np.float32)
    return write_wav(path, audio, sample_rate)
