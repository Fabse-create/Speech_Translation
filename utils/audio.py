from typing import Optional

import numpy as np


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    try:
        import torchaudio
    except ImportError:
        torchaudio = None

    if torchaudio is not None:
        waveform, sample_rate = torchaudio.load(path)
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)
        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sr
            )
        return waveform.numpy()

    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "Audio loading requires torchaudio or librosa. "
            "Install with: pip install torchaudio or pip install librosa"
        ) from exc

    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    if not isinstance(audio, np.ndarray):
        audio = np.asarray(audio)
    return audio.astype(np.float32)
