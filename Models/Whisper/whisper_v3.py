from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import whisper

from utils.load_config import load_config


class WhisperV3:
    def __init__(self, config_path: str = "Config/whisper_config.json", device: Optional[str] = None):
        config = load_config(config_path)
        v3_config = config.get("v3", {})

        model_name = v3_config.get("model_name", "large-v3")
        download_root = v3_config.get("download_root")
        configured_device = device or v3_config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        if configured_device == "cuda" and not torch.cuda.is_available():
            configured_device = "cpu"

        self.config = v3_config
        self.device = configured_device
        self.model = whisper.load_model(
            model_name, device=configured_device, download_root=download_root
        )
        self.model.eval()

    def predict(self, wav_path: str, **decode_options: Any) -> Dict[str, Any]:
        options = dict(self.config.get("transcribe_options", {}))
        options.update(decode_options)
        return self.model.transcribe(wav_path, **options)

    def extract_embeddings(self, wav_path: str) -> torch.Tensor:
        audio = whisper.load_audio(wav_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)

        with torch.no_grad():
            embeddings = self.model.encoder(mel)

        return embeddings

    def save_embeddings(
        self,
        wav_path: str,
        output_dir: Path = Path("Data/embeddings/whisper_v3_embeddings"),
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        embeddings = self.extract_embeddings(wav_path)
        embedding_id = Path(wav_path).name
        output_path = output_dir / f"{embedding_id}.npy"
        np.save(output_path, embeddings.detach().cpu().numpy())
        return output_path