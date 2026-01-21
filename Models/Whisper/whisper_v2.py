from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from transformers import WhisperForConditionalGeneration, WhisperProcessor

from utils.audio import load_audio
from utils.load_config import load_config


class WhisperV2:
    def __init__(self, config_path: str = "Config/whisper_config.json", device: Optional[str] = None):
        config = load_config(config_path)
        v2_config = config.get("v2", {})

        model_name = v2_config.get("model_name", "openai/whisper-large-v2")
        download_root = v2_config.get("download_root")
        configured_device = device or v2_config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        if configured_device == "cuda" and not torch.cuda.is_available():
            configured_device = "cpu"

        self.config = v2_config
        self.device = configured_device
        self.model_name = self._normalize_model_name(model_name)
        self.processor = WhisperProcessor.from_pretrained(
            self.model_name, cache_dir=download_root
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name, cache_dir=download_root
        ).to(self.device)
        self.model.eval()

    def predict(self, wav_path: str, **decode_options: Any) -> Dict[str, Any]:
        options = dict(self.config.get("transcribe_options", {}))
        options.update(decode_options)

        audio = load_audio(wav_path)
        input_features = self.processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device)

        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=options.get("language"),
            task=options.get("task", "transcribe"),
        )
        generated_ids = self.model.generate(
            input_features, forced_decoder_ids=forced_decoder_ids
        )
        text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return {"text": text}

    def extract_embeddings(self, wav_path: str) -> torch.Tensor:
        audio = load_audio(wav_path)
        input_features = self.processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device)

        with torch.no_grad():
            encoder_outputs = self.model.model.encoder(input_features)

        return encoder_outputs.last_hidden_state

    def save_embeddings(
        self,
        wav_path: str,
        output_dir: Path = Path("Data/embeddings/whisper_v2_embeddings"),
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        embeddings = self.extract_embeddings(wav_path)
        embedding_id = Path(wav_path).name
        output_path = output_dir / f"{embedding_id}.npy"
        np.save(output_path, embeddings.detach().cpu().numpy())
        return output_path

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        if "/" in model_name:
            return model_name
        if model_name.startswith("whisper-"):
            return f"openai/{model_name}"
        return f"openai/whisper-{model_name}"