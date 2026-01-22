import json
from pathlib import Path
from typing import Any, Dict, Optional

from Data.datapreprocessing import WhisperDataLoader
from Models.Whisper.whisper_v2 import WhisperV2
from Models.Whisper.whisper_v3 import WhisperV3
from utils.load_config import load_config


def _select_model(name: str):
    if name.lower() == "v2":
        return WhisperV2()
    if name.lower() == "v3":
        return WhisperV3()
    raise ValueError("whisper_model must be 'v2' or 'v3'.")


def _default_output_dir(model_name: str) -> Path:
    return Path("Data/embeddings/whisper_v3_embeddings" if model_name == "v3" else "Data/embeddings/whisper_v2_embeddings")


def extract_embeddings(config_path: str = "Config/embedding_extraction.json") -> Path:
    config = load_config(config_path)

    model_name = config.get("whisper_model", "v2").lower()
    model = _select_model(model_name)

    output_dir = Path(config.get("output_dir") or _default_output_dir(model_name))
    mapping_path = Path(
        config.get("mapping_path")
        or (output_dir / "mapping.json")
    )
    overwrite = bool(config.get("overwrite", False))

    data_config_path = config.get("data_config_path", "Config/dataloader_config.json")
    data_mode = config.get("data_mode", "default")
    data_override: Optional[Dict[str, Any]] = config.get("data_config_override")

    loader = WhisperDataLoader(
        config_path=data_config_path,
        mode=data_mode,
        config=data_override,
    )

    samples = loader.sample()
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    mapping = []
    for sample in samples:
        wav_path = sample["wav_path"]
        sample_id = sample["id"]
        contributor_id = sample.get("contributor_id")
        wav_filename = Path(wav_path).name
        embedding_dir = output_dir / contributor_id if contributor_id else output_dir
        embedding_path = embedding_dir / f"{wav_filename}.npy"

        if overwrite or not embedding_path.exists():
            embedding_path = model.save_embeddings(
                wav_path,
                output_dir=embedding_dir,
                embedding_id=wav_filename,
            )

        mapping.append(
            {
                "id": sample_id,
                "contributor_id": contributor_id,
                "wav_filename": wav_filename,
                "wav_path": wav_path,
                "embedding_path": str(embedding_path),
            }
        )

    with mapping_path.open("w", encoding="utf-8") as mapping_file:
        json.dump(mapping, mapping_file, ensure_ascii=False, indent=2)

    return mapping_path


if __name__ == "__main__":
    extract_embeddings()