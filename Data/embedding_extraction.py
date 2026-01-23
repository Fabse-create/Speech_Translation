import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from Data.datapreprocessing import WhisperDataLoader
from Models.Whisper.whisper_v2 import WhisperV2
from Models.Whisper.whisper_v3 import WhisperV3
from utils.load_config import load_config

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _select_model(name: str, device: Optional[str] = None):
    if name.lower() == "v2":
        return WhisperV2(device=device)
    if name.lower() == "v3":
        return WhisperV3(device=device)
    raise ValueError("whisper_model must be 'v2' or 'v3'.")


def _default_output_dir(model_name: str) -> Path:
    return Path("Data/embeddings/whisper_v3_embeddings" if model_name == "v3" else "Data/embeddings/whisper_v2_embeddings")


def _load_existing_mapping(mapping_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load existing mapping for resume capability."""
    if not mapping_path.exists():
        return {}
    try:
        with mapping_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Index by sample_id for fast lookup
        return {entry["id"]: entry for entry in data}
    except (json.JSONDecodeError, KeyError):
        return {}


def _save_mapping(mapping: List[Dict[str, Any]], mapping_path: Path) -> None:
    """Save mapping to disk."""
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def extract_embeddings(
    config_path: str = "Config/embedding_extraction.json",
    checkpoint_interval: int = 100,
) -> Path:
    """
    Extract embeddings from audio files using Whisper encoder.
    
    OPTIMIZATIONS:
    - Incremental mapping saves every checkpoint_interval samples
    - Resume capability: skips already processed samples
    - Progress tracking with tqdm
    - Memory management with periodic cache clearing
    """
    config = load_config(config_path)

    model_name = config.get("whisper_model", "v2").lower()
    device = config.get("device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # Load existing mapping for resume capability
    existing_mapping = _load_existing_mapping(mapping_path)
    print(f"Found {len(existing_mapping)} existing embeddings")

    # Initialize model (lazy loading to save memory if all samples are cached)
    model = None
    mapping: List[Dict[str, Any]] = []

    # Add existing entries first (maintain order)
    processed_ids = set()
    for entry in existing_mapping.values():
        if Path(entry["embedding_path"]).exists():
            mapping.append(entry)
            processed_ids.add(entry["id"])

    # Filter samples that need processing
    samples_to_process = []
    for sample in samples:
        sample_id = sample["id"]
        if sample_id in processed_ids and not overwrite:
            continue
        samples_to_process.append(sample)

    print(f"Total samples: {len(samples)}")
    print(f"Already processed: {len(processed_ids)}")
    print(f"Samples to process: {len(samples_to_process)}")

    if not samples_to_process:
        print("All samples already processed!")
        _save_mapping(mapping, mapping_path)
        return mapping_path

    # Only load model if we have samples to process
    print(f"Loading Whisper {model_name} model on {device}...")
    model = _select_model(model_name, device)

    # Process samples with progress tracking
    iterator = samples_to_process
    if tqdm is not None:
        iterator = tqdm(samples_to_process, desc="Extracting embeddings")

    newly_processed = 0
    for idx, sample in enumerate(iterator):
        wav_path = sample["wav_path"]
        sample_id = sample["id"]
        contributor_id = sample.get("contributor_id")
        wav_filename = Path(wav_path).name
        embedding_dir = output_dir / contributor_id if contributor_id else output_dir
        embedding_path = embedding_dir / f"{wav_filename}.npy"

        try:
            if overwrite or not embedding_path.exists():
                embedding_path = model.save_embeddings(
                    wav_path,
                    output_dir=embedding_dir,
                    embedding_id=wav_filename,
                )
                newly_processed += 1

            # Update mapping
            entry = {
                "id": sample_id,
                "contributor_id": contributor_id,
                "wav_filename": wav_filename,
                "wav_path": wav_path,
                "embedding_path": str(embedding_path),
            }

            # Update or append
            existing_idx = next(
                (i for i, e in enumerate(mapping) if e["id"] == sample_id),
                None
            )
            if existing_idx is not None:
                mapping[existing_idx] = entry
            else:
                mapping.append(entry)

        except Exception as e:
            print(f"\nWarning: Failed to process {sample_id}: {e}")
            continue

        # OPTIMIZATION: Incremental checkpoint saves
        if (idx + 1) % checkpoint_interval == 0:
            _save_mapping(mapping, mapping_path)
            if tqdm is not None and isinstance(iterator, tqdm):
                iterator.set_postfix(saved=len(mapping), new=newly_processed)

            # Clear CUDA cache periodically to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Final save
    _save_mapping(mapping, mapping_path)
    print(f"\nExtraction complete!")
    print(f"Total embeddings: {len(mapping)}")
    print(f"Newly processed: {newly_processed}")

    return mapping_path


def extract_embeddings_batched(
    config_path: str = "Config/embedding_extraction.json",
    batch_size: int = 8,
    checkpoint_interval: int = 100,
) -> Path:
    """
    EXPERIMENTAL: Batched embedding extraction for faster processing.
    
    Note: This requires the Whisper model to support batched inference,
    which may require modifications to the model wrapper.
    """
    # For now, fall back to single-sample extraction
    # Batched extraction would require modifying WhisperV2/V3 to accept
    # multiple audio files and return batched embeddings
    print("Note: Batched extraction not yet fully implemented. Using sequential extraction.")
    return extract_embeddings(config_path, checkpoint_interval)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract Whisper embeddings")
    parser.add_argument(
        "--config",
        default="Config/embedding_extraction.json",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save mapping every N samples"
    )
    args = parser.parse_args()
    extract_embeddings(args.config, args.checkpoint_interval)
