import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from Data.datapreprocessing import WhisperDataLoader
from Evaluation.evaluate_WER import WERScorer
from Models.Gating_Model.gating_model import GatingModel
from utils.audio import load_audio
from utils.load_config import load_config

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:  # Optional dependencies for LoRA fine-tuning
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover - optional dependency
    LoraConfig = None
    get_peft_model = None

try:  # Optional dependency for Hugging Face Whisper training
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
except ImportError:  # pragma: no cover - optional dependency
    WhisperForConditionalGeneration = None
    WhisperProcessor = None


@dataclass
class TrainingConfig:
    model_name: str
    language: Optional[str]
    task: Optional[str]
    data_config_path: str
    data_mode: str
    data_config_override: Optional[Dict[str, Any]]
    gating_model_checkpoint: Optional[str]
    gating_model_config: str
    num_experts: int
    top_k_experts: int
    expert_fraction: float
    max_samples_per_expert: Optional[int]
    val_split: float
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int
    num_workers: int
    use_lora: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]
    output_dir: str
    fp16: bool
    # New optimization parameters
    embeddings_dir: Optional[str]  # Path to cached embeddings
    gradient_accumulation_steps: int
    pin_memory: bool


class WhisperMoEDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class WhisperCollator:
    def __init__(self, processor: "WhisperProcessor") -> None:
        self.processor = processor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_list = [load_audio(item["wav_path"]) for item in batch]
        input_features = self.processor.feature_extractor(
            audio_list, sampling_rate=16000, return_tensors="pt"
        ).input_features

        texts = [item["transcript"] for item in batch]
        labels = self.processor.tokenizer(
            texts, return_tensors="pt", padding=True
        ).input_ids
        labels = labels.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"input_features": input_features, "labels": labels}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sample_key(sample: Dict[str, Any]) -> str:
    contributor_id = sample.get("contributor_id")
    sample_id = sample.get("id")
    if contributor_id:
        return f"{contributor_id}/{sample_id}"
    return str(sample_id)


def _ensure_dependencies(use_lora: bool) -> None:
    if WhisperForConditionalGeneration is None or WhisperProcessor is None:
        raise ImportError(
            "transformers is required for expert pre-training. "
            "Install with: pip install transformers"
        )
    if use_lora and (LoraConfig is None or get_peft_model is None):
        raise ImportError(
            "peft is required for LoRA fine-tuning. "
            "Install with: pip install peft"
        )


def _pool_embedding(embedding: np.ndarray) -> np.ndarray:
    pooled = embedding
    while pooled.ndim > 1:
        pooled = pooled.mean(axis=0)
    return pooled.astype(np.float32)


def _load_gating_model(
    checkpoint_path: Path, config_path: str, device: torch.device
) -> GatingModel:
    model = GatingModel(config_path=config_path).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _load_embedding_mapping(embeddings_dir: Path) -> Dict[str, Path]:
    """Load embedding mapping.json to get exact sample_id -> embedding_path mapping."""
    mapping_path = embeddings_dir / "mapping.json"
    if not mapping_path.exists():
        return {}
    try:
        with mapping_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Build lookup: sample_id -> embedding_path
        return {entry["id"]: Path(entry["embedding_path"]) for entry in data if "id" in entry and "embedding_path" in entry}
    except (json.JSONDecodeError, KeyError):
        return {}


def _find_embedding_file(embeddings_dir: Path, sample_id: str, contributor_id: Optional[str], mapping: Optional[Dict[str, Path]] = None) -> Optional[Path]:
    """Find embedding file using multiple search strategies."""
    # Strategy 0: Use mapping.json if available (most reliable)
    if mapping and sample_id in mapping:
        path = mapping[sample_id]
        if path.exists():
            return path

    # Strategy 1: Direct path with contributor subdirectory
    if contributor_id:
        path = embeddings_dir / contributor_id / f"{sample_id}.npy"
        if path.exists():
            return path
        # Try with .wav extension
        path = embeddings_dir / contributor_id / f"{sample_id}.wav.npy"
        if path.exists():
            return path

    # Strategy 2: Direct in root
    path = embeddings_dir / f"{sample_id}.npy"
    if path.exists():
        return path

    # Strategy 3: Search recursively (slower but catches edge cases)
    for candidate in embeddings_dir.rglob(f"*{sample_id}*.npy"):
        return candidate

    return None


def _assign_experts_from_cached_embeddings(
    samples: List[Dict[str, Any]],
    num_experts: int,
    gating_model: GatingModel,
    embeddings_dir: Path,
    top_k_experts: int,
    device: torch.device,
) -> Dict[str, List[int]]:
    """
    OPTIMIZED: Use cached embeddings instead of recomputing through Whisper encoder.
    This is MUCH faster for large datasets.
    """
    assignments: Dict[str, List[int]] = {}
    missing_count = 0

    # Load embedding mapping for reliable ID matching
    embedding_mapping = _load_embedding_mapping(embeddings_dir)
    if embedding_mapping:
        print(f"Loaded embedding mapping with {len(embedding_mapping)} entries")

    iterator = samples
    if tqdm is not None:
        iterator = tqdm(samples, desc="Assigning experts from cached embeddings")

    with torch.no_grad():
        for sample in iterator:
            sample_id = _sample_key(sample)
            contributor_id = sample.get("contributor_id")

            # Find cached embedding using mapping if available
            embedding_path = _find_embedding_file(
                embeddings_dir,
                sample.get("id", ""),
                contributor_id,
                mapping=embedding_mapping
            )

            if embedding_path is None or not embedding_path.exists():
                missing_count += 1
                continue

            # Load cached embedding
            embedding = np.load(embedding_path)
            embedding = _pool_embedding(embedding)

            # Get gating decision
            logits = gating_model(torch.from_numpy(embedding).unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=-1)
            k = max(1, min(int(top_k_experts), probs.shape[-1]))
            top_indices = torch.topk(probs, k=k, dim=-1).indices
            assignments[sample_id] = [int(idx) for idx in top_indices[0].cpu().tolist()]

    if missing_count > 0:
        print(f"Warning: {missing_count}/{len(samples)} samples had missing embeddings")

    return assignments


def _assign_experts_compute_on_fly(
    samples: List[Dict[str, Any]],
    num_experts: int,
    gating_model: GatingModel,
    model_name: str,
    top_k_experts: int,
    device: torch.device,
) -> Dict[str, List[int]]:
    """
    Fallback: Compute embeddings on the fly if no cached embeddings available.
    This is SLOW but works as a fallback.
    """
    from Models.Whisper.whisper_v2 import WhisperV2

    assignments: Dict[str, List[int]] = {}
    whisper_model = WhisperV2(device=str(device))

    iterator = samples
    if tqdm is not None:
        iterator = tqdm(samples, desc="Computing embeddings and assigning experts (slow)")

    with torch.no_grad():
        for sample in iterator:
            sample_id = _sample_key(sample)
            try:
                embedding_tensor = whisper_model.extract_embeddings(sample["wav_path"])
                if embedding_tensor is None:
                    continue
                embedding = _pool_embedding(embedding_tensor.detach().cpu().numpy())
                logits = gating_model(torch.from_numpy(embedding).unsqueeze(0).to(device))
                probs = torch.softmax(logits, dim=-1)
                k = max(1, min(int(top_k_experts), probs.shape[-1]))
                top_indices = torch.topk(probs, k=k, dim=-1).indices
                assignments[sample_id] = [int(idx) for idx in top_indices[0].cpu().tolist()]
            except Exception as e:
                print(f"Warning: Failed to process {sample_id}: {e}")
                continue

    return assignments


def _assign_experts(
    samples: List[Dict[str, Any]],
    num_experts: int,
    gating_model_checkpoint: Optional[Path],
    gating_model_config: str,
    top_k_experts: int,
    device: torch.device,
    embeddings_dir: Optional[Path] = None,
    model_name: str = "openai/whisper-large-v2",
) -> Dict[str, List[int]]:
    """Main dispatcher for expert assignment."""
    if gating_model_checkpoint is None:
        raise ValueError("gating_model_checkpoint is required.")

    gating_model = _load_gating_model(gating_model_checkpoint, gating_model_config, device)

    # OPTIMIZATION: Use cached embeddings if available
    if embeddings_dir is not None and embeddings_dir.exists():
        embedding_files = list(embeddings_dir.rglob("*.npy"))
        if embedding_files:
            print(f"Found {len(embedding_files)} cached embeddings in {embeddings_dir}")
            return _assign_experts_from_cached_embeddings(
                samples=samples,
                num_experts=num_experts,
                gating_model=gating_model,
                embeddings_dir=embeddings_dir,
                top_k_experts=top_k_experts,
                device=device,
            )

    # Fallback to computing on the fly
    print("Warning: No cached embeddings found. Computing on the fly (this is slow)...")
    return _assign_experts_compute_on_fly(
        samples=samples,
        num_experts=num_experts,
        gating_model=gating_model,
        model_name=model_name,
        top_k_experts=top_k_experts,
        device=device,
    )


def _prepare_expert_indices(
    samples: List[Dict[str, Any]],
    assignments: Dict[str, List[int]],
    num_experts: int,
    expert_fraction: float,
    max_samples_per_expert: Optional[int],
    seed: int,
) -> Dict[int, List[int]]:
    rng = random.Random(seed)
    indices: Dict[int, List[int]] = {expert_id: [] for expert_id in range(num_experts)}

    for idx, sample in enumerate(samples):
        labels = assignments.get(_sample_key(sample))
        if not labels:
            continue
        for label in labels:
            if label < 0 or label >= num_experts:
                continue
            indices[label].append(idx)

    for expert_id, expert_indices in indices.items():
        rng.shuffle(expert_indices)
        if expert_fraction < 1.0:
            keep = max(1, int(round(len(expert_indices) * expert_fraction)))
            expert_indices = expert_indices[:keep]
        if max_samples_per_expert is not None:
            expert_indices = expert_indices[: int(max_samples_per_expert)]
        indices[expert_id] = expert_indices
    return indices


def _build_model(
    config: TrainingConfig, processor: "WhisperProcessor"
) -> "WhisperForConditionalGeneration":
    model = WhisperForConditionalGeneration.from_pretrained(config.model_name)

    if config.language or config.task:
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=config.language, task=config.task
        )
        model.config.forced_decoder_ids = forced_decoder_ids

    for param in model.parameters():
        param.requires_grad = not config.use_lora

    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )
        model = get_peft_model(model, lora_config)
    return model


def _trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def _forward_model(model: nn.Module, **kwargs: torch.Tensor) -> Any:
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "forward"):
        return base_model(**kwargs)
    return model(**kwargs)


def _evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = _forward_model(
                    model,
                    input_features=batch["input_features"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
            batch_size = batch["input_features"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / max(total_samples, 1)


def _evaluate_wer(
    model: "WhisperForConditionalGeneration",
    data_loader: DataLoader,
    processor: "WhisperProcessor",
    device: torch.device,
) -> float:
    model.eval()
    scorer = WERScorer(normalize=True)
    references: List[str] = []
    hypotheses: List[str] = []

    with torch.no_grad():
        for batch in data_loader:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            generated_ids = model.generate(input_features)
            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)

            labels = labels.clone()
            labels[labels == -100] = processor.tokenizer.pad_token_id
            refs = processor.batch_decode(labels, skip_special_tokens=True)

            references.extend(refs)
            hypotheses.extend(preds)

    return scorer.corpus_wer(references, hypotheses)


def _train_expert(
    expert_id: int,
    dataset: Dataset,
    indices: List[int],
    config: TrainingConfig,
    processor: "WhisperProcessor",
    device: torch.device,
    early_stopping_patience: int = 5,
) -> None:
    if not indices:
        return

    subset = Subset(dataset, indices)
    val_size = int(len(subset) * config.val_split)
    train_size = len(subset) - val_size
    generator = torch.Generator().manual_seed(config.seed)
    train_set, val_set = random_split(subset, [train_size, val_size], generator=generator)

    collator = WhisperCollator(processor)
    use_pin_memory = config.pin_memory and torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=use_pin_memory,
    )

    model = _build_model(config, processor).to(device)
    optimizer = torch.optim.AdamW(
        _trainable_parameters(model),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)
    accumulation_steps = config.gradient_accumulation_steps

    metrics_path = Path("Evaluation/expert_training_results") / f"expert_{expert_id}" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_history: List[Dict[str, float]] = []
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as metrics_file:
            try:
                metrics_history = json.load(metrics_file)
            except json.JSONDecodeError:
                metrics_history = []

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        iterator = train_loader
        if tqdm is not None:
            iterator = tqdm(train_loader, desc=f"Expert {expert_id} Epoch {epoch}", leave=False)

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(iterator):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=config.fp16):
                outputs = _forward_model(
                    model,
                    input_features=batch["input_features"],
                    labels=batch["labels"],
                )
                loss = outputs.loss

            batch_size = batch["input_features"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Gradient accumulation
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # Handle remaining gradients
        if (batch_idx + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss = total_loss / max(total_samples, 1)
        current_loss = train_loss  # Default to train_loss for early stopping if no val
        if val_size > 0:
            val_loss = _evaluate(model, val_loader, device, config.fp16)
            val_wer = _evaluate_wer(model, val_loader, processor, device)
            current_loss = val_loss
            metrics_history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_wer": float(val_wer),
                }
            )
            print(
                f"Expert {expert_id} epoch {epoch}: "
                f"val_loss={val_loss:.4f} val_wer={val_wer:.4f}"
            )
        else:
            metrics_history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_loss": None,
                    "val_wer": None,
                }
            )
            print(
                f"Expert {expert_id} epoch {epoch}: "
                f"train_loss={train_loss:.4f} (no val split)"
            )

        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            json.dump(metrics_history, metrics_file, indent=2)

        # Early stopping check
        if current_loss < best_val_loss:
            best_val_loss = current_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Expert {expert_id}: Early stopping after {epoch} epochs (no improvement for {early_stopping_patience} epochs)")
                break

    output_dir = Path(config.output_dir) / f"expert_{expert_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    del model
    torch.cuda.empty_cache()


def _load_training_config(config_path: str) -> TrainingConfig:
    config = load_config(config_path)
    lora_config = config.get("lora", {})

    return TrainingConfig(
        model_name=config.get("model_name", "openai/whisper-large-v2"),
        language=config.get("language"),
        task=config.get("task", "transcribe"),
        data_config_path=config.get("data_config_path", "Config/dataloader_config.json"),
        data_mode=config.get("data_mode", "default"),
        data_config_override=config.get("data_config_override"),
        gating_model_checkpoint=config.get("gating_model_checkpoint"),
        gating_model_config=config.get("gating_model_config", "Config/gating_model_config.json"),
        num_experts=int(config.get("num_experts", 8)),
        top_k_experts=int(config.get("top_k_experts", 2)),
        expert_fraction=float(config.get("expert_fraction", 0.5)),
        max_samples_per_expert=config.get("max_samples_per_expert"),
        val_split=float(config.get("val_split", 0.1)),
        batch_size=int(config.get("batch_size", 2)),
        epochs=int(config.get("epochs", 3)),
        learning_rate=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        seed=int(config.get("seed", 42)),
        num_workers=int(config.get("num_workers", 0)),
        use_lora=bool(config.get("use_lora", True)),
        lora_r=int(lora_config.get("r", 16)),
        lora_alpha=int(lora_config.get("alpha", 32)),
        lora_dropout=float(lora_config.get("dropout", 0.05)),
        lora_target_modules=list(lora_config.get("target_modules", ["q_proj", "v_proj"])),
        output_dir=config.get("output_dir", "checkpoints/experts"),
        fp16=bool(config.get("fp16", True)),
        # New optimization parameters
        embeddings_dir=config.get("embeddings_dir"),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        pin_memory=bool(config.get("pin_memory", True)),
    )


def train(config_path: str, max_samples: Optional[int] = None) -> None:
    config = _load_training_config(config_path)
    if max_samples is not None:
        if config.data_config_override is None:
            config.data_config_override = {}
        config.data_config_override["max_samples"] = int(max_samples)
    _ensure_dependencies(config.use_lora)
    _set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not config.gating_model_checkpoint:
        raise ValueError("gating_model_checkpoint is required for expert pre-training.")

    loader = WhisperDataLoader(
        config_path=config.data_config_path,
        mode=config.data_mode,
        config=config.data_config_override,
    )
    samples = [
        sample
        for sample in loader.sample()
        if sample.get("prompt")
    ]

    if not samples:
        raise ValueError("No samples with transcripts available for training.")

    print(f"Loaded {len(samples)} samples with transcripts")

    # OPTIMIZATION: Use embeddings_dir for cached embeddings
    embeddings_dir = Path(config.embeddings_dir) if config.embeddings_dir else None

    assignments = _assign_experts(
        samples=samples,
        num_experts=config.num_experts,
        gating_model_checkpoint=Path(config.gating_model_checkpoint)
        if config.gating_model_checkpoint
        else None,
        gating_model_config=config.gating_model_config,
        top_k_experts=config.top_k_experts,
        device=device,
        embeddings_dir=embeddings_dir,
        model_name=config.model_name,
    )

    indices_by_expert = _prepare_expert_indices(
        samples,
        assignments,
        config.num_experts,
        config.expert_fraction,
        config.max_samples_per_expert,
        config.seed,
    )

    # Print expert distribution
    print("\nExpert sample distribution:")
    for expert_id, indices in indices_by_expert.items():
        print(f"  Expert {expert_id}: {len(indices)} samples")

    dataset = WhisperMoEDataset(
        [
            {
                "id": _sample_key(sample),
                "wav_path": sample["wav_path"],
                "transcript": sample.get("prompt", ""),
            }
            for sample in samples
        ]
    )

    processor = WhisperProcessor.from_pretrained(config.model_name)

    for expert_id, indices in indices_by_expert.items():
        if not indices:
            print(f"\nSkipping expert {expert_id}: no assigned samples.")
            continue
        print(f"\n{'='*50}")
        print(f"Training expert {expert_id} with {len(indices)} samples")
        print(f"{'='*50}")
        _train_expert(
            expert_id=expert_id,
            dataset=dataset,
            indices=indices,
            config=config,
            processor=processor,
            device=device,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-train MoE experts (Whisper + LoRA) with a frozen gate."
    )
    parser.add_argument(
        "--config",
        default="Config/expert_pre_training.json",
        help="Path to JSON config.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override max_samples for data selection (e.g., 50).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args.config, args.max_samples)
