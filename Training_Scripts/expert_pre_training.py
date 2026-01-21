import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from Data.datapreprocessing import WhisperDataLoader
from Models.Gating_Model.gating_model import GatingModel
from utils.audio import load_audio
from utils.load_config import load_config

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
    cluster_labels_path: Optional[str]
    embeddings_dir: Optional[str]
    gating_model_checkpoint: Optional[str]
    gating_model_config: str
    use_gating_model: bool
    drop_noise: bool
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


def _load_cluster_labels(
    labels_path: Path, num_experts: int, drop_noise: bool, top_k_experts: int
) -> Dict[str, List[int]]:
    with labels_path.open("r", encoding="utf-8") as labels_file:
        data = json.load(labels_file)

    if not isinstance(data, list):
        raise ValueError("Cluster labels file must contain a list of entries.")
    if not data:
        raise ValueError("Cluster labels file is empty.")
    if not isinstance(data[0], dict) or "id" not in data[0]:
        raise ValueError("Cluster labels must include explicit 'id' entries.")

    assignments: Dict[str, List[int]] = {}
    for entry in data:
        entry_id = entry.get("id")
        if entry_id is None:
            continue
        if "label" in entry:
            label = int(entry["label"])
            if label < 0 and drop_noise:
                continue
            assignments[entry_id] = [label]
            continue
        if "probs" in entry:
            probs = np.asarray(entry["probs"], dtype=np.float32)
            if drop_noise and probs.shape[0] == num_experts + 1:
                probs = probs[:num_experts]
            if probs.size == 0:
                continue
            k = max(1, min(int(top_k_experts), probs.shape[0]))
            top_indices = np.argsort(probs)[-k:][::-1]
            assignments[entry_id] = [int(idx) for idx in top_indices]
            continue
        raise ValueError("Cluster entry must include 'label' or 'probs'.")
    return assignments


def _load_embeddings(
    embeddings_dir: Path,
) -> Dict[str, np.ndarray]:
    embedding_files = list(embeddings_dir.glob("*.npy"))
    if not embedding_files:
        raise FileNotFoundError(f"No embeddings found in {embeddings_dir}")

    embeddings: Dict[str, np.ndarray] = {}
    for embedding_path in embedding_files:
        sample_id = embedding_path.name
        if sample_id.endswith(".npy"):
            sample_id = sample_id[: -len(".npy")]
        embeddings[sample_id] = _pool_embedding(np.load(embedding_path))
    return embeddings


def _load_gating_model(
    checkpoint_path: Path, config_path: str, device: torch.device
) -> GatingModel:
    model = GatingModel(config_path=config_path).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _assign_experts(
    samples: List[Dict[str, Any]],
    num_experts: int,
    labels_path: Optional[Path],
    embeddings_dir: Optional[Path],
    gating_model_checkpoint: Optional[Path],
    gating_model_config: str,
    use_gating_model: bool,
    drop_noise: bool,
    top_k_experts: int,
    device: torch.device,
) -> Dict[str, List[int]]:
    assignments: Dict[str, List[int]] = {}
    if labels_path is not None:
        assignments.update(
            _load_cluster_labels(labels_path, num_experts, drop_noise, top_k_experts)
        )

    if use_gating_model:
        if embeddings_dir is None or gating_model_checkpoint is None:
            raise ValueError("embeddings_dir and gating_model_checkpoint are required.")
        embeddings = _load_embeddings(embeddings_dir)
        gating_model = _load_gating_model(
            gating_model_checkpoint, gating_model_config, device
        )
        with torch.no_grad():
            for sample in samples:
                sample_id = sample["id"]
                embedding = embeddings.get(sample_id)
                if embedding is None:
                    continue
                logits = gating_model(torch.from_numpy(embedding).to(device))
                probs = torch.softmax(logits, dim=-1)
                k = max(1, min(int(top_k_experts), probs.shape[-1]))
                top_indices = torch.topk(probs, k=k, dim=-1).indices
                assignments[sample_id] = [int(idx) for idx in top_indices.cpu().tolist()]
    return assignments


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
        labels = assignments.get(sample["id"])
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
                outputs = model(**batch)
                loss = outputs.loss
            batch_size = batch["input_features"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / max(total_samples, 1)


def _train_expert(
    expert_id: int,
    dataset: Dataset,
    indices: List[int],
    config: TrainingConfig,
    processor: "WhisperProcessor",
    device: torch.device,
) -> None:
    if not indices:
        return

    subset = Subset(dataset, indices)
    val_size = int(len(subset) * config.val_split)
    train_size = len(subset) - val_size
    generator = torch.Generator().manual_seed(config.seed)
    train_set, val_set = random_split(subset, [train_size, val_size], generator=generator)

    collator = WhisperCollator(processor)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collator,
    )

    model = _build_model(config, processor).to(device)
    optimizer = torch.optim.AdamW(
        _trainable_parameters(model),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=config.fp16):
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if val_size > 0:
            val_loss = _evaluate(model, val_loader, device, config.fp16)
            print(f"Expert {expert_id} epoch {epoch}: val_loss={val_loss:.4f}")

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
        cluster_labels_path=config.get("cluster_labels_path"),
        embeddings_dir=config.get("embeddings_dir"),
        gating_model_checkpoint=config.get("gating_model_checkpoint"),
        gating_model_config=config.get("gating_model_config", "Config/gating_model_config.json"),
        use_gating_model=bool(config.get("use_gating_model", False)),
        drop_noise=bool(config.get("drop_noise", True)),
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
    )


def train(config_path: str) -> None:
    config = _load_training_config(config_path)
    _ensure_dependencies(config.use_lora)
    _set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    assignments = _assign_experts(
        samples=samples,
        num_experts=config.num_experts,
        labels_path=Path(config.cluster_labels_path) if config.cluster_labels_path else None,
        embeddings_dir=Path(config.embeddings_dir) if config.embeddings_dir else None,
        gating_model_checkpoint=Path(config.gating_model_checkpoint)
        if config.gating_model_checkpoint
        else None,
        gating_model_config=config.gating_model_config,
        use_gating_model=config.use_gating_model,
        drop_noise=config.drop_noise,
        top_k_experts=config.top_k_experts,
        device=device,
    )

    indices_by_expert = _prepare_expert_indices(
        samples,
        assignments,
        config.num_experts,
        config.expert_fraction,
        config.max_samples_per_expert,
        config.seed,
    )

    dataset = WhisperMoEDataset(
        [
            {
                "id": sample["id"],
                "wav_path": sample["wav_path"],
                "transcript": sample.get("prompt", ""),
            }
            for sample in samples
        ]
    )

    processor = WhisperProcessor.from_pretrained(config.model_name)

    for expert_id, indices in indices_by_expert.items():
        if not indices:
            print(f"Skipping expert {expert_id}: no assigned samples.")
            continue
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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args.config)
