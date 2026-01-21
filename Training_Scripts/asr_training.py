import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

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
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperModel,
        WhisperProcessor,
    )
except ImportError:  # pragma: no cover - optional dependency
    WhisperForConditionalGeneration = None
    WhisperModel = None
    WhisperProcessor = None


@dataclass
class TrainingConfig:
    model_name: str
    language: Optional[str]
    task: Optional[str]
    data_config_path: str
    data_mode: str
    data_config_override: Optional[Dict[str, Any]]
    num_experts: int
    top_k_experts: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int
    num_workers: int
    val_split: float
    load_balance_coef: float
    gating_model_config: str
    gating_checkpoint: Optional[str]
    use_lora: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]
    experts_dir: Optional[str]
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
            "transformers is required for ASR training. "
            "Install with: pip install transformers"
        )
    if use_lora and (LoraConfig is None or get_peft_model is None):
        raise ImportError(
            "peft is required for LoRA fine-tuning. "
            "Install with: pip install peft"
        )


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
        num_experts=int(config.get("num_experts", 8)),
        top_k_experts=int(config.get("top_k_experts", 2)),
        batch_size=int(config.get("batch_size", 2)),
        epochs=int(config.get("epochs", 3)),
        learning_rate=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        seed=int(config.get("seed", 42)),
        num_workers=int(config.get("num_workers", 0)),
        val_split=float(config.get("val_split", 0.1)),
        load_balance_coef=float(config.get("load_balance_coef", 0.01)),
        gating_model_config=config.get("gating_model_config", "Config/gating_model_config.json"),
        gating_checkpoint=config.get("gating_checkpoint"),
        use_lora=bool(config.get("use_lora", True)),
        lora_r=int(lora_config.get("r", 16)),
        lora_alpha=int(lora_config.get("alpha", 32)),
        lora_dropout=float(lora_config.get("dropout", 0.05)),
        lora_target_modules=list(lora_config.get("target_modules", ["q_proj", "v_proj"])),
        experts_dir=config.get("experts_dir"),
        output_dir=config.get("output_dir", "checkpoints/asr"),
        fp16=bool(config.get("fp16", True)),
    )


def _build_embedding_model(config: TrainingConfig, device: torch.device) -> "WhisperModel":
    model = WhisperModel.from_pretrained(config.model_name).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _build_moe_model(
    config: TrainingConfig,
) -> "WhisperForConditionalGeneration":
    model = WhisperForConditionalGeneration.from_pretrained(config.model_name)

    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )
        model = get_peft_model(model, lora_config, adapter_name="expert_0")
        for expert_id in range(1, config.num_experts):
            model.add_adapter(f"expert_{expert_id}", lora_config)
        model.set_adapter("expert_0")
    return model


def _load_expert_adapters(
    model: nn.Module, experts_dir: Path, num_experts: int
) -> None:
    for expert_id in range(num_experts):
        adapter_dir = experts_dir / f"expert_{expert_id}"
        if not adapter_dir.exists():
            continue
        model.load_adapter(adapter_dir, adapter_name=f"expert_{expert_id}")


def _load_gating_model(
    config: TrainingConfig, device: torch.device
) -> GatingModel:
    model = GatingModel(config_path=config.gating_model_config).to(device)
    if config.gating_checkpoint:
        state = torch.load(config.gating_checkpoint, map_location=device)
        model.load_state_dict(state)
    return model


def _sequence_loss(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    vocab_size = logits.size(-1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
    loss = loss.view(labels.size(0), labels.size(1))
    mask = labels != -100
    loss = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return loss


def _load_balance_loss(
    probs: torch.Tensor, num_experts: int
) -> torch.Tensor:
    avg_probs = probs.mean(dim=0)
    return torch.sum(avg_probs * torch.log(avg_probs * num_experts + 1e-8))


def _train_epoch(
    model: nn.Module,
    gating_model: nn.Module,
    embedding_model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    device: torch.device,
    processor: "WhisperProcessor",
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    gating_model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        input_features = batch["input_features"]
        labels = batch["labels"]

        with torch.no_grad():
            encoder_outputs = embedding_model.encoder(input_features)
            pooled = encoder_outputs.last_hidden_state.mean(dim=1)

        gate_logits = gating_model(pooled)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        k = max(1, min(config.top_k_experts, gate_probs.size(-1)))
        topk_values, topk_indices = torch.topk(gate_probs, k=k, dim=-1)
        topk_weights = topk_values / topk_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        weights = torch.zeros_like(gate_probs)
        weights.scatter_(1, topk_indices, topk_weights)

        per_sample_loss = torch.zeros(labels.size(0), device=device)
        unique_experts = torch.unique(topk_indices).tolist()

        for expert_id in unique_experts:
            if config.use_lora:
                model.set_adapter(f"expert_{expert_id}")
            with torch.cuda.amp.autocast(enabled=config.fp16):
                outputs = model(input_features=input_features, labels=labels)
                logits = outputs.logits
                loss_per_sample = _sequence_loss(logits, labels)
            weight = weights[:, expert_id]
            per_sample_loss += weight * loss_per_sample

        main_loss = per_sample_loss.mean()
        balance_loss = _load_balance_loss(gate_probs, config.num_experts)
        loss = main_loss + config.load_balance_coef * balance_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def _evaluate(
    model: nn.Module,
    gating_model: nn.Module,
    embedding_model: nn.Module,
    data_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
) -> float:
    model.eval()
    gating_model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            input_features = batch["input_features"]
            labels = batch["labels"]

            encoder_outputs = embedding_model.encoder(input_features)
            pooled = encoder_outputs.last_hidden_state.mean(dim=1)
            gate_logits = gating_model(pooled)
            gate_probs = torch.softmax(gate_logits, dim=-1)
            k = max(1, min(config.top_k_experts, gate_probs.size(-1)))
            topk_values, topk_indices = torch.topk(gate_probs, k=k, dim=-1)
            topk_weights = topk_values / topk_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            weights = torch.zeros_like(gate_probs)
            weights.scatter_(1, topk_indices, topk_weights)

            per_sample_loss = torch.zeros(labels.size(0), device=device)
            unique_experts = torch.unique(topk_indices).tolist()
            for expert_id in unique_experts:
                if config.use_lora:
                    model.set_adapter(f"expert_{expert_id}")
                outputs = model(input_features=input_features, labels=labels)
                logits = outputs.logits
                loss_per_sample = _sequence_loss(logits, labels)
                weight = weights[:, expert_id]
                per_sample_loss += weight * loss_per_sample

            main_loss = per_sample_loss.mean()
            balance_loss = _load_balance_loss(gate_probs, config.num_experts)
            loss = main_loss + config.load_balance_coef * balance_loss

            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


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

    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(config.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    processor = WhisperProcessor.from_pretrained(config.model_name)
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

    embedding_model = _build_embedding_model(config, device)
    model = _build_moe_model(config).to(device)
    gating_model = _load_gating_model(config, device)

    if config.experts_dir and config.use_lora:
        _load_expert_adapters(model, Path(config.experts_dir), config.num_experts)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=config.language, task=config.task
    )
    model.config.forced_decoder_ids = forced_decoder_ids

    params = list(gating_model.parameters()) + [
        param for param in model.parameters() if param.requires_grad
    ]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_path = output_dir / "best.json"
    for epoch in range(1, config.epochs + 1):
        train_loss = _train_epoch(
            model=model,
            gating_model=gating_model,
            embedding_model=embedding_model,
            data_loader=train_loader,
            optimizer=optimizer,
            config=config,
            device=device,
            processor=processor,
            scaler=scaler,
        )
        val_loss = train_loss
        if val_size > 0:
            val_loss = _evaluate(
                model=model,
                gating_model=gating_model,
                embedding_model=embedding_model,
                data_loader=val_loader,
                config=config,
                device=device,
            )

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(gating_model.state_dict(), output_dir / "gating_model.pt")
            if config.use_lora:
                for expert_id in range(config.num_experts):
                    model.set_adapter(f"expert_{expert_id}")
                    adapter_dir = output_dir / f"expert_{expert_id}"
                    adapter_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(adapter_dir)
                    processor.save_pretrained(adapter_dir)
            else:
                model.save_pretrained(output_dir / "model")
                processor.save_pretrained(output_dir / "model")

            with best_path.open("w", encoding="utf-8") as best_file:
                json.dump({"loss": best_loss, "epoch": epoch}, best_file, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint MoE training (gating model + Whisper LoRA experts)."
    )
    parser.add_argument(
        "--config",
        default="Config/asr_training.json",
        help="Path to JSON config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args.config)
