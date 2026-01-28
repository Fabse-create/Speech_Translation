import argparse
import json
import random
import shutil
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

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
    metrics_dir: str
    # New optimization parameters
    gradient_accumulation_steps: int
    eval_every_n_epochs: int
    save_every_n_epochs: int
    pin_memory: bool
    # Routing curriculum parameters
    soft_routing_epochs: int
    topk_routing_epochs: int
    routing_temperature_start: float
    routing_temperature_end: float
    min_expert_usage_fraction: float
    routing_temperature_min: float
    router_noise_std: float
    routing_alignment_coef: float
    routing_alignment_temperature: float
    routing_entropy_coef: float
    save_every_epoch: bool
    rebase_output_dir: bool


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
        features = self.processor.feature_extractor(
            audio_list,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = features.input_features
        attention_mask = features.get("attention_mask")

        texts = [item["transcript"] for item in batch]
        max_label_length = getattr(self.processor.tokenizer, "model_max_length", 448)
        if max_label_length is None or max_label_length > 1000:
            max_label_length = 448
        label_batch = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_label_length,
            return_attention_mask=True,
        )
        labels = label_batch.input_ids.clone()
        decoder_attention_mask = label_batch.attention_mask
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        batch_out: Dict[str, torch.Tensor] = {
            "input_features": input_features,
            "labels": labels,
        }
        if attention_mask is not None:
            batch_out["attention_mask"] = attention_mask
        if decoder_attention_mask is not None:
            batch_out["decoder_attention_mask"] = decoder_attention_mask
        return batch_out


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
        metrics_dir=config.get("metrics_dir", "Evaluation/asr_training_results"),
        # New optimization parameters with sensible defaults
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        eval_every_n_epochs=int(config.get("eval_every_n_epochs", 1)),
        save_every_n_epochs=int(config.get("save_every_n_epochs", 5)),
        pin_memory=bool(config.get("pin_memory", True)),
        # Routing curriculum parameters
        soft_routing_epochs=int(config.get("soft_routing_epochs", 3)),
        topk_routing_epochs=int(config.get("topk_routing_epochs", 6)),
        routing_temperature_start=float(config.get("routing_temperature_start", 2.0)),
        routing_temperature_end=float(config.get("routing_temperature_end", 0.5)),
        min_expert_usage_fraction=float(config.get("min_expert_usage_fraction", 0.05)),
        routing_temperature_min=float(
            config.get(
                "routing_temperature_min",
                config.get("routing_temperature_end", 0.5),
            )
        ),
        router_noise_std=float(config.get("router_noise_std", 0.0)),
        routing_alignment_coef=float(config.get("routing_alignment_coef", 0.0)),
        routing_alignment_temperature=float(config.get("routing_alignment_temperature", 1.0)),
        routing_entropy_coef=float(config.get("routing_entropy_coef", 0.0)),
        save_every_epoch=bool(config.get("save_every_epoch", True)),
        rebase_output_dir=bool(config.get("rebase_output_dir", True)),
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


def _forward_model(model: nn.Module, **kwargs: torch.Tensor) -> Any:
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "forward"):
        return base_model(**kwargs)
    return model(**kwargs)


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
        state = torch.load(config.gating_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
    return model


def _configure_generation(
    model: "WhisperForConditionalGeneration",
    language: Optional[str],
    task: Optional[str],
) -> None:
    generation_config = getattr(model, "generation_config", model.config)
    if language:
        generation_config.language = language
    if task:
        generation_config.task = task
    if hasattr(generation_config, "forced_decoder_ids"):
        generation_config.forced_decoder_ids = None


def _resolve_max_length(
    model: "WhisperForConditionalGeneration",
    processor: "WhisperProcessor",
) -> int:
    """Resolve max generation length, same logic as WER benchmark."""
    generation_config = getattr(model, "generation_config", model.config)
    max_length = getattr(generation_config, "max_length", None)
    if max_length is None:
        max_length = getattr(model.config, "max_length", None)
    if max_length is None and processor is not None:
        max_length = processor.tokenizer.model_max_length
    if max_length is None or max_length <= 0 or max_length > 2048:
        max_length = 448
    return int(max_length)


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
    """
    KL divergence between a uniform target and the empirical expert usage.
    """
    importance = probs.mean(dim=0)
    uniform = torch.full_like(importance, 1.0 / num_experts)
    eps = 1e-8
    return torch.sum(uniform * torch.log((uniform + eps) / (importance + eps)))


def _gating_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()


def _routing_alignment_loss(
    gate_probs: torch.Tensor,
    expert_losses: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    safe_temp = max(1e-4, float(temperature))
    with torch.no_grad():
        target = torch.softmax(-expert_losses / safe_temp, dim=-1)
    log_probs = torch.log(gate_probs + 1e-8)
    return F.kl_div(log_probs, target, reduction="batchmean")


def _compute_routing_mode(epoch_idx: int, config: TrainingConfig) -> str:
    if epoch_idx < config.soft_routing_epochs:
        return "soft"
    if epoch_idx < config.topk_routing_epochs:
        return "topk_soft"
    return "topk_hard"


def _compute_routing_temperature(epoch_idx: int, config: TrainingConfig) -> float:
    max_epoch = max(1, config.topk_routing_epochs)
    progress = min(1.0, epoch_idx / max_epoch)
    temperature = (
        config.routing_temperature_start * (1 - progress)
        + config.routing_temperature_end * progress
    )
    return max(temperature, config.routing_temperature_min)


def _compute_gating_probabilities(
    gate_logits: torch.Tensor,
    temperature: float,
    min_prob: float = 0.0,
) -> torch.Tensor:
    safe_temp = max(1e-4, float(temperature))
    gate_probs = torch.softmax(gate_logits / safe_temp, dim=-1)
    if min_prob > 0:
        gate_probs = torch.clamp(gate_probs, min=min_prob)
        gate_probs = gate_probs / gate_probs.sum(dim=-1, keepdim=True)
    return gate_probs


def _rebase_output_dir(output_dir: Path, run_id: str, enabled: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not enabled:
        return
    existing_items = [
        item for item in output_dir.iterdir() if item.name != "archive"
    ]
    if not existing_items:
        return
    archive_dir = output_dir / "archive" / run_id
    archive_dir.mkdir(parents=True, exist_ok=False)
    for item in existing_items:
        shutil.move(str(item), str(archive_dir / item.name))


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
    epoch_idx: int = 0,
) -> float:
    """
    Train one epoch with curriculum routing (soft -> topk_soft -> topk_hard).
    """
    model.train()
    gating_model.train()
    total_loss = 0.0
    total_batches = 0
    accumulation_steps = config.gradient_accumulation_steps

    iterator = data_loader
    if tqdm is not None:
        iterator = tqdm(data_loader, desc="Training", leave=False)

    optimizer.zero_grad()

    routing_mode = _compute_routing_mode(epoch_idx, config)
    temperature = _compute_routing_temperature(epoch_idx, config)
    min_prob = 0.0
    if routing_mode == "soft":
        min_prob = min(
            max(config.min_expert_usage_fraction, 0.0),
            1.0 / max(1, config.num_experts),
        )
    print(
        f"  [Epoch {epoch_idx + 1}] Routing mode: {routing_mode}, "
        f"Temperature: {temperature:.2f}"
    )

    for batch_idx, batch in enumerate(iterator):
        batch = {k: v.to(device) for k, v in batch.items()}
        input_features = batch["input_features"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask")
        decoder_attention_mask = batch.get("decoder_attention_mask")
        batch_size = input_features.size(0)

        # OPTIMIZATION: Run encoder ONCE for gating decisions
        with torch.no_grad():
            encoder_outputs = embedding_model.encoder(
                input_features, attention_mask=attention_mask
            )
            pooled = encoder_outputs.last_hidden_state.mean(dim=1)

        # Get gating decisions
        gate_logits = gating_model(pooled)
        if config.router_noise_std > 0 and routing_mode != "topk_hard":
            gate_logits = gate_logits + torch.randn_like(gate_logits) * config.router_noise_std
        gate_probs = _compute_gating_probabilities(
            gate_logits,
            temperature=temperature,
            min_prob=min_prob,
        )
        gating_entropy = _gating_entropy(gate_probs)
        routing_alignment_loss = None

        if routing_mode == "soft":
            expert_losses = torch.zeros(
                batch_size, config.num_experts, device=device
            )
            for expert_id in range(config.num_experts):
                if config.use_lora:
                    model.set_adapter(f"expert_{expert_id}")

                with torch.cuda.amp.autocast(enabled=config.fp16):
                    model_kwargs = {
                        "input_features": input_features,
                        "labels": labels,
                    }
                    if attention_mask is not None:
                        model_kwargs["attention_mask"] = attention_mask
                    if decoder_attention_mask is not None:
                        model_kwargs["decoder_attention_mask"] = decoder_attention_mask
                    outputs = _forward_model(model, **model_kwargs)
                    loss_per_sample = _sequence_loss(outputs.logits, labels)

                expert_losses[:, expert_id] = loss_per_sample

            per_sample_loss = (gate_probs * expert_losses).sum(dim=-1)
            routing_alignment_loss = _routing_alignment_loss(
                gate_probs,
                expert_losses,
                config.routing_alignment_temperature,
            )
        else:
            per_sample_loss = torch.zeros(batch_size, device=device)
            k = max(1, min(config.top_k_experts, gate_probs.size(-1)))
            topk_values, topk_indices = torch.topk(gate_probs, k=k, dim=-1)
            topk_weights = topk_values / topk_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            for rank in range(k):
                experts_at_rank = topk_indices[:, rank]
                weights_at_rank = topk_weights[:, rank]

                for expert_id in torch.unique(experts_at_rank).tolist():
                    if config.use_lora:
                        model.set_adapter(f"expert_{expert_id}")

                    mask = experts_at_rank == expert_id
                    if not mask.any():
                        continue

                    expert_features = input_features[mask]
                    expert_labels = labels[mask]
                    expert_attention_mask = (
                        attention_mask[mask] if attention_mask is not None else None
                    )
                    expert_decoder_attention_mask = (
                        decoder_attention_mask[mask] if decoder_attention_mask is not None else None
                    )
                    expert_weights = weights_at_rank[mask]

                    with torch.cuda.amp.autocast(enabled=config.fp16):
                        model_kwargs = {
                            "input_features": expert_features,
                            "labels": expert_labels,
                        }
                        if expert_attention_mask is not None:
                            model_kwargs["attention_mask"] = expert_attention_mask
                        if expert_decoder_attention_mask is not None:
                            model_kwargs["decoder_attention_mask"] = expert_decoder_attention_mask
                        outputs = _forward_model(model, **model_kwargs)
                        loss_per_sample = _sequence_loss(outputs.logits, expert_labels)

                    indices = torch.where(mask)[0]
                    per_sample_loss[indices] += expert_weights * loss_per_sample

        main_loss = per_sample_loss.mean()
        balance_loss = _load_balance_loss(gate_probs, config.num_experts)
        loss = main_loss + config.load_balance_coef * balance_loss
        if routing_mode == "soft" and routing_alignment_loss is not None:
            loss = loss + config.routing_alignment_coef * routing_alignment_loss
        if routing_mode == "soft" and config.routing_entropy_coef > 0:
            loss = loss - config.routing_entropy_coef * gating_entropy

        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()

        # Step optimizer every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        total_batches += 1

        # Diagnostic output: expert usage statistics
        if batch_idx % 50 == 0:
            expert_usage = gate_probs.mean(dim=0)  # [num_experts]
            entropy = gating_entropy
            max_usage = expert_usage.max().item()
            min_usage = expert_usage.min().item()
            print(
                f"    Batch {batch_idx}: entropy={entropy:.3f}, "
                f"max_expert={max_usage:.3f}, min_expert={min_usage:.3f}, "
                f"usage_range={max_usage - min_usage:.3f}"
            )

        if tqdm is not None and isinstance(iterator, tqdm):
            iterator.set_postfix({
                "loss": f"{loss.item() * accumulation_steps:.4f}",
                "mode": routing_mode,
            })

    # Handle remaining gradients
    if total_batches % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / max(total_batches, 1)


def _evaluate(
    model: nn.Module,
    gating_model: nn.Module,
    embedding_model: nn.Module,
    data_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    epoch_idx: int = 0,
) -> float:
    """Evaluate with the same routing curriculum as training."""
    model.eval()
    gating_model.eval()
    total_loss = 0.0
    total_batches = 0

    iterator = data_loader
    if tqdm is not None:
        iterator = tqdm(data_loader, desc="Evaluating", leave=False)

    routing_mode = _compute_routing_mode(epoch_idx, config)
    temperature = _compute_routing_temperature(epoch_idx, config)
    min_prob = 0.0
    if routing_mode == "soft":
        min_prob = min(
            max(config.min_expert_usage_fraction, 0.0),
            1.0 / max(1, config.num_experts),
        )

    with torch.no_grad():
        for batch in iterator:
            batch = {k: v.to(device) for k, v in batch.items()}
            input_features = batch["input_features"]
            labels = batch["labels"]
            attention_mask = batch.get("attention_mask")
            decoder_attention_mask = batch.get("decoder_attention_mask")
            batch_size = input_features.size(0)

            encoder_outputs = embedding_model.encoder(
                input_features, attention_mask=attention_mask
            )
            pooled = encoder_outputs.last_hidden_state.mean(dim=1)
            gate_logits = gating_model(pooled)
            gate_probs = _compute_gating_probabilities(
                gate_logits,
                temperature=temperature,
                min_prob=min_prob,
            )
            gating_entropy = _gating_entropy(gate_probs)
            routing_alignment_loss = None

            if routing_mode == "soft":
                expert_losses = torch.zeros(
                    batch_size, config.num_experts, device=device
                )
                for expert_id in range(config.num_experts):
                    if config.use_lora:
                        model.set_adapter(f"expert_{expert_id}")

                    model_kwargs = {
                        "input_features": input_features,
                        "labels": labels,
                    }
                    if attention_mask is not None:
                        model_kwargs["attention_mask"] = attention_mask
                    if decoder_attention_mask is not None:
                        model_kwargs["decoder_attention_mask"] = decoder_attention_mask
                    outputs = _forward_model(model, **model_kwargs)
                    loss_per_sample = _sequence_loss(outputs.logits, labels)
                    expert_losses[:, expert_id] = loss_per_sample

                per_sample_loss = (gate_probs * expert_losses).sum(dim=-1)
                routing_alignment_loss = _routing_alignment_loss(
                    gate_probs,
                    expert_losses,
                    config.routing_alignment_temperature,
                )
            else:
                per_sample_loss = torch.zeros(batch_size, device=device)
                k = max(1, min(config.top_k_experts, gate_probs.size(-1)))
                topk_values, topk_indices = torch.topk(gate_probs, k=k, dim=-1)
                topk_weights = topk_values / topk_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)

                for rank in range(k):
                    experts_at_rank = topk_indices[:, rank]
                    weights_at_rank = topk_weights[:, rank]

                    for expert_id in torch.unique(experts_at_rank).tolist():
                        if config.use_lora:
                            model.set_adapter(f"expert_{expert_id}")

                        mask = experts_at_rank == expert_id
                        if not mask.any():
                            continue

                        expert_features = input_features[mask]
                        expert_labels = labels[mask]
                        expert_attention_mask = (
                            attention_mask[mask] if attention_mask is not None else None
                        )
                        expert_decoder_attention_mask = (
                            decoder_attention_mask[mask] if decoder_attention_mask is not None else None
                        )
                        expert_weights = weights_at_rank[mask]

                        model_kwargs = {
                            "input_features": expert_features,
                            "labels": expert_labels,
                        }
                        if expert_attention_mask is not None:
                            model_kwargs["attention_mask"] = expert_attention_mask
                        if expert_decoder_attention_mask is not None:
                            model_kwargs["decoder_attention_mask"] = expert_decoder_attention_mask
                        outputs = _forward_model(model, **model_kwargs)
                        loss_per_sample = _sequence_loss(outputs.logits, expert_labels)

                        indices = torch.where(mask)[0]
                        per_sample_loss[indices] += expert_weights * loss_per_sample

            main_loss = per_sample_loss.mean()
            balance_loss = _load_balance_loss(gate_probs, config.num_experts)
            loss = main_loss + config.load_balance_coef * balance_loss
            if routing_mode == "soft" and routing_alignment_loss is not None:
                loss = loss + config.routing_alignment_coef * routing_alignment_loss
            if routing_mode == "soft" and config.routing_entropy_coef > 0:
                loss = loss - config.routing_entropy_coef * gating_entropy

            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


def _evaluate_wer(
    model: nn.Module,
    gating_model: nn.Module,
    embedding_model: nn.Module,
    data_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    processor: "WhisperProcessor",
    epoch_idx: int = 0,
) -> float:
    model.eval()
    gating_model.eval()
    scorer = WERScorer(normalize=True)
    references: List[str] = []
    hypotheses: List[str] = []

    iterator = data_loader
    if tqdm is not None:
        iterator = tqdm(data_loader, desc="Computing WER", leave=False)

    routing_mode = _compute_routing_mode(epoch_idx, config)
    temperature = _compute_routing_temperature(epoch_idx, config)
    min_prob = 0.0
    if routing_mode == "soft":
        min_prob = min(
            max(config.min_expert_usage_fraction, 0.0),
            1.0 / max(1, config.num_experts),
        )

    with torch.no_grad():
        for batch in iterator:
            batch = {k: v.to(device) for k, v in batch.items()}
            input_features = batch["input_features"]
            labels = batch["labels"]
            attention_mask = batch.get("attention_mask")

            encoder_outputs = embedding_model.encoder(
                input_features, attention_mask=attention_mask
            )
            pooled = encoder_outputs.last_hidden_state.mean(dim=1)
            gate_logits = gating_model(pooled)
            gate_probs = _compute_gating_probabilities(
                gate_logits,
                temperature=temperature,
                min_prob=min_prob,
            )
            if routing_mode == "soft":
                top1_indices = torch.argmax(gate_probs, dim=-1)
            else:
                k = max(1, min(config.top_k_experts, gate_probs.size(-1)))
                _, topk_indices = torch.topk(gate_probs, k=k, dim=-1)
                top1_indices = topk_indices[:, 0]

            generated = [""] * input_features.size(0)
            max_length = _resolve_max_length(model, processor)
            for expert_id in torch.unique(top1_indices).tolist():
                if config.use_lora:
                    model.set_adapter(f"expert_{expert_id}")
                mask = top1_indices == expert_id
                if not mask.any():
                    continue
                feats = input_features[mask]
                feats_mask = attention_mask[mask] if attention_mask is not None else None
                generated_ids = model.generate(
                    input_features=feats,
                    attention_mask=feats_mask,
                    max_length=max_length,
                )
                preds = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                for idx, pred in zip(torch.where(mask)[0].tolist(), preds):
                    generated[idx] = pred

            labels = labels.clone()
            labels[labels == -100] = processor.tokenizer.pad_token_id
            refs = processor.batch_decode(labels, skip_special_tokens=True)

            references.extend(refs)
            hypotheses.extend(generated)

    return scorer.corpus_wer(references, hypotheses)


def _save_expert_adapters(
    model: nn.Module,
    processor: "WhisperProcessor",
    output_dir: Path,
    num_experts: int,
) -> None:
    for expert_id in range(num_experts):
        adapter_name = f"expert_{expert_id}"
        adapter_dir = output_dir / adapter_name
        adapter_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "set_adapter"):
            model.set_adapter(adapter_name)
        saved = False
        if hasattr(model, "save_adapter"):
            try:
                model.save_adapter(str(adapter_dir), adapter_name=adapter_name)
                saved = True
            except TypeError:
                try:
                    model.save_adapter(str(adapter_dir), adapter_name)
                    saved = True
                except TypeError:
                    pass
        if not saved:
            try:
                model.save_pretrained(adapter_dir, selected_adapters=[adapter_name])
                saved = True
            except TypeError:
                pass
        if not saved:
            model.save_pretrained(adapter_dir)
        processor.save_pretrained(adapter_dir)


def _save_checkpoint(
    model: nn.Module,
    gating_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    config: TrainingConfig,
    processor: "WhisperProcessor",
    output_dir: Path,
    is_best: bool = False,
    loss: float = None,
    wer: Optional[float] = None,
) -> None:
    """Save periodic checkpoints for resumability."""
    checkpoint_dir = output_dir / f"checkpoint_epoch_{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save training state
    torch.save({
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss": loss,
    }, checkpoint_dir / "training_state.pt")

    # Save gating model
    torch.save(gating_model.state_dict(), checkpoint_dir / "gating_model.pt")

    # Save expert adapters
    if config.use_lora:
        _save_expert_adapters(model, processor, checkpoint_dir, config.num_experts)
    else:
        model.save_pretrained(checkpoint_dir / "model")

    processor.save_pretrained(checkpoint_dir)

    # Update best checkpoint symlink/copy if this is best
    if is_best:
        best_dir = output_dir / "best"
        if best_dir.exists():
            import shutil
            shutil.rmtree(best_dir)
        import shutil
        shutil.copytree(checkpoint_dir, best_dir)

        with (output_dir / "best.json").open("w", encoding="utf-8") as f:
            payload = {"loss": loss, "epoch": epoch}
            if wer is not None:
                payload["wer"] = wer
            json.dump(payload, f, indent=2)


def train(config_path: str, max_samples: Optional[int] = None, exclude_impairment: Optional[str] = None) -> None:
    config = _load_training_config(config_path)
    if config.data_config_override is None:
        config.data_config_override = {}
    if max_samples is not None:
        config.data_config_override["max_samples"] = int(max_samples)
    if exclude_impairment is not None:
        config.data_config_override["exclude_impairment"] = exclude_impairment
    _ensure_dependencies(config.use_lora)
    _set_seed(config.seed)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    print(f"Train set: {len(train_set)}, Val set: {len(val_set)}")

    processor = WhisperProcessor.from_pretrained(config.model_name)
    collator = WhisperCollator(processor)

    # OPTIMIZATION: Added pin_memory for faster data transfer to GPU
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

    print(f"Building models...")
    embedding_model = _build_embedding_model(config, device)
    model = _build_moe_model(config).to(device)
    gating_model = _load_gating_model(config, device)

    if config.experts_dir and config.use_lora:
        _load_expert_adapters(model, Path(config.experts_dir), config.num_experts)

    _configure_generation(model, config.language, config.task)

    params = list(gating_model.parameters()) + [
        param for param in model.parameters() if param.requires_grad
    ]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

    output_dir = Path(config.output_dir)
    _rebase_output_dir(output_dir, run_id, config.rebase_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_id.txt").open("w", encoding="utf-8") as f:
        f.write(run_id)

    metrics_dir = Path(config.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"metrics_{run_id}.json"
    latest_metrics_path = metrics_dir / "metrics.json"
    metrics_history: List[Dict[str, float]] = []
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as metrics_file:
            try:
                metrics_history = json.load(metrics_file)
            except json.JSONDecodeError:
                metrics_history = []

    best_loss = float("inf")
    best_wer: Optional[float] = None
    best_path = output_dir / "best.json"
    if best_path.exists():
        with best_path.open("r", encoding="utf-8") as f:
            best_state = json.load(f)
            best_loss = best_state.get("loss", float("inf"))
            best_wer = best_state.get("wer")

    # Early stopping configuration
    early_stopping_patience = 5
    epochs_without_improvement = 0

    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"Batch size: {config.batch_size}, Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")

    for epoch in range(1, config.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config.epochs}")
        print(f"{'='*50}")

        epoch_idx = epoch - 1

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
            epoch_idx=epoch_idx,
        )

        val_loss = train_loss
        val_wer = None

        # OPTIMIZATION: Only evaluate every N epochs to save time
        should_eval = (epoch % config.eval_every_n_epochs == 0) or (epoch == config.epochs)

        if val_size > 0 and should_eval:
            val_loss = _evaluate(
                model=model,
                gating_model=gating_model,
                embedding_model=embedding_model,
                data_loader=val_loader,
                config=config,
                device=device,
                epoch_idx=epoch_idx,
            )
            try:
                val_wer = _evaluate_wer(
                    model=model,
                    gating_model=gating_model,
                    embedding_model=embedding_model,
                    data_loader=val_loader,
                    config=config,
                    device=device,
                    processor=processor,
                    epoch_idx=epoch_idx,
                )
            except Exception as exc:
                print(f"[WARNING] WER validation failed at epoch {epoch}: {exc}")
                val_wer = None

        metrics_entry = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss) if should_eval else None,
            "val_wer": float(val_wer) if val_wer is not None else None,
        }
        metrics_history.append(metrics_entry)
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            json.dump(metrics_history, metrics_file, indent=2)
        if metrics_path != latest_metrics_path:
            shutil.copyfile(metrics_path, latest_metrics_path)

        if val_wer is None:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        else:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_wer={val_wer:.4f}"
            )

        # OPTIMIZATION: Periodic checkpointing
        metric_improved = False
        if val_wer is not None:
            if best_wer is None or val_wer < best_wer:
                best_wer = val_wer
                metric_improved = True
        else:
            if val_loss < best_loss:
                metric_improved = True

        if val_loss < best_loss:
            best_loss = val_loss

        if metric_improved:
            epochs_without_improvement = 0
            if best_wer is not None:
                print(f"  -> New best WER: {best_wer:.4f}")
            else:
                print(f"  -> New best loss: {best_loss:.4f}")
        else:
            epochs_without_improvement += 1

        should_save = (
            config.save_every_epoch
            or (epoch % config.save_every_n_epochs == 0)
            or (epoch == config.epochs)
            or metric_improved
        )
        if should_save:
            print(f"  -> Saving checkpoint...")
            _save_checkpoint(
                model=model,
                gating_model=gating_model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                config=config,
                processor=processor,
                output_dir=output_dir,
                is_best=metric_improved,
                loss=val_loss,
                wer=val_wer,
            )

        # Clear CUDA cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {early_stopping_patience} epochs)")
            break

    if best_wer is None:
        print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    else:
        print(f"\nTraining complete! Best loss: {best_loss:.4f} | Best WER: {best_wer:.4f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint MoE training (gating model + Whisper LoRA experts)."
    )
    parser.add_argument(
        "--config",
        default="Config/asr_training.json",
        help="Path to JSON config.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override max_samples for data selection (e.g., 50).",
    )
    parser.add_argument(
        "--exclude-impairment",
        type=str,
        default=None,
        help="Exclude samples with a specific impairment/etiology from training (e.g., 'ALS', 'Parkinson's Disease').",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args.config, args.max_samples, args.exclude_impairment)
