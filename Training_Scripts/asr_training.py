import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
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
        max_label_length = getattr(self.processor.tokenizer, "model_max_length", 448)
        if max_label_length is None or max_label_length > 1000:
            max_label_length = 448
        labels = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_label_length,
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
        metrics_dir=config.get("metrics_dir", "Evaluation/asr_training_results"),
        # New optimization parameters with sensible defaults
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        eval_every_n_epochs=int(config.get("eval_every_n_epochs", 1)),
        save_every_n_epochs=int(config.get("save_every_n_epochs", 5)),
        pin_memory=bool(config.get("pin_memory", True)),
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
    """
    OPTIMIZED: Routes per-expert SUBSETS instead of running full batch per expert.
    This reduces compute from O(batch * num_experts) to O(batch * top_k_avg).
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

    for batch_idx, batch in enumerate(iterator):
        batch = {k: v.to(device) for k, v in batch.items()}
        input_features = batch["input_features"]
        labels = batch["labels"]
        batch_size = input_features.size(0)

        # OPTIMIZATION: Run encoder ONCE for gating decisions
        with torch.no_grad():
            encoder_outputs = embedding_model.encoder(input_features)
            pooled = encoder_outputs.last_hidden_state.mean(dim=1)

        # Get gating decisions
        gate_logits = gating_model(pooled)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        k = max(1, min(config.top_k_experts, gate_probs.size(-1)))
        topk_values, topk_indices = torch.topk(gate_probs, k=k, dim=-1)
        topk_weights = topk_values / topk_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # OPTIMIZATION: Route per-expert SUBSETS instead of full batch per expert
        # For each sample, find its primary expert (highest weight) and route there
        primary_expert = topk_indices[:, 0]  # [batch_size]
        unique_experts = torch.unique(primary_expert).tolist()

        per_sample_loss = torch.zeros(batch_size, device=device)

        for expert_id in unique_experts:
            if config.use_lora:
                model.set_adapter(f"expert_{expert_id}")

            # Only process samples routed to this expert
            mask = primary_expert == expert_id
            if not mask.any():
                continue

            expert_features = input_features[mask]
            expert_labels = labels[mask]
            expert_weights = topk_weights[mask, 0]  # Primary expert weight

            with torch.cuda.amp.autocast(enabled=config.fp16):
                outputs = _forward_model(
                    model, input_features=expert_features, labels=expert_labels
                )
                logits = outputs.logits
                loss_per_sample = _sequence_loss(logits, expert_labels)

            # Weight the loss by expert weight
            weighted_loss = expert_weights * loss_per_sample
            # Scatter back to original positions
            indices = torch.where(mask)[0]
            per_sample_loss[indices] = weighted_loss

        # Handle secondary experts if top_k > 1
        if k > 1:
            secondary_expert = topk_indices[:, 1]  # [batch_size]
            unique_secondary = torch.unique(secondary_expert).tolist()

            for expert_id in unique_secondary:
                if config.use_lora:
                    model.set_adapter(f"expert_{expert_id}")

                mask = secondary_expert == expert_id
                if not mask.any():
                    continue

                expert_features = input_features[mask]
                expert_labels = labels[mask]
                expert_weights = topk_weights[mask, 1]  # Secondary expert weight

                with torch.cuda.amp.autocast(enabled=config.fp16):
                    outputs = _forward_model(
                        model, input_features=expert_features, labels=expert_labels
                    )
                    logits = outputs.logits
                    loss_per_sample = _sequence_loss(logits, expert_labels)

                weighted_loss = expert_weights * loss_per_sample
                indices = torch.where(mask)[0]
                per_sample_loss[indices] += weighted_loss

        main_loss = per_sample_loss.mean()
        balance_loss = _load_balance_loss(gate_probs, config.num_experts)
        loss = main_loss + config.load_balance_coef * balance_loss

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

        if tqdm is not None and isinstance(iterator, tqdm):
            iterator.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}")

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
) -> float:
    """OPTIMIZED: Same per-expert subset routing as training."""
    model.eval()
    gating_model.eval()
    total_loss = 0.0
    total_batches = 0

    iterator = data_loader
    if tqdm is not None:
        iterator = tqdm(data_loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in iterator:
            batch = {k: v.to(device) for k, v in batch.items()}
            input_features = batch["input_features"]
            labels = batch["labels"]
            batch_size = input_features.size(0)

            encoder_outputs = embedding_model.encoder(input_features)
            pooled = encoder_outputs.last_hidden_state.mean(dim=1)
            gate_logits = gating_model(pooled)
            gate_probs = torch.softmax(gate_logits, dim=-1)
            k = max(1, min(config.top_k_experts, gate_probs.size(-1)))
            topk_values, topk_indices = torch.topk(gate_probs, k=k, dim=-1)
            topk_weights = topk_values / topk_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            primary_expert = topk_indices[:, 0]
            unique_experts = torch.unique(primary_expert).tolist()

            per_sample_loss = torch.zeros(batch_size, device=device)

            for expert_id in unique_experts:
                if config.use_lora:
                    model.set_adapter(f"expert_{expert_id}")

                mask = primary_expert == expert_id
                if not mask.any():
                    continue

                expert_features = input_features[mask]
                expert_labels = labels[mask]
                expert_weights = topk_weights[mask, 0]

                outputs = _forward_model(
                    model, input_features=expert_features, labels=expert_labels
                )
                logits = outputs.logits
                loss_per_sample = _sequence_loss(logits, expert_labels)

                weighted_loss = expert_weights * loss_per_sample
                indices = torch.where(mask)[0]
                per_sample_loss[indices] = weighted_loss

            if k > 1:
                secondary_expert = topk_indices[:, 1]
                unique_secondary = torch.unique(secondary_expert).tolist()

                for expert_id in unique_secondary:
                    if config.use_lora:
                        model.set_adapter(f"expert_{expert_id}")

                    mask = secondary_expert == expert_id
                    if not mask.any():
                        continue

                    expert_features = input_features[mask]
                    expert_labels = labels[mask]
                    expert_weights = topk_weights[mask, 1]

                    outputs = _forward_model(
                        model, input_features=expert_features, labels=expert_labels
                    )
                    logits = outputs.logits
                    loss_per_sample = _sequence_loss(logits, expert_labels)

                    weighted_loss = expert_weights * loss_per_sample
                    indices = torch.where(mask)[0]
                    per_sample_loss[indices] += weighted_loss

            main_loss = per_sample_loss.mean()
            balance_loss = _load_balance_loss(gate_probs, config.num_experts)
            loss = main_loss + config.load_balance_coef * balance_loss

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
) -> float:
    model.eval()
    gating_model.eval()
    scorer = WERScorer(normalize=True)
    references: List[str] = []
    hypotheses: List[str] = []

    iterator = data_loader
    if tqdm is not None:
        iterator = tqdm(data_loader, desc="Computing WER", leave=False)

    with torch.no_grad():
        for batch in iterator:
            batch = {k: v.to(device) for k, v in batch.items()}
            input_features = batch["input_features"]
            labels = batch["labels"]

            encoder_outputs = embedding_model.encoder(input_features)
            pooled = encoder_outputs.last_hidden_state.mean(dim=1)
            gate_logits = gating_model(pooled)
            gate_probs = torch.softmax(gate_logits, dim=-1)
            top1_indices = torch.argmax(gate_probs, dim=-1)

            generated = [""] * input_features.size(0)
            for expert_id in torch.unique(top1_indices).tolist():
                if config.use_lora:
                    model.set_adapter(f"expert_{expert_id}")
                mask = top1_indices == expert_id
                if not mask.any():
                    continue
                feats = input_features[mask]
                generated_ids = model.generate(input_features=feats)
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
            json.dump({"loss": loss, "epoch": epoch}, f, indent=2)


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

    metrics_dir = Path(config.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "metrics.json"
    metrics_history: List[Dict[str, float]] = []
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as metrics_file:
            try:
                metrics_history = json.load(metrics_file)
            except json.JSONDecodeError:
                metrics_history = []

    best_loss = float("inf")
    best_path = output_dir / "best.json"
    if best_path.exists():
        with best_path.open("r", encoding="utf-8") as f:
            best_loss = json.load(f).get("loss", float("inf"))

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
            )
            val_wer = _evaluate_wer(
                model=model,
                gating_model=gating_model,
                embedding_model=embedding_model,
                data_loader=val_loader,
                config=config,
                device=device,
                processor=processor,
            )

        metrics_entry = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss) if should_eval else None,
            "val_wer": float(val_wer) if val_wer is not None else None,
        }
        metrics_history.append(metrics_entry)
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            json.dump(metrics_history, metrics_file, indent=2)

        if val_wer is None:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        else:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_wer={val_wer:.4f}"
            )

        # OPTIMIZATION: Periodic checkpointing
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            epochs_without_improvement = 0
            print(f"  -> New best loss: {best_loss:.4f}")
        else:
            epochs_without_improvement += 1

        should_save = (epoch % config.save_every_n_epochs == 0) or (epoch == config.epochs) or is_best
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
                is_best=is_best,
                loss=val_loss,
            )

        # Clear CUDA cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {early_stopping_patience} epochs)")
            break

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")


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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args.config, args.max_samples)
