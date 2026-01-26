# Curriculum Routing: From Soft to Hard Expert Routing
## Complete Implementation Guide for Cursor

---

## Overview

**Goal**: Train MoE without expert collapse by using curriculum learning for routing:
- **Phase 1 (Epochs 0–2)**: Soft routing – all experts see gradients
- **Phase 2 (Epochs 3–5)**: Top‑k soft – specialization with safety net
- **Phase 3 (Epochs 6+)**: Top‑k hard – final optimization (if needed)

**Key insight**: Experts need time to specialize. Hard routing before convergence locks the gate onto the best pre‑trained expert and causes collapse.

---

## Step 0: Update `TrainingConfig` (asr_training.py)

Add these fields to the `TrainingConfig` dataclass:

```python
@dataclass
class TrainingConfig:
    # ... existing fields ...
    
    # NEW: Routing curriculum schedule
    soft_routing_epochs: int  # Epochs with soft routing (all experts)
    topk_routing_epochs: int  # Epochs with top-k soft routing
    routing_temperature_start: float  # Initial temperature (higher = softer)
    routing_temperature_end: float  # Final temperature (lower = sharper)
    min_expert_usage_fraction: float  # Prevent expert death in hard phase
    load_balance_coef: float  # Strength of load-balancing loss
Update _load_training_config():

python
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
        load_balance_coef=float(config.get("load_balance_coef", 0.1)),  # stronger default
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
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        eval_every_n_epochs=int(config.get("eval_every_n_epochs", 1)),
        save_every_n_epochs=int(config.get("save_every_n_epochs", 5)),
        pin_memory=bool(config.get("pin_memory", True)),

        # NEW: Routing curriculum
        soft_routing_epochs=int(config.get("soft_routing_epochs", 3)),
        topk_routing_epochs=int(config.get("topk_routing_epochs", 6)),
        routing_temperature_start=float(config.get("routing_temperature_start", 2.0)),
        routing_temperature_end=float(config.get("routing_temperature_end", 0.5)),
        min_expert_usage_fraction=float(config.get("min_expert_usage_fraction", 0.05)),
    )
Step 1: Replace _load_balance_loss() (asr_training.py)
Replace the old implementation with:

python
def _load_balance_loss(probs: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    KL divergence between empirical expert distribution and uniform.
    
    Args:
        probs: [batch_size, num_experts] gating probabilities
        num_experts: number of experts
    
    Returns:
        Scalar loss encouraging uniform expert usage
    """
    # Empirical distribution: average gating probability per expert
    importance = probs.mean(dim=0)  # [num_experts]
    
    # Uniform target distribution
    uniform = torch.full_like(importance, 1.0 / num_experts)
    
    # KL divergence: D_KL(uniform || importance)
    loss = torch.sum(uniform * torch.log(uniform / (importance + 1e-8) + 1e-8))
    
    return loss
Step 2: Add Helper Functions (after _load_balance_loss())
python
def _compute_routing_mode(epoch_idx: int, config: TrainingConfig) -> str:
    """
    Determine routing mode based on training progress.
    
    Returns:
        "soft": All experts receive gradients (Phase 1)
        "topk_soft": Top-k experts with soft weights (Phase 2)
        "topk_hard": Top-k with hard selection (Phase 3)
    """
    if epoch_idx < config.soft_routing_epochs:
        return "soft"
    elif epoch_idx < config.topk_routing_epochs:
        return "topk_soft"
    else:
        return "topk_hard"


def _compute_routing_temperature(epoch_idx: int, config: TrainingConfig) -> float:
    """
    Compute routing temperature for current epoch (annealed schedule).
    """
    max_epoch = max(1, config.topk_routing_epochs)
    progress = min(1.0, epoch_idx / max_epoch)
    
    temperature = (
        config.routing_temperature_start * (1 - progress)
        + config.routing_temperature_end * progress
    )
    
    return temperature


def _compute_gating_probabilities(
    gate_logits: torch.Tensor, 
    temperature: float,
    min_prob: float = 0.0,
) -> torch.Tensor:
    """
    Compute gating probabilities with temperature control.
    """
    gate_probs = torch.softmax(gate_logits / temperature, dim=-1)
    
    if min_prob > 0:
        gate_probs = torch.clamp(gate_probs, min=min_prob)
        gate_probs = gate_probs / gate_probs.sum(dim=-1, keepdim=True)
    
    return gate_probs
Step 3: Replace _train_epoch() (core curriculum routing)
Replace the whole function with:

python
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
    epoch_idx: int = 0,  # NEW
) -> float:
    """
    Train one epoch with curriculum routing (soft → topk_soft → topk_hard).
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

    # Determine routing mode and temperature for this epoch
    routing_mode = _compute_routing_mode(epoch_idx, config)
    temperature = _compute_routing_temperature(epoch_idx, config)
    
    print(f"  [Epoch {epoch_idx}] Routing mode: {routing_mode}, Temperature: {temperature:.2f}")

    for batch_idx, batch in enumerate(iterator):
        batch = {k: v.to(device) for k, v in batch.items()}
        input_features = batch["input_features"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask")
        decoder_attention_mask = batch.get("decoder_attention_mask")
        batch_size = input_features.size(0)

        # Encoder once
        with torch.no_grad():
            encoder_outputs = embedding_model.encoder(
                input_features, attention_mask=attention_mask
            )
            pooled = encoder_outputs.last_hidden_state.mean(dim=1)

        gate_logits = gating_model(pooled)
        gate_probs = _compute_gating_probabilities(
            gate_logits, 
            temperature=temperature,
            min_prob=0.0,
        )

        per_sample_loss = torch.zeros(batch_size, device=device)

        # -------- Phase 1: soft routing --------
        if routing_mode == "soft":
            for expert_id in range(config.num_experts):
                if config.use_lora:
                    model.set_adapter(f"expert_{expert_id}")

                expert_weight = gate_probs[:, expert_id]
                if expert_weight.mean() < 1e-4:
                    continue

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

                per_sample_loss += expert_weight * loss_per_sample

        # -------- Phase 2: top-k soft --------
        elif routing_mode == "topk_soft":
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

                    indices = torch.where(mask)
                    per_sample_loss[indices] += expert_weights * loss_per_sample

        # -------- Phase 3: top-k hard (same math as topk_soft, just later epochs) --------
        else:  # "topk_hard"
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

                    indices = torch.where(mask)
                    per_sample_loss[indices] += expert_weights * loss_per_sample

        main_loss = per_sample_loss.mean()
        balance_loss = _load_balance_loss(gate_probs, config.num_experts)
        loss = main_loss + config.load_balance_coef * balance_loss

        loss = loss / accumulation_steps
        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        total_batches += 1

        if tqdm is not None and isinstance(iterator, tqdm):
            iterator.set_postfix({
                "loss": f"{loss.item() * accumulation_steps:.4f}",
                "mode": routing_mode,
            })

    if total_batches % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / max(total_batches, 1)
Step 4: Update _evaluate() to mirror training routing
python
def _evaluate(
    model: nn.Module,
    gating_model: nn.Module,
    embedding_model: nn.Module,
    data_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    epoch_idx: int = 0,  # NEW
) -> float:
    """Evaluate with same routing mode as training."""
    model.eval()
    gating_model.eval()
    total_loss = 0.0
    total_batches = 0

    iterator = data_loader
    if tqdm is not None:
        iterator = tqdm(data_loader, desc="Evaluating", leave=False)

    routing_mode = _compute_routing_mode(epoch_idx, config)
    temperature = _compute_routing_temperature(epoch_idx, config)

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
                gate_logits, temperature=temperature, min_prob=0.0
            )

            per_sample_loss = torch.zeros(batch_size, device=device)

            if routing_mode == "soft":
                for expert_id in range(config.num_experts):
                    if config.use_lora:
                        model.set_adapter(f"expert_{expert_id}")

                    expert_weight = gate_probs[:, expert_id]
                    if expert_weight.mean() < 1e-4:
                        continue

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
                    per_sample_loss += expert_weight * loss_per_sample

            else:  # topk_soft or topk_hard
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

                        indices = torch.where(mask)
                        per_sample_loss[indices] += expert_weights * loss_per_sample

            main_loss = per_sample_loss.mean()
            balance_loss = _load_balance_loss(gate_probs, config.num_experts)
            loss = main_loss + config.load_balance_coef * balance_loss

            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)
Step 5: Update _evaluate_wer() with temperature (still top‑1 for decoding)
python
def _evaluate_wer(
    model: nn.Module,
    gating_model: nn.Module,
    embedding_model: nn.Module,
    data_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    processor: "WhisperProcessor",
    epoch_idx: int = 0,  # NEW
) -> float:
    """Evaluate WER with temperature-controlled routing."""
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
                gate_logits, temperature=temperature, min_prob=0.0
            )

            batch_size = input_features.size(0)
            generated = [""] * batch_size

            if routing_mode == "soft":
                top1_indices = torch.argmax(gate_probs, dim=-1)
                for expert_id in torch.unique(top1_indices).tolist():
                    if config.use_lora:
                        model.set_adapter(f"expert_{expert_id}")
                    mask = top1_indices == expert_id
                    if not mask.any():
                        continue
                    feats = input_features[mask]
                    feats_mask = attention_mask[mask] if attention_mask is not None else None
                    generated_ids = model.generate(
                        input_features=feats, attention_mask=feats_mask
                    )
                    preds = processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )
                    for idx, pred in zip(torch.where(mask).tolist(), preds):
                        generated[idx] = pred

            else:
                k = max(1, min(config.top_k_experts, gate_probs.size(-1)))
                topk_values, topk_indices = torch.topk(gate_probs, k=k, dim=-1)
                top1_experts = topk_indices[:, 0]
                for expert_id in torch.unique(top1_experts).tolist():
                    if config.use_lora:
                        model.set_adapter(f"expert_{expert_id}")
                    mask = top1_experts == expert_id
                    if not mask.any():
                        continue
                    feats = input_features[mask]
                    feats_mask = attention_mask[mask] if attention_mask is not None else None
                    generated_ids = model.generate(
                        input_features=feats, attention_mask=feats_mask
                    )
                    preds = processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )
                    for idx, pred in zip(torch.where(mask).tolist(), preds):
                        generated[idx] = pred

            labels = labels.clone()
            labels[labels == -100] = processor.tokenizer.pad_token_id
            refs = processor.batch_decode(labels, skip_special_tokens=True)

            references.extend(refs)
            hypotheses.extend(generated)

    return scorer.corpus_wer(references, hypotheses)
Step 6: Update training loop to pass epoch_idx
In train():

python
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
        epoch_idx=epoch_idx,  # NEW
    )

    val_loss = train_loss
    val_wer = None

    should_eval = (epoch % config.eval_every_n_epochs == 0) or (epoch == config.epochs)

    if val_size > 0 and should_eval:
        val_loss = _evaluate(
            model=model,
            gating_model=gating_model,
            embedding_model=embedding_model,
            data_loader=val_loader,
            config=config,
            device=device,
            epoch_idx=epoch_idx,  # NEW
        )
        val_wer = _evaluate_wer(
            model=model,
            gating_model=gating_model,
            embedding_model=embedding_model,
            data_loader=val_loader,
            config=config,
            device=device,
            processor=processor,
            epoch_idx=epoch_idx,  # NEW
        )
    # ... rest unchanged ...
Step 7: Update Config/asr_training.json
Add routing parameters:

json
{
  "model_name": "openai/whisper-large-v2",
  "language": null,
  "task": "transcribe",
  "data_config_path": "Config/dataloader_config.json",
  "data_mode": "default",
  "num_experts": 8,
  "top_k_experts": 2,
  "batch_size": 4,
  "epochs": 9,
  "learning_rate": 1e-4,
  "weight_decay": 0.0,
  "seed": 42,
  "num_workers": 4,
  "val_split": 0.1,

  "use_lora": true,
  "lora": {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"]
  },

  "gating_model_config": "Config/gating_model_config.json",
  "output_dir": "checkpoints/asr",
  "metrics_dir": "Evaluation/asr_training_results",
  "fp16": true,

  "gradient_accumulation_steps": 1,
  "eval_every_n_epochs": 1,
  "save_every_n_epochs": 1,
  "pin_memory": true,

  "load_balance_coef": 0.1,

  "soft_routing_epochs": 3,
  "topk_routing_epochs": 6,
  "routing_temperature_start": 2.0,
  "routing_temperature_end": 0.5,
  "min_expert_usage_fraction": 0.05
}
Step 8 (Optional): Diagnostics
Inside _train_epoch(), after gate_probs:

python
entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=-1).mean()
expert_usage = gate_probs.mean(dim=0)

if batch_idx % 50 == 0:
    print(
        f"  Batch {batch_idx}: entropy={entropy:.3f}, "
        f"max_expert_prob={expert_usage.max():.3f}, "
        f"min_expert_prob={expert_usage.min():.3f}"
    )
Quick run
bash
python Training_Scripts/asr_training.py \
  --config Config/asr_training.json \
  --max-samples 50
You should see:

Routing mode change from soft → topk_soft → topk_hard

Entropy decreasing over epochs

Expert usage becoming diverse instead of collapsing to a single expert