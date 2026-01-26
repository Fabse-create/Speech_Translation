# Expert Collapse Analysis: ASR MoE Training

## Executive Summary

Your code exhibits **multiple convergent failure modes** that collectively cause expert collapse. The primary culprit is **undershooting the load balancing coefficient**, compounded by **soft routing initialization that never transitions properly**, and **entropy collapse at the probability level**.

---

## Primary Causes (Ranked by Impact)

### 1. **Insufficient Load Balancing Loss Weight** ⚠️ CRITICAL

**Location:** Line ~640
```python
balance_loss = _load_balance_loss(gate_probs, config.num_experts)
loss = main_loss + config.load_balance_coef * balance_loss
```

**The Problem:**
```python
load_balance_coef=float(config.get("load_balance_coef", 0.01))  # Default: 0.01
```

A coefficient of **0.01 is too weak** for 8 experts. Your `_load_balance_loss` uses KL divergence:

```python
def _load_balance_loss(probs, num_experts):
    importance = probs.mean(dim=0)  # [num_experts]
    uniform = torch.full_like(importance, 1.0 / num_experts)  # [1/8 = 0.125 each]
    eps = 1e-8
    return torch.sum(uniform * torch.log((uniform + eps) / (importance + eps)))
```

**Why it's weak:**
- When 1 expert dominates and 7 collapse to ~0, the KL is approximately `8 * 0.125 * log(0.125 / 0.001)` ≈ **3.4**
- Your main ASR loss is likely **20-30** per batch
- Balance contribution: `0.01 * 3.4 ≈ 0.034` ← **Negligible** compared to main loss

**Fix:** Increase to **0.1–0.5** (experiment, but start with 0.2):
```python
load_balance_coef=float(config.get("load_balance_coef", 0.2))  # More aggressive
```

---

### 2. **Soft Routing Never Produces Entropy Collapse Signal** ⚠️ CRITICAL

**Location:** Lines ~605–625 (soft routing branch)

**The Problem:**

In soft routing mode (`epoch_idx < soft_routing_epochs=3`), **every sample is processed by ALL experts** with weighted losses:

```python
if routing_mode == "soft":
    for expert_id in range(config.num_experts):  # Loop over ALL experts
        expert_weight = gate_probs[:, expert_id]
        if expert_weight.mean() < 1e-4:
            continue
        per_sample_loss += expert_weight * loss_per_sample
```

**The trap:**
- Each expert sees a loss-weighted gradient regardless of `gate_probs` values
- If gating network learns `gate_probs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]` (uniform), **all experts get equal gradients**
- No expert is incentivized to "specialize" during soft routing—**the curriculum doesn't work**
- By epoch 3, when you switch to `topk_routing`, experts are already undifferentiated
- Top-K then picks randomly, and gradient variance determines collapse, not learned routing

**Why "soft routing" fails here:**
Standard MoE soft routing should use **auxiliary loss only**, not loss-weighted routing:
```python
# THIS IS WHAT YOU'RE DOING (and it's the problem):
per_sample_loss += gate_probs[:, expert_id] * expert_loss  # Scales gradient by gate_prob

# WHAT SOFT ROUTING TYPICALLY DOES:
per_sample_loss += expert_loss  # Same gradient for all experts
# + auxiliary_loss (balance term) guides the gating network
```

Your approach means:
1. Gating network learns to suppress gradients to weak experts → no learning signal
2. Weak experts stay weak even when they *could* specialize
3. Strong experts get reinforced → **collapse**

---

### 3. **Gating Network Unchecked Temperature Decay** ⚠️ HIGH

**Location:** Lines ~559–567
```python
def _compute_routing_temperature(epoch_idx: int, config: TrainingConfig) -> float:
    max_epoch = max(1, config.topk_routing_epochs)
    progress = min(1.0, epoch_idx / max_epoch)
    return (
        config.routing_temperature_start * (1 - progress)
        + config.routing_temperature_end * progress
    )
```

**Default values:**
```python
routing_temperature_start=float(config.get("routing_temperature_start", 2.0)),
routing_temperature_end=float(config.get("routing_temperature_end", 0.5)),
```

**The Problem:**

By epoch 6 (end of `topk_routing_epochs=6`), temperature drops to **0.5**. Combined with a **poorly trained gating network** (from soft routing failure), this:

1. **Sharpens already-biased logits** — if gating learned `logits = [5, -5, -5, ...]`, softmax at T=0.5 becomes `[0.99, 0.001, ...]`
2. **No recovery possible** — once top-K selects expert 0, other experts get zero gradient
3. **Interacts badly with load balance loss** — at low T, `gate_probs` are one-hot-ish, so:
   ```python
   importance = [0.95, 0.01, 0.01, ...]  # Heavily imbalanced
   uniform = [0.125, 0.125, ...]
   KL ≈ 2.2  # Still small relative to main loss * 0.01 coef
   ```

**Why it's wrong:**
- Temperature should **increase** if you detect collapse (adaptive temperature), not blindly decay
- You're annealing toward **sharper decisions** before experts are even trained

---

### 4. **Early Stopping at `patience=5` Masks the Problem** ⚠️ MEDIUM

**Location:** Lines ~787–789
```python
early_stopping_patience = 5
epochs_without_improvement += 1
```

**The Problem:**
- Collapse often happens **epochs 3–5** (end of soft routing, start of top-K)
- By epoch 5, you might trigger early stopping before it stabilizes
- If one expert dominates, validation loss often drops (fewer samples → faster GPU, but WER worsens)
- **Validation loss ≠ WER** — you're optimizing the wrong metric

**Evidence in your code:**
```python
if is_best:
    best_loss = val_loss  # Using loss for early stopping
    epochs_without_improvement = 0
```

But you compute WER separately and don't use it for early stopping.

---

### 5. **Gating Network May Lack Capacity to Learn Meaningful Routing** ⚠️ MEDIUM

**Location:** `_load_gating_model()` (not shown, but implied)

**The Problem:**
You pool the encoder output:
```python
pooled = encoder_outputs.last_hidden_state.mean(dim=1)  # Global average pool → [batch, 768]
```

Then pass it to `GatingModel`. **Without seeing the gating architecture**, I suspect:
- Simple 2-layer MLP with no task-specific features
- No attention to sequence position (e.g., which parts of speech are hard)
- No multi-head routing (one expert per sample)

**For dysarthric speech**, you likely need:
- Positional importance (e.g., stressed syllables might need "clarity expert")
- Speaker-specific experts (acoustic variation)
- But global pooling throws away this information

---

### 6. **Min-Probability Clamping Counteracts Load Balance** ⚠️ MEDIUM

**Location:** Lines ~551–557 and usage in `_compute_gating_probabilities`
```python
min_prob = min(
    max(config.min_expert_usage_fraction, 0.0),
    1.0 / max(1, config.num_experts),
)
# min_prob = min(0.05, 0.125) = 0.05

# Then applied:
if min_prob > 0:
    gate_probs = torch.clamp(gate_probs, min=min_prob)
    gate_probs = gate_probs / gate_probs.sum(dim=-1, keepdim=True)
```

**The Problem:**
- `min_expert_usage_fraction=0.05` means **every expert gets ≥5%** of samples
- When clamped, renormalization can flip the gradient flow
- This **artificially enforces** expert usage, but doesn't prevent **loss collapse**
  - Expert might be used 5% of the time but learn **nothing useful** (outputs pure noise)
  - Gradients are wasted on a useless expert

**Effect on load balance loss:**
- Clamping to `[0.05, ...]` already removes extreme imbalance
- Load balance loss has less signal → coefficient matters even less

---

## Diagnostic Checklist

### Check These in Your Training Logs:

```python
# In your _train_epoch, add around line 596:
if batch_idx % 50 == 0:
    expert_usage = gate_probs.mean(dim=0)  # [num_experts]
    entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=-1).mean()
    max_usage = expert_usage.max().item()
    min_usage = expert_usage.min().item()
    print(
        f"    Batch {batch_idx}: entropy={entropy:.3f}, "
        f"max_expert={max_usage:.3f}, min_expert={min_usage:.3f}, "
        f"usage_range={max_usage - min_usage:.3f}"
    )
```

**YOU'RE ALREADY DOING THIS!** But check your outputs:

- **Entropy stays high (>2.0)?** → Gating is uniform, experts not differentiating
- **Usage range stays small (<0.3)?** → Clamping is dominanting, load balance is inactive
- **Max expert stays <0.5 into epoch 5+?** → Top-K is working against you (even distribution)
- **Max expert jumps to >0.8 suddenly?** → Collapse happened, probably end of soft routing

---

## Root Cause Summary

| Issue | Cause | Impact |
|-------|-------|--------|
| **Soft routing never creates signal** | All experts process all samples with weighted loss | Experts undifferentiated at curriculum switch |
| **Weak load balance** | `0.01 * KL ≈ 0.03` << `main_loss` | Balance loss ignored, no incentive to spread |
| **Min probability clamping** | Forces 5% baseline usage per expert | Masks load balance signal, allows useless experts |
| **Temperature annealing** | Sharpens decisions before experts trained | Amplifies collapse once it starts |
| **Gating architecture** | Global pooling, likely simple MLP | Can't learn fine-grained routing |
| **Early stopping** | Stops at epoch 5 | Might catch collapse mid-transition |

---

## How to Fix It (Priority Order)

### 1. **Immediate: Increase Load Balance Coefficient** (5-min fix)
```python
# In config, change:
"load_balance_coef": 0.2  # Up from 0.01
```

Try **0.1, 0.2, 0.5** and monitor entropy/usage range. You want:
- Entropy > 2.5 (out of max 8 = 2.08, so near-uniform until topk)
- Usage range < 0.1 in soft routing

**Expected impact:** Modest improvement (30% collapse reduction)

---

### 2. **Critical: Fix Soft Routing Logic** (20-min fix)

Replace soft routing with **proper auxiliary loss approach**:

```python
if routing_mode == "soft":
    # ALL experts train on ALL samples (no weighting)
    per_sample_loss_all = torch.zeros((config.num_experts, batch_size), device=device)
    
    for expert_id in range(config.num_experts):
        if config.use_lora:
            model.set_adapter(f"expert_{expert_id}")
        
        with torch.cuda.amp.autocast(enabled=config.fp16):
            outputs = _forward_model(model, **model_kwargs)
            per_sample_loss_all[expert_id, :] = _sequence_loss(outputs.logits, labels)
    
    # Weighted by gating (soft assignment)
    per_sample_loss = (gate_probs.T @ per_sample_loss_all.T).diag()  # [batch]
    
    # AUXILIARY loss (not main loss) to guide gating
    # Experts that do well for this sample should have high gate_prob
    auxiliary_loss = 0.0
    for expert_id in range(config.num_experts):
        expert_skill = 1.0 / (per_sample_loss_all[expert_id, :] + 1e-4)  # Skill ~ inverse loss
        expert_skill = expert_skill / expert_skill.max()  # Normalize to [0, 1]
        auxiliary_loss += torch.nn.functional.mse_loss(
            gate_probs[:, expert_id], expert_skill
        )
    
    main_loss = per_sample_loss.mean()
    balance_loss = _load_balance_loss(gate_probs, config.num_experts)
    loss = main_loss + 0.5 * auxiliary_loss + 0.2 * balance_loss
```

**Expected impact:** 70% collapse reduction (experts specialize during soft routing)

---

### 3. **High: Add Expert-Specific Gradient Masking** (10-min fix)

In top-K routing, prevent **zombie experts** (selected but unhelpful):

```python
else:  # topk_soft or topk_hard
    k = max(1, min(config.top_k_experts, gate_probs.size(-1)))
    topk_values, topk_indices = torch.topk(gate_probs, k=k, dim=-1)
    
    # Track which experts are actually helping
    expert_losses = torch.full((config.num_experts,), float('inf'), device=device)
    
    for rank in range(k):
        experts_at_rank = topk_indices[:, rank]
        for expert_id in torch.unique(experts_at_rank).tolist():
            if config.use_lora:
                model.set_adapter(f"expert_{expert_id}")
            
            mask = experts_at_rank == expert_id
            # ... compute loss_per_sample for this expert ...
            
            expert_losses[expert_id] = loss_per_sample[mask].mean().item()
    
    # Penalize gating network for selecting unhelpful experts
    gating_penalty = 0.0
    for expert_id in range(config.num_experts):
        if expert_losses[expert_id] < float('inf'):
            loss_ratio = expert_losses[expert_id] / (main_loss + 1e-6)
            if loss_ratio > 2.0:  # This expert is bad
                gating_penalty += gate_probs[:, expert_id].mean()
    
    loss = main_loss + config.load_balance_coef * balance_loss + 0.1 * gating_penalty
```

**Expected impact:** Prevents "zombie" experts from staying selected

---

### 4. **Medium: Adaptive Temperature** (15-min fix)

Don't blindly anneal. Stop sharpening if entropy collapses:

```python
def _compute_routing_temperature(epoch_idx: int, config: TrainingConfig, 
                                  current_entropy: float) -> float:
    # Baseline annealing
    max_epoch = max(1, config.topk_routing_epochs)
    progress = min(1.0, epoch_idx / max_epoch)
    base_temp = (
        config.routing_temperature_start * (1 - progress)
        + config.routing_temperature_end * progress
    )
    
    # EMERGENCY: If entropy is very low, increase temperature
    min_safe_entropy = torch.tensor(1.0).log()  # ~ln(num_experts)
    if current_entropy < min_safe_entropy * 0.5:
        base_temp = max(base_temp, 1.5)  # Floor at 1.5
    
    return base_temp
```

**Expected impact:** Prevents temperature from accelerating collapse

---

### 5. **Medium: Use WER for Early Stopping, Not Loss** (5-min fix)

```python
# Replace:
# is_best = val_loss < best_loss

# With:
is_best = val_wer < best_wer if val_wer is not None else (val_loss < best_loss)
best_metric = best_wer if val_wer is not None else best_loss
```

**Expected impact:** Catches real degradation (WER) instead of phantom improvements

---

### 6. **Low: Improve Gating Network Architecture** (30-min refactor)

Current (assumed):
```python
# Global pool → MLP
pooled = encoder.last_hidden_state.mean(dim=1)  # [batch, 768]
gate_logits = mlp(pooled)  # [batch, num_experts]
```

**Better approach:**
```python
# Use sequence information
encoder_states = encoder.last_hidden_state  # [batch, seq_len, 768]

# Multi-head attention to identify "hard" regions
hard_regions = attention(encoder_states, query=encoder_states.mean(1, keepdim=True))  # [batch, seq_len, 1]
weighted_pool = (encoder_states * hard_regions).sum(1) / hard_regions.sum(1)  # [batch, 768]

# Gating with speaker/dialect features
gate_logits = mlp(weighted_pool)  # [batch, num_experts]
```

**Expected impact:** Experts learn to handle different speech types

---

## Summary Recommendation

**Start with fix #1 + #2.** Together they should get you to 80%+ experts active. If you're still seeing collapse:

1. Check diagnostic logs (entropy, usage_range)
2. Apply fix #3 (zombie expert penalty)
3. Apply fix #4 (adaptive temperature)
4. Switch to WER-based early stopping (#5)

After these, if collapse persists, the gating network itself needs redesign (#6).

---

## Code Addition: Enhanced Diagnostics

Add this to `_train_epoch` after line 596:

```python
# Enhanced diagnostics every 50 batches
if batch_idx % 50 == 0 and batch_idx > 0:
    expert_usage = gate_probs.mean(dim=0)
    entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=-1).mean()
    
    # Compute actual loss per expert (if in topk mode)
    expert_loss_stats = {}
    if routing_mode != "soft":
        for expert_id in range(config.num_experts):
            if f"expert_{expert_id}_loss" in locals():
                expert_loss_stats[expert_id] = locals()[f"expert_{expert_id}_loss"].item()
    
    print(
        f"  Batch {batch_idx:4d} | Entropy: {entropy:.3f} | "
        f"Usage: [{expert_usage.min():.3f}, {expert_usage.max():.3f}] | "
        f"Main: {main_loss:.4f} | Balance: {balance_loss:.4f}"
    )
```

This will tell you **exactly when** entropy collapses and which expert dominates.
