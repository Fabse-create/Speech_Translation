# Testing Curriculum Routing Functions

This guide explains how to test the newly implemented curriculum routing functions that prevent expert collapse.

## Quick Test: Unit Tests

Run the unit test script to verify all routing functions work correctly:

```bash
python test_curriculum_routing.py
```

This will test:
- ✅ Routing mode transitions (soft → topk_soft → topk_hard)
- ✅ Temperature annealing schedule
- ✅ Gating probability computation with temperature
- ✅ Load balance loss (uniform vs skewed distributions)
- ✅ Minimum probability clamping

Expected output: All tests should pass with ✓ marks.

## Integration Test: Quick Training Run

Test the full training pipeline with a small dataset:

```bash
python Training_Scripts/asr_training.py --config Config/asr_training.json --max-samples 50
```

### What to Look For:

1. **Routing Mode Transitions** (printed at start of each epoch):
   ```
   [Epoch 1] Routing mode: soft, Temperature: 2.00
   [Epoch 2] Routing mode: soft, Temperature: 1.67
   [Epoch 3] Routing mode: soft, Temperature: 1.33
   [Epoch 4] Routing mode: topk_soft, Temperature: 1.00
   [Epoch 5] Routing mode: topk_soft, Temperature: 0.83
   [Epoch 6] Routing mode: topk_soft, Temperature: 0.67
   [Epoch 7] Routing mode: topk_hard, Temperature: 0.50
   ```

2. **Expert Usage Diagnostics** (printed every 50 batches):
   ```
   Batch 0: entropy=2.079, max_expert=0.125, min_expert=0.125, usage_range=0.000
   Batch 50: entropy=1.856, max_expert=0.142, min_expert=0.108, usage_range=0.034
   ```
   
   **Key indicators of healthy expert usage:**
   - **Entropy**: Should start high (~2.0 for 8 experts) and gradually decrease, but not collapse to near 0
   - **Usage range**: Should be small (ideally < 0.1) indicating balanced usage
   - **Min expert**: Should stay above `min_expert_usage_fraction` (default 0.05)

3. **No Expert Collapse**: 
   - All experts should receive some samples throughout training
   - Max expert usage should not exceed ~0.3-0.4 (for 8 experts, uniform would be 0.125)
   - Min expert usage should stay above 0.05

## Full Training Test

For a more realistic test, run with your full dataset:

```bash
python Training_Scripts/asr_training.py --config Config/asr_training.json
```

Monitor the metrics file for expert usage over time:

```bash
# Check metrics after training
cat Evaluation/asr_training_results/metrics.json
```

## Manual Function Testing

You can also test individual functions in a Python REPL:

```python
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from Training_Scripts.asr_training import (
    TrainingConfig,
    _compute_routing_mode,
    _compute_routing_temperature,
    _compute_gating_probabilities,
    _load_balance_loss,
)
import torch

# Create a test config
config = TrainingConfig(
    model_name="openai/whisper-large-v2",
    language=None,
    task=None,
    data_config_path="",
    data_mode="default",
    data_config_override=None,
    num_experts=8,
    top_k_experts=2,
    batch_size=4,
    epochs=9,
    learning_rate=1e-4,
    weight_decay=0.0,
    seed=42,
    num_workers=0,
    val_split=0.1,
    load_balance_coef=0.1,
    gating_model_config="",
    gating_checkpoint=None,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules=["q_proj", "v_proj"],
    experts_dir=None,
    output_dir="",
    fp16=True,
    metrics_dir="",
    gradient_accumulation_steps=1,
    eval_every_n_epochs=1,
    save_every_n_epochs=1,
    pin_memory=True,
    soft_routing_epochs=3,
    topk_routing_epochs=6,
    routing_temperature_start=2.0,
    routing_temperature_end=0.5,
    min_expert_usage_fraction=0.05,
)

# Test routing modes
for epoch in range(9):
    mode = _compute_routing_mode(epoch, config)
    temp = _compute_routing_temperature(epoch, config)
    print(f"Epoch {epoch}: {mode:12} (temp={temp:.2f})")

# Test gating probabilities
gate_logits = torch.randn(4, 8)
probs_high = _compute_gating_probabilities(gate_logits, temperature=2.0)
probs_low = _compute_gating_probabilities(gate_logits, temperature=0.5)
print(f"High temp entropy: {-(probs_high * torch.log(probs_high + 1e-8)).sum(dim=-1).mean():.3f}")
print(f"Low temp entropy: {-(probs_low * torch.log(probs_low + 1e-8)).sum(dim=-1).mean():.3f}")

# Test load balance loss
uniform = torch.ones(10, 8) / 8
skewed = torch.zeros(10, 8)
skewed[:, 0] = 1.0
print(f"Uniform loss: {_load_balance_loss(uniform, 8):.4f}")
print(f"Skewed loss: {_load_balance_loss(skewed, 8):.4f}")
```

## Verifying Expert Collapse Prevention

After training, check that experts are being used:

1. **Check expert usage distribution**: The diagnostic output should show balanced usage
2. **Check metrics**: Loss should decrease while maintaining expert diversity
3. **Visual inspection**: If you have plotting scripts, check that all experts show activity

## Troubleshooting

### All experts collapse to one
- Increase `load_balance_coef` (try 0.1 or 0.2)
- Increase `soft_routing_epochs` (give more time for specialization)
- Check that `min_expert_usage_fraction` is being applied

### Training is too slow
- Reduce `soft_routing_epochs` (soft routing is compute-heavy)
- Reduce `epochs` for testing
- Use `--max-samples` to limit dataset size

### Temperature not changing
- Verify config values are loaded correctly
- Check that `epoch_idx` is being passed correctly to functions

## Expected Behavior

**Healthy training should show:**
- Gradual transition from soft → topk_soft → topk_hard routing
- Temperature decreasing smoothly from start to end
- Expert usage entropy decreasing but not collapsing
- All experts receiving at least 5% of samples (with default settings)
- Load balance loss decreasing over time

**Signs of expert collapse:**
- One expert gets >50% of samples
- Min expert usage drops below 0.01
- Entropy drops to near 0 (<0.5)
- Load balance loss increases or stays high
