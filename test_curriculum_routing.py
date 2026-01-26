"""
Test script for curriculum routing functions in asr_training.py

This script tests:
1. Routing mode computation (soft -> topk_soft -> topk_hard)
2. Temperature annealing
3. Gating probability computation
4. Load balance loss computation
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from Training_Scripts.asr_training import (
    TrainingConfig,
    _compute_routing_mode,
    _compute_routing_temperature,
    _compute_gating_probabilities,
    _load_balance_loss,
)


def test_routing_mode():
    """Test routing mode transitions."""
    print("=" * 60)
    print("Testing Routing Mode Computation")
    print("=" * 60)
    
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
    
    # Test mode transitions
    expected_modes = [
        (0, "soft"),
        (1, "soft"),
        (2, "soft"),
        (3, "topk_soft"),
        (4, "topk_soft"),
        (5, "topk_soft"),
        (6, "topk_hard"),
        (7, "topk_hard"),
        (8, "topk_hard"),
    ]
    
    print("\nEpoch -> Expected Mode:")
    all_passed = True
    for epoch_idx, expected_mode in expected_modes:
        actual_mode = _compute_routing_mode(epoch_idx, config)
        status = "[PASS]" if actual_mode == expected_mode else "[FAIL]"
        print(f"  Epoch {epoch_idx}: {actual_mode:12} (expected: {expected_mode:12}) {status}")
        if actual_mode != expected_mode:
            all_passed = False
    
    return all_passed


def test_temperature_annealing():
    """Test temperature annealing schedule."""
    print("\n" + "=" * 60)
    print("Testing Temperature Annealing")
    print("=" * 60)
    
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
    
    print("\nEpoch -> Temperature:")
    temps = []
    for epoch_idx in range(9):
        temp = _compute_routing_temperature(epoch_idx, config)
        temps.append(temp)
        print(f"  Epoch {epoch_idx}: {temp:.3f}")
    
    # Check that temperature decreases
    is_decreasing = all(temps[i] >= temps[i + 1] for i in range(len(temps) - 1))
    print(f"\n  Temperature decreasing: {'[PASS]' if is_decreasing else '[FAIL]'}")
    
    # Check start and end values
    start_ok = abs(temps[0] - config.routing_temperature_start) < 0.1
    end_ok = abs(temps[-1] - config.routing_temperature_end) < 0.1
    print(f"  Start temperature correct: {'[PASS]' if start_ok else '[FAIL]'}")
    print(f"  End temperature correct: {'[PASS]' if end_ok else '[FAIL]'}")
    
    return is_decreasing and start_ok and end_ok


def test_gating_probabilities():
    """Test gating probability computation with temperature."""
    print("\n" + "=" * 60)
    print("Testing Gating Probability Computation")
    print("=" * 60)
    
    num_experts = 8
    batch_size = 4
    
    # Create dummy gate logits
    gate_logits = torch.randn(batch_size, num_experts)
    
    # Test with high temperature (softer)
    high_temp = 2.0
    probs_high = _compute_gating_probabilities(gate_logits, high_temp)
    
    # Test with low temperature (sharper)
    low_temp = 0.5
    probs_low = _compute_gating_probabilities(gate_logits, low_temp)
    
    # High temp should have more uniform distribution (higher entropy)
    entropy_high = -(probs_high * torch.log(probs_high + 1e-8)).sum(dim=-1).mean()
    entropy_low = -(probs_low * torch.log(probs_low + 1e-8)).sum(dim=-1).mean()
    
    print(f"\n  High temp ({high_temp}): entropy = {entropy_high:.3f}")
    print(f"  Low temp ({low_temp}): entropy = {entropy_low:.3f}")
    print(f"  High temp has higher entropy: {'[PASS]' if entropy_high > entropy_low else '[FAIL]'}")
    
    # Check probabilities sum to 1
    sum_high = probs_high.sum(dim=-1)
    sum_low = probs_low.sum(dim=-1)
    sums_ok = (torch.allclose(sum_high, torch.ones_like(sum_high)) and
               torch.allclose(sum_low, torch.ones_like(sum_low)))
    print(f"  Probabilities sum to 1: {'[PASS]' if sums_ok else '[FAIL]'}")
    
    return entropy_high > entropy_low and sums_ok


def test_load_balance_loss():
    """Test load balance loss computation."""
    print("\n" + "=" * 60)
    print("Testing Load Balance Loss")
    print("=" * 60)
    
    num_experts = 8
    batch_size = 10
    
    # Uniform distribution (should have low loss)
    uniform_probs = torch.ones(batch_size, num_experts) / num_experts
    uniform_loss = _load_balance_loss(uniform_probs, num_experts)
    
    # Skewed distribution (one expert dominates)
    skewed_probs = torch.zeros(batch_size, num_experts)
    skewed_probs[:, 0] = 1.0  # All samples go to expert 0
    skewed_loss = _load_balance_loss(skewed_probs, num_experts)
    
    print(f"\n  Uniform distribution loss: {uniform_loss:.4f}")
    print(f"  Skewed distribution loss: {skewed_loss:.4f}")
    print(f"  Skewed has higher loss: {'[PASS]' if skewed_loss > uniform_loss else '[FAIL]'}")
    
    # Check that loss is non-negative
    non_neg = uniform_loss >= 0 and skewed_loss >= 0
    print(f"  Loss is non-negative: {'[PASS]' if non_neg else '[FAIL]'}")
    
    return skewed_loss > uniform_loss and non_neg


def test_min_prob_clamping():
    """Test minimum probability clamping."""
    print("\n" + "=" * 60)
    print("Testing Minimum Probability Clamping")
    print("=" * 60)
    
    num_experts = 8
    batch_size = 4
    min_prob = 0.05
    
    # Create logits that would produce very small probabilities without clamping
    gate_logits = torch.tensor([
        [5.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
        [5.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
        [5.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
        [5.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
    ])
    
    # Test without min_prob (baseline)
    probs_no_clamp = _compute_gating_probabilities(gate_logits, temperature=1.0, min_prob=0.0)
    min_no_clamp = probs_no_clamp.min().item()
    
    # Test with min_prob
    probs = _compute_gating_probabilities(gate_logits, temperature=1.0, min_prob=min_prob)
    min_probs = probs.min(dim=-1)[0]
    min_with_clamp = min_probs.min().item()
    
    # After normalization, minimum might be slightly less than min_prob, but should be
    # significantly higher than without clamping
    improved = min_with_clamp > min_no_clamp * 1.5
    reasonable = min_with_clamp >= min_prob * 0.5  # At least half of requested min
    
    print(f"\n  Minimum probability set: {min_prob}")
    print(f"  Min without clamping: {min_no_clamp:.4f}")
    print(f"  Min with clamping: {min_with_clamp:.4f}")
    print(f"  Clamping improves minimum: {'[PASS]' if improved else '[FAIL]'}")
    print(f"  Minimum is reasonable (>= {min_prob * 0.5:.3f}): {'[PASS]' if reasonable else '[FAIL]'}")
    
    # Check still sums to 1
    sums = probs.sum(dim=-1)
    sums_ok = torch.allclose(sums, torch.ones_like(sums))
    print(f"  Still sums to 1 after clamping: {'[PASS]' if sums_ok else '[FAIL]'}")
    
    return improved and reasonable and sums_ok


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Curriculum Routing Function Tests")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("Routing Mode", test_routing_mode()))
    except Exception as e:
        print(f"\n[FAIL] Routing Mode test failed: {e}")
        results.append(("Routing Mode", False))
    
    try:
        results.append(("Temperature Annealing", test_temperature_annealing()))
    except Exception as e:
        print(f"\n[FAIL] Temperature Annealing test failed: {e}")
        results.append(("Temperature Annealing", False))
    
    try:
        results.append(("Gating Probabilities", test_gating_probabilities()))
    except Exception as e:
        print(f"\n[FAIL] Gating Probabilities test failed: {e}")
        results.append(("Gating Probabilities", False))
    
    try:
        results.append(("Load Balance Loss", test_load_balance_loss()))
    except Exception as e:
        print(f"\n[FAIL] Load Balance Loss test failed: {e}")
        results.append(("Load Balance Loss", False))
    
    try:
        results.append(("Min Prob Clamping", test_min_prob_clamping()))
    except Exception as e:
        print(f"\n[FAIL] Min Prob Clamping test failed: {e}")
        results.append(("Min Prob Clamping", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {test_name:30} {status}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
