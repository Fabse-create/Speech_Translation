# Curriculum Routing Implementation Analysis

## Summary of Changes

We implemented a **curriculum routing** system to prevent expert collapse in MoE training:

### 1. **Routing Curriculum Schedule**
- **Phase 1 (soft_routing_epochs)**: Soft routing - all experts receive gradients
- **Phase 2 (topk_routing_epochs)**: Top-k soft routing - specialization with safety net
- **Phase 3 (epochs 6+)**: Top-k hard routing - final optimization

### 2. **Temperature Annealing**
- Starts at `routing_temperature_start` (default: 2.0) - softer distribution
- Ends at `routing_temperature_end` (default: 0.5) - sharper distribution
- Gradually decreases over training to allow specialization

### 3. **Improved Load Balance Loss**
- Changed from entropy-based to KL divergence: `D_KL(uniform || importance)`
- More effective at penalizing skewed expert usage
- Default coefficient increased from 0.01 to 0.05

### 4. **Minimum Expert Usage Protection**
- `min_expert_usage_fraction` (default: 0.05) prevents expert death
- Applied via probability clamping before normalization

### 5. **Diagnostic Output**
- Added expert usage statistics every 50 batches
- Shows entropy, max/min expert usage, and usage range

## Can Soft Routing Handle Missing Expert Parameters?

### ✅ **YES - Soft Routing Works with Missing Pre-trained Experts**

Here's why:

### Expert Adapter Creation Flow

1. **`_build_moe_model()` creates ALL adapters** (lines 216-234):
   ```python
   model = get_peft_model(model, lora_config, adapter_name="expert_0")
   for expert_id in range(1, config.num_experts):
       model.add_adapter(f"expert_{expert_id}", lora_config)
   ```
   - All adapters are created with **random initialization**
   - Adapter structure exists for all experts (0 to num_experts-1)

2. **`_load_expert_adapters()` only loads if exists** (lines 244-251):
   ```python
   for expert_id in range(num_experts):
       adapter_dir = experts_dir / f"expert_{expert_id}"
       if not adapter_dir.exists():
           continue  # Skip if no pre-trained weights
       model.load_adapter(adapter_dir, adapter_name=f"expert_{expert_id}")
   ```
   - **Optional**: Only loads pre-trained weights if they exist
   - If missing, adapter keeps random initialization from step 1

3. **Soft routing uses all adapters** (lines 395-416):
   ```python
   if routing_mode == "soft":
       for expert_id in range(config.num_experts):
           if config.use_lora:
               model.set_adapter(f"expert_{expert_id}")  # Works even without pre-trained weights
   ```
   - `set_adapter()` works on any adapter created by `add_adapter()`
   - Doesn't require pre-trained weights to exist

### Why This Works

1. **PEFT (LoRA) Library Behavior**:
   - `add_adapter()` creates adapter structure with random weights
   - `load_adapter()` is optional - only overwrites weights if called
   - `set_adapter()` works on any existing adapter regardless of weight source

2. **Soft Routing Design**:
   - Purpose is to let ALL experts learn from the start
   - Random initialization is acceptable - experts will learn during training
   - The curriculum approach specifically allows experts to specialize gradually

3. **Safety Checks**:
   - Low-weight experts are skipped: `if expert_weight.mean() < 1e-4: continue`
   - This prevents wasting compute on experts with negligible contribution

### Potential Edge Cases

1. **All experts missing pre-trained weights**:
   - ✅ **Works fine**: All experts start from random initialization
   - This is actually the intended behavior for soft routing phase

2. **Some experts missing pre-trained weights**:
   - ✅ **Works fine**: Mixed initialization (some pre-trained, some random)
   - Pre-trained experts may perform better initially, but random ones will catch up

3. **Expert adapter not created**:
   - ❌ **Would fail**: But `_build_moe_model()` creates all adapters, so this shouldn't happen

### Recommendations

1. **For best results**: Pre-train all experts before ASR training
   - Ensures all experts start with reasonable initialization
   - Reduces training time needed for convergence

2. **If some experts missing**: Still works, but:
   - Consider increasing `soft_routing_epochs` to give random experts more time
   - Monitor expert usage diagnostics to ensure all experts are learning

3. **If all experts missing**: Works, but:
   - Training will take longer (experts learning from scratch)
   - Consider reducing `soft_routing_epochs` if compute is limited
   - The curriculum routing will still prevent collapse

## Code Flow Verification

```
ASR Training Start:
├── _build_moe_model()
│   └── Creates adapters: expert_0, expert_1, ..., expert_N (random init)
├── _load_expert_adapters() [if experts_dir specified]
│   └── Loads pre-trained weights for existing adapters only
│       └── Missing adapters keep random initialization
└── Training Loop:
    └── Soft Routing Phase:
        └── For each expert_id in range(num_experts):
            ├── model.set_adapter(f"expert_{expert_id}") ✅ Works
            ├── Forward pass with expert
            └── Backward pass updates expert weights
```

## Conclusion

**Soft routing CAN handle missing pre-trained expert parameters.** The implementation is robust because:

1. All adapters are created upfront with random initialization
2. Pre-trained weights are optional (loaded only if they exist)
3. `set_adapter()` works on any existing adapter
4. The curriculum design allows experts to learn from scratch

The only requirement is that `_build_moe_model()` successfully creates all adapters, which it does by design.
