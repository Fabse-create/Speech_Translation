# Whisper v2 Finetuning Guide

This guide explains how to finetune Whisper Large v2 using a single LoRA adapter with the training pipeline.

## Quick Start

### Basic Finetuning Command

```bash
python Training_Scripts/train_pipeline.py --finetune-only \
  --asr-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --seed 42
```

### With All Recommended Settings (A100 40GB)

```bash
python Training_Scripts/train_pipeline.py --finetune-only \
  --seed 42 \
  --asr-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --num-workers 4 \
  --fp16 \
  --eval-every-n-epochs 2 \
  --save-every-n-epochs 5 \
  --log-file Runs/finetune/training.log
```

## What Does `--finetune-only` Do?

When you use the `--finetune-only` flag, the pipeline:

1. **Skips MoE setup**: No embedding extraction, clustering, gating model pre-training, or expert pre-training
2. **Creates a dummy gating model**: A minimal gating model that always routes to a single expert (required by the ASR training code)
3. **Trains a single LoRA adapter**: Uses 100% of your training data to finetune one LoRA adapter on the base Whisper Large v2 model
4. **Saves checkpoints**: Checkpoints are saved to `checkpoints/asr/` (same location as MoE training)

## Key Parameters

### Required
- `--finetune-only`: Enable finetune-only mode (skips MoE setup)

### Recommended for GPU Memory Management
- `--asr-batch-size 2`: Batch size for training (use 2 on A100 40GB to avoid OOM)
- `--gradient-accumulation-steps 8`: Accumulate gradients over 8 batches (effective batch size = 2 × 8 = 16)

### Optional Optimization
- `--fp16`: Enable mixed precision training (faster, uses less memory)
- `--num-workers 4`: Number of DataLoader workers
- `--eval-every-n-epochs 2`: Run validation every 2 epochs (reduce overhead)
- `--save-every-n-epochs 5`: Save checkpoint every 5 epochs
- `--log-file`: Path to save training logs

### Data Configuration
- `--dataset-root Data/extracted_data`: Root directory containing Train/Dev splits (default)
- `--data-percent 100`: Use 100% of data (default in finetune-only mode)

## Output Locations

- **Checkpoints**: `checkpoints/asr/`
  - `gating_model.pt`: Dummy gating model (always routes to expert 0)
  - `expert_0/`: Trained LoRA adapter
  - `best.json`: Best checkpoint metadata
- **Metrics**: `Evaluation/asr_training_results/metrics.json`
- **Benchmarks**: `Evaluation/asr_benchmark_results/` (if benchmarking is enabled)

## Example: Full Training Run

```bash
python Training_Scripts/train_pipeline.py --finetune-only \
  --seed 42 \
  --asr-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --num-workers 4 \
  --fp16 \
  --eval-every-n-epochs 2 \
  --save-every-n-epochs 5 \
  --log-file Runs/finetune/training.log \
  --benchmark-max-samples 200
```

## Differences from MoE Training

| Feature | MoE Training | Finetune-Only |
|---------|-------------|---------------|
| Experts | Multiple (default: 8) | Single (1) |
| Data Usage | 5% embeddings → 15% experts → 100% ASR | 100% ASR |
| Pre-training | Gating + Experts | None |
| Training Time | ~4-5 days (full pipeline) | ~1-2 days (ASR only) |
| Use Case | Specialized experts for different data types | General-purpose finetuning |

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size further
--asr-batch-size 1 --gradient-accumulation-steps 16
```

### Slow Training
```bash
# Reduce evaluation frequency
--eval-every-n-epochs 5

# Reduce checkpoint frequency  
--save-every-n-epochs 10
```

## Notes

- The finetune-only mode uses the same ASR training code as MoE training, so all features (validation, checkpoints, benchmarking) work the same way
- Validation uses 2.5% of training data (automatically split)
- Benchmarking runs after each checkpoint save (if enabled)
- The dummy gating model is created automatically and always routes to expert 0
