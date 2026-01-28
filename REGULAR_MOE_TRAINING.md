# Regular MoE Training Guide

This guide explains how to use the `--skip-pretraining` flag for standard Mixture of Experts (MoE) training, where the gating network and experts train jointly from scratch without any pretraining stages.

## Overview

The standard training pipeline includes several stages:
1. **Embedding extraction** - Extract Whisper embeddings from audio
2. **Clustering** - Cluster embeddings to determine expert assignments
3. **Gating pretraining** - Pretrain the gating network on cluster labels
4. **Expert pretraining** - Pretrain each expert on its assigned cluster
5. **ASR training** - Joint training of the full MoE model

With `--skip-pretraining`, you can skip steps 1-4 and go directly to step 5, training everything end-to-end from scratch.

## When to Use

Use `--skip-pretraining` when you want to:
- Train a standard MoE model end-to-end without pretraining
- Compare pretrained vs. non-pretrained MoE approaches
- Speed up experimentation (skip time-consuming pretraining stages)
- Train with a fixed number of experts without clustering

## Basic Usage

### Minimal Example

```bash
python Training_Scripts/train_pipeline.py --mode full --seed 42 \
  --skip-pretraining \
  --num-experts 8 \
  --asr-batch-size 2 \
  --gradient-accumulation-steps 8
```

### With Custom Settings

```bash
python Training_Scripts/train_pipeline.py --mode full --seed 42 \
  --skip-pretraining \
  --num-experts 8 \
  --asr-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --fp16 \
  --num-workers 4 \
  --eval-every-n-epochs 2 \
  --save-every-n-epochs 5 \
  --log-file Runs/regular_moe/training.log
```

### Quick Test Run

```bash
python Training_Scripts/train_pipeline.py --mode quick --seed 42 \
  --skip-pretraining \
  --num-experts 4 \
  --asr-batch-size 2 \
  --gradient-accumulation-steps 4
```

## Key Differences

### Standard Pipeline (with pretraining)
- Extracts embeddings from 5% of data
- Clusters embeddings to determine expert structure
- Pretrains gating model on cluster labels
- Pretrains each expert on its assigned cluster
- Trains full ASR model with pretrained components

### Regular MoE Training (`--skip-pretraining`)
- **Skips** embedding extraction
- **Skips** clustering
- **Skips** gating pretraining
- **Skips** expert pretraining
- **Directly** trains full ASR model with randomly initialized components

## Required Parameters

When using `--skip-pretraining`, you **must** specify:
- `--num-experts`: Number of experts to use (e.g., `--num-experts 8`)

This replaces the clustering step that would normally determine the number of experts.

## What Gets Initialized

When skipping pretraining:
- **Gating model**: Randomly initialized (no pretrained checkpoint)
- **Experts**: Randomly initialized LoRA adapters (no pretrained adapters)
- **Base Whisper model**: Loaded from HuggingFace (e.g., `openai/whisper-large-v2`)

All components train jointly from the start on the ASR task.

## Example: Full Training Run

```bash
python Training_Scripts/train_pipeline.py --mode full \
  --skip-pretraining \
  --num-experts 8 \
  --seed 42 \
  --asr-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --fp16 \
  --num-workers 4 \
  --eval-every-n-epochs 2 \
  --save-every-n-epochs 5 \
  --log-file Runs/regular_moe/training.log
```

## Example: With Data Percentage

You can still use `--data-percent` to scale the training data:

```bash
python Training_Scripts/train_pipeline.py --mode full \
  --skip-pretraining \
  --num-experts 8 \
  --data-percent 10 \
  --asr-batch-size 2 \
  --gradient-accumulation-steps 8
```

Note: With `--skip-pretraining`, `--data-percent` only affects the ASR training stage (since embedding/expert pretraining stages are skipped).

## Comparison: Pretrained vs. Regular MoE

| Aspect | Standard Pipeline | Regular MoE (`--skip-pretraining`) |
|--------|-------------------|-----------------------------------|
| Embedding extraction | ✅ Required | ❌ Skipped |
| Clustering | ✅ Required | ❌ Skipped |
| Gating pretraining | ✅ Required | ❌ Skipped |
| Expert pretraining | ✅ Required | ❌ Skipped |
| ASR training | ✅ With pretrained components | ✅ With random initialization |
| Total time | ~4-5 days (A100 40GB) | ~60-80 hours (A100 40GB) |
| Expert structure | Determined by clustering | Fixed by `--num-experts` |

## Tips

1. **Start with fewer experts**: Try `--num-experts 4` or `--num-experts 8` first
2. **Use spectral clustering equivalent**: If you want balanced experts like spectral clustering, just set `--num-experts` to your desired number
3. **Monitor training**: Regular MoE training may need more epochs since components start from scratch
4. **Memory**: Same memory requirements as standard ASR training stage

## Troubleshooting

### Error: "num_experts must be specified"
- **Solution**: Add `--num-experts N` where N is your desired number of experts

### Training seems slower to converge
- **Normal**: Without pretraining, the model needs to learn routing and expert specialization from scratch
- **Solution**: Train for more epochs or adjust learning rate

### Out of memory errors
- **Solution**: Reduce `--asr-batch-size` and increase `--gradient-accumulation-steps` to maintain effective batch size

## See Also

- [Quickstart.md](Quickstart.md) - Full pipeline documentation
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Detailed usage guide
