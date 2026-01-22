# Training Scripts Usage

This document covers how to run the training scripts that are implemented so far:

- `Training_Scripts/gating_model_pre_training.py`
- `Training_Scripts/expert_pre_training.py`
- `Training_Scripts/asr_training.py`

## 1) Gating Model Pre-Training

### Without Docker

```
python Training_Scripts/gating_model_pre_training.py
```

Configuration is loaded from `Config/gating_model_config.json`.

### With Docker

Build:

```
docker build -f docker/gating_model/Dockerfile -t st-gating-model .
```

Run:

```
docker run --rm --name st-gating-model -v "%cd%:/app" st-gating-model
```

Stop:

```
docker stop st-gating-model
```

Stop and delete cache:

```
docker stop st-gating-model && docker builder prune -f
```

PowerShell:

```
docker stop st-gating-model; docker builder prune -f
```

## 2) Expert Pre-Training

### Without Docker

```
python Training_Scripts/expert_pre_training.py --config Config/expert_pre_training.json
```

The expert pre-training flow can either:
- Use cluster labels/embeddings from disk, or
- Let Whisper v2 extract embeddings on the fly, then route samples through the frozen gating model.

To enable on-the-fly routing, set in `Config/expert_pre_training.json`:
- `"use_gating_model": true`
- `"use_whisper_embeddings_for_gating": true`
- `"gating_model_checkpoint": "checkpoints/gating_model/best.pt"`

Data selection follows the dataloader settings (percent, sampling, max_samples).
You can override max samples from the CLI:

```
python Training_Scripts/expert_pre_training.py --config Config/expert_pre_training.json --max-samples 50
```

### With Docker

Build:

```
docker build -f docker/expert_pre_training/Dockerfile -t st-expert-pretrain .
```

Run:

```
docker run --rm --name st-expert-pretrain -v "%cd%:/app" st-expert-pretrain
```

Stop:

```
docker stop st-expert-pretrain
```

Stop and delete cache:

```
docker stop st-expert-pretrain && docker builder prune -f
```

PowerShell:

```
docker stop st-expert-pretrain; docker builder prune -f
```

To use a different config:

```
docker run --rm --name st-expert-pretrain -v "%cd%:/app" st-expert-pretrain \
  python Training_Scripts/expert_pre_training.py --config Config/expert_pre_training.json
```

Stop:

```
docker stop st-expert-pretrain
```

Stop and delete cache:

```
docker stop st-expert-pretrain && docker builder prune -f
```

PowerShell:

```
docker stop st-expert-pretrain; docker builder prune -f
```

## 3) Joint ASR Training (Gating + Experts)

### Without Docker

```
python Training_Scripts/asr_training.py --config Config/asr_training.json
```

### With Docker

Build:

```
docker build -f docker/asr_training/Dockerfile -t st-asr-training .
```

Run:

```
docker run --rm --name st-asr-training -v "%cd%:/app" st-asr-training
```

Stop:

```
docker stop st-asr-training
```

Stop and delete cache:

```
docker stop st-asr-training && docker builder prune -f
```

PowerShell:

```
docker stop st-asr-training; docker builder prune -f
```

To use a different config:

```
docker run --rm --name st-asr-training -v "%cd%:/app" st-asr-training \
  python Training_Scripts/asr_training.py --config Config/asr_training.json
```

Stop:

```
docker stop st-asr-training
```

Stop and delete cache:

```
docker stop st-asr-training && docker builder prune -f
```

PowerShell:

```
docker stop st-asr-training; docker builder prune -f
```
