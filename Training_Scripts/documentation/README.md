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
docker run --rm -v "%cd%:/app" st-gating-model
```

## 2) Expert Pre-Training

### Without Docker

```
python Training_Scripts/expert_pre_training.py --config Config/expert_pre_training.json
```

### With Docker

Build:

```
docker build -f docker/expert_pre_training/Dockerfile -t st-expert-pretrain .
```

Run:

```
docker run --rm -v "%cd%:/app" st-expert-pretrain
```

To use a different config:

```
docker run --rm -v "%cd%:/app" st-expert-pretrain \
  python Training_Scripts/expert_pre_training.py --config Config/expert_pre_training.json
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
docker run --rm -v "%cd%:/app" st-asr-training
```

To use a different config:

```
docker run --rm -v "%cd%:/app" st-asr-training \
  python Training_Scripts/asr_training.py --config Config/asr_training.json
```
