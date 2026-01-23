# Training Scripts Usage

This document covers how to run the training scripts that are implemented so far:

- `Training_Scripts/gating_model_pre_training.py`
- `Training_Scripts/expert_pre_training.py`
- `Training_Scripts/asr_training.py`

## 1) Gating Model Pre-Training

### Run

```
python Training_Scripts/gating_model_pre_training.py
```

Configuration is loaded from `Config/gating_model_config.json`.

Metrics (train loss, val loss, val accuracy) are stored at:
`Evaluation/gating_model_results/metrics.json`

## 2) Expert Pre-Training

### Run

```
python Training_Scripts/expert_pre_training.py --config Config/expert_pre_training.json
```

The expert pre-training flow always:
- Extracts Whisper v2 embeddings on the fly
- Routes samples through the frozen gating model

Ensure `Config/expert_pre_training.json` points to a trained gate:
- `"gating_model_checkpoint": "checkpoints/gating_model/best.pt"`

Data selection follows the dataloader settings (percent, sampling, max_samples).
You can override max samples from the CLI:

```
python Training_Scripts/expert_pre_training.py --config Config/expert_pre_training.json --max-samples 50
```

Metrics (train loss, val loss, val WER) are stored at:
`Evaluation/expert_training_results/expert_<id>/metrics.json`

## 3) Joint ASR Training (Gating + Experts)

### Run

```
python Training_Scripts/asr_training.py --config Config/asr_training.json
```

The ASR training loop:
- Samples data via the dataloader (percent/sampling/max_samples)
- Extracts Whisper encoder embeddings
- Routes through the gating model
- Trains LoRA experts jointly with an auxiliary load-balance loss

To quickly test with a limited subset:

```
python Training_Scripts/asr_training.py --config Config/asr_training.json --max-samples 50
```

Metrics (train loss, val loss, val WER) are stored at:
`Evaluation/asr_training_results/metrics.json`
