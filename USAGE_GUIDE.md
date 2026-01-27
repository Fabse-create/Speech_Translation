# Comprehensive Usage Guide

This guide explains how to use the project end-to-end or step-by-step. It covers
setup, data requirements, the training pipeline, standalone scripts, parameter
meanings, and benchmarking.

## Table of Contents
- Project layout and prerequisites
- Data layout and sampling
- End-to-end training pipeline
- Running each stage separately
- Configuration files
- ASR training behavior and anti-collapse logic
- WER benchmarking (top-1 and top-k mixture)
- Outputs and where artifacts are stored
- Troubleshooting and tips

## Project layout and prerequisites

### Python environment
- Python 3.10+ recommended
- GPU recommended for all training stages

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional dependencies used in training/evaluation:

```bash
pip install transformers peft torchaudio
```

### Key directories (high level)
- `Training_Scripts/` : training scripts + pipeline
- `Data/` : dataset loading, embedding extraction, clustering utilities
- `Models/` : gating model and Whisper wrappers
- `Evaluation/` : metrics, WER, plots
- `WER_Benchmark/` : benchmarking script
- `Config/` : JSON configs used across stages
- `checkpoints/` : model artifacts
- `Runs/` : pipeline run artifacts (embeddings/cluster outputs)

## Data layout and sampling

The pipeline expects a dataset at `Data/extracted_data/` with this structure:

```
Data/extracted_data/
  Train/
    <speaker_id>/
      <speaker_id>.json
      <audio files...>
  Dev/
    <speaker_id>/
      <speaker_id>.json
      <audio files...>
```

Each `<speaker_id>.json` file should contain:
- `Etiology` (string)
- `Files` list, where each item has:
  - `Filename`
  - `Prompt.Transcript`

Sampling happens via `WhisperDataLoader` and supports:
- `sampling: random` or `stratified`
- `percent` of dataset used
- `max_samples` cap
- `seed` for reproducibility

## End-to-end training pipeline

The pipeline runs:
1) Embedding extraction
2) Clustering
3) Gating pre-training
4) Expert pre-training
5) Full ASR training
6) Post-run WER benchmarks (top-1 + top-k mixture)

### Quick smoke test

```bash
python Training_Scripts/train_pipeline.py --mode quick --seed 42
```

### Full run example

```bash
python Training_Scripts/train_pipeline.py --mode full --seed 42 \
  --asr-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --log-file Runs/full/training.log
```

### Resume after crash

```bash
python Training_Scripts/train_pipeline.py --mode full --seed 42 --resume
```

### Key pipeline parameters and effects

**Core**
- `--mode {quick,full}`: quick uses tiny sample sizes for smoke tests.
- `--seed`: global seed for reproducibility across stages.
- `--resume`: skip completed stages if outputs exist.
- `--no-plot`: skip metric plots for headless servers.

**Data selection**
- `--dataset-root`: dataset root directory (default `Data/extracted_data`).
- `--data-percent`: scales percent for all stages proportionally.

**Clustering**
- `--clustering-algorithm {hdbscan,spectral}`:
  - `hdbscan` is density-based and may yield fewer clusters.
  - `spectral` forces exactly `num_experts` clusters (often more balanced).
- `--num-experts`: target number of experts.
- `--min-clusters`: min clusters for HDBSCAN.
- `--min-cluster-size`: minimum cluster size (lower => more clusters).
- `--min-samples`: HDBSCAN min_samples (if unset, uses min_cluster_size).
- `--metric`: distance metric for HDBSCAN (e.g., euclidean, cosine).
- `--allow-single-cluster`: allow single cluster instead of all noise.
- `--no-reduce-experts`: fail if HDBSCAN yields fewer clusters.
- `--min-experts`: minimum number of experts if reduction is allowed.
- `--max-retries`: HDBSCAN retries with smaller min_cluster_size.

**Batch sizes**
- `--gating-batch-size`: gating pre-train batch size (embeddings only).
- `--expert-batch-size`: LoRA expert pre-train batch size.
- `--asr-batch-size`: full ASR batch size (use small values on limited GPUs).
- `--num-workers`: DataLoader workers.

**Training optimization**
- `--fp16` / `--no-fp16`: enable/disable mixed precision.
- `--gradient-accumulation-steps`: increase effective batch size.
- `--eval-every-n-epochs`: less frequent eval for faster training.
- `--save-every-n-epochs`: periodic checkpoints.

**Benchmarking**
- `--benchmark-split {Train,Dev}`: benchmark split.
- `--benchmark-percent`: percent of split used for benchmarks.
- `--benchmark-max-samples`: default 200 samples for comparability.
- `--benchmark-batch-size`: benchmark batch size.
- `--benchmark-top-k`: top-k mixture decoding size.
- `--benchmark-seed`: fixed seed for benchmark sample manifest.

The pipeline writes benchmarks to:

```
Evaluation/asr_benchmark_results/
  benchmark_<run_id>_top1.json
  benchmark_<run_id>_topkK.json
  gating_<run_id>_top1.jsonl
  gating_<run_id>_topkK.jsonl
  benchmarks_index.jsonl
  benchmark_samples_<dataset>_...json   # sample manifest
```

Delete the manifest file if you want to regenerate samples.

## Running each stage separately

### 1) Embedding extraction

```bash
python -m Data.embedding_extraction
```

### 2) Clustering

```bash
python -m Data.clustering --algorithm spectral --pooling mean --reduce pca --reduce-dim 50
```

### 3) Gating model pre-training

```bash
python Training_Scripts/gating_model_pre_training.py
```

### 4) Expert pre-training

```bash
python Training_Scripts/expert_pre_training.py --config Config/expert_pre_training.json
```

### 5) Joint ASR training

```bash
python Training_Scripts/asr_training.py --config Config/asr_training.json
```

### 6) WER benchmarking

Top-1 (default):

```bash
python WER_Benchmark/wer_benchmark.py \
  --models finetuned \
  --output Evaluation/asr_benchmark_results/benchmark_top1.json
```

Top-k mixture (weighted logit decoding):

```bash
python WER_Benchmark/wer_benchmark.py \
  --models finetuned \
  --moe-top-k 2 \
  --moe-mixture \
  --output Evaluation/asr_benchmark_results/benchmark_topk.json \
  --gating-output Evaluation/asr_benchmark_results/gating_topk.jsonl
```

## Configuration files

### `Config/dataloader_config.json`
Controls dataset root, split, sampling method, and percent for data loading.

### `Config/gating_model_config.json`
Defines gating MLP shape and pre-training parameters. Key fields:
- `input_dim`, `hidden_dim`, `num_experts`
- `training.*` fields: batch size, epochs, learning rate, embeddings and labels path

### `Config/expert_pre_training.json`
Defines expert pre-training behavior. Key fields:
- `num_experts`
- `gating_model_checkpoint`
- `data_config_override` (sampling)
- `fp16`, `batch_size`, `gradient_accumulation_steps`

### `Config/asr_training.json`
Defines joint ASR training. Common fields:
- `num_experts`, `top_k_experts`, `batch_size`, `epochs`, `learning_rate`
- `load_balance_coef`: higher keeps routing more uniform
- `soft_routing_epochs`, `topk_routing_epochs`: routing curriculum
- `routing_temperature_start/end/min`: controls routing sharpness
- `router_noise_std`: adds exploration during routing
- `routing_alignment_coef`: aligns gate probs to expert losses (anti-collapse)
- `routing_entropy_coef`: encourages higher entropy early
- `min_expert_usage_fraction`: clamp in soft routing to prevent dead experts
- `save_every_epoch`, `save_every_n_epochs`: checkpoint cadence
- `rebase_output_dir`: archive previous checkpoints on new runs

## ASR training behavior and anti-collapse logic

During joint ASR training:
- **Soft routing** (early epochs): all experts are evaluated; router learns to
  favor experts that minimize loss. This avoids undifferentiated experts.
- **Top-k routing** (middle epochs): only the k best experts contribute to loss.
- **Hard routing** (late epochs): top-k selection is fixed, no mixing.

Anti-collapse mechanisms:
- **Load-balance loss**: nudges usage toward uniform distribution.
- **Routing alignment**: aligns gate probs with inverse expert loss.
- **Entropy bonus**: keeps router exploratory early.
- **Router noise**: prevents early deterministic collapse.
- **Temperature floor**: stops routing from getting too sharp too early.

These choices keep experts learning useful specializations before routing
hardens.

## Outputs and artifacts

### Checkpoints
- `checkpoints/gating_model/`: gating checkpoints
- `checkpoints/experts/`: expert LoRA adapters
- `checkpoints/asr/`: joint ASR checkpoints, `best.json`, and run archive

### Metrics
- `Evaluation/gating_model_results/metrics.json`
- `Evaluation/expert_training_results/`
- `Evaluation/asr_training_results/metrics.json`
- `Evaluation/asr_benchmark_results/benchmarks_index.jsonl`

## Troubleshooting and tips

**CUDA OOM**
- Reduce `--asr-batch-size` and increase `--gradient-accumulation-steps`.

**HDBSCAN yields too few clusters**
- Use `--clustering-algorithm spectral`, or allow expert reduction.

**Routing collapse**
- Increase `load_balance_coef`.
- Keep `routing_temperature_min` >= 1.0 during early training.
- Enable `router_noise_std` and `routing_entropy_coef`.

**Benchmark comparability**
- Keep `--benchmark-seed` fixed.
- Do not delete the benchmark sample manifest unless you want a new sample set.

---

If you want this guide to live somewhere else (e.g., replace root `README.md`),
say the word and I will reorganize it.
