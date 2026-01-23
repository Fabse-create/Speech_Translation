# Running the Pipeline with Spectral Clustering

This guide explains how to run the full MoE training pipeline using **Spectral Clustering** instead of HDBSCAN.

## Key Differences from HDBSCAN

| Feature | HDBSCAN | Spectral Clustering |
|---------|---------|---------------------|
| Cluster count | Discovered automatically | Fixed via `--num-experts` |
| Soft labels | Native probability estimates | Computed via centroid cosine similarity + softmax |
| Noise handling | Can label points as noise (-1) | All points assigned to a cluster |
| Best for | Unknown cluster structure | Known number of experts |

## Are Soft Labels Used?

**Yes!** The pipeline uses soft labels from spectral clustering:

1. After fitting, `get_soft_clusters()` computes centroids for each cluster
2. Cosine similarity is calculated between each embedding and all centroids
3. Softmax is applied to produce probability distributions
4. Saved to `Spectral_soft.json` and used for gating model training

This means the gating model learns from soft cluster assignments, not just hard labels.

---

## Basic Command

```bash
python Training_Scripts/train_pipeline.py --mode full \
  --clustering-algorithm spectral \
  --num-experts 8
```

## Full Command with All Options

```bash
python Training_Scripts/train_pipeline.py --mode full \
  --clustering-algorithm spectral \
  --num-experts 8 \
  --seed 42 \
  --whisper-model v2 \
  --pooling mean \
  --reduce pca \
  --reduce-dim 50 \
  --plot-method umap \
  --fp16 \
  --gating-batch-size 512 \
  --expert-batch-size 4 \
  --asr-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --num-workers 4 \
  --eval-every-n-epochs 1 \
  --save-every-n-epochs 1 \
  --log-file Runs/full/training.log
```

## Resume from Checkpoint

```bash
python Training_Scripts/train_pipeline.py --mode full \
  --clustering-algorithm spectral \
  --num-experts 8 \
  --resume \
  --fp16 \
  --gating-batch-size 512 \
  --expert-batch-size 4 \
  --asr-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --num-workers 4 \
  --log-file Runs/full/training.log
```

## Quick Test Run

```bash
python Training_Scripts/train_pipeline.py --mode quick \
  --clustering-algorithm spectral \
  --num-experts 8 \
  --fp16
```

---

## Pipeline Arguments Reference

### Clustering Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--clustering-algorithm` | `hdbscan` | Use `spectral` for spectral clustering |
| `--num-experts` | `8` | Number of clusters/experts (directly used by spectral) |
| `--pooling` | `mean` | How to pool frame-level embeddings (`mean`, `flatten`, `none`) |
| `--reduce` | `pca` | Dimensionality reduction before clustering (`none`, `pca`, `umap`) |
| `--reduce-dim` | `50` | Target dimension for reduction |
| `--plot-method` | `umap` | Visualization method for cluster plots (`pca`, `umap`) |

### Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `full` | `quick` for smoke test, `full` for real training |
| `--seed` | `42` | Random seed for reproducibility |
| `--fp16` | auto | Force mixed precision (auto-enabled with CUDA) |
| `--no-fp16` | - | Disable mixed precision |
| `--whisper-model` | `v2` | Whisper model for embeddings (`v2`, `v3`) |

### Batch Size & Workers

| Argument | Default | Description |
|----------|---------|-------------|
| `--gating-batch-size` | config | Override gating model batch size |
| `--expert-batch-size` | config | Override expert pre-training batch size |
| `--asr-batch-size` | config | Override ASR training batch size |
| `--num-workers` | config | DataLoader workers |
| `--gradient-accumulation-steps` | `1` | Gradient accumulation for larger effective batches |

### Checkpointing

| Argument | Default | Description |
|----------|---------|-------------|
| `--resume` | - | Resume from existing checkpoints |
| `--eval-every-n-epochs` | `1` | Evaluation frequency |
| `--save-every-n-epochs` | `1` | Checkpoint save frequency |
| `--log-file` | - | Path to log file |
| `--no-plot` | - | Skip plotting (faster, headless servers) |

### Data Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset-root` | `Data/extracted_data` | Root directory for dataset |
| `--data-percent` | - | Scale data percentage for all stages |

---

## HDBSCAN-Only Options (Ignored for Spectral)

These options only affect HDBSCAN and are ignored when using spectral clustering:

- `--min-clusters`
- `--min-cluster-size`
- `--min-samples`
- `--metric`
- `--allow-single-cluster`
- `--hdbscan-algorithm`
- `--no-reduce-experts`
- `--min-experts`
- `--max-retries`

---

## Standalone Clustering (Outside Pipeline)

You can also run clustering separately:

```bash
python -m Data.clustering \
  --embedding-dir Runs/full/embeddings/whisper_v2_embeddings \
  --algorithm spectral \
  --output-dir Runs/full/clustered \
  --pooling mean \
  --reduce pca \
  --reduce-dim 50 \
  --plot umap
```

---

## Output Files

After running with spectral clustering, you'll find:

```
Runs/full/
├── embeddings/whisper_v2_embeddings/
│   ├── mapping.json
│   └── *.npy (embeddings)
├── clustered/
│   ├── Spectral.json          # Hard labels
│   ├── Spectral_soft.json     # Soft labels (used for gating)
│   └── spectral_umap.png      # Cluster visualization
├── gating_model/
│   ├── best.pt
│   └── gating_pretrain_config.json
├── asr/
│   ├── best.json
│   ├── gating_model.pt
│   └── expert_*/
└── training.log
```

---

## Example: Memory-Constrained Training

For GPUs with limited VRAM:

```bash
python Training_Scripts/train_pipeline.py --mode full \
  --clustering-algorithm spectral \
  --num-experts 8 \
  --fp16 \
  --gating-batch-size 256 \
  --expert-batch-size 2 \
  --asr-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --num-workers 2 \
  --resume \
  --log-file Runs/full/training.log
```

## Example: Fast Iteration (Reduced Data)

For quick experiments with less data:

```bash
python Training_Scripts/train_pipeline.py --mode full \
  --clustering-algorithm spectral \
  --num-experts 8 \
  --data-percent 10 \
  --fp16 \
  --no-plot \
  --log-file Runs/full/training.log
```
