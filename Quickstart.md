# Quickstart

This repo supports both an end-to-end training pipeline and running each step
individually. The pipeline enforces stratified sampling by illness (`etiology`)
and uses reproducible seeds throughout.

## End-to-end Pipeline

### Quick smoke test (10/5/5 samples)

```bash
python Training_Scripts/train_pipeline.py --mode quick --seed 42
```

### Full run (10% embeddings → 15% expert pretrain → 100% ASR)

```bash
python Training_Scripts/train_pipeline.py --mode full --seed 42
```

Defaults:
- `num_experts=8`
- Clustering uses HDBSCAN with dimensionality reduction and a plot.
- Outputs are written under `Runs/<mode>/`.

---

## New Features: Resume & Optimization

### Resume after crash or interruption

The pipeline now supports resuming from existing checkpoints. This is **critical** 
for long-running full training jobs that may take days:

```bash
# First run
python Training_Scripts/train_pipeline.py --mode full --seed 42

# If it crashes, resume with:
python Training_Scripts/train_pipeline.py --mode full --seed 42 --resume
```

The `--resume` flag will:
- Skip embedding extraction if embeddings already exist
- Skip clustering if cluster labels already exist
- Skip gating pre-training if checkpoint exists
- Skip expert pre-training if expert directories exist
- Skip ASR training if best checkpoint exists

### Skip plotting (headless servers)

For faster runs or servers without display:

```bash
python Training_Scripts/train_pipeline.py --mode full --no-plot
```

### Gradient accumulation for larger effective batch sizes

If you're memory-constrained but want larger effective batch sizes:

```bash
python Training_Scripts/train_pipeline.py --mode full \
  --expert-batch-size 4 \
  --asr-batch-size 4 \
  --gradient-accumulation-steps 4
# Effective batch size = 4 × 4 = 16
```

### Reduce evaluation overhead

Evaluation (especially WER) is expensive. For faster training:

```bash
python Training_Scripts/train_pipeline.py --mode full \
  --eval-every-n-epochs 5 \
  --save-every-n-epochs 10
```

---

## Recommended Settings by GPU

### A100 40GB

```bash
python Training_Scripts/train_pipeline.py --mode full \
  --fp16 \
  --gating-batch-size 512 \
  --expert-batch-size 4 \
  --asr-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --num-workers 4 \
  --eval-every-n-epochs 2 \
  --save-every-n-epochs 5
```

### A100 80GB

```bash
python Training_Scripts/train_pipeline.py --mode full \
  --fp16 \
  --gating-batch-size 1024 \
  --expert-batch-size 8 \
  --asr-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --num-workers 4 \
  --eval-every-n-epochs 2 \
  --save-every-n-epochs 5
```

### H100 80GB

```bash
python Training_Scripts/train_pipeline.py --mode full \
  --fp16 \
  --gating-batch-size 2048 \
  --expert-batch-size 12 \
  --asr-batch-size 12 \
  --gradient-accumulation-steps 4 \
  --num-workers 8 \
  --eval-every-n-epochs 2 \
  --save-every-n-epochs 5
```

### Production run with all optimizations

```bash
python Training_Scripts/train_pipeline.py --mode full \
  --resume \
  --no-plot \
  --fp16 \
  --gating-batch-size 1024 \
  --expert-batch-size 8 \
  --asr-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --num-workers 4 \
  --eval-every-n-epochs 5 \
  --save-every-n-epochs 10
```

---

## Common Options

### Clustering options

```bash
# Use UMAP for dimensionality reduction (default is PCA)
python Training_Scripts/train_pipeline.py --mode full --reduce umap --reduce-dim 50

# Use spectral clustering instead of HDBSCAN
python Training_Scripts/train_pipeline.py --mode full --clustering-algorithm spectral --num-experts 8

# Fail if HDBSCAN yields fewer clusters than requested
python Training_Scripts/train_pipeline.py --mode full --no-reduce-experts
```

### HDBSCAN fallback behavior

- The pipeline retries HDBSCAN with a decreasing `min_cluster_size`.
- If HDBSCAN still yields zero clusters, it falls back to Spectral
  (only when expert reduction is allowed).
- Quick mode enables expert reduction by default to avoid failing on tiny sample sizes.

---

## Memory & Performance Notes

### Embedding size estimate

- Each `.npy` embedding is ~7.3 MiB (about 1500 × 1280 float32).
- Clustering loads all embeddings into RAM, so 10k embeddings ≈ 73 GiB RAM.

### Key optimizations implemented

1. **Per-expert subset routing** (ASR training): Only processes samples routed 
   to each expert instead of running all experts on the full batch → ~4-8× speedup

2. **Cached embedding reuse** (Expert pre-training): Uses embeddings from the 
   extraction phase instead of recomputing → ~10-100× speedup for expert assignment

3. **Incremental checkpointing** (Embedding extraction): Saves progress every 
   100 samples so crashes don't lose all work

4. **Periodic model checkpointing** (ASR training): Saves checkpoints at 
   configurable intervals, not just on best loss

---

## Run Each Step Individually

### 1) Embedding extraction

Uses stratified sampling by `etiology` in the dataloader.

```bash
python -m Data.embedding_extraction
```

Example (10% of Train, stratified):
```bash
python -c "import json, tempfile; cfg={'data_config_path':'Config/dataloader_config.json','data_mode':'default','data_config_override':{'dataset_root':'Data/extracted_data','split':'Train','percent':10,'sampling':'stratified','seed':42,'max_samples':None,'modes':{}},'whisper_model':'v2','output_dir':'Data/embeddings/whisper_v2_embeddings','mapping_path':'Data/embeddings/whisper_v2_embeddings/mapping.json','overwrite':True}; f=tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False); json.dump(cfg,f); f.close(); from Data.embedding_extraction import extract_embeddings; print(extract_embeddings(f.name))"
```

### 2) Clustering + visualization

```bash
python -m Data.clustering --algorithm spectral --pooling mean --reduce pca --reduce-dim 50 --plot umap
```

### 3) Gating network pre-training

```bash
python Training_Scripts/gating_model_pre_training.py
```

Plot metrics:
```bash
python Evaluation/plot_gating_metrics.py
```

### 4) Expert pre-training

```bash
python Training_Scripts/expert_pre_training.py --config Config/expert_pre_training.json
```

Plot metrics:
```bash
python Evaluation/plot_expert_metrics.py
```

### 5) Joint ASR training

```bash
python Training_Scripts/asr_training.py --config Config/asr_training.json
```

Plot metrics:
```bash
python Evaluation/plot_asr_metrics.py
```

---

## CLI Reference

### Pipeline Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `full` | `quick` for smoke test, `full` for production |
| `--seed` | `42` | Random seed for reproducibility |
| `--num-experts` | `8` | Number of MoE experts |
| `--resume` | `false` | Resume from existing checkpoints |
| `--no-plot` | `false` | Skip plotting (faster on headless servers) |

### Batch Size & Workers

| Argument | Default | Description |
|----------|---------|-------------|
| `--gating-batch-size` | `256` | Batch size for gating model training |
| `--expert-batch-size` | `4` | Batch size for expert pre-training |
| `--asr-batch-size` | `4` | Batch size for ASR training |
| `--num-workers` | `4` | DataLoader worker processes |
| `--fp16` | auto | Enable FP16 mixed precision |

### Training Optimization

| Argument | Default | Description |
|----------|---------|-------------|
| `--gradient-accumulation-steps` | `1` | Accumulate gradients over N batches |
| `--eval-every-n-epochs` | `1` | Evaluate validation set every N epochs |
| `--save-every-n-epochs` | `5` | Save checkpoint every N epochs |

### Clustering

| Argument | Default | Description |
|----------|---------|-------------|
| `--clustering-algorithm` | `hdbscan` | `hdbscan` or `spectral` |
| `--reduce` | `pca` | Dimensionality reduction: `none`, `pca`, `umap` |
| `--reduce-dim` | `50` | Target dimension for reduction |
| `--plot-method` | `umap` | Visualization method for clusters |
| `--min-cluster-size` | `5` | HDBSCAN minimum cluster size |

---

## Estimated Training Times

| Stage | A100 40GB | A100 80GB | H100 80GB |
|-------|-----------|-----------|-----------|
| Embedding Extraction | ~4-5 hours | ~4-5 hours | ~3-4 hours |
| Clustering | ~30 min | ~30 min | ~20 min |
| Gating Pre-training | ~30 min | ~30 min | ~20 min |
| Expert Pre-training (8) | ~30-40 hours | ~20-25 hours | ~15-20 hours |
| Full ASR Training | ~60-80 hours | ~40-50 hours | ~30-40 hours |
| **Total** | **~4-5 days** | **~3-4 days** | **~2-3 days** |

---

## Notes on Reproducibility

- All steps accept a seed in their configs or CLI.
- The pipeline sets deterministic PyTorch settings and always uses the same seed
  for sampling, clustering, and training.
- Outputs are written to separate `Runs/<mode>/` directories to avoid
  accidentally resuming from older checkpoints (unless `--resume` is used).
