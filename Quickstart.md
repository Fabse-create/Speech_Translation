# Quickstart

This repo supports both an end-to-end training pipeline and running each step
individually. The pipeline enforces stratified sampling by illness (`etiology`)
and uses reproducible seeds throughout.

---

## End-to-end Pipeline

### Quick smoke test (10/5/5 samples)

```bash
python Training_Scripts/train_pipeline.py --mode quick --seed 42
```

### Full run (5% embeddings → 15% expert pretrain → 100% ASR)

```bash
python Training_Scripts/train_pipeline.py --mode full --seed 42 \
  --asr-batch-size 2 \
  --gradient-accumulation-steps 8
```

### Regular MoE training (skip pretraining, train end-to-end)

For regular MoE training where gating and experts train jointly from scratch:

```bash
python Training_Scripts/train_pipeline.py --mode full --seed 42 \
  --skip-pretraining \
  --num-experts 8 \
  --asr-batch-size 2 \
  --gradient-accumulation-steps 8
```

This will:
- Skip embedding extraction (not needed for end-to-end training)
- Skip clustering (not needed - uses `--num-experts` directly)
- Skip gating model pretraining
- Skip expert pretraining  
- Start ASR training directly with randomly initialized gating and experts

**Note:** When using `--skip-pretraining`, you must specify `--num-experts` to define the number of experts. The clustering step is bypassed entirely.

### Test with 10% of data (with file logging)

Useful for verifying the full pipeline before committing to a long training run:

```bash
python Training_Scripts/train_pipeline.py --mode full --seed 42 \
  --data-percent 10 \
  --asr-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --log-file Runs/full/training.log
```

The `--data-percent` flag **scales proportionally** while maintaining the stage ratios (5%/15%/100%):
- `--data-percent 10` → embeddings: 0.5%, experts: 1.5%, ASR: 10%
- `--data-percent 50` → embeddings: 2.5%, experts: 7.5%, ASR: 50%
- `--data-percent 100` → embeddings: 5%, experts: 15%, ASR: 100% (default full run)

This ensures the relative data distribution between stages stays consistent.

### Test with Spectral Clustering (recommended for balanced experts)

HDBSCAN can produce imbalanced clusters (most samples in one cluster). Spectral clustering 
forces exactly N balanced clusters, which is often better for MoE training:

```bash
python Training_Scripts/train_pipeline.py --mode full --seed 42 \
  --clustering-algorithm spectral \
  --num-experts 8 \
  --data-percent 10 \
  --asr-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --log-file Runs/full/training.log
```

For a fuller test with 50% of data:

```bash
python Training_Scripts/train_pipeline.py --mode full --seed 42 \
  --clustering-algorithm spectral \
  --num-experts 8 \
  --data-percent 50 \
  --asr-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --log-file Runs/full/training.log
```

Defaults:
- `num_experts=8`
- Clustering uses HDBSCAN with dimensionality reduction and a plot.
- Outputs are written under `Runs/<mode>/` (e.g., `Runs/full/` or `Runs/quick/`).

---

## Resume & Recovery

### Resume after crash or interruption

The pipeline now supports resuming from existing checkpoints. This is **critical** 
for long-running full training jobs that may take days:

```bash
# First run
python Training_Scripts/train_pipeline.py --mode full --seed 42 \
  --asr-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --log-file Runs/full/training.log

# If it crashes, resume with:
python Training_Scripts/train_pipeline.py --mode full --seed 42 \
  --asr-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --log-file Runs/full/training.log \
  --resume
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
python Training_Scripts/train_pipeline.py --mode full --no-plot \
  --asr-batch-size 2 \
  --gradient-accumulation-steps 8
```

---

## Recommended Settings by GPU

> **Note:** All commands use reduced ASR batch size (2) with increased gradient 
> accumulation (8) to prevent OOM errors. This gives the same effective batch size 
> while using less peak memory.

### A100 40GB (Recommended)

```bash
python Training_Scripts/train_pipeline.py --mode full \
  --resume \
  --fp16 \
  --gating-batch-size 512 \
  --expert-batch-size 4 \
  --asr-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --num-workers 4 \
  --eval-every-n-epochs 2 \
  --save-every-n-epochs 5 \
  --log-file Runs/full/training.log
```

### A100 80GB

```bash
python Training_Scripts/train_pipeline.py --mode full \
  --resume \
  --fp16 \
  --gating-batch-size 1024 \
  --expert-batch-size 8 \
  --asr-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --num-workers 4 \
  --eval-every-n-epochs 2 \
  --save-every-n-epochs 5 \
  --log-file Runs/full/training.log
```

### H100 80GB

```bash
python Training_Scripts/train_pipeline.py --mode full \
  --resume \
  --fp16 \
  --gating-batch-size 2048 \
  --expert-batch-size 12 \
  --asr-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --num-workers 8 \
  --eval-every-n-epochs 2 \
  --save-every-n-epochs 5 \
  --log-file Runs/full/training.log
```

### Production run with all optimizations

```bash
python Training_Scripts/train_pipeline.py --mode full \
  --resume \
  --no-plot \
  --fp16 \
  --gating-batch-size 512 \
  --expert-batch-size 4 \
  --asr-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --num-workers 4 \
  --eval-every-n-epochs 5 \
  --save-every-n-epochs 10 \
  --log-file Runs/full/training.log
```

---

## Parameter Reference

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--mode` | str | `full` | Pipeline mode: `quick` (smoke test with 10/5/5 samples) or `full` (production training) |
| `--seed` | int | `42` | Random seed for reproducibility across all stages |
| `--resume` | flag | `false` | Resume from existing checkpoints instead of starting fresh |
| `--no-plot` | flag | `false` | Skip generating plots (useful for headless servers) |
| `--skip-pretraining` | flag | `false` | Skip gating and expert pretraining, start ASR training directly (regular MoE training mode) |

### Data Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data-percent` | float | `None` | Scale all data percentages proportionally. Example: `10` gives 0.5% embeddings, 1.5% experts, 10% ASR |
| `--dataset-root` | path | `Data/extracted_data` | Root directory containing Train/Dev splits |
| `--whisper-model` | str | `v2` | Whisper model version for embedding extraction: `v2` or `v3` |

### Model Architecture

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--num-experts` | int | `8` | Number of MoE experts to create |

### Batch Sizes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--gating-batch-size` | int | `None` | Batch size for gating model pre-training (can be large, embeddings are small) |
| `--expert-batch-size` | int | `None` | Batch size for expert pre-training (LoRA fine-tuning) |
| `--asr-batch-size` | int | `None` | Batch size for full ASR training. **Use 2 on A100 40GB to avoid OOM** |
| `--num-workers` | int | `None` | Number of DataLoader worker processes |

### Training Optimization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--gradient-accumulation-steps` | int | `1` | Accumulate gradients over N batches before optimizer step. **Use 8 with batch-size 2 for effective batch of 16** |
| `--fp16` | flag | auto | Enable FP16 mixed precision training (recommended for speed) |
| `--no-fp16` | flag | `false` | Disable FP16 even if GPU supports it |
| `--eval-every-n-epochs` | int | `1` | Run validation every N epochs (increase to reduce overhead) |
| `--save-every-n-epochs` | int | `5` | Save checkpoint every N epochs (in addition to best model) |

### Clustering Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--clustering-algorithm` | str | `hdbscan` | Clustering algorithm: `hdbscan` (density-based, variable clusters) or `spectral` (fixed N clusters, more balanced) |
| `--min-cluster-size` | int | `5` | HDBSCAN minimum cluster size (lower = more clusters) |
| `--min-clusters` | int | `8` | Minimum number of clusters required (HDBSCAN only) |
| `--min-samples` | int | `None` | HDBSCAN min_samples (defaults to min_cluster_size) |
| `--metric` | str | `euclidean` | HDBSCAN distance metric (e.g., `euclidean`, `cosine`) |
| `--allow-single-cluster` | flag | `false` | HDBSCAN: allow single cluster instead of all noise |
| `--hdbscan-algorithm` | str | `best` | HDBSCAN backend: `best`, `generic`, `prims_kdtree`, `prims_balltree`, `boruvka_kdtree`, `boruvka_balltree` |
| `--no-reduce-experts` | flag | `false` | Fail if clustering produces fewer experts than requested |
| `--min-experts` | int | `2` | Minimum experts if reduction is allowed |
| `--max-retries` | int | `5` | Retries for HDBSCAN with decreasing min_cluster_size |

### Dimensionality Reduction & Visualization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--reduce` | str | `pca` | Dimensionality reduction before clustering: `none`, `pca`, or `umap` |
| `--reduce-dim` | int | `50` | Target dimensions for reduction |
| `--pooling` | str | `mean` | Pooling for clustering embeddings: `mean`, `flatten`, or `none` |
| `--plot-method` | str | `umap` | Visualization method for cluster plots: `pca` or `umap` |

### Logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--log-file` | path | `None` | Path to save training logs. Essential for long runs on servers |

---

## Understanding Batch Size & Memory

### Why reduced batch size?

The full ASR training loads:
- Base Whisper Large v2 model (~3GB)
- 8 LoRA expert adapters
- Gating network
- Optimizer states

On A100 40GB, this leaves limited memory for activations during forward/backward pass.

### Effective batch size calculation

```
Effective Batch = batch_size × gradient_accumulation_steps
```

Examples:
- `--asr-batch-size 4 --gradient-accumulation-steps 4` → Effective: 16
- `--asr-batch-size 2 --gradient-accumulation-steps 8` → Effective: 16 (same, but lower peak memory)
- `--asr-batch-size 1 --gradient-accumulation-steps 16` → Effective: 16 (minimum memory)

### Memory-safe defaults for A100 40GB

```bash
--asr-batch-size 2 --gradient-accumulation-steps 8
```

---

## Common Options

### Clustering options

```bash
# Use UMAP for dimensionality reduction (default is PCA)
python Training_Scripts/train_pipeline.py --mode full \
  --reduce umap --reduce-dim 50 \
  --asr-batch-size 2 --gradient-accumulation-steps 8

# Use spectral clustering instead of HDBSCAN (recommended for balanced clusters)
python Training_Scripts/train_pipeline.py --mode full \
  --clustering-algorithm spectral --num-experts 8 \
  --asr-batch-size 2 --gradient-accumulation-steps 8

# Fail if HDBSCAN yields fewer clusters than requested
python Training_Scripts/train_pipeline.py --mode full \
  --no-reduce-experts \
  --asr-batch-size 2 --gradient-accumulation-steps 8
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

1. **GPU memory cleanup** between stages: Prevents OOM when transitioning from 
   expert pre-training to full ASR training.

2. **Per-expert subset routing** (ASR training): Only processes samples routed 
   to each expert instead of running all experts on the full batch → ~4-8× speedup

3. **Cached embedding reuse** (Expert pre-training): Uses embeddings from the 
   extraction phase instead of recomputing → ~10-100× speedup for expert assignment

4. **Incremental checkpointing** (Embedding extraction): Saves progress every 
   100 samples so crashes don't lose all work

5. **Early stopping**: All training stages monitor validation loss and stop if 
   no improvement for 5 epochs (prevents wasted compute and overfitting)

6. **Periodic model checkpointing** (ASR training): Saves checkpoints at 
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

## Estimated Training Times

| Stage | A100 40GB | A100 80GB | H100 80GB |
|-------|-----------|-----------|-----------|
| Embedding Extraction | ~4-5 hours | ~4-5 hours | ~3-4 hours |
| Clustering | ~30 min | ~30 min | ~20 min |
| Gating Pre-training | ~30 min | ~30 min | ~20 min |
| Expert Pre-training (8) | ~30-40 hours | ~20-25 hours | ~15-20 hours |
| Full ASR Training | ~60-80 hours | ~40-50 hours | ~30-40 hours |
| **Total** | **~4-5 days** | **~3-4 days** | **~2-3 days** |

> Note: Early stopping may significantly reduce these times if the model converges early.

---

## Notes on Reproducibility

- All steps accept a seed in their configs or CLI.
- The pipeline sets deterministic PyTorch settings and always uses the same seed
  for sampling, clustering, and training.
- Outputs are written to `Runs/<mode>/` directories (e.g., `Runs/full/`, `Runs/quick/`).

---

## Troubleshooting

### CUDA Out of Memory

If you see OOM errors during ASR training:

```bash
# Reduce batch size further
--asr-batch-size 1 --gradient-accumulation-steps 16

# Or for expert training
--expert-batch-size 2 --gradient-accumulation-steps 8
```

### Training stuck or slow

```bash
# Reduce evaluation frequency
--eval-every-n-epochs 5

# Reduce checkpoint frequency
--save-every-n-epochs 10
```

### Missing embeddings during expert training

The pipeline uses a mapping file to find embeddings. If you see warnings about missing embeddings:
1. Ensure embedding extraction completed successfully
2. Check that `mapping.json` exists in the embeddings directory
3. Re-run with `--resume` to skip completed stages
