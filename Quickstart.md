# Quickstart

This repo supports both an end-to-end training pipeline and running each step
individually. The pipeline enforces stratified sampling by illness (`etiology`)
and uses reproducible seeds throughout.

## End-to-end pipeline

### Quick smoke test (10/5/5 samples)

```
python Training_Scripts/train_pipeline.py --mode quick --seed 42
```

### Full run (10% embeddings → 15% expert pretrain → 100% ASR)

```
python Training_Scripts/train_pipeline.py --mode full --seed 42
```

Defaults:
- `num_experts=8`
- Clustering uses HDBSCAN with dimensionality reduction and a plot.
- Outputs are written under `Runs/<mode>/`.

Common options:
```
python Training_Scripts/train_pipeline.py --mode full --seed 123 --reduce umap --reduce-dim 50
```

Use spectral clustering (experimental):
```
python Training_Scripts/train_pipeline.py --mode full --clustering-algorithm spectral --num-experts 8
```

Fail if HDBSCAN yields fewer clusters than requested:
```
python Training_Scripts/train_pipeline.py --mode full --no-reduce-experts
```

HDBSCAN fallback behavior:
- The pipeline retries HDBSCAN with a decreasing `min_cluster_size`.
- If HDBSCAN still yields zero clusters, it falls back to Spectral
  (only when expert reduction is allowed).
- Quick mode enables expert reduction by default to avoid failing on tiny sample sizes.

GPU/memory controls (safe defaults for A100/H100, override if needed):
```
python Training_Scripts/train_pipeline.py --mode full --fp16 --expert-batch-size 2 --asr-batch-size 2 --gating-batch-size 16
```

Embedding size estimate (from the quick run):
- Each `.npy` embedding is ~7.3 MiB (about 1500 x 1280 float32).
- Clustering loads all embeddings into RAM, so 10k embeddings ≈ 73 GiB RAM.

Conservative run command based on that estimate:
```
python Training_Scripts/train_pipeline.py --mode full --fp16 --expert-batch-size 2 --asr-batch-size 2 --gating-batch-size 16 --num-workers 4
```

## Run each step individually

### 1) Embedding extraction (pretraining only)

Uses stratified sampling by `etiology` in the dataloader.

```
python -m Data.embedding_extraction
```

Example (10% of Train, stratified):
```
python -c "import json, tempfile; cfg={'data_config_path':'Config/dataloader_config.json','data_mode':'default','data_config_override':{'dataset_root':'Data/extracted_data','split':'Train','percent':10,'sampling':'stratified','seed':42,'max_samples':None,'modes':{}},'whisper_model':'v2','output_dir':'Data/embeddings/whisper_v2_embeddings','mapping_path':'Data/embeddings/whisper_v2_embeddings/mapping.json','overwrite':True}; f=tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False); json.dump(cfg,f); f.close(); from Data.embedding_extraction import extract_embeddings; print(extract_embeddings(f.name))"
```

### 2) Clustering + visualization

```
python -m Data.clustering --algorithm spectral --pooling mean --reduce pca --reduce-dim 50 --plot umap
```

### 3) Gating network pre-training

```
python Training_Scripts/gating_model_pre_training.py
```

Plot metrics:
```
python Evaluation/plot_gating_metrics.py
```

### 4) Expert pre-training (fresh embeddings + frozen gate)

```
python Training_Scripts/expert_pre_training.py --config Config/expert_pre_training.json
```

Plot metrics:
```
python Evaluation/plot_expert_metrics.py
```

### 5) Joint ASR training

```
python Training_Scripts/asr_training.py --config Config/asr_training.json
```

Plot metrics:
```
python Evaluation/plot_asr_metrics.py
```

## Notes on reproducibility

- All steps accept a seed in their configs or CLI.
- The pipeline sets deterministic PyTorch settings and always uses the same seed
  for sampling, clustering, and training.
- Outputs are written to separate `Runs/<mode>/` directories to avoid
  accidentally resuming from older checkpoints.
