# Pipeline Optimization Guide for A100/H100 GPUs

This document summarizes all optimizations made to the MoE Speech Translation pipeline for efficient long-running training on high-end GPUs.

## Summary of Changes

### 1. Requirements Updated
Added missing dependencies:
- `umap-learn` - for UMAP dimensionality reduction
- `matplotlib` - for plotting
- `tqdm` - for progress tracking

### 2. ASR Training Optimizations (`asr_training.py`)

**Critical Fix: Per-Expert Subset Routing**
- **Before**: Ran full forward pass through ALL experts for every sample in batch → O(batch × num_experts)
- **After**: Routes samples to their primary/secondary experts only → O(batch × top_k)
- **Speedup**: ~4× faster with 8 experts and top_k=2

**New Features:**
- Gradient accumulation support (`gradient_accumulation_steps`)
- Periodic checkpoint saving (`save_every_n_epochs`)
- Configurable evaluation frequency (`eval_every_n_epochs`)
- Progress bars with tqdm
- Pin memory for faster GPU data transfer

### 3. Expert Pre-Training Optimizations (`expert_pre_training.py`)

**Critical Fix: Use Cached Embeddings**
- **Before**: Re-computed embeddings through Whisper encoder for every sample during expert assignment
- **After**: Uses cached embeddings from the embedding extraction phase
- **Speedup**: ~10-100× faster for expert assignment

**New Features:**
- Gradient accumulation support
- Progress tracking with tqdm
- Pin memory for GPU

### 4. Embedding Extraction Optimizations (`embedding_extraction.py`)

**New Features:**
- Incremental checkpoint saves (every 100 samples by default)
- Resume capability - skips already processed embeddings
- Progress tracking with tqdm
- Periodic CUDA cache clearing

### 5. Pipeline Resume Capability (`train_pipeline.py`)

**Critical Fix: No More Deleting Progress**
- Added `--resume` flag to continue from existing checkpoints
- Each stage checks for completed outputs before re-running
- No more losing days of work on crashes

**New CLI Arguments:**
```bash
--resume                      # Resume from existing checkpoints
--no-plot                     # Skip plotting (faster on headless servers)
--gradient-accumulation-steps # For larger effective batch sizes
--eval-every-n-epochs         # Evaluate less frequently (faster training)
--save-every-n-epochs         # Periodic checkpoint saves
```

---

## Recommended Settings by GPU

### A100 40GB
```bash
python Training_Scripts/train_pipeline.py \
  --mode full \
  --gating-batch-size 512 \
  --expert-batch-size 4 \
  --asr-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --num-workers 4 \
  --eval-every-n-epochs 2 \
  --save-every-n-epochs 5 \
  --fp16
```

### A100 80GB
```bash
python Training_Scripts/train_pipeline.py \
  --mode full \
  --gating-batch-size 1024 \
  --expert-batch-size 8 \
  --asr-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --num-workers 4 \
  --eval-every-n-epochs 2 \
  --save-every-n-epochs 5 \
  --fp16
```

### H100 80GB
```bash
python Training_Scripts/train_pipeline.py \
  --mode full \
  --gating-batch-size 2048 \
  --expert-batch-size 12 \
  --asr-batch-size 12 \
  --gradient-accumulation-steps 4 \
  --num-workers 8 \
  --eval-every-n-epochs 2 \
  --save-every-n-epochs 5 \
  --fp16
```

---

## Estimated Training Times

| Stage | A100 40GB (100% data) | A100 80GB (100% data) | H100 80GB (100% data) |
|-------|----------------------|----------------------|----------------------|
| Embedding Extraction | ~4-5 hours | ~4-5 hours | ~3-4 hours |
| Clustering | ~30 min | ~30 min | ~20 min |
| Gating Pre-training | ~30 min | ~30 min | ~20 min |
| Expert Pre-training (8) | ~30-40 hours | ~20-25 hours | ~15-20 hours |
| Full ASR Training | ~60-80 hours | ~40-50 hours | ~30-40 hours |
| **Total** | **~4-5 days** | **~3-4 days** | **~2-3 days** |

---

## Usage Examples

### Quick Test (5 min)
```bash
python Training_Scripts/train_pipeline.py --mode quick
```

### Full Training (Fresh Start)
```bash
python Training_Scripts/train_pipeline.py --mode full --fp16 --num-workers 4
```

### Resume After Crash
```bash
python Training_Scripts/train_pipeline.py --mode full --resume --fp16 --num-workers 4
```

### Headless Server (No Plotting)
```bash
python Training_Scripts/train_pipeline.py --mode full --no-plot --fp16 --num-workers 4
```

### Maximum Throughput
```bash
python Training_Scripts/train_pipeline.py \
  --mode full \
  --resume \
  --no-plot \
  --fp16 \
  --gating-batch-size 1024 \
  --expert-batch-size 8 \
  --asr-batch-size 8 \
  --gradient-accumulation-steps 8 \
  --num-workers 4 \
  --eval-every-n-epochs 5 \
  --save-every-n-epochs 10
```

---

## Config Files Updated

### `Config/asr_training.json`
- `batch_size`: 4 (up from 2)
- `num_workers`: 4 (up from 0)
- `gradient_accumulation_steps`: 4 (new)
- `eval_every_n_epochs`: 2 (new)
- `save_every_n_epochs`: 5 (new)
- `pin_memory`: true (new)

### `Config/expert_pre_training.json`
- `num_experts`: 8 (aligned with ASR config)
- `batch_size`: 4 (up from 2)
- `num_workers`: 4 (up from 0)
- `gradient_accumulation_steps`: 4 (new)
- `embeddings_dir`: null (set dynamically by pipeline)
- `pin_memory`: true (new)

### `Config/gating_model_config.json`
- `num_experts`: 8 (aligned with other configs)
- `batch_size`: 256 (up from 32)
- `num_workers`: 4 (up from 0)
- `epochs`: 20 (up from 10)
- `weight_decay`: 0.01 (added regularization)

---

## Key Architecture Changes

### Before (Inefficient)
```
For each batch:
  1. Run encoder on batch → get embeddings
  2. Run gating → get expert assignments
  3. For EACH expert (0 to num_experts-1):
     - Run FULL forward pass on ENTIRE batch
     - Weight losses by gating probabilities
```

### After (Optimized)
```
For each batch:
  1. Run encoder ONCE → get embeddings
  2. Run gating → get top-k experts per sample
  3. For each unique expert in top-k:
     - Extract ONLY samples routed to this expert
     - Run forward pass on SUBSET only
     - Weight losses by routing probabilities
```

This change alone provides **4-8× speedup** depending on num_experts and top_k settings.

---

## Troubleshooting

### OOM Errors
1. Reduce `batch_size`
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce `num_workers`

### Slow Data Loading
1. Increase `num_workers` (up to number of CPU cores)
2. Ensure `pin_memory: true`
3. Use SSD storage for dataset

### Resuming Fails
1. Check that checkpoint files exist in expected locations
2. Ensure config files haven't changed between runs
3. Delete partial outputs if corrupted

### Clustering Issues
1. Use `--reduce pca` instead of UMAP for large datasets
2. Increase `--min-cluster-size` if getting too many clusters
3. Use `--clustering-algorithm spectral` for more stable results
