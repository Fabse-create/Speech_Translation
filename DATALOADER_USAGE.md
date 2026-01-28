# DataLoader Usage Guide: Impairment Exclusion

This guide explains how to use the `WhisperDataLoader` with impairment exclusion functionality, both standalone and as part of the training pipeline.

## Overview

The `WhisperDataLoader` now supports excluding samples from a specific impairment/etiology during data loading. This is useful for:
- Training models without certain impairment types
- Testing model generalization
- Creating ablation studies

## Available Impairments

Common impairments in the dataset include:
- `ALS`
- `Parkinson's Disease`
- `Cerebral Palsy`
- `Stroke`
- `Down Syndrome`

Note: The exact impairment name must match exactly as it appears in the dataset metadata (case-sensitive).

## Using the DataLoader Standalone

### Basic Usage

```python
from Data.datapreprocessing import WhisperDataLoader

# Create a loader that excludes ALS samples
loader = WhisperDataLoader(
    config_path="Config/dataloader_config.json",
    mode="default",
    config={
        "dataset_root": "Data/extracted_data",
        "split": "Train",
        "percent": 100,
        "sampling": "stratified",
        "seed": 42,
        "exclude_impairment": "ALS"  # Exclude ALS samples
    }
)

# Get samples (ALS samples will be filtered out)
samples = loader.sample()

print(f"Loaded {len(samples)} samples (excluding ALS)")
```

### Using Configuration File

You can also specify `exclude_impairment` in the configuration file:

```json
{
  "dataset_root": "Data/extracted_data",
  "split": "Train",
  "percent": 100,
  "sampling": "stratified",
  "seed": 42,
  "exclude_impairment": "ALS",
  "modes": {
    "default": {},
    "no_als": {
      "exclude_impairment": "ALS"
    }
  }
}
```

Then use it with:

```python
loader = WhisperDataLoader(
    config_path="Config/dataloader_config.json",
    mode="no_als"
)
samples = loader.sample()
```

### Filtering Logic

The exclusion happens **before** sampling (random or stratified). This means:
1. All samples are loaded from metadata files
2. Samples matching the excluded impairment are filtered out
3. The remaining samples are sampled according to the specified strategy (random/stratified)
4. Percent and max_samples limits are applied to the filtered dataset

## Using with Training Pipeline

### Full Training Pipeline

To exclude an impairment during the full training pipeline:

```bash
python Training_Scripts/train_pipeline.py \
    --mode full \
    --exclude-impairment "ALS" \
    --num-experts 8 \
    --seed 42
```

This will:
- Exclude ALS samples during embedding extraction
- Exclude ALS samples during expert pre-training
- Exclude ALS samples during ASR training

### Standalone ASR Training

To exclude an impairment when running ASR training directly:

```bash
python Training_Scripts/asr_training.py \
    --config Config/asr_training.json \
    --exclude-impairment "ALS"
```

### Combined with Other Options

You can combine impairment exclusion with other options:

```bash
python Training_Scripts/train_pipeline.py \
    --mode full \
    --exclude-impairment "Parkinson's Disease" \
    --data-percent 50 \
    --num-experts 8 \
    --seed 42 \
    --batch-size 4
```

## Examples

### Example 1: Exclude ALS for Quick Test

```bash
python Training_Scripts/train_pipeline.py \
    --mode quick \
    --exclude-impairment "ALS" \
    --num-experts 4
```

### Example 2: Exclude Cerebral Palsy with Custom Data Percentage

```bash
python Training_Scripts/train_pipeline.py \
    --mode full \
    --exclude-impairment "Cerebral Palsy" \
    --data-percent 25 \
    --num-experts 8
```

### Example 3: Standalone ASR Training Without Stroke Samples

```bash
python Training_Scripts/asr_training.py \
    --config Config/asr_training.json \
    --exclude-impairment "Stroke" \
    --max-samples 1000
```

## Important Notes

1. **Case Sensitivity**: The impairment name must match exactly (case-sensitive). Check your dataset metadata for exact names.

2. **Filtering Order**: Exclusion happens before sampling, so stratified sampling will maintain proportions among the remaining impairments.

3. **Empty Results**: If excluding an impairment results in no samples, the loader will return an empty list and training will fail with an appropriate error message.

4. **Logging**: When using the training pipeline, excluded impairments are logged at the start of the pipeline run.

5. **Consistency**: When using the full pipeline, the same impairment exclusion is applied consistently across all stages (embedding extraction, expert training, ASR training).

## Troubleshooting

### No samples found after exclusion
- Check that the impairment name matches exactly (including capitalization and punctuation)
- Verify that other impairments exist in your dataset
- Try without exclusion to confirm the dataset is accessible

### Impairment name not recognized
- Check the dataset metadata files (`{contributor_id}.json`) to see exact impairment names
- Look at `WER_Benchmark/results.json` for examples of impairment names used in the dataset

### Verification

To verify exclusion is working, you can check the samples:

```python
from Data.datapreprocessing import WhisperDataLoader

loader = WhisperDataLoader(
    config_path="Config/dataloader_config.json",
    mode="default",
    config={"exclude_impairment": "ALS"}
)

samples = loader.sample()
etiologies = [s.get("etiology") for s in samples]
print(f"Unique etiologies: {set(etiologies)}")
# Should not contain "ALS"
```
