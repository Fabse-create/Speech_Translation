# Plot ASR Training Metrics

This script reads `metrics.json` from ASR joint training and generates PNG plots
for training/validation loss and validation WER.

## Run

```
python Evaluation/plot_asr_metrics.py
```

Outputs (default):
- `Evaluation/asr_training_results/asr_loss.png`
- `Evaluation/asr_training_results/asr_wer.png`

## Options

- `--metrics-path` (default: `Evaluation/asr_training_results/metrics.json`)
- `--output-dir` (default: `Evaluation/asr_training_results`)

Example:

```
python Evaluation/plot_asr_metrics.py --metrics-path Evaluation/asr_training_results/metrics.json --output-dir Evaluation/asr_training_results
```
