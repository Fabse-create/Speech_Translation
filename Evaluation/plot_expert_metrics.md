# Plot Expert Training Metrics

This script reads per-expert `metrics.json` files (written during expert
pre-training) and generates PNG plots for validation loss and WER.

## Run

```
python Evaluation/plot_expert_metrics.py
```

Outputs (default):
- `Evaluation/expert_training_results/expert_<id>_loss.png`
- `Evaluation/expert_training_results/expert_<id>_wer.png`

## Options

- `--metrics-root` (default: `checkpoints/experts`)
- `--output-dir` (default: `Evaluation/expert_training_results`)

Example:

```
python Evaluation/plot_expert_metrics.py --metrics-root checkpoints/experts --output-dir Evaluation/expert_training_results
```
