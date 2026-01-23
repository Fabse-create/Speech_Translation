# Plot Gating Model Metrics

This script reads `metrics.json` from gating model pre-training and generates
PNG plots for loss and validation accuracy.

## Run

```
python Evaluation/plot_gating_metrics.py
```

Outputs (default):
- `Evaluation/gating_model_results/gating_model_loss.png`
- `Evaluation/gating_model_results/gating_model_accuracy.png`

## Options

- `--metrics-path` (default: `Evaluation/gating_model_results/metrics.json`)
- `--output-dir` (default: `Evaluation/gating_model_results`)

Example:

```
python Evaluation/plot_gating_metrics.py --metrics-path Evaluation/gating_model_results/metrics.json --output-dir Evaluation/gating_model_results
```
