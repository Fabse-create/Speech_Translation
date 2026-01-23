import argparse
import json
from pathlib import Path
from typing import Dict, List


def _load_metrics(metrics_path: Path) -> List[Dict[str, float]]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as metrics_file:
        data = json.load(metrics_file)
    if not isinstance(data, list) or not data:
        raise ValueError("Metrics file must contain a non-empty list.")
    return data


def _extract_series(metrics: List[Dict[str, float]], key: str) -> List[float]:
    values = []
    for entry in metrics:
        if key not in entry:
            raise ValueError(f"Missing '{key}' in metrics entry: {entry}")
        value = entry[key]
        values.append(float(value) if value is not None else float("nan"))
    return values


def _plot_expert(metrics_path: Path, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("matplotlib is required for plotting.") from exc

    metrics = _load_metrics(metrics_path)
    epochs = _extract_series(metrics, "epoch")
    val_loss = _extract_series(metrics, "val_loss")
    val_wer = _extract_series(metrics, "val_wer")

    expert_name = metrics_path.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, val_loss, label="val_loss")
    ax.set_title(f"{expert_name} Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig(output_dir / f"{expert_name}_loss.png", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, val_wer, label="val_wer")
    ax.set_title(f"{expert_name} Validation WER")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("WER")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.savefig(output_dir / f"{expert_name}_wer.png", bbox_inches="tight")


def plot_metrics(metrics_root: Path, output_dir: Path) -> None:
    if metrics_root.is_file():
        _plot_expert(metrics_root, output_dir)
        return
    if not metrics_root.exists():
        raise FileNotFoundError(f"Metrics root not found: {metrics_root}")

    for metrics_path in metrics_root.rglob("metrics.json"):
        _plot_expert(metrics_path, output_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot expert training metrics.")
    parser.add_argument(
        "--metrics-root",
        default="Evaluation/expert_training_results",
        help="Root directory containing expert metrics.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default="Evaluation/expert_training_results",
        help="Directory to write PNG plots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_metrics(Path(args.metrics_root), Path(args.output_dir))
