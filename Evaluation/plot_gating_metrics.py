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
        values.append(float(entry[key]))
    return values


def _ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_metrics(metrics_path: Path, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("matplotlib is required for plotting.") from exc

    metrics = _load_metrics(metrics_path)
    epochs = _extract_series(metrics, "epoch")
    train_loss = _extract_series(metrics, "train_loss")
    val_loss = _extract_series(metrics, "val_loss")
    val_acc = _extract_series(metrics, "val_accuracy")

    _ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="train_loss")
    ax.plot(epochs, val_loss, label="val_loss")
    ax.set_title("Gating Model Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig(output_dir / "gating_model_loss.png", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, val_acc, label="val_accuracy")
    ax.set_title("Gating Model Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.savefig(output_dir / "gating_model_accuracy.png", bbox_inches="tight")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot gating model metrics.")
    parser.add_argument(
        "--metrics-path",
        default="Evaluation/gating_model_results/metrics.json",
        help="Path to metrics.json produced by gating pre-training.",
    )
    parser.add_argument(
        "--output-dir",
        default="Evaluation/gating_model_results",
        help="Directory to write PNG plots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_metrics(Path(args.metrics_path), Path(args.output_dir))
