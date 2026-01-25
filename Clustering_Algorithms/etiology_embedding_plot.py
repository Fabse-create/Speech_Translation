import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Data.clustering import _load_embeddings, _reduce_embeddings
from Data.datapreprocessing import WhisperDataLoader


def _load_mapping(mapping_path: Path, embedding_dir: Path) -> Dict[str, Tuple[Optional[str], str]]:
    """Map embedding id -> (contributor_id, wav_filename) using mapping.json."""
    if not mapping_path.exists():
        return {}
    try:
        with mapping_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, list):
        return {}

    id_map: Dict[str, Tuple[Optional[str], str]] = {}
    for entry in data:
        embedding_path = Path(entry.get("embedding_path", ""))
        contributor_id = entry.get("contributor_id")
        wav_filename = entry.get("wav_filename")
        if not wav_filename:
            continue
        rel_id = None
        if embedding_path:
            try:
                rel_id = embedding_path.relative_to(embedding_dir).with_suffix("").as_posix()
            except ValueError:
                rel_id = None
        if rel_id is None:
            if contributor_id:
                rel_id = f"{contributor_id}/{Path(wav_filename).name}"
            else:
                rel_id = Path(wav_filename).name
        id_map[rel_id] = (contributor_id, Path(wav_filename).name)
    return id_map


def _build_etiology_index(
    dataset_root: Optional[str],
    splits: Iterable[str],
) -> Dict[Tuple[str, str], str]:
    """Build (contributor_id, filename) -> etiology mapping."""
    index: Dict[Tuple[str, str], str] = {}
    for split in splits:
        loader = WhisperDataLoader(
            config_path="Config/dataloader_config.json",
            mode="default",
            config={
                "dataset_root": dataset_root or "Data/extracted_data",
                "split": split,
                "percent": 100,
                "sampling": "random",
                "seed": 42,
                "max_samples": None,
            },
        )
        for sample in loader.build_index():
            contributor_id = sample.get("contributor_id")
            filename = Path(sample.get("id", "")).name
            if contributor_id and filename:
                index[(contributor_id, filename)] = sample.get("etiology", "Unknown")
    return index


def _resolve_etiologies(
    ids: List[str],
    mapping: Dict[str, Tuple[Optional[str], str]],
    etiology_index: Dict[Tuple[str, str], str],
) -> List[str]:
    resolved: List[str] = []
    for item_id in ids:
        contributor_id: Optional[str] = None
        filename: Optional[str] = None

        if item_id in mapping:
            contributor_id, filename = mapping[item_id]
        elif "/" in item_id:
            contributor_id, filename = item_id.split("/", 1)
        else:
            filename = item_id

        if contributor_id and filename:
            etiology = etiology_index.get((contributor_id, Path(filename).name), "Unknown")
        else:
            etiology = "Unknown"
        resolved.append(etiology)
    return resolved


def _plot_by_etiology(
    embeddings_2d: np.ndarray,
    etiologies: List[str],
    output_path: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("matplotlib is required for plotting.") from exc

    labels = np.asarray(etiologies)
    unique = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(9, 7))

    for idx, label in enumerate(unique):
        mask = labels == label
        color = cmap(idx % 20)
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            s=10,
            alpha=0.85,
            label=f"{label} ({mask.sum()})",
            color=color,
        )

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(loc="best", fontsize="small", frameon=True)
    fig.savefig(output_path, bbox_inches="tight")


def _write_csv(
    output_path: Path,
    ids: List[str],
    etiologies: List[str],
    embeddings_2d: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("id,etiology,x,y\n")
        for item_id, etiology, coords in zip(ids, etiologies, embeddings_2d):
            handle.write(f"{item_id},{etiology},{coords[0]},{coords[1]}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Whisper embeddings colored by illness/etiology."
    )
    parser.add_argument(
        "--embedding-dir",
        default="Data/embeddings/whisper_v2_embeddings",
        help="Directory containing .npy embeddings.",
    )
    parser.add_argument(
        "--mapping-path",
        default=None,
        help="Optional mapping.json path (defaults to embedding_dir/mapping.json).",
    )
    parser.add_argument(
        "--dataset-root",
        default="Data/extracted_data",
        help="Root directory for extracted dataset.",
    )
    parser.add_argument(
        "--splits",
        default="Train",
        help="Comma-separated splits to use for etiology lookup (Train,Dev).",
    )
    parser.add_argument(
        "--pooling",
        default="mean",
        choices=["mean", "flatten", "none"],
        help="Pooling to apply to embeddings before reduction.",
    )
    parser.add_argument(
        "--reduce",
        default="pca",
        choices=["none", "pca", "umap"],
        help="Dimensionality reduction (same as clustering).",
    )
    parser.add_argument(
        "--reduce-dim",
        type=int,
        default=50,
        help="Target dimension for reduction.",
    )
    parser.add_argument(
        "--plot-method",
        default="umap",
        choices=["pca", "umap"],
        help="2D projection method for visualization.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="Data/embeddings/whisper_v2_embeddings_etiology",
        help="Output directory for plots and CSV.",
    )
    parser.add_argument(
        "--plot-file",
        default="etiology_plot.png",
        help="Output plot filename.",
    )
    parser.add_argument(
        "--output-csv",
        default="etiology_embeddings.csv",
        help="CSV filename for 2D embedding export.",
    )

    args = parser.parse_args()

    embedding_dir = Path(args.embedding_dir)
    mapping_path = Path(args.mapping_path) if args.mapping_path else (embedding_dir / "mapping.json")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings, ids = _load_embeddings(embedding_dir, args.pooling)
    reduced = _reduce_embeddings(embeddings, args.reduce, args.reduce_dim, args.seed)
    embeddings_2d = _reduce_embeddings(reduced, args.plot_method, 2, args.seed)

    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    mapping = _load_mapping(mapping_path, embedding_dir)
    etiology_index = _build_etiology_index(args.dataset_root, splits)
    etiologies = _resolve_etiologies(ids, mapping, etiology_index)

    plot_path = output_dir / args.plot_file
    title = f"Embeddings by Etiology ({args.plot_method.upper()})"
    _plot_by_etiology(embeddings_2d, etiologies, plot_path, title)

    csv_path = output_dir / args.output_csv
    _write_csv(csv_path, ids, etiologies, embeddings_2d)

    print(f"Saved plot to: {plot_path}")
    print(f"Saved CSV to: {csv_path}")


if __name__ == "__main__":
    main()
