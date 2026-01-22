import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

from Clustering_Algorithms.hdbscan import HDBSCAN
from Clustering_Algorithms.spectral_clustering import Spectral_Clustering


def _pool_embedding(embedding: np.ndarray, pooling: str) -> np.ndarray:
    data = np.asarray(embedding)
    if data.ndim == 1:
        return data
    if pooling == "none":
        raise ValueError("pooling='none' requires 1D embeddings.")
    if pooling == "flatten":
        return data.reshape(-1)
    if pooling == "mean":
        while data.ndim > 1:
            data = data.mean(axis=0)
        return data
    raise ValueError("pooling must be 'mean', 'flatten', or 'none'.")


def _load_embeddings(embedding_dir: Path, pooling: str) -> Tuple[np.ndarray, List[str]]:
    embedding_files = sorted(embedding_dir.rglob("*.npy"))
    if not embedding_files:
        raise FileNotFoundError(f"No .npy embeddings found in {embedding_dir}")

    embeddings = []
    ids = []
    for embedding_path in embedding_files:
        embedding = np.load(embedding_path)
        embeddings.append(_pool_embedding(embedding, pooling))
        wav_id = embedding_path.relative_to(embedding_dir).with_suffix("")
        ids.append(wav_id.as_posix())

    try:
        return np.vstack(embeddings), ids
    except ValueError as exc:
        raise ValueError("Embeddings have incompatible shapes; consider pooling='mean'.") from exc


def _reduce_embeddings(
    embeddings: np.ndarray,
    method: str,
    n_components: int,
    random_state: int,
) -> np.ndarray:
    if method == "none":
        return embeddings
    if method == "pca":
        try:
            from sklearn.decomposition import PCA
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("scikit-learn is required for PCA reduction.") from exc
        return PCA(n_components=n_components, random_state=random_state).fit_transform(embeddings)
    if method == "umap":
        try:
            import umap
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("umap-learn is required for UMAP reduction.") from exc
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(embeddings)
    raise ValueError("reduce must be 'none', 'pca', or 'umap'.")


def _plot_embeddings(
    embeddings_2d: np.ndarray,
    labels: Iterable[int],
    output_path: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("matplotlib is required for plotting.") from exc

    fig, ax = plt.subplots(figsize=(8, 6))
    labels_array = np.asarray(list(labels))
    noise_mask = labels_array == -1
    cluster_mask = ~noise_mask

    if cluster_mask.any():
        scatter = ax.scatter(
            embeddings_2d[cluster_mask, 0],
            embeddings_2d[cluster_mask, 1],
            c=labels_array[cluster_mask],
            s=8,
            cmap="tab20",
            alpha=0.9,
        )
        fig.colorbar(scatter, ax=ax, shrink=0.8)
    if noise_mask.any():
        ax.scatter(
            embeddings_2d[noise_mask, 0],
            embeddings_2d[noise_mask, 1],
            c="lightgrey",
            s=8,
            alpha=0.6,
            label="noise",
        )
        ax.legend(loc="best", fontsize="small")
    ax.set_title(title)
    fig.savefig(output_path, bbox_inches="tight")


def _save_hard_labels(
    ids: List[str],
    labels: Iterable[int],
    output_dir: Path,
    filename: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    payload = [{"id": item_id, "label": int(label)} for item_id, label in zip(ids, labels)]
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    return output_path


def _save_soft_probs(
    ids: List[str],
    probs: np.ndarray,
    output_dir: Path,
    filename: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    payload = [{"id": item_id, "probs": row.tolist()} for item_id, row in zip(ids, probs)]
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    return output_path


def _cluster_hdbscan(
    embeddings: np.ndarray,
    ids: List[str],
    output_dir: Path,
    min_cluster_size: int,
    min_samples: Optional[int],
    metric: str,
    allow_single_cluster: bool,
    hdbscan_algorithm: str,
) -> np.ndarray:
    algo = hdbscan_algorithm
    if metric == "cosine" and algo == "best":
        algo = "generic"
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        algorithm=algo,
        allow_single_cluster=allow_single_cluster,
    )
    labels = model.fit(embeddings)
    _save_hard_labels(ids, labels, output_dir, "HDBSCAN.json")
    soft_probs = model.get_soft_clusters(include_noise=True)
    _save_soft_probs(ids, soft_probs, output_dir, "HDBSCAN_soft.json")
    return labels


def _cluster_spectral(embeddings: np.ndarray, ids: List[str], output_dir: Path) -> np.ndarray:
    model = Spectral_Clustering()
    labels = model.fit(embeddings)
    _save_hard_labels(ids, labels, output_dir, "Spectral.json")
    soft_probs = model.get_soft_clusters(embeddings)
    _save_soft_probs(ids, soft_probs, output_dir, "Spectral_soft.json")
    return labels


def run(
    embedding_dir: Path = Path("Data/embeddings/whisper_v2_embeddings"),
    algorithm: str = "hdbscan",
    output_dir: Path = Path("Data/embeddings/whisper_v2_embeddings_clustered"),
    pooling: str = "mean",
    reduce: str = "none",
    reduce_dim: int = 50,
    plot: str = "none",
    plot_path: Optional[Path] = None,
    random_state: int = 42,
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None,
    metric: str = "euclidean",
    allow_single_cluster: bool = False,
    hdbscan_algorithm: str = "best",
) -> None:
    embeddings, ids = _load_embeddings(embedding_dir, pooling)
    reduced = _reduce_embeddings(embeddings, reduce, reduce_dim, random_state)
    output_dir.mkdir(parents=True, exist_ok=True)

    algo = algorithm.lower()
    if algo == "hdbscan":
        labels = _cluster_hdbscan(
            reduced,
            ids,
            output_dir,
            min_cluster_size,
            min_samples,
            metric,
            allow_single_cluster,
            hdbscan_algorithm,
        )
    elif algo in {"spectral", "spectral_clustering"}:
        labels = _cluster_spectral(reduced, ids, output_dir)
    else:
        raise ValueError("algorithm must be 'hdbscan' or 'spectral'.")

    plot_method = plot.lower()
    if plot_method != "none":
        plot_embeddings = _reduce_embeddings(reduced, plot_method, 2, random_state)
        target_path = plot_path or (output_dir / f"{algo}_{plot_method}.png")
        _plot_embeddings(plot_embeddings, labels, target_path, f"{algorithm.upper()} clusters ({plot_method})")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster Whisper embeddings.")
    parser.add_argument(
        "--embedding-dir",
        default="Data/embeddings/whisper_v2_embeddings",
        help="Directory with .npy embeddings.",
    )
    parser.add_argument(
        "--algorithm",
        default="hdbscan",
        choices=["hdbscan", "spectral"],
        help="Clustering algorithm to use.",
    )
    parser.add_argument(
        "--output-dir",
        default="Data/embeddings/whisper_v2_embeddings_clustered",
        help="Output directory for clustered JSON files.",
    )
    parser.add_argument(
        "--pooling",
        default="mean",
        choices=["mean", "flatten", "none"],
        help="How to pool frame-level embeddings into a single vector.",
    )
    parser.add_argument(
        "--reduce",
        default="none",
        choices=["none", "pca", "umap"],
        help="Dimensionality reduction before clustering.",
    )
    parser.add_argument(
        "--reduce-dim",
        type=int,
        default=50,
        help="Target dimension when using --reduce.",
    )
    parser.add_argument(
        "--plot",
        default="none",
        choices=["none", "pca", "umap"],
        help="Create a 2D plot of clusters using PCA or UMAP.",
    )
    parser.add_argument(
        "--plot-path",
        default=None,
        help="Optional path for saving the plot image.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reduction/plotting.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="HDBSCAN: minimum cluster size.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="HDBSCAN: min_samples (defaults to min_cluster_size if unset).",
    )
    parser.add_argument(
        "--metric",
        default="euclidean",
        help="HDBSCAN: distance metric (e.g., euclidean, cosine).",
    )
    parser.add_argument(
        "--allow-single-cluster",
        action="store_true",
        help="HDBSCAN: allow a single cluster instead of all noise.",
    )
    parser.add_argument(
        "--hdbscan-algorithm",
        default="best",
        choices=["best", "generic", "prims_kdtree", "prims_balltree", "boruvka_kdtree", "boruvka_balltree"],
        help="HDBSCAN: backend algorithm (generic supports cosine).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        Path(args.embedding_dir),
        args.algorithm,
        Path(args.output_dir),
        args.pooling,
        args.reduce,
        args.reduce_dim,
        args.plot,
        Path(args.plot_path) if args.plot_path else None,
        args.random_state,
        args.min_cluster_size,
        args.min_samples,
        args.metric,
        args.allow_single_cluster,
        args.hdbscan_algorithm,
    )