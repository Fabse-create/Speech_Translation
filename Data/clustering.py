import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from Clustering_Algorithms.hdbscan import HDBSCAN
from Clustering_Algorithms.spectral_clustering import Spectral_Clustering


def _load_embeddings(embedding_dir: Path) -> Tuple[np.ndarray, List[str]]:
    embedding_files = sorted(embedding_dir.glob("*.npy"))
    if not embedding_files:
        raise FileNotFoundError(f"No .npy embeddings found in {embedding_dir}")

    embeddings = []
    ids = []
    for embedding_path in embedding_files:
        embeddings.append(np.load(embedding_path))
        wav_id = embedding_path.name
        if wav_id.endswith(".npy"):
            wav_id = wav_id[: -len(".npy")]
        ids.append(wav_id)

    return np.asarray(embeddings), ids


def _cluster_hdbscan(embeddings: np.ndarray, ids: List[str], output_dir: Path) -> None:
    model = HDBSCAN()
    model.save_clusters(embeddings, output_dir=output_dir, soft=False, ids=ids)
    model.save_clusters(embeddings, output_dir=output_dir, soft=True, ids=ids)


def _cluster_spectral(embeddings: np.ndarray, ids: List[str], output_dir: Path) -> None:
    model = Spectral_Clustering()
    model.save_clusters(embeddings, output_dir=output_dir, soft=False, ids=ids)
    model.save_clusters(embeddings, output_dir=output_dir, soft=True, ids=ids)


def run(
    embedding_dir: Path = Path("Data/embeddings/whisper_v2_embeddings"),
    algorithm: str = "hdbscan",
    output_dir: Path = Path("Data/embeddings/whisper_v2_embeddings_clustered"),
) -> None:
    embeddings, ids = _load_embeddings(embedding_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    algo = algorithm.lower()
    if algo == "hdbscan":
        _cluster_hdbscan(embeddings, ids, output_dir)
    elif algo in {"spectral", "spectral_clustering"}:
        _cluster_spectral(embeddings, ids, output_dir)
    else:
        raise ValueError("algorithm must be 'hdbscan' or 'spectral'.")


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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(Path(args.embedding_dir), args.algorithm, Path(args.output_dir))