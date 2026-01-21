import json
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

try:
    from sklearn.cluster import SpectralClustering
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("scikit-learn is required for Spectral Clustering.") from exc


class Spectral_Clustering:
    def __init__(
        self,
        n_clusters: int = 10,
        affinity: str = "nearest_neighbors",
        assign_labels: str = "kmeans",
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.model = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            assign_labels=assign_labels,
            random_state=random_state,
            **kwargs,
        )
        self.labels_: Optional[np.ndarray] = None
        self.soft_clusters_: Optional[np.ndarray] = None

    @staticmethod
    def _to_numpy(embeddings: Any) -> np.ndarray:
        if torch is not None and isinstance(embeddings, torch.Tensor):
            return embeddings.detach().cpu().numpy()
        return np.asarray(embeddings)

    def fit(self, embeddings: Any) -> np.ndarray:
        data = self._to_numpy(embeddings)
        self.labels_ = self.model.fit_predict(data)
        self.soft_clusters_ = None
        return self.labels_

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    @staticmethod
    def _cosine_similarity(matrix: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
        return matrix_norm @ centroids_norm.T

    def get_soft_clusters(self, embeddings: Any) -> np.ndarray:
        if self.labels_ is None:
            raise RuntimeError("Model is not fitted yet.")

        data = self._to_numpy(embeddings)
        unique_labels = np.unique(self.labels_)
        centroids = np.vstack(
            [data[self.labels_ == label].mean(axis=0) for label in unique_labels]
        )
        similarity = self._cosine_similarity(data, centroids)
        soft_clusters = self._softmax(similarity)
        self.soft_clusters_ = soft_clusters
        return soft_clusters

    def cluster(self, embeddings: Any, soft: bool = False) -> np.ndarray:
        self.fit(embeddings)
        if soft:
            return self.get_soft_clusters(embeddings)
        return self.labels_

    def save_clusters(
        self,
        embeddings: Any,
        output_dir: Path = Path("Data/embeddings/whisper_v2_embeddings_clustered"),
        soft: bool = False,
        filename: Optional[str] = None,
        ids: Optional[Iterable[str]] = None,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = "Spectral_soft.json" if soft else "Spectral.json"
        output_path = output_dir / filename

        result = self.cluster(embeddings, soft=soft)
        if ids is not None:
            ids_list = list(ids)
            if len(ids_list) != len(result):
                raise ValueError("Length of ids must match number of embeddings.")
            if soft:
                payload = [
                    {"id": item_id, "probs": probs.tolist()}
                    for item_id, probs in zip(ids_list, result)
                ]
            else:
                payload = [
                    {"id": item_id, "label": int(label)}
                    for item_id, label in zip(ids_list, result)
                ]
        else:
            payload = result.tolist()
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
        return output_path

    def plot_umap(
        self,
        embeddings: Any,
        labels: Optional[Iterable[int]] = None,
        save_path: Optional[Path] = None,
        random_state: int = 42,
        **umap_kwargs: Any,
    ) -> Tuple[Any, Any]:
        try:
            import umap
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("umap-learn and matplotlib are required for plotting.") from exc

        data = self._to_numpy(embeddings)
        if labels is None:
            if self.labels_ is None:
                raise RuntimeError("Provide labels or fit the model before plotting.")
            labels = self.labels_

        reducer = umap.UMAP(random_state=random_state, **umap_kwargs)
        reduced = reducer.fit_transform(data)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, s=8, cmap="tab20")
        ax.set_title("Spectral clusters (UMAP)")
        fig.colorbar(scatter, ax=ax, shrink=0.8)

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")

        return fig, ax