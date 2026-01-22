import json
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

try:
    import hdbscan as hdbscan_lib
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("hdbscan is required for HDBSCAN clustering.") from exc


class HDBSCAN:
    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        metric: str = "euclidean",
        algorithm: str = "best",
        **kwargs: Any,
    ) -> None:
        self.model = hdbscan_lib.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            algorithm=algorithm,
            prediction_data=True,
            **kwargs,
        )
        self.labels_: Optional[np.ndarray] = None
        self.soft_clusters_: Optional[np.ndarray] = None

    @staticmethod
    def _to_numpy(embeddings: Any) -> np.ndarray:
        if torch is not None and isinstance(embeddings, torch.Tensor):
            return embeddings.detach().cpu().numpy().astype(np.float64, copy=False)
        return np.asarray(embeddings, dtype=np.float64)

    def fit(self, embeddings: Any) -> np.ndarray:
        data = self._to_numpy(embeddings)
        self.model.fit(data)
        self.labels_ = self.model.labels_
        self.soft_clusters_ = None
        return self.labels_

    def get_soft_clusters(self, include_noise: bool = True) -> np.ndarray:
        if self.labels_ is None:
            raise RuntimeError("Model is not fitted yet.")

        try:
            memberships = hdbscan_lib.all_points_membership_vectors(self.model)
        except AttributeError:
            labels = np.asarray(self.labels_)
            if labels.size == 0:
                raise RuntimeError("No labels available for soft clustering.")
            valid_labels = labels[labels >= 0]
            n_clusters = int(valid_labels.max()) + 1 if valid_labels.size else 0
            if n_clusters == 0:
                memberships = np.zeros((labels.size, 0), dtype=np.float64)
            else:
                memberships = np.zeros((labels.size, n_clusters), dtype=np.float64)
                for idx, label in enumerate(labels):
                    if label >= 0:
                        memberships[idx, int(label)] = 1.0
            if include_noise:
                noise_prob = (labels == -1).astype(np.float64)[:, None]
                memberships_with_noise = np.concatenate([memberships, noise_prob], axis=1)
                self.soft_clusters_ = memberships_with_noise
                return memberships_with_noise
            self.soft_clusters_ = memberships
            return memberships
        if memberships.ndim == 1:
            memberships = memberships[:, None]
        if not include_noise:
            self.soft_clusters_ = memberships
            return memberships

        noise_prob = 1.0 - memberships.sum(axis=1, keepdims=True)
        noise_prob = np.clip(noise_prob, 0.0, 1.0)
        memberships_with_noise = np.concatenate([memberships, noise_prob], axis=1)
        self.soft_clusters_ = memberships_with_noise
        return memberships_with_noise

    def cluster(self, embeddings: Any, soft: bool = False, include_noise: bool = True) -> np.ndarray:
        self.fit(embeddings)
        if soft:
            return self.get_soft_clusters(include_noise=include_noise)
        return self.labels_

    def save_clusters(
        self,
        embeddings: Any,
        output_dir: Path = Path("Data/embeddings/whisper_v2_embeddings_clustered"),
        soft: bool = False,
        include_noise: bool = True,
        filename: Optional[str] = None,
        ids: Optional[Iterable[str]] = None,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = "HDBSCAN_soft.json" if soft else "HDBSCAN.json"
        output_path = output_dir / filename

        result = self.cluster(embeddings, soft=soft, include_noise=include_noise)
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
        ax.set_title("HDBSCAN clusters (UMAP)")
        fig.colorbar(scatter, ax=ax, shrink=0.8)

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")

        return fig, ax