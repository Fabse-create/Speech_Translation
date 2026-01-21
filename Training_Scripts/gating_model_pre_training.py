import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from Evaluation.evaluate_model_training import evaluate_model
from Models.Gating_Model.gating_model import GatingModel
from utils.load_config import load_config


class EmbeddingClusterDataset(Dataset):
    def __init__(
        self,
        embeddings_dir: Path,
        labels_path: Path,
        num_experts: int,
    ) -> None:
        self.embeddings_dir = embeddings_dir
        self.num_experts = num_experts
        self.entries = self._load_entries(labels_path)

    @staticmethod
    def _load_entries(labels_path: Path) -> List[Dict[str, object]]:
        with labels_path.open("r", encoding="utf-8") as labels_file:
            data = json.load(labels_file)

        if not isinstance(data, list):
            raise ValueError("Labels file must contain a list of entries.")
        if not data:
            raise ValueError("Labels file is empty.")

        if isinstance(data[0], dict) and "id" in data[0]:
            return data
        raise ValueError("Labels must include explicit 'id' entries.")

    @staticmethod
    def _pool_embedding(embedding: np.ndarray) -> np.ndarray:
        pooled = embedding
        while pooled.ndim > 1:
            pooled = pooled.mean(axis=0)
        return pooled.astype(np.float32)

    def _to_target(self, label_entry: Dict[str, object]) -> np.ndarray:
        if "probs" in label_entry:
            probs = np.asarray(label_entry["probs"], dtype=np.float32)
            return probs
        if "label" in label_entry:
            label = int(label_entry["label"])
            target = np.zeros(self.num_experts, dtype=np.float32)
            target[label] = 1.0
            return target
        raise ValueError("Label entry must include 'probs' or 'label'.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self.entries[idx]
        sample_id = entry["id"]
        embedding_path = self.embeddings_dir / f"{sample_id}.npy"
        if not embedding_path.exists():
            raise FileNotFoundError(f"Missing embedding file: {embedding_path}")

        embedding = np.load(embedding_path)
        embedding = self._pool_embedding(embedding)
        target = self._to_target(entry)

        if embedding.shape[0] != 0 and target.shape[0] != self.num_experts:
            raise ValueError(
                f"Target size {target.shape[0]} does not match num_experts {self.num_experts}."
            )

        return torch.from_numpy(embedding), torch.from_numpy(target)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(config_path: str = "Config/gating_model_config.json") -> Path:
    config = load_config(config_path)
    training_cfg = config.get("training", {})

    embeddings_dir = Path(training_cfg.get("embeddings_dir", "Data/embeddings/whisper_v2_embeddings"))
    labels_path = Path(training_cfg.get("labels_path", "Data/embeddings/whisper_v2_embeddings_clustered/HDBSCAN_soft.json"))
    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "checkpoints/gating_model"))
    batch_size = int(training_cfg.get("batch_size", 32))
    epochs = int(training_cfg.get("epochs", 10))
    learning_rate = float(training_cfg.get("learning_rate", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    val_split = float(training_cfg.get("val_split", 0.1))
    seed = int(training_cfg.get("seed", 42))
    num_workers = int(training_cfg.get("num_workers", 0))

    _set_seed(seed)

    num_experts = int(config.get("num_experts", 8))
    dataset = EmbeddingClusterDataset(embeddings_dir, labels_path, num_experts)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatingModel(config_path=config_path).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.KLDivLoss(reduction="batchmean")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_meta_path = checkpoint_dir / "best.json"
    best_loss = float("inf")
    if best_meta_path.exists():
        with best_meta_path.open("r", encoding="utf-8") as meta_file:
            best_loss = json.load(meta_file).get("loss", best_loss)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = loss_fn(log_probs, targets)
            loss.backward()
            optimizer.step()

            batch_size_actual = inputs.size(0)
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual

        train_loss = total_loss / max(total_samples, 1)
        metrics = {"loss": train_loss, "accuracy": 0.0}
        if val_size > 0:
            metrics = evaluate_model(model, val_loader, device)

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_path = checkpoint_dir / "best.pt"
            torch.save(model.state_dict(), best_path)
            with best_meta_path.open("w", encoding="utf-8") as meta_file:
                json.dump({"loss": best_loss, "epoch": epoch}, meta_file, indent=2)

    return checkpoint_dir / "best.pt"


if __name__ == "__main__":
    train()