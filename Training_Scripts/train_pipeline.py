import argparse
import json
import logging
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging to console and optionally to a file."""
    logger = logging.getLogger("train_pipeline")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_path}")
    
    return logger

from Clustering_Algorithms.hdbscan import HDBSCAN
from Clustering_Algorithms.spectral_clustering import Spectral_Clustering
from Data.clustering import _load_embeddings, _plot_embeddings, _reduce_embeddings
from Data.datapreprocessing import WhisperDataLoader
from Data.embedding_extraction import extract_embeddings
from Evaluation.plot_asr_metrics import plot_metrics as plot_asr_metrics
from Evaluation.plot_expert_metrics import plot_metrics as plot_expert_metrics
from Evaluation.plot_gating_metrics import plot_metrics as plot_gating_metrics
from Training_Scripts import asr_training
from Training_Scripts.expert_pre_training import train as train_experts
from Training_Scripts.gating_model_pre_training import train as train_gate
from utils.load_config import load_config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def _remove_if_exists(path: Path) -> None:
    if path.is_file():
        path.unlink()


def _remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _clear_expert_metrics(metrics_root: Path) -> None:
    if not metrics_root.exists():
        return
    for metrics_path in metrics_root.rglob("metrics.json"):
        _remove_if_exists(metrics_path)


def _build_data_override(
    dataset_root: str,
    split: str,
    percent: int,
    sampling: str,
    seed: int,
    max_samples: Optional[int],
) -> Dict[str, Any]:
    return {
        "dataset_root": dataset_root,
        "split": split,
        "percent": percent,
        "sampling": sampling,
        "seed": seed,
        "max_samples": max_samples,
        "modes": {},
    }


def _extract_embeddings(
    dataset_root: str,
    split: str,
    percent: int,
    max_samples: Optional[int],
    seed: int,
    output_dir: Path,
    mapping_path: Path,
    whisper_model: str,
    sampling: str,
) -> Path:
    config = {
        "data_config_path": "Config/dataloader_config.json",
        "data_mode": "default",
        "data_config_override": _build_data_override(
            dataset_root, split, percent, sampling, seed, max_samples
        ),
        "whisper_model": whisper_model,
        "output_dir": str(output_dir),
        "mapping_path": str(mapping_path),
        "overwrite": True,
    }
    tmp_path = output_dir / "embedding_extraction_config.json"
    _write_json(tmp_path, config)
    return extract_embeddings(str(tmp_path))


def _save_hard_labels(ids: List[str], labels: Iterable[int], output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    payload = [{"id": item_id, "label": int(label)} for item_id, label in zip(ids, labels)]
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    return output_path


def _save_soft_labels(ids: List[str], probs: np.ndarray, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    payload = [{"id": item_id, "probs": row.tolist()} for item_id, row in zip(ids, probs)]
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    return output_path


def _merge_clusters_to_max(
    reduced: np.ndarray,
    labels: np.ndarray,
    soft_probs: np.ndarray,
    max_clusters: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    valid_labels = labels[labels >= 0]
    n_clusters = int(valid_labels.max()) + 1 if valid_labels.size else 0
    if n_clusters <= max_clusters:
        return labels, soft_probs, n_clusters
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("scikit-learn is required for cluster merging.") from exc

    centroids = np.vstack(
        [reduced[labels == label].mean(axis=0) for label in range(n_clusters)]
    )
    merger = AgglomerativeClustering(n_clusters=max_clusters)
    merged_ids = merger.fit_predict(centroids)
    mapping = {cluster_id: int(merged_id) for cluster_id, merged_id in enumerate(merged_ids)}

    merged_labels = labels.copy()
    for idx, label in enumerate(labels):
        if label >= 0:
            merged_labels[idx] = mapping[int(label)]

    has_noise = soft_probs.shape[1] == n_clusters + 1
    target_cols = max_clusters + (1 if has_noise else 0)
    merged_probs = np.zeros((soft_probs.shape[0], target_cols), dtype=soft_probs.dtype)
    for cluster_id in range(n_clusters):
        merged_probs[:, mapping[cluster_id]] += soft_probs[:, cluster_id]
    if has_noise:
        merged_probs[:, -1] = soft_probs[:, -1]

    return merged_labels, merged_probs, max_clusters


def _cluster_embeddings(
    embedding_dir: Path,
    output_dir: Path,
    algorithm: str,
    num_experts: int,
    min_clusters: int,
    min_cluster_size: int,
    min_samples: Optional[int],
    metric: str,
    allow_single_cluster: bool,
    hdbscan_algorithm: str,
    allow_reduce_experts: bool,
    min_experts: int,
    max_retries: int,
    pooling: str,
    reduce: str,
    reduce_dim: int,
    plot_method: str,
    seed: int,
) -> Tuple[Path, int]:
    embeddings, ids = _load_embeddings(embedding_dir, pooling)
    if reduce in {"pca", "umap"}:
        max_dim = int(min(embeddings.shape[0], embeddings.shape[1]))
        if max_dim <= 0:
            raise ValueError("Embeddings are empty after loading.")
        reduce_dim = min(int(reduce_dim), max_dim)
    reduced = _reduce_embeddings(embeddings, reduce, reduce_dim, seed)
    algo = algorithm.lower()
    if algo == "hdbscan":
        current_min_cluster_size = max(2, int(min_cluster_size))
        attempt = 0
        num_clusters = 0
        labels = np.array([], dtype=np.int32)
        soft_probs = np.zeros((0, 0), dtype=np.float32)
        while attempt <= max_retries:
            model = HDBSCAN(
                min_cluster_size=current_min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                algorithm=hdbscan_algorithm,
                allow_single_cluster=allow_single_cluster,
            )
            labels = model.fit(reduced)
            soft_probs = model.get_soft_clusters(include_noise=True)
            valid_labels = labels[labels >= 0]
            num_clusters = int(valid_labels.max()) + 1 if valid_labels.size else 0
            if num_clusters >= min_clusters:
                break
            if current_min_cluster_size <= 2:
                break
            current_min_cluster_size = max(2, current_min_cluster_size - 1)
            attempt += 1

        if num_clusters == 0:
            if allow_reduce_experts:
                fallback_clusters = min(num_experts, reduced.shape[0])
                fallback_clusters = max(int(min_experts), int(fallback_clusters))
                if fallback_clusters < 2:
                    raise ValueError(
                        "HDBSCAN produced zero clusters and fallback spectral clustering needs at least 2 samples."
                    )
                model = Spectral_Clustering(n_clusters=fallback_clusters, random_state=seed)
                labels = model.fit(reduced)
                _save_hard_labels(ids, labels, output_dir, "Spectral.json")
                soft_probs = model.get_soft_clusters(reduced)
                _save_soft_labels(ids, soft_probs, output_dir, "Spectral_soft.json")
                plot_embeddings = _reduce_embeddings(reduced, plot_method, 2, seed)
                plot_path = output_dir / f"spectral_{plot_method}.png"
                _plot_embeddings(
                    plot_embeddings,
                    labels,
                    plot_path,
                    f"SPECTRAL clusters ({plot_method})",
                )
                return output_dir / "Spectral_soft.json", fallback_clusters
            raise ValueError("HDBSCAN produced zero clusters (all noise).")
        if num_clusters < min_clusters:
            if allow_reduce_experts:
                min_clusters = max(int(min_experts), num_clusters)
            else:
                raise ValueError(
                    f"HDBSCAN produced {num_clusters} clusters; minimum required is {min_clusters}."
                )

        labels, soft_probs, resolved_clusters = _merge_clusters_to_max(
            reduced, labels, soft_probs, max_clusters=min(num_experts, num_clusters)
        )

        _save_hard_labels(ids, labels, output_dir, "HDBSCAN.json")
        _save_soft_labels(ids, soft_probs, output_dir, "HDBSCAN_soft.json")
        plot_embeddings = _reduce_embeddings(reduced, plot_method, 2, seed)
        plot_path = output_dir / f"hdbscan_{plot_method}.png"
        _plot_embeddings(
            plot_embeddings,
            labels,
            plot_path,
            f"HDBSCAN clusters ({plot_method})",
        )
        return output_dir / "HDBSCAN_soft.json", resolved_clusters

    if algo == "spectral":
        if reduced.shape[0] < num_experts:
            raise ValueError(
                f"Need at least {num_experts} embeddings for spectral clustering, got {reduced.shape[0]}."
            )
        model = Spectral_Clustering(n_clusters=num_experts, random_state=seed)
        labels = model.fit(reduced)
        _save_hard_labels(ids, labels, output_dir, "Spectral.json")
        soft_probs = model.get_soft_clusters(reduced)
        _save_soft_labels(ids, soft_probs, output_dir, "Spectral_soft.json")

        plot_embeddings = _reduce_embeddings(reduced, plot_method, 2, seed)
        plot_path = output_dir / f"spectral_{plot_method}.png"
        _plot_embeddings(
            plot_embeddings,
            labels,
            plot_path,
            f"SPECTRAL clusters ({plot_method})",
        )
        return output_dir / "Spectral_soft.json", num_experts

    raise ValueError("algorithm must be 'hdbscan' or 'spectral'.")


def _load_samples(
    dataset_root: str,
    split: str,
    percent: int,
    sampling: str,
    seed: int,
    max_samples: Optional[int],
) -> List[Dict[str, Any]]:
    loader = WhisperDataLoader(
        config_path="Config/dataloader_config.json",
        mode="default",
        config=_build_data_override(dataset_root, split, percent, sampling, seed, max_samples),
    )
    return [sample for sample in loader.sample() if sample.get("prompt")]


def _stratified_split(
    samples: List[Dict[str, Any]],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_ratio, val_ratio, test_ratio = ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train/val/test ratios must sum to 1.0.")
    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[sample.get("etiology", "Unknown")].append(sample)

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    for group in groups.values():
        rng.shuffle(group)
        n_total = len(group)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        n_train = min(n_train, n_total)
        n_val = min(n_val, n_total - n_train)
        n_test = n_total - n_train - n_val
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val : n_train + n_val + n_test])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _build_moe_dataset(samples: List[Dict[str, Any]]) -> "asr_training.WhisperMoEDataset":
    return asr_training.WhisperMoEDataset(
        [
            {
                "id": sample["id"],
                "wav_path": sample["wav_path"],
                "transcript": sample.get("prompt", ""),
            }
            for sample in samples
        ]
    )


def _run_asr_training(
    config_path: Path,
    samples: List[Dict[str, Any]],
    seed: int,
    output_dir: Path,
    metrics_dir: Path,
) -> None:
    config = asr_training._load_training_config(str(config_path))
    _set_seed(seed)

    train_samples, val_samples, test_samples = _stratified_split(
        samples, (0.8, 0.1, 0.1), seed
    )
    train_set = _build_moe_dataset(train_samples)
    val_set = _build_moe_dataset(val_samples)
    test_set = _build_moe_dataset(test_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = asr_training.WhisperProcessor.from_pretrained(config.model_name)
    collator = asr_training.WhisperCollator(processor)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collator,
    )

    embedding_model = asr_training._build_embedding_model(config, device)
    model = asr_training._build_moe_model(config).to(device)
    gating_model = asr_training._load_gating_model(config, device)
    if config.experts_dir and config.use_lora:
        asr_training._load_expert_adapters(model, Path(config.experts_dir), config.num_experts)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=config.language, task=config.task
    )
    model.config.forced_decoder_ids = forced_decoder_ids

    params = list(gating_model.parameters()) + [
        param for param in model.parameters() if param.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        params, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "metrics.json"
    _remove_if_exists(metrics_path)
    metrics_history: List[Dict[str, float]] = []

    best_loss = float("inf")
    best_path = output_dir / "best.json"
    _remove_if_exists(best_path)

    # Early stopping configuration
    early_stopping_patience = 5
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = asr_training._train_epoch(
            model=model,
            gating_model=gating_model,
            embedding_model=embedding_model,
            data_loader=train_loader,
            optimizer=optimizer,
            config=config,
            device=device,
            processor=processor,
            scaler=scaler,
        )
        val_loss = train_loss
        if len(val_set) > 0:
            val_loss = asr_training._evaluate(
                model=model,
                gating_model=gating_model,
                embedding_model=embedding_model,
                data_loader=val_loader,
                config=config,
                device=device,
            )
            val_wer = asr_training._evaluate_wer(
                model=model,
                gating_model=gating_model,
                embedding_model=embedding_model,
                data_loader=val_loader,
                config=config,
                device=device,
                processor=processor,
            )
        else:
            val_wer = None

        metrics_history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_wer": None if val_wer is None else float(val_wer),
            }
        )
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            json.dump(metrics_history, metrics_file, indent=2)

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save(gating_model.state_dict(), output_dir / "gating_model.pt")
            if config.use_lora:
                for expert_id in range(config.num_experts):
                    model.set_adapter(f"expert_{expert_id}")
                    adapter_dir = output_dir / f"expert_{expert_id}"
                    adapter_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(adapter_dir)
                    processor.save_pretrained(adapter_dir)
            else:
                model.save_pretrained(output_dir / "model")
                processor.save_pretrained(output_dir / "model")
            with best_path.open("w", encoding="utf-8") as best_file:
                json.dump({"loss": best_loss, "epoch": epoch}, best_file, indent=2)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs (no improvement for {early_stopping_patience} epochs)")
                break

    test_metrics_path = metrics_dir / "test_metrics.json"
    _remove_if_exists(test_metrics_path)
    if len(test_set) > 0:
        test_loss = asr_training._evaluate(
            model=model,
            gating_model=gating_model,
            embedding_model=embedding_model,
            data_loader=test_loader,
            config=config,
            device=device,
        )
        test_wer = asr_training._evaluate_wer(
            model=model,
            gating_model=gating_model,
            embedding_model=embedding_model,
            data_loader=test_loader,
            config=config,
            device=device,
            processor=processor,
        )
        with test_metrics_path.open("w", encoding="utf-8") as metrics_file:
            json.dump({"test_loss": float(test_loss), "test_wer": float(test_wer)}, metrics_file, indent=2)


def _build_gating_config(
    base_path: str,
    num_experts: int,
    embeddings_dir: Path,
    labels_path: Path,
    checkpoint_dir: Path,
    metrics_dir: Path,
    seed: int,
    batch_size: Optional[int],
    num_workers: Optional[int],
) -> Path:
    config = load_config(base_path)
    config["num_experts"] = num_experts
    training_cfg = config.get("training", {})
    training_cfg["embeddings_dir"] = str(embeddings_dir)
    training_cfg["labels_path"] = str(labels_path)
    training_cfg["checkpoint_dir"] = str(checkpoint_dir)
    training_cfg["metrics_dir"] = str(metrics_dir)
    training_cfg["seed"] = seed
    if batch_size is not None:
        training_cfg["batch_size"] = int(batch_size)
    if num_workers is not None:
        training_cfg["num_workers"] = int(num_workers)
    config["training"] = training_cfg
    output_path = checkpoint_dir / "gating_pretrain_config.json"
    return _write_json(output_path, config)


def _build_expert_config(
    base_path: str,
    num_experts: int,
    gating_checkpoint: Path,
    gating_config: str,
    data_override: Dict[str, Any],
    seed: int,
    batch_size: Optional[int],
    num_workers: Optional[int],
    fp16: bool,
    embeddings_dir: Optional[Path] = None,
    gradient_accumulation_steps: int = 1,
) -> Path:
    config = load_config(base_path)
    config["num_experts"] = num_experts
    config["gating_model_checkpoint"] = str(gating_checkpoint)
    config["gating_model_config"] = gating_config
    config["data_config_override"] = data_override
    config["seed"] = seed
    if batch_size is not None:
        config["batch_size"] = int(batch_size)
    if num_workers is not None:
        config["num_workers"] = int(num_workers)
    config["fp16"] = bool(fp16)
    # OPTIMIZATION: Pass cached embeddings directory
    if embeddings_dir is not None:
        config["embeddings_dir"] = str(embeddings_dir)
    config["gradient_accumulation_steps"] = gradient_accumulation_steps
    config["pin_memory"] = True
    output_path = Path(config.get("output_dir", "checkpoints/experts")) / "expert_pretrain_config.json"
    return _write_json(output_path, config)


def _build_asr_config(
    base_path: str,
    num_experts: int,
    gating_checkpoint: Path,
    gating_config: str,
    experts_dir: Path,
    data_override: Dict[str, Any],
    seed: int,
    output_dir: Path,
    metrics_dir: Path,
    batch_size: Optional[int],
    num_workers: Optional[int],
    fp16: bool,
    gradient_accumulation_steps: int = 1,
    eval_every_n_epochs: int = 1,
    save_every_n_epochs: int = 5,
) -> Path:
    config = load_config(base_path)
    config["num_experts"] = num_experts
    config["gating_checkpoint"] = str(gating_checkpoint)
    config["gating_model_config"] = gating_config
    config["experts_dir"] = str(experts_dir)
    config["data_config_override"] = data_override
    config["seed"] = seed
    config["val_split"] = 0.1
    config["output_dir"] = str(output_dir)
    config["metrics_dir"] = str(metrics_dir)
    if batch_size is not None:
        config["batch_size"] = int(batch_size)
    if num_workers is not None:
        config["num_workers"] = int(num_workers)
    config["fp16"] = bool(fp16)
    # OPTIMIZATION: Add gradient accumulation and checkpoint settings
    config["gradient_accumulation_steps"] = gradient_accumulation_steps
    config["eval_every_n_epochs"] = eval_every_n_epochs
    config["save_every_n_epochs"] = save_every_n_epochs
    config["pin_memory"] = True
    output_path = output_dir / "asr_training_config.json"
    return _write_json(output_path, config)


def run_pipeline(
    mode: str,
    seed: int,
    num_experts: int,
    clustering_algorithm: str,
    min_clusters: int,
    min_cluster_size: int,
    min_samples: Optional[int],
    metric: str,
    allow_single_cluster: bool,
    hdbscan_algorithm: str,
    allow_reduce_experts: bool,
    min_experts: int,
    max_retries: int,
    gating_batch_size: Optional[int],
    expert_batch_size: Optional[int],
    asr_batch_size: Optional[int],
    num_workers: Optional[int],
    fp16: Optional[bool],
    dataset_root: str,
    whisper_model: str,
    pooling: str,
    reduce: str,
    reduce_dim: int,
    plot_method: str,
    resume: bool = False,
    no_plot: bool = False,
    gradient_accumulation_steps: int = 1,
    eval_every_n_epochs: int = 1,
    save_every_n_epochs: int = 5,
    data_percent: Optional[float] = None,
    log_file: Optional[str] = None,
) -> None:
    _set_seed(seed)

    # Setup logging
    logger = _setup_logging(log_file)
    logger.info(f"Starting pipeline in '{mode}' mode with seed={seed}")
    if data_percent is not None:
        logger.info(f"Using custom data percentage: {data_percent}%")

    run_root = Path("Runs") / mode
    embeddings_dir = run_root / "embeddings" / f"whisper_{whisper_model}_embeddings"
    embeddings_mapping = embeddings_dir / "mapping.json"
    clustered_dir = run_root / "clustered"

    gating_ckpt_dir = run_root / "gating_model"
    gating_metrics_dir = run_root / "gating_model_results"

    asr_output_dir = run_root / "asr"
    asr_metrics_dir = run_root / "asr_results"

    if mode == "quick":
        embedding_percent = 100
        embedding_max = 10
        expert_percent = 100
        expert_max = 5
        asr_percent = 100
        asr_max = 5
        include_dev = False
        if min_clusters > num_experts:
            min_clusters = num_experts
        if not allow_reduce_experts:
            allow_reduce_experts = True
    else:
        embedding_percent = 5
        embedding_max = None
        expert_percent = 15
        expert_max = None
        asr_percent = 100
        asr_max = None
        include_dev = True

    # Override percentages if data_percent is specified (scales proportionally)
    if data_percent is not None:
        # Scale each stage proportionally: stage_percent * (data_percent / 100)
        scale_factor = data_percent / 100.0
        embedding_percent = embedding_percent * scale_factor
        expert_percent = expert_percent * scale_factor
        asr_percent = asr_percent * scale_factor
        embedding_max = None  # Use percentage, not fixed count
        expert_max = None
        asr_max = None
        logger.info(f"Scaling data to {data_percent}% - effective percentages: embedding={embedding_percent:.2f}%, expert={expert_percent:.2f}%, asr={asr_percent:.2f}%")

    # OPTIMIZATION: Only remove run directory if not resuming
    if not resume:
        _remove_dir(run_root)
    else:
        logger.info(f"Resuming from existing run at {run_root}")

    # STEP 1: Embedding extraction (with resume support)
    skip_embeddings = resume and embeddings_mapping.exists() and len(list(embeddings_dir.rglob("*.npy"))) > 0
    if skip_embeddings:
        logger.info(f"[RESUME] Skipping embedding extraction - found {len(list(embeddings_dir.rglob('*.npy')))} embeddings")
    else:
        logger.info("[STEP 1/5] Extracting embeddings...")
        _extract_embeddings(
            dataset_root=dataset_root,
            split="Train",
            percent=embedding_percent,
            max_samples=embedding_max,
            seed=seed,
            output_dir=embeddings_dir,
            mapping_path=embeddings_mapping,
            whisper_model=whisper_model,
            sampling="stratified",
        )

    # STEP 2: Clustering (with resume support)
    soft_labels_path = clustered_dir / "HDBSCAN_soft.json"
    spectral_labels_path = clustered_dir / "Spectral_soft.json"
    skip_clustering = resume and (soft_labels_path.exists() or spectral_labels_path.exists())

    if skip_clustering:
        # Determine which labels file exists and count experts
        if soft_labels_path.exists():
            labels_path = soft_labels_path
        else:
            labels_path = spectral_labels_path
        import json as _json
        with labels_path.open("r") as f:
            labels_data = _json.load(f)
        if labels_data and "probs" in labels_data[0]:
            resolved_experts = len(labels_data[0]["probs"])
            # Adjust for noise column if present
            if resolved_experts > num_experts:
                resolved_experts = num_experts
        else:
            resolved_experts = num_experts
        logger.info(f"[RESUME] Skipping clustering - found labels at {labels_path} with {resolved_experts} experts")
    else:
        logger.info("[STEP 2/5] Clustering embeddings...")
        labels_path, resolved_experts = _cluster_embeddings(
            embedding_dir=embeddings_dir,
            output_dir=clustered_dir,
            algorithm=clustering_algorithm,
            num_experts=num_experts,
            min_clusters=min_clusters,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            allow_single_cluster=allow_single_cluster,
            hdbscan_algorithm=hdbscan_algorithm,
            allow_reduce_experts=allow_reduce_experts,
            min_experts=min_experts,
            max_retries=max_retries,
            pooling=pooling,
            reduce=reduce,
            reduce_dim=reduce_dim,
            plot_method=plot_method if not no_plot else "pca",  # Use PCA if not plotting
            seed=seed,
        )

    if fp16 is None:
        fp16 = torch.cuda.is_available()

    # STEP 3: Gating model pre-training (with resume support)
    gating_best_checkpoint = gating_ckpt_dir / "best.pt"
    skip_gating = resume and gating_best_checkpoint.exists()

    if skip_gating:
        gating_checkpoint = gating_best_checkpoint
        gating_config = gating_ckpt_dir / "gating_pretrain_config.json"
        logger.info(f"[RESUME] Skipping gating pre-training - found checkpoint at {gating_checkpoint}")
    else:
        logger.info("[STEP 3/5] Training gating model...")
        gating_config = _build_gating_config(
            base_path="Config/gating_model_config.json",
            num_experts=resolved_experts,
            embeddings_dir=embeddings_dir,
            labels_path=labels_path,
            checkpoint_dir=gating_ckpt_dir,
            metrics_dir=gating_metrics_dir,
            seed=seed,
            batch_size=gating_batch_size,
            num_workers=num_workers,
        )
        if not resume:
            _remove_if_exists(gating_metrics_dir / "metrics.json")
            _remove_if_exists(gating_ckpt_dir / "best.json")
        gating_checkpoint = train_gate(str(gating_config))
        if not no_plot:
            plot_gating_metrics(gating_metrics_dir / "metrics.json", gating_metrics_dir)

    # STEP 4: Expert pre-training (with resume support)
    experts_output_dir = Path("checkpoints/experts")
    expert_dirs_exist = [
        (experts_output_dir / f"expert_{i}").exists() 
        for i in range(resolved_experts)
    ]
    skip_experts = resume and any(expert_dirs_exist)

    if skip_experts:
        logger.info(f"[RESUME] Skipping expert pre-training - found {sum(expert_dirs_exist)}/{resolved_experts} expert directories")
    else:
        logger.info("[STEP 4/5] Training experts...")
        expert_override = _build_data_override(
            dataset_root, "Train", expert_percent, "stratified", seed, expert_max
        )
        expert_config = _build_expert_config(
            base_path="Config/expert_pre_training.json",
            num_experts=resolved_experts,
            gating_checkpoint=gating_checkpoint,
            gating_config=str(gating_config),
            data_override=expert_override,
            seed=seed,
            batch_size=expert_batch_size,
            num_workers=num_workers,
            fp16=fp16,
            embeddings_dir=embeddings_dir,  # OPTIMIZATION: Pass cached embeddings
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        if not resume:
            _clear_expert_metrics(Path("Evaluation/expert_training_results"))
        train_experts(str(expert_config))
        if not no_plot:
            plot_expert_metrics(Path("Evaluation/expert_training_results"), Path("Evaluation/expert_training_results"))

    # Clear GPU memory before ASR training to prevent OOM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        logger.info("Cleared GPU memory before ASR training")

    # STEP 5: Full ASR training (with resume support)
    asr_best_checkpoint = asr_output_dir / "best.json"
    skip_asr = resume and asr_best_checkpoint.exists()

    if skip_asr:
        logger.info(f"[RESUME] Skipping ASR training - found checkpoint at {asr_best_checkpoint}")
    else:
        logger.info("[STEP 5/5] Training full ASR model...")
        asr_override = _build_data_override(
            dataset_root, "Train", asr_percent, "stratified", seed, asr_max
        )
        asr_config = _build_asr_config(
            base_path="Config/asr_training.json",
            num_experts=resolved_experts,
            gating_checkpoint=gating_checkpoint,
            gating_config=str(gating_config),
            experts_dir=Path("checkpoints/experts"),
            data_override=asr_override,
            seed=seed,
            output_dir=asr_output_dir,
            metrics_dir=asr_metrics_dir,
            batch_size=asr_batch_size,
            num_workers=num_workers,
            fp16=fp16,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_every_n_epochs=eval_every_n_epochs,
            save_every_n_epochs=save_every_n_epochs,
        )

        train_samples = _load_samples(
            dataset_root,
            split="Train",
            percent=asr_percent,
            sampling="stratified",
            seed=seed,
            max_samples=asr_max,
        )
        dev_samples: List[Dict[str, Any]] = []
        if include_dev:
            dev_samples = _load_samples(
                dataset_root,
                split="Dev",
                percent=asr_percent,
                sampling="stratified",
                seed=seed,
                max_samples=asr_max,
            )
        all_samples = train_samples + dev_samples
        if not all_samples:
            raise ValueError("No ASR samples with transcripts available for training.")

        _run_asr_training(
            config_path=asr_config,
            samples=all_samples,
            seed=seed,
            output_dir=asr_output_dir,
            metrics_dir=asr_metrics_dir,
        )
        if not no_plot:
            plot_asr_metrics(asr_metrics_dir / "metrics.json", asr_metrics_dir)

    logger.info("="*50)
    logger.info("Pipeline complete!")
    logger.info("="*50)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end training pipeline.")
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="full",
        help="Run a quick smoke test or the full pipeline.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-experts", type=int, default=8, help="Number of experts.")
    parser.add_argument(
        "--clustering-algorithm",
        choices=["hdbscan", "spectral"],
        default="hdbscan",
        help="Clustering algorithm (hdbscan is standard).",
    )
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=8,
        help="Minimum number of clusters required (HDBSCAN only).",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="HDBSCAN min_cluster_size.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="HDBSCAN min_samples (defaults to min_cluster_size).",
    )
    parser.add_argument(
        "--metric",
        default="euclidean",
        help="HDBSCAN distance metric (e.g., euclidean, cosine).",
    )
    parser.add_argument(
        "--allow-single-cluster",
        action="store_true",
        help="HDBSCAN allow single cluster instead of all noise.",
    )
    parser.add_argument(
        "--hdbscan-algorithm",
        default="best",
        choices=[
            "best",
            "generic",
            "prims_kdtree",
            "prims_balltree",
            "boruvka_kdtree",
            "boruvka_balltree",
        ],
        help="HDBSCAN backend algorithm.",
    )
    parser.add_argument(
        "--no-reduce-experts",
        action="store_true",
        help="Fail if HDBSCAN yields fewer clusters than requested.",
    )
    parser.add_argument(
        "--min-experts",
        type=int,
        default=2,
        help="Minimum experts if reduction is allowed.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retries for HDBSCAN with decreasing min_cluster_size.",
    )
    parser.add_argument(
        "--gating-batch-size",
        type=int,
        default=None,
        help="Override gating pre-training batch size.",
    )
    parser.add_argument(
        "--expert-batch-size",
        type=int,
        default=None,
        help="Override expert pre-training batch size.",
    )
    parser.add_argument(
        "--asr-batch-size",
        type=int,
        default=None,
        help="Override ASR training batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override DataLoader num_workers for training.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Force fp16 for expert/ASR training (default: on when CUDA is available).",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 even if CUDA is available.",
    )
    parser.add_argument(
        "--dataset-root",
        default="Data/extracted_data",
        help="Root directory for extracted dataset.",
    )
    parser.add_argument(
        "--whisper-model",
        choices=["v2", "v3"],
        default="v2",
        help="Whisper model for embedding extraction.",
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "flatten", "none"],
        default="mean",
        help="Pooling for clustering embeddings.",
    )
    parser.add_argument(
        "--reduce",
        choices=["none", "pca", "umap"],
        default="pca",
        help="Dimensionality reduction for clustering.",
    )
    parser.add_argument(
        "--reduce-dim", type=int, default=50, help="Target dimension for reduction."
    )
    parser.add_argument(
        "--plot-method",
        choices=["pca", "umap"],
        default="umap",
        help="Visualization method for clusters.",
    )
    # NEW: Resume and optimization arguments
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoints instead of starting fresh.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (useful for headless servers or faster runs).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps for larger effective batch sizes.",
    )
    parser.add_argument(
        "--eval-every-n-epochs",
        type=int,
        default=1,
        help="Evaluate every N epochs (higher = faster training).",
    )
    parser.add_argument(
        "--save-every-n-epochs",
        type=int,
        default=5,
        help="Save checkpoints every N epochs.",
    )
    parser.add_argument(
        "--data-percent",
        type=float,
        default=None,
        help="Override data percentage for all stages (e.g., 0.5, 1, 10, 100).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file. If specified, all output is logged to this file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        mode=args.mode,
        seed=args.seed,
        num_experts=args.num_experts,
        clustering_algorithm=args.clustering_algorithm,
        min_clusters=args.min_clusters,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric=args.metric,
        allow_single_cluster=args.allow_single_cluster,
        hdbscan_algorithm=args.hdbscan_algorithm,
        allow_reduce_experts=not args.no_reduce_experts,
        min_experts=args.min_experts,
        max_retries=args.max_retries,
        gating_batch_size=args.gating_batch_size,
        expert_batch_size=args.expert_batch_size,
        asr_batch_size=args.asr_batch_size,
        num_workers=args.num_workers,
        fp16=(True if args.fp16 else False) if args.no_fp16 else (True if args.fp16 else None),
        dataset_root=args.dataset_root,
        whisper_model=args.whisper_model,
        pooling=args.pooling,
        reduce=args.reduce,
        reduce_dim=args.reduce_dim,
        plot_method=args.plot_method,
        # NEW: Resume and optimization arguments
        resume=args.resume,
        no_plot=args.no_plot,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_every_n_epochs=args.eval_every_n_epochs,
        save_every_n_epochs=args.save_every_n_epochs,
        # Data percentage and logging
        data_percent=args.data_percent,
        log_file=args.log_file,
    )
