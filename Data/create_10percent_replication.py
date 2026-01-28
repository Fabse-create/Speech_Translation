import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
from Data.datapreprocessing import WhisperDataLoader


def _build_data_override(
    dataset_root: str,
    split: str,
    percent: float,
    sampling: str,
    seed: int,
    max_samples: Optional[int],
) -> Dict[str, Any]:
    """
    Mirror the override structure used elsewhere so sampling behavior matches
    the main training pipeline / dataloader usage.
    """
    return {
        "dataset_root": dataset_root,
        "split": split,
        "percent": percent,
        "sampling": sampling,
        "seed": seed,
        "max_samples": max_samples,
        "modes": {},
    }


def _load_representative_samples(
    source_root: Path,
    split: str,
    percent: float,
    sampling: str,
    seed: int,
    max_samples: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Use WhisperDataLoader to select a representative subset of samples for a split.
    """
    override = _build_data_override(
        dataset_root=str(source_root),
        split=split,
        percent=percent,
        sampling=sampling,
        seed=seed,
        max_samples=max_samples,
    )
    loader = WhisperDataLoader(
        config_path="Config/dataloader_config.json",
        mode="default",
        config=override,
    )
    samples = [s for s in loader.sample() if s.get("prompt")]
    return samples


def _copy_split(
    source_root: Path,
    target_root: Path,
    split: str,
    percent: float,
    sampling: str,
    seed: int,
    max_samples: Optional[int],
) -> int:
    """
    Sample a split and copy JSON + audio into the replicated dataset root,
    preserving structure and metadata but only for the selected subset.
    """
    split_src = source_root / split
    if not split_src.exists():
        print(f"[WARN] Split '{split}' not found under {source_root}, skipping.")
        return 0

    samples = _load_representative_samples(
        source_root=source_root,
        split=split,
        percent=percent,
        sampling=sampling,
        seed=seed,
        max_samples=max_samples,
    )
    if not samples:
        print(f"[WARN] No samples selected for split '{split}', skipping.")
        return 0

    print(
        f"[INFO] Selected {len(samples)} samples for split '{split}' "
        f"({percent}% with sampling='{sampling}', seed={seed})."
    )

    # Group samples by contributor to reconstruct per-speaker JSON files.
    by_contributor: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        contributor_id = s.get("contributor_id")
        if not contributor_id:
            # Fallback: infer contributor from directory name
            wav_path = Path(s["wav_path"])
            contributor_id = wav_path.parent.name
        by_contributor[contributor_id].append(s)

    copied_count = 0
    for contributor_id, group in by_contributor.items():
        # All samples in the group share the same directory / metadata file.
        example_wav = Path(group[0]["wav_path"])
        src_contrib_dir = example_wav.parent
        src_meta_path = src_contrib_dir / f"{contributor_id}.json"
        if not src_meta_path.exists():
            print(f"[WARN] Missing metadata for contributor '{contributor_id}': {src_meta_path}")
            continue

        # Load original metadata and filter Files to the subset of filenames.
        with src_meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        files = meta.get("Files", [])
        # Build filename set for fast filtering
        keep_filenames = {Path(s["wav_path"]).name for s in group}
        filtered_files = [fi for fi in files if fi.get("Filename") in keep_filenames]
        if not filtered_files:
            continue

        meta["Files"] = filtered_files

        # Prepare target paths
        dest_contrib_dir = target_root / split / contributor_id
        dest_contrib_dir.mkdir(parents=True, exist_ok=True)
        dest_meta_path = dest_contrib_dir / f"{contributor_id}.json"

        # Write filtered metadata
        with dest_meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # Copy audio files
        for s in group:
            wav_path = Path(s["wav_path"])
            dest_wav_path = dest_contrib_dir / wav_path.name
            if not dest_wav_path.exists():
                shutil.copy2(wav_path, dest_wav_path)
                copied_count += 1

    print(
        f"[INFO] Finished split '{split}': {copied_count} audio files copied "
        f"into {target_root / split}."
    )
    return copied_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 10%% (or custom) representative replication of the dataset "
            "with identical structure, using the same sampling behavior as the "
            "WhisperDataLoader."
        )
    )
    parser.add_argument(
        "--source-root",
        default="Data/extracted_data",
        help="Path to original extracted_data root.",
    )
    parser.add_argument(
        "--target-root",
        default="10percent_data_replication",
        help="Path where the replicated subset will be written.",
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=10.0,
        help="Percent of data to sample per split (default: 10.0).",
    )
    parser.add_argument(
        "--sampling",
        choices=["random", "stratified"],
        default="stratified",
        help="Sampling strategy (must match training).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on total samples per split (None = no cap).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Train", "Dev"],
        help="Which splits to replicate (default: Train Dev).",
    )

    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    target_root = Path(args.target_root).resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    print(f"[INFO] Source dataset root: {source_root}")
    print(f"[INFO] Target replication root: {target_root}")
    print(f"[INFO] Sampling {args.percent}% with strategy='{args.sampling}', seed={args.seed}")

    total_copied = 0
    for split in args.splits:
        total_copied += _copy_split(
            source_root=source_root,
            target_root=target_root,
            split=split,
            percent=args.percent,
            sampling=args.sampling,
            seed=args.seed,
            max_samples=args.max_samples,
        )

    print(
        f"[DONE] Replication complete. Total audio files copied: {total_copied}."
        f"\n       You can now (in theory) point the pipeline to '{target_root}' "
        f"by setting dataset_root accordingly."
    )


if __name__ == "__main__":
    main()

