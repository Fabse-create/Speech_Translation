import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _collect_contributor_metadata_files(split_dir: Path) -> tuple[List[Path], List[Path]]:
    metadata_files: List[Path] = []
    missing_metadata_dirs: List[Path] = []

    for contributor_dir in split_dir.iterdir():
        if not contributor_dir.is_dir():
            continue
        metadata_path = contributor_dir / f"{contributor_dir.name}.json"
        if metadata_path.exists():
            metadata_files.append(metadata_path)
        else:
            missing_metadata_dirs.append(contributor_dir)

    return metadata_files, missing_metadata_dirs


def _parse_metadata(metadata_path: Path) -> List[Dict[str, Any]]:
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        data = json.load(metadata_file)
    return data.get("Files", [])


def _resolve_wav_path(metadata_path: Path, filename: str) -> Path:
    return metadata_path.parent / filename


def check_dataset(
    dataset_root: Path,
    split: str,
) -> Dict[str, Any]:
    split_dir = dataset_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    missing: List[Dict[str, str]] = []
    missing_metadata: List[Dict[str, str]] = []
    total_files = 0

    metadata_files, missing_dirs = _collect_contributor_metadata_files(split_dir)
    for contributor_dir in missing_dirs:
        missing_metadata.append({"contributor_dir": str(contributor_dir)})

    for metadata_path in metadata_files:
        # Skip any non-metadata files that do not have the expected structure.
        files = _parse_metadata(metadata_path)
        if not isinstance(files, list):
            continue
        for file_info in files:
            filename = file_info.get("Filename")
            if not filename:
                continue
            total_files += 1
            wav_path = _resolve_wav_path(metadata_path, filename)
            if not wav_path.exists():
                missing.append(
                    {
                        "metadata": str(metadata_path),
                        "filename": filename,
                        "expected_path": str(wav_path),
                    }
                )

    return {
        "split": split,
        "total_files": total_files,
        "missing_count": len(missing),
        "missing_metadata_count": len(missing_metadata),
        "missing": missing,
        "missing_metadata": missing_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check if WAV files referenced by metadata exist."
    )
    parser.add_argument(
        "--dataset-root",
        default="Data/extracted_data",
        help="Path to dataset root (default: Data/extracted_data).",
    )
    parser.add_argument(
        "--split",
        choices=["Train", "Dev", "all"],
        default="Train",
        help="Split to check (Train, Dev, or all).",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Optional path to write JSON report.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    splits = ["Train", "Dev"] if args.split == "all" else [args.split]

    results = [check_dataset(dataset_root, split) for split in splits]

    for result in results:
        split = result["split"]
        total = result["total_files"]
        missing_count = result["missing_count"]
        missing_metadata_count = result["missing_metadata_count"]
        print(
            f"{split}: {missing_count} missing WAVs of {total} files; "
            f"{missing_metadata_count} contributor folders missing metadata"
        )
        if missing_count > 0:
            print("  Example missing entries:")
            for entry in result["missing"][:5]:
                print(f"  - {entry['expected_path']} (from {entry['metadata']})")
        if missing_metadata_count > 0:
            print("  Example contributor folders without metadata:")
            for entry in result["missing_metadata"][:5]:
                print(f"  - {entry['contributor_dir']}")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as report_file:
            json.dump(results, report_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
