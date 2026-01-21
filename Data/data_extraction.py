import tarfile
from pathlib import Path
from typing import Iterable, Set


SOURCE_ROOT = Path("Data/raw_data/Downsampled")
DEST_ROOT = Path("Data/extracted_data")


def _safe_extract(tar: tarfile.TarFile, dest_dir: Path) -> None:
    dest_dir = dest_dir.resolve()
    for member in tar.getmembers():
        member_path = dest_dir / member.name
        if not str(member_path.resolve()).startswith(str(dest_dir)):
            raise RuntimeError(f"Blocked path traversal in tar: {member.name}")
    tar.extractall(dest_dir)


def _should_skip_tar(tar_path: Path) -> bool:
    sibling_dir = tar_path.parent / tar_path.stem
    return sibling_dir.is_dir()


def _extract_tar(tar_path: Path, dest_dir: Path, processed: Set[Path]) -> None:
    tar_path = tar_path.resolve()
    if tar_path in processed:
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        _safe_extract(tar, dest_dir)
        members = [member.name for member in tar.getmembers()]

    processed.add(tar_path)

    for member_name in members:
        if not member_name.lower().endswith(".tar"):
            continue
        nested_tar = dest_dir / member_name
        if nested_tar.is_file():
            _extract_tar(nested_tar, nested_tar.parent, processed)


def iter_source_tars(source_root: Path) -> Iterable[Path]:
    yield from source_root.rglob("*.tar")


def extract_all(source_root: Path = SOURCE_ROOT, dest_root: Path = DEST_ROOT) -> None:
    processed: Set[Path] = set()
    for tar_path in iter_source_tars(source_root):
        if _should_skip_tar(tar_path):
            continue
        relative_dir = tar_path.parent.relative_to(source_root)
        dest_dir = dest_root / relative_dir
        _extract_tar(tar_path, dest_dir, processed)


if __name__ == "__main__":
    extract_all()