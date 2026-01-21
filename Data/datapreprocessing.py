import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from utils.load_config import load_config


class WhisperDataLoader:
    def __init__(
        self,
        config_path: str = "Config/dataloader_config.json",
        mode: str = "default",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        config_data = config if config is not None else load_config(config_path)
        mode_config = config_data.get("modes", {}).get(mode, {})

        self.dataset_root = Path(
            mode_config.get("dataset_root", config_data.get("dataset_root", "Data/extracted_data"))
        )
        self.split = mode_config.get("split", config_data.get("split", "Train"))
        self.percent = mode_config.get("percent", config_data.get("percent", 100))
        self.sampling = mode_config.get("sampling", config_data.get("sampling", "random"))
        self.seed = mode_config.get("seed", config_data.get("seed", 42))
        self.max_samples = mode_config.get("max_samples", config_data.get("max_samples"))

        if self.split not in {"Train", "Dev"}:
            raise ValueError("split must be 'Train' or 'Dev'.")
        if not (0 <= self.percent <= 100):
            raise ValueError("percent must be between 0 and 100.")
        if self.sampling not in {"random", "stratified"}:
            raise ValueError("sampling must be 'random' or 'stratified'.")

    def _iter_metadata_files(self) -> Iterable[Path]:
        split_dir = self.dataset_root / self.split
        yield from split_dir.rglob("*.json")

    @staticmethod
    def _parse_metadata(metadata_path: Path) -> List[Dict[str, Any]]:
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            data = json.load(metadata_file)

        etiology = data.get("Etiology", "Unknown")
        files = data.get("Files", [])

        samples: List[Dict[str, Any]] = []
        for file_info in files:
            filename = file_info.get("Filename")
            if not filename:
                continue
            wav_path = metadata_path.parent / filename
            samples.append(
                {
                    "id": filename,
                    "wav_path": str(wav_path),
                    "etiology": etiology,
                    "prompt": file_info.get("Prompt", {}).get("Transcript"),
                }
            )
        return samples

    def build_index(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for metadata_path in self._iter_metadata_files():
            samples.extend(self._parse_metadata(metadata_path))
        return samples

    def _apply_random_sampling(
        self, samples: List[Dict[str, Any]], target: int, rng: random.Random
    ) -> List[Dict[str, Any]]:
        if target >= len(samples):
            rng.shuffle(samples)
            return samples
        return rng.sample(samples, target)

    def _apply_stratified_sampling(
        self, samples: List[Dict[str, Any]], target: int, rng: random.Random
    ) -> List[Dict[str, Any]]:
        if target >= len(samples):
            rng.shuffle(samples)
            return samples

        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for sample in samples:
            groups[sample.get("etiology", "Unknown")].append(sample)

        total = len(samples)
        fractional: List[tuple[str, float]] = []
        counts: Dict[str, int] = {}
        for etiology, group in groups.items():
            exact = (len(group) / total) * target
            count = int(math.floor(exact))
            counts[etiology] = min(count, len(group))
            fractional.append((etiology, exact - count))

        selected_total = sum(counts.values())
        remainder = target - selected_total

        fractional.sort(key=lambda item: item[1], reverse=True)
        for etiology, _ in fractional:
            if remainder <= 0:
                break
            if counts[etiology] < len(groups[etiology]):
                counts[etiology] += 1
                remainder -= 1

        if remainder > 0:
            available = [
                etiology
                for etiology, group in groups.items()
                if counts[etiology] < len(group)
            ]
            while remainder > 0 and available:
                etiology = rng.choice(available)
                counts[etiology] += 1
                remainder -= 1
                if counts[etiology] >= len(groups[etiology]):
                    available.remove(etiology)

        sampled: List[Dict[str, Any]] = []
        for etiology, group in groups.items():
            group_count = counts[etiology]
            if group_count <= 0:
                continue
            sampled.extend(rng.sample(group, group_count))

        rng.shuffle(sampled)
        return sampled

    def sample(self) -> List[Dict[str, Any]]:
        samples = self.build_index()
        rng = random.Random(self.seed)

        target = int(round(len(samples) * (self.percent / 100.0)))
        if self.max_samples is not None:
            target = min(target, int(self.max_samples))
        target = max(0, target)

        if target == 0:
            return []

        if self.sampling == "random":
            return self._apply_random_sampling(samples, target, rng)
        return self._apply_stratified_sampling(samples, target, rng)

    def get_wav_paths(self) -> List[str]:
        return [sample["wav_path"] for sample in self.sample()]