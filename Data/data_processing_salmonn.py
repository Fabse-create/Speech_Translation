#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path

BASE_PATH = "/pfs/data6/home/ka/ka_stud/ka_uynlv/Speech_Translation/Data/extracted_data/Train"

def iter_json_files(input_dir: Path):
    for p in sorted(input_dir.rglob("*.json")):
        if p.is_file():
            yield p

def load_annotations_from_one_json(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    contributor_id = obj.get("Contributor ID")
    if not contributor_id:
        return []

    files = obj.get("Files", [])
    out = []

    for entry in files:
        filename = entry.get("Filename")
        prompt = entry.get("Prompt") or {}
        text = prompt.get("Transcript")
        translation = prompt.get("Translation")

        # Skip incomplete examples
        if not filename or text is None or translation is None:
            continue

          # Build POSIX-style path (forward slashes, no escaping issues)
        full_path = f"{BASE_PATH}/{contributor_id}/{filename}"

        ann = {
            "path": str(full_path),
            "text": str(text).strip(),
            "translation": str(translation).strip(),
            "task": "st",
        }
        out.append(ann)

    return out

def write_json(out_path: Path, annotations):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"annotation": annotations}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path, required=True, help="Path to downsampled folder containing JSONs")
    ap.add_argument("--output_dir", type=Path, required=True, help="Where to write sa_data_{train,dev,test}.json")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    all_annotations = []
    for jp in iter_json_files(args.input_dir):
        all_annotations.extend(load_annotations_from_one_json(jp))

    rng = random.Random(args.seed)
    rng.shuffle(all_annotations)

    n = len(all_annotations)
    n_train = int(0.8 * n)
    n_dev = int(0.1 * n)
    # ensure all remaining go to test
    n_test = n - n_train - n_dev

    train = all_annotations[:n_train]
    dev = all_annotations[n_train:n_train + n_dev]
    test = all_annotations[n_train + n_dev:]

    write_json(args.output_dir / "sa_data_train.json", train)
    write_json(args.output_dir / "sa_data_dev.json", dev)
    write_json(args.output_dir / "sa_data_test.json", test)

    print(f"Total: {n} | train: {len(train)} | dev: {len(dev)} | test: {len(test)}")

if __name__ == "__main__":
    main()

# python3 build_splits.py \
#   --input_dir /path/to/downsampled \
#   --output_dir /path/to/output \
#   --seed 13
