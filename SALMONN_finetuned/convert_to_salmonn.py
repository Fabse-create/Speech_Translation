import json
import glob
import os
import argparse

def extract_file_records(obj):
    # Your source JSON has a "Files" list with records containing
    # Filename / Transcript / Translation. [file:2]
    if isinstance(obj, dict):
        if "Files" in obj and isinstance(obj["Files"], list):
            return obj["Files"]
        for v in obj.values():
            rec = extract_file_records(v)
            if rec is not None:
                return rec
    elif isinstance(obj, list):
        for v in obj:
            rec = extract_file_records(v)
            if rec is not None:
                return rec
    return None

def norm(x):
    return "" if x is None else str(x).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder containing the 800 source .json files")
    ap.add_argument("--output_json", required=True, help="Path to write the merged Salmonn JSON")
    ap.add_argument("--task", default="st", help="Task label (default: st)")
    ap.add_argument("--path_prefix", default="data/wav/", help='Prefix for wav path (default: "data/wav/")')
    args = ap.parse_args()

    annotations = []
    json_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))

    for jp in json_paths:
        with open(jp, "r", encoding="utf-8") as f:
            src = json.load(f)

        records = extract_file_records(src)
        if not records:
            raise ValueError(f"No 'Files' list found in {jp}")

        for r in records:
            fn = norm(r.get("Filename"))
            if not fn:
                continue

            annotations.append({
                "path": args.path_prefix + fn,          # "data/wav/{filename}"
                "text": norm(r["Prompt"].get("Transcript")),      # source Transcript
                "translation": norm(r["Prompt"].get("Translation")),  # source Translation
                "task": args.task
            })

    out = {"annotation": annotations}

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(annotations)} annotation items to {args.output_json}")

if __name__ == "__main__":
    main()
