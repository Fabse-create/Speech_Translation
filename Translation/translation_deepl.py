import json
import os
import time
import requests

INPUT_FOLDER = r"D:\Speech_translation\SpeechAccessibility_2025-11-02_Train_Only_Json_downsampled_500k"
OUTPUT_FOLDER = r"D:\Speech_translation\downsampled_500k_translated_deepl"

# Recommended: put your key in an environment variable instead of hardcoding.
DEEPL_AUTH_KEY = "92a6da85-1898-4629-b996-dec0c19d0115:fx"

# If you use DeepL API Free, use api-free.deepl.com. If Pro, use api.deepl.com. [web:4]
DEEPL_BASE_URL = os.environ.get("DEEPL_BASE_URL", "https://api-free.deepl.com").strip()
DEEPL_TRANSLATE_URL = f"{DEEPL_BASE_URL}/v2/translate"

SOURCE_LANG = "EN"
TARGET_LANG = "DE"

MAX_TEXTS_PER_REQUEST = 40
SLEEP_BETWEEN_REQUESTS_SEC = 0.2

OUT_FIELD_NAME = "Translation"  # stored under file_entry["Prompt"][OUT_FIELD_NAME]


def norm(s: str) -> str:
    return " ".join((s or "").split()).strip()


def get_transcript(entry: dict) -> str:
    prompt = entry.get("Prompt", {})
    if isinstance(prompt, dict):
        return prompt.get("Transcript") or ""
    return ""


def deepl_translate_batch(texts: list[str]) -> list[str]:
    """Translate a list of strings in one request; returns translated strings in same order. [web:4]"""
    if not texts:
        return []

    headers = {
        "Authorization": f"DeepL-Auth-Key {DEEPL_AUTH_KEY}",  # [web:4]
        "Content-Type": "application/json",
    }
    payload = {
        "text": texts,               # array of strings [web:4]
        "source_lang": SOURCE_LANG,  # optional [web:4]
        "target_lang": TARGET_LANG,  # required [web:4]
    }

    r = requests.post(DEEPL_TRANSLATE_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    translations = data.get("translations", [])
    return [t["text"] for t in translations]


def process_file(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("Files", [])
    if not isinstance(entries, list) or not entries:
        return data

    to_translate: list[str] = []
    index_map: list[int] = []

    for i, file_entry in enumerate(entries):
        if not isinstance(file_entry, dict):
            continue

        # Ensure Prompt exists as a dict
        prompt = file_entry.get("Prompt")
        if not isinstance(prompt, dict):
            continue  # no prompt -> nothing to translate in your current logic

        # Skip if already translated (inside Prompt)
        if OUT_FIELD_NAME in prompt:
            continue

        src = norm(get_transcript(file_entry))
        if not src:
            continue

        to_translate.append(src)
        index_map.append(i)

    # Batch translate
    for start in range(0, len(to_translate), MAX_TEXTS_PER_REQUEST):
        batch_texts = to_translate[start:start + MAX_TEXTS_PER_REQUEST]
        batch_idxs = index_map[start:start + MAX_TEXTS_PER_REQUEST]

        batch_trans = deepl_translate_batch(batch_texts)

        for idx, tgt in zip(batch_idxs, batch_trans):
            p = entries[idx].get("Prompt")
            if not isinstance(p, dict):
                p = {}
                entries[idx]["Prompt"] = p
            p[OUT_FIELD_NAME] = tgt  # write inside Prompt

        if SLEEP_BETWEEN_REQUESTS_SEC:
            time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)

    data["Files"] = entries
    return data


def main():
    if not DEEPL_AUTH_KEY:
        raise SystemExit("Set DEEPL_AUTH_KEY first.")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for fname in os.listdir(INPUT_FOLDER):
        if not fname.lower().endswith(".json"):
            continue

        infile = os.path.join(INPUT_FOLDER, fname)
        try:
            augmented = process_file(infile)

            outfile = os.path.join(OUTPUT_FOLDER, fname)
            with open(outfile, "w", encoding="utf-8") as out_f:
                json.dump(augmented, out_f, ensure_ascii=False, indent=2)

            print(f"Processed: {fname}")
        except Exception as e:
            print(f"Error processing {infile}: {e}")


if __name__ == "__main__":
    main()
