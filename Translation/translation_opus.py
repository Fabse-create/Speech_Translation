import os
import json
import time
import ctranslate2
from transformers import AutoTokenizer

INPUT_FOLDER  = r"D:\Speech_translation\SpeechAccessibility_2025-11-02_Train_Only_Json"
OUTPUT_FOLDER = r"D:\Speech_translation\opus_fast_json"

# Pfad zum konvertierten CT2-Modellordner
CT2_MODEL_DIR = r"D:\models\opus-mt-en-de-ct2"

# Tuning
BATCH_SIZE = 64          # 32/64 sind oft gut auf CPU
BEAM_SIZE = 1            # 1 = schnell (greedy). 4 = bessere Qualität, langsamer
INTRA_THREADS = max(1, os.cpu_count() - 1)  # nutzt CPU-Kerne
INTER_THREADS = 1

def normalize_spaces(s: str) -> str:
    return " ".join(s.split()).strip()

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    translator = ctranslate2.Translator(
        CT2_MODEL_DIR,
        device="cpu",
        intra_threads=INTRA_THREADS,
        inter_threads=INTER_THREADS,
    )

    # Cache gegen Duplikate (spart massiv Zeit, falls viele Sätze mehrfach vorkommen)
    cache = {}

    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".json")]
    total_files = len(files)

    start = time.time()
    total_sentences_translated = 0

    for idx_file, fname in enumerate(files, 1):
        infile = os.path.join(INPUT_FOLDER, fname)

        with open(infile, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "Files" in data:
            # Sammeln, deduplizieren, batch-übersetzen
            pending_texts = []
            pending_positions = []  # (entry_index)

            for i, entry in enumerate(data["Files"]):
                prompt = entry.get("Prompt", {})
                transcript = prompt.get("Transcript", "")

                if not transcript:
                    continue
                if "Translation" in prompt and prompt["Translation"]:
                    continue

                t_norm = normalize_spaces(transcript)

                # Cache-Hit
                if t_norm in cache:
                    data["Files"][i]["Prompt"]["Translation"] = cache[t_norm]
                    continue

                pending_texts.append(t_norm)
                pending_positions.append(i)

            # Batch translation
            for b in range(0, len(pending_texts), BATCH_SIZE):
                batch_texts = pending_texts[b:b+BATCH_SIZE]
                batch_pos = pending_positions[b:b+BATCH_SIZE]

                # Tokenize -> tokens für CT2
                tok = tokenizer(batch_texts, padding=True, truncation=True, max_length=512)
                input_ids = tok["input_ids"]

                token_batches = []
                for ids in input_ids:
                    # convert ids -> tokens, pads entfernen
                    tokens = tokenizer.convert_ids_to_tokens(ids)
                    tokens = [t for t in tokens if t != tokenizer.pad_token]
                    token_batches.append(tokens)

                results = translator.translate_batch(
                    token_batches,
                    beam_size=BEAM_SIZE,
                    max_decoding_length=256,
                )

                # Decode
                for pos, res, src_text in zip(batch_pos, results, batch_texts):
                    out_tokens = res.hypotheses[0]
                    out_ids = tokenizer.convert_tokens_to_ids(out_tokens)
                    translation = tokenizer.decode(out_ids, skip_special_tokens=True)
                    translation = normalize_spaces(translation)

                    data["Files"][pos]["Prompt"]["Translation"] = translation
                    cache[src_text] = translation
                    total_sentences_translated += 1

        outfile = os.path.join(OUTPUT_FOLDER, fname)
        with open(outfile, "w", encoding="utf-8") as out_f:
            json.dump(data, out_f, ensure_ascii=False, indent=2)

        if idx_file % 10 == 0 or idx_file == total_files:
            elapsed_min = (time.time() - start) / 60
            print(f"{idx_file}/{total_files} files | translated: {total_sentences_translated:,} | {elapsed_min:.1f} min elapsed")

    print("Done.")

if __name__ == "__main__":
    main()
