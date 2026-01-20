import os
import json
from collections import defaultdict

INPUT_FOLDER = r"D:\Speech_translation\SpeechAccessibility_2025-11-02_Train_Only_Json_downsampled_500k"
# INPUT_FOLDER = r"D:\Speech_translation\SpeechAccessibility_2025-11-02_Train_Only_Json"

def norm(s: str) -> str:
    return " ".join((s or "").split()).strip()

def get_transcript(entry: dict) -> str:
    # NOTE: this version only reads Prompt.* like your snippet
    prompt = entry.get("Prompt", {})
    if isinstance(prompt, dict):
        return prompt.get("Transcript")
    return ""

def main():
    speakers_per_et = defaultdict(int)
    unique_sentences_per_et = defaultdict(set)

    # duplicates-included totals (what translation costs if you translate every occurrence)
    total_chars_per_et = defaultdict(int)
    total_occurs_per_et = defaultdict(int)

    unreadable = 0
    missing_et = 0

    for fname in os.listdir(INPUT_FOLDER):
        if not fname.lower().endswith(".json"):
            continue

        path = os.path.join(INPUT_FOLDER, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            unreadable += 1
            continue

        et = norm(data.get("Etiology"))
        if not et:
            et = "UNKNOWN"
            missing_et += 1

        speakers_per_et[et] += 1

        for entry in data.get("Files", []):
            s = norm(get_transcript(entry))
            if not s:
                continue

            unique_sentences_per_et[et].add(s)

            total_chars_per_et[et] += len(s)
            total_occurs_per_et[et] += 1

    # Build rows (now includes TotalOccur + TotalChars)
    rows = []
    for et in speakers_per_et:
        uniq = unique_sentences_per_et.get(et, set())
        uniq_sent = len(uniq)
        uniq_chars = sum(len(x) for x in uniq)

        tot_occ = total_occurs_per_et.get(et, 0)
        tot_chars = total_chars_per_et.get(et, 0)

        rows.append({
            "Etiology": et,
            "Speakers": speakers_per_et[et],
            "TotalOccur": tot_occ,
            "TotalChars": tot_chars,
            "UniqueSentences": uniq_sent,
            "UniqueChars": uniq_chars,
        })

    # Global totals
    total_speakers = sum(r["Speakers"] for r in rows)
    total_occurs_all = sum(r["TotalOccur"] for r in rows)
    total_chars_all = sum(r["TotalChars"] for r in rows)

    total_unique_sent = sum(r["UniqueSentences"] for r in rows)  # per-et sum; can double-count across etiologies
    total_unique_chars = sum(r["UniqueChars"] for r in rows)     # per-et sum; can double-count across etiologies

    # Ratio based on translation cost share
    for r in rows:
        r["Ratio%"] = (100.0 * r["TotalChars"] / total_chars_all) if total_chars_all else 0.0

    # Sort (change if you want)
    rows.sort(key=lambda r: r["Speakers"], reverse=True)

    # SUM row
    rows.append({
        "Etiology": "SUM",
        "Speakers": total_speakers,
        "TotalOccur": total_occurs_all,
        "TotalChars": total_chars_all,
        "UniqueSentences": total_unique_sent,
        "UniqueChars": total_unique_chars,
        "Ratio%": 100.0 if total_chars_all else 0.0,
    })

    def fmt_int(x: int) -> str:
        return f"{x:,}"

    headers = ["Etiology", "Speakers", "TotalOccur", "TotalChars", "UniqueSentences", "UniqueChars", "Ratio%"]

    # Compute column widths
    col_width = {h: len(h) for h in headers}
    for r in rows:
        col_width["Etiology"] = max(col_width["Etiology"], len(str(r["Etiology"])))
        for h in ["Speakers", "TotalOccur", "TotalChars", "UniqueSentences", "UniqueChars"]:
            col_width[h] = max(col_width[h], len(fmt_int(int(r[h]))))
        col_width["Ratio%"] = max(col_width["Ratio%"], len(f"{r['Ratio%']:.1f}"))

    # Print
    print(f"Unreadable JSON: {unreadable}")
    print(f"Missing Etiology: {missing_et}\n")

    header_line = (
        f"{'Etiology':<{col_width['Etiology']}}  "
        f"{'Speakers':>{col_width['Speakers']}}  "
        f"{'TotalOccur':>{col_width['TotalOccur']}}  "
        f"{'TotalChars':>{col_width['TotalChars']}}  "
        f"{'UniqueSentences':>{col_width['UniqueSentences']}}  "
        f"{'UniqueChars':>{col_width['UniqueChars']}}  "
        f"{'Ratio%':>{col_width['Ratio%']}}"
    )
    print(header_line)
    print("-" * len(header_line))

    for r in rows:
        print(
            f"{str(r['Etiology']):<{col_width['Etiology']}}  "
            f"{fmt_int(int(r['Speakers'])):>{col_width['Speakers']}}  "
            f"{fmt_int(int(r['TotalOccur'])):>{col_width['TotalOccur']}}  "
            f"{fmt_int(int(r['TotalChars'])):>{col_width['TotalChars']}}  "
            f"{fmt_int(int(r['UniqueSentences'])):>{col_width['UniqueSentences']}}  "
            f"{fmt_int(int(r['UniqueChars'])):>{col_width['UniqueChars']}}  "
            f"{r['Ratio%']:{col_width['Ratio%']}.1f}"
        )

if __name__ == "__main__":
    main()
