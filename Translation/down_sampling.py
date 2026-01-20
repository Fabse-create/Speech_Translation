import os
import json
import math
import random
from collections import defaultdict

# =====================
# CONFIG
# =====================
INPUT_FOLDER  = r"D:\Speech_translation\SpeechAccessibility_2025-11-02_Train_Only_Json"
OUTPUT_FOLDER = r"D:\Speech_translation\SpeechAccessibility_2025-11-02_Train_Only_Json_downsampled_500k"

TOTAL_BUDGET_CHARS = 500_000
SLACK_CHARS = 10_000          # keep slack so you don't accidentally exceed
MIN_TRANSCRIPT_LEN = 1        # set to 3 or 5 to drop very short transcripts from budgeting/stats
SEED = 42

PROPORTIONAL_BY_ETIOLOGY = True
PICK_SMALLEST_FIRST = True

# =====================
# HELPERS
# =====================
def norm(s: str) -> str:
    return " ".join((s or "").split()).strip()

def get_transcript(entry: dict) -> str:
    prompt = entry.get("Prompt", {})
    if isinstance(prompt, dict):
        return prompt.get("Transcript")
    return ""

def speaker_transcript_list(data: dict) -> list[str]:
    """All transcript occurrences (duplicates kept)."""
    lst = []
    for entry in data.get("Files", []):
        s = norm(get_transcript(entry))
        if len(s) >= MIN_TRANSCRIPT_LEN:
            lst.append(s)
    return lst

def speaker_unique_set_from_list(lst: list[str]) -> set[str]:
    return set(lst)

def speaker_unique_char_cost_from_set(uniq: set[str]) -> int:
    # Budgeting stays char-based (as before), but computed from unique transcripts.
    return sum(len(x) for x in uniq)

def safe_load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def fmt_int(n: int) -> str:
    return f"{n:,}"

# =====================
# MAIN
# =====================
def main():
    random.seed(SEED)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1) Scan input, compute per-speaker costs and totals per Etiology
    # speakers: {path,fname,etiology,cost,uniq,transcripts,data}
    speakers = []
    total_cost_by_et = defaultdict(int)

    unreadable = 0
    for fname in os.listdir(INPUT_FOLDER):
        if not fname.lower().endswith(".json"):
            continue

        path = os.path.join(INPUT_FOLDER, fname)
        data = safe_load_json(path)
        if not isinstance(data, dict):
            unreadable += 1
            continue

        et = norm(data.get("Etiology")) or "UNKNOWN"

        transcripts = speaker_transcript_list(data)          # occurrences
        uniq = speaker_unique_set_from_list(transcripts)  # keep this for "new sentence" scoring
        cost = sum(len(x) for x in transcripts)           # IMPORTANT: duplicates count in budget


        speakers.append({
            "path": path,
            "fname": fname,
            "etiology": et,
            "cost": cost,
            "uniq": uniq,
            "transcripts": transcripts,
            "data": data,
        })
        total_cost_by_et[et] += cost

    if not speakers:
        print("No JSON files found/loaded.")
        return

    # 2) Compute per-etiology budgets
    target_total = max(0, TOTAL_BUDGET_CHARS - SLACK_CHARS)

    et_list = sorted(total_cost_by_et.keys())
    budgets = {}

    if PROPORTIONAL_BY_ETIOLOGY:
        grand_total = sum(total_cost_by_et.values())
        for et in et_list:
            share = (total_cost_by_et[et] / grand_total) if grand_total else 0.0
            budgets[et] = int(math.floor(target_total * share))
    else:
        per = target_total // len(et_list)
        for et in et_list:
            budgets[et] = per

    leftover = target_total - sum(budgets.values())
    if leftover > 0:
        biggest = max(et_list, key=lambda e: total_cost_by_et[e])
        budgets[biggest] += leftover

    # 3) Select speakers within each Etiology budget  (GREEDY: maximize new sentences per etiology)
    speakers_by_et = defaultdict(list)
    for sp in speakers:
        speakers_by_et[sp["etiology"]].append(sp)

    selected = []
    selected_cost_by_et = defaultdict(int)

    for et in et_list:
        group = speakers_by_et[et]
        budget = budgets[et]

        seen_et = set()   # transcripts already covered within this etiology selection

        remaining = group[:]
        while True:
            best = None
            best_new = 0
            best_add_cost = 0
            best_score = 0.0

            for sp in remaining:
                new_set = sp["uniq"] - seen_et
                if not new_set:
                    continue

                # incremental cost = chars of ONLY the new sentences this file contributes
                add_cost = sp["cost"]  # full file cost (duplicates included)

                if add_cost <= 0:
                    continue
                if selected_cost_by_et[et] + add_cost > budget:
                    continue

                # Option A (your request): maximize new sentences (tie-break by smaller cost)
                # score = float(len(new_set))
                # Option B (usually better): maximize new sentences per added char cost
                score = len(new_set) / add_cost

                if (score > best_score) or (score == best_score and len(new_set) > best_new) or (score == best_score and len(new_set) == best_new and add_cost < best_add_cost):
                    best = sp
                    best_new = len(new_set)
                    best_add_cost = add_cost
                    best_score = score

            if best is None:
                break

            # select it (duplicates-included cost)
            new_set = best["uniq"] - seen_et
            add_cost = best["cost"]  # full file cost (duplicates included)

            selected.append(best)
            selected_cost_by_et[et] += add_cost
            seen_et |= new_set
            remaining.remove(best)



    # 4) Top up globally if still under total budget
    selected_fnames = set(sp["fname"] for sp in selected)
    selected_total = sum(selected_cost_by_et.values())

    if selected_total < target_total:
        remaining = [sp for sp in speakers if sp["fname"] not in selected_fnames]
        if PICK_SMALLEST_FIRST:
            remaining = sorted(remaining, key=lambda x: (x["cost"], x["fname"]))
        else:
            random.shuffle(remaining)

        for sp in remaining:
            if selected_total + sp["cost"] <= target_total:
                selected.append(sp)
                selected_fnames.add(sp["fname"])
                selected_cost_by_et[sp["etiology"]] += sp["cost"]
                selected_total += sp["cost"]

    # 5) Write out selected JSON files unchanged
    for sp in selected:
        out_path = os.path.join(OUTPUT_FOLDER, sp["fname"])
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sp["data"], f, ensure_ascii=False, indent=2)

    # 6) Transcript diversity checks (selected set only)  <<< CHANGED
    # Overall occurrences + unique transcript strings
    overall_occurs = 0
    overall_unique = set()

    for sp in selected:
        overall_occurs += len(sp["transcripts"])
        overall_unique |= sp["uniq"]

    overall_unique_count = len(overall_unique)
    overall_dupe_occurs = overall_occurs - overall_unique_count
    overall_dupe_ratio = (overall_dupe_occurs / overall_occurs) if overall_occurs else 0.0

    # Per-etiology occurrences + unique
    trans_stats_by_et = {}
    per_et_unique_sets = {}

    for et in et_list:
        group = [sp for sp in selected if sp["etiology"] == et]
        occurs = 0
        uniq = set()
        for sp in group:
            occurs += len(sp["transcripts"])
            uniq |= sp["uniq"]

        per_et_unique_sets[et] = uniq
        uniq_count = len(uniq)
        dupe_occurs = occurs - uniq_count
        dupe_ratio = (dupe_occurs / occurs) if occurs else 0.0

        # (occurs, unique_count, duplicate_occurs, duplicate_ratio)
        trans_stats_by_et[et] = (occurs, uniq_count, dupe_occurs, dupe_ratio)

    # Cross-etiology duplicates:
    # if a transcript appears in multiple etiologies, sum(per-et unique) > overall unique
    sum_per_et_unique = sum(len(per_et_unique_sets[et]) for et in et_list)
    cross_et_dup_count = sum_per_et_unique - overall_unique_count

    # 7) Print report (with SUM row)  <<< CHANGED columns
    print(f"Input folder:  {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Unreadable JSON: {unreadable}\n")

    print(f"Target budget (with slack): {fmt_int(target_total)} chars")
    print(f"Selected total (sum per-et): {fmt_int(selected_total)} chars")
    print(f"Selected speakers: {fmt_int(len(selected))} / {fmt_int(len(speakers))}\n")

    # Budget + transcript-diversity table
    rows = []
    for et in et_list:
        b = budgets[et]
        s = selected_cost_by_et[et]
        util = (100.0 * s / b) if b else 0.0
        spk_count = sum(1 for x in selected if x["etiology"] == et)

        occurs, uniq_count, dupe_occ, dupe_ratio = trans_stats_by_et[et]

        rows.append({
            "Etiology": et,
            "Budget": b,
            "Selected": s,
            "Util%": util,
            "Speakers": spk_count,
            "Occur": occurs,
            "Unique": uniq_count,
            "DupeOccur": dupe_occ,
            "Dupe%": dupe_ratio * 100.0,
        })

    # Add SUM row
    sum_budget = sum(budgets.values())
    sum_selected = sum(selected_cost_by_et.values())
    sum_speakers = len(selected)

    sum_occurs = sum(r["Occur"] for r in rows)
    sum_unique = sum(r["Unique"] for r in rows)          # can exceed overall unique if cross-etiology duplicates exist
    sum_dupe_occ = sum(r["DupeOccur"] for r in rows)

    rows.append({
        "Etiology": "SUM",
        "Budget": sum_budget,
        "Selected": sum_selected,
        "Util%": (100.0 * sum_selected / sum_budget) if sum_budget else 0.0,
        "Speakers": sum_speakers,
        "Occur": sum_occurs,
        "Unique": sum_unique,
        "DupeOccur": sum_dupe_occ,
        "Dupe%": (100.0 * sum_dupe_occ / sum_occurs) if sum_occurs else 0.0,
    })

    headers = ["Etiology","Budget","Selected","Util%","Speakers","Occur","Unique","DupeOccur","Dupe%"]

    def cell(h, r):
        if h == "Etiology":
            return str(r[h])
        if h in ("Util%","Dupe%"):
            return f"{r[h]:.1f}"
        return fmt_int(int(r[h]))

    colw = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            colw[h] = max(colw[h], len(cell(h, r)))

    line = (
        f"{'Etiology':<{colw['Etiology']}}  "
        f"{'Budget':>{colw['Budget']}}  "
        f"{'Selected':>{colw['Selected']}}  "
        f"{'Util%':>{colw['Util%']}}  "
        f"{'Speakers':>{colw['Speakers']}}  "
        f"{'Occur':>{colw['Occur']}}  "
        f"{'Unique':>{colw['Unique']}}  "
        f"{'DupeOccur':>{colw['DupeOccur']}}  "
        f"{'Dupe%':>{colw['Dupe%']}}"
    )
    print(line)
    print("-" * len(line))

    for r in rows:
        print(
            f"{cell('Etiology', r):<{colw['Etiology']}}  "
            f"{cell('Budget', r):>{colw['Budget']}}  "
            f"{cell('Selected', r):>{colw['Selected']}}  "
            f"{cell('Util%', r):>{colw['Util%']}}  "
            f"{cell('Speakers', r):>{colw['Speakers']}}  "
            f"{cell('Occur', r):>{colw['Occur']}}  "
            f"{cell('Unique', r):>{colw['Unique']}}  "
            f"{cell('DupeOccur', r):>{colw['DupeOccur']}}  "
            f"{cell('Dupe%', r):>{colw['Dupe%']}}"
        )

    # 8) Overall + cross-etiology transcript duplication summary  <<< CHANGED
    print("\nTranscript duplication checks (selected only):")
    print(f"- Overall transcript occurrences: {fmt_int(overall_occurs)}")
    print(f"- Overall unique transcripts:      {fmt_int(overall_unique_count)}")
    print(f"- Overall duplicate occurrences:   {fmt_int(overall_dupe_occurs)} ({overall_dupe_ratio*100:.1f}%)")
    print(f"- Cross-etiology duplicate count (sum per-et unique - overall unique): {fmt_int(cross_et_dup_count)}")

if __name__ == "__main__":
    main()
