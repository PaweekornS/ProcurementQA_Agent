import json
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from run import match_legal_section, match_category

def recalculate_file(json_path: str, stats_path: str = None):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    total = len(cases)
    sec_hits = 0
    cat_hits = 0

    for case in cases:
        exp_sec = case.get("expected_section", [])
        pred_laws = case.get("predicted_laws", [])
        
        is_sec_hit = False
        for ts in exp_sec:
            ts_clean = str(ts).strip()
            if not ts_clean:
                continue
            for pl in pred_laws:
                if match_legal_section(ts_clean, str(pl)):
                    is_sec_hit = True
                    break
            if is_sec_hit:
                break
        
        case["is_section_hit"] = is_sec_hit
        if is_sec_hit:
            sec_hits += 1

        exp_cat = case.get("expected_category", [])
        pred_cat = case.get("predicted_category", [])
        is_cat_hit = False
        for tc in exp_cat:
            tc_clean = str(tc).strip()
            if not tc_clean:
                continue
            for pc in pred_cat:
                if match_category(tc_clean, str(pc)):
                    is_cat_hit = True
                    break
            if is_cat_hit:
                break
        
        case["is_category_hit"] = is_cat_hit
        if is_cat_hit:
            cat_hits += 1

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    sec_rate = (sec_hits / total * 100) if total > 0 else 0.0
    cat_rate = (cat_hits / total * 100) if total > 0 else 0.0

    print(f"Total Cases: {total}")
    print(f"Section Hits: {sec_hits}/{total} ({sec_rate:.1f}%)")
    print(f"Category Hits: {cat_hits}/{total} ({cat_rate:.1f}%)")

    if stats_path and os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        stats["section_hits"] = sec_hits
        stats["section_hit_rate"] = sec_rate
        stats["category_hits"] = cat_hits
        stats["category_hit_rate"] = cat_rate
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Updated {stats_path}")

if __name__ == "__main__":
    combined_file = sys.argv[1] if len(sys.argv) > 1 else "./outputs/THAI/openrouter_results_combined.json"
    stats_file = sys.argv[2] if len(sys.argv) > 2 else "./outputs/THAI/openrouter_stats.json"
    recalculate_file(combined_file, stats_file)
