import json
import os
import re
import sys

TH_TO_AR = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

def normalize_legal_text(text: str) -> str:
    if not text:
        return ""
    t = str(text).translate(TH_TO_AR)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def match_legal_section(expected: str, candidate: str) -> bool:
    exp_norm = normalize_legal_text(expected)
    cand_norm = normalize_legal_text(candidate)
    
    if not exp_norm or not cand_norm:
        return False
        
    if exp_norm in cand_norm or cand_norm in exp_norm:
        return True
        
    m_exp = re.search(r"(มาตรา|ม\.)\s*(\d+)", exp_norm)
    if m_exp:
        sec_num = m_exp.group(2)
        if re.search(rf"(มาตรา|ม\.)\s*{sec_num}\b", cand_norm):
            return True

    k_exp = re.search(r"ข้อ\s*(\d+)", exp_norm)
    if k_exp:
        clause_num = k_exp.group(1)
        if re.search(rf"ข้อ\s*{clause_num}\b", cand_norm):
            return True

    ch_exp = re.search(r"หมวด\s*(\d+)", exp_norm)
    if ch_exp:
        ch_num = ch_exp.group(1)
        if re.search(rf"หมวด\s*{ch_num}\b", cand_norm):
            return True

    return False

def match_document(expected_file: str, candidate_text: str) -> bool:
    """Check if candidate text matches the expected source document title."""
    if not expected_file or not candidate_text:
        return False
    base_name = os.path.basename(expected_file)
    base_title = re.sub(r"\.md$", "", base_name, flags=re.IGNORECASE).strip()
    
    cand_norm = normalize_legal_text(candidate_text)
    base_norm = normalize_legal_text(base_title)
    
    if base_norm in cand_norm or cand_norm in base_norm:
        return True
        
    m = re.search(r"(พระราชบัญญัติ[^\n|]+?๒๕๖๐|พระราชบัญญัติ[^\n|]+?2560|ระเบียบกระทรวงการคลัง[^\n|]+?๒๕๖๐|ระเบียบกระทรวงการคลัง[^\n|]+?2560|กฎกระทรวง[^\n|]+?๒๕๖๑|กฎกระทรวง[^\n|]+?2561)", base_norm)
    if m and m.group(1) in cand_norm:
        return True
    return False

def match_doc_and_section(expected_file: str, expected_section: str, candidate: str) -> bool:
    """Strict AND condition: candidate must match BOTH the document and the section/clause."""
    if not match_legal_section(expected_section, candidate):
        return False
    if not expected_file:
        return True
    return match_document(expected_file, candidate)

def recalculate_file(json_path: str, stats_path: str = None, dataset_path: str = "./datasets/crime_data_THAI_small.json"):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    # Load ground truth dataset to get source_files if needed
    source_files_by_id = {}
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
            for item in gt_data:
                cid = item.get("id")
                s_files = item.get("source_files", [])
                if not s_files and item.get("source_file"):
                    s_files = [s.strip() for s in item["source_file"].split(";") if s.strip()]
                source_files_by_id[cid] = s_files

    with open(json_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    total = len(cases)
    doc_hits = 0
    sec_hits = 0
    both_hits = 0

    for case in cases:
        cid = case.get("id")
        # Keep single standard key: expected_section
        exp_sec = case.get("expected_section") or case.get("law_article") or case.get("laws", [])
        if not isinstance(exp_sec, list):
            exp_sec = [exp_sec] if exp_sec else []
        case["expected_section"] = exp_sec
        
        # Expected documents
        exp_docs = case.get("expected_documents") or source_files_by_id.get(cid, [])
        case["expected_documents"] = exp_docs

        # Remove redundant legacy key
        case.pop("law_article", None)

        # Remove all category keys
        case.pop("expected_category", None)
        case.pop("predicted_category", None)
        case.pop("is_category_hit", None)

        pred_laws = case.get("predicted_laws", [])
        
        # 1. Section Hit
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
        
        # 2. Document Hit
        is_doc_hit = False
        for ed in exp_docs:
            for pl in pred_laws:
                if match_document(ed, str(pl)):
                    is_doc_hit = True
                    break
            if is_doc_hit:
                break

        # 3. Strict AND Condition: Must hit BOTH Document AND Section
        is_both = False
        for ed in exp_docs:
            for ts in exp_sec:
                ts_clean = str(ts).strip()
                if not ts_clean:
                    continue
                for pl in pred_laws:
                    if match_doc_and_section(ed, ts_clean, str(pl)):
                        is_both = True
                        break
                if is_both:
                    break
            if is_both:
                break

        case["is_section_hit"] = is_sec_hit
        case["is_document_hit"] = is_doc_hit
        case["is_both_hit"] = is_both

        if is_sec_hit:
            sec_hits += 1
        if is_doc_hit:
            doc_hits += 1
        if is_both:
            both_hits += 1

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    doc_rate = (doc_hits / total * 100) if total > 0 else 0.0
    sec_rate = (sec_hits / total * 100) if total > 0 else 0.0
    both_rate = (both_hits / total * 100) if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Total Cases: {total}")
    print(f"Document Hit Rate (Recall@k): {doc_hits}/{total} ({doc_rate:.1f}%)")
    print(f"Section Hit Rate (Recall@k): {sec_hits}/{total} ({sec_rate:.1f}%)")
    print(f"Strict Hit Rate [Doc AND Section] (Recall@k): {both_hits}/{total} ({both_rate:.1f}%)")
    print(f"{'='*60}\n")

    if stats_path and os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        stats["total_cases"] = total
        stats["document_hits"] = doc_hits
        stats["document_hit_rate"] = doc_rate
        stats["section_hits"] = sec_hits
        stats["section_hit_rate"] = sec_rate
        stats["both_hits"] = both_hits
        stats["doc_and_section_hit_rate"] = both_rate
        stats.pop("category_hits", None)
        stats.pop("category_hit_rate", None)
        stats.pop("correct_count", None)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Updated {stats_path}")

if __name__ == "__main__":
    combined_file = sys.argv[1] if len(sys.argv) > 1 else "./outputs/THAI/openrouter_results_combined.json"
    stats_file = sys.argv[2] if len(sys.argv) > 2 else "./outputs/THAI/openrouter_stats.json"
    recalculate_file(combined_file, stats_file)
