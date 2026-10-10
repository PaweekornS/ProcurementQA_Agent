# -*- coding: utf-8 -*-
"""
evaluation/evaluate_rag_triad.py

Production Evaluation Suite for ProcurementQA Agent.
Evaluates both Retrieval and Generation layers:

1. Retrieval Layer Metrics:
   - Precision@k
   - Recall@k
   - Strict Hit Rate (Doc + Section)

2. Generation Layer Metrics (Custom LLM-as-a-Judge + Deterministic Matching):
   - Faithfulness (Grounding to retrieved context without hallucination)
   - Completeness (Coverage of critical conditions vs Ground Truth)
   - Answer Relevancy (Direct alignment to user inquiry)
   - Citation Precision, Recall, and F1 (Deterministic Section/Clause matching)
"""

import os
import sys
import re
import json
import argparse
import time
import concurrent.futures
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from evaluation.evidence import evidence_metrics  # noqa: E402

TH_TO_AR = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def normalize_legal_text(text: str) -> str:
    if not text:
        return ""
    t = str(text).translate(TH_TO_AR)
    t = re.sub(r"[+_]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()


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
        if re.search(rf"ข้อ\s*[^\n|]*?\b{clause_num}\b", cand_norm) or re.search(rf"\bข้อ\s*{clause_num}\b", cand_norm):
            return True

    ch_exp = re.search(r"หมวด\s*(\d+)", exp_norm)
    if ch_exp:
        ch_num = ch_exp.group(1)
        if re.search(rf"หมวด\s*[^\n|]*?\b{ch_num}\b", cand_norm) or re.search(rf"\bหมวด\s*{ch_num}\b", cand_norm):
            return True

    core_exp = re.sub(r"(ประกาศ|แบบสัญญา|เรื่อง|หลักเกณฑ์|ระเบียบฯ|ระเบียบ|พ\.ร\.บ\.|พรบ|กฎกระทรวง|ข้อ|มาตรา)", "", exp_norm).strip()
    if len(core_exp) >= 4 and core_exp in cand_norm:
        return True

    return False


def match_document(expected_file: str, candidate_text: str) -> bool:
    if not expected_file or not candidate_text:
        return False
    base_name = os.path.basename(expected_file)
    base_title = re.sub(r"\.md$", "", base_name, flags=re.IGNORECASE).strip()
    
    cand_norm = normalize_legal_text(candidate_text)
    base_norm = normalize_legal_text(base_title)
    
    if base_norm in cand_norm or cand_norm in base_norm:
        return True
        
    m = re.search(r"(พระราชบัญญัติ[^\n|]+?2560|ระเบียบกระทรวงการคลัง[^\n|]+?2560|กฎกระทรวง[^\n|]+?2561|กฎกระทรวง[^\n|]+?2563|ข้อตกลงคุณธรรม)", base_norm)
    if m and m.group(1) in cand_norm:
        return True
    return False


def match_doc_and_section(expected_file: str, expected_section: str, candidate: str) -> bool:
    if not match_legal_section(expected_section, candidate):
        return False
    if not expected_file:
        return True
    return match_document(expected_file, candidate)


def _unit_ref(text: str) -> Optional[Tuple[str, int]]:
    """('มาตรา', 56) / ('ข้อ', 27) for the first section/clause reference in `text`, else None."""
    m = re.search(r"(มาตรา|ข้อ)\s*(\d+)", normalize_legal_text(text))
    return (m.group(1), int(m.group(2))) if m else None


def _same_document(expected_file: str, chunk_doc: str) -> bool:
    """
    Expected document (OCR path) vs the chunk's document title. Corpus titles can be truncated
    (~80 chars), so containment is checked both ways; the reverse direction requires a long
    enough title to avoid matching a generic prefix.
    """
    if not expected_file:
        return True
    exp = re.sub(r"\s+", "", normalize_legal_text(re.sub(r"\.md$", "", os.path.basename(expected_file), flags=re.IGNORECASE)))
    doc = re.sub(r"\s+", "", normalize_legal_text(chunk_doc))
    if not exp or not doc:
        return False
    return exp in doc or (len(doc) >= 20 and doc in exp)


def match_chunk(pair: Dict[str, str], item: Any) -> bool:
    """
    A retrieved chunk is relevant to a ground-truth (document, section) pair when the chunk's
    own label is that section of that document. Chunks labelled "<document> | <มาตรา/ข้อ N>"
    are matched structurally; a chunk merely *mentioning* "ข้อ N" in its body does not count.
    Unlabelled chunks fall back to the legacy text matcher.
    """
    entry = (item.get("law_entry") or item.get("entry") or "") if isinstance(item, dict) else str(item)
    if "|" not in entry:
        text = entry
        if isinstance(item, dict):
            text = f"{entry} {item.get('snippet', '')} {item.get('text', '')}".strip()
        return match_doc_and_section(pair.get("doc", ""), pair.get("section", ""), text)

    chunk_doc, _, label = entry.partition("|")
    if not _same_document(pair.get("doc", ""), chunk_doc):
        return False
    section = pair.get("section", "")
    expected_ref = _unit_ref(section)
    if expected_ref is not None:
        return _unit_ref(label) == expected_ref
    if "หมวด" in section or _SUB_ITEM.match(normalize_legal_text(section)):
        # Chapter-level or unresolved sub-item references cannot be read from chunk labels
        text = f"{entry} {item.get('snippet', '')} {item.get('text', '')}" if isinstance(item, dict) else entry
        return match_doc_and_section(pair.get("doc", ""), section, text)
    # Document-level ground truth (e.g. "ประกาศกรมบัญชีกลาง เรื่อง ..."): the document match suffices
    return bool(pair.get("doc"))


_SUB_ITEM = re.compile(r"^\s*(\(\s*[\dก-ฮ]+\s*\)\s*)+$")


def resolve_expected_pairs(pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Ground truth sometimes splits one citation into two pairs, e.g. "มาตรา ๙๖ วรรคหนึ่ง (๒)" followed
    by a bare "(๕)" meaning มาตรา ๙๖ (๕). A bare sub-item inherits the section of the preceding
    pair from the same document.
    """
    resolved, last_ref_by_doc = [], {}
    for pair in pairs or []:
        pair = dict(pair)
        doc, section = pair.get("doc", ""), pair.get("section", "")
        ref = _unit_ref(section)
        if ref is not None:
            last_ref_by_doc[doc] = ref
        elif _SUB_ITEM.match(normalize_legal_text(section)) and doc in last_ref_by_doc:
            kind, num = last_ref_by_doc[doc]
            pair["section"] = f"{kind} {num} {section.strip()}"
        resolved.append(pair)
    return resolved


# ==============================================================================
# 1. RETRIEVAL LAYER METRICS (STANDARD INFORMATION RETRIEVAL)
# ==============================================================================

def compute_retrieval_metrics(
    retrieved_items: List[Any],
    expected_pairs: List[Dict[str, str]],
    ground_truth_sections: Optional[List[str]] = None,
    k: int = 5
) -> Dict[str, float]:
    """
    Computes standard Information Retrieval metrics for top-k retrieved evidence chunks:
    1. Chunk Precision@k: (Relevant chunks in top-k) / k
    2. Ground Truth Recall@k: (Covered GT laws in top-k) / (Total GT laws)
    3. Strict Hit@k: 1.0 if at least one target document + section is retrieved in top-k, else 0.0
    """
    # Build list of target pairs if not explicitly provided
    targets = resolve_expected_pairs(expected_pairs)
    if not targets and ground_truth_sections:
        targets = [{"doc": "", "section": s} for s in ground_truth_sections]

    if not targets:
        return {"precision_at_k": 1.0, "adjusted_precision_at_k": 1.0, "recall_at_k": 1.0, "hit_at_k": 1.0, "mrr_at_k": 1.0}

    top_k_chunks = (retrieved_items or [])[:k]
    actual_k = max(len(top_k_chunks), 1)

    relevant_chunks = 0
    covered_gt_indices = set()
    first_relevant_rank = None

    for rank_idx, item in enumerate(top_k_chunks, 1):
        is_chunk_relevant = False
        for p_idx, pair in enumerate(targets):
            if match_chunk(pair, item):
                is_chunk_relevant = True
                covered_gt_indices.add(p_idx)

        if is_chunk_relevant:
            relevant_chunks += 1
            if first_relevant_rank is None:
                first_relevant_rank = rank_idx

    precision = relevant_chunks / float(k)
    adj_precision = relevant_chunks / float(min(k, len(targets)))
    recall = len(covered_gt_indices) / float(len(targets))
    hit = 1.0 if relevant_chunks > 0 else 0.0
    mrr = (1.0 / first_relevant_rank) if first_relevant_rank is not None else 0.0

    return {
        "precision_at_k": round(precision, 4),
        "adjusted_precision_at_k": round(min(adj_precision, 1.0), 4),
        "recall_at_k": round(recall, 4),
        "hit_at_k": round(hit, 4),
        "mrr_at_k": round(mrr, 4)
    }


def extract_sections(text: str) -> List[str]:
    """Extract standard section and clause identifiers from text (used for citation matching)."""
    if not text:
        return []
    norm = normalize_legal_text(text)
    identifiers = []
    for m in re.findall(r"(?:มาตรา|ม\.)\s*(\d+)", norm):
        identifiers.append(f"มาตรา {m}")
    for m in re.findall(r"(?:ข้อ|ระเบียบฯ\s*ข้อ)\s*(\d+)", norm):
        identifiers.append(f"ข้อ {m}")
    return list(dict.fromkeys(identifiers))


# ==============================================================================
# 2. GENERATION LAYER METRICS (DETERMINISTIC + LLM-AS-A-JUDGE)
# ==============================================================================

def compute_citation_f1(
    generated_text: str,
    ground_truth_sections: List[str]
) -> Dict[str, float]:
    """
    Deterministic Precision, Recall, and F1 calculation for legal citations.
    """
    gt_secs = set()
    for s in ground_truth_sections or []:
        gt_secs.update(extract_sections(s))

    if not gt_secs:
        return {"citation_precision": 1.0, "citation_recall": 1.0, "citation_f1": 1.0}

    cand_secs = set(extract_sections(generated_text))
    if not cand_secs:
        return {"citation_precision": 0.0, "citation_recall": 0.0, "citation_f1": 0.0}

    hits = gt_secs.intersection(cand_secs)
    recall = len(hits) / len(gt_secs)
    precision = len(hits) / len(cand_secs)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "citation_precision": round(precision, 4),
        "citation_recall": round(recall, 4),
        "citation_f1": round(f1, 4)
    }


JUDGE_EVAL_PROMPT = """คุณคือผู้เชี่ยวชาญตรวจสอบความถูกต้องของระบบถามตอบกฎหมายจัดซื้อจัดจ้างภาครัฐ (AI Legal Evaluation Judge)
หน้าที่ของคุณคือประเมินคำตอบของระบบ (Candidate Answer) เทียบกับคำถาม (Question), บริบทที่ค้นมาได้ (Contexts), และเฉลย (Ground Truth)

เกณฑ์การให้คะแนน (0.0 ถึง 1.0):

1. **Faithfulness (ความซื่อสัตย์ต่อหลักฐาน ไม่มโน):**
   - คำตอบและเลขมาตรา มีระบุหรืออิงจากหลักฐานใน Contexts จริงหรือไม่
   - (ข้อสังเกตสำคัญ: หากใน Contexts มีบทบัญญัติหลักหรือเนื้อความของมาตรานั้น แล้วคำตอบระบุวรรค/อนุมาตรา/ข้อย่อย (เช่น มาตรา 56 วรรคหนึ่ง (2) (ข) หรือ ข้อ 79) ที่สอดคล้องกับหลักการ ให้ถือว่าได้ 1.0 ไม่ถือว่าแต่งเลขมาตราขึ้นเอง)
   - 1.0 = อิงจากหลักฐานและตัวบทกฎหมายใน Contexts ชัดเจน ไม่มีเลขมาตราที่แต่งขึ้นเอง
   - 0.5 = มีเนื้อหาหรือเลขมาตราส่วนใหญ่ในบริบท แต่มีบางส่วนไม่ปรากฏในหลักฐาน
   - 0.0 = แต่งเลขมาตราหรือเนื้อหาขัดแย้งกับหลักฐานทั้งหมด

2. **Completeness (ความครบถ้วนสมบูรณ์ของเนื้อหา):**
   - คำตอบครอบคลุมประเด็น ข้อยกเว้น และเงื่อนไขที่ระบุไว้ใน Ground Truth หรือไม่
   - (ข้อสังเกต: หากคำตอบครอบคลุม Ground Truth ครบถ้วนแล้ว และมีการอธิบายขั้นตอนหรือหลักเกณฑ์ทางกฎหมายเสริมอย่างถูกต้อง ให้ถือว่าได้ 1.0 ไม่หักคะแนน)
   - 1.0 = ครอบคลุมประเด็นหลักและเงื่อนไขของ Ground Truth ครบถ้วน
   - 0.7 = ตอบถูกประเด็นหลักแต่ขาดเงื่อนไขสำคัญบางส่วน
   - 0.3 = ตอบไม่ครบอย่างมีนัยสำคัญ
   - 0.0 = ตอบผิดหรือไม่ตอบ

3. **Answer Relevancy (ความตรงประเด็น):**
   - คำตอบตอบตรงกับสิ่งที่ผู้ใช้ถามใน Question หรือไม่
   - 1.0 = ตอบตรงประเด็นคำถามชัดเจน
   - 0.5 = มีการตอบนอกเรื่องหรืออ้อมค้อม
   - 0.0 = ตอบไม่ตรงคำถาม

ข้อมูลสำหรับการประเมิน:
[Question]: {question}
[Contexts]: {contexts}
[Ground Truth]: {ground_truth}
[Candidate Answer]: {candidate_answer}

จงตอบเป็น JSON รูปแบบนี้เท่านั้น:
{{
  "faithfulness": 1.0,
  "completeness": 1.0,
  "answer_relevancy": 1.0,
  "rationale": "คำอธิบายสั้นๆ ภาษาไทย"
}}
"""


class GenerationEvaluator:
    """Evaluates the Generation Layer via deterministic metrics and LLM-as-a-Judge."""

    def __init__(self, chatbot=None, model: Optional[str] = None):
        if chatbot is not None:
            self.chatbot = chatbot
        else:
            try:
                from core.models.openrouter import OpenRouterChatbot
                self.chatbot = OpenRouterChatbot(model_name=model)
            except Exception as e:
                print(f"[Warning] Failed to initialize OpenRouterChatbot for Judge: {e}")
                self.chatbot = None

    def evaluate_sample(
        self,
        question: str,
        candidate_answer: str,
        ground_truth: str,
        ground_truth_sections: List[str],
        retrieved_contexts: List[Any],
        use_llm_judge: bool = True
    ) -> Dict[str, Any]:
        """Evaluates a single sample."""
        # 1. Deterministic Citation Metrics
        citation_metrics = compute_citation_f1(candidate_answer, ground_truth_sections)

        # 2. LLM-as-a-Judge Metrics
        llm_metrics = {
            "faithfulness": None,
            "completeness": None,
            "answer_relevancy": None,
            "rationale": "Deterministic mode only (LLM Judge skipped)"
        }

        if use_llm_judge and self.chatbot:
            try:
                ctx_parts = []
                for c in (retrieved_contexts or [])[:5]:
                    if isinstance(c, dict):
                        txt = c.get("content") or c.get("snippet") or c.get("text") or str(c)
                    elif isinstance(c, str):
                        txt = c
                    else:
                        txt = str(c)
                    if txt and txt.strip():
                        ctx_parts.append(txt.strip())
                ctx_summary = "\n\n".join(ctx_parts) if ctx_parts else "ไม่มีบริบท"

                prompt = JUDGE_EVAL_PROMPT.format(
                    question=question,
                    contexts=ctx_summary[:3500],
                    ground_truth=ground_truth[:1500],
                    candidate_answer=candidate_answer[:1500]
                )
                resp_text = self.chatbot.generate_response(prompt, max_length=500, temperature=0.0)
                # Parse JSON block
                json_match = re.search(r"\{[\s\S]*\}", resp_text)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    llm_metrics = {
                        "faithfulness": float(parsed.get("faithfulness", 1.0)),
                        "completeness": float(parsed.get("completeness", 0.8)),
                        "answer_relevancy": float(parsed.get("answer_relevancy", 1.0)),
                        "rationale": parsed.get("rationale", "")
                    }
                else:
                    llm_metrics["rationale"] = f"LLM Judge returned non-JSON: {resp_text[:100]}"
            except Exception as e:
                llm_metrics["rationale"] = f"Judge Evaluation Error: {e}"

        return {
            **citation_metrics,
            **llm_metrics
        }


# ==============================================================================
# 3. BATCH BENCHMARK RUNNER
# ==============================================================================

def run_rag_triad_evaluation(
    results_file: str,
    test_cases_file: str,
    output_report_file: Optional[str] = None,
    sample_limit: Optional[int] = None,
    use_llm_judge: bool = True,
    workers: int = 5,
    top_k: int = 5,
    skip_no_law_found: bool = True,
    corpus_chunks: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs end-to-end evaluation on output results JSON from run.py.

    corpus_chunks: chunks.jsonl written by scripts/build_corpus.py. Result files only keep chunk
    labels, so the corpus is needed to (a) score datasets whose ground truth is given as evidence
    spans (general documents, tables) and (b) give the LLM judge full chunk text instead of
    250-character snippets."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    if not os.path.exists(test_cases_file):
        raise FileNotFoundError(f"Test cases file not found: {test_cases_file}")

    with open(results_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        pred_results = raw_data.get("results", raw_data.get("cases", raw_data.get("data", [])))
    else:
        pred_results = raw_data

    with open(test_cases_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # Build lookup map of ground truth cases by ID or question text
    gt_map = {}
    for idx, c in enumerate(test_cases):
        cid = str(c.get("id", idx))
        gt_map[cid] = c
        if c.get("fact"):
            gt_map[c["fact"].strip()] = c

    chunk_by_entry: Dict[str, Dict[str, str]] = {}
    if corpus_chunks:
        with open(corpus_chunks, "r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                chunk_by_entry[c["entry"]] = {"doc": c["doc_title"], "text": c["content"]}
        print(f"[*] Loaded {len(chunk_by_entry)} corpus chunks for evidence matching and judge context")
    if any(c.get("evidence") for c in test_cases) and not chunk_by_entry:
        print("[!] Dataset has evidence spans but no --corpus-chunks: evidence metrics fall back to 250-char snippets")

    def hydrate(item: Any) -> Dict[str, str]:
        entry = (item.get("law_entry") or item.get("entry") or "") if isinstance(item, dict) else str(item)
        hit = chunk_by_entry.get(entry)
        if hit:
            return hit
        doc = entry.partition("|")[0].strip()
        return {"doc": doc, "text": f"{entry} {item.get('snippet', '') if isinstance(item, dict) else ''}"}

    evaluator = GenerationEvaluator()
    items_to_eval = pred_results[:sample_limit] if sample_limit else pred_results
    total_count = len(items_to_eval)
    print(f"[*] Starting RAG Triad Evaluation on {total_count} samples (Workers: {workers}, LLM Judge: {use_llm_judge}, Top-K: {top_k}, Skip NO_LAW_FOUND: {skip_no_law_found})...")

    def process_item(item_tuple: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
        idx, pred = item_tuple
        q_text = pred.get("question", "") or pred.get("fact", "")
        cid = str(pred.get("caseId") or pred.get("id", idx))
        gt_case = gt_map.get(cid) or gt_map.get(q_text.strip()) or {}

        gt_sections = gt_case.get("laws", gt_case.get("law", []))
        gt_answer = gt_case.get("ground_truth", "") or gt_case.get("answer", "")
        cand_answer = pred.get("pred_direct_answer", "") or pred.get("direct_answer", "") or pred.get("response", "")

        pred_status = (
            pred.get("status")
            or pred.get("judge_result", {}).get("status")
            or pred.get("pred_status", "")
        )
        is_no_law_found = (
            pred_status == "NO_LAW_FOUND"
            or "ไม่พบข้อกฎหมาย" in cand_answer
            or "NO_LAW_FOUND" in cand_answer
        )

        evidence = pred.get("analysis", {}).get("top_retrieved_evidence", [])
        evidence_entries = [e.get("law_entry", "") for e in evidence if isinstance(e, dict) and e.get("law_entry")]
        evidence_snippets = [e.get("snippet", "") for e in evidence if isinstance(e, dict) and e.get("snippet")]

        retrieved_laws = (
            pred.get("retrieved_laws")
            or pred.get("used_laws")
            or evidence_entries
            or pred.get("predicted_laws", [])
        )

        # Expected pairs matching run.py
        expected_pairs = pred.get("expected_pairs") or gt_case.get("expected_pairs", [])

        # 1. Retrieval Layer Evaluation (Chunk Precision@K, Recall@K, Hit@K)
        # Rank over the full context the generator received when run.py recorded it; older
        # result files only carry the top-20 evidence, which caps every cut-off at 20.
        ranked_context = pred.get("analysis", {}).get("retrieved_context") or evidence or retrieved_laws
        ret_metrics = compute_retrieval_metrics(
            retrieved_items=ranked_context,
            expected_pairs=expected_pairs,
            ground_truth_sections=gt_sections,
            k=top_k
        )
        eval_cutoffs = sorted({5, 10, 15, 20, top_k})
        ret_metrics["by_k"] = {
            str(k): compute_retrieval_metrics(ranked_context, expected_pairs, gt_sections, k=k)
            for k in eval_cutoffs
        }
        # Context recall: share of ground-truth sections present anywhere in the generator's context
        ret_metrics["context_size"] = len(ranked_context)
        ret_metrics["context_recall"] = compute_retrieval_metrics(
            ranked_context, expected_pairs, gt_sections, k=max(len(ranked_context), 1)
        )["recall_at_k"]
        ret_metrics["full_context_available"] = bool(pred.get("analysis", {}).get("retrieved_context"))

        # Evidence-span relevance (general documents / tables): chunker-independent
        gt_evidence = gt_case.get("evidence") or []
        if gt_evidence:
            ranked_chunks = [hydrate(item) for item in ranked_context]
            ret_metrics["evidence"] = {
                str(k): evidence_metrics(gt_evidence, ranked_chunks, k) for k in eval_cutoffs
            }
            ret_metrics["evidence"]["context"] = evidence_metrics(gt_evidence, ranked_chunks, max(len(ranked_chunks), 1))
        ret_metrics["question_type"] = gt_case.get("question_type")

        # If NO_LAW_FOUND and skipping is enabled: skip LLM Judge & exclude from aggregate scoring
        if is_no_law_found and skip_no_law_found:
            return {
                "sample_index": idx + 1,
                "question": q_text,
                "answer": cand_answer,
                "status": "NO_LAW_FOUND",
                "skipped": True,
                "skip_reason": "NO_LAW_FOUND",
                "retrieval": ret_metrics,
                "generation": {
                    "faithfulness": None,
                    "completeness": None,
                    "answer_relevancy": None,
                    "citation_precision": 0.0,
                    "citation_recall": 0.0,
                    "citation_f1": 0.0,
                    "rationale": "Skipped evaluation because system declared NO_LAW_FOUND"
                }
            }

        # Build rich, authentic statutory context for the Judge:
        # 1. Decisive quotes that the model retrieved and cited
        # 2. Top retrieved evidence chunks
        judge_contexts = []
        for q in pred.get("decisive_quotes", []):
            if isinstance(q, dict) and q.get("quote"):
                judge_contexts.append(f"[{q.get('law', '')}]: {q.get('quote', '')}")

        for e in evidence:
            if isinstance(e, dict):
                entry = e.get("law_entry", "")
                snip = chunk_by_entry.get(entry, {}).get("text", "")[:2000] or e.get("snippet", "")
                if entry and not any(entry in jc for jc in judge_contexts):
                    judge_contexts.append(f"[{entry}]: {snip}")

        if not judge_contexts:
            judge_contexts = retrieved_laws

        gen_metrics = evaluator.evaluate_sample(
            question=q_text,
            candidate_answer=cand_answer + "\n(ข้อกฎหมายที่อ้างอิง: " + ", ".join(pred.get("predicted_laws", [])) + ")",
            ground_truth=gt_answer,
            ground_truth_sections=gt_sections,
            retrieved_contexts=judge_contexts,
            use_llm_judge=use_llm_judge
        )

        sample_res = {
            "sample_index": idx + 1,
            "question": q_text,
            "answer": cand_answer,
            "status": pred_status or "ANSWERED",
            "skipped": False,
            "retrieval": ret_metrics,
            "generation": gen_metrics
        }
        return sample_res

    evaluated_samples = []
    item_tuples = list(enumerate(items_to_eval))

    if workers > 1 and total_count > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_item, itm) for itm in item_tuples]
            with tqdm(total=total_count, desc="Evaluating Triad") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    evaluated_samples.append(res)
                    pbar.update(1)
        evaluated_samples.sort(key=lambda x: x["sample_index"])
    else:
        for itm in item_tuples:
            res = process_item(itm)
            evaluated_samples.append(res)
            idx = itm[0]
            if res.get("skipped"):
                print(f"  [{idx+1}/{total_count}] [SKIPPED] NO_LAW_FOUND")
            else:
                ret_metrics = res["retrieval"]
                gen_metrics = res["generation"]
                print(f"  [{idx+1}/{total_count}] Prec@{top_k}: {ret_metrics['precision_at_k']:.2f} | Recall@{top_k}: {ret_metrics['recall_at_k']:.2f} | Hit@{top_k}: {ret_metrics['hit_at_k']:.2f} | Faith: {gen_metrics['faithfulness'] if gen_metrics.get('faithfulness') is None else round(gen_metrics['faithfulness'], 2)}")

    # Split into valid (included in scoring) and skipped
    valid_samples = [s for s in evaluated_samples if not s.get("skipped")]
    skipped_count = len(evaluated_samples) - len(valid_samples)

    total_precision_k = [s["retrieval"]["precision_at_k"] for s in valid_samples]
    total_adj_precision_k = [s["retrieval"]["adjusted_precision_at_k"] for s in valid_samples]
    total_recall_k = [s["retrieval"]["recall_at_k"] for s in valid_samples]
    total_hit_k = [s["retrieval"]["hit_at_k"] for s in valid_samples]
    total_mrr_k = [s["retrieval"]["mrr_at_k"] for s in valid_samples]
    total_faithfulness = [s["generation"]["faithfulness"] for s in valid_samples if s["generation"].get("faithfulness") is not None]
    total_completeness = [s["generation"]["completeness"] for s in valid_samples if s["generation"].get("completeness") is not None]
    total_relevancy = [s["generation"]["answer_relevancy"] for s in valid_samples if s["generation"].get("answer_relevancy") is not None]

    def safe_avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    retrieval_summary = {
        f"mean_hit_at_{top_k}": safe_avg(total_hit_k),
        f"mean_mrr_at_{top_k}": safe_avg(total_mrr_k),
        f"mean_adjusted_precision_at_{top_k}": safe_avg(total_adj_precision_k),
        f"mean_precision_at_{top_k}": safe_avg(total_precision_k),
        f"mean_recall_at_{top_k}": safe_avg(total_recall_k),
        "evaluated_k": top_k
    }
    k_benchmarks = [5, 10, 15, 20]
    for k_val in k_benchmarks:
        k_str = str(k_val)
        rows = [s["retrieval"]["by_k"][k_str] for s in valid_samples if k_str in s["retrieval"].get("by_k", {})]
        retrieval_summary[f"at_{k_str}"] = {
            "hit": safe_avg([r["hit_at_k"] for r in rows]),
            "mrr": safe_avg([r["mrr_at_k"] for r in rows]),
            "recall": safe_avg([r["recall_at_k"] for r in rows]),
            "adjusted_precision": safe_avg([r["adjusted_precision_at_k"] for r in rows]),
            "precision": safe_avg([r["precision_at_k"] for r in rows]),
        }
    retrieval_summary["mean_context_recall"] = safe_avg([s["retrieval"].get("context_recall", 0.0) for s in valid_samples])
    retrieval_summary["mean_context_size"] = safe_avg([s["retrieval"].get("context_size", 0) for s in valid_samples])
    retrieval_summary["full_context_available"] = all(s["retrieval"].get("full_context_available") for s in valid_samples)

    with_evidence = [s for s in valid_samples if s["retrieval"].get("evidence")]
    if with_evidence:
        def evidence_summary(samples):
            out = {"n": len(samples)}
            for cut in [str(k) for k in sorted({5, 10, 15, 20, top_k})] + ["context"]:
                rows = [s["retrieval"]["evidence"][cut] for s in samples if cut in s["retrieval"]["evidence"]]
                out[f"at_{cut}"] = {m: safe_avg([r[m] for r in rows]) for m in ("recall", "hit", "all", "mrr", "precision")}
            gen = [s["generation"] for s in samples]
            for m in ("faithfulness", "completeness", "answer_relevancy"):
                out[m] = safe_avg([g[m] for g in gen if g.get(m) is not None])
            return out

        retrieval_summary["evidence"] = evidence_summary(with_evidence)
        types = sorted({s["retrieval"].get("question_type") or "unknown" for s in with_evidence})
        retrieval_summary["evidence_by_question_type"] = {
            t: evidence_summary([s for s in with_evidence if (s["retrieval"].get("question_type") or "unknown") == t])
            for t in types
        }

    summary_report = {
        "total_cases": total_count,
        "evaluated_cases": len(valid_samples),
        "skipped_no_law_found_cases": skipped_count,
        "retrieval_layer": retrieval_summary,
        "generation_layer": {
            "mean_faithfulness": safe_avg(total_faithfulness),
            "mean_completeness": safe_avg(total_completeness),
            "mean_answer_relevancy": safe_avg(total_relevancy),
        },
        "samples": evaluated_samples
    }

    if output_report_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_report_file)), exist_ok=True)
        with open(output_report_file, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, ensure_ascii=False, indent=2)
        print(f"[+] Evaluation Report saved to: {output_report_file}")

    print("\n" + "=" * 66)
    print("🎯 RAG TRIAD & LEGAL EVALUATION SUMMARY REPORT")
    print("=" * 66)
    print(f"📊 Evaluated: {len(valid_samples)}/{total_count} cases | Skipped (NO_LAW_FOUND): {skipped_count}")
    print("-" * 66)
    print("1. RETRIEVAL LAYER (Multi-Cutoff Benchmark: K = 5, 10, 15, 20):")
    print(f"   {'Cutoff':<8} | {'MRR@K':<10} | {'Recall@K':<12} | {'Adj Precision@K':<18}")
    print("   " + "-" * 60)
    for k_val in k_benchmarks:
        k_str = str(k_val)
        if f"at_{k_str}" in retrieval_summary:
            m = retrieval_summary[f"at_{k_str}"]
            print(f"   K={k_val:<5} | {m['mrr']*100:>8.2f}% | {m['recall']*100:>10.2f}% | {m['adjusted_precision']*100:>16.2f}%")
    print(f"   * Mean Context Recall: {retrieval_summary['mean_context_recall']*100:.2f}% (Average context pool: {retrieval_summary['mean_context_size']:.1f} chunks)")
    if "evidence" in retrieval_summary:
        print("   Evidence-span retrieval (by question type):")
        for t, m in [("all", retrieval_summary["evidence"])] + list(retrieval_summary["evidence_by_question_type"].items()):
            print(f"   {t:<16} n={m['n']:<3} recall@5={m['at_5']['recall']*100:6.2f}%  recall@10={m['at_10']['recall']*100:6.2f}%"
                  f"  context recall={m['at_context']['recall']*100:6.2f}%  faithfulness={m['faithfulness']*100:6.2f}%")
    print("-" * 66)
    print("2. GENERATION LAYER:")
    if use_llm_judge and total_faithfulness:
        print(f"   - Faithfulness (Grounding):           {summary_report['generation_layer']['mean_faithfulness'] * 100:.2f}%")
        print(f"   - Completeness:                      {summary_report['generation_layer']['mean_completeness'] * 100:.2f}%")
        print(f"   - Answer Relevancy:                  {summary_report['generation_layer']['mean_answer_relevancy'] * 100:.2f}%")
    else:
        print("   - LLM Judge:                         [Skipped with --no-llm-judge]")
    print("=" * 66)

    return summary_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG Triad: Retrieval & Generation Layers")
    parser.add_argument("results_pos", nargs="?", default=None, help="Optional positional path to predictions JSON from run.py")
    parser.add_argument("--results", default=None, help="Path to predictions JSON from run.py")
    parser.add_argument("--datasets", default="./datasets/crime_data_THAI_small.json", help="Path to ground truth dataset JSON")
    parser.add_argument("--output", default=None, help="Output path for evaluation report (defaults to outputs/<dataset>/<mode>_rag_triad_report.json)")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for quick smoke test")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent workers for LLM Judge evaluation")
    parser.add_argument("--no-llm-judge", action="store_true", help="Skip LLM Judge and run deterministic evaluation only")
    parser.add_argument("--k", "-k", type=int, default=20, help="Number of evidence chunks to evaluate (default: 5)")
    parser.add_argument("--include-no-law-found", action="store_true", help="Include NO_LAW_FOUND cases in average score calculations (default is to skip them)")
    parser.add_argument("--corpus-chunks", default=None, help="chunks.jsonl from scripts/build_corpus.py: enables evidence-span metrics and full-text judge context")
    args = parser.parse_args()

    # 1. Resolve results_file safely
    results_file = args.results_pos or args.results

    # Safeguard: if user accidentally passed a results JSON file to --output without --results, redirect it as input
    if args.output and ("_results.json" in args.output) and not results_file:
        print(f"[*] Detected results file passed as --output: {args.output}. Using it as input --results.")
        results_file = args.output
        args.output = None

    if not results_file or not os.path.exists(results_file):
        candidates = [
            "./outputs/THAI/crag_results.json",
            "./outputs/THAI/agentic_results.json",
            "./outputs/THAI/openrouter_crag_results.json",
            "./outputs/THAI/openrouter_agentic_results.json",
            "./outputs/crag_results.json",
            "./outputs/agentic_results.json",
            "./outputs/THAI/openrouter_results.json",
            "./outputs/openrouter_crag_results.json",
            "./outputs/openrouter_results.json"
        ]
        for cand in candidates:
            if os.path.exists(cand):
                results_file = cand
                print(f"[*] Auto-detected results file: {results_file}")
                break

    if not results_file or not os.path.exists(results_file):
        raise FileNotFoundError(f"Predictions results file not found: {args.results or args.results_pos}")

    # 2. Resolve output_report_file safely (crag_triad_report.json / agentic_triad_report.json)
    base_name = os.path.basename(results_file).lower()
    results_dir = os.path.dirname(os.path.abspath(results_file))
    
    if "crag" in base_name:
        default_report_name = "crag_triad_report.json"
    elif "agentic" in base_name:
        default_report_name = "agentic_triad_report.json"
    else:
        stem = re.sub(r"(_results|\.json)$", "", base_name, flags=re.IGNORECASE)
        stem = re.sub(r"^(openrouter_|openai_|gemini_)", "", stem, flags=re.IGNORECASE)
        default_report_name = f"{stem}_triad_report.json"

    output_report_file = args.output
    if not output_report_file:
        output_report_file = os.path.join(results_dir, default_report_name)
    else:
        # Crucial safeguard: Never overwrite input results file
        if os.path.abspath(output_report_file) == os.path.abspath(results_file):
            print(f"[!] Warning: Specified --output matches input --results ({output_report_file}).")
            output_report_file = os.path.join(results_dir, default_report_name)
            print(f"[*] Diverted output to prevent data loss: {output_report_file}")

    run_rag_triad_evaluation(
        results_file=results_file,
        test_cases_file=args.datasets,
        output_report_file=output_report_file,
        sample_limit=args.limit,
        use_llm_judge=not args.no_llm_judge,
        workers=args.workers,
        top_k=args.k,
        skip_no_law_found=not args.include_no_law_found,
        corpus_chunks=args.corpus_chunks,
    )
