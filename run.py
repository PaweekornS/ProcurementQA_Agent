import argparse
import os
import sys
import json
from tqdm import tqdm
import multiprocessing
import threading
import concurrent.futures
import time
from typing import List, Dict, Any, Optional
import re

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from core.pipeline import ProcurementQAPipeline, PipelineConfig
from evaluation.evaluate_rag_triad import compute_retrieval_metrics


DATASET_ALIASES = {
    "statute": "statute_qa.json",
    "statute_sample": "statute_qa_sample.json",
    "general": "general_docs_qa.json",
}


def sanitize_dataset_name(dataset_input: str) -> str:
    """Output-folder name for a dataset: 'datasets/statute_qa.json' -> 'statute_qa', 'general' -> 'general_docs_qa'."""
    if not dataset_input:
        return "statute_qa"
    base = DATASET_ALIASES.get(dataset_input, os.path.basename(dataset_input.replace("\\", "/").rstrip("/")))
    return re.sub(r"\.json$", "", base, flags=re.IGNORECASE) or "statute_qa"


def load_test_cases(datasets: str, datasets_path: str = "./datasets") -> List[Dict[str, Any]]:
    """A dataset path, a file name under datasets_path, or an alias (statute, statute_sample, general)."""
    if os.path.isfile(datasets):
        case_file = datasets
    else:
        name = DATASET_ALIASES.get(datasets, datasets if datasets.endswith(".json") else f"{datasets}.json")
        case_file = os.path.join(datasets_path, name)
        if not os.path.isfile(case_file):
            raise FileNotFoundError(f"Test dataset not found: {case_file} (aliases: {', '.join(DATASET_ALIASES)})")
    with open(case_file, "r", encoding="utf-8") as f:
        return json.load(f)


TH_TO_AR = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def normalize_legal_text(text: str) -> str:
    if not text:
        return ""
    t = str(text).translate(TH_TO_AR)
    # Normalize URL encoding and special separators
    t = re.sub(r"[+_]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()


def match_legal_section(expected: str, candidate: str) -> bool:
    exp_norm = normalize_legal_text(expected)
    cand_norm = normalize_legal_text(candidate)
    
    if not exp_norm or not cand_norm:
        return False
        
    # 1. Direct normalized substring check
    if exp_norm in cand_norm or cand_norm in exp_norm:
        return True
        
    # 2. Section number matching (e.g. มาตรา 56 or ม. 56)
    m_exp = re.search(r"(มาตรา|ม\.)\s*(\d+)", exp_norm)
    if m_exp:
        sec_num = m_exp.group(2)
        if re.search(rf"(มาตรา|ม\.)\s*{sec_num}\b", cand_norm):
            return True

    # 3. Clause/Regulation matching (e.g. ข้อ 79, ข้อ 13)
    k_exp = re.search(r"ข้อ\s*(\d+)", exp_norm)
    if k_exp:
        clause_num = k_exp.group(1)
        # Direct "ข้อ 13" or part of a list "ข้อ 9, 13, 14"
        if re.search(rf"ข้อ\s*[^\n|]*?\b{clause_num}\b", cand_norm) or re.search(rf"\bข้อ\s*{clause_num}\b", cand_norm):
            return True

    # 4. Chapter matching (e.g. หมวด 7)
    ch_exp = re.search(r"หมวด\s*(\d+)", exp_norm)
    if ch_exp:
        ch_num = ch_exp.group(1)
        if re.search(rf"หมวด\s*[^\n|]*?\b{ch_num}\b", cand_norm) or re.search(rf"\bหมวด\s*{ch_num}\b", cand_norm):
            return True

    # 5. Non-numbered Core Title / Charter matching (e.g. ข้อตกลงคุณธรรม, integrity pact)
    core_exp = re.sub(r"(ประกาศ|แบบสัญญา|เรื่อง|หลักเกณฑ์|ระเบียบฯ|ระเบียบ|พ\.ร\.บ\.|พรบ|กฎกระทรวง|ข้อ|มาตรา)", "", exp_norm).strip()
    if len(core_exp) >= 4 and core_exp in cand_norm:
        return True

    return False


def match_document(expected_file: str, candidate_text: str) -> bool:
    """Check if candidate text matches the expected source document title."""
    if not expected_file or not candidate_text:
        return False
    # Clean file name from path like 'พรบ/พระราชบัญญัติการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. 2560.md'
    base_name = os.path.basename(expected_file)
    base_title = re.sub(r"\.md$", "", base_name, flags=re.IGNORECASE).strip()
    
    cand_norm = normalize_legal_text(candidate_text)
    base_norm = normalize_legal_text(base_title)
    
    if base_norm in cand_norm or cand_norm in base_norm:
        return True
        
    # Core keyword match (e.g. พระราชบัญญัติการจัดซื้อจัดจ้าง, ระเบียบกระทรวงการคลัง, กฎกระทรวง, ข้อตกลงคุณธรรม)
    m = re.search(r"(พระราชบัญญัติ[^\n|]+?2560|ระเบียบกระทรวงการคลัง[^\n|]+?2560|กฎกระทรวง[^\n|]+?2561|กฎกระทรวง[^\n|]+?2563|ข้อตกลงคุณธรรม)", base_norm)
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


def extract_case_analysis(case_res: List[Dict[str, Any]], max_evidence: int = 20, snippet_len: int = 250) -> Dict[str, Any]:
    """
    Extracts high-value diagnostic features and top retrieval evidence with clean snippets.
    Omits bloated multi-page raw law text dumps.
    """
    if not case_res or not isinstance(case_res, list) or len(case_res) == 0:
        return {}
    
    first_res = case_res[0]
    raw_feature = first_res.get("feature", {})
    
    from core.agent.decomposer import FEATURE_FIELDS
    extracted_features = {field: raw_feature.get(field, []) for field in FEATURE_FIELDS}
    
    evidence_list = []
    candidate_laws = first_res.get("used_laws") or first_res.get("retrieved_laws") or []
    
    for rank, law in enumerate(candidate_laws[:max_evidence], start=1):
        raw_desc = law.get("description", "")
        clean_desc = re.sub(r"\s+", " ", str(raw_desc)).strip()
        snippet = clean_desc[:snippet_len] + ("..." if len(clean_desc) > snippet_len else "")
        
        evidence_list.append({
            "rank": rank,
            "law_entry": law.get("entry", ""),
            "rerank_score": round(float(law.get("rerank_score", 1.0)), 4),
            "snippet": snippet,
        })
        
    crag_meta = first_res.get("crag_meta", {})
    # Full ranked context the generator received (labels only, to keep result files small) so the
    # evaluator can score retrieval against what generation actually used, not just the top 20.
    retrieved_context = [
        {"rank": rank, "law_entry": law.get("entry", ""), "rerank_score": round(float(law.get("rerank_score", 0.0)), 4)}
        for rank, law in enumerate(candidate_laws, start=1)
    ]

    return {
        "extracted_features": extracted_features,
        "top_retrieved_evidence": evidence_list,
        "retrieved_context": retrieved_context,
        "crag_meta": crag_meta,
    }


def process_cases_worker(
    cases: List[Dict[str, Any]],
    config_dict: Dict[str, Any],
    device: str,
    output_file: str,
    model_name: str
):
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    config = PipelineConfig.from_dict(config_dict)
    
    config.model.device = device
    config.model.model_name = model_name

    rag = ProcurementQAPipeline(config=config)
    
    results = []
    section_hits = 0
    document_hits = 0
    both_hits = 0
    
    try:
        for case in tqdm(cases, desc=f"Processing on {device} with {model_name}"):
            question = case.get("question", "")
            true_section = case.get("expected_sections", [])
            ground_truth = case.get("ground_truth", "")
            
            case_res = rag.analyze_case(case)
            
            pred_answer = ""
            pred_direct_answer = ""
            pred_laws = []
            exceptions = ""
            
            pred_quotes = []
            
            if case_res and isinstance(case_res, list) and len(case_res) > 0:
                answer_result = case_res[0].get("answer_result", {})
                pred_status = answer_result.get("status", "COMPLIANT")
                pred_direct_answer = answer_result.get("direct_answer", "")
                pred_quotes = answer_result.get("decisive_quotes", [])
                pred_laws = list(answer_result.get("applicable_laws", answer_result.get("law_article", [])))
                exceptions = answer_result.get("exceptions_or_conditions", "")
                
                if pred_status == "NO_LAW_FOUND":
                    pred_laws = []
                    pred_quotes = []

            expected_docs = case.get("source_files", [])
            if not expected_docs and case.get("source_file"):
                expected_docs = [s.strip() for s in case["source_file"].split(";") if s.strip()]

            expected_pairs = case.get("expected_pairs", [])
            if not expected_pairs:
                if len(expected_docs) == len(true_section) and len(expected_docs) > 0:
                    expected_pairs = [{"doc": d, "section": s} for d, s in zip(expected_docs, true_section)]
                else:
                    expected_pairs = [{"doc": d, "section": s} for d in expected_docs for s in true_section]

            # Candidate laws for evaluating retrieval hit rate:
            eval_chunks = []
            seen_chunk_keys = set()
            if case_res and isinstance(case_res, list) and len(case_res) > 0:
                first_item = case_res[0]
                cand_pool = list(first_item.get("used_laws", [])) + list(first_item.get("retrieved_laws", []))
                for ul in cand_pool:
                    if isinstance(ul, dict):
                        entry = ul.get("entry", "")
                        desc = ul.get("description", "") or ul.get("text", "")
                        key = (entry, desc[:40])
                        if key not in seen_chunk_keys and (entry or desc):
                            seen_chunk_keys.add(key)
                            eval_chunks.append({
                                "law_entry": entry,
                                "entry": entry,
                                "text": desc,
                                "snippet": desc[:300] if desc else ""
                            })
                    elif isinstance(ul, str) and ul.strip():
                        if ul.strip() not in seen_chunk_keys:
                            seen_chunk_keys.add(ul.strip())
                            eval_chunks.append({
                                "law_entry": ul.strip(),
                                "entry": ul.strip(),
                                "text": ul.strip()
                            })

            for pl in pred_laws:
                if isinstance(pl, str) and pl.strip() and pl.strip() not in seen_chunk_keys:
                    seen_chunk_keys.add(pl.strip())
                    eval_chunks.append({
                        "law_entry": pl.strip(),
                        "entry": pl.strip(),
                        "text": pl.strip()
                    })

            # Exact IR metrics calculation matching evaluate_rag_triad.py
            eval_k = max(len(eval_chunks), 5)
            ret_metrics = compute_retrieval_metrics(
                retrieved_items=eval_chunks,
                expected_pairs=expected_pairs,
                ground_truth_sections=true_section,
                k=eval_k
            )

            is_both_hit = (ret_metrics.get("hit_at_k", 0.0) > 0)
            recall_at_k = ret_metrics.get("recall_at_k", 0.0)
            prec_at_k = ret_metrics.get("precision_at_k", 0.0)
            mrr_at_k = ret_metrics.get("mrr_at_k", 0.0)

            # Section Hit & Document Hit
            is_section_hit = False
            for ts in true_section:
                ts_clean = str(ts).strip()
                if not ts_clean:
                    continue
                for item in eval_chunks:
                    chunk_text = f"{item.get('law_entry', '')} {item.get('text', '')}".strip()
                    if match_legal_section(ts_clean, chunk_text):
                        is_section_hit = True
                        break
                if is_section_hit:
                    break

            is_document_hit = False
            for ed in expected_docs:
                for item in eval_chunks:
                    chunk_text = f"{item.get('law_entry', '')} {item.get('text', '')}".strip()
                    if match_document(ed, chunk_text):
                        is_document_hit = True
                        break
                if is_document_hit:
                    break

            if is_section_hit:
                section_hits += 1
            if is_document_hit:
                document_hits += 1
            if is_both_hit:
                both_hits += 1

            results.append({
                "id": case.get("id"),
                "status": pred_status,
                "question": question,
                "direct_answer": pred_direct_answer,
                "decisive_quotes": pred_quotes,
                "ground_truth": ground_truth,
                "expected_pairs": expected_pairs,
                "predicted_laws": pred_laws,
                "is_section_hit": is_section_hit,
                "is_document_hit": is_document_hit,
                "is_both_hit": is_both_hit,
                "recall_at_k": recall_at_k,
                "hit_at_k": 1.0 if is_both_hit else 0.0,
                "precision_at_k": prec_at_k,
                "mrr_at_k": mrr_at_k,
                "exceptions_or_conditions": exceptions,
                "analysis": extract_case_analysis(case_res),
            })
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return section_hits, document_hits, both_hits, len(cases)
    
    finally:
        if hasattr(rag, 'model') and hasattr(rag.model, 'release_model'):
            try:
                rag.model.release_model()
            except:
                pass


def run_evaluation(
    model_name: str,
    datasets: str = "statute",
    dotenv_path: str = "configs/thai_procurement.env",
    devices: Optional[List[str]] = None,
    datasets_path: str = "./datasets",
    limit: Optional[int] = None,
    workers: int = 4,
    rag_mode: str = "agentic"
):
    config = PipelineConfig.from_env_file(dotenv_path)
    os.environ["RAG_MODE"] = (rag_mode or "agentic").lower()
    clean_dataset = sanitize_dataset_name(datasets)
    output_dir = os.path.join(config.data.output_dir, clean_dataset)
    os.makedirs(output_dir, exist_ok=True)
    if datasets_path == "./datasets" and config.data.datasets_path:
        datasets_path = config.data.datasets_path
    # Retrieval needs no preparation here: the tri-store is seeded by the migration, and local mode
    # builds its in-memory index from outputs/corpus (data_ocr/) on first use.
    if os.getenv("USE_TRI_STORE", "false").lower() in ("true", "1", "yes"):
        print("USE_TRI_STORE=true: evaluating against the tri-store.")
    else:
        print("Local mode: evaluating against the in-memory store built from data_ocr/.")

    test_cases = load_test_cases(datasets, datasets_path)
    print(f"Loaded {len(test_cases)} test cases from {datasets} dataset")
    if limit is not None and limit > 0:
        test_cases = test_cases[:limit]
        print(f"Limiting evaluation to first {len(test_cases)} test cases (--limit {limit})")
    
    # Multi-GPU support via multiprocessing pool when multiple distinct GPUs are specified
    if devices is not None and len(devices) > 1:
        num_processes = len(devices)
        chunks = [[] for _ in range(num_processes)]
        for i, case in enumerate(test_cases):
            chunk_index = i % num_processes
            chunks[chunk_index].append(case)
        
        print(f"Split {len(test_cases)} cases into {num_processes} distinct GPU processes")
        for i, chunk in enumerate(chunks):
            print(f"  Process {i} ({devices[i]}): {len(chunk)} cases")
        
        config_dict = config.to_dict()
        pool = multiprocessing.Pool(processes=num_processes)
        async_results = []
        time_before = time.time()
        
        for i, chunk in enumerate(chunks):
            output_file = os.path.join(output_dir, f"{model_name}_results_part_{i}.json")
            async_results.append(
                pool.apply_async(
                    process_cases_worker,
                    args=(chunk, config_dict, devices[i], output_file, model_name),
                )
            )
        pool.close()
        pool.join()
        time_after = time.time()
        elapsed_time = time_after - time_before
        
        total_document_hits, total_section_hits, total_both_hits, total_cases = 0, 0, 0, 0
        for res in async_results:
            sec_hits, doc_hits, both_hits, count = res.get()
            total_section_hits += sec_hits
            total_document_hits += doc_hits
            total_both_hits += both_hits
            total_cases += count
            
        combined_results = []
        for i in range(len(chunks)):
            part_file = os.path.join(output_dir, f"{model_name}_results_part_{i}.json")
            if os.path.exists(part_file):
                with open(part_file, "r", encoding="utf-8") as f:
                    combined_results.extend(json.load(f))
                os.remove(part_file)
    else:
        # High-efficiency ThreadPool concurrent execution:
        # Shares one retriever (store + GPU CrossEncoder, lock-guarded, ~1.2GB VRAM) across threads.
        # OpenRouter API calls run in parallel, cutting total inference time by 4x-10x!
        device = devices[0] if (devices and len(devices) > 0) else (config.model.device or "cpu")
        print(f"Starting concurrent inference with {workers} worker threads on {device} (model: {model_name})...")
        
        config.model.device = device
        config.model.model_name = model_name
        
        rag = ProcurementQAPipeline(config=config)
        
        # Warmup embedder and reranker in main thread to prevent multi-threaded CUDA initialization races
        try:
            from core.retrieval.embedding import get_embedding
            from core.retrieval.retriever import get_retriever
            _ = get_embedding("warmup query")
            _ = get_retriever()  # loads the store and the reranker once, before the threads start
            print("[Warmup] Local embedder and GPU reranker successfully initialized in main thread.")
        except Exception as e:
            print(f"[Warmup Warning] {e}")

        combined_results = []
        total_document_hits = 0
        total_section_hits = 0
        total_both_hits = 0
        total_recall = 0.0
        results_lock = threading.Lock()
        pbar = tqdm(total=len(test_cases), desc=f"Evaluating ({workers} workers)")
        
        def process_single_case(case):
            nonlocal total_document_hits, total_section_hits, total_both_hits, total_recall
            question = case.get("question", "")
            true_category = case.get("topics", [])
            true_section = case.get("expected_sections", [])
            ground_truth = case.get("ground_truth", "")
            
            case_res = rag.analyze_case(case)
            
            pred_answer = ""
            pred_direct_answer = ""
            pred_laws = []
            exceptions = ""
            
            pred_quotes = []
            
            if case_res and isinstance(case_res, list) and len(case_res) > 0:
                answer_result = case_res[0].get("answer_result", {})
                pred_status = answer_result.get("status", "COMPLIANT")
                pred_direct_answer = answer_result.get("direct_answer", "")
                pred_quotes = answer_result.get("decisive_quotes", [])
                pred_laws = list(answer_result.get("applicable_laws", answer_result.get("law_article", [])))
                exceptions = answer_result.get("exceptions_or_conditions", "")
                pred_issues_breakdown = answer_result.get("issues_breakdown", case_res[0].get("issues_breakdown", []))
                
                if pred_status == "NO_LAW_FOUND":
                    pred_laws = []
                    pred_quotes = []

            expected_docs = case.get("source_files", [])
            if not expected_docs and case.get("source_file"):
                expected_docs = [s.strip() for s in case["source_file"].split(";") if s.strip()]

            expected_pairs = case.get("expected_pairs", [])
            if not expected_pairs:
                if len(expected_docs) == len(true_section) and len(expected_docs) > 0:
                    expected_pairs = [{"doc": d, "section": s} for d, s in zip(expected_docs, true_section)]
                else:
                    expected_pairs = [{"doc": d, "section": s} for d in expected_docs for s in true_section]

            # Candidate laws for evaluating retrieval hit rate:
            eval_chunks = []
            seen_chunk_keys = set()
            if case_res and isinstance(case_res, list) and len(case_res) > 0:
                first_item = case_res[0]
                cand_pool = list(first_item.get("used_laws", [])) + list(first_item.get("retrieved_laws", []))
                for ul in cand_pool:
                    if isinstance(ul, dict):
                        entry = ul.get("entry", "")
                        desc = ul.get("description", "") or ul.get("text", "")
                        key = (entry, desc[:40])
                        if key not in seen_chunk_keys and (entry or desc):
                            seen_chunk_keys.add(key)
                            eval_chunks.append({
                                "law_entry": entry,
                                "entry": entry,
                                "text": desc,
                                "snippet": desc[:300] if desc else ""
                            })
                    elif isinstance(ul, str) and ul.strip():
                        if ul.strip() not in seen_chunk_keys:
                            seen_chunk_keys.add(ul.strip())
                            eval_chunks.append({
                                "law_entry": ul.strip(),
                                "entry": ul.strip(),
                                "text": ul.strip()
                            })

            for pl in pred_laws:
                if isinstance(pl, str) and pl.strip() and pl.strip() not in seen_chunk_keys:
                    seen_chunk_keys.add(pl.strip())
                    eval_chunks.append({
                        "law_entry": pl.strip(),
                        "entry": pl.strip(),
                        "text": pl.strip()
                    })

            # Exact IR metrics calculation matching evaluate_rag_triad.py
            eval_k = max(len(eval_chunks), 5)
            ret_metrics = compute_retrieval_metrics(
                retrieved_items=eval_chunks,
                expected_pairs=expected_pairs,
                ground_truth_sections=true_section,
                k=eval_k
            )

            is_both_hit = (ret_metrics.get("hit_at_k", 0.0) > 0)
            recall_at_k = ret_metrics.get("recall_at_k", 0.0)
            prec_at_k = ret_metrics.get("precision_at_k", 0.0)
            mrr_at_k = ret_metrics.get("mrr_at_k", 0.0)

            # Section Hit & Document Hit
            is_section_hit = False
            for ts in true_section:
                ts_clean = str(ts).strip()
                if not ts_clean:
                    continue
                for item in eval_chunks:
                    chunk_text = f"{item.get('law_entry', '')} {item.get('text', '')}".strip()
                    if match_legal_section(ts_clean, chunk_text):
                        is_section_hit = True
                        break
                if is_section_hit:
                    break

            is_document_hit = False
            for ed in expected_docs:
                for item in eval_chunks:
                    chunk_text = f"{item.get('law_entry', '')} {item.get('text', '')}".strip()
                    if match_document(ed, chunk_text):
                        is_document_hit = True
                        break
                if is_document_hit:
                    break

            case_data = {
                "id": case.get("id"),
                "status": pred_status,
                "question": question,
                "direct_answer": pred_direct_answer,
                "decisive_quotes": pred_quotes,
                "ground_truth": ground_truth,
                "expected_pairs": expected_pairs,
                "predicted_laws": pred_laws,
                "issues_breakdown": pred_issues_breakdown,
                "is_section_hit": is_section_hit,
                "is_document_hit": is_document_hit,
                "is_both_hit": is_both_hit,
                "recall_at_k": recall_at_k,
                "hit_at_k": 1.0 if is_both_hit else 0.0,
                "precision_at_k": prec_at_k,
                "mrr_at_k": mrr_at_k,
                "exceptions_or_conditions": exceptions,
                "analysis": extract_case_analysis(case_res),
            }
            
            with results_lock:
                if is_section_hit:
                    total_section_hits += 1
                if is_document_hit:
                    total_document_hits += 1
                if is_both_hit:
                    total_both_hits += 1
                total_recall += recall_at_k
                combined_results.append(case_data)
                pbar.update(1)

        time_before = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_single_case, c): c for c in test_cases}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    c = futures[future]
                    print(f"\n[Worker Error] Failed processing case {c.get('id')}: {e}")
        pbar.close()
        time_after = time.time()
        elapsed_time = time_after - time_before
        
        combined_results.sort(key=lambda x: x.get("id", 0))
        total_cases = len(combined_results)
    
    doc_rate = (total_document_hits / total_cases * 100) if total_cases > 0 else 0.0
    sec_rate = (total_section_hits / total_cases * 100) if total_cases > 0 else 0.0
    strict_hit_rate = (total_both_hits / total_cases * 100) if total_cases > 0 else 0.0
    mean_recall = (total_recall / total_cases * 100) if total_cases > 0 else 0.0
    
    active_mode = os.environ.get("RAG_MODE", "agentic")
    print(f"\n{'='*65}")
    print(f"Model: {model_name} | Dataset: {clean_dataset} | Mode: {active_mode}")
    print(f"Total Questions Evaluated: {total_cases}")
    print(f"🎯 STRICT HIT RATE [Doc AND Section] (Hit@k): {total_both_hits}/{total_cases} ({strict_hit_rate:.1f}%)")
    print(f"   ├─ Ground Truth Recall@k: {mean_recall:.2f}%")
    print(f"   ├─ Document Hit Rate:     {total_document_hits}/{total_cases} ({doc_rate:.1f}%)")
    print(f"   └─ Section Hit Rate:      {total_section_hits}/{total_cases} ({sec_rate:.1f}%)")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"{'='*65}\n")
    
    mode_results_file = os.path.join(output_dir, f"{active_mode}_results.json")
    stats = {
        "model_name": model_name,
        "rag_mode": active_mode,
        "dataset": clean_dataset,
        "total_cases": total_cases,
        "strict_hits": total_both_hits,
        "strict_hit_rate": strict_hit_rate,
        "mean_recall_at_k": mean_recall,
        "document_hits": total_document_hits,
        "document_hit_rate": doc_rate,
        "section_hits": total_section_hits,
        "section_hit_rate": sec_rate,
        "elapsed_time": elapsed_time
    }
    output_payload = {
        "stats": stats,
        "results": combined_results
    }
    with open(mode_results_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)
    
    print(f"Results and statistics saved to {mode_results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Legal Case Analysis with Different Models using ProcurementQA Agent"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openrouter",
        help="Model to use for analysis (e.g. openrouter, google/gemma-3-4b-it, qwen3, gpt4o_mini, etc.)",
    )
    default_dotenv = "configs/thai_procurement.env" if os.path.exists("configs/thai_procurement.env") else ".env"
    parser.add_argument(
        "--dotenv_path",
        type=str,
        default=default_dotenv,
        help="Path to the .env file (default: configs/thai_procurement.env)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="statute",
        help="Dataset path or alias: statute (default), statute_sample, general",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="# of query to inference",
    )
    parser.add_argument(
        "--datasets_path",
        type=str,
        default=None,
        help="Path to datasets directory (default: ./datasets)",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=None,
        help="GPU devices to use (e.g., cuda:2 cuda:3)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent worker threads for inference (default: 8)",
    )
    parser.add_argument(
        "--rag-mode",
        type=str,
        choices=["agentic", "crag"],
        default="agentic",
        help="RAG Pipeline Execution Mode (default: 'agentic' - LangGraph self-reflective Agentic RAG)",
    )
    
    args = parser.parse_args()
    
    multiprocessing.set_start_method("spawn", force=True)
    
    run_evaluation(
        model_name=args.model,
        datasets=args.datasets,
        dotenv_path=args.dotenv_path,
        devices=args.devices,
        datasets_path=args.datasets_path if args.datasets_path else "./datasets",
        limit=args.limit,
        workers=args.workers,
        rag_mode=args.rag_mode
    )
