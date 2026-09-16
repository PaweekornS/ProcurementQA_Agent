#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_thai_corpus.py

Prepares Thai procurement corpus files for LegalGraphRAG:
1. Ingests procurement_data/typhoon_ocr (markdown statutes) and procurement_data/FAQ_กรมบัญชีกลาง.xlsx (Q&A pairs)
   to build the vectorDB / knowledge graph database.
2. Ingests procurement_data/qa_*.csv (qa_single_doc_test.csv, qa_multi_doc_test.csv) as strict test sets
   for benchmark evaluation, preventing test set leakage into the knowledge graph.
3. Generates:
   - datas/thai_corpus/law_to_crime.json: Law and regulation nodes with candidate topics.
   - datas/thai_corpus/cases_with_feature.json: FAQ inquiry cases with legal features (knowledge graph only).
   - datasets/crime_data_THAI_small.json: Combined benchmark evaluation questions (single + multi doc).
   - datasets/crime_data_THAI_single_small.json: Single-statute benchmark questions.
   - datasets/crime_data_THAI_multi_small.json: Cross-statute benchmark questions.
"""

import os
import sys
import json
import re
import glob
import argparse
from typing import List, Dict, Any, Optional
import pandas as pd


def find_file(relative_paths: List[str], base_dirs: List[str]) -> str:
    """Search for an existing file across multiple base directories."""
    for b in base_dirs:
        for r in relative_paths:
            candidate = os.path.normpath(os.path.join(b, r))
            if os.path.exists(candidate):
                return candidate
    return ""


def chunk_statute_markdown(
    content: str,
    doc_title: str,
    category: str,
    rel_source: str,
    max_chunk_chars: int = 2500
) -> List[Dict[str, Any]]:
    """
    Splits Thai statute/procurement markdown by sections/articles (มาตรา / ข้อ / หมวด)
    while keeping sub-paragraphs (วรรค / อนุมาตรา) together.
    """
    # 1. Clean page markers and horizontal rules
    cleaned = re.sub(r"<!--\s*Page\s+\d+\s+of\s+\d+\s*-->", "", content, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n\s*---\s*\n", "\n", cleaned)
    cleaned = re.sub(r"\r\n", "\n", cleaned)

    # 2. Match section boundaries: มาตรา ..., ข้อ ..., หมวด ...
    # Pattern detects lines starting with Section/Article/Chapter keywords
    section_pattern = re.compile(r"(?m)^(?:#{1,4}\s*)?(มาตรา\s+[0-9๑-๙]+|ข้อ\s+[0-9๑-๙]+|หมวด\s+[0-9๑-๙]+|ส่วนที่\s+[0-9๑-๙]+)")
    
    splits = []
    last_idx = 0
    current_sec_label = ""
    current_chapter = ""
    
    matches = list(section_pattern.finditer(cleaned))
    
    if not matches:
        # Fallback: combine paragraphs into macro chunks of ~2000 chars
        paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
        cur_text = ""
        chunk_num = 1
        for p in paragraphs:
            if len(cur_text) + len(p) > max_chunk_chars and cur_text:
                splits.append({
                    "text": cur_text.strip(),
                    "section": f"ส่วนที่ {chunk_num}",
                    "chapter": ""
                })
                cur_text = p + "\n\n"
                chunk_num += 1
            else:
                cur_text += p + "\n\n"
        if cur_text.strip():
            splits.append({
                "text": cur_text.strip(),
                "section": f"ส่วนที่ {chunk_num}",
                "chapter": ""
            })
    else:
        # Preamble before the first section match
        if matches[0].start() > 0:
            preamble = cleaned[:matches[0].start()].strip()
            if len(preamble) > 100:
                splits.append({
                    "text": preamble,
                    "section": "คำนำ/บททั่วไป",
                    "chapter": ""
                })

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
            block = cleaned[start:end].strip()
            header_text = m.group(1).strip()

            if "หมวด" in header_text:
                current_chapter = header_text
                # If block is short (just chapter title), keep chapter and continue
                lines = block.split("\n")
                if len(lines) <= 2:
                    continue

            sec_name = header_text if ("มาตรา" in header_text or "ข้อ" in header_text) else current_sec_label or header_text
            
            # If block is too large (> 3500 chars), split into sub-windows
            if len(block) > 3500:
                paras = [p.strip() for p in block.split("\n\n") if p.strip()]
                sub_text = ""
                sub_idx = 1
                for p in paras:
                    if len(sub_text) + len(p) > max_chunk_chars and sub_text:
                        splits.append({
                            "text": sub_text.strip(),
                            "section": f"{sec_name} (ตอนที่ {sub_idx})",
                            "chapter": current_chapter
                        })
                        sub_text = p + "\n\n"
                        sub_idx += 1
                    else:
                        sub_text += p + "\n\n"
                if sub_text.strip():
                    splits.append({
                        "text": sub_text.strip(),
                        "section": f"{sec_name} (ตอนที่ {sub_idx})",
                        "chapter": current_chapter
                    })
            else:
                if len(block) > 40:  # ignore tiny artifact lines
                    splits.append({
                        "text": block,
                        "section": sec_name,
                        "chapter": current_chapter
                    })

    # 3. Create rich chunks with context header
    result_chunks = []
    for c_idx, s in enumerate(splits):
        sec = s["section"]
        chap = s["chapter"]
        body = s["text"]

        context_parts = [f"[{doc_title}]"]
        if chap:
            context_parts.append(f"[{chap}]")
        if sec:
            context_parts.append(f"[{sec}]")
        header_prefix = " ".join(context_parts)

        rich_content = f"{header_prefix}\n{body}"
        chunk_id = f"{doc_title}_{c_idx+1}"

        sections_covered = [sec] if sec and sec not in ("คำนำ/บททั่วไป",) else []

        result_chunks.append({
            "chunk_id": chunk_id,
            "source_file": rel_source,
            "category": category,
            "heading": f"{doc_title} - {sec}",
            "doc_title": doc_title,
            "sections_covered": sections_covered,
            "content": rich_content,
            "raw_text": body
        })

    return result_chunks


def load_raw_chunks(
    chunks_file: Optional[str] = None,
    typhoon_dir: Optional[str] = None,
    faq_file: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Load chunks from chunks_file if present,
    or build structure-aware section chunks directly from typhoon_dir markdown files and faq_file.
    """
    if chunks_file and os.path.exists(chunks_file):
        print(f"Loading pre-processed chunks from: {chunks_file}")
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks.")
        return chunks

    chunks = []

    # 1. Parse typhoon_ocr markdown files using structure-aware section chunking
    if typhoon_dir and os.path.exists(typhoon_dir):
        print(f"Extracting statutory chunks from: {typhoon_dir}")
        for root, _, files in os.walk(typhoon_dir):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    rel_source = os.path.relpath(file_path, os.path.dirname(typhoon_dir)).replace("\\", "/")
                    category = os.path.basename(root)
                    doc_title = os.path.splitext(file)[0]
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        statute_chunks = chunk_statute_markdown(
                            content=content,
                            doc_title=doc_title,
                            category=category,
                            rel_source=rel_source
                        )
                        chunks.extend(statute_chunks)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    # 2. Parse FAQ excel file (1 question-answer pair = 1 cohesive chunk)
    if faq_file and os.path.exists(faq_file):
        print(f"Extracting FAQ chunks from: {faq_file}")
        try:
            df = pd.read_excel(faq_file)
            for i, row in df.iterrows():
                q = str(row.get("Question", "")).strip()
                a = str(row.get("Answer", "")).strip()
                chunk_id = f"faq_cgd_{i+1}"
                content = f"[แนวทางคำถาม-คำตอบ (FAQ) กรมบัญชีกลาง | ข้อที่ {i+1}]\nคำถาม: {q}\nคำตอบ: {a}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "source_file": os.path.basename(faq_file),
                    "category": "FAQ กรมบัญชีกลาง",
                    "heading": f"FAQ ข้อที่ {i+1}: {q[:50]}",
                    "doc_title": "แนวทางคำถาม-คำตอบ (FAQ) กรมบัญชีกลาง",
                    "sections_covered": [f"FAQ ข้อที่ {i+1}"],
                    "content": content,
                    "raw_text": f"คำถาม: {q}\nคำตอบ: {a}"
                })
        except Exception as e:
            print(f"Error reading FAQ file: {e}")

    print(f"Constructed {len(chunks)} structure-aware knowledge chunks directly from procurement sources.")
    return chunks


def build_law_to_crime(chunks: List[Dict[str, Any]], output_path: str) -> List[Dict[str, Any]]:
    """
    Build law_to_crime.json mapping each statutory/FAQ chunk to topics, consideration criteria, and citations.
    """
    law_to_crime_list = []

    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id", ""))
        sections = chunk.get("sections_covered", [])
        category = chunk.get("category", "การจัดซื้อจัดจ้างภาครัฐ")
        doc_title = chunk.get("doc_title", "")
        heading = chunk.get("heading", "")
        content = chunk.get("content", "")
        source_file = chunk.get("source_file", "")

        id_parts = [chunk_id]
        if sections:
            id_parts.extend([str(s) for s in sections])
        entry_id = " | ".join(id_parts)

        crimes = [category]
        if doc_title and doc_title != category:
            crimes.append(doc_title)

        judge_dep = [heading] if heading else []
        related_laws = [source_file] if source_file else []
        if sections:
            related_laws.extend(sections)

        law_to_crime_list.append({
            "id": entry_id,
            "items": [
                {
                    "text": content,
                    "crime": crimes,
                    "judge_dep": judge_dep,
                    "related_laws": related_laws
                }
            ]
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(law_to_crime_list, f, ensure_ascii=False, indent=2)

    print(f"Generated law_to_crime with {len(law_to_crime_list)} nodes -> {output_path}")
    return law_to_crime_list


def build_cases_with_feature(
    faq_file: str,
    output_path: str
) -> List[Dict[str, Any]]:
    """
    Build cases_with_feature.json from FAQ Q&A entries ONLY.
    Strictly excludes test questions to avoid knowledge graph / vectorDB contamination.
    """
    cases = []
    case_idx = 0

    if faq_file and os.path.exists(faq_file):
        try:
            df_faq = pd.read_excel(faq_file)
            for i, row in df_faq.iterrows():
                q = str(row.get("Question", "")).strip()
                a = str(row.get("Answer", "")).strip()
                fact_text = f"ข้อหารือ/คำถาม: {q}\nแนวทางวินิจฉัย/คำตอบ: {a}"

                cases.append({
                    "id": case_idx,
                    "name": ["ผู้สอบถาม (หน่วยงานของรัฐ / ผู้ประกอบการ)"],
                    "fact": fact_text,
                    "crime": ["FAQ กรมบัญชีกลาง", "การปฏิบัติการจัดซื้อจัดจ้าง"],
                    "law": [f"FAQ ข้อที่ {i+1}", f"faq_cgd_{i+1}", "ระเบียบกระทรวงการคลังว่าด้วยการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. 2560"],
                    "laws": [f"FAQ ข้อที่ {i+1}", f"faq_cgd_{i+1}", "ระเบียบกระทรวงการคลังว่าด้วยการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. 2560"],
                    "term_of_imprisonment": {
                        "death_penalty": False,
                        "imprisonment": 0,
                        "life_imprisonment": False
                    },
                    "features": {
                        "defendant_info": ["หน่วยงานของรัฐ / ผู้ปฏิบัติงานพัสดุ"],
                        "criminal_acts": [f"ประเด็นข้อหารือ {q[:40]}"],
                        "victim_property_details": ["ระบบการจัดซื้อจัดจ้างภาครัฐ (e-GP) / พัสดุ"],
                        "intent_remorse": []
                    }
                })
                case_idx += 1
        except Exception as e:
            print(f"Error reading FAQ file: {e}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    print(f"Generated cases_with_feature (FAQ knowledge cases only) with {len(cases)} entries -> {output_path}")
    return cases


def parse_qa_csv(csv_path: str, start_id: int = 0) -> List[Dict[str, Any]]:
    """Parse a single qa_*.csv file into benchmark format."""
    dataset = []
    if not os.path.exists(csv_path):
        print(f"Warning: QA test file not found: {csv_path}")
        return dataset

    df = pd.read_csv(csv_path)
    for i, row in df.iterrows():
        q = str(row.get("Question", "")).strip()
        gt = str(row.get("Ground_Truth", "")).strip()
        sec = str(row.get("Section", "")).strip()
        cat = str(row.get("Category", "การจัดซื้อจัดจ้างภาครัฐ")).strip()
        src = str(row.get("Source_File", "")).strip()
        q_type = str(row.get("Question_Type", "")).strip()

        # Split multiple sections by comma or semicolon for accurate retrieval matching
        sec_clean = sec.replace(";", ",")
        laws_list = [s.strip() for s in sec_clean.split(",") if s.strip()]
        if not laws_list and sec:
            laws_list = [sec]

        src_list = [s.strip() for s in src.replace(";", "\n").split("\n") if s.strip()]

        dataset.append({
            "id": start_id + int(i),
            "name": ["ผู้สอบถาม"],
            "fact": q,
            "crime": [cat],
            "laws": laws_list,
            "ground_truth": gt,
            "source_file": src,
            "source_files": src_list,
            "question_type": q_type,
            "test_source": os.path.basename(csv_path),
            "term_of_imprisonment": {
                "death_penalty": False,
                "imprisonment": 0,
                "life_imprisonment": False
            }
        })
    return dataset


def build_benchmark_datasets(
    qa_files: List[str],
    datasets_dir: str
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build evaluation datasets from qa_*.csv files:
    - crime_data_THAI_small.json (combined)
    - crime_data_THAI_single_small.json (from qa_single_doc_test.csv)
    - crime_data_THAI_multi_small.json (from qa_multi_doc_test.csv)
    """
    os.makedirs(datasets_dir, exist_ok=True)
    all_cases = []
    current_id = 0
    results_by_name = {}

    for qf in qa_files:
        if not os.path.exists(qf):
            continue
        base = os.path.basename(qf)
        cases = parse_qa_csv(qf, start_id=current_id)
        current_id += len(cases)
        all_cases.extend(cases)

        # Output specific sub-dataset
        if "single" in base.lower():
            sub_path = os.path.join(datasets_dir, "crime_data_THAI_single_small.json")
            with open(sub_path, "w", encoding="utf-8") as f:
                json.dump(cases, f, ensure_ascii=False, indent=2)
            print(f"Generated single-doc benchmark dataset ({len(cases)} items) -> {sub_path}")
            results_by_name["single"] = cases
        elif "multi" in base.lower():
            sub_path = os.path.join(datasets_dir, "crime_data_THAI_multi_small.json")
            with open(sub_path, "w", encoding="utf-8") as f:
                json.dump(cases, f, ensure_ascii=False, indent=2)
            print(f"Generated multi-doc benchmark dataset ({len(cases)} items) -> {sub_path}")
            results_by_name["multi"] = cases

    # Combined main benchmark dataset
    combined_path = os.path.join(datasets_dir, "crime_data_THAI_small.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    print(f"Generated combined benchmark dataset ({len(all_cases)} items) -> {combined_path}")
    results_by_name["combined"] = all_cases

    return results_by_name


def main():
    parser = argparse.ArgumentParser(description="Prepare Thai Procurement Corpus for LegalGraphRAG")
    parser.add_argument("--procurement_dir", type=str, default="./procurement_data", help="Path to procurement_data directory")
    parser.add_argument("--chunks_file", type=str, default="", help="Optional path to precomputed chunks.json")
    parser.add_argument("--typhoon_dir", type=str, default="", help="Path to typhoon_ocr directory")
    parser.add_argument("--faq_file", type=str, default="", help="Path to FAQ_กรมบัญชีกลาง.xlsx")
    parser.add_argument("--output_dir", type=str, default="./datas/thai_corpus", help="Output directory for corpus files")
    parser.add_argument("--datasets_dir", type=str, default="./datasets", help="Output directory for test datasets")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dirs = [os.getcwd(), project_root]

    # Resolve procurement_dir
    procurement_dir = args.procurement_dir
    if not os.path.exists(procurement_dir):
        procurement_dir = find_file(["procurement_data", "data"], base_dirs) or "./procurement_data"

    # Resolve input paths with solid procurement_data defaults
    typhoon_dir = args.typhoon_dir or os.path.join(procurement_dir, "typhoon_ocr")
    if not os.path.exists(typhoon_dir):
        typhoon_dir = find_file(["procurement_data/typhoon_ocr", "typhoon_ocr"], base_dirs)

    faq_file = args.faq_file or os.path.join(procurement_dir, "FAQ_กรมบัญชีกลาง.xlsx")
    if not os.path.exists(faq_file):
        faq_file = find_file(["procurement_data/FAQ_กรมบัญชีกลาง.xlsx", "FAQ_กรมบัญชีกลาง.xlsx"], base_dirs)

    chunks_file = args.chunks_file or find_file(["procurement_data/chunks.json", "storage/chunks.json"], base_dirs)

    # Resolve test csv files (begins with qa_...)
    qa_files = []
    if os.path.exists(procurement_dir):
        for f in os.listdir(procurement_dir):
            if f.startswith("qa_") and f.endswith(".csv"):
                qa_files.append(os.path.join(procurement_dir, f))
    qa_files.sort()

    print("Resolved Procurement Inputs:")
    print(f"  Procurement Dir: {procurement_dir}")
    print(f"  Typhoon Dir:     {typhoon_dir} (Exists: {os.path.exists(typhoon_dir)})")
    print(f"  FAQ File:        {faq_file} (Exists: {os.path.exists(faq_file)})")
    print(f"  QA Test Files:   {qa_files}")
    print(f"  Corpus Outputs:  {args.output_dir}")
    print(f"  Dataset Outputs: {args.datasets_dir}")
    print()

    # 1. Load Chunks for VectorDB / Knowledge Graph
    chunks = load_raw_chunks(chunks_file, typhoon_dir, faq_file)

    # 2. Build datas/thai_corpus/law_to_crime.json
    law_output = os.path.join(args.output_dir, "law_to_crime.json")
    build_law_to_crime(chunks, law_output)

    # 3. Build datas/thai_corpus/cases_with_feature.json (FAQ only, NO test questions)
    cases_output = os.path.join(args.output_dir, "cases_with_feature.json")
    build_cases_with_feature(faq_file, cases_output)

    # 4. Build datasets/crime_data_THAI_*.json test suites
    build_benchmark_datasets(qa_files, args.datasets_dir)

    print("\nThai Procurement Corpus preparation complete!")


if __name__ == "__main__":
    main()
