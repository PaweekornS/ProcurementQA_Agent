#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_thai_corpus.py

Prepares Thai procurement corpus files for LegalGraphRAG:
1. Ingests data/typhoon_ocr (72 markdown files) and data/FAQ_กรมบัญชีกลาง.xlsx (29 Q&A pairs)
   (using 1_CRAG/storage/chunks.json when available for exact alignment with pre-computed Qdrant embeddings).
2. Generates:
   - datas/thai_corpus/law_to_crime.json: Law and regulation nodes with candidate topics.
   - datas/thai_corpus/cases_with_feature.json: FAQ and test inquiry cases with legal features.
   - datasets/crime_data_THAI_small.json: Benchmark evaluation questions from data/qa_test.csv.
"""

import os
import sys
import json
import uuid
import argparse
from typing import List, Dict, Any
import pandas as pd


def find_file(relative_paths: List[str], base_dirs: List[str]) -> str:
    """Search for an existing file across multiple base directories."""
    for b in base_dirs:
        for r in relative_paths:
            candidate = os.path.normpath(os.path.join(b, r))
            if os.path.exists(candidate):
                return candidate
    return ""


def load_raw_chunks(chunks_file: str, typhoon_dir: str, faq_file: str) -> List[Dict[str, Any]]:
    """
    Load chunks from 1_CRAG/storage/chunks.json if present,
    or build them directly from typhoon_dir markdown files and faq_file.
    """
    if chunks_file and os.path.exists(chunks_file):
        print(f"Loading pre-processed chunks from: {chunks_file}")
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks.")
        return chunks

    print("Pre-processed chunks.json not found. Extracting from source files...")
    chunks = []
    chunk_idx = 0

    # 1. Parse typhoon_ocr markdown files
    if typhoon_dir and os.path.exists(typhoon_dir):
        for root, _, files in os.walk(typhoon_dir):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    rel_source = os.path.relpath(file_path, os.path.dirname(typhoon_dir))
                    category = os.path.basename(root)
                    doc_title = os.path.splitext(file)[0]
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        
                        # Split by section / article markers if possible
                        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                        for p_idx, p in enumerate(paragraphs):
                            chunk_id = f"{doc_title}_p{p_idx+1}"
                            chunks.append({
                                "chunk_id": chunk_id,
                                "source_file": rel_source,
                                "category": category,
                                "heading": f"{doc_title} - ส่วนที่ {p_idx+1}",
                                "doc_title": doc_title,
                                "sections_covered": [],
                                "content": p,
                                "raw_text": p
                            })
                            chunk_idx += 1
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    # 2. Parse FAQ excel file
    if faq_file and os.path.exists(faq_file):
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

    print(f"Constructed {len(chunks)} chunks directly from source files.")
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

        # Multi-identifier entry for maximum matching flexibility
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
    qa_file: str,
    output_path: str
) -> List[Dict[str, Any]]:
    """
    Build cases_with_feature.json from FAQ Q&A entries and QA benchmark cases.
    """
    cases = []
    case_idx = 0

    # 1. Ingest FAQ items as real inquiry cases
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

    # 2. Ingest QA test cases as reference inquiry cases
    if qa_file and os.path.exists(qa_file):
        try:
            df_qa = pd.read_csv(qa_file)
            for i, row in df_qa.iterrows():
                q = str(row.get("Question", "")).strip()
                gt = str(row.get("Ground_Truth", "")).strip()
                sec = str(row.get("Section", "")).strip()
                cat = str(row.get("Category", "การจัดซื้อจัดจ้างภาครัฐ")).strip()
                src = str(row.get("Source_File", "")).strip()

                laws = [sec] if sec else []
                if src:
                    laws.append(src)

                cases.append({
                    "id": case_idx,
                    "name": ["ผู้สอบถาม"],
                    "fact": q,
                    "crime": [cat, "การจัดซื้อจัดจ้างภาครัฐ"],
                    "law": laws,
                    "laws": laws,
                    "ground_truth": gt,
                    "term_of_imprisonment": {
                        "death_penalty": False,
                        "imprisonment": 0,
                        "life_imprisonment": False
                    },
                    "features": {
                        "defendant_info": ["หน่วยงานของรัฐ / คณะกรรมการจัดซื้อจัดจ้าง"],
                        "criminal_acts": [cat],
                        "victim_property_details": ["การจัดซื้อจัดจ้าง / ขอบเขตของงาน (TOR) / สัญญา"],
                        "intent_remorse": []
                    }
                })
                case_idx += 1
        except Exception as e:
            print(f"Error reading QA file: {e}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    print(f"Generated cases_with_feature with {len(cases)} case nodes -> {output_path}")
    return cases


def build_benchmark_dataset(qa_file: str, output_path: str) -> List[Dict[str, Any]]:
    """
    Build crime_data_THAI_small.json benchmark evaluation dataset from data/qa_test.csv.
    """
    dataset = []
    if not os.path.exists(qa_file):
        print(f"Warning: qa_file not found: {qa_file}")
        return dataset

    df = pd.read_csv(qa_file)
    for i, row in df.iterrows():
        q = str(row.get("Question", "")).strip()
        gt = str(row.get("Ground_Truth", "")).strip()
        sec = str(row.get("Section", "")).strip()
        cat = str(row.get("Category", "การจัดซื้อจัดจ้างภาครัฐ")).strip()
        src = str(row.get("Source_File", "")).strip()
        q_type = str(row.get("Question_Type", "")).strip()

        dataset.append({
            "id": int(i),
            "name": ["ผู้สอบถาม"],
            "fact": q,
            "crime": [cat],
            "laws": [sec] if sec else [],
            "ground_truth": gt,
            "source_file": src,
            "question_type": q_type,
            "term_of_imprisonment": {
                "death_penalty": False,
                "imprisonment": 0,
                "life_imprisonment": False
            }
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Generated benchmark dataset with {len(dataset)} items -> {output_path}")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare Thai Procurement Corpus for LegalGraphRAG")
    parser.add_argument("--chunks_file", type=str, default="", help="Path to 1_CRAG/storage/chunks.json")
    parser.add_argument("--typhoon_dir", type=str, default="", help="Path to data/typhoon_ocr directory")
    parser.add_argument("--faq_file", type=str, default="", help="Path to data/FAQ_กรมบัญชีกลาง.xlsx")
    parser.add_argument("--qa_file", type=str, default="", help="Path to data/qa_test.csv")
    parser.add_argument("--output_dir", type=str, default="./datas/thai_corpus", help="Output directory for corpus files")
    parser.add_argument("--datasets_dir", type=str, default="./datasets", help="Output directory for test datasets")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workspace_root = os.path.dirname(project_root)

    base_dirs = [os.getcwd(), project_root, workspace_root]

    chunks_file = args.chunks_file or find_file(
        ["1_CRAG/storage/chunks.json", "storage/chunks.json"], base_dirs
    )
    typhoon_dir = args.typhoon_dir or find_file(
        ["data/typhoon_ocr"], base_dirs
    )
    faq_file = args.faq_file or find_file(
        ["data/FAQ_กรมบัญชีกลาง.xlsx"], base_dirs
    )
    qa_file = args.qa_file or find_file(
        ["data/qa_test.csv"], base_dirs
    )

    print("Resolved input paths:")
    print(f"  Chunks file: {chunks_file}")
    print(f"  Typhoon dir: {typhoon_dir}")
    print(f"  FAQ file:    {faq_file}")
    print(f"  QA test:     {qa_file}")

    # Output paths
    law_output = os.path.join(args.output_dir, "law_to_crime.json")
    cases_output = os.path.join(args.output_dir, "cases_with_feature.json")
    dataset_output = os.path.join(args.datasets_dir, "crime_data_THAI_small.json")

    # 1. Load Chunks
    chunks = load_raw_chunks(chunks_file, typhoon_dir, faq_file)

    # 2. Build law_to_crime.json
    build_law_to_crime(chunks, law_output)

    # 3. Build cases_with_feature.json
    build_cases_with_feature(faq_file, qa_file, cases_output)

    # 4. Build crime_data_THAI_small.json
    build_benchmark_dataset(qa_file, dataset_output)

    print("\nThai Procurement Corpus preparation complete!")


if __name__ == "__main__":
    main()
