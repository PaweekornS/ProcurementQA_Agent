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


from pathlib import Path


def clean_text(text: str) -> str:
    """Normalize whitespace and remove excessive linebreaks."""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_document_title(file_path: Path, content: str) -> str:
    """Extract official title from markdown heading or filename stem."""
    match = re.search(r"^#+\s+(.+)$", content, re.MULTILINE)
    if match:
        title = match.group(1).strip()
        if len(title) < 5 or title in ["พระราชบัญญัติ", "ระเบียบ", "ประกาศ", "กฎกระทรวง", "หน้า"]:
            return file_path.stem
        return title
    return file_path.stem


def extract_major_structural_breaks(text: str) -> List[str]:
    """Extract prominent structural headings (Schedules, Chapters, Tables, Forms) from text."""
    breaks = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if re.match(r"^#{1,3}\s+", s):
            h = re.sub(r"^#{1,3}\s+", "", s).strip()
            if not re.match(r"^หน้า\s*[0-9๐-๙]+$", h) and len(h) > 2:
                breaks.append(h)
        elif re.match(r"^(?:บัญชี(?:เอกสาร)?แนบท้าย|ตารางหลักเกณฑ์|ตารางแนบท้าย|หมวด\s*[๐-๙0-9]+|ส่วนที่\s*[๐-๙0-9]+|แบบสัญญา)", s):
            breaks.append(s)
    return breaks


def build_group_chunk(
    group: List[Dict[str, Any]],
    file_path: Path,
    doc_title: str,
    category: str,
    rel_path: str,
) -> Dict[str, Any]:
    """Assemble a macro chunk from one or more contiguous pages belonging to the same legal section."""
    pages_covered = [p["page_num"] for p in group]
    start_page = pages_covered[0]
    end_page = pages_covered[-1]
    total_pages = group[0]["total_pages"]

    all_sections = []
    for p in group:
        for s in p.get("sections", []):
            if s not in all_sections:
                all_sections.append(s)

    first_header = ""
    for p in group:
        if p.get("headers"):
            first_header = p["headers"][0]
            break

    if len(pages_covered) == 1:
        page_str = f"หน้า {start_page}/{total_pages}" if total_pages > 0 else f"หน้า {start_page}"
        chunk_id = f"{file_path.stem}_p{start_page}"
    else:
        page_str = f"หน้า {start_page}-{end_page}/{total_pages}" if total_pages > 0 else f"หน้า {start_page}-{end_page}"
        chunk_id = f"{file_path.stem}_p{start_page}_p{end_page}"

    heading_parts = [page_str]
    if first_header:
        heading_parts.append(f"[{first_header}]")
    if all_sections:
        heading_parts.append(f"[บทบัญญัติ: {', '.join(all_sections[:5])}]")

    heading_str = " ".join(heading_parts)
    merged_body = "\n\n".join(p["text"] for p in group)
    chunk_text = f"[{doc_title} | {heading_str}]\n{merged_body}"

    return {
        "chunk_id": chunk_id,
        "source_file": rel_path,
        "category": category,
        "heading": heading_str,
        "doc_title": doc_title,
        "page_num": start_page,
        "pages_covered": pages_covered,
        "total_pages": total_pages,
        "sections_covered": all_sections,
        "content": chunk_text,
        "raw_text": merged_body,
    }


def chunk_legal_document(
    file_path: Path,
    max_macro_size: int = 8000,
    max_pages_per_chunk: int = 4,
) -> List[Dict[str, Any]]:
    """
    Split a Thai legal markdown document into structure-aware macro chunks.
    Groups contiguous pages belonging to the same schedule, chapter, table, or section
    up to max_macro_size (default 8,000 characters) while preserving page lineage and
    statutory boundaries.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        try:
            content = file_path.read_text(encoding="utf-8-sig")
        except Exception:
            return []

    content = clean_text(content)
    if not content:
        return []

    doc_title = extract_document_title(file_path, content)
    category = file_path.parent.name
    rel_path = str(file_path.name)

    # 1. Primary Page Marker Pattern: <!-- Page X of Y -->
    page_marker_pattern = re.compile(r"<!--\s*Page\s*(\d+)\s*of\s*(\d+)\s*-->", re.IGNORECASE)
    page_matches = list(page_marker_pattern.finditer(content))

    chunks: List[Dict[str, Any]] = []

    if page_matches:
        pages_data = []
        for idx, match in enumerate(page_matches):
            p_num = int(match.group(1))
            total_pages = int(match.group(2))
            start_pos = match.start()
            end_pos = page_matches[idx + 1].start() if idx + 1 < len(page_matches) else len(content)

            page_raw = content[start_pos:end_pos].strip()
            clean_raw = re.sub(r"<!--\s*Page\s*\d+\s*of\s*\d+\s*-->", "", page_raw).strip()
            if not clean_raw:
                continue

            sections = re.findall(r"(?:มาตรา|ข้อ)\s+[๐-๙0-9]+(?:\s*(?:ทวิ|ตรี|จัตวา|เบญจ))?", clean_raw)
            unique_sections = []
            for s in sections:
                if s not in unique_sections:
                    unique_sections.append(s)

            headers = extract_major_structural_breaks(clean_raw)
            pages_data.append({
                "page_num": p_num,
                "total_pages": total_pages,
                "text": clean_raw,
                "sections": unique_sections,
                "headers": headers,
            })

        curr_group = []
        curr_len = 0

        for p in pages_data:
            has_major_new_section = False
            if p["headers"]:
                for h in p["headers"]:
                    if any(k in h for k in ["บัญชีเอกสารแนบท้าย", "บัญชีแนบท้าย", "หมวด", "แบบสัญญา", "ตารางหลักเกณฑ์", "ตารางแนบท้าย"]):
                        has_major_new_section = True
                        break

            can_merge = (
                curr_group and
                not has_major_new_section and
                (curr_len + len(p["text"]) <= max_macro_size) and
                (len(curr_group) < max_pages_per_chunk)
            )

            if can_merge:
                curr_group.append(p)
                curr_len += len(p["text"])
            else:
                if curr_group:
                    chunks.append(build_group_chunk(curr_group, file_path, doc_title, category, rel_path))
                curr_group = [p]
                curr_len = len(p["text"])

        if curr_group:
            chunks.append(build_group_chunk(curr_group, file_path, doc_title, category, rel_path))

    else:
        # Fallback for documents without <!-- Page X of Y --> markers
        alt_page_pattern = re.compile(r"^(?:#+\s*)หน้า\s*([0-9๐-๙]+)", re.MULTILINE)
        alt_matches = list(alt_page_pattern.finditer(content))

        if alt_matches:
            pages_data = []
            for idx, match in enumerate(alt_matches):
                p_num_raw = match.group(1)
                p_num = int(p_num_raw) if p_num_raw.isdigit() else idx + 1
                start_pos = match.start()
                end_pos = alt_matches[idx + 1].start() if idx + 1 < len(alt_matches) else len(content)

                page_raw = content[start_pos:end_pos].strip()
                if not page_raw:
                    continue

                headers = extract_major_structural_breaks(page_raw)
                pages_data.append({
                    "page_num": p_num,
                    "total_pages": len(alt_matches),
                    "text": page_raw,
                    "sections": [],
                    "headers": headers,
                })

            curr_group = []
            curr_len = 0
            for p in pages_data:
                can_merge = (
                    curr_group and
                    (curr_len + len(p["text"]) <= max_macro_size) and
                    (len(curr_group) < max_pages_per_chunk)
                )
                if can_merge:
                    curr_group.append(p)
                    curr_len += len(p["text"])
                else:
                    if curr_group:
                        chunks.append(build_group_chunk(curr_group, file_path, doc_title, category, rel_path))
                    curr_group = [p]
                    curr_len = len(p["text"])

            if curr_group:
                chunks.append(build_group_chunk(curr_group, file_path, doc_title, category, rel_path))
        else:
            if len(content) <= max_macro_size:
                chunks.append({
                    "chunk_id": f"{file_path.stem}_p1",
                    "source_file": rel_path,
                    "category": category,
                    "heading": doc_title,
                    "doc_title": doc_title,
                    "page_num": 1,
                    "pages_covered": [1],
                    "total_pages": 1,
                    "sections_covered": [],
                    "content": f"[{doc_title}]\n{content}",
                    "raw_text": content,
                })
            else:
                paragraphs = content.split("\n\n")
                curr_parts = []
                curr_len = 0
                part_idx = 1
                for p in paragraphs:
                    p = p.strip()
                    if not p:
                        continue
                    if curr_len + len(p) > max_macro_size and curr_parts:
                        sub_text = "\n\n".join(curr_parts)
                        heading_str = f"ส่วนที่ {part_idx}"
                        chunks.append({
                            "chunk_id": f"{file_path.stem}_p{part_idx}",
                            "source_file": rel_path,
                            "category": category,
                            "heading": heading_str,
                            "doc_title": doc_title,
                            "page_num": part_idx,
                            "pages_covered": [part_idx],
                            "total_pages": 0,
                            "sections_covered": [],
                            "content": f"[{doc_title} | {heading_str}]\n{sub_text}",
                            "raw_text": sub_text,
                        })
                        part_idx += 1
                        curr_parts = [p]
                        curr_len = len(p)
                    else:
                        curr_parts.append(p)
                        curr_len += len(p)

                if curr_parts:
                    sub_text = "\n\n".join(curr_parts)
                    heading_str = f"ส่วนที่ {part_idx}"
                    chunks.append({
                        "chunk_id": f"{file_path.stem}_p{part_idx}",
                        "source_file": rel_path,
                        "category": category,
                        "heading": heading_str,
                        "doc_title": doc_title,
                        "page_num": part_idx,
                        "pages_covered": [part_idx],
                        "total_pages": part_idx,
                        "sections_covered": [],
                        "content": f"[{doc_title} | {heading_str}]\n{sub_text}",
                        "raw_text": sub_text,
                    })

    return chunks


def chunk_faq_excel(file_path: Path) -> List[Dict[str, Any]]:
    """
    Parse FAQ Excel file into structured Q&A knowledge chunks.
    Each Q&A record is treated as an authoritative knowledge chunk.
    """
    if not file_path.exists():
        print(f"[Chunker] FAQ Excel file not found: {file_path}")
        return []

    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"[Chunker] Error reading FAQ Excel {file_path}: {e}")
        return []

    col_q = next((c for c in df.columns if "question" in str(c).lower() or "คำถาม" in str(c)), None)
    col_a = next((c for c in df.columns if "answer" in str(c).lower() or "คำตอบ" in str(c)), None)

    if not col_q or not col_a:
        if len(df.columns) >= 2:
            col_q, col_a = df.columns[0], df.columns[1]
        else:
            print(f"[Chunker] Unable to identify Q&A columns in {file_path}")
            return []

    faq_chunks = []
    total_rows = len(df)
    doc_title = "แนวทางคำถาม-คำตอบ (FAQ) กรมบัญชีกลาง"
    category = "FAQ กรมบัญชีกลาง"

    for idx, row in df.iterrows():
        q = str(row[col_q]).strip() if pd.notna(row[col_q]) else ""
        a = str(row[col_a]).strip() if pd.notna(row[col_a]) else ""
        if not q or not a:
            continue

        item_idx = idx + 1
        heading_str = f"FAQ ข้อที่ {item_idx}: {q}"
        content = (
            f"[{doc_title} | {heading_str}]\n\n"
            f"คำถาม: {q}\n\n"
            f"คำตอบ: {a}"
        )
        faq_chunks.append({
            "chunk_id": f"faq_cgd_{item_idx}",
            "source_file": file_path.name,
            "category": category,
            "heading": heading_str,
            "doc_title": doc_title,
            "page_num": item_idx,
            "pages_covered": [item_idx],
            "total_pages": total_rows,
            "sections_covered": [f"ข้อที่ {item_idx}"],
            "content": content,
            "raw_text": f"คำถาม: {q}\nคำตอบ: {a}",
        })

    print(f"[Chunker] Loaded {len(faq_chunks)} chunks from FAQ Excel '{file_path.name}'")
    return faq_chunks


def load_raw_chunks(
    chunks_file: Optional[str] = None,
    typhoon_dir: Optional[str] = None,
    faq_file: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Load chunks from chunks_file if present,
    or build structure-aware macro chunks directly from typhoon_dir markdown files and faq_file.
    """
    if chunks_file and os.path.exists(chunks_file):
        print(f"Loading pre-processed chunks from: {chunks_file}")
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks.")
        return chunks

    all_chunks = []

    # 1. Parse typhoon_ocr markdown files using page-aware macro chunking
    if typhoon_dir and os.path.exists(typhoon_dir):
        print(f"Extracting statutory chunks from: {typhoon_dir}")
        data_ocr_path = Path(typhoon_dir)
        raw_files = list(data_ocr_path.rglob("*.md"))

        priority_map = {
            "พรบ": 0,
            "ระเบียบกระทรวงการคลัง": 1,
            "กฎกระทรวง": 2,
            "ประกาศคกกนโยบาย": 3,
            "ประกาศคกกราคากลาง": 4,
            "ประกาศกรมบัญชีกลาง": 5,
        }

        def sort_key(p: Path):
            parent = p.parent.name
            return (priority_map.get(parent, 99), str(p.name))

        md_files = sorted(raw_files, key=sort_key)

        for file_path in md_files:
            if ".ipynb_checkpoints" in str(file_path):
                continue
            chunks = chunk_legal_document(file_path)
            all_chunks.extend(chunks)

        print(f"[Chunker] Loaded {len(all_chunks)} statutory macro chunks from {len(md_files)} markdown files.")

    # 2. Parse FAQ Excel file
    if faq_file and os.path.exists(faq_file):
        faq_chunks = chunk_faq_excel(Path(faq_file))
        all_chunks.extend(faq_chunks)

    print(f"Constructed total {len(all_chunks)} structure-aware knowledge chunks directly from procurement sources.")
    return all_chunks



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
    parser.add_argument("--output_dir", type=str, default="./datas", help="Output directory for corpus files")
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
