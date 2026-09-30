#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
poc_split_sections.py

Proof-of-Concept: Transform page-level Thai legal corpus (datas/law_to_crime.json)
into granular, high-precision Section-level chunks (datas/law_to_crime_section_level.json).
"""

import os
import sys
import json
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


TH_TO_AR = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
AR_TO_TH = str.maketrans("0123456789", "๐๑๒๓๔๕๖๗๘๙")


def normalize_num(text: str) -> str:
    return str(text).translate(TH_TO_AR)


def clean_header_metadata(text: str) -> str:
    """Strips the top page-level metadata tag e.g. [Document | หน้า X-Y ...]."""
    text = re.sub(r"^\[.*?\]\n*", "", text, flags=re.DOTALL)
    # Also strip recurring page markers if any
    text = re.sub(r"<!--\s*Page\s*\d+\s*of\s*\d+\s*-->", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_doc_name(chunk_id: str) -> str:
    """Extract clean document name from chunk_id (before _p or |)."""
    base = chunk_id.split("|")[0].strip()
    base = re.sub(r"_p\d+(_p\d+)?$", "", base)
    base = re.sub(r"_\d+$", "", base)
    return base.strip()


def split_text_by_sections(doc_name: str, full_doc_text: str) -> List[Dict[str, Any]]:
    """
    Splits continuous document text into individual sections/clauses (มาตรา / ข้อ).
    Handles headers, preambles, and chapters cleanly.
    """
    # Pattern to match the start of a Section or Clause:
    # e.g. "มาตรา ๕๖", "ข้อ ๗๙", "ข้อ ๑", "มาตรา ๕ ทวิ"
    # Must be at the start of a line or after double newline
    pattern = re.compile(
        r"(?=(?:^|\n)(?:(?:หมวด|ส่วนที่)\s+[๐-๙0-9]+[^\n]*\n+)?(?:มาตรา|ข้อ)\s+([๐-๙0-9]+)(?:\s*(?:ทวิ|ตรี|จัตวา|เบญจ))?)",
        flags=re.MULTILINE
    )
    
    parts = pattern.split(full_doc_text)
    
    # Alternatively, use finditer to accurately slice start and end
    # We match both Chapter/Part headers and Section identifiers
    section_regex = re.compile(
        r"(?:^|\n)(?:((?:หมวด|ส่วนที่)\s+[๐-๙0-9]+[^\n]*)\n+)?((?:มาตรา|ข้อ)\s+([๐-๙0-9]+(?:\s*(?:ทวิ|ตรี|จัตวา|เบญจ))?))",
        flags=re.MULTILINE
    )
    
    matches = list(section_regex.finditer(full_doc_text))
    if not matches:
        # Document has no standard sections (e.g. general announcement or short form)
        return [{
            "section_id": "ทั่วไป",
            "section_title": "บททั่วไป",
            "body": full_doc_text.strip()
        }]
    
    sections = []
    
    # Preamble (before first section)
    first_start = matches[0].start()
    preamble = full_doc_text[:first_start].strip()
    if len(preamble) > 100:
        sections.append({
            "section_id": "คำนำ/บทนำ",
            "section_title": "บทนำและคำปรารภ",
            "chapter": "",
            "body": preamble
        })
        
    current_chapter = ""
    for idx, m in enumerate(matches):
        start_pos = m.start()
        end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_doc_text)
        
        chapter_header = m.group(1) or ""
        if chapter_header.strip():
            current_chapter = chapter_header.strip()
            
        sec_title = m.group(2).strip()
        sec_num_thai = m.group(3).strip()
        
        sec_body = full_doc_text[start_pos:end_pos].strip()
        
        sections.append({
            "section_id": sec_title,
            "section_number": sec_num_thai,
            "section_title": sec_title,
            "chapter": current_chapter,
            "body": sec_body
        })
        
    return sections


def run_poc_extraction(input_path: str = "datas/law_to_crime.json", output_path: str = "datas/law_to_crime_section_level.json"):
    print(f"[*] Loading raw corpus from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)
        
    print(f"[+] Loaded {len(raw_chunks)} raw page-level macro chunks.")
    
    # 1. Group chunks by document to assemble full continuous document bodies
    doc_groups = defaultdict(list)
    for c in raw_chunks:
        cid = c.get("id", "")
        doc_name = extract_doc_name(cid)
        items = c.get("items", [])
        text = items[0].get("text", "") if items else ""
        clean_text = clean_header_metadata(text)
        
        # Try to extract page number from cid e.g. _p19_p21 -> 19
        p_match = re.search(r"_p(\d+)", cid)
        page_num = int(p_match.group(1)) if p_match else 9999
        doc_groups[doc_name].append({
            "cid": cid,
            "page_num": page_num,
            "text": clean_text,
            "crimes": items[0].get("crime", []) if items else []
        })
        
    print(f"[+] Found {len(doc_groups)} unique legal documents.")
    
    section_chunks = []
    stats = defaultdict(int)
    
    for doc_name, chunks_list in doc_groups.items():
        # Sort chunks by page number to ensure continuous order
        chunks_list.sort(key=lambda x: x["page_num"])
        
        # Merge text
        full_text = "\n\n".join(c["text"] for c in chunks_list if c["text"].strip())
        
        # Extract sections
        sections = split_text_by_sections(doc_name, full_text)
        stats["documents"] += 1
        stats["total_sections"] += len(sections)
        
        for sec in sections:
            sec_id = sec["section_id"]
            sec_body = sec["body"]
            chapter = sec.get("chapter", "")
            
            heading_prefix = f"[{doc_name}"
            if chapter:
                heading_prefix += f" | {chapter}"
            heading = f"{heading_prefix} | {sec_id}]"
            
            # If body is within acceptable token budget (<= 5,000 chars), keep intact
            max_limit = 5000
            if len(sec_body) <= max_limit:
                formatted_text = f"{heading}\n{sec_body}"
                chunk_full_id = f"{doc_name} | {sec_id}"
                section_chunks.append({
                    "id": chunk_full_id,
                    "doc_name": doc_name,
                    "section": sec_id,
                    "chapter": chapter,
                    "char_length": len(formatted_text),
                    "items": [
                        {
                            "text": formatted_text,
                            "crime": [f"{doc_name}_{sec_id}"],
                            "judge_dep": [doc_name],
                            "related_laws": [sec_id]
                        }
                    ]
                })
            else:
                # Split large tables / annexes into sub-chunks of ~3500 chars at paragraph boundaries
                paragraphs = sec_body.split("\n\n")
                curr_sub = []
                curr_sub_len = 0
                part_idx = 1
                
                for p in paragraphs:
                    p = p.strip()
                    if not p:
                        continue
                    if curr_sub_len + len(p) > 3500 and curr_sub:
                        sub_text = "\n\n".join(curr_sub)
                        sub_heading = f"{heading_prefix} | {sec_id} (ตอนที่ {part_idx})]"
                        sub_formatted = f"{sub_heading}\n{sub_text}"
                        section_chunks.append({
                            "id": f"{doc_name} | {sec_id} (ตอนที่ {part_idx})",
                            "doc_name": doc_name,
                            "section": sec_id,
                            "chapter": chapter,
                            "char_length": len(sub_formatted),
                            "items": [
                                {
                                    "text": sub_formatted,
                                    "crime": [f"{doc_name}_{sec_id}"],
                                    "judge_dep": [doc_name],
                                    "related_laws": [sec_id]
                                }
                            ]
                        })
                        part_idx += 1
                        curr_sub = [p]
                        curr_sub_len = len(p)
                    else:
                        curr_sub.append(p)
                        curr_sub_len += len(p)
                        
                if curr_sub:
                    sub_text = "\n\n".join(curr_sub)
                    sub_heading = f"{heading_prefix} | {sec_id} (ตอนที่ {part_idx})]" if part_idx > 1 else heading
                    sub_formatted = f"{sub_heading}\n{sub_text}"
                    sub_id = f"{doc_name} | {sec_id} (ตอนที่ {part_idx})" if part_idx > 1 else f"{doc_name} | {sec_id}"
                    section_chunks.append({
                        "id": sub_id,
                        "doc_name": doc_name,
                        "section": sec_id,
                        "chapter": chapter,
                        "char_length": len(sub_formatted),
                        "items": [
                            {
                                "text": sub_formatted,
                                "crime": [f"{doc_name}_{sec_id}"],
                                "judge_dep": [doc_name],
                                "related_laws": [sec_id]
                            }
                        ]
                    })
            
    # Deduplicate section chunks by ID to prevent duplicate slots
    seen_ids = set()
    deduped_chunks = []
    for c in section_chunks:
        cid = c["id"]
        if cid not in seen_ids:
            seen_ids.add(cid)
            deduped_chunks.append(c)

    section_chunks = deduped_chunks

    print(f"\n[+] Extraction Summary:")
    print(f"  Total Documents Processed: {stats['documents']}")
    print(f"  Total Section Chunks (Deduplicated): {len(section_chunks)}")
    
    # Length distribution analysis
    lengths = [c["char_length"] for c in section_chunks]
    avg_len = sum(lengths) / max(len(lengths), 1)
    print(f"  Average Section Length: {avg_len:.1f} chars (vs ~6,000+ chars in page-level)")
    print(f"  Min Length: {min(lengths)} chars | Max Length: {max(lengths)} chars")
    
    # Check specific target sections for verification
    test_targets = [
        "พระราชบัญญัติการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. 2560 | มาตรา ๕๖",
        "ระเบียบกระทรวงการคลังว่าด้วยการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. 2560 | ข้อ ๗๙",
        "ระเบียบกระทรวงการคลังว่าด้วยการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. 2560 | ข้อ ๒๘",
        "กฎกระทรวง กำหนดกรณีการจัดซื้อจัดจ้างพัสดุโดยวิธีเฉพาะเจาะจง พ.ศ. 2561 | ข้อ ๑"
    ]
    
    print("\n🔍 Target Sections Verification:")
    for target in test_targets:
        matches = [c for c in section_chunks if target in c["id"]]
        if matches:
            m = matches[0]
            print(f"  ✅ FOUND: {m['id']} (len: {m['char_length']} chars)")
            print(f"     Preview: {m['items'][0]['text'][:180].replace(chr(10), ' ')}...")
        else:
            print(f"  ❌ NOT FOUND: {target}")
            
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(section_chunks, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Saved section-level corpus to: {output_path}")


if __name__ == "__main__":
    run_poc_extraction()
