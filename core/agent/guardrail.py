# -*- coding: utf-8 -*-
"""
core/agent/guardrail.py

Deterministic Grounding & Fact-Checking Guardrail (0-LLM, ~5ms).
Replaces the expensive LLM-based CompletenessAuditor.

Responsibilities:
1. Scan synthesized response for statutory citations (e.g. มาตรา 56, ข้อ 79, ว.845).
2. Cross-reference citations against the retrieved contexts in `AgentState`.
3. Intercept ungrounded or fictitious legal sections (Anti-Hallucination).
4. Validate financial threshold ceilings (e.g. 500,000 THB ceiling for specific selection).
"""

import re
from typing import Dict, Any, List, Set, Tuple

from core.compliance_constants import SPECIAL_CASE_QUALIFIERS, SPECIFIC_METHOD_CEILING_THB, fmt_thb

# Spelled-out amounts the LLM tends to copy verbatim from statute text
TH_WORD_AMOUNTS = {"หนึ่งแสน": 100000, "ห้าแสน": 500000, "หนึ่งล้าน": 1000000, "สองล้าน": 2000000, "ห้าล้าน": 5000000}

TH_TO_AR = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def normalize_digits(text: str) -> str:
    """Normalize Thai numerals to Arabic digits."""
    if not text:
        return ""
    return str(text).translate(TH_TO_AR)


class GroundingGuardrail:
    """High-speed deterministic legal guardrail."""

    SECTION_PATTERN = re.compile(r"(?:มาตรา|ม\.)\s*([0-9๐-๙]+(?:\s*\([0-9๐-๙a-zA-Zก-ฮ]+\))*)")
    CLAUSE_PATTERN = re.compile(r"(?:ข้อ|ระเบียบฯ\s*ข้อ)\s*([0-9๐-๙]+(?:\s*\([0-9๐-๙a-zA-Zก-ฮ]+\))*)")
    CIRCULAR_PATTERN = re.compile(r"(?:ว\s*[0-9๐-๙]+|หนังสือเวียน\s*ที่\s*[^\s,]+)")

    @classmethod
    def extract_citations(cls, text: str) -> Dict[str, Set[str]]:
        """Extract sections, clauses, and circular citations from text."""
        norm_text = normalize_digits(text or "")
        sections = set(re.findall(r"(?:มาตรา|ม\.)\s*(\d+)", norm_text))
        clauses = set(re.findall(r"(?:ข้อ|ระเบียบฯ\s*ข้อ)\s*(\d+)", norm_text))
        circulars = set(re.findall(r"(?:ว\s*\d+)", norm_text))
        return {
            "sections": sections,
            "clauses": clauses,
            "circulars": circulars
        }

    @classmethod
    def build_grounded_index(cls, contexts: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
        """Builds lookup set of legal sections actually present in retrieved contexts."""
        grounded_sections = set()
        grounded_clauses = set()
        grounded_circulars = set()

        for ctx in contexts or []:
            combined = f"{ctx.get('entry', '')} {ctx.get('content_thai', '')} {ctx.get('text', '')} {str(ctx.get('related_laws', ''))}"
            norm = normalize_digits(combined)
            grounded_sections.update(re.findall(r"(?:มาตรา|ม\.)\s*(\d+)", norm))
            grounded_clauses.update(re.findall(r"(?:ข้อ|ระเบียบฯ\s*ข้อ)\s*(\d+)", norm))
            grounded_circulars.update(re.findall(r"(?:ว\s*\d+)", norm))
            
            # Check numerical section_num and clause_num directly
            if ctx.get("section_num"):
                grounded_sections.add(str(ctx["section_num"]))
            if ctx.get("clause_num"):
                grounded_clauses.add(str(ctx["clause_num"]))

        return {
            "sections": grounded_sections,
            "clauses": grounded_clauses,
            "circulars": grounded_circulars
        }

    @staticmethod
    def extract_stated_ceilings(norm_text: str) -> Set[int]:
        """Amounts phrased as a ceiling ("ไม่เกิน X บาท") in digits or spelled-out Thai."""
        found: Set[int] = set()
        for m in re.finditer(r"ไม่เกิน\s*(?:วงเงิน\s*)?([\d,]{5,})\s*บาท", norm_text):
            found.add(int(m.group(1).replace(",", "")))
        for word, amount in TH_WORD_AMOUNTS.items():
            if re.search(rf"ไม่เกิน\s*(?:วงเงิน\s*)?{word}บาท", norm_text):
                found.add(amount)
        return found

    @classmethod
    def audit(cls, synthesized_text: str, retrieved_contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes deterministic audit in ~2-5ms:
        Returns:
            passed (bool), grounded_citations (list), ungrounded_citations (list), warnings (list)
        """
        if not synthesized_text:
            return {"passed": False, "score": 0.0, "reason": "Empty synthesized text"}

        cited = cls.extract_citations(synthesized_text)
        grounded = cls.build_grounded_index(retrieved_contexts)

        ungrounded_sections = [s for s in cited["sections"] if s not in grounded["sections"]]
        ungrounded_clauses = [c for c in cited["clauses"] if c not in grounded["clauses"]]
        
        warnings = []
        is_passed = True

        if ungrounded_sections:
            warnings.append(f"อ้างอิงมาตราที่ไม่มีในหลักฐานที่ค้นพบ: มาตรา {', '.join(ungrounded_sections)}")
            # If majority of cited sections are ungrounded, fail grounding check
            if len(ungrounded_sections) > len(cited["sections"]) / 2:
                is_passed = False

        if ungrounded_clauses:
            warnings.append(f"อ้างอิงข้อระเบียบที่ไม่มีในหลักฐานที่ค้นพบ: ข้อ {', '.join(ungrounded_clauses)}")

        # Verification of specific selection financial ceiling
        norm_synth = normalize_digits(synthesized_text)
        if "เฉพาะเจาะจง" in norm_synth:
            # Check if mentioned budget > 500,000 without citing exception (ข้อยกเว้น)
            has_exception = any(kw in norm_synth for kw in ["ข้อยกเว้น", "จำเป็นเร่งด่วน", "ฉุกเฉิน", "มีรายเดียว", "วรรคสอง", "(๒)"])
            budget_matches = [int(b.replace(",", "")) for b in re.findall(r"(?:วงเงิน|งบประมาณ)\s*([\d,]+)\s*บาท", norm_synth)]
            for b in budget_matches:
                if b > SPECIFIC_METHOD_CEILING_THB and not has_exception:
                    warnings.append(f"วงเงิน {b:,} บาท เกิน {fmt_thb(SPECIFIC_METHOD_CEILING_THB)} บาทสำหรับวิธีเฉพาะเจาะจง แต่คำตอบไม่ได้ระบุเงื่อนไขข้อยกเว้นอย่างชัดเจน")

            # A ceiling other than the general one is only valid for a special case (e.g. MoE schools: 1,000,000)
            stated = cls.extract_stated_ceilings(norm_synth)
            # lower amounts are other legit thresholds (e.g. มาตรา 96: 100,000), so only higher ones are suspect
            wrong = sorted(c for c in stated if c > SPECIFIC_METHOD_CEILING_THB)
            if wrong and not any(q in norm_synth for q in SPECIAL_CASE_QUALIFIERS):
                is_passed = False
                warnings.append(
                    f"คำตอบระบุเพดานวิธีเฉพาะเจาะจง {', '.join(fmt_thb(c) for c in wrong)} บาท ซึ่งสูงกว่าเพดานทั่วไป "
                    f"{fmt_thb(SPECIFIC_METHOD_CEILING_THB)} บาท และไม่ได้ระบุว่าเป็นกรณีเฉพาะ (เช่น สถานศึกษาสังกัดกระทรวงศึกษาธิการ)"
                )

        # Grounding Score Calculation
        total_cited = len(cited["sections"]) + len(cited["clauses"])
        grounded_count = total_cited - (len(ungrounded_sections) + len(ungrounded_clauses))
        score = (grounded_count / total_cited) if total_cited > 0 else 1.0

        return {
            "passed": is_passed,
            "grounding_score": round(score, 2),
            "cited_sections": list(cited["sections"]),
            "ungrounded_sections": ungrounded_sections,
            "ungrounded_clauses": ungrounded_clauses,
            "warnings": warnings,
        }
