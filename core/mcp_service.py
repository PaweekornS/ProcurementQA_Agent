# -*- coding: utf-8 -*-
"""
core/mcp_service.py

Domain service providing decoupled access to LegalGraphRAG capabilities:
- Exact statutory clause lookup (0-LLM, ~10ms)
- Hybrid search over statutory clauses without synthesis (~100-300ms)
- FAQ precedent search over Comptroller General cases (~50ms)
- Knowledge graph neighbor and subordinate legislation traversal (~10ms)
- Rule-based procurement compliance verification against statutory thresholds
- CRAG reasoning (fast and deep modes)
- Resources (thresholds & catalog)
"""

import os
import sys
import re
import json
from typing import Dict, Any, List, Optional, Tuple

from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig
from core.graph_construct.graph_db import GraphDBManager


# Thai to Arabic digits mapping and vice versa
THAI_TO_ARABIC = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
ARABIC_TO_THAI = str.maketrans("0123456789", "๐๑๒๓๔๕๖๗๘๙")


def normalize_digits(text: str) -> str:
    """Convert Thai numerals to Arabic digits for normalized comparison."""
    if not text:
        return ""
    return str(text).translate(THAI_TO_ARABIC)


def to_thai_digits(text: str) -> str:
    """Convert Arabic numerals to Thai numerals."""
    if not text:
        return ""
    return str(text).translate(ARABIC_TO_THAI)


class ProcurementService:
    """
    Singleton service managing LegalGraphRAG components, cached statutory lookups,
    and compliance rule-checking.
    """
    _instance: Optional["ProcurementService"] = None

    def __init__(
        self,
        dotenv_path: Optional[str] = None,
        config: Optional[LegalGraphRAGConfig] = None,
        auto_build: Optional[bool] = None
    ):
        self.dotenv_path = dotenv_path or os.getenv("DOTENV_PATH", ".env")
        self.config = config or LegalGraphRAGConfig.from_env_file(self.dotenv_path)
        
        # Explicit override
        if auto_build is not None:
            self.config.graph.auto_build = auto_build
        elif os.getenv("AUTO_BUILD") is not None:
            self.config.graph.auto_build = os.getenv("AUTO_BUILD", "True").lower() in ("true", "1", "yes")
        elif os.getenv("DISABLE_AUTO_BUILD") == "1":
            self.config.graph.auto_build = False
            
        self.rag = LegalGraphRAG(config=self.config)
        self._section_index: Dict[str, List[Dict[str, Any]]] = {}
        self._build_section_lookup_index()
        self._warmup_models()

    def _warmup_models(self):
        try:
            from core.graph_construct.feature_graph import get_embedding
            from core.graph_construct.hybrid_reranker import get_reranker, is_reranker_enabled

            if is_reranker_enabled():
                import torch
                reranker_model = os.getenv("reranker_model", "BAAI/bge-reranker-v2-m3")
                reranker_device = os.getenv("reranker_device", "cuda:0" if (torch and torch.cuda.is_available()) else "cpu")
                reranker_thresh = float(os.getenv("reranker_threshold", "0.20"))

                print(f"[ProcurementService] Pre-warming CrossEncoder '{reranker_model}' on '{reranker_device}'...")
                reranker = get_reranker(model_name=reranker_model, device=reranker_device, threshold=reranker_thresh)
                if reranker and reranker.model:
                    print(f"[ProcurementService] CrossEncoder successfully pre-warmed on '{reranker.device}'.")
            else:
                print("[ProcurementService] Reranker is disabled (enable_reranker=false). Running in Pure Hybrid Retriever mode.")

            _ = get_embedding("warmup query")
            print("[ProcurementService] Embedder successfully pre-warmed.")
        except Exception as e:
            print(f"[ProcurementService Warmup Warning] {e}")

    @classmethod
    def get_instance(
        cls,
        dotenv_path: Optional[str] = None,
        config: Optional[LegalGraphRAGConfig] = None,
        auto_build: Optional[bool] = None
    ) -> "ProcurementService":
        if cls._instance is None:
            cls._instance = cls(dotenv_path=dotenv_path, config=config, auto_build=auto_build)
        return cls._instance

    # --------------------------------------------------------------------------
    # Tier 1: Indexing & Atomic Lookup
    # --------------------------------------------------------------------------

    def _build_section_lookup_index(self):
        """Build fast in-memory index mapping section keys to statutory nodes."""
        laws = self.rag.law_to_crime or []
        for law in laws:
            items = law.get("items", [])
            entry_id = law.get("id", "")
            for item in items:
                related = item.get("related_laws", [])
                text = item.get("text", "")
                topics = item.get("crime", [])

                node_entry = {
                    "id": entry_id,
                    "text": text,
                    "topics": topics,
                    "related_laws": related,
                    "judge_dep": item.get("judge_dep", [])
                }

                # Index by exact related items (e.g. 'มาตรา ๕๖', 'ข้อ ๗๙')
                for r in related:
                    r_norm = normalize_digits(r.strip())
                    if r_norm:
                        self._section_index.setdefault(r_norm.lower(), []).append(node_entry)
                        # Also index just the number if starts with มาตรา or ข้อ
                        num_match = re.search(r"(\d+)", r_norm)
                        if num_match:
                            num = num_match.group(1)
                            if "มาตรา" in r:
                                self._section_index.setdefault(f"มาตรา {num}", []).append(node_entry)
                                self._section_index.setdefault(f"sec_{num}", []).append(node_entry)
                            elif "ข้อ" in r:
                                self._section_index.setdefault(f"ข้อ {num}", []).append(node_entry)
                                self._section_index.setdefault(f"rule_{num}", []).append(node_entry)

    def lookup_section(self, section: str, doc_title: Optional[str] = None) -> Dict[str, Any]:
        """
        Exact statutory section lookup without generative overhead.
        
        Args:
            section: Section string e.g. "มาตรา 56", "56", "ข้อ 79", "มาตรา ๕๖ (๒) (ข)"
            doc_title: Optional filter by document title (e.g. "พระราชบัญญัติ", "ระเบียบ")
        """
        if not section or not section.strip():
            return {"found": False, "error": "section parameter is required"}

        sec_norm = normalize_digits(section.strip()).lower()
        candidates = self._section_index.get(sec_norm, [])

        if not candidates:
            # Try matching base section like "มาตรา 56" or "ข้อ 79" from sub-clause "มาตรา 56 (2) (ข)"
            m_base = re.search(r"(มาตรา|ข้อ)\s*(\d+)", sec_norm)
            if m_base:
                base_key = f"{m_base.group(1)} {m_base.group(2)}"
                candidates = self._section_index.get(base_key, [])

        if not candidates:
            # Try prepending มาตรา or ข้อ if user supplied just a number
            if re.match(r"^\d+", sec_norm):
                candidates = self._section_index.get(f"มาตรา {sec_norm}", [])
                if not candidates:
                    candidates = self._section_index.get(f"ข้อ {sec_norm}", [])

        if not candidates:
            # Substring scan across keys
            for k, val in self._section_index.items():
                if sec_norm in k or k in sec_norm:
                    candidates.extend(val)
                    break

        if not candidates:
            return {
                "found": False,
                "section": section,
                "message": f"No statutory clause found matching section: {section}"
            }

        # Filter by doc_title if specified
        matched_item = None
        if doc_title:
            doc_norm = doc_title.strip().lower()
            for cand in candidates:
                cand_id = cand["id"].lower()
                cand_topics = " ".join(cand["topics"]).lower()
                if doc_norm in cand_id or doc_norm in cand_topics:
                    matched_item = cand
                    break

        if not matched_item:
            matched_item = candidates[0]

        # Extract focused paragraph for the specific section from the macro chunk if possible
        full_text = matched_item["text"]
        focused_text = self._extract_focused_section_text(full_text, section)

        return {
            "found": True,
            "section": section,
            "source_id": matched_item["id"],
            "topics": matched_item["topics"],
            "related_laws": matched_item.get("related_laws", []),
            "judge_dep": matched_item.get("judge_dep", []),
            "focused_content": focused_text or full_text[:1500],
            "full_macro_chunk": full_text
        }

    def _extract_focused_section_text(self, macro_text: str, section: str) -> Optional[str]:
        """Extract only the lines relevant to the requested section."""
        sec_num = normalize_digits(section)
        num_match = re.search(r"(\d+)", sec_num)
        if not num_match:
            return None
        target_num = num_match.group(1)
        thai_num = to_thai_digits(target_num)

        # Pattern matching 'มาตรา 56' or 'มาตรา ๕๖' or 'ข้อ 79' or 'ข้อ ๗๙'
        pattern = rf"(?:มาตรา|ข้อ)\s*(?:{target_num}|{thai_num})\b[\s\S]*?(?=(?:มาตรา|ข้อ)\s*(?:\d+|[๐-๙]+)\b|#|\Z)"
        match = re.search(pattern, macro_text)
        if match:
            return match.group(0).strip()
        return None

    def search_clauses(self, query: str, top_k: int = 5, doc_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Direct hybrid search (Dense Vector + Thai BM25 + GPU Cross-Encoder Reranker)
        over statutory clauses without LLM synthesis.
        """
        if not query or not query.strip():
            return []

        from core.graph_construct.feature_graph import search_similar_nodes_direct, get_embedding

        query_emb = get_embedding(query.strip())
        cases, laws = search_similar_nodes_direct(
            self.rag.model,
            query_emb,
            query.strip(),
            top_k=top_k * 2 if doc_filter else top_k
        )

        results = []
        for law in laws:
            data = law.get("data", {}) or law
            entry = data.get("entry") or law.get("id", "")
            desc = data.get("description") or law.get("description", "")
            topics = data.get("crimes", data.get("crime", []))

            if doc_filter:
                combined_meta = f"{entry} {' '.join(topics)}".lower()
                if doc_filter.lower() not in combined_meta:
                    continue

            results.append({
                "id": law.get("id"),
                "entry": entry,
                "topics": topics,
                "content": desc[:800],
                "score": round(float(law.get("rerank_score", law.get("similarity", 0.0))), 4)
            })

            if len(results) >= top_k:
                break

        return results

    def search_faqs(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search historical rulings and Comptroller General FAQs (cases_with_feature.json).
        """
        if not query or not query.strip():
            return []

        q_terms = set(re.findall(r"\w+", query.lower()))
        cases = self.rag.cases_db or []
        scored_cases = []

        for c in cases:
            fact = c.get("fact", "")
            laws = c.get("law", [])
            topics = c.get("crime", [])
            text_to_match = f"{fact} {' '.join(laws)} {' '.join(topics)}".lower()

            # Simple token overlap score + length normalization
            match_count = sum(1 for term in q_terms if term in text_to_match)
            if match_count > 0:
                score = match_count / (len(q_terms) + 0.1)
                scored_cases.append((score, c))

        scored_cases.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, c in scored_cases[:top_k]:
            fact = c.get("fact", "")
            parts = fact.split("แนวทางวินิจฉัย/คำตอบ:")
            question_part = parts[0].replace("ข้อหารือ/คำถาม:", "").strip()
            answer_part = parts[1].strip() if len(parts) > 1 else fact

            results.append({
                "faq_id": c.get("id"),
                "question": question_part,
                "answer": answer_part,
                "laws": c.get("law", []),
                "topics": c.get("crime", []),
                "relevance_score": round(score, 3)
            })

        return results

    # --------------------------------------------------------------------------
    # Tier 2: Knowledge Graph Traversal
    # --------------------------------------------------------------------------

    def traverse_regulations(self, section_reference: str) -> Dict[str, Any]:
        """
        Traverse the Knowledge Graph to find subordinate rules, ministerial regulations,
        or circular letters linked to a parent statutory section.
        """
        if not section_reference or not section_reference.strip():
            return {"parent": section_reference, "related_nodes": []}

        db = GraphDBManager.get_db()
        norm_ref = normalize_digits(section_reference.strip()).lower()

        # 1. Search for matching node in the graph
        matched_node_id = None
        for node_id in db.nodes_data:
            if norm_ref in normalize_digits(node_id).lower():
                matched_node_id = node_id
                break

        related_results = []
        if matched_node_id:
            # Get neighbors from graph
            neighbors = list(db.graph.neighbors(matched_node_id))
            for n_id in neighbors:
                n_info = db.nodes_data.get(n_id, {})
                edge_data = db.graph.get_edge_data(matched_node_id, n_id) or {}
                related_results.append({
                    "node_id": n_id,
                    "type": n_info.get("type", "Unknown"),
                    "relation": list(edge_data.keys())[0] if isinstance(edge_data, dict) and edge_data else "RELATED",
                    "description": n_info.get("data", {}).get("description", "")[:400]
                })

        # 2. Also retrieve related items from section lookup index
        lookup_res = self.lookup_section(section_reference)
        cross_laws = []
        if lookup_res.get("found"):
            cross_laws = lookup_res.get("topics", [])

        return {
            "target": section_reference,
            "matched_graph_node": matched_node_id,
            "graph_neighbors_count": len(related_results),
            "related_nodes": related_results,
            "associated_topics": cross_laws
        }

    # --------------------------------------------------------------------------
    # Tier 3: Compliance Engine & CRAG Pipeline
    # --------------------------------------------------------------------------

    def verify_compliance(
        self,
        procurement_item: str,
        estimated_budget: float,
        proposed_method: str,
        justification_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluates structured procurement project parameters against Thai statutory thresholds:
        - พระราชบัญญัติการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. 2560 (ม. 55, 56)
        - กฎกระทรวงกำหนดวงเงินการจัดซื้อจัดจ้างพัสดุโดยวิธีเฉพาะเจาะจง พ.ศ. 2560
        - ระเบียบกระทรวงการคลังว่าด้วยการจัดซื้อจัดจ้างฯ พ.ศ. 2560 (ข้อ 79)
        """
        method_clean = proposed_method.strip()
        reason = (justification_reason or "").strip()
        risks = []
        approvals = ["หัวหน้าเจ้าหน้าที่", "หัวหน้าหน่วยงานของรัฐ"]
        is_compliant = True
        status = "PASSED"
        statutory_threshold = ""
        legal_basis = []

        is_specific = "เฉพาะเจาะจง" in method_clean
        is_ebidding = "e-bidding" in method_clean.lower() or "ประกวดราคา" in method_clean
        is_emarket = "e-market" in method_clean.lower() or "ตลาดอิเล็กทรอนิกส์" in method_clean
        is_selection = "คัดเลือก" in method_clean

        if is_specific:
            statutory_threshold = "วงเงินไม่เกิน 500,000 บาท ตามกฎกระทรวงกำหนดวงเงินฯ พ.ศ. 2560"
            legal_basis.append("พ.ร.บ. จัดซื้อจัดจ้างฯ 2560 มาตรา ๕๖ (๒) (ข)")
            legal_basis.append("ระเบียบกระทรวงการคลังฯ 2560 ข้อ ๗๙")

            if estimated_budget > 500000.0:
                # Check whether special justification allows specific method > 500,000 THB
                valid_exceptions = [
                    "เร่งด่วน", "ฉุกเฉิน", "ราชการลับ", "ที่ดิน", "สิ่งปลูกสร้าง",
                    "ไม่มีผู้ยื่น", "ยกเลิกการประกวดราคา", "ตัวแทนจำหน่ายแต่ผู้เดียว",
                    "สิทธิบัตร", "จำเป็นต้องใช้โดยตรง"
                ]
                has_valid_exception = any(kw in reason for kw in valid_exceptions)

                if not has_valid_exception:
                    is_compliant = False
                    status = "VIOLATION"
                    risks.append(
                        f"วงเงิน {estimated_budget:,.2f} บาท เกินเพดานวิธีเฉพาะเจาะจง 500,000 บาท "
                        "และไม่มีเหตุผลยกเว้นตามมาตรา ๕๖ (๒) (ก), (ค), (ง), (จ), (ฉ), (ช) หรือ (ซ)"
                    )
                else:
                    status = "FLAGGED"
                    risks.append(
                        f"วงเงินเกิน 500,000 บาท ต้องมีบันทึกรายงานความจำเป็นชี้แจงเหตุผลความเร่งด่วน/ความเฉพาะเจาะจง "
                        "พร้อมเอกสารหลักฐานประกอบอย่างเคร่งครัด"
                    )
                    approvals.append("คณะกรรมการหรือผู้มีอำนาจสั่งซื้อสั่งจ้างตามระเบียบฯ")

            if estimated_budget <= 100000.0:
                # Small amount exemption for agreement in writing
                legal_basis.append("พ.ร.บ. มาตรา ๙๖ วรรคสอง (การจัดทำข้อตกลงเป็นหนังสือ)")

        elif is_emarket or is_ebidding:
            statutory_threshold = "วงเงินเกิน 500,000 บาทขึ้นไป (วิธีประกาศเชิญชวนทั่วไป)"
            legal_basis.append("พ.ร.บ. จัดซื้อจัดจ้างฯ 2560 มาตรา ๕๕ (๑)")
            legal_basis.append("ระเบียบกระทรวงการคลังฯ 2560 ข้อ ๒๙")

            if estimated_budget <= 500000.0:
                status = "FLAGGED"
                risks.append(
                    f"วงเงิน {estimated_budget:,.2f} บาท ไม่เกิน 500,000 บาท โดยปกติสามารถใช้วิธีเฉพาะเจาะจงได้ "
                    "เพื่อความคล่องตัวและประหยัดระยะเวลา"
                )

        # Anti-splitting check (ห้ามแบ่งซื้อแบ่งจ้าง มาตรา ๖๕)
        risks.append(
            "ข้อควรระวัง: ห้ามมิให้แบ่งวงเงินเพื่อลดวงเงินจัดซื้อจัดจ้างโดยมุ่งหมายให้อำนาจสั่งซื้อสั่งจ้างเปลี่ยนแปลงไป "
            "หรือเพื่อหลีกเลี่ยงการจัดซื้อจัดจ้างโดยวิธีประกาศเชิญชวนทั่วไป (มาตรา ๖๕)"
        )

        return {
            "procurement_item": procurement_item,
            "estimated_budget": estimated_budget,
            "proposed_method": proposed_method,
            "is_compliant": is_compliant,
            "compliance_status": status,
            "statutory_threshold": statutory_threshold,
            "legal_basis": legal_basis,
            "required_approvals": approvals,
            "potential_risks": risks
        }

    def ask_procurement_law(self, question: str, mode: str = "deep") -> Dict[str, Any]:
        """
        Execute CRAG synthesis pipeline.
        mode="deep": full CRAG with Issue Decomposer, Synthesizer, Auditor, Refiner Retry.
        mode="fast": single-pass hybrid retrieval + Synthesizer without auditor retry loop.
        """
        if not question or not question.strip():
            return {
                "status": "ERROR",
                "direct_answer": "",
                "decisive_quotes": [],
                "applicable_laws": [],
                "exceptions_or_conditions": "",
                "citations": [],
                "crag_meta": {},
                "error": "`question` must be a non-empty string."
            }

        case = {"fact": question.strip(), "name": "ผู้สอบถาม"}

        # Temporarily override CRAG retry configuration based on mode
        original_retry = getattr(self.rag.config.crag, "max_retry", 1)
        original_enabled = getattr(self.rag.config.crag, "enabled", True)

        try:
            if mode == "fast":
                self.rag.config.crag.max_retry = 0
            else:
                self.rag.config.crag.max_retry = max(1, original_retry)

            results: List[Dict[str, Any]] = self.rag.analyze_case(case)
        finally:
            self.rag.config.crag.max_retry = original_retry
            self.rag.config.crag.enabled = original_enabled

        if not results:
            return {
                "status": "ERROR",
                "direct_answer": "",
                "decisive_quotes": [],
                "applicable_laws": [],
                "exceptions_or_conditions": "",
                "citations": [],
                "crag_meta": {},
                "error": "Pipeline returned no results."
            }

        item = results[0]
        judge_result = item.get("judge_result", {}) or {}
        used_laws = item.get("used_laws", []) or []

        raw_quotes = judge_result.get("decisive_quotes") or judge_result.get("decisive_quote") or []
        enriched_quotes = self._enrich_decisive_quotes(raw_quotes, used_laws)

        return {
            "status": judge_result.get("status", "OK"),
            "mode": mode,
            "direct_answer": judge_result.get("direct_answer", ""),
            "decisive_quotes": enriched_quotes,
            "applicable_laws": judge_result.get("applicable_laws", []),
            "exceptions_or_conditions": judge_result.get("exceptions_or_conditions", ""),
            "citations": [
                {
                    "entry": law.get("entry", ""),
                    "topics": law.get("crimes", law.get("crime", []))
                }
                for law in used_laws
            ],
            "crag_meta": item.get("crag_meta", {})
        }

    def _enrich_decisive_quotes(
        self,
        raw_quotes: List[Any],
        used_laws: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enrich decisive quotes with statutory source metadata:
        - filename: source document markdown filename
        - page: page number or page range in document (e.g. '4-6/42')
        - law: exact section or rule identifier
        - quote: verbatim quoted text
        """
        enriched = []
        if not raw_quotes:
            return enriched

        for q in raw_quotes:
            if not isinstance(q, dict):
                if isinstance(q, str) and q.strip():
                    enriched.append({
                        "filename": "ไม่ระบุ",
                        "page": "ไม่ระบุ",
                        "law": "ไม่ระบุ",
                        "quote": q.strip()
                    })
                continue

            law_name = str(q.get("law", "")).strip()
            quote_text = str(q.get("quote", "")).strip()

            filename = ""
            page = ""

            # 1. Match against used_laws from retrieval
            norm_law = normalize_digits(law_name).lower()
            law_num_match = re.search(r"(\d+)", norm_law)
            law_num = law_num_match.group(1) if law_num_match else None

            matched_law = None
            if law_name or quote_text:
                for cand in used_laws:
                    cand_entry = normalize_digits(str(cand.get("entry", ""))).lower()
                    cand_id = normalize_digits(str(cand.get("id", ""))).lower()
                    cand_desc = str(cand.get("description", ""))
                    cand_related = [normalize_digits(str(r)).lower() for r in cand.get("related_laws", [])]

                    # Match by exact section identifier or number
                    if (norm_law and (norm_law in cand_entry or norm_law in cand_id or any(norm_law in r for r in cand_related))) or \
                       (law_num and (f" {law_num}" in cand_entry or f" {law_num}" in cand_id or any(f" {law_num}" in r for r in cand_related))) or \
                       (quote_text and len(quote_text) > 15 and quote_text[:30] in cand_desc):
                        matched_law = cand
                        break

            # 2. Extract metadata from matched used_law
            if matched_law:
                # Find filename from related_laws or id
                related = matched_law.get("related_laws", [])
                for r in related:
                    if str(r).endswith((".md", ".pdf")):
                        filename = str(r)
                        break
                if not filename:
                    entry_raw = matched_law.get("entry") or matched_law.get("id") or ""
                    parts = str(entry_raw).split("|")[0].strip()
                    parts = re.sub(r"_p\d+.*$", "", parts).strip()
                    if parts:
                        filename = parts if parts.endswith(".md") else f"{parts}.md"

                # Find page from judge_dep or id or description
                judge_dep = str(matched_law.get("judge_dep", ""))
                p_match = re.search(r"หน้า\s*([0-9\-\/]+)", judge_dep)
                if p_match:
                    page = p_match.group(1)
                else:
                    id_raw = str(matched_law.get("id", ""))
                    pid_match = re.search(r"_p(\d+(?:_p\d+)?)", id_raw)
                    if pid_match:
                        page = pid_match.group(1).replace("_p", "-")

            # 3. Fallback to lookup_section index if filename or page is still missing
            if (not filename or not page) and law_name:
                lookup_res = self.lookup_section(law_name)
                if lookup_res.get("found"):
                    source_id = str(lookup_res.get("source_id", ""))
                    topics = lookup_res.get("topics", [])
                    raw_text = lookup_res.get("full_macro_chunk", "")
                    related = lookup_res.get("related_laws", [])
                    judge_dep = str(lookup_res.get("judge_dep", ""))

                    if not filename:
                        for r in related:
                            if str(r).endswith((".md", ".pdf")):
                                filename = str(r)
                                break
                    if not filename:
                        fn_match = re.search(r"^\[(.*?)\s*\|", raw_text)
                        if fn_match:
                            fn_title = fn_match.group(1).strip()
                            filename = fn_title if fn_title.endswith(".md") else f"{fn_title}.md"
                        elif topics:
                            filename = f"{topics[-1]}.md"

                    if not page:
                        p_match = re.search(r"หน้า\s*([0-9\-\/]+)", judge_dep) or re.search(r"หน้า\s*([0-9\-\/]+)", raw_text)
                        if p_match:
                            page = p_match.group(1)
                        else:
                            pid_match = re.search(r"_p(\d+(?:_p\d+)?)", source_id)
                            if pid_match:
                                page = pid_match.group(1).replace("_p", "-")

            enriched.append({
                "filename": filename or "ไม่ระบุ",
                "page": page or "ไม่ระบุ",
                "law": law_name or "ไม่ระบุ",
                "quote": quote_text
            })

        return enriched

    # Alias for consistent high-level agent naming
    procurement_qa = ask_procurement_law
    check_procurement_threshold = verify_compliance

    # --------------------------------------------------------------------------
    # Tier 4: Resources
    # --------------------------------------------------------------------------

    def get_thresholds_resource(self) -> str:
        """Markdown summary of statutory monetary thresholds and procedural rules."""
        return """# เกณฑ์วงเงินและข้อกำหนดตามกฎหมายจัดซื้อจัดจ้างภาครัฐไทย (พ.ร.บ. 2560)

## 1. วิธีการจัดซื้อจัดจ้างพัสดุ (มาตรา 55, 56)
- **วิธีเฉพาะเจาะจง (Specific Selection)**:
  - วงเงินเล็กน้อย: **ไม่เกิน 500,000 บาท** (ตามกฎกระทรวงกำหนดวงเงินฯ พ.ศ. 2560)
  - วงเงินเกิน 500,000 บาท ต้องมีข้อยกเว้นตามมาตรา 56 (2) เช่น จำเป็นเร่งด่วนฉุกเฉิน (ง), เป็นพัสดุที่มีตัวแทนจำหน่ายแต่เพียงผู้เดียว (ค), ยกเลิก e-bidding แล้วไม่มีผู้ยื่น (ก)
- **วิธีประกาศเชิญชวนทั่วไป (General Invitation)**:
  - วงเงิน **เกิน 500,000 บาทขึ้นไป**
  - **e-Market**: พัสดุมีมาตรฐาน อยู่ในระบบ e-catalog
  - **e-Bidding**: พัสดุที่มีความซับซ้อน หรือไม่อยู่ใน e-catalog
- **วิธีคัดเลือก (Selective Method)**:
  - เชิญชวนผู้ประกอบการไม่น้อยกว่า 3 ราย ตามเงื่อนไขมาตรา 56 (1)

## 2. ข้อยกเว้นการทำสัญญาเป็นหนังสือ (มาตรา 96)
- วงเงิน **ไม่เกิน 100,000 บาท** จะไม่ทำข้อตกลงเป็นหนังสือก็ได้ โดยใช้ใบสั่งซื้อสั่งจ้างหรือใบเสร็จรับเงินแทน

## 3. สิทธิและการยื่นอุทธรณ์ (หมวด 6 มาตรา 114 - 119)
- **กำหนดเวลายื่นอุทธรณ์**: ต้องยื่นต่อหน่วยงานของรัฐ **ภายใน 7 วันทำการ** นับแต่วันประกาศผลการจัดซื้อจัดจ้างในระบบ e-GP
- **ข้อยกเว้นการอุทธรณ์ (มาตรา 115)**: อุทธรณ์ไม่ได้ในกรณีการเลือกวิธีจัดซื้อจัดจ้าง, การยกเลิกการจัดซื้อจัดจ้าง, หรือเรื่องการปฏิเสธไม่รับข้อเสนอที่ผิดเงื่อนไขสำคัญ

## 4. ข้อห้ามการแบ่งซื้อแบ่งจ้าง (มาตรา 65)
- ห้ามแบ่งซื้อแบ่งจ้างพัสดุโดยเจตนาเพื่อลดวงเงินให้ต่ำกว่าเกณฑ์ เพื่อให้เปลี่ยนวิธีจัดซื้อจัดจ้างหรือเปลี่ยนผู้มีอำนาจสั่งซื้อสั่งจ้าง
"""

    def get_catalog_resource(self) -> Dict[str, Any]:
        """Catalog of indexed statutes, regulations, and cases."""
        laws = self.rag.law_to_crime or []
        cases = self.rag.cases_db or []
        doc_titles = set()
        for law in laws:
            topics = law.get("items", [{}])[0].get("crime", [])
            for t in topics:
                if len(t) > 5:
                    doc_titles.add(t)

        return {
            "total_statute_macro_nodes": len(laws),
            "total_faq_cases": len(cases),
            "indexed_documents": sorted(list(doc_titles))[:30],
            "embedding_model": self.rag.config.graph.embedding_model,
            "llm_model": self.rag.config.model.model_name
        }
