# -*- coding: utf-8 -*-
"""
core/mcp_service.py

Domain service providing decoupled access to ProcurementQA Agent capabilities:
- Exact statutory clause lookup (0-LLM, ~10ms)
- Hybrid search over statutory clauses without synthesis (~100-300ms)
- FAQ precedent search over Comptroller General cases (~50ms)
- Knowledge graph neighbor and subordinate legislation traversal (~10ms)
- CRAG reasoning (fast and deep modes)
- Resources (thresholds & catalog)
"""

import os
import sys
import re
import json
import threading
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple

from core.pipeline import ProcurementQAPipeline, PipelineConfig
from core.retrieval.retriever import get_retriever
from core.chunking.page_locator import format_page_range
from core.utils.settings import env


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


# Each agentic QA run holds an LLM conversation and CPU reranker work for ~1-2 minutes. Bounding
# concurrency protects the CPU, memory and the LLM provider's rate limit; excess callers queue
# briefly and then get ServiceBusyError (HTTP 503) instead of degrading everyone's latency.
_QA_MAX_CONCURRENCY = int(os.getenv("QA_MAX_CONCURRENCY", "4"))
_QA_SLOTS = threading.BoundedSemaphore(_QA_MAX_CONCURRENCY)


from core.compliance_constants import SPECIFIC_METHOD_CEILING_THB, fmt_thb


class ServiceBusyError(RuntimeError):
    """Raised when no QA slot frees up within QA_QUEUE_TIMEOUT_SECONDS."""


class ProcurementService:
    """
    Singleton service managing ProcurementQA Agent components, cached statutory lookups,
    and compliance rule-checking.
    """
    _instance: Optional["ProcurementService"] = None

    def __init__(
        self,
        dotenv_path: Optional[str] = None,
        config: Optional[PipelineConfig] = None,
    ):
        self.dotenv_path = dotenv_path or os.getenv("DOTENV_PATH", ".env")
        self.config = config or PipelineConfig.from_env_file(self.dotenv_path)
        self.rag = ProcurementQAPipeline(config=self.config)
        self._warmup_models()

    def _warmup_models(self):
        try:
            from core.retrieval.embedding import get_embedding
            from core.retrieval.reranker import get_reranker, is_reranker_enabled
            # Load the store (and, locally, build the BM25/dense index) before the first request
            get_retriever()

            if is_reranker_enabled():
                import torch
                reranker_model = env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
                reranker_device = env("RERANKER_DEVICE", "cuda:0" if (torch and torch.cuda.is_available()) else "cpu")
                reranker_thresh = float(env("RERANKER_THRESHOLD", "0.20"))

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
        config: Optional[PipelineConfig] = None,
    ) -> "ProcurementService":
        if cls._instance is None:
            cls._instance = cls(dotenv_path=dotenv_path, config=config)
        return cls._instance

    # --------------------------------------------------------------------------
    # Tier 1: Indexing & Atomic Lookup
    # --------------------------------------------------------------------------

    def lookup_section(self, section: str, doc_title: Optional[str] = None, org_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Exact statutory section lookup without generative overhead.

        Args:
            section: Section string e.g. "มาตรา 56", "56", "ข้อ 79", "มาตรา ๕๖ (๒) (ข)"
            doc_title: Optional document title keyword (e.g. "พระราชบัญญัติ", "ระเบียบ")
            org_id: Tenant scope; PUBLIC records are always visible
        """
        if not section or not section.strip():
            return {"found": False, "error": "section parameter is required"}

        active_org = org_id or os.getenv("DEFAULT_ORG_ID", "DGA")
        sec_norm = normalize_digits(section.strip())
        num_m = re.search(r"(\d+)", sec_norm)
        if not num_m:
            return {"found": False, "section": section, "message": f"No section/clause number in: {section}"}
        # 'มาตรา N' -> Act sections, 'ข้อ N' -> clauses, a bare number -> sections first, then clauses
        kind = "clause" if "ข้อ" in sec_norm else ("section" if "มาตรา" in sec_norm else None)
        records = get_retriever().store.lookup_unit(kind, int(num_m.group(1)), doc_title or "", active_org)
        if not records:
            return {"found": False, "section": section,
                    "message": f"No statutory clause found matching section: {section}"}

        top = records[0]
        full_text = top["content"]
        result = {
            "found": True,
            "section": section,
            "source_id": top["entry"],
            "chunk_id": top["chunk_id"],
            "doc_title": top.get("doc_title"),
            "source_file": top.get("source_file"),
            "page": format_page_range(top.get("page_start"), top.get("page_end"), top.get("total_pages")),
            "focused_content": self._extract_focused_section_text(full_text, section) or full_text[:1500],
            "full_macro_chunk": full_text,
        }
        # The same number exists in many documents; tell the caller instead of silently picking one
        other_docs = list(dict.fromkeys(
            r.get("doc_title") for r in records[1:] if r.get("doc_title") and r.get("doc_title") != top.get("doc_title")
        ))
        if other_docs and not doc_title:
            result["ambiguous"] = True
            result["other_documents_with_same_number"] = other_docs[:5]
        return result

    def _extract_focused_section_text(self, macro_text: str, section: str) -> Optional[str]:
        """Extract only the lines relevant to the requested section."""
        sec_num = normalize_digits(section)
        num_match = re.search(r"(\d+)", sec_num)
        if not num_match:
            return None
        target_num = num_match.group(1)
        thai_num = to_thai_digits(target_num)

        # Remove leading bracket header if present: [พระราชบัญญัติ... | มาตรา ๕๖]
        cleaned = re.sub(r"^\[[^\]]+\]\s*", "", macro_text.strip())

        # Pattern matching 'มาตรา 56' or 'มาตรา ๕๖' or 'ข้อ 79' or 'ข้อ ๗๙'
        pattern = rf"(?:^|\n)\s*(?:มาตรา|ข้อ)\s*(?:{target_num}|{thai_num})[\s\S]*?(?=(?:\n\s*(?:มาตรา|ข้อ)\s*(?:\d+|[๐-๙]+))|#|\Z)"
        match = re.search(pattern, cleaned)
        if match and len(match.group(0).strip()) > 10:
            return match.group(0).strip()
        return cleaned if len(cleaned) > 10 else macro_text

    def search_clauses(self, query: str, top_k: int = 5, doc_filter: Optional[str] = None, org_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Direct hybrid search (Dense Vector + Thai BM25 + GPU Cross-Encoder Reranker)
        over statutory clauses without LLM synthesis, filtered by tenant org_id.
        """
        if not query or not query.strip():
            return []

        from core.retrieval.retriever import get_retriever

        active_org = org_id or os.getenv("DEFAULT_ORG_ID", "DGA")
        laws = get_retriever().retrieve(
            [query.strip()], active_org, top_k=top_k * 2 if doc_filter else top_k
        ).laws()

        results = []
        for law in laws:
            data = law.get("data", {}) or law
            entry = law.get("entry") or data.get("entry") or law.get("id", "")
            desc = data.get("description") or law.get("description", "")
            topics = law.get("topics", [])

            if doc_filter:
                combined_meta = f"{entry} {' '.join(topics)}".lower()
                if doc_filter.lower() not in combined_meta:
                    continue

            results.append({
                "id": law.get("id"),
                "entry": entry,
                "topics": topics,
                "content": desc[:800],
                "description": desc[:800],
                "score": round(float(law.get("rerank_score", law.get("similarity", 0.0))), 4),
                "source_type": law.get("source_type", "statute"),
            })

            if len(results) >= top_k:
                break

        return results

    def search_faqs(self, query: str, top_k: int = 3, org_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search the Comptroller General's Department FAQ (one question/answer pair per hit)."""
        if not query or not query.strip():
            return []
        from core.retrieval.embedding import get_embedding
        active_org = org_id or os.getenv("DEFAULT_ORG_ID", "DGA")
        return get_retriever().store.search_faq(query.strip(), get_embedding(query.strip()), active_org, top_k)

    # --------------------------------------------------------------------------
    # Tier 2: Knowledge Graph Traversal
    # --------------------------------------------------------------------------

    def traverse_regulations(self, section_reference: str, org_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Neighbours of a statutory section in the knowledge graph: adjacent sections, clauses it
        cites, subordinate rules that derive from it and FAQ cases about it.
        """
        if not section_reference or not section_reference.strip():
            return {"target": section_reference, "related_nodes": []}

        active_org = org_id or os.getenv("DEFAULT_ORG_ID", "DGA")
        found = self.lookup_section(section_reference, org_id=active_org)
        if not found.get("found"):
            return {"target": section_reference, "matched_graph_node": None,
                    "graph_neighbors_count": 0, "related_nodes": []}
        related = get_retriever().store.related(found["chunk_id"], active_org)
        return {
            "target": section_reference,
            "matched_graph_node": found["chunk_id"],
            "matched_entry": found["source_id"],
            "graph_neighbors_count": len(related),
            "related_nodes": related,
        }

    def get_related_clauses(self, section_reference: str, org_id: Optional[str] = None, max_hops: int = 1) -> List[Dict[str, Any]]:
        """
        Convenience wrapper returning the list of related nodes discovered through
        Knowledge Graph traversal for a given statute section or ministerial rule.
        """
        res = self.traverse_regulations(section_reference=section_reference, org_id=org_id)
        return res.get("related_nodes", [])

    # --------------------------------------------------------------------------
    # Tier 3: Compliance Engine & CRAG Pipeline
    # --------------------------------------------------------------------------

    def ask_procurement_law(self, question: str, org_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute LangGraph Agentic RAG legal analysis workflow, scoped to tenant org_id.

        At most QA_MAX_CONCURRENCY workflows run at once (shared by REST and MCP); a caller that
        cannot get a slot within QA_QUEUE_TIMEOUT_SECONDS gets ServiceBusyError. Every attempt,
        including failures, is recorded in query_audit_logs and the result carries its query_id.
        """
        if not question or not question.strip():
            return {
                "status": "ERROR",
                "direct_answer": "",
                "decisive_quotes": [],
                "applicable_laws": [],
                "exceptions_or_conditions": "",
                "issues_breakdown": [],
                "error": "`question` must be a non-empty string."
            }

        active_org = org_id or os.getenv("DEFAULT_ORG_ID", "DGA")
        if not _QA_SLOTS.acquire(timeout=float(os.getenv("QA_QUEUE_TIMEOUT_SECONDS", "30"))):
            raise ServiceBusyError(f"All {_QA_MAX_CONCURRENCY} QA slots are busy; retry later.")

        query_id = str(uuid.uuid4())
        started = time.monotonic()
        result: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        try:
            result = self._run_qa(question.strip(), active_org)
            result["query_id"] = query_id
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _QA_SLOTS.release()
            self._write_audit_log(query_id, active_org, question.strip(), result, error,
                                  int((time.monotonic() - started) * 1000))

    def _run_qa(self, question: str, active_org: str) -> Dict[str, Any]:
        case = {"question": question, "org_id": active_org}

        from core.agent import ProcurementAgenticWorkflow
        workflow = ProcurementAgenticWorkflow(self.rag.model, max_retries=self.rag.config.agentic_max_retries)
        agent_res = workflow.invoke(case)
        judge = agent_res.get("judge_result", {})
        used_laws = agent_res.get("used_laws", [])

        raw_quotes = judge.get("decisive_quotes", [])
        enriched_quotes = self._enrich_decisive_quotes(raw_quotes, used_laws, active_org)

        return {
            "status": judge.get("status", "COMPLIANT"),
            "direct_answer": judge.get("direct_answer", ""),
            "decisive_quotes": enriched_quotes,
            "applicable_laws": judge.get("applicable_laws", []),
            "exceptions_or_conditions": judge.get("exceptions_or_conditions", ""),
            "issues_breakdown": judge.get("issues_breakdown", []),
            "citations": [
                {
                    "entry": law.get("entry", ""),
                    "topics": law.get("topics", [])
                }
                for law in used_laws
            ],
            "retrieved_clause_ids": [
                law.get("chunk_id") or law.get("id") for law in used_laws if law.get("chunk_id") or law.get("id")
            ],
            "crag_meta": agent_res.get("crag_meta", {}),
            "guardrail_verdict": judge.get("guardrail_verdict", {}),
            "org_id": active_org
        }

    def _write_audit_log(
        self,
        query_id: str,
        org_id: str,
        question: str,
        result: Optional[Dict[str, Any]],
        error: Optional[str],
        latency_ms: int,
    ) -> None:
        """Best-effort audit record; an audit failure must never fail the user's request."""
        if os.getenv("USE_TRI_STORE", "false").lower() not in ("true", "1", "yes"):
            return
        try:
            from core.database import StorageManager
            result = result or {}
            verdict = result.get("guardrail_verdict") or {}
            StorageManager.get_instance().pg.insert_audit_log({
                "query_id": query_id,
                "org_id": org_id,
                "user_query": question,
                "status": result.get("status") or ("ERROR" if error else None),
                "decomposed_issues": result.get("issues_breakdown") or [],
                "retrieved_clause_ids": result.get("retrieved_clause_ids") or [],
                "citations": result.get("applicable_laws") or [],
                "synthesized_answer": result.get("direct_answer"),
                "grounded": verdict.get("passed"),
                "grounding_score": verdict.get("grounding_score"),
                "latency_ms": latency_ms,
                "error": error,
            })
        except Exception as exc:
            print(f"[ProcurementService] WARNING: audit log write failed for {query_id}: {exc}", file=sys.stderr)

    def _clause_source_record(self, chunk_id: Optional[str], org_id: str) -> Optional[Dict[str, Any]]:
        """Authoritative clause row (with source_file / page range) from PostgreSQL in tri-store mode."""
        if not chunk_id or os.getenv("USE_TRI_STORE", "false").lower() not in ("true", "1", "yes"):
            return None
        try:
            from core.database import StorageManager
            return StorageManager.get_instance().pg.get_chunk_by_id(chunk_id, org_id=org_id)
        except Exception:
            return None

    def _tenant_chunk_source(self, law_name: str, quote_text: str, used_laws: List[Dict[str, Any]], org_id: str):
        """
        (source_file, page) of the retrieved tenant chunk a quote came from: the verbatim quote
        appears in the chunk text, or the quote's law label names the document title.
        """
        from core.tenant_documents import is_tenant_chunk_id
        norm = lambda s: re.sub(r"\s+", "", normalize_digits(s or ""))
        target = norm(law_name.split("|")[0])
        quote = norm(quote_text)[:60]
        tenant_cands = [c for c in used_laws if is_tenant_chunk_id(c.get("chunk_id") or c.get("id"))]
        # A paraphrased quote labelled only "หน้า N" is attributed only when one document has that page
        page_m = re.search(r"หน้า\s*(\d+)", normalize_digits(law_name))
        on_page = [c for c in tenant_cands if page_m and str(c.get("entry", "")).endswith(f"หน้า {page_m.group(1)}")]
        page_unique = len({str(c.get("entry", "")).split("|")[0] for c in on_page}) == 1
        for cand in tenant_cands:
            cid = cand.get("chunk_id") or cand.get("id")
            title = norm(str(cand.get("entry", "")).split("|")[0])
            quoted = len(quote) >= 10 and quote in norm(cand.get("description", ""))
            titled = bool(title and target) and (title in target or target in title)
            paged = page_unique and cand in on_page
            if quoted or titled or paged:
                try:
                    from core.database import StorageManager
                    rows = StorageManager.get_instance().pg.get_tenant_chunks_by_ids([cid], org_id=org_id)
                except Exception:
                    rows = []
                if rows:
                    r = rows[0]
                    return r.get("source_file") or r.get("title"), format_page_range(r.get("page_start"), r.get("page_end"), r.get("total_pages"))
        return None

    def _enrich_decisive_quotes(
        self,
        raw_quotes: List[Any],
        used_laws: List[Dict[str, Any]],
        org_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Enrich decisive quotes with statutory source metadata:
        - filename: source document markdown filename
        - page: page number or page range in document (e.g. '4-6/42')
        - law: exact section or rule identifier
        - quote: verbatim quoted text

        A quote is attributed to a source only when both the document and the มาตรา/ข้อ number
        match; unknown metadata is None (never a same-numbered clause of another document).
        """
        from core.graph.citation_linker import parse_unit_label

        def split_law(law: str):
            """'ระเบียบ... พ.ศ. 2560 ข้อ 86' -> ('ระเบียบ... พ.ศ. 2560', 'clause', 86)."""
            norm = normalize_digits(re.sub(r"\s+", " ", law.replace("|", " "))).strip()
            m = re.search(r"(มาตรา|ข้อ)\s*(\d+)", norm)
            if not m:
                return norm, None, None
            kind, num, _, _ = parse_unit_label(f"{m.group(1)} {m.group(2)}")
            return norm[:m.start()].strip(), kind, num

        def same_doc(a: str, b: str) -> bool:
            a, b = normalize_digits(a).replace(" ", ""), normalize_digits(b).replace(" ", "")
            return bool(a) and bool(b) and (a in b or b in a)

        def page_of(*texts: str) -> Optional[str]:
            for t in texts:
                m = re.search(r"หน้า\s*([0-9\-\/]+)", t) or re.search(r"_p(\d+(?:_p\d+)?)", t)
                if m:
                    return m.group(1).replace("_p", "-")
            return None

        enriched = []
        for q in raw_quotes or []:
            if not isinstance(q, dict):
                if isinstance(q, str) and q.strip():
                    enriched.append({"filename": None, "page": None, "law": None, "quote": q.strip()})
                continue

            law_name = str(q.get("law", "")).strip()
            quote_text = str(q.get("quote", "")).strip()

            # Tenant documents first: their 'ข้อ N' is a TOR item, not a regulation clause
            tenant_hit = self._tenant_chunk_source(law_name, quote_text, used_laws, org_id)
            if tenant_hit:
                enriched.append({"filename": tenant_hit[0], "page": tenant_hit[1], "law": law_name or None, "quote": quote_text})
                continue

            doc_part, kind, num = split_law(law_name)

            # 1. A retrieved chunk of the same document and the same มาตรา/ข้อ
            matched = None  # (doc_name, entry, text, chunk_id)
            for cand in used_laws:
                entry = str(cand.get("entry", "") or cand.get("id", ""))
                cand_doc, _, cand_label = entry.partition("|")
                cand_kind, cand_num, _, _ = parse_unit_label(cand_label)
                if kind and cand_kind == kind and cand_num == num and same_doc(doc_part, cand_doc):
                    cid = cand.get("chunk_id") or (cand.get("data") or {}).get("chunk_id")
                    matched = (cand_doc.strip(), entry, str(cand.get("description", "")), cid)
                    break

            # 2. Otherwise (or when the workflow dropped chunk_id while merging candidates)
            #    an exact store lookup scoped to that document
            if (not matched or not matched[3]) and kind and doc_part:
                res = self.lookup_section(f"{'มาตรา' if kind == 'section' else 'ข้อ'} {num}", doc_title=doc_part)
                if res.get("found") and same_doc(doc_part, str(res.get("source_id", "")).split("|")[0]):
                    entry = str(res.get("source_id", ""))
                    matched = (entry.split("|")[0].strip(), entry, str(res.get("full_macro_chunk", "")), res.get("chunk_id"))

            filename, page = None, None
            if matched:
                doc_name, entry, text, chunk_id = matched
                record = self._clause_source_record(chunk_id, org_id)
                if record:
                    # Tri-store: OCR-relative path and page range recovered at ingestion
                    filename = record.get("source_file") or f"{doc_name}.md"
                    page = format_page_range(record.get("page_start"), record.get("page_end"), record.get("total_pages"))
                else:
                    filename = f"{doc_name}.md"
                    page = page_of(entry, text[:300])

            enriched.append({
                "filename": filename,
                "page": page,
                "law": law_name or None,
                "quote": quote_text
            })

        return enriched

    # Alias for consistent high-level agent naming
    procurement_qa = ask_procurement_law

    # --------------------------------------------------------------------------
    # Tier 4: Resources
    # --------------------------------------------------------------------------

    def get_thresholds_resource(self) -> str:
        """Markdown summary of statutory monetary thresholds and procedural rules."""
        return f"""# เกณฑ์วงเงินและข้อกำหนดตามกฎหมายจัดซื้อจัดจ้างภาครัฐไทย (พ.ร.บ. 2560)

## 1. วิธีการจัดซื้อจัดจ้างพัสดุ (มาตรา 55, 56)
- **วิธีเฉพาะเจาะจง (Specific Selection)**:
  - วงเงินเล็กน้อย: **ไม่เกิน {fmt_thb(SPECIFIC_METHOD_CEILING_THB)} บาท** (ตามกฎกระทรวงกำหนดวงเงินฯ พ.ศ. 2560)
  - วงเงินเกิน {fmt_thb(SPECIFIC_METHOD_CEILING_THB)} บาท ต้องมีข้อยกเว้นตามมาตรา 56 (2) เช่น จำเป็นเร่งด่วนฉุกเฉิน (ง), เป็นพัสดุที่มีตัวแทนจำหน่ายแต่เพียงผู้เดียว (ค), ยกเลิก e-bidding แล้วไม่มีผู้ยื่น (ก)
- **วิธีประกาศเชิญชวนทั่วไป (General Invitation)**:
  - วงเงิน **เกิน {fmt_thb(SPECIFIC_METHOD_CEILING_THB)} บาทขึ้นไป**
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
        """Catalog of indexed documents, chunks and FAQ pairs."""
        cat = get_retriever().store.catalog(os.getenv("DEFAULT_ORG_ID", "DGA"))
        return {
            "total_chunks": cat["chunks"],
            "total_faq_pairs": cat["faq_pairs"],
            "indexed_documents": cat["documents"],
            "embedding_model": self.rag.config.embedding.tokenmind_model
            if self.rag.config.embedding.provider == "tokenmind" else self.rag.config.embedding.model,
            "llm_model": self.rag.config.model.model_name,
        }
