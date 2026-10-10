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
from core.graph.local_graph import GraphDBManager
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
        auto_build: Optional[bool] = None
    ):
        self.dotenv_path = dotenv_path or os.getenv("DOTENV_PATH", ".env")
        self.config = config or PipelineConfig.from_env_file(self.dotenv_path)
        
        # Explicit override
        if auto_build is not None:
            self.config.graph.auto_build = auto_build
        elif env("AUTO_BUILD") is not None:
            self.config.graph.auto_build = env("AUTO_BUILD", "True").lower() in ("true", "1", "yes")
        elif os.getenv("DISABLE_AUTO_BUILD") == "1":
            self.config.graph.auto_build = False
            
        self.rag = ProcurementQAPipeline(config=self.config)
        self._section_index: Dict[str, List[Dict[str, Any]]] = {}
        self._build_section_lookup_index()
        self._warmup_models()

    def _warmup_models(self):
        try:
            from core.retrieval.search import get_embedding
            from core.retrieval.reranker import get_reranker, is_reranker_enabled

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

    def lookup_section(self, section: str, doc_title: Optional[str] = None, org_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Exact statutory section lookup without generative overhead.

        Args:
            section: Section string e.g. "มาตรา 56", "56", "ข้อ 79", "มาตรา ๕๖ (๒) (ข)"
            doc_title: Optional filter by document title (e.g. "พระราชบัญญัติ", "ระเบียบ")
            org_id: Tenant scope (tri-store mode); PUBLIC records are always visible
        """
        if not section or not section.strip():
            return {"found": False, "error": "section parameter is required"}

        if os.getenv("USE_TRI_STORE", "false").lower() in ("true", "1", "yes"):
            return self._lookup_section_tri_store(section, doc_title, org_id)

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

    def _lookup_section_tri_store(self, section: str, doc_title: Optional[str], org_id: Optional[str]) -> Dict[str, Any]:
        """PostgreSQL-backed lookup with the same response contract as the in-memory index."""
        from core.database import StorageManager

        active_org = org_id or os.getenv("DEFAULT_ORG_ID", "DGA")
        sec_norm = normalize_digits(section.strip())
        num_m = re.search(r"(\d+)", sec_norm)
        if not num_m:
            return {"found": False, "section": section, "message": f"No section/clause number in: {section}"}

        pg = StorageManager.get_instance().pg
        num = int(num_m.group(1))
        if "ข้อ" in sec_norm:
            records = pg.lookup_clause(doc_title or "", num, org_id=active_org)
        else:
            # 'มาตรา N' or a bare number: Act sections first, then regulation clauses
            records = pg.lookup_section(doc_title or "", num, org_id=active_org)
            if not records and "มาตรา" not in sec_norm:
                records = pg.lookup_clause(doc_title or "", num, org_id=active_org)

        if not records:
            return {
                "found": False,
                "section": section,
                "message": f"No statutory clause found matching section: {section}"
            }

        top = records[0]
        full_text = top.get("content_thai", "")
        result = {
            "found": True,
            "section": section,
            "source_id": top.get("entry", ""),
            "clause_id": top.get("clause_id"),
            "doc_title": top.get("doc_title"),
            "source_file": top.get("source_file"),
            "page": format_page_range(top.get("page_start"), top.get("page_end"), top.get("total_pages")),
            "topics": top.get("topics", []),
            "related_laws": top.get("related_laws", []),
            "judge_dep": top.get("judge_dep", []),
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
                "description": desc[:800],
                "score": round(float(law.get("rerank_score", law.get("similarity", 0.0))), 4),
                "source_type": law.get("source_type", "statute"),
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

    def traverse_regulations(self, section_reference: str, org_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Traverse the Knowledge Graph to find subordinate rules, ministerial regulations,
        or circular letters linked to a parent statutory section, filtered by organization.
        """
        if not section_reference or not section_reference.strip():
            return {"parent": section_reference, "related_nodes": []}

        active_org = org_id or os.getenv("DEFAULT_ORG_ID", "DGA")
        if os.getenv("USE_TRI_STORE", "false").lower() in ("true", "1", "yes"):
            from core.database import StorageManager
            storage = StorageManager.get_instance()
            norm_ref = normalize_digits(section_reference.strip()).lower()
            num_m = re.search(r"(\d+)", norm_ref)
            clause_records = []
            if "ข้อ" in norm_ref and num_m:
                clause_records = storage.pg.lookup_clause("", int(num_m.group(1)), org_id=active_org)
            elif num_m:
                clause_records = storage.pg.lookup_section("", int(num_m.group(1)), org_id=active_org)

            related_results = []
            matched_cid = None
            if clause_records:
                matched_cid = clause_records[0]["clause_id"]
                graph_ctx = storage.traverse_clause_graph(matched_cid, org_id=active_org)
                for adj in graph_ctx.get("adjacent_sections", []):
                    related_results.append({
                        "node_id": adj.get("clause_id"),
                        "type": "Laws",
                        "relation": "ADJACENT_SECTION",
                        "description": f"Adjacent: {adj.get('entry', '')}"
                    })
                for cited in graph_ctx.get("cited_clauses", []):
                    related_results.append({
                        "node_id": cited.get("clause_id"),
                        "type": "Laws",
                        "relation": "CITES_CLAUSE",
                        "description": f"Cited: {cited.get('quote', '')}"
                    })
                for sub in graph_ctx.get("subordinate_laws", []):
                    related_results.append({
                        "node_id": sub.get("clause_id"),
                        "type": "Laws",
                        "relation": "SUBORDINATE_RULE",
                        "description": f"{sub.get('document_title', '')}: {sub.get('entry', '')}"
                    })
                for cs in graph_ctx.get("related_cases", []):
                    related_results.append({
                        "node_id": cs.get("case_id"),
                        "type": "Cases",
                        "relation": "RELATES_TO_LAW",
                        "description": cs.get("question", "")
                    })

            return {
                "target": section_reference,
                "matched_graph_node": matched_cid,
                "graph_neighbors_count": len(related_results),
                "related_nodes": related_results,
                "associated_topics": clause_records[0].get("topics", []) if clause_records else []
            }

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
        case = {"fact": question, "name": "ผู้สอบถาม", "org_id": active_org}

        from core.agent import ProcurementAgenticWorkflow
        retrieve_config = self.rag.config.retrieve.to_dict()
        max_retries = int(os.getenv("AGENTIC_MAX_RETRIES", "2"))
        workflow = ProcurementAgenticWorkflow(
            self.rag.model,
            retrieve_config=retrieve_config,
            max_retries=max_retries
        )
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
                    "topics": law.get("crimes", law.get("crime", []))
                }
                for law in used_laws
            ],
            "retrieved_clause_ids": [
                law.get("clause_id") or law.get("id") for law in used_laws if law.get("clause_id") or law.get("id")
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

    def _clause_source_record(self, clause_id: Optional[str], org_id: str) -> Optional[Dict[str, Any]]:
        """Authoritative clause row (with source_file / page range) from PostgreSQL in tri-store mode."""
        if not clause_id or os.getenv("USE_TRI_STORE", "false").lower() not in ("true", "1", "yes"):
            return None
        try:
            from core.database import StorageManager
            return StorageManager.get_instance().pg.get_clause_by_id(clause_id, org_id=org_id)
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
        tenant_cands = [c for c in used_laws if is_tenant_chunk_id(c.get("clause_id") or c.get("id"))]
        # A paraphrased quote labelled only "หน้า N" is attributed only when one document has that page
        page_m = re.search(r"หน้า\s*(\d+)", normalize_digits(law_name))
        on_page = [c for c in tenant_cands if page_m and str(c.get("entry", "")).endswith(f"หน้า {page_m.group(1)}")]
        page_unique = len({str(c.get("entry", "")).split("|")[0] for c in on_page}) == 1
        for cand in tenant_cands:
            cid = cand.get("clause_id") or cand.get("id")
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
            matched = None  # (doc_name, entry, text, clause_id)
            for cand in used_laws:
                entry = str(cand.get("entry", "") or cand.get("id", ""))
                cand_doc, _, cand_label = entry.partition("|")
                cand_kind, cand_num, _, _ = parse_unit_label(cand_label)
                if kind and cand_kind == kind and cand_num == num and same_doc(doc_part, cand_doc):
                    cid = cand.get("clause_id") or (cand.get("data") or {}).get("clause_id")
                    matched = (cand_doc.strip(), entry, str(cand.get("description", "")), cid)
                    break

            # 2. Otherwise (or when the workflow dropped clause_id while merging candidates)
            #    an exact store lookup scoped to that document
            if (not matched or not matched[3]) and kind and doc_part:
                res = self.lookup_section(f"{'มาตรา' if kind == 'section' else 'ข้อ'} {num}", doc_title=doc_part)
                if res.get("found") and same_doc(doc_part, str(res.get("source_id", "")).split("|")[0]):
                    entry = str(res.get("source_id", ""))
                    matched = (entry.split("|")[0].strip(), entry, str(res.get("full_macro_chunk", "")), res.get("clause_id"))

            filename, page = None, None
            if matched:
                doc_name, entry, text, clause_id = matched
                record = self._clause_source_record(clause_id, org_id)
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
