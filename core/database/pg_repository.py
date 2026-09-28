# -*- coding: utf-8 -*-
"""
core/database/pg_repository.py

PostgreSQL Data Access Layer (SSOT) for Thai Procurement LegalGraphRAG.
Provides transactional persistence for legal documents, statutory macro-clauses,
Comptroller General FAQ cases, and QA audit logs.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import (
    create_engine, text, inspect
)
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv

load_dotenv(override=False)
logger = logging.getLogger("pg_repository")


class PostgresRepository:
    """PostgreSQL Repository for Legal System of Record (SSOT)."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        dbname: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.host = host or os.getenv("POSTGRES_HOST", "localhost")
        self.port = int(port or os.getenv("POSTGRES_PORT", 5435))
        self.dbname = dbname or os.getenv("POSTGRES_DB", "procurement_rag")
        self.user = user or os.getenv("POSTGRES_USER", "procurement_user")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "procurement_secret123")

        self.db_url = f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"
        self.engine = create_engine(
            self.db_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=1800,
        )

    def init_schema(self):
        """Idempotently create tables, constraints, and indexes."""
        ddl = """
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE EXTENSION IF NOT EXISTS "pg_trgm";

        CREATE TABLE IF NOT EXISTS legal_documents (
            doc_id VARCHAR(128) PRIMARY KEY,
            title TEXT NOT NULL,
            doc_type VARCHAR(64) NOT NULL,
            year_be INT,
            source_file VARCHAR(255),
            total_pages INT,
            metadata JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS statute_clauses (
            clause_id VARCHAR(255) PRIMARY KEY,
            doc_id VARCHAR(128) NOT NULL REFERENCES legal_documents(doc_id) ON DELETE CASCADE,
            entry TEXT NOT NULL,
            chapter_num INT,
            section_num INT,
            clause_num INT,
            page_start INT,
            page_end INT,
            content_thai TEXT NOT NULL,
            judge_dep JSONB DEFAULT '[]'::jsonb,
            related_laws JSONB DEFAULT '[]'::jsonb,
            topics JSONB DEFAULT '[]'::jsonb,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS faq_cases (
            case_id VARCHAR(128) PRIMARY KEY,
            source VARCHAR(64) DEFAULT 'FAQ_CGD',
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            features JSONB NOT NULL,
            cited_laws JSONB DEFAULT '[]'::jsonb,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS query_audit_logs (
            query_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_query TEXT NOT NULL,
            decomposed_issues JSONB,
            retrieved_clause_ids JSONB,
            synthesized_answer TEXT,
            grounding_score FLOAT,
            latency_ms INT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_statute_doc_sec ON statute_clauses(doc_id, section_num);
        CREATE INDEX IF NOT EXISTS idx_statute_doc_cls ON statute_clauses(doc_id, clause_num);
        CREATE INDEX IF NOT EXISTS idx_statute_chapter ON statute_clauses(doc_id, chapter_num);
        CREATE INDEX IF NOT EXISTS idx_statute_content_trgm ON statute_clauses USING gin (content_thai gin_trgm_ops);
        """
        with self.engine.begin() as conn:
            conn.execute(text(ddl))
        logger.info("PostgreSQL schema successfully initialized.")

    def upsert_documents(self, documents: List[Dict[str, Any]]):
        """Batch upsert legal documents."""
        if not documents:
            return
        query = text("""
            INSERT INTO legal_documents (doc_id, title, doc_type, year_be, source_file, total_pages, metadata)
            VALUES (:doc_id, :title, :doc_type, :year_be, :source_file, :total_pages, :metadata)
            ON CONFLICT (doc_id) DO UPDATE SET
                title = EXCLUDED.title,
                doc_type = EXCLUDED.doc_type,
                year_be = EXCLUDED.year_be,
                source_file = EXCLUDED.source_file,
                total_pages = EXCLUDED.total_pages,
                metadata = EXCLUDED.metadata;
        """)
        formatted = []
        for doc in documents:
            formatted.append({
                "doc_id": str(doc["doc_id"]),
                "title": str(doc["title"]),
                "doc_type": str(doc.get("doc_type", "ACT")),
                "year_be": doc.get("year_be"),
                "source_file": doc.get("source_file"),
                "total_pages": doc.get("total_pages"),
                "metadata": json.dumps(doc.get("metadata", {}), ensure_ascii=False)
            })
        with self.engine.begin() as conn:
            conn.execute(query, formatted)

    def upsert_clauses(self, clauses: List[Dict[str, Any]], batch_size: int = 250):
        """Batch upsert statutory clauses."""
        if not clauses:
            return
        query = text("""
            INSERT INTO statute_clauses (
                clause_id, doc_id, entry, chapter_num, section_num, clause_num,
                page_start, page_end, content_thai, judge_dep, related_laws, topics
            )
            VALUES (
                :clause_id, :doc_id, :entry, :chapter_num, :section_num, :clause_num,
                :page_start, :page_end, :content_thai, :judge_dep, :related_laws, :topics
            )
            ON CONFLICT (clause_id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                entry = EXCLUDED.entry,
                chapter_num = EXCLUDED.chapter_num,
                section_num = EXCLUDED.section_num,
                clause_num = EXCLUDED.clause_num,
                page_start = EXCLUDED.page_start,
                page_end = EXCLUDED.page_end,
                content_thai = EXCLUDED.content_thai,
                judge_dep = EXCLUDED.judge_dep,
                related_laws = EXCLUDED.related_laws,
                topics = EXCLUDED.topics;
        """)
        for i in range(0, len(clauses), batch_size):
            chunk = clauses[i:i + batch_size]
            formatted = []
            for c in chunk:
                formatted.append({
                    "clause_id": str(c["clause_id"]),
                    "doc_id": str(c["doc_id"]),
                    "entry": str(c["entry"]),
                    "chapter_num": c.get("chapter_num"),
                    "section_num": c.get("section_num"),
                    "clause_num": c.get("clause_num"),
                    "page_start": c.get("page_start"),
                    "page_end": c.get("page_end"),
                    "content_thai": str(c["content_thai"]),
                    "judge_dep": json.dumps(c.get("judge_dep", []), ensure_ascii=False),
                    "related_laws": json.dumps(c.get("related_laws", []), ensure_ascii=False),
                    "topics": json.dumps(c.get("topics", []), ensure_ascii=False)
                })
            with self.engine.begin() as conn:
                conn.execute(query, formatted)

    def upsert_faq_cases(self, cases: List[Dict[str, Any]]):
        """Batch upsert FAQ cases."""
        if not cases:
            return
        query = text("""
            INSERT INTO faq_cases (case_id, source, question, answer, features, cited_laws)
            VALUES (:case_id, :source, :question, :answer, :features, :cited_laws)
            ON CONFLICT (case_id) DO UPDATE SET
                source = EXCLUDED.source,
                question = EXCLUDED.question,
                answer = EXCLUDED.answer,
                features = EXCLUDED.features,
                cited_laws = EXCLUDED.cited_laws;
        """)
        formatted = []
        for cs in cases:
            formatted.append({
                "case_id": str(cs["case_id"]),
                "source": str(cs.get("source", "FAQ_CGD")),
                "question": str(cs["question"]),
                "answer": str(cs["answer"]),
                "features": json.dumps(cs.get("features", {}), ensure_ascii=False),
                "cited_laws": json.dumps(cs.get("cited_laws", []), ensure_ascii=False)
            })
        with self.engine.begin() as conn:
            conn.execute(query, formatted)

    def get_clause_by_id(self, clause_id: str) -> Optional[Dict[str, Any]]:
        """Fetch full statutory clause record by clause_id."""
        query = text("SELECT * FROM statute_clauses WHERE clause_id = :cid")
        with self.engine.connect() as conn:
            row = conn.execute(query, {"cid": clause_id}).mappings().first()
            if row:
                return dict(row)
        return None

    def get_clauses_by_ids(self, clause_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch multiple statutory clauses by their IDs."""
        if not clause_ids:
            return []
        query = text("SELECT * FROM statute_clauses WHERE clause_id IN :cids")
        with self.engine.connect() as conn:
            rows = conn.execute(query, {"cids": tuple(clause_ids)}).mappings().all()
            return [dict(r) for r in rows]

    def lookup_section(self, doc_id_or_keyword: str, section_num: int) -> List[Dict[str, Any]]:
        """Fast relational lookup for a section number in a statute."""
        query = text("""
            SELECT sc.*, ld.title as doc_title
            FROM statute_clauses sc
            JOIN legal_documents ld ON sc.doc_id = ld.doc_id
            WHERE (sc.doc_id ILIKE :kw OR ld.title ILIKE :kw)
              AND sc.section_num = :sec
            ORDER BY sc.page_start ASC NULLS LAST;
        """)
        kw = f"%{doc_id_or_keyword.strip()}%"
        with self.engine.connect() as conn:
            rows = conn.execute(query, {"kw": kw, "sec": section_num}).mappings().all()
            return [dict(r) for r in rows]

    def lookup_clause(self, doc_id_or_keyword: str, clause_num: int) -> List[Dict[str, Any]]:
        """Fast relational lookup for a regulation clause number (ข้อ)."""
        query = text("""
            SELECT sc.*, ld.title as doc_title
            FROM statute_clauses sc
            JOIN legal_documents ld ON sc.doc_id = ld.doc_id
            WHERE (sc.doc_id ILIKE :kw OR ld.title ILIKE :kw)
              AND sc.clause_num = :cls
            ORDER BY sc.page_start ASC NULLS LAST;
        """)
        kw = f"%{doc_id_or_keyword.strip()}%"
        with self.engine.connect() as conn:
            rows = conn.execute(query, {"kw": kw, "cls": clause_num}).mappings().all()
            return [dict(r) for r in rows]

    def count_stats(self) -> Dict[str, int]:
        """Return counts of all tables."""
        with self.engine.connect() as conn:
            doc_cnt = conn.execute(text("SELECT count(*) FROM legal_documents")).scalar() or 0
            cls_cnt = conn.execute(text("SELECT count(*) FROM statute_clauses")).scalar() or 0
            faq_cnt = conn.execute(text("SELECT count(*) FROM faq_cases")).scalar() or 0
            return {
                "legal_documents": int(doc_cnt),
                "statute_clauses": int(cls_cnt),
                "faq_cases": int(faq_cnt)
            }
