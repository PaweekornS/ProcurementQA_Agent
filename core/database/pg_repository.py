# -*- coding: utf-8 -*-
"""
core/database/pg_repository.py

PostgreSQL Data Access Layer (SSOT) for ProcurementQA Agent.
Provides transactional persistence for legal documents, statutory macro-clauses,
Comptroller General FAQ cases, and QA audit logs.

Tenant isolation is enforced by the database with Row-Level Security (RLS):
- Two connection roles. The owner (POSTGRES_USER) runs schema setup, ingestion and admin stats.
  The runtime role (POSTGRES_APP_USER, not superuser, not table owner) serves tenant-scoped
  reads and writes and is always subject to the RLS policies.
- Every runtime transaction sets `app.org_id` (transaction-local, so pooled connections never
  carry another tenant's value). Policies let a tenant read PUBLIC rows plus its own, and write
  only its own. Without `app.org_id` only PUBLIC rows are visible (fail-closed).
- The explicit `org_id IN ('PUBLIC', :org_id)` predicates stay in the queries as a second layer
  and so the planner keeps using the org_id indexes.
"""

import os
import re
import json
import logging
from contextlib import contextmanager
from typing import Iterator, List, Dict, Any, Optional
from sqlalchemy import (
    create_engine, text, inspect
)
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv

load_dotenv(override=False)
logger = logging.getLogger("pg_repository")


_ROLE_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")

# Tables holding tenant data; each carries an org_id column ('PUBLIC' = shared corpus)
RLS_TABLES = (
    "legal_documents", "statute_clauses", "faq_cases", "query_audit_logs",
    "tenant_documents", "tenant_chunks",
)


def _policy_ddl(table: str, role: str) -> Dict[str, str]:
    """Read PUBLIC + own tenant; insert/update/delete own tenant only."""
    tenant = "current_setting('app.org_id', true)"
    return {
        "tenant_read": (
            f'CREATE POLICY tenant_read ON {table} FOR SELECT TO "{role}" '
            f"USING (org_id = 'PUBLIC' OR org_id = {tenant})"
        ),
        "tenant_insert": (
            f'CREATE POLICY tenant_insert ON {table} FOR INSERT TO "{role}" WITH CHECK (org_id = {tenant})'
        ),
        "tenant_update": (
            f'CREATE POLICY tenant_update ON {table} FOR UPDATE TO "{role}" '
            f"USING (org_id = {tenant}) WITH CHECK (org_id = {tenant})"
        ),
        "tenant_delete": (
            f'CREATE POLICY tenant_delete ON {table} FOR DELETE TO "{role}" USING (org_id = {tenant})'
        ),
    }


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

        # Runtime role subject to RLS. Without it, tenant queries fall back to the owner engine,
        # which bypasses RLS (owner / superuser), leaving only the WHERE-clause filtering.
        self.app_user = os.getenv("POSTGRES_APP_USER", "procurement_app").strip()
        self.app_password = os.getenv("POSTGRES_APP_PASSWORD", "").strip()
        if not _ROLE_NAME.match(self.app_user):
            raise ValueError(f"Invalid POSTGRES_APP_USER: {self.app_user!r}")
        if self.app_password and self.app_user != self.user:
            self.app_engine = create_engine(
                f"postgresql+psycopg2://{self.app_user}:{self.app_password}@{self.host}:{self.port}/{self.dbname}",
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=1800,
            )
            self.rls_enforced = True
        else:
            logger.warning(
                "POSTGRES_APP_PASSWORD is not set: tenant queries run as the owner role, which "
                "bypasses Row-Level Security. Set POSTGRES_APP_USER/POSTGRES_APP_PASSWORD to enforce RLS."
            )
            self.app_engine = self.engine
            self.rls_enforced = False

    @contextmanager
    def tenant_connection(self, org_id: Optional[str]) -> Iterator[Any]:
        """Runtime transaction scoped to one tenant (RLS reads app.org_id for its lifetime only)."""
        with self.app_engine.begin() as conn:
            conn.execute(text("SELECT set_config('app.org_id', :org, true)"), {"org": org_id or ""})
            yield conn

    def _init_rls(self):
        """
        Idempotently create the runtime role, grant it table access, and install RLS policies.
        Runs as the owner. Policies are created only when missing, so concurrent startups
        (migrate + api containers) do not race on DROP/CREATE.
        """
        if not self.app_password or self.app_user == self.user:
            return
        with self.engine.begin() as conn:
            exists = conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": self.app_user}).first()
            verb = "ALTER" if exists else "CREATE"
            # psycopg2 interpolates %s client-side, so the password is safely quoted in DDL
            conn.exec_driver_sql(
                f'{verb} ROLE "{self.app_user}" LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD %s',
                (self.app_password,),
            )
            conn.exec_driver_sql(f'GRANT USAGE ON SCHEMA public TO "{self.app_user}"')
            conn.exec_driver_sql(
                f'GRANT SELECT, INSERT, UPDATE, DELETE ON {", ".join(RLS_TABLES)} TO "{self.app_user}"'
            )
            for table in RLS_TABLES:
                conn.exec_driver_sql(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
                existing = {
                    r[0] for r in conn.execute(
                        text("SELECT policyname FROM pg_policies WHERE schemaname = 'public' AND tablename = :t"),
                        {"t": table},
                    )
                }
                for name, ddl in _policy_ddl(table, self.app_user).items():
                    if name not in existing:
                        conn.exec_driver_sql(ddl)
        logger.info("PostgreSQL Row-Level Security enabled for role '%s' on %s.", self.app_user, ", ".join(RLS_TABLES))

    def init_schema(self):
        """Idempotently create tables, constraints, and indexes, including multi-tenancy columns."""
        ddl = """
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE EXTENSION IF NOT EXISTS "pg_trgm";

        CREATE TABLE IF NOT EXISTS legal_documents (
            doc_id VARCHAR(128) PRIMARY KEY,
            org_id VARCHAR(64) NOT NULL DEFAULT 'PUBLIC',
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
            org_id VARCHAR(64) NOT NULL DEFAULT 'PUBLIC',
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
            org_id VARCHAR(64) NOT NULL DEFAULT 'PUBLIC',
            source VARCHAR(64) DEFAULT 'FAQ_CGD',
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            features JSONB NOT NULL,
            cited_laws JSONB DEFAULT '[]'::jsonb,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS query_audit_logs (
            query_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            org_id VARCHAR(64) NOT NULL DEFAULT 'PUBLIC',
            user_query TEXT NOT NULL,
            decomposed_issues JSONB,
            retrieved_clause_ids JSONB,
            synthesized_answer TEXT,
            grounding_score FLOAT,
            latency_ms INT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- Add org_id column if tables were created previously without it
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='legal_documents' AND column_name='org_id') THEN
                ALTER TABLE legal_documents ADD COLUMN org_id VARCHAR(64) NOT NULL DEFAULT 'PUBLIC';
            END IF;
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='statute_clauses' AND column_name='org_id') THEN
                ALTER TABLE statute_clauses ADD COLUMN org_id VARCHAR(64) NOT NULL DEFAULT 'PUBLIC';
            END IF;
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='faq_cases' AND column_name='org_id') THEN
                ALTER TABLE faq_cases ADD COLUMN org_id VARCHAR(64) NOT NULL DEFAULT 'PUBLIC';
            END IF;
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='query_audit_logs' AND column_name='org_id') THEN
                ALTER TABLE query_audit_logs ADD COLUMN org_id VARCHAR(64) NOT NULL DEFAULT 'PUBLIC';
            END IF;
        END $$;

        -- Audit trail columns added after the initial schema (idempotent for existing databases)
        ALTER TABLE query_audit_logs ADD COLUMN IF NOT EXISTS status VARCHAR(64);
        ALTER TABLE query_audit_logs ADD COLUMN IF NOT EXISTS citations JSONB;
        ALTER TABLE query_audit_logs ADD COLUMN IF NOT EXISTS grounded BOOLEAN;
        ALTER TABLE query_audit_logs ADD COLUMN IF NOT EXISTS error TEXT;
        CREATE INDEX IF NOT EXISTS idx_audit_org_created ON query_audit_logs(org_id, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_doc_org_id ON legal_documents(org_id);
        CREATE INDEX IF NOT EXISTS idx_statute_org_id ON statute_clauses(org_id);
        CREATE INDEX IF NOT EXISTS idx_faq_org_id ON faq_cases(org_id);

        CREATE INDEX IF NOT EXISTS idx_statute_doc_sec ON statute_clauses(doc_id, section_num);
        CREATE INDEX IF NOT EXISTS idx_statute_doc_cls ON statute_clauses(doc_id, clause_num);
        CREATE INDEX IF NOT EXISTS idx_statute_chapter ON statute_clauses(doc_id, chapter_num);
        CREATE INDEX IF NOT EXISTS idx_statute_content_trgm ON statute_clauses USING gin (content_thai gin_trgm_ops);

        -- Tenant-private documents (TOR, BOQ, contracts...) ingested from the OCR service
        CREATE TABLE IF NOT EXISTS tenant_documents (
            doc_id VARCHAR(128) PRIMARY KEY,
            org_id VARCHAR(64) NOT NULL,
            title TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_file TEXT,
            total_pages INT,
            metadata JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS tenant_chunks (
            chunk_id VARCHAR(160) PRIMARY KEY,
            doc_id VARCHAR(128) NOT NULL REFERENCES tenant_documents(doc_id) ON DELETE CASCADE,
            org_id VARCHAR(64) NOT NULL,
            chunk_index INT NOT NULL,
            page_start INT,
            page_end INT,
            content TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_tenant_doc_org ON tenant_documents(org_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_tenant_chunk_org ON tenant_chunks(org_id);
        CREATE INDEX IF NOT EXISTS idx_tenant_chunk_doc ON tenant_chunks(doc_id, chunk_index);
        """
        with self.engine.begin() as conn:
            conn.execute(text(ddl))
        self._init_rls()
        logger.info("PostgreSQL schema successfully initialized with multi-tenancy.")

    def delete_corpus(self, org_id: str) -> Dict[str, int]:
        """Remove one tenant's seeded corpus (documents, clauses, FAQ cases) before a re-ingest.
        Tenant-uploaded documents live in other tables and are not touched."""
        with self.engine.begin() as conn:
            faq = conn.execute(text("DELETE FROM faq_cases WHERE org_id = :o"), {"o": org_id}).rowcount
            clauses = conn.execute(text("DELETE FROM statute_clauses WHERE org_id = :o"), {"o": org_id}).rowcount
            docs = conn.execute(text("DELETE FROM legal_documents WHERE org_id = :o"), {"o": org_id}).rowcount
        return {"legal_documents": docs, "statute_clauses": clauses, "faq_cases": faq}

    def upsert_documents(self, documents: List[Dict[str, Any]], org_id: str = "PUBLIC"):
        """Batch upsert legal documents."""
        if not documents:
            return
        query = text("""
            INSERT INTO legal_documents (doc_id, org_id, title, doc_type, year_be, source_file, total_pages, metadata)
            VALUES (:doc_id, :org_id, :title, :doc_type, :year_be, :source_file, :total_pages, :metadata)
            ON CONFLICT (doc_id) DO UPDATE SET
                org_id = EXCLUDED.org_id,
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
                "org_id": str(doc.get("org_id", org_id)),
                "title": str(doc["title"]),
                "doc_type": str(doc.get("doc_type", "ACT")),
                "year_be": doc.get("year_be"),
                "source_file": doc.get("source_file"),
                "total_pages": doc.get("total_pages"),
                "metadata": json.dumps(doc.get("metadata", {}), ensure_ascii=False)
            })
        with self.engine.begin() as conn:
            conn.execute(query, formatted)

    def upsert_clauses(self, clauses: List[Dict[str, Any]], org_id: str = "PUBLIC", batch_size: int = 250):
        """Batch upsert statutory clauses."""
        if not clauses:
            return
        query = text("""
            INSERT INTO statute_clauses (
                clause_id, doc_id, org_id, entry, chapter_num, section_num, clause_num,
                page_start, page_end, content_thai, judge_dep, related_laws, topics
            )
            VALUES (
                :clause_id, :doc_id, :org_id, :entry, :chapter_num, :section_num, :clause_num,
                :page_start, :page_end, :content_thai, :judge_dep, :related_laws, :topics
            )
            ON CONFLICT (clause_id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                org_id = EXCLUDED.org_id,
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
                    "org_id": str(c.get("org_id", org_id)),
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

    def upsert_faq_cases(self, cases: List[Dict[str, Any]], org_id: str = "PUBLIC"):
        """Batch upsert FAQ cases."""
        if not cases:
            return
        query = text("""
            INSERT INTO faq_cases (case_id, org_id, source, question, answer, features, cited_laws)
            VALUES (:case_id, :org_id, :source, :question, :answer, :features, :cited_laws)
            ON CONFLICT (case_id) DO UPDATE SET
                org_id = EXCLUDED.org_id,
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
                "org_id": str(cs.get("org_id", org_id)),
                "source": str(cs.get("source", "FAQ_CGD")),
                "question": str(cs["question"]),
                "answer": str(cs["answer"]),
                "features": json.dumps(cs.get("features", {}), ensure_ascii=False),
                "cited_laws": json.dumps(cs.get("cited_laws", []), ensure_ascii=False)
            })
        with self.engine.begin() as conn:
            conn.execute(query, formatted)

    def get_clause_by_id(self, clause_id: str, org_id: str = "DGA") -> Optional[Dict[str, Any]]:
        """Fetch full statutory clause record by clause_id with tenant filtering."""
        query = text("""
            SELECT sc.*, ld.title AS doc_title, ld.source_file, ld.total_pages
            FROM statute_clauses sc
            JOIN legal_documents ld ON sc.doc_id = ld.doc_id
            WHERE sc.clause_id = :cid AND sc.org_id IN ('PUBLIC', :org_id)
        """)
        with self.tenant_connection(org_id) as conn:
            row = conn.execute(query, {"cid": clause_id, "org_id": org_id}).mappings().first()
            if row:
                return dict(row)
        return None

    def get_clauses_by_ids(self, clause_ids: List[str], org_id: str = "DGA") -> List[Dict[str, Any]]:
        """Fetch multiple statutory clauses by their IDs with tenant filtering."""
        if not clause_ids:
            return []
        query = text("""
            SELECT sc.*, ld.title AS doc_title, ld.source_file, ld.total_pages
            FROM statute_clauses sc
            JOIN legal_documents ld ON sc.doc_id = ld.doc_id
            WHERE sc.clause_id IN :cids AND sc.org_id IN ('PUBLIC', :org_id)
        """)
        with self.tenant_connection(org_id) as conn:
            rows = conn.execute(query, {"cids": tuple(clause_ids), "org_id": org_id}).mappings().all()
            return [dict(r) for r in rows]

    def get_faq_cases_by_ids(self, case_ids: List[str], org_id: str = "DGA") -> List[Dict[str, Any]]:
        if not case_ids:
            return []
        query = text("SELECT case_id, question, answer, cited_laws FROM faq_cases "
                     "WHERE case_id IN :ids AND org_id IN ('PUBLIC', :org_id)")
        with self.tenant_connection(org_id) as conn:
            return [dict(r) for r in conn.execute(query, {"ids": tuple(case_ids), "org_id": org_id}).mappings().all()]

    def list_document_titles(self, org_id: str = "DGA") -> List[str]:
        query = text("SELECT title FROM legal_documents WHERE org_id IN ('PUBLIC', :org_id) ORDER BY title")
        with self.tenant_connection(org_id) as conn:
            return [r[0] for r in conn.execute(query, {"org_id": org_id}).all()]

    def lookup_section(self, doc_id_or_keyword: str, section_num: int, org_id: str = "DGA") -> List[Dict[str, Any]]:
        """Fast relational lookup for a section number in a statute with tenant filtering."""
        query = text("""
            SELECT sc.*, ld.title AS doc_title, ld.source_file, ld.total_pages
            FROM statute_clauses sc
            JOIN legal_documents ld ON sc.doc_id = ld.doc_id
            WHERE (sc.doc_id ILIKE :kw OR ld.title ILIKE :kw)
              AND sc.section_num = :sec
              AND sc.org_id IN ('PUBLIC', :org_id)
            -- The parent Act first (circulars may quote 'มาตรา N'), then continuation parts in order
            ORDER BY (ld.doc_type = 'ACT') DESC, sc.page_start ASC NULLS LAST, length(sc.entry), sc.entry;
        """)
        kw = f"%{doc_id_or_keyword.strip()}%"
        with self.tenant_connection(org_id) as conn:
            rows = conn.execute(query, {"kw": kw, "sec": section_num, "org_id": org_id}).mappings().all()
            return [dict(r) for r in rows]

    def lookup_clause(self, doc_id_or_keyword: str, clause_num: int, org_id: str = "DGA") -> List[Dict[str, Any]]:
        """Fast relational lookup for a regulation clause number (ข้อ) with tenant filtering."""
        query = text("""
            SELECT sc.*, ld.title AS doc_title, ld.source_file, ld.total_pages
            FROM statute_clauses sc
            JOIN legal_documents ld ON sc.doc_id = ld.doc_id
            WHERE (sc.doc_id ILIKE :kw OR ld.title ILIKE :kw)
              AND sc.clause_num = :cls
              AND sc.org_id IN ('PUBLIC', :org_id)
            -- 'ข้อ N' exists in dozens of documents. Without a document keyword, prefer regulations
            -- over ministerial rules / circulars, then the most comprehensive document (most clauses),
            -- then continuation parts in reading order.
            ORDER BY CASE ld.doc_type WHEN 'REGULATION' THEN 0 WHEN 'MINISTERIAL_RULE' THEN 1 ELSE 2 END,
                     (SELECT count(*) FROM statute_clauses s2 WHERE s2.doc_id = sc.doc_id) DESC,
                     sc.page_start ASC NULLS LAST, length(sc.entry), sc.entry;
        """)
        kw = f"%{doc_id_or_keyword.strip()}%"
        with self.tenant_connection(org_id) as conn:
            rows = conn.execute(query, {"kw": kw, "cls": clause_num, "org_id": org_id}).mappings().all()
            return [dict(r) for r in rows]

    def insert_audit_log(self, record: Dict[str, Any]):
        """Persist one QA audit record (who asked what, what was answered, how it was grounded)."""
        query = text("""
            INSERT INTO query_audit_logs (
                query_id, org_id, user_query, status, decomposed_issues, retrieved_clause_ids,
                citations, synthesized_answer, grounded, grounding_score, latency_ms, error
            )
            VALUES (
                :query_id, :org_id, :user_query, :status, :decomposed_issues, :retrieved_clause_ids,
                :citations, :synthesized_answer, :grounded, :grounding_score, :latency_ms, :error
            )
        """)
        params = dict(record)
        for key in ("decomposed_issues", "retrieved_clause_ids", "citations"):
            params[key] = json.dumps(params.get(key) or [], ensure_ascii=False)
        with self.tenant_connection(params.get("org_id")) as conn:
            conn.execute(query, params)

    def replace_tenant_document(self, org_id: str, doc: Dict[str, Any], chunks: List[Dict[str, Any]]) -> None:
        """Atomically (re)write one tenant document and its chunks under the tenant's RLS scope."""
        with self.tenant_connection(org_id) as conn:
            conn.execute(text("DELETE FROM tenant_documents WHERE doc_id = :d"), {"d": doc["doc_id"]})
            conn.execute(
                text("""
                    INSERT INTO tenant_documents (doc_id, org_id, title, source_id, source_file, total_pages, metadata)
                    VALUES (:doc_id, :org_id, :title, :source_id, :source_file, :total_pages, :metadata)
                """),
                {**doc, "org_id": org_id, "metadata": json.dumps(doc.get("metadata") or {}, ensure_ascii=False)},
            )
            if chunks:
                conn.execute(
                    text("""
                        INSERT INTO tenant_chunks (chunk_id, doc_id, org_id, chunk_index, page_start, page_end, content)
                        VALUES (:chunk_id, :doc_id, :org_id, :chunk_index, :page_start, :page_end, :content)
                    """),
                    [{**c, "doc_id": doc["doc_id"], "org_id": org_id} for c in chunks],
                )

    def delete_tenant_document(self, org_id: str, doc_id: str) -> bool:
        """Delete a tenant document (chunks cascade). False when it does not exist for this tenant."""
        with self.tenant_connection(org_id) as conn:
            res = conn.execute(
                text("DELETE FROM tenant_documents WHERE doc_id = :d AND org_id = :o"),
                {"d": doc_id, "o": org_id},
            )
            return res.rowcount > 0

    def list_tenant_documents(self, org_id: str) -> List[Dict[str, Any]]:
        query = text("""
            SELECT d.doc_id, d.title, d.source_id, d.source_file, d.total_pages, d.metadata, d.created_at,
                   (SELECT count(*) FROM tenant_chunks c WHERE c.doc_id = d.doc_id) AS chunks
            FROM tenant_documents d
            WHERE d.org_id = :o
            ORDER BY d.created_at DESC
        """)
        with self.tenant_connection(org_id) as conn:
            return [dict(r) for r in conn.execute(query, {"o": org_id}).mappings().all()]

    def get_tenant_chunks_by_ids(self, chunk_ids: List[str], org_id: str) -> List[Dict[str, Any]]:
        if not chunk_ids:
            return []
        query = text("""
            SELECT c.*, d.title, d.source_file, d.total_pages
            FROM tenant_chunks c
            JOIN tenant_documents d ON c.doc_id = d.doc_id
            WHERE c.chunk_id IN :ids AND c.org_id = :o
        """)
        with self.tenant_connection(org_id) as conn:
            rows = conn.execute(query, {"ids": tuple(chunk_ids), "o": org_id}).mappings().all()
            return [dict(r) for r in rows]

    def count_stats(self) -> Dict[str, Any]:
        """Return counts of all tables (owner role: totals across every tenant)."""
        with self.engine.connect() as conn:
            doc_cnt = conn.execute(text("SELECT count(*) FROM legal_documents")).scalar() or 0
            cls_cnt = conn.execute(text("SELECT count(*) FROM statute_clauses")).scalar() or 0
            faq_cnt = conn.execute(text("SELECT count(*) FROM faq_cases")).scalar() or 0
            return {
                "legal_documents": int(doc_cnt),
                "statute_clauses": int(cls_cnt),
                "faq_cases": int(faq_cnt),
                "rls_enforced": self.rls_enforced,
            }
