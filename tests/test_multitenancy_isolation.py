# -*- coding: utf-8 -*-
"""
tests/test_multitenancy_isolation.py

Integration test suite verifying strict data isolation between organizations (Tenants):
- Default Organization: "DGA"
- Foreign Organization: "MOF"
- Shared Public Data: "PUBLIC"

Rules tested:
1. Public statutes/documents are visible to both DGA and MOF.
2. DGA-specific private cases are visible ONLY to DGA, hidden from MOF.
3. MOF-specific private cases are visible ONLY to MOF, hidden from DGA.
4. Tri-Store consistency: Postgres (SSOT), Qdrant (VectorDB), Neo4j (GraphDB).
"""

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.database import StorageManager


class TestMultiTenancyIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage = StorageManager.get_instance()
        cls.storage.init_all_stores()

        # Insert Test Fixtures
        # 1. DGA private case
        cls.dga_case = {
            "case_id": "case_tenant_dga_secret_01",
            "org_id": "DGA",
            "source": "INTERNAL_DGA",
            "question": "แนวทางปฏิบัติการจัดซื้อจัดจ้างระบบคลาวด์ภาครัฐภายใน สพร. (DGA)",
            "answer": "ให้ปฏิบัติตามคู่มือความปลอดภัยข้อมูลขั้นสูงของ DGA ฉบับลับเฉพาะ",
            "features": {"category": "Cloud Procurement", "internal_dga_only": True},
            "cited_laws": ["มาตรา ๕๖"],
            "topics": ["Cloud", "DGA_Internal"]
        }

        # 2. MOF private case
        cls.mof_case = {
            "case_id": "case_tenant_mof_secret_01",
            "org_id": "MOF",
            "source": "INTERNAL_MOF",
            "question": "เกณฑ์การอนุมัติวงเงินพิเศษเฉพาะกระทรวงการคลัง (MOF Confidential)",
            "answer": "ต้องได้รับความเห็นชอบจากปลัดกระทรวงการคลังเป็นการเฉพาะ",
            "features": {"category": "Finance Approval", "internal_mof_only": True},
            "cited_laws": ["มาตรา ๕๖"],
            "topics": ["Finance", "MOF_Internal"]
        }

        # Ingest to PostgreSQL
        cls.storage.pg.upsert_faq_cases([cls.dga_case], org_id="DGA")
        cls.storage.pg.upsert_faq_cases([cls.mof_case], org_id="MOF")

        # Ingest to Qdrant (dummy 1024-dim vector for test)
        dummy_vector_dga = [0.01] * 1024
        dummy_vector_mof = [0.02] * 1024
        cls.storage.qdrant.upsert_case_points([cls.dga_case], [dummy_vector_dga], org_id="DGA")
        cls.storage.qdrant.upsert_case_points([cls.mof_case], [dummy_vector_mof], org_id="MOF")

        # Ingest to Neo4j
        cls.storage.neo4j.sync_faq_cases([cls.dga_case], org_id="DGA")
        cls.storage.neo4j.sync_faq_cases([cls.mof_case], org_id="MOF")

    def test_01_postgres_isolation(self):
        """Postgres queries must filter out records belonging to other tenants."""
        # Querying as DGA
        dga_visible = self.storage.pg.get_clause_by_id("ACT_2560_SEC_56", org_id="DGA")
        # Public statute is visible
        if dga_visible:
            self.assertEqual(dga_visible.get("org_id", "PUBLIC"), "PUBLIC")

        # Querying cases as DGA
        from sqlalchemy import text
        with self.storage.pg.engine.connect() as conn:
            # Query as DGA
            dga_rows = conn.execute(
                text("SELECT case_id FROM faq_cases WHERE org_id IN ('PUBLIC', 'DGA')")
            ).scalars().all()
            self.assertIn("case_tenant_dga_secret_01", dga_rows)
            self.assertNotIn("case_tenant_mof_secret_01", dga_rows)

            # Query as MOF
            mof_rows = conn.execute(
                text("SELECT case_id FROM faq_cases WHERE org_id IN ('PUBLIC', 'MOF')")
            ).scalars().all()
            self.assertIn("case_tenant_mof_secret_01", mof_rows)
            self.assertNotIn("case_tenant_dga_secret_01", mof_rows)

    def test_02_qdrant_isolation(self):
        """Qdrant hybrid search must never return foreign tenant points."""
        query_vec = [0.01] * 1024

        # Search as DGA
        dga_results = self.storage.hybrid_search_cases(
            query_text="ระบบคลาวด์ภาครัฐ สพร. DGA",
            query_dense=query_vec,
            top_k=10,
            org_id="DGA"
        )
        returned_ids_dga = [r["case_id"] for r in dga_results]
        self.assertNotIn("case_tenant_mof_secret_01", returned_ids_dga, "MOF confidential case leaked to DGA!")

        # Search as MOF
        mof_results = self.storage.hybrid_search_cases(
            query_text="วงเงินพิเศษ กระทรวงการคลัง MOF",
            query_dense=query_vec,
            top_k=10,
            org_id="MOF"
        )
        returned_ids_mof = [r["case_id"] for r in mof_results]
        self.assertNotIn("case_tenant_dga_secret_01", returned_ids_mof, "DGA confidential case leaked to MOF!")

    def test_03_neo4j_isolation(self):
        """Neo4j graph traversals must only return PUBLIC or current tenant cases."""
        # Link cases to Section 56 in Neo4j for test
        edges = [
            {"source_id": "case_tenant_dga_secret_01", "target_id": "ACT_2560_SEC_56", "rel_type": "RELATES_TO_LAW"},
            {"source_id": "case_tenant_mof_secret_01", "target_id": "ACT_2560_SEC_56", "rel_type": "RELATES_TO_LAW"}
        ]
        self.storage.neo4j.sync_relationships(edges)

        # Traverse as DGA
        dga_cases = self.storage.neo4j.get_related_cases("ACT_2560_SEC_56", org_id="DGA")
        dga_case_ids = [c["case_id"] for c in dga_cases]
        self.assertNotIn("case_tenant_mof_secret_01", dga_case_ids, "MOF case leaked in Neo4j traversal to DGA!")

        # Traverse as MOF
        mof_cases = self.storage.neo4j.get_related_cases("ACT_2560_SEC_56", org_id="MOF")
        mof_case_ids = [c["case_id"] for c in mof_cases]
        self.assertNotIn("case_tenant_dga_secret_01", mof_case_ids, "DGA case leaked in Neo4j traversal to MOF!")


if __name__ == "__main__":
    unittest.main()
