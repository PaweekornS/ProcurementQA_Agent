# -*- coding: utf-8 -*-
"""
tests/test_postgres_rls.py

PostgreSQL Row-Level Security: isolation must hold even when a query forgets its
`WHERE org_id` filter. Every query below deliberately omits the tenant predicate, so only
the database policies can stop cross-tenant reads and writes.

Fixtures are two tenant FAQ rows inserted with the owner role and removed afterwards.
"""

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(override=False)

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from core.database import StorageManager

FIXTURES = {
    "RLS_ORG_A": "case_rls_test_org_a",
    "RLS_ORG_B": "case_rls_test_org_b",
}


class TestPostgresRLS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pg = StorageManager.get_instance().pg
        cls.pg.init_schema()
        if not cls.pg.rls_enforced:
            raise unittest.SkipTest("POSTGRES_APP_PASSWORD not set: RLS runtime role is disabled")
        for org, case_id in FIXTURES.items():
            cls.pg.upsert_faq_cases([{
                "case_id": case_id, "org_id": org, "source": "RLS_TEST",
                "question": f"private question of {org}", "answer": "private",
                "features": {}, "cited_laws": [],
            }], org_id=org)

    @classmethod
    def tearDownClass(cls):
        with cls.pg.engine.begin() as conn:
            conn.execute(text("DELETE FROM faq_cases WHERE case_id IN :ids"), {"ids": tuple(FIXTURES.values())})

    def _visible_fixture_orgs(self, org_id):
        with self.pg.tenant_connection(org_id) as conn:
            rows = conn.execute(
                text("SELECT org_id FROM faq_cases WHERE case_id IN :ids"),  # no tenant predicate
                {"ids": tuple(FIXTURES.values())},
            ).scalars().all()
        return set(rows)

    def test_01_runtime_role_cannot_bypass_rls(self):
        with self.pg.app_engine.connect() as conn:
            role = conn.execute(text(
                "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
            )).mappings().one()
        self.assertFalse(role["rolsuper"])
        self.assertFalse(role["rolbypassrls"])

    def test_02_tenant_sees_only_its_own_rows(self):
        self.assertEqual(self._visible_fixture_orgs("RLS_ORG_A"), {"RLS_ORG_A"})
        self.assertEqual(self._visible_fixture_orgs("RLS_ORG_B"), {"RLS_ORG_B"})

    def test_03_unset_tenant_is_fail_closed(self):
        self.assertEqual(self._visible_fixture_orgs(None), set())

    def test_04_public_corpus_visible_to_every_tenant(self):
        with self.pg.tenant_connection("RLS_ORG_A") as conn:
            public = conn.execute(text("SELECT count(*) FROM statute_clauses WHERE org_id = 'PUBLIC'")).scalar()
        self.assertGreater(public, 0)

    def test_05_tenant_cannot_write_into_another_tenant(self):
        with self.assertRaises(DBAPIError):
            with self.pg.tenant_connection("RLS_ORG_A") as conn:
                conn.execute(text(
                    "INSERT INTO faq_cases (case_id, org_id, question, answer, features) "
                    "VALUES ('case_rls_forged', 'RLS_ORG_B', 'q', 'a', '{}'::jsonb)"
                ))

    def test_06_tenant_cannot_modify_public_corpus(self):
        with self.pg.tenant_connection("RLS_ORG_A") as conn:
            updated = conn.execute(text(
                "UPDATE statute_clauses SET content_thai = content_thai WHERE org_id = 'PUBLIC'"
            )).rowcount
        self.assertEqual(updated, 0)


if __name__ == "__main__":
    unittest.main()
