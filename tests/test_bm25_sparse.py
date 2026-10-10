# -*- coding: utf-8 -*-
"""
ThaiSparseVectorizer must make Qdrant's sparse scoring (Modifier.IDF) equal Okapi BM25.
Qdrant is simulated here: score = sum_t q(t) * d(t) * idf(t), idf(t) = ln(1 + (N - n + 0.5)/(n + 0.5)).
"""

import math
import unittest
from collections import Counter

from rank_bm25 import BM25Okapi

from core.database.qdrant_repository import ThaiSparseVectorizer
from core.retrieval.tokenize import thai_tokens

CORPUS = [
    "การจัดซื้อจัดจ้างโดยวิธีเฉพาะเจาะจง ให้หน่วยงานของรัฐเชิญผู้ประกอบการที่มีคุณสมบัติโดยตรงเข้ายื่นข้อเสนอ",
    "การจัดซื้อจัดจ้างโดยวิธีประกาศเชิญชวนทั่วไป ให้ดำเนินการด้วยวิธีตลาดอิเล็กทรอนิกส์หรือวิธีประกวดราคาอิเล็กทรอนิกส์",
    "คณะกรรมการตรวจรับพัสดุ ให้ตรวจรับพัสดุ ณ ที่ทำการของผู้ใช้พัสดุ",
    "หลักประกันการเสนอราคา ให้ใช้เงินสด เช็ค หรือหนังสือค้ำประกันของธนาคาร",
    "วงเงินการจัดซื้อจัดจ้างโดยวิธีเฉพาะเจาะจง ครั้งหนึ่งไม่เกินห้าแสนบาท",
    "การบริหารสัญญาและการตรวจรับพัสดุ การแก้ไขสัญญา การบอกเลิกสัญญา และค่าปรับ",
]
QUERIES = ["วิธีเฉพาะเจาะจง วงเงิน", "ตรวจรับพัสดุ", "หลักประกันการเสนอราคา", "ประกวดราคาอิเล็กทรอนิกส์"]


def qdrant_scores(vec, query, docs):
    """What Qdrant computes for a sparse vector query on an IDF-modified collection."""
    doc_vecs = [dict(zip(*vec.vectorize_document(d))) for d in docs]
    n_docs = len(docs)
    df = Counter(i for dv in doc_vecs for i in dv)
    idf = {i: math.log(1 + (n_docs - n + 0.5) / (n + 0.5)) for i, n in df.items()}
    q = dict(zip(*vec.vectorize_query(query)))
    return [sum(w * dv.get(i, 0.0) * idf.get(i, 0.0) for i, w in q.items()) for dv in doc_vecs]


def reference_bm25(query, docs, k1=1.2, b=0.75):
    """Textbook Okapi BM25 with Qdrant's IDF, written independently of the vectorizer."""
    toks = [thai_tokens(d) for d in docs]
    avg = sum(map(len, toks)) / len(toks)
    df = Counter(t for ts in toks for t in set(ts))
    scores = []
    for ts in toks:
        tf = Counter(ts)
        s = 0.0
        for t in set(thai_tokens(query)):
            if tf[t]:
                idf = math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(ts) / avg))
        scores.append(s)
    return scores


class TestBM25Sparse(unittest.TestCase):
    def setUp(self):
        avg = sum(len(thai_tokens(d)) for d in CORPUS) / len(CORPUS)
        self.vec = ThaiSparseVectorizer(avg_doc_len=avg)

    def test_query_weights_are_one_per_distinct_term(self):
        idx, vals = self.vec.vectorize_query("พัสดุ พัสดุ ตรวจรับ")
        self.assertEqual(len(idx), 2)
        self.assertEqual(vals, [1.0, 1.0])
        self.assertEqual(idx, sorted(idx))

    def test_term_frequency_saturates(self):
        one = dict(zip(*self.vec.vectorize_document("พัสดุ " + "คำอื่น " * 20)))
        many = dict(zip(*self.vec.vectorize_document("พัสดุ " * 10 + "คำอื่น " * 11)))
        key = self.vec._token_to_idx("พัสดุ")
        self.assertGreater(many[key], one[key])
        self.assertLess(many[key], one[key] * 10 / 2)  # nowhere near linear in tf
        self.assertLess(many[key], self.vec.K1 + 1)      # BM25 upper bound

    def test_longer_documents_are_normalised(self):
        short = dict(zip(*self.vec.vectorize_document("พัสดุ ตรวจรับ")))
        long_ = dict(zip(*self.vec.vectorize_document("พัสดุ ตรวจรับ " + "ข้อความอื่น " * 60)))
        key = self.vec._token_to_idx("พัสดุ")
        self.assertGreater(short[key], long_[key])

    def test_qdrant_scoring_equals_okapi_bm25(self):
        for q in QUERIES:
            got, want = qdrant_scores(self.vec, q, CORPUS), reference_bm25(q, CORPUS)
            for g, w in zip(got, want):
                self.assertAlmostEqual(g, w, places=3, msg=q)

    def test_ranking_matches_local_rank_bm25(self):
        local = BM25Okapi([thai_tokens(d) for d in CORPUS])
        for q in QUERIES:
            ours = qdrant_scores(self.vec, q, CORPUS)
            theirs = local.get_scores(thai_tokens(q))
            self.assertEqual(max(range(len(CORPUS)), key=ours.__getitem__),
                             max(range(len(CORPUS)), key=lambda i: theirs[i]), msg=q)

    def test_empty_text(self):
        self.assertEqual(self.vec.vectorize_document(""), ([], []))
        self.assertEqual(self.vec.vectorize_query("   "), ([], []))


if __name__ == "__main__":
    unittest.main()
