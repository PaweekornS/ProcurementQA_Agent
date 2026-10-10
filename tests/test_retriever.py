# -*- coding: utf-8 -*-
"""Unit tests for core/retrieval/retriever.py with an in-process fake store and scorer (no DB, no model)."""

import unittest

from core.retrieval.retriever import GRAPH, Candidate, Retriever, RetrieverConfig


class FakeStore:
    def __init__(self, results, edges=None):
        self.results = results          # query -> [chunk_id, ...] in rank order
        self.edges = edges or {}        # seed chunk_id -> [neighbour chunk_id, ...]
        self.expand_calls = 0

    def search(self, query, vector, org_id, top_k):
        return [Candidate(chunk_id=c, entry=f"doc | {c}", text=f"text {c}") for c in self.results.get(query, [])[:top_k]]

    def expand(self, seeds, org_id):
        self.expand_calls += 1
        return {s.chunk_id: [Candidate(chunk_id=n, entry=f"doc | {n}", text=f"text {n}", origin=GRAPH,
                                       seed_id=s.chunk_id, relation="CITES_CLAUSE")
                             for n in self.edges.get(s.chunk_id, [])]
                for s in seeds}


class FakeReranker:
    """Scores from a table keyed by (query, chunk_id); counts calls to prove a single pass."""

    def __init__(self, table):
        self.table = table
        self.calls = 0
        self.model = object()

    def _predict_pairs(self, pairs, batch_size=8):
        self.calls += 1
        out = []
        for q, text in pairs:
            cid = text.split()[-1]  # FakeStore chunk text is "text <chunk_id>"
            out.append(self.table.get((q, cid), -5.0))
        return out


def make(store, table, **cfg):
    config = RetrieverConfig(search_top_k=10, seeds=10, per_query=1, top_k=3, graph_per_seed=1,
                             rerank_threshold=0.0, **cfg)
    reranker = FakeReranker(table)
    return Retriever(store, embed=lambda q: None, reranker=reranker, config=config), reranker


class TestRetriever(unittest.TestCase):
    def test_reranks_once_over_all_query_variants(self):
        store = FakeStore({"q1": ["a", "b"], "q2": ["c"]})
        r, reranker = make(store, {("q1", "a"): 3.0, ("q2", "c"): 2.0})
        r.retrieve(["q1", "q2"], "DGA")
        self.assertEqual(reranker.calls, 1)
        self.assertEqual(store.expand_calls, 1)

    def test_secondary_sub_question_keeps_a_slot(self):
        # 'c' only answers q2 and scores below a/b against q1; per-query slots must still keep it
        store = FakeStore({"q1": ["a", "b", "d"], "q2": ["c"]})
        table = {("q1", "a"): 9.0, ("q1", "b"): 8.0, ("q1", "d"): 7.0, ("q2", "c"): 1.0}
        r, _ = make(store, table, )
        ids = [c.chunk_id for c in r.retrieve(["q1", "q2"], "DGA").candidates]
        self.assertIn("c", ids)
        self.assertIn("a", ids)

    def test_graph_neighbour_is_carried_not_reranked(self):
        # 'z' is cited by 'a' and would fail the threshold if it were scored on wording
        store = FakeStore({"q": ["a", "b"]}, edges={"a": ["z"]})
        r, _ = make(store, {("q", "a"): 5.0, ("q", "b"): 1.0, ("q", "z"): -9.0})
        result = r.retrieve(["q"], "DGA").candidates
        ids = [c.chunk_id for c in result]
        self.assertEqual(ids, ["a", "z", "b"])  # the cited clause follows the seed that cites it
        z = next(c for c in result if c.chunk_id == "z")
        self.assertEqual(z.origin, GRAPH)
        self.assertEqual(z.seed_id, "a")

    def test_neighbours_of_dropped_seeds_are_not_carried(self):
        store = FakeStore({"q": ["a", "b"]}, edges={"b": ["z"]})
        r, _ = make(store, {("q", "a"): 5.0, ("q", "b"): -1.0}, )  # b fails threshold 0.0
        ids = [c.chunk_id for c in r.retrieve(["q"], "DGA").candidates]
        self.assertNotIn("b", ids)
        self.assertNotIn("z", ids)

    def test_all_below_threshold_keeps_best(self):
        store = FakeStore({"q": ["a", "b"]})
        r, _ = make(store, {("q", "a"): -3.0, ("q", "b"): -1.0})
        ids = [c.chunk_id for c in r.retrieve(["q"], "DGA").candidates]
        self.assertEqual(ids, ["b"])

    def test_fusion_rewards_agreement_across_variants(self):
        store = FakeStore({"q1": ["a", "b"], "q2": ["b", "c"]})
        r = Retriever(store, embed=lambda q: None, reranker=None,
                      config=RetrieverConfig(search_top_k=10, seeds=10, top_k=3, graph_per_seed=0))
        ids = [c.chunk_id for c in r.retrieve(["q1", "q2"], "DGA").candidates]
        self.assertEqual(ids[0], "b")

    def test_graph_failure_does_not_fail_retrieval(self):
        class Broken(FakeStore):
            def expand(self, seeds, org_id):
                raise RuntimeError("neo4j down")
        r, _ = make(Broken({"q": ["a"]}), {("q", "a"): 1.0})
        self.assertEqual([c.chunk_id for c in r.retrieve(["q"], "DGA").candidates], ["a"])

    def test_empty_queries(self):
        r, reranker = make(FakeStore({}), {})
        self.assertEqual(r.retrieve(["", "  "], "DGA").candidates, [])
        self.assertEqual(reranker.calls, 0)


if __name__ == "__main__":
    unittest.main()
