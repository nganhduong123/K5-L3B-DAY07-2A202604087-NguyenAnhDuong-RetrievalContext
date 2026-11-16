"""Unit tests for src/ (mocked LLM / retrievers; they test code paths, not benchmark quality)."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import agent as agent_mod  # noqa: E402
from src.retrievers import BM25Retriever, HybridRetriever  # noqa: E402

CHUNKS = [
    {"id": "d:p1:B:0", "text": "A rhetorical situation has two elements: agents and conditions.", "source_id": "d",
     "source": "x.pdf", "page": 1, "metadata": {"chapter": 1}},
    {"id": "d:p2:B:0", "text": "The iceberg is often used as a metaphor for culture.", "source_id": "d",
     "source": "x.pdf", "page": 2, "metadata": {"chapter": 2}},
    {"id": "d:p3:B:0", "text": "Identity encompasses all parts of yourself.", "source_id": "d",
     "source": "x.pdf", "page": 3, "metadata": {"chapter": 2}},
]


class FakeRanker:
    def __init__(self, name, order):
        self.name, self.order = name, order

    def search(self, query, *, filters=None, top_k=3):
        return [{**CHUNKS[i], "score": 1.0 / (r + 1), "rank": r + 1} for r, i in enumerate(self.order[:top_k])]


class RetrieverTests(unittest.TestCase):
    def test_bm25_ranks_and_filters(self):
        bm25 = BM25Retriever(CHUNKS)
        self.assertEqual(bm25.search("iceberg metaphor culture", top_k=1)[0]["id"], "d:p2:B:0")
        self.assertEqual(bm25.search("iceberg metaphor", filters={"chapter": 1}, top_k=3), [])
        with self.assertRaises(ValueError):
            bm25.search("x", filters={"page": 1})

    def test_rrf_uses_ranks_and_rewards_agreement(self):
        hybrid = HybridRetriever(FakeRanker("dense", [0, 1, 2]), FakeRanker("bm25", [1, 2, 0]), candidates=3)
        res = hybrid.search("q", top_k=3)
        self.assertEqual(res[0]["id"], "d:p2:B:0")          # rank 2 + rank 1 beats rank 1 + rank 3
        self.assertEqual([r["rank"] for r in res], [1, 2, 3])
        self.assertEqual(res[0]["ranks"], {"dense": 2, "bm25": 1})


def _msg(content=None, query=None):
    if query is None:
        return {"message": {"content": content}, "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0}}
    return {"message": {"content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
        "name": "search_documents", "arguments": f'{{"query": "{query}"}}'}}]},
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0}}


class AgentTests(unittest.TestCase):
    def run_with(self, replies, **kw):
        with mock.patch.object(agent_mod, "chat", side_effect=replies):
            return agent_mod.run_agent("What are the two elements?", FakeRanker("dense", [0, 1, 2]), model="m", **kw)

    def test_answer_with_valid_citation(self):
        out = self.run_with([_msg(query="two elements"), _msg("Agents and conditions [d:p1:B:0].")])
        self.assertEqual(out["status"], "answered")
        self.assertEqual(out["citations"], ["d:p1:B:0"])

    def test_citation_outside_context_is_abstain(self):
        out = self.run_with([_msg(query="two elements"), _msg("Agents [d:p9:B:0].")], top_k=1)
        self.assertEqual(out["status"], "abstain")

    def test_abstain_tag(self):
        out = self.run_with([_msg(query="x"), _msg("INSUFFICIENT_EVIDENCE: not stated.")])
        self.assertEqual(out["status"], "abstain")

    def test_budget_skips_chunks_and_dedup(self):
        out = self.run_with([_msg(query="a"), _msg(query="b"), _msg("Agents [d:p1:B:0].")], budget=150)
        first, second = [e for e in out["events"] if e["event"] == "search"]
        self.assertEqual(first["added_to_context"], ["d:p1:B:0"])
        self.assertTrue(first["skipped_budget"])
        self.assertIn("d:p1:B:0", second["skipped_duplicate"])

    def test_seed_uses_original_question(self):
        out = self.run_with([_msg("Agents [d:p1:B:0].")], seed_original=True)
        search = [e for e in out["events"] if e["event"] == "search"][0]
        self.assertEqual(search["query"], "What are the two elements?")
        self.assertTrue(search["seeded_original_question"])

    def test_provider_error_is_recorded_not_faked(self):
        with mock.patch.object(agent_mod, "chat", side_effect=agent_mod.LLMError("402")):
            out = agent_mod.run_agent("q?", FakeRanker("dense", [0]), model="m")
        self.assertEqual(out["status"], "error")
        self.assertIsNone(out["answer"])


if __name__ == "__main__":
    unittest.main()
