"""Unit tests for the advanced pipeline helpers (no model calls)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.advanced.agent import parse_answer  # noqa: E402

A = "openstax-business-ethics:p175:table6.2"
F = "openstax-business-ethics:p163:fig6.6"


class ParseAnswerTests(unittest.TestCase):
    def test_multiple_ids_in_one_bracket(self):  # bug found on dev
        p = parse_answer(f"US 12% [{A}, {F}].", {A, F})
        self.assertEqual(p["status"], "answered")
        self.assertEqual(p["citations"], [A, F])

    def test_citation_outside_context_becomes_abstain(self):
        p = parse_answer(f"Answer [{A}].", {F})
        self.assertEqual(p["status"], "abstain")
        self.assertTrue(p["invalid_citation"])

    def test_abstain_tag(self):
        p = parse_answer(f"INSUFFICIENT_EVIDENCE: not in table [{A}].", {A})
        self.assertEqual(p["status"], "abstain")
        self.assertEqual(p["citations"], [])

    def test_uncaptioned_figure_id(self):
        i = "openstax-business-ethics:p323:figp323-img0"
        self.assertEqual(parse_answer(f"x [{i}]", {i})["citations"], [i])


if __name__ == "__main__":
    unittest.main()
