"""Write manual review labels (reviews.json) for the four graded agent runs, then
they are aggregated with starter/agent_metrics.py.

Labels were produced by reading every answer and every cited chunk (not by a verifier
model). Default rule after reading: an answered answerable case whose cited chunks state
the answer -> grade 2, each citation supported; abstained answerable -> 0; unanswerable
abstained without extra claims -> correct. Every deviation is listed in NOTES with a reason.
Run: python -m src.make_reviews
"""
import json

from .config import RESULTS_DIR

RUNS = [("dev", "agent_dense_b800"), ("dev", "agent_rerank_b2400_v2_seed"),
        ("test", "agent_dense_b800"), ("test", "agent_rerank_b2400_v2_seed")]

# (split, config, query_id) -> overrides + reason
NOTES = {
    ("dev", "agent_dense_b800", "q01"): {"reason": "Cited p24:B:0 (not the gold chunk) but it states 'the conditions ... and the agents' -> valid alternative evidence."},
    ("dev", "agent_dense_b800", "q09"): {"reason": "Cited p55:B:3 (figure caption, not gold chunk): 'Icebergs often serve as a metaphor for the aspects of culture' -> supports."},
    ("dev", "agent_rerank_b2400_v2_seed", "q06"): {"reason": "p31:B:0 states the condition (part of your text); 2nd citation p40:B:4 supports the extra sentence that all paraphrases need citation."},
    ("dev", "agent_rerank_b2400_v2_seed", "u02"): {"abstention_correct": False, "citation_support": [False],
        "reason": "Should abstain. Corpus gives pages (1-2), not words; '250-500 words' is outside knowledge. p35:B:1 does not support the word count."},
    ("test", "agent_rerank_b2400_v2_seed", "t06"): {"reason": "Correct (Becoming, 2018). Name 'Michelle Obama' is in p56:B:1, not the cited p56:B:2; not asked, so grade kept 2."},
    ("test", "agent_rerank_b2400_v2_seed", "t10"): {"reason": "Verdict 'not guilty as charged' correct and in p41:B:0; 'due to' insanity claim is a slight causal inference (chunk says defense claimed insanity). Grade 2, noted."},
}


def main() -> None:
    for split, cfg in RUNS:
        traces = json.loads((RESULTS_DIR / f"agent-{split}" / cfg / "traces.json").read_text(encoding="utf-8"))
        reviews = []
        for t in traces:
            if t["status"] == "error":
                continue
            r = {"query_id": t["query_id"], "citation_support": [True] * len(t["citations"])}
            if t["answerable"]:
                if t["status"] == "answered":
                    r.update(answer_grade=2, reason="Correct and complete; cited chunk contains the evidence sentence.")
                else:
                    r.update(answer_grade=0, citation_support=[],
                             reason="Refused an answerable question (graded 0). Final context: "
                                    + ("gold present" if t["context_evidence_hit"] else "gold absent"))
            else:
                if t["status"] == "abstain":
                    r.update(abstention_correct=True, citation_support=[], reason="Searched, then abstained without adding claims.")
                else:
                    r.update(abstention_correct=False, reason="Answered an unanswerable question.")
            r.update(NOTES.get((split, cfg, t["query_id"]), {}))
            if "abstention_correct" in r:
                r.pop("answer_grade", None)
            reviews.append(r)
        path = RESULTS_DIR / f"agent-{split}" / cfg / "reviews.json"
        path.write_text(json.dumps(reviews, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("wrote", path.relative_to(RESULTS_DIR.parent))


if __name__ == "__main__":
    main()
