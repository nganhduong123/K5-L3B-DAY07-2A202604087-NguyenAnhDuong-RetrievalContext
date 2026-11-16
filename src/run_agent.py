"""Run the real LLM agent over one split for one or more agent configurations.

Run: python -m src.run_agent --split dev --configs agent_dense_b800 agent_dense_b2400 agent_rerank_b2400

Only question text + user filter go to the agent. Gold evidence is loaded afterwards,
purely to compute the Context Evidence Rate of the final context.
"""
import argparse
import json
import statistics
import sys
from collections import Counter

from .agent import run_agent
from .config import AGENT_MODEL, CASES_PATH, QUESTIONS_PATH, RESULTS_DIR, ROOT
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever

sys.path.insert(0, str(ROOT / "starter"))
from baseline import hit  # noqa: E402

BASE = {"top_k": 3, "budget": 2400, "max_searches": 3, "prompt": "v1", "seed": False, "model": AGENT_MODEL}
AGENT_CONFIGS = {
    # baseline = starter defaults (top-3, 800 chars, 4 searches) on Day 6 dense
    "agent_dense_b800": {**BASE, "retriever": "dense", "budget": 800, "max_searches": 4},
    # improvement A: context budget 800 -> 2400 chars
    "agent_dense_b2400": {**BASE, "retriever": "dense"},
    # improvement B: + hybrid RRF + cross-encoder rerank
    "agent_rerank_b2400": {**BASE, "retriever": "hybrid_rerank"},
    # improvement C: stricter grounding prompt (after dev errors u02, q19)
    "agent_rerank_b2400_v2": {**BASE, "retriever": "hybrid_rerank", "prompt": "v2"},
    # improvement D: first search always uses the original question (q19: agent rewrite lost gold)
    "agent_rerank_b2400_v2_seed": {**BASE, "retriever": "hybrid_rerank", "prompt": "v2", "seed": True},
    # improvement E: same as D with a stronger model (u02: unit conversion from outside knowledge)
    "agent_rerank_b2400_v2_seed_41mini": {**BASE, "retriever": "hybrid_rerank", "prompt": "v2", "seed": True,
                                          "model": "openai/gpt-4.1-mini"},
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--configs", nargs="+", required=True, choices=list(AGENT_CONFIGS))
    args = p.parse_args()

    cases = [c for c in json.loads(CASES_PATH.read_text(encoding="utf-8")) if c["split"] == args.split]
    gold = {q["id"]: q["gold"] for q in json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))}
    dense, bm25 = DenseRetriever(), BM25Retriever()
    hybrid = HybridRetriever(dense, bm25)
    retrievers = {"dense": dense, "hybrid_rerank": RerankRetriever(hybrid)}
    by_id = {c["id"]: c for c in bm25.chunks}

    for name in args.configs:
        cfg = AGENT_CONFIGS[name]
        rname, top_k, budget, max_searches, model = cfg["retriever"], cfg["top_k"], cfg["budget"], cfg["max_searches"], cfg["model"]
        traces = []
        for case in cases:
            res = run_agent(case["question"], retrievers[rname], model=model, filters=case["filter"],
                            top_k=top_k, budget=budget, max_searches=max_searches, prompt=cfg["prompt"],
                            seed_original=cfg["seed"])
            ctx_hit = None
            if case["answerable"]:      # gold used only here, after the agent finished
                ctx_hit = any(hit(by_id[i], gold[case["id"]]) for i in res["context_ids"])
            traces.append({"query_id": case["id"], "split": case["split"], "answerable": case["answerable"],
                           "question": case["question"], "config": name, "model": model,
                           "retriever": rname, "top_k": top_k, "budget_chars": budget, "max_searches": max_searches,
                           "prompt": cfg["prompt"], "seed_original": cfg["seed"],
                           "context_evidence_hit": ctx_hit, **res})
            print(f"  {name} {case['id']:4} {res['status']:9} searches={sum(e['event'] == 'search' for e in res['events'])} "
                  f"ctx_hit={ctx_hit} {res['elapsed_s']}s  {(res['answer'] or res.get('error') or '')[:90]!r}")
        ok = [t for t in traces if t["status"] != "error"]
        ans = [t for t in ok if t["answerable"]]
        searches = Counter(sum(e["event"] == "search" for e in t["events"]) for t in ok)
        summary = {"config": name, "split": args.split, "model": model, "retriever": rname, "top_k": top_k,
                   "prompt": cfg["prompt"], "seed_original": cfg["seed"],
                   "budget_chars": budget, "max_searches": max_searches, "n": len(traces),
                   "errors": len(traces) - len(ok),
                   "tool_use": f"{sum(any(e['event'] == 'search' for e in t['events']) for t in ok)}/{len(ok)}",
                   "searches_distribution": dict(sorted(searches.items())),
                   "Context Evidence Rate": f"{sum(t['context_evidence_hit'] for t in ans)}/{len(ans)}",
                   "answered_answerable": f"{sum(t['status'] == 'answered' for t in ans)}/{len(ans)}",
                   "abstained_unanswerable": f"{sum(t['status'] == 'abstain' for t in ok if not t['answerable'])}/{sum(not t['answerable'] for t in ok)}",
                   "avg_context_chars": round(statistics.mean(t.get("context_chars", 0) for t in ok)) if ok else 0,
                   "median_latency_s": round(statistics.median(t["elapsed_s"] for t in ok), 2) if ok else None,
                   "total_prompt_tokens": sum(t["usage"]["prompt_tokens"] for t in traces),
                   "total_completion_tokens": sum(t["usage"]["completion_tokens"] for t in traces),
                   "total_cost_usd": round(sum(t["usage"]["cost_usd"] for t in traces), 6)}
        out = RESULTS_DIR / f"agent-{args.split}" / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
