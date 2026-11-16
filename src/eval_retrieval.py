"""Benchmark retrieval methods + context packing on one split (no LLM involved).

Run: python -m src.eval_retrieval --split dev
     python -m src.eval_retrieval --split test --configs dense_k3_b800 hybrid_rerank_k3_b2400   (after freezing)

Hit uses the starter definition: same source_id + page and the evidence string is
contained in the chunk text. Hit@1/Hit@3 are always computed on the top-3 ranking.
Context Evidence Rate: a gold chunk survives `pack_context` under the char budget.
Unanswerable cases are excluded from the denominator (they have no gold).
"""
import argparse
import json
import statistics
import sys
import time

from .config import QUESTIONS_PATH, RESULTS_DIR, ROOT
from .retrievers import build_all

sys.path.insert(0, str(ROOT / "starter"))
from baseline import hit, pack_context  # noqa: E402

# name -> (retriever, top_k used for context, context budget in characters incl. headers)
CONFIGS = {
    "dense_k3_b800": ("dense", 3, 800),            # baseline: Day 6 dense + starter budget
    "bm25_k3_b800": ("bm25", 3, 800),              # method 2
    "hybrid_rrf_k3_b800": ("hybrid_rrf", 3, 800),  # method 3
    "hybrid_rerank_k3_b800": ("hybrid_rerank", 3, 800),   # method 4 / improvement 1
    "dense_k3_b2400": ("dense", 3, 2400),          # improvement 2: context budget
    "hybrid_rrf_k3_b2400": ("hybrid_rrf", 3, 2400),
    "hybrid_rerank_k3_b2400": ("hybrid_rerank", 3, 2400),
    "hybrid_rerank_k5_b4000": ("hybrid_rerank", 5, 4000),  # improvement 3: wider context
}


def run_config(name, retriever, top_k, budget, questions):
    traces = []
    for q in questions:
        start = time.perf_counter()
        results = retriever.search(q["question"], filters=q.get("filter"), top_k=max(3, top_k))
        elapsed = (time.perf_counter() - start) * 1000
        top3 = results[:3]
        context, ids = pack_context(results[:top_k], budget=budget)
        marked = [{"id": r["id"], "page": r["page"], "rank": r["rank"], "score": round(r["score"], 5),
                   "hit": hit(r, q["gold"]), **({"ranks": r["ranks"]} if "ranks" in r else {}),
                   **({"pre_rerank_rank": r["pre_rerank_rank"]} if "pre_rerank_rank" in r else {})}
                  for r in results[:top_k]]
        traces.append({"query_id": q["id"], "question": q["question"], "filter": q.get("filter"),
                       "gold_pages": [g["page"] for g in q["gold"]], "results": marked,
                       "context_ids": ids, "context_chars": len(context), "elapsed_ms": round(elapsed, 1),
                       "hit1": bool(top3 and hit(top3[0], q["gold"])),
                       "hit3": any(hit(r, q["gold"]) for r in top3),
                       "context_hit": any(hit(r, q["gold"]) and r["id"] in ids for r in results[:top_k])})
    n = len(traces)
    fmt = lambda k: f"{sum(t[k] for t in traces)}/{n} ({sum(t[k] for t in traces) / n:.0%})"
    summary = {"config": name, "retriever": retriever.name, "top_k_context": top_k, "budget_chars": budget,
               "n": n, "Hit@1": fmt("hit1"), "Evidence Hit@3": fmt("hit3"), "Context Evidence Rate": fmt("context_hit"),
               "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces)),
               "avg_context_tokens_approx": round(statistics.mean(t["context_chars"] for t in traces) / 4),
               "avg_chunks_in_context": round(statistics.mean(len(t["context_ids"]) for t in traces), 2),
               "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 1),
               "cost_usd": 0.0}
    return summary, traces


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--configs", nargs="*", default=list(CONFIGS))
    args = p.parse_args()
    questions = [q for q in json.loads(QUESTIONS_PATH.read_text(encoding="utf-8")) if q["split"] == args.split]
    retrievers = build_all()
    out = RESULTS_DIR / f"retrieval-{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    for r in retrievers.values():               # warm-up so model loading is not counted as latency
        r.search("warm up", top_k=1)
    summaries = []
    for name in args.configs:
        rname, top_k, budget = CONFIGS[name]
        summary, traces = run_config(name, retrievers[rname], top_k, budget, questions)
        summaries.append(summary)
        (out / f"{name}.traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{name:26} Hit@1 {summary['Hit@1']:12} Hit@3 {summary['Evidence Hit@3']:12} "
              f"Ctx {summary['Context Evidence Rate']:12} chars {summary['avg_context_chars']:5} {summary['median_ms']} ms")
    (out / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
