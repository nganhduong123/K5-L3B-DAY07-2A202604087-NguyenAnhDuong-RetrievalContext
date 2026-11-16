"""Run advanced agent configurations on a split and save traces + automatic coverage metrics.

Run: python -m src.advanced.run --split dev --configs adv_dense_noview adv_rerank_noview adv_rerank_view
Only the question goes to the agent. Gold evidence units are used afterwards to compute:
- first_search_hit@k: any required unit in the results of the seeded (original-question) search
- context_full: ALL required units are in the final context (multi-hop needs every unit)
Pass/fail is graded manually afterwards (grade.py), not by these numbers.
"""
import argparse
import json
import statistics

from ..config import RESULTS_DIR, ROOT
from ..retrievers import BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever
from .agent import run
from .gemini import AGENT_MODEL

BENCH = ROOT / "data" / "advanced" / "benchmark.json"
INDEX = ROOT / "data" / "advanced" / "index_units.json"
BASE = {"top_k": 5, "budget": 9000, "max_searches": 4, "max_views": 3, "seed": True,
        "model": AGENT_MODEL, "vision_model": AGENT_MODEL}
CONFIGS = {
    "adv_dense_noview": {**BASE, "retriever": "dense", "allow_view": False},       # baseline
    "adv_rerank_noview": {**BASE, "retriever": "hybrid_rerank", "allow_view": False},
    "adv_rerank_view": {**BASE, "retriever": "hybrid_rerank", "allow_view": True},
}


class Enrich:
    """Wrap a retriever so results carry the full index unit (type, crop, bbox, section)."""
    def __init__(self, base, units):
        self.base, self.units, self.name = base, units, base.name

    def search(self, query, *, filters=None, top_k=5):
        return [{**self.units[r["id"]], "score": r["score"], "rank": r["rank"]}
                for r in self.base.search(query, filters=filters, top_k=top_k)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--configs", nargs="+", required=True, choices=list(CONFIGS))
    ap.add_argument("--only", nargs="*", help="subset of question ids (debug)")
    args = ap.parse_args()
    bench = [b for b in json.loads(BENCH.read_text(encoding="utf-8")) if b["split"] == args.split]
    if args.only:
        bench = [b for b in bench if b["id"] in args.only]
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    units = {u["id"]: u for u in index}
    dense, bm25 = DenseRetriever(strategy="ADV"), BM25Retriever(index)
    hybrid = HybridRetriever(dense, bm25)
    retrievers = {"dense": Enrich(dense, units), "hybrid_rerank": Enrich(RerankRetriever(hybrid), units)}
    for name in args.configs:
        cfg = CONFIGS[name]
        out_dir = RESULTS_DIR / "advanced" / f"{args.split}" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        traces = []
        for b in bench:   # question text only goes to the agent
            res = run(b["question"], retrievers[cfg["retriever"]], model=cfg["model"], vision_model=cfg["vision_model"],
                      top_k=cfg["top_k"], budget=cfg["budget"], max_searches=cfg["max_searches"],
                      max_views=cfg["max_views"], allow_view=cfg["allow_view"], seed=cfg["seed"])
            req = [e["unit"] for e in b.get("evidence", [])]
            first = next((e for e in res["events"] if e["event"] == "search"), None)
            t = {"query_id": b["id"], "split": b["split"], "group": b["group"], "answerable": b["answerable"],
                 "question": b["question"], "config": name, **{k: v for k, v in cfg.items()}, **res,
                 "required_units": req,
                 "first_search_hit": bool(first and any(r["id"] in req for r in first["results"])) if req else None,
                 "context_full": all(u in res["context_ids"] for u in req) if req else None,
                 "cites_all_required": all(u in res["citations"] for u in req) if req else None}
            traces.append(t)
            (out_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(f"{name} {b['id']:5} {b['group']:12} {res['status']:8} ctx_full={t['context_full']} "
                  f"cites_all={t['cites_all_required']} {res['elapsed_s']}s {(res['answer'] or res.get('error') or '')[:110]!r}", flush=True)
        ok = [t for t in traces if t["status"] != "error"]
        ans = [t for t in traces if t["answerable"]]
        summary = {"config": name, "split": args.split, **cfg, "n": len(traces), "errors": len(traces) - len(ok),
                   "first_search_hit": f"{sum(bool(t['first_search_hit']) for t in ans)}/{len(ans)}",
                   "context_full": f"{sum(bool(t['context_full']) for t in ans)}/{len(ans)}",
                   "cites_all_required": f"{sum(bool(t['cites_all_required']) for t in ans)}/{len(ans)}",
                   "answered_answerable": f"{sum(t['status'] == 'answered' for t in ans)}/{len(ans)}",
                   "abstained_unanswerable": f"{sum(t['status'] == 'abstain' for t in traces if not t['answerable'])}/{sum(not t['answerable'] for t in traces)}",
                   "median_latency_s": statistics.median(t["elapsed_s"] for t in traces),
                   "llm_calls": sum(t["usage"]["llm_calls"] for t in traces),
                   "vision_calls": sum(t["usage"]["vision_calls"] for t in traces),
                   "prompt_tokens": sum(t["usage"]["prompt_tokens"] for t in traces),
                   "completion_tokens": sum(t["usage"]["completion_tokens"] for t in traces),
                   "cost_usd": 0.0, "cost_note": "Gemini API free tier (no charge)"}
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
