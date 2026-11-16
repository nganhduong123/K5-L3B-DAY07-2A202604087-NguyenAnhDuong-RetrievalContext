"""Re-apply the fixed citation parser (parse_answer) to saved traces without new model calls.
The model outputs (answer text) and final context are unchanged; only post-processing is redone.

Run: python -m src.advanced.rescore --split dev --configs adv_dense_noview adv_rerank_noview adv_rerank_view
"""
import argparse
import json

from ..config import RESULTS_DIR
from .agent import parse_answer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    a = ap.parse_args()
    for cfg in a.configs:
        d = RESULTS_DIR / "advanced" / a.split / cfg
        traces = json.loads((d / "traces.json").read_text(encoding="utf-8"))
        changed = []
        for t in traces:
            if t["status"] == "error":
                continue
            p = parse_answer(t["answer"], set(t["context_ids"]))
            if p["status"] != t["status"] or p["citations"] != t["citations"]:
                changed.append(f"{t['query_id']}: {t['status']}->{p['status']}")
            t.update(status=p["status"], citations=p["citations"], raw_citations=p["raw_citations"],
                     parser="v2 (multi-id brackets)")
            req = t["required_units"]
            t["cites_all_required"] = all(u in t["citations"] for u in req) if req else None
        (d / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        ans = [t for t in traces if t["answerable"]]
        s.update({"cites_all_required": f"{sum(bool(t['cites_all_required']) for t in ans)}/{len(ans)}",
                  "answered_answerable": f"{sum(t['status'] == 'answered' for t in ans)}/{len(ans)}",
                  "abstained_unanswerable": f"{sum(t['status'] == 'abstain' for t in traces if not t['answerable'])}/{sum(not t['answerable'] for t in traces)}",
                  "parser": "v2 (multi-id brackets)", "rescored_changes": changed})
        (d / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(cfg, {k: s[k] for k in ("first_search_hit", "context_full", "cites_all_required", "answered_answerable",
                                       "abstained_unanswerable", "median_latency_s", "llm_calls", "vision_calls")}, changed)


if __name__ == "__main__":
    main()
