"""Aggregate MANUAL pass/fail labels for an advanced run (labels written after reading every answer,
its citations and the original page/crop). System errors count as fail and stay in the denominator.

Run: python -m src.advanced.grade --split test --config adv_rerank_view
Needs results/advanced/<split>/<config>/labels.json: [{query_id, pass: bool, reason}]
"""
import argparse
import json
import math
import statistics

from ..config import RESULTS_DIR

GROUPS = ("table", "image", "chart", "multihop", "text", "unanswerable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    d = RESULTS_DIR / "advanced" / a.split / a.config
    traces = {t["query_id"]: t for t in json.loads((d / "traces.json").read_text(encoding="utf-8"))}
    labels = {l["query_id"]: l for l in json.loads((d / "labels.json").read_text(encoding="utf-8"))}
    missing = sorted(set(traces) - set(labels))
    if missing:
        raise SystemExit(f"missing labels for {missing}")
    rows = []
    for qid, t in traces.items():
        passed = bool(labels[qid]["pass"]) and t["status"] != "error"
        rows.append((t["group"], passed, t["status"] == "error"))
    n, k = len(rows), sum(p for _, p, _ in rows)
    out = {"split": a.split, "config": a.config, "passed": f"{k}/{n}", "pass_rate_pct": round(100 * k / n, 2),
           "threshold_90pct_needs": math.ceil(0.9 * n), "meets_90pct": k >= math.ceil(0.9 * n),
           "by_group": {g: f"{sum(p for gg, p, _ in rows if gg == g)}/{sum(gg == g for gg, _, _ in rows)}" for g in GROUPS},
           "system_errors": sum(e for _, _, e in rows),
           "median_latency_s": statistics.median(t["elapsed_s"] for t in traces.values()),
           "llm_calls": sum(t["usage"]["llm_calls"] for t in traces.values()),
           "vision_calls": sum(t["usage"]["vision_calls"] for t in traces.values()),
           "cost_usd": 0.0}
    (d / "grade.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
