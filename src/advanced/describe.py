"""Advanced step 2 — describe every figure crop with a vision model (cached, one call per figure).

Run: python -m src.advanced.describe
Output: data/advanced/figure_descriptions/<figure_id>.json  (re-runs skip cached figures)
The description is model-generated: it is checked against the original crops for every figure
used in the benchmark (see report), and the crop path is kept so the agent can re-read the image.
"""
import json
import sys
import time

from ..config import ROOT
from .gemini import VISION_MODEL, GeminiError, chat, image_part

UNITS = ROOT / "data" / "advanced" / "units.json"
OUT = ROOT / "data" / "advanced" / "figure_descriptions"

PROMPT = """You are indexing a figure from the OpenStax textbook "Business Ethics" (PDF page {page}).
Printed caption: "{caption}"

Describe ONLY what is visible in the image, for search and question answering. Use this format:
TYPE: one of bar chart | line chart | pie chart | map | diagram | infographic | table image | photo | painting | other
TITLE: the title printed in the image, or "none"
TEXT IN IMAGE: every readable word/number/label, verbatim (axis labels, legend, categories, data labels, notes, sources)
DATA: for charts give axes, units and the values of every series/category. Write printed data labels exactly;
      for values read from the axis without a label write "≈" before the number. For maps list which regions
      have which category/color as given by the legend. For diagrams list every box/node and the arrows/relations.
KEY FACTS: 3-6 bullet points with the most important comparisons, highest/lowest values, trends.
For photos/paintings: describe who/what is shown, setting and any visible text; do not guess identities that
are not stated in the caption or image. Never invent numbers that are not visible."""


# Fallback when Gemini's RECITATION filter blocks verbatim transcription (seen for figs 4.5, 6.11, 7.4):
# paraphrase long sentences but still give every label and number exactly.
PROMPT_NO_VERBATIM = PROMPT.replace(
    "TEXT IN IMAGE: every readable word/number/label, verbatim (axis labels, legend, categories, data labels, notes, sources)",
    "TEXT IN IMAGE: titles, axis labels, legend entries, row/column headers and category names exactly; paraphrase "
    "long sentences in your own words (do not copy them verbatim); skip source/citation lines")

PROMPT_SUMMARY = ("This is a figure from the textbook \"Business Ethics\" (page {page}), caption: \"{caption}\". "
                  "Summarize it in your own words for search. Give the figure type, every heading/label exactly, and all "
                  "key numbers, durations, distances, conditions and relationships. Do not copy sentences verbatim.")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figures = [u for u in json.loads(UNITS.read_text(encoding="utf-8")) if u["type"] == "figure"]
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else VISION_MODEL
    if "--reverse" in sys.argv:          # second worker for parallel runs; cache prevents duplicates
        figures = figures[::-1]
    done = errors = 0
    for u in figures:
        path = OUT / f"{u['figure_id']}.json"
        if path.exists():
            continue
        msg = [{"role": "user", "content": [
            {"type": "text", "text": PROMPT.format(page=u["page"], caption=u["caption"][:600])},
            image_part(ROOT / u["crop"])]}]
        try:
            r = chat(model, msg, max_tokens=6000, retries=2)
            variant = "default"
            if len((r["message"].get("content") or "").strip()) < 80:
                print(f"  fig {u['figure_id']}: finish_reason={r.get('finish_reason')} -> retry with no-verbatim prompt", flush=True)
                msg[0]["content"][0]["text"] = PROMPT_NO_VERBATIM.format(page=u["page"], caption=u["caption"][:600])
                r = chat(model, msg, max_tokens=6000, retries=2)
                variant = "no_verbatim (RECITATION fallback)"
                if len((r["message"].get("content") or "").strip()) < 80:
                    print(f"  fig {u['figure_id']}: finish_reason={r.get('finish_reason')} -> retry with summary prompt", flush=True)
                    msg[0]["content"][0]["text"] = PROMPT_SUMMARY.format(page=u["page"], caption=u["caption"][:600])
                    r = chat(model, msg, max_tokens=6000, retries=2)
                    variant = "summary (2nd RECITATION fallback)"
        except GeminiError as e:
            errors += 1
            print(f"ERROR fig {u['figure_id']}: {e}", flush=True)
            continue
        text = (r["message"].get("content") or "").strip()
        if len(text) < 80:               # empty/truncated answers are errors, never cached
            errors += 1
            print(f"ERROR fig {u['figure_id']}: empty/short description ({len(text)} chars), usage={r['usage']}", flush=True)
            continue
        path.write_text(json.dumps({"figure_id": u["figure_id"], "page": u["page"], "crop": u["crop"],
                                    "model": model, "prompt_variant": variant, "finish_reason": r.get("finish_reason"), "described_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "latency_s": r["latency_s"], "usage": r["usage"], "description": text},
                                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        done += 1
        print(f"fig {u['figure_id']:8} p{u['page']:<4} {r['latency_s']:5}s  {text[:70]!r}", flush=True)
        time.sleep(6)                    # stay under the free-tier requests-per-minute limit
    total = len(list(OUT.glob("*.json")))
    print(f"described now: {done}, errors: {errors}, cached total: {total}/{len(figures)}")


if __name__ == "__main__":
    main()
