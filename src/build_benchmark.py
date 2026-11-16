"""Build the Day 7 benchmark (dev/test) and validate every evidence against the corpus.

Run: python -m src.build_benchmark

Split policy
- dev: all 20 Day 6 questions (q01-q20). The Day 6 test split (q16-q20) was already
  looked at during Day 6, so it is moved to dev and NOT used as held-out test here.
- test: 10 NEW answerable questions written from pages that had no Day 6 question,
  + 3 new unanswerable questions. Written before any Day 7 retrieval run.
- unanswerable: near-topic questions; absence checked with full-corpus keyword search
  (see `absence_check`), not just with top-k retrieval.

`filter` is user-supplied metadata (the user explicitly says "In chapter N"); it is
passed to retrieval as a user filter. `gold` / `expected_answer` never reach the agent.
"""
import json
import sys
from pathlib import Path

from .config import CASES_PATH, CHUNKS_PATH, QUESTIONS_PATH, ROOT

sys.path.insert(0, str(ROOT / "starter"))
from baseline import validate  # noqa: E402  (starter validator: evidence must sit inside one chunk)

# Copy of the Day 6 benchmark (repo K5-L3B-DAY06-...-DataIngestion/benchmark/questions.json)
DAY6_QUESTIONS = ROOT / "data" / "day6" / "questions_day6.json"
SRC = "openstax-writing-guide"

NEW_TEST = [
    ("t01", "direct", None, "Where was Selena Gomez born?", "Grand Prairie, Texas (raised outside of Dallas)",
     26, "Gomez was born in Grand Prairie, Texas, but raised outside of Dallas."),
    ("t02", "direct", None, "How does the textbook define authority in its list of rhetorical key terms?",
     "Credibility; background that reflects experience, knowledge, or understanding of a situation",
     30, "Authority: credibility; background that reflects experience, knowledge, or understanding of a situation."),
    ("t03", "direct", None, "When and where did Jean Gianini murder his former teacher?",
     "In 1914, in the village of Poland, New York", 40,
     "In 1914, in the village of Poland, New York, sixteen-year-old Jean Gianini murdered his former teacher Lydia Beecher."),
    ("t04", "condition", None, "If the text you are responding to is argumentative, what should you examine?",
     "The claim, the reasoning, and each piece of supporting evidence", 40,
     "If the text is argumentative, examine the claim, reasoning, and each piece of supporting evidence."),
    ("t05", "direct", None, "What does the term academia refer to?",
     "The life, community, or environment of teachers, schools, and education", 46,
     "The term academia refers to “the life, community, or environment of teachers, schools, and education.”"),
    ("t06", "paraphrase", None, "In which book did the former First Lady write that your personal story is something to own?",
     "Becoming (2018), by Michelle Obama", 56, "writes in her book Becoming (2018) that your personal story"),
    ("t07", "direct", None, "At approximately what mental age did Goddard estimate Jean Gianini?",
     "Approximately ten years of age", 42, "he estimated his mentality at approximately ten years of age"),
    ("t08", "metadata", {"chapter": 1}, "In chapter 1, how is a meme defined in the list of key terms?",
     "An image (usually) with accompanying text that calls for a response or elicits a reaction", 30,
     "Meme: image (usually) with accompanying text that calls for a response or elicits a reaction."),
    ("t09", "metadata", {"chapter": 2}, "In chapter 2, is culture described as something static?",
     "No; culture is not static, it changes and grows dynamically in response to many variables", 56,
     "Culture is not static; it changes and grows dynamically in response to any number of variables."),
    ("t10", "paraphrase", None, "What verdict did the jury reach in the Jean Gianini murder case?",
     "Not guilty (as charged)", 41, "We find the defendant in this case not guilty as charged"),
]

UNANSWERABLE = [
    ("u01", "dev", "How many TikTok followers did Selena Gomez have in 2021?",
     "Corpus gives 2021 Twitter (~65M) and Instagram (~260M) counts only; keyword 'TikTok' has 0 hits.",
     ["TikTok"]),
    ("u02", "dev", "What is the required word count for the chapter 1 critical response assignment?",
     "Assignment length is given only in pages (1-2 pages); no word count/limit in corpus.",
     ["word count", "word limit", "words long"]),
    ("u03", "dev", "Which literary award did Cathy Park Hong's Minor Feelings win?",
     "Near-miss trap: Minor Feelings is described (p57-58) but no award is given for it. p57 says a DIFFERENT "
     "book (Dance Dance Revolution, 2007) won the Barnard Women Poets Prize; answering with that prize is wrong.",
     ["National Book Critics", "Minor Feelings won", "Minor Feelings received"]),
    ("u04", "test", "To which institution was Jean Gianini committed after his trial?",
     "Corpus covers the not-guilty verdict and Goddard's testimony; no post-trial institution ('asylum', 'hospital', 'Matteawan': 0 hits).",
     ["asylum", "hospital", "Matteawan"]),
    ("u05", "test", "In what year did Richard Dawkins publish The Selfish Gene?",
     "Dawkins appears once (p29, coined 'meme' in the 1970s); book title 'Selfish' has 0 hits.",
     ["Selfish"]),
    ("u06", "test", "What percentage of Americans speak African American Vernacular English?",
     "AAVE is discussed (p60, p72) but no share of speakers; the only 'percentage' refers to General American English.",
     ["AAVE speakers", "percent of Americans"]),
]


def main() -> None:
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    day6 = json.loads(DAY6_QUESTIONS.read_text(encoding="utf-8"))
    questions = []
    for q in day6:
        pages = q["gold_page"] if isinstance(q["gold_page"], list) else [q["gold_page"]]
        questions.append({"id": q["id"], "split": "dev", "type": q["type"], "question": q["question"],
                          "expected_answer": q["expected_answer"],
                          "gold": [{"source_id": q["gold_source_id"], "page": p, "evidence": q["evidence_quote"]} for p in pages],
                          "filter": q.get("filter"), "origin": f"day6-{q['split']}"})
    for qid, qtype, flt, text, ans, page, ev in NEW_TEST:
        questions.append({"id": qid, "split": "test", "type": qtype, "question": text, "expected_answer": ans,
                          "gold": [{"source_id": SRC, "page": page, "evidence": ev}], "filter": flt, "origin": "day7-new"})
    validate(chunks, questions)              # raises if any evidence is not inside a chunk of the right page

    cases = [{"id": q["id"], "split": q["split"], "question": q["question"], "filter": q["filter"],
              "answerable": True, "expected_behavior": "search_and_answer"} for q in questions]
    for qid, split, text, reason, keywords in UNANSWERABLE:
        hits = [c["id"] for c in chunks for k in keywords if k in c["text"]]
        if hits:
            raise ValueError(f"{qid}: keyword search found {hits[:3]} - re-check absence")
        cases.append({"id": qid, "split": split, "question": text, "filter": None, "answerable": False,
                      "expected_behavior": "search_and_abstain", "absence_reason": reason,
                      "absence_check": {"keywords": keywords, "hits": 0}})

    QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUESTIONS_PATH.write_text(json.dumps(questions, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    CASES_PATH.write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for split in ("dev", "test"):
        a = sum(c["split"] == split and c["answerable"] for c in cases)
        u = sum(c["split"] == split and not c["answerable"] for c in cases)
        print(f"{split}: {a} có đáp án + {u} thiếu evidence")


if __name__ == "__main__":
    main()
