"""Export Day 6 chunks (strategy B) from pgvector to the Day 7 chunk JSON contract.

Run: python -m src.export_day6
The vectors stay in pgvector (dense retrieval queries the DB directly); this JSON
is used by the lexical retriever, the context packer and the evaluation.
"""
import json
from pathlib import Path

import psycopg

from .config import DATABASE_URL, DAY6_STRATEGY, ROOT


def main() -> None:
    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(
            """SELECT chunk_id, document_id, source, page, chapter, text, metadata
               FROM chunks WHERE strategy = %s
               ORDER BY page, (metadata->>'chunk_index')::int""", (DAY6_STRATEGY,)).fetchall()
    chunks = [{"id": r[0], "source_id": r[1], "source": r[2], "page": r[3], "text": r[5],
               "metadata": {**r[6], "chapter": r[4]}} for r in rows]
    if not chunks:
        raise RuntimeError(f"Không có chunk nào cho strategy {DAY6_STRATEGY}; kiểm tra Day 6 DB")
    out = ROOT / "data" / "corpus" / "chunks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(chunks, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Đã xuất {len(chunks)} chunk -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
