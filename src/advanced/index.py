"""Advanced step 3 — build the searchable index (text + tables + figure caption/description).

Run: python -m src.advanced.index
- data/advanced/index_units.json : units in the chunk contract {id,text,source_id,source,page,metadata}
  figure text = "Figure x.y (page p) [type]\nCaption: ...\nVisual description (model-generated): ..."
- pgvector table `chunks`, strategy 'ADV' : E5-small passage embeddings (same model as Day 6, 384d)
Re-running upserts by id (no duplicates) and removes ADV rows that no longer exist.
"""
import json
import time

import psycopg

from ..config import DATABASE_URL, EMBEDDING_DIM, EMBEDDING_MODEL, ROOT

UNITS = ROOT / "data" / "advanced" / "units.json"
DESC = ROOT / "data" / "advanced" / "figure_descriptions"
OUT = ROOT / "data" / "advanced" / "index_units.json"
STRATEGY = "ADV"
SOURCE = "Business Ethics (OpenStax) business-ethics.pdf"


def build_units() -> list[dict]:
    out, missing = [], []
    for u in json.loads(UNITS.read_text(encoding="utf-8")):
        meta = {k: u.get(k) for k in ("type", "chapter", "section", "bbox", "figure_id", "table_id", "crop") if u.get(k) is not None}
        text = u["text"]
        if u["type"] == "figure":
            d = DESC / f"{u['figure_id']}.json"
            desc = json.loads(d.read_text(encoding="utf-8"))["description"] if d.exists() else None
            if desc is None:
                missing.append(u["figure_id"])
            text = (f"Figure {u['figure_id']} (page {u['page']})\nCaption: {u['caption']}\n"
                    f"Visual description (model-generated, verify with view_figure): {desc or 'NOT AVAILABLE'}")
        elif u["type"] == "table":
            text = f"Table {u['table_id']} (page {u['page']})\n{text}"
        out.append({"id": u["id"], "text": text, "source_id": "openstax-business-ethics", "source": SOURCE,
                    "page": u["page"], "metadata": meta})
    if missing:
        print("WARNING figures without description:", missing)
    return out


def main() -> None:
    from sentence_transformers import SentenceTransformer
    units = build_units()
    OUT.write_text(json.dumps(units, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    model = SentenceTransformer(EMBEDDING_MODEL)
    t0 = time.perf_counter()
    vecs = model.encode([f"passage: {u['text']}" for u in units], batch_size=32, normalize_embeddings=True,
                        show_progress_bar=False).tolist()
    embed_s = time.perf_counter() - t0
    ids = [u["id"] for u in units]
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute("DELETE FROM chunks WHERE strategy = %s AND NOT (chunk_id = ANY(%s))", (STRATEGY, ids))
        with conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO chunks (strategy, chunk_id, document_id, source, page, chapter, text, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (strategy, chunk_id) DO UPDATE SET text = EXCLUDED.text, page = EXCLUDED.page,
                    chapter = EXCLUDED.chapter, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding""",
                [(STRATEGY, u["id"], u["source_id"], u["source"], u["page"], u["metadata"].get("chapter"), u["text"],
                  json.dumps(u["metadata"], ensure_ascii=False), "[" + ",".join(f"{x:.7f}" for x in v) + "]")
                 for u, v in zip(units, vecs)])
        n = conn.execute("SELECT count(*) FROM chunks WHERE strategy = %s", (STRATEGY,)).fetchone()[0]
    stats = {"units": len(units), "db_rows": n, "embedding_model": EMBEDDING_MODEL, "dim": EMBEDDING_DIM,
             "embed_seconds": round(embed_s, 1),
             "by_type": {t: sum(u["metadata"]["type"] == t for u in units) for t in ("text", "table", "figure")}}
    (ROOT / "data" / "advanced" / "index_stats.json").write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    print(stats)


if __name__ == "__main__":
    main()
