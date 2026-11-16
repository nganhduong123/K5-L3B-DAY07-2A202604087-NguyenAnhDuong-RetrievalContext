"""Rebuild the Day 6 pgvector index from data/corpus/chunks.json (for graders without the Day 6 repo).

Same schema, embedding model, prefix ("passage: ") and normalisation as Day 6, so query
embeddings stay compatible. Upsert by (strategy, chunk_id): re-running does not duplicate rows.

Run (needs a PostgreSQL + pgvector reachable at DATABASE_URL, e.g. the Day 6 docker-compose):
    python -m src.build_index                 # writes strategy "B" (what the retrievers query)
    python -m src.build_index --strategy B_check   # write elsewhere, e.g. to compare with Day 6
"""
import argparse
import json
import time

import psycopg

from .config import CHUNKS_PATH, DATABASE_URL, DAY6_STRATEGY, EMBEDDING_DIM, EMBEDDING_MODEL


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default=DAY6_STRATEGY)
    args = p.parse_args()
    from sentence_transformers import SentenceTransformer

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    model = SentenceTransformer(EMBEDDING_MODEL)
    if int(model.get_embedding_dimension()) != EMBEDDING_DIM:
        raise RuntimeError("Embedding model dimension does not match the index")
    t0 = time.perf_counter()
    vecs = model.encode([f"passage: {c['text']}" for c in chunks], batch_size=32,
                        normalize_embeddings=True, show_progress_bar=False).tolist()
    embed_s = time.perf_counter() - t0
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(f"""CREATE TABLE IF NOT EXISTS chunks (
            strategy text NOT NULL, chunk_id text NOT NULL, document_id text NOT NULL,
            source text NOT NULL, page int NOT NULL, chapter int, text text NOT NULL,
            metadata jsonb NOT NULL, embedding vector({EMBEDDING_DIM}) NOT NULL,
            PRIMARY KEY (strategy, chunk_id))""")
        with conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO chunks (strategy, chunk_id, document_id, source, page, chapter, text, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (strategy, chunk_id) DO UPDATE SET text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding""",
                [(args.strategy, c["id"], c["source_id"], c["source"], c["page"], c["metadata"].get("chapter"),
                  c["text"], json.dumps(c["metadata"], ensure_ascii=False),
                  "[" + ",".join(f"{x:.7f}" for x in v) + "]") for c, v in zip(chunks, vecs)])
        n = conn.execute("SELECT count(*) FROM chunks WHERE strategy = %s", (args.strategy,)).fetchone()[0]
    print(f"strategy={args.strategy}: {n} rows, embed {embed_s:.1f}s, model {EMBEDDING_MODEL} ({EMBEDDING_DIM}d)")


if __name__ == "__main__":
    main()
