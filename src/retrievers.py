"""Retrieval methods for Day 7. Every retriever returns ranked chunks:
{id, text, source_id, source, page, metadata, score, rank, strategy}.

1. DenseRetriever   - E5 query embedding -> Day 6 pgvector table (cosine), SQL metadata filter.
2. BM25Retriever    - lexical BM25 (rank_bm25) over the same chunks.
3. HybridRetriever  - Reciprocal Rank Fusion of dense + BM25 ranks (no raw score mixing).
4. RerankRetriever  - hybrid candidates re-scored by a cross-encoder.
"""
import json
import re
from typing import Any

import psycopg

from .config import (CHUNKS_PATH, DATABASE_URL, DAY6_STRATEGY, EMBEDDING_DIM, EMBEDDING_MODEL,
                     QUERY_PREFIX, RERANK_MODEL)

ALLOWED_FILTERS = {"chapter"}
STOPWORDS = set("""a an the of to in on for and or is are was were be been being what which who whom whose
when where why how does do did that this these those it its as by with from at into about your you their
there than then so if not no can could should would will shall may might must has have had i me my we our
they them he she his her""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.casefold()) if t not in STOPWORDS]


def _check_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    filters = filters or {}
    bad = set(filters) - ALLOWED_FILTERS
    if bad:
        raise ValueError(f"Filter không được phép: {sorted(bad)}")
    return filters


def _matches(chunk: dict[str, Any], filters: dict[str, Any]) -> bool:
    return all(chunk["metadata"].get(k) == v for k, v in filters.items())


def load_chunks() -> list[dict[str, Any]]:
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


class DenseRetriever:
    name = "dense"

    def __init__(self, strategy: str = DAY6_STRATEGY) -> None:
        from sentence_transformers import SentenceTransformer
        self.strategy = strategy
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        dim = int(self.model.get_embedding_dimension())
        if dim != EMBEDDING_DIM:
            raise RuntimeError(f"Embedding model có {dim} chiều, index Day 6 cần {EMBEDDING_DIM}")
        self.conn = psycopg.connect(DATABASE_URL)       # fails loudly if Docker/pgvector is down

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        filters = _check_filters(filters)
        vec = self.model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
        literal = "[" + ",".join(f"{x:.7f}" for x in vec) + "]"
        where, params = ["strategy = %s"], [self.strategy]
        for key, value in filters.items():
            where.append(f"{key} = %s")                 # key is whitelisted above
            params.append(value)
        rows = self.conn.execute(
            f"""SELECT chunk_id, document_id, source, page, chapter, text, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM chunks WHERE {' AND '.join(where)}
                ORDER BY embedding <=> %s::vector LIMIT %s""", [literal, *params, literal, top_k]).fetchall()
        return [{"id": r[0], "source_id": r[1], "source": r[2], "page": r[3], "text": r[5],
                 "metadata": {**r[6], "chapter": r[4]}, "score": float(r[7]), "rank": i + 1, "strategy": self.name}
                for i, r in enumerate(rows)]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[dict[str, Any]] | None = None) -> None:
        from rank_bm25 import BM25Okapi
        self.chunks = chunks or load_chunks()
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        filters = _check_filters(filters)
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(((s, c) for s, c in zip(scores, self.chunks) if s > 0 and _matches(c, filters)),
                        key=lambda x: (-x[0], x[1]["id"]))[:top_k]
        return [{**c, "score": float(s), "rank": i + 1, "strategy": self.name} for i, (s, c) in enumerate(ranked)]


class HybridRetriever:
    """RRF: score(d) = sum over retrievers of 1 / (k + rank_d). Uses ranks only, so the
    different scales of cosine similarity and BM25 never get added together."""
    name = "hybrid_rrf"

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, *, candidates: int = 20, rrf_k: int = 60) -> None:
        self.dense, self.bm25, self.candidates, self.rrf_k = dense, bm25, candidates, rrf_k

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}
        for retriever in (self.dense, self.bm25):
            for r in retriever.search(query, filters=filters, top_k=self.candidates):
                item = fused.setdefault(r["id"], {**r, "score": 0.0, "ranks": {}})
                item["score"] += 1.0 / (self.rrf_k + r["rank"])
                item["ranks"][retriever.name] = r["rank"]
        ranked = sorted(fused.values(), key=lambda c: (-c["score"], c["id"]))[:top_k]
        return [{**c, "rank": i + 1, "strategy": self.name} for i, c in enumerate(ranked)]


class RerankRetriever:
    name = "hybrid_rerank"

    def __init__(self, base: HybridRetriever, *, candidates: int = 20) -> None:
        from sentence_transformers import CrossEncoder
        self.base, self.candidates = base, candidates
        self.model = CrossEncoder(RERANK_MODEL)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        cands = self.base.search(query, filters=filters, top_k=self.candidates)
        if not cands:
            return []
        scores = self.model.predict([(query, c["text"]) for c in cands], show_progress_bar=False)
        ranked = sorted(zip(scores, cands), key=lambda x: (-float(x[0]), x[1]["id"]))[:top_k]
        return [{**c, "pre_rerank_rank": c["rank"], "score": float(s), "rank": i + 1, "strategy": self.name}
                for i, (s, c) in enumerate(ranked)]


def build_all() -> dict[str, Any]:
    dense = DenseRetriever()
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(dense, bm25)
    return {"dense": dense, "bm25": bm25, "hybrid_rrf": hybrid, "hybrid_rerank": RerankRetriever(hybrid)}
