"""Shared configuration. Secrets are read from .env (gitignored), never hard-coded."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_env(ROOT / ".env")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
AGENT_MODEL = os.environ.get("AGENT_MODEL", "openai/gpt-4o-mini")

# Day 6 index: same embedding model / prefixes / dimension as the stored vectors.
DAY6_STRATEGY = "B"                       # paragraph chunking, best Day 6 config
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384
QUERY_PREFIX = "query: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNKS_PATH = ROOT / "data" / "corpus" / "chunks.json"
QUESTIONS_PATH = ROOT / "data" / "benchmark" / "questions.json"
CASES_PATH = ROOT / "data" / "benchmark" / "cases.json"
RESULTS_DIR = ROOT / "results"
