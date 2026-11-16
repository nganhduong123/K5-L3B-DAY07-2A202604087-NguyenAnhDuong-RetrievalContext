"""Gemini client via the OpenAI-compatible endpoint (stdlib only): text, images, tool calling.
Free-tier friendly: retries on 429/5xx/timeouts with backoff; never fabricates output."""
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..config import ROOT  # noqa: F401  (loads .env)

URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
VISION_MODEL = "gemini-3.5-flash"
AGENT_MODEL = "gemini-3.5-flash-lite"


class GeminiError(RuntimeError):
    pass


def image_part(path: Path) -> dict[str, Any]:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}


def chat(model: str, messages: list[dict[str, Any]], *, tools: list | None = None, tool_choice: str | None = None,
         max_tokens: int = 4000, reasoning_effort: str = "low", timeout: int = 180, retries: int = 5) -> dict[str, Any]:
    key = os.environ.get("GEMINI_API_KEY", "").strip().strip('"')
    if not key:
        raise GeminiError("GEMINI_API_KEY chưa có trong .env")
    body: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens,
                            "temperature": 0, "reasoning_effort": reasoning_effort}
    if tools:
        body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
    data = json.dumps(body).encode()
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(URL, data=data, headers={"Authorization": f"Bearer {key}",
                                                              "Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read())
            if not payload.get("choices"):
                raise GeminiError(f"empty response: {str(payload)[:200]}")
            return {"message": payload["choices"][0]["message"], "usage": payload.get("usage", {}),
                    "finish_reason": payload["choices"][0].get("finish_reason"),
                    "latency_s": round(time.perf_counter() - t0, 2), "model": model, "attempts": attempt + 1}
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
            if e.code not in (429, 500, 502, 503, 504):
                raise GeminiError(last) from e
        except (urllib.error.URLError, TimeoutError, GeminiError, json.JSONDecodeError) as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(90, 10 * 2 ** attempt))
    raise GeminiError(last or "unknown error")
