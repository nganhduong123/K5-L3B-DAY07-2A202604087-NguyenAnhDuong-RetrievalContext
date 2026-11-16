"""Minimal OpenRouter chat-completions client with tool calling (stdlib only)."""
import json
import time
import urllib.error
import urllib.request
from typing import Any

from .config import OPENROUTER_API_KEY

URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMError(RuntimeError):
    pass


def chat(model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
         tool_choice: str | None = None, temperature: float = 0.0, retries: int = 2,
         max_tokens: int = 800) -> dict[str, Any]:
    """Return {'message': ..., 'usage': ..., 'latency_s': ...}. Raises LLMError; never fakes output."""
    if not OPENROUTER_API_KEY:
        raise LLMError("OPENROUTER_API_KEY chưa được đặt trong .env")
    body: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                            "usage": {"include": True}}
    if tools:
        body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
    data = json.dumps(body).encode("utf-8")
    last_error = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(URL, data=data, headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"})
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            detail = exc.read().decode("utf-8", "replace")[:300] if isinstance(exc, urllib.error.HTTPError) else ""
            last_error = f"{type(exc).__name__}: {exc} {detail}".strip()
            time.sleep(2 * (attempt + 1))
            continue
        if "error" in payload or not payload.get("choices"):
            last_error = f"provider error: {str(payload.get('error', payload))[:300]}"
            time.sleep(2 * (attempt + 1))
            continue
        return {"message": payload["choices"][0]["message"], "usage": payload.get("usage", {}),
                "latency_s": time.perf_counter() - start, "model": payload.get("model", model)}
    raise LLMError(last_error or "unknown LLM error")
