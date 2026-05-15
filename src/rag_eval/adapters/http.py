"""HTTP-endpoint adapter — posts {prompt, conversation} and parses the response."""

from __future__ import annotations

from typing import Any

import httpx

from rag_eval.adapters.base import (
    Citation,
    ContextDoc,
    Message,
    RagAdapter,
    RagResponse,
    ToolCall,
)


class HttpAdapter(RagAdapter):
    """Generic adapter for a RAG agent exposed over HTTP JSON.

    Expects POST {endpoint} with:
        {"prompt": str, "conversation": [{"role": str, "content": str}, ...]}

    Returns:
        {
          "text": str,
          "citations": [{"id": str, "source_id": str, "span": str}, ...],
          "tool_calls": [{"name": str, "args": {}, "result": ..., "latency_ms": int, "error": null}, ...],
          "retrieved_context": [{"source_id": str, "chunk": str, "score": float}, ...]
        }
    """

    def __init__(self, *, endpoint: str, headers: dict[str, str], timeout_s: float) -> None:
        self.endpoint = endpoint
        self.headers = headers
        self.timeout_s = timeout_s

    def query(
        self,
        prompt: str,
        *,
        conversation: list[Message] | None = None,
    ) -> RagResponse:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "conversation": [
                {"role": m.role, "content": m.content} for m in (conversation or [])
            ],
        }
        resp = httpx.post(
            self.endpoint,
            json=payload,
            headers=self.headers,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return RagResponse(
            text=data.get("text", ""),
            citations=[Citation(**c) for c in data.get("citations", [])],
            tool_calls=[ToolCall(**t) for t in data.get("tool_calls", [])],
            retrieved_context=[ContextDoc(**c) for c in data.get("retrieved_context", [])],
            raw=data,
        )
