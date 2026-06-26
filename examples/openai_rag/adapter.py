"""OpenAI-backed RAG adapter — starter template.

This file is a **starter** you copy into your own project and modify. It exists to
show *the shape* of an OpenAI-based RAG adapter:

  embed(query) → retrieve(top_k) → build_prompt → chat_completion → parse_response

The default implementation uses a tiny in-memory substring index instead of real
OpenAI embeddings + a vector store, so you can run this example without any API key
and see the wiring end-to-end. Replace the four marked methods with calls into your
real stack.

To use as a starting template:
  1. Copy this file into your project (e.g., `my_app/rag_adapter.py`).
  2. Install the real OpenAI SDK: `pip install openai>=1.0`.
  3. Replace each `# REPLACE:` block with a call into your retrieval + generation
     pipeline.
  4. Point your `config.yaml` `adapter.module` at the new path.
"""

from __future__ import annotations

from collections.abc import Iterable

from ragverdict.adapters.base import (
    Citation,
    ContextDoc,
    Message,
    RagAdapter,
    RagResponse,
    SourceDoc,
    ToolCall,
    ToolSpec,
)

# Tiny in-memory corpus for the runnable demo. In your real project these come from
# your vector store / document database.
_CORPUS: dict[str, str] = {
    "PRODUCT": (
        "Acme's flagship product is RouteOps, a real-time logistics routing platform "
        "used by over 200 enterprise customers."
    ),
    "PRICING": (
        "Enterprise pricing starts at $4,000/month for up to 50 active routes."
    ),
}


class OpenAIRagAdapter(RagAdapter):
    """Subclass of RagAdapter that demonstrates the OpenAI integration pattern.

    The real version of this would hold an `openai.OpenAI()` client + a vector
    store handle. The mock version below uses substring matching so it runs offline.
    """

    def __init__(self, *, openai_client: object | None = None) -> None:
        # REPLACE: store your OpenAI client + vector store handle here.
        #   self._openai = openai_client or openai.OpenAI()
        #   self._index = qdrant_client.QdrantClient(...)
        self._openai = openai_client  # unused in the mock

    # ---- ragverdict contract -----------------------------------------------

    def query(
        self,
        prompt: str,
        *,
        conversation: list[Message] | None = None,
    ) -> RagResponse:
        retrieved = self._retrieve(prompt, top_k=3)
        tool_calls = self._tool_calls_for(prompt)
        answer_text, citations = self._generate(prompt, retrieved, conversation or [])
        return RagResponse(
            text=answer_text,
            citations=citations,
            tool_calls=tool_calls,
            retrieved_context=retrieved,
        )

    def available_tools(self) -> list[ToolSpec]:
        # REPLACE: enumerate the tools your OpenAI function-calling agent exposes.
        return [
            ToolSpec(
                name="search_kb",
                description="Search the Acme knowledge base.",
                trigger_prompt="Use search_kb to find pricing information.",
            ),
        ]

    def corpus(self) -> Iterable[SourceDoc]:
        # REPLACE: yield from your real source corpus so citation_audit can validate.
        for source_id, content in _CORPUS.items():
            yield SourceDoc(source_id=source_id, content=content, title=source_id.title())

    # ---- internals (the four methods to replace) -------------------------

    def _retrieve(self, query: str, *, top_k: int) -> list[ContextDoc]:
        """REPLACE: real implementation calls OpenAI embeddings + your vector store.

        Real version (sketch):
            embedding = self._openai.embeddings.create(
                model="text-embedding-3-small",
                input=query,
            ).data[0].embedding
            hits = self._index.search(embedding, limit=top_k)
            return [
                ContextDoc(source_id=h.source_id, chunk=h.text, score=h.score)
                for h in hits
            ]
        """
        # Mock: substring match against the in-memory corpus. Require ≥2 token hits
        # so out-of-corpus queries with one incidental match (e.g. "stock price" → "pricing")
        # refuse cleanly instead of dragging in a barely-relevant chunk.
        q_lower = query.lower()
        scored: list[tuple[float, str, str]] = []
        for source_id, content in _CORPUS.items():
            hits = sum(1 for w in q_lower.split() if len(w) > 3 and w in content.lower())
            if hits >= 2:
                scored.append((float(hits), source_id, content))
        scored.sort(reverse=True)
        return [
            ContextDoc(source_id=sid, chunk=chunk, score=score)
            for score, sid, chunk in scored[:top_k]
        ]

    def _generate(
        self,
        prompt: str,
        retrieved: list[ContextDoc],
        conversation: list[Message],
    ) -> tuple[str, list[Citation]]:
        """REPLACE: real implementation calls OpenAI chat completions.

        Real version (sketch):
            context_block = "\\n\\n".join(
                f"[src:{doc.source_id}] {doc.chunk}" for doc in retrieved
            )
            messages = [
                {"role": "system", "content": "Answer using only the provided context. Cite sources as [src:ID]."},
                *[{"role": m.role, "content": m.content} for m in conversation],
                {"role": "user", "content": f"Context:\\n{context_block}\\n\\nQuestion: {prompt}"},
            ]
            resp = self._openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            text = resp.choices[0].message.content or ""
            citations = _parse_citations(text, retrieved)
            return text, citations
        """
        # Mock: refuse cleanly when no context retrieved; otherwise quote the top chunk.
        if not retrieved:
            return (
                "I could not find any relevant documents for that question.",
                [],
            )
        top = retrieved[0]
        text = f"{top.chunk} [src:{top.source_id}]"
        citations = [
            Citation(id=f"cite-{top.source_id}", source_id=top.source_id, span=top.chunk)
        ]
        return text, citations

    def _tool_calls_for(self, prompt: str) -> list[ToolCall]:
        """REPLACE: in a real OpenAI function-calling agent, tool calls come from
        the model's response (resp.choices[0].message.tool_calls).

        Real version (sketch):
            tool_calls = []
            for tc in (resp.choices[0].message.tool_calls or []):
                tool_calls.append(ToolCall(
                    name=tc.function.name,
                    args=json.loads(tc.function.arguments),
                    latency_ms=tc.latency_ms,
                ))
            return tool_calls
        """
        # Mock: fire search_kb for any prompt mentioning the knowledge base.
        if any(k in prompt.lower() for k in ("search", "pricing", "product")):
            return [
                ToolCall(name="search_kb", args={"query": prompt}, result="ok", latency_ms=12)
            ]
        return []
