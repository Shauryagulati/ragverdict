"""DemoAdapter — a tiny deterministic RAG over a markdown corpus.

This is the reference adapter used in `rag-eval`'s own demo and tests. It deliberately
exposes a small surface so all three V0 evaluators (tool_coverage, rag_quality,
citation_audit) have something concrete to bite into.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

from rag_eval.adapters.base import (
    Citation,
    ContextDoc,
    Message,
    RagAdapter,
    RagResponse,
    SourceDoc,
    ToolCall,
    ToolSpec,
)

CORPUS_DIR = Path(__file__).parent / "corpus"


class DemoAdapter(RagAdapter):
    """Substring-retrieval "RAG" over the bundled Acme markdown corpus."""

    def __init__(self) -> None:
        self._docs: dict[str, SourceDoc] = {}
        for md in sorted(CORPUS_DIR.glob("*.md")):
            source_id = md.stem.upper()
            self._docs[source_id] = SourceDoc(
                source_id=source_id,
                content=md.read_text(),
                title=md.stem.replace("-", " ").title(),
            )

    def available_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="search_corpus",
                description="Keyword search over the Acme document corpus.",
                trigger_prompt="Search the Acme corpus for the term 'revenue'.",
            ),
            ToolSpec(
                name="get_company_profile",
                description="Return the high-level Acme company profile.",
                trigger_prompt="Get the Acme company profile.",
            ),
        ]

    def corpus(self) -> Iterable[SourceDoc]:
        return self._docs.values()

    def query(
        self,
        prompt: str,
        *,
        conversation: list[Message] | None = None,
    ) -> RagResponse:
        t0 = time.perf_counter()
        retrieved = self._retrieve(prompt)
        tool_calls = self._tool_calls_for(prompt)

        if not retrieved:
            # Hallucination guardrail: refuse cleanly rather than make something up.
            return RagResponse(
                text=(
                    "I could not find any Acme documents that answer that question, "
                    "so I cannot give a grounded answer."
                ),
                citations=[],
                tool_calls=tool_calls,
                retrieved_context=[],
                raw={"latency_ms": int((time.perf_counter() - t0) * 1000)},
            )

        # Build a synthesis-style answer by quoting the top match.
        top = retrieved[0]
        text = (
            f"According to Acme records, {top.chunk.strip()} "
            f"[src:{top.source_id}]"
        )
        citations = [Citation(id=f"cite-{top.source_id}", source_id=top.source_id, span=top.chunk)]
        return RagResponse(
            text=text,
            citations=citations,
            tool_calls=tool_calls,
            retrieved_context=retrieved,
            raw={"latency_ms": int((time.perf_counter() - t0) * 1000)},
        )

    # ---------- internals ----------

    def _retrieve(self, prompt: str) -> list[ContextDoc]:
        lower = prompt.lower()
        # Score each doc by counting unique prompt-term hits in its body. Cheap and
        # deterministic — fine for a demo, terrible for production. That is the point.
        scored: list[tuple[float, SourceDoc, str]] = []
        for doc in self._docs.values():
            chunk = self._best_chunk(doc.content, lower)
            if chunk is None:
                continue
            hits = sum(1 for tok in self._tokens(lower) if tok in doc.content.lower())
            scored.append((float(hits), doc, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            ContextDoc(source_id=doc.source_id, chunk=chunk, score=score)
            for score, doc, chunk in scored[:3]
            if score > 0
        ]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [t for t in "".join(c if c.isalnum() else " " for c in text).split() if len(t) > 3]

    @staticmethod
    def _best_chunk(body: str, prompt_lower: str) -> str | None:
        # Return the first paragraph that contains any 4+ char token from the prompt.
        for para in (p.strip() for p in body.split("\n\n")):
            if not para or para.startswith("#"):
                continue
            tokens = DemoAdapter._tokens(prompt_lower)
            if any(tok in para.lower() for tok in tokens):
                return para
        return None

    def _tool_calls_for(self, prompt: str) -> list[ToolCall]:
        calls: list[ToolCall] = []
        lower = prompt.lower()
        if "search" in lower or any(w in lower for w in ("revenue", "product", "risk", "leader")):
            calls.append(
                ToolCall(name="search_corpus", args={"query": prompt}, result="ok", latency_ms=8)
            )
        if "profile" in lower or "company" in lower or "about" in lower:
            calls.append(
                ToolCall(
                    name="get_company_profile",
                    args={},
                    result={"name": "Acme"},
                    latency_ms=3,
                )
            )
        return calls
