"""DemoAdapter — a tiny deterministic RAG over a markdown corpus.

This is the reference adapter used in `ragverdict`'s own demo and tests. It deliberately
exposes a small surface so all three V0 evaluators (tool_coverage, rag_quality,
citation_audit) have something concrete to bite into.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from pathlib import Path

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
        # Lowercased corpus text — the groundedness guard checks entity names
        # against this to decide whether the agent actually has any record of them.
        self._corpus_text = " ".join(d.content for d in self._docs.values()).lower()

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

        # Edge-case handling — keeps `edge_cases` happy without complicating the
        # golden-path code that drives the other three evaluators.
        if not prompt.strip():
            return RagResponse(
                text="I cannot process empty input — please provide a question.",
                raw={"latency_ms": int((time.perf_counter() - t0) * 1000)},
            )

        # Fold prior user turns into retrieval so multi-turn references work
        # ("What was the name you mentioned?" needs the earlier "Who is the CEO?"
        # to find Jane Smith in leadership.md). Keeps the demo deterministic
        # rather than re-asking the model.
        search_text = prompt
        if conversation:
            prior_user_turns = " ".join(
                m.content for m in conversation if m.role == "user"
            )
            if prior_user_turns:
                search_text = f"{prior_user_turns} {prompt}"

        retrieved = self._retrieve(search_text)
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

        # Groundedness guard: if the question names an entity we have no record of,
        # push back instead of dressing up a loosely-matched chunk as an answer —
        # e.g. don't validate "Acme's acquisition of XYZ Corp" just because the
        # company-overview doc happens to mention "Acme Corp".
        unknown = self._unknown_entities(prompt)
        if unknown:
            return RagResponse(
                text=(
                    f"I have no record of {unknown[0]} in the Acme corpus, so I "
                    f"cannot speak to that premise."
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

    # Minimum prompt-token hits in a paragraph for it to count as a match. Anything below
    # this threshold triggers a refusal — the demo's stand-in for an "out of corpus" check.
    MIN_HITS = 2

    _STOPWORDS = frozenset(
        {
            "the", "and", "for", "with", "what", "who", "did", "are", "was", "has",
            "is", "of", "in", "on", "at", "to", "a", "an", "by", "or", "be", "it",
            "this", "that", "any", "all", "from", "have", "had", "does", "do", "as",
        }
    )

    # Capitalized multi-word phrases ("XYZ Corp", "Chief Marketing Officer") — a
    # lightweight, dependency-free stand-in for named-entity recognition. A real
    # agent would use NER or let the judge assess groundedness.
    _ENTITY_RE = re.compile(r"\b[A-Z][\w]*(?:\s+[A-Z][\w]*)+\b")

    def _unknown_entities(self, prompt: str) -> list[str]:
        """Multi-word proper nouns in `prompt` that appear nowhere in the corpus.

        The agent declines on these rather than answer a question built on an
        entity it has no record of — the check that makes it push back on a false
        premise instead of confabulating around a loosely-matched chunk.
        """
        matches: list[str] = self._ENTITY_RE.findall(prompt)
        return [m for m in matches if m.lower() not in self._corpus_text]

    def _retrieve(self, prompt: str) -> list[ContextDoc]:
        tokens = self._tokens(prompt.lower())
        if not tokens:
            return []
        scored: list[tuple[int, SourceDoc, str]] = []
        for doc in self._docs.values():
            hits, chunk = self._best_chunk(doc.content, tokens)
            if hits >= self.MIN_HITS and chunk is not None:
                scored.append((hits, doc, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            ContextDoc(source_id=doc.source_id, chunk=chunk, score=float(score))
            for score, doc, chunk in scored[:3]
        ]

    @classmethod
    def _tokens(cls, text: str) -> list[str]:
        return [
            t
            for t in "".join(c if c.isalnum() else " " for c in text).split()
            if len(t) >= 2 and t not in cls._STOPWORDS
        ]

    @classmethod
    def _best_chunk(cls, body: str, tokens: list[str]) -> tuple[int, str | None]:
        """Pick the paragraph with the most token hits. Returns (hit_count, chunk)."""
        best_hits = 0
        best_chunk: str | None = None
        for para in (p.strip() for p in body.split("\n\n")):
            if not para or para.startswith("#"):
                continue
            lower = para.lower()
            hits = sum(1 for tok in tokens if tok in lower)
            if hits > best_hits:
                best_hits = hits
                best_chunk = para
        return best_hits, best_chunk

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
