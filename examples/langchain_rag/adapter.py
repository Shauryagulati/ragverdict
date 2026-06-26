"""LangChain-backed RAG adapter — starter template.

This file is a **starter** you copy into your own project and modify. It shows the
shape of a LangChain-based RAG adapter:

  vectorstore.as_retriever(k=3) → RetrievalQA.invoke → parse result + source_documents

The default implementation uses an in-memory substring index instead of a real
LangChain vector store + chain, so it runs offline without any dependencies. Replace
the four marked methods with calls into your real LangChain pipeline.

To use as a starting template:
  1. Copy this file into your project (e.g., `my_app/rag_adapter.py`).
  2. Install the real LangChain stack: `pip install langchain langchain-openai
     langchain-chroma` (or whichever providers you use).
  3. Replace each `# REPLACE:` block with a call into your real chain.
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

# Tiny in-memory corpus for the runnable demo. In your real project these would be
# loaded into a Chroma / FAISS / Qdrant vector store via LangChain's document loaders.
_CORPUS: dict[str, str] = {
    "OVERVIEW": (
        "Acme Logistics operates a fleet management platform for last-mile delivery, "
        "with 200+ enterprise customers across North America and Europe."
    ),
    "INTEGRATIONS": (
        "Acme integrates with Salesforce, NetSuite, and SAP via REST APIs and "
        "supports webhook-based event streaming for real-time fleet updates."
    ),
}


class LangChainRagAdapter(RagAdapter):
    """Subclass of RagAdapter that demonstrates the LangChain integration pattern.

    The real version would hold a `RetrievalQA` chain (or any LCEL chain that returns
    `{"result": str, "source_documents": [Document, ...]}`). The mock version below
    uses substring matching so it runs offline.
    """

    def __init__(self, *, chain: object | None = None) -> None:
        # REPLACE: hold your LangChain chain handle here.
        #   from langchain.chains import RetrievalQA
        #   from langchain_openai import ChatOpenAI
        #   from langchain_chroma import Chroma
        #   vectorstore = Chroma(persist_directory="./chroma", embedding_function=...)
        #   self._chain = RetrievalQA.from_chain_type(
        #       llm=ChatOpenAI(model="gpt-4o-mini"),
        #       retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        #       return_source_documents=True,
        #   )
        self._chain = chain  # unused in the mock

    # ---- ragverdict contract -----------------------------------------------

    def query(
        self,
        prompt: str,
        *,
        conversation: list[Message] | None = None,
    ) -> RagResponse:
        # In a real LangChain adapter, conversation history typically goes through a
        # memory abstraction (ConversationBufferMemory) attached to the chain. Here
        # we fold it into the search text so multi-turn cases work in the mock.
        search_text = prompt
        if conversation:
            prior = " ".join(m.content for m in conversation if m.role == "user")
            if prior:
                search_text = f"{prior} {prompt}"

        retrieved = self._retrieve(search_text, top_k=3)
        tool_calls = self._tool_calls_for(prompt)
        answer_text, citations = self._generate(prompt, retrieved)
        return RagResponse(
            text=answer_text,
            citations=citations,
            tool_calls=tool_calls,
            retrieved_context=retrieved,
        )

    def available_tools(self) -> list[ToolSpec]:
        # REPLACE: if your LangChain agent uses tools (via the Tools/AgentExecutor
        # APIs), list them here so `tool_coverage` can fire each one.
        return [
            ToolSpec(
                name="retriever",
                description="Vector-store retrieval over Acme docs.",
                trigger_prompt="Search for information about Salesforce integration.",
            ),
        ]

    def corpus(self) -> Iterable[SourceDoc]:
        # REPLACE: yield from your real LangChain document store so citation_audit
        # can validate `[src:ID]` references resolve to real source documents.
        for source_id, content in _CORPUS.items():
            yield SourceDoc(source_id=source_id, content=content, title=source_id.title())

    # ---- internals (the four methods to replace) -------------------------

    def _retrieve(self, query: str, *, top_k: int) -> list[ContextDoc]:
        """REPLACE: real implementation calls your LangChain retriever.

        Real version (sketch):
            docs = self._chain.retriever.invoke(query)
            return [
                ContextDoc(
                    source_id=doc.metadata.get("source_id", "unknown"),
                    chunk=doc.page_content,
                    score=doc.metadata.get("score", 0.0),
                )
                for doc in docs[:top_k]
            ]
        """
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
    ) -> tuple[str, list[Citation]]:
        """REPLACE: real implementation invokes your LangChain RetrievalQA chain.

        Real version (sketch):
            result = self._chain.invoke({"query": prompt})
            text = result["result"]
            source_docs = result.get("source_documents", [])
            citations = [
                Citation(
                    id=f"cite-{doc.metadata['source_id']}",
                    source_id=doc.metadata["source_id"],
                    span=doc.page_content[:200],
                )
                for doc in source_docs
            ]
            return text, citations
        """
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
        """REPLACE: if you wrap your chain in an AgentExecutor, intermediate steps
        contain tool invocations.

        Real version (sketch):
            result = self._chain.invoke({"input": prompt}, return_intermediate_steps=True)
            tool_calls = []
            for action, observation in result["intermediate_steps"]:
                tool_calls.append(ToolCall(
                    name=action.tool,
                    args=action.tool_input if isinstance(action.tool_input, dict) else {"input": str(action.tool_input)},
                    result=str(observation),
                ))
            return tool_calls
        """
        # Mock: fire the retriever tool for any prompt likely to trigger retrieval.
        if any(k in prompt.lower() for k in ("search", "salesforce", "integration", "platform")):
            return [
                ToolCall(name="retriever", args={"query": prompt}, result="ok", latency_ms=15)
            ]
        return []
