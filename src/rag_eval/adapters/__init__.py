"""Adapters connect rag-eval to user-defined RAG systems."""

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

__all__ = [
    "Citation",
    "ContextDoc",
    "Message",
    "RagAdapter",
    "RagResponse",
    "SourceDoc",
    "ToolCall",
    "ToolSpec",
]
