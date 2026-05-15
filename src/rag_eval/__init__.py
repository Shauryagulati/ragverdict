"""rag-eval — pytest for RAG agents."""

__version__ = "0.1.0"

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
from rag_eval.evaluators.base import Evaluator, TestResult, Verdict

__all__ = [
    "Citation",
    "ContextDoc",
    "Evaluator",
    "Message",
    "RagAdapter",
    "RagResponse",
    "SourceDoc",
    "TestResult",
    "ToolCall",
    "ToolSpec",
    "Verdict",
    "__version__",
]
