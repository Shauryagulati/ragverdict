"""ragverdict — pytest for RAG agents."""

__version__ = "0.3.0"

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
from ragverdict.evaluators.base import Evaluator, TestResult, Verdict

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
