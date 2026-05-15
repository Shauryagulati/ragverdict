"""Core protocol types every RAG adapter speaks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Message:
    role: Literal["user", "assistant", "system"]
    content: str


@dataclass
class Citation:
    """A reference from a response back to a source document."""

    id: str
    source_id: str
    span: str = ""


@dataclass
class ToolCall:
    """A tool invocation the agent made while answering."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    latency_ms: int = 0
    error: str | None = None


@dataclass
class ContextDoc:
    """A retrieved chunk the agent considered while answering."""

    source_id: str
    chunk: str
    score: float = 0.0


@dataclass
class ToolSpec:
    """Describes a tool the adapter exposes for tool_coverage testing."""

    name: str
    description: str = ""
    trigger_prompt: str | None = None


@dataclass
class SourceDoc:
    """A document in the adapter's source corpus for citation_audit."""

    source_id: str
    content: str
    title: str = ""


@dataclass
class RagResponse:
    """What an adapter returns from `query`."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    retrieved_context: list[ContextDoc] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class RagAdapter(ABC):
    """Subclass and implement `query` to connect rag-eval to your RAG system."""

    @abstractmethod
    def query(
        self,
        prompt: str,
        *,
        conversation: list[Message] | None = None,
    ) -> RagResponse: ...

    def available_tools(self) -> list[ToolSpec]:
        """Tools the adapter exposes. Used by tool_coverage. Default: none."""
        return []

    def corpus(self) -> Iterable[SourceDoc] | None:
        """The source corpus citations resolve against. Default: not available."""
        return None
