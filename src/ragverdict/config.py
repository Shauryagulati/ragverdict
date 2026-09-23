"""Pydantic schemas + YAML loader for ragverdict configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ConfigError(Exception):
    """Raised when a config file is missing, malformed, or fails validation."""


class PythonAdapterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["python"] = "python"
    module: str
    cls: str = Field(alias="class")


class HttpAdapterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: Literal["http"] = "http"
    endpoint: str
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = 30.0


AdapterConfig = PythonAdapterConfig | HttpAdapterConfig


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["anthropic", "jev", "cascade"] = "anthropic"
    model: str = "claude-sonnet-4-6"
    thinking: Literal["model_default", "disabled"] = "model_default"
    max_concurrency: int = 4
    fixtures_path: Path | None = None
    jev_model: str = "typesafe/jev-1.13"
    jev_base_url: str = "https://openrouter.ai/api"
    cascade_band: tuple[float, float] = (0.3, 0.7)

    @field_validator("cascade_band")
    @classmethod
    def _band_in_range(cls, band: tuple[float, float]) -> tuple[float, float]:
        lo, hi = band
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError(f"cascade_band must satisfy 0 <= lo < hi <= 1, got {band}")
        return band


class Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    faithfulness_pass: float = 0.85
    faithfulness_weak: float = 0.7
    relevance_pass: float = 0.85
    relevance_weak: float = 0.7
    citation_support_pass: float = 0.95
    citation_support_weak: float = 0.8


class RagQualityCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    expects_citations: bool = False
    must_mention: list[str] = Field(default_factory=list)
    must_not_cite: bool = False
    must_refuse: bool = False


class ToolCoverageSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    require_all_tools: bool = True
    trigger_prompts: dict[str, str] = Field(default_factory=dict)


class CitationAuditSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_queries: list[str] = Field(default_factory=list)
    sample_size: int = 10


class TestSpec(BaseModel):
    """One entry in the top-level `tests:` list.

    `evaluator` selects which evaluator runs; the remaining fields are evaluator-specific
    and validated by the evaluator at run time. We keep them as a free-form dict here so
    evaluators can evolve their case shapes without churning Config.
    """

    __test__ = False  # tell pytest this is not a test class
    model_config = ConfigDict(extra="allow")

    name: str
    evaluator: str


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapter: AdapterConfig
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    tests: list[TestSpec] = Field(default_factory=list)


def load_config(path: Path) -> Config:
    """Read a YAML config file and validate it. Raises ConfigError on any failure."""
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw: Any = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"config validation failed:\n{exc}") from exc
