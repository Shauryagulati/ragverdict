"""Build the judge a config asks for."""

from __future__ import annotations

from ragverdict.config import JudgeConfig
from ragverdict.judges.base import Judge
from ragverdict.judges.cascade import CascadeJudge
from ragverdict.judges.jev_judge import JevJudge
from ragverdict.judges.llm_judge import LLMJudge


def build_judge(cfg: JudgeConfig) -> Judge:
    """Construct the configured judge. Raises JudgeError when credentials are missing."""
    if cfg.provider == "anthropic":
        return LLMJudge(model=cfg.model, thinking=cfg.thinking)
    jev = JevJudge(model=cfg.jev_model, base_url=cfg.jev_base_url)
    if cfg.provider == "jev":
        return jev
    llm = LLMJudge(model=cfg.model, thinking=cfg.thinking)
    return CascadeJudge(jev, llm, band=cfg.cascade_band)
