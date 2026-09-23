"""build_judge + runner judge wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragverdict.config import Config, JudgeConfig
from ragverdict.judges.cascade import CascadeJudge
from ragverdict.judges.factory import build_judge
from ragverdict.judges.jev_judge import JevJudge
from ragverdict.judges.llm_judge import LLMJudge
from ragverdict.runner import Runner


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")


def test_anthropic_provider(keys: None) -> None:
    judge = build_judge(JudgeConfig(model="claude-sonnet-5", thinking="disabled"))
    assert isinstance(judge, LLMJudge)
    assert judge.model == "claude-sonnet-5"
    assert judge.thinking == "disabled"


def test_jev_provider(keys: None) -> None:
    judge = build_judge(JudgeConfig(provider="jev"))
    assert isinstance(judge, JevJudge)
    assert judge.model == "typesafe/jev-1.13"


def test_cascade_provider(keys: None) -> None:
    judge = build_judge(JudgeConfig(provider="cascade", cascade_band=(0.2, 0.8)))
    assert isinstance(judge, CascadeJudge)
    assert isinstance(judge.primary, JevJudge)
    assert isinstance(judge.fallback, LLMJudge)
    assert (judge.lo, judge.hi) == (0.2, 0.8)


def _config(provider: str) -> Config:
    return Config.model_validate(
        {"adapter": {"type": "python", "module": "m", "class": "C"}, "judge": {"provider": provider}}
    )


def test_runner_without_jev_key_runs_without_judge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    runner = Runner(_config("jev"), tmp_path)
    assert runner.judge is None
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_runner_builds_cascade_when_keys_present(keys: None, tmp_path: Path) -> None:
    assert isinstance(Runner(_config("cascade"), tmp_path).judge, CascadeJudge)
