"""OpenRouter chat judges over httpx.MockTransport — no network."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from ragverdict.bench.openrouter_llm import (
    DEEPSEEK_FLASH,
    GLM_FLASH,
    ChatJudgeConfig,
    chat_body,
    parse_chat_response,
    run_chat_judge,
)
from ragverdict.bench.predict import PredictionStore
from ragverdict.bench.ragtruth import Example
from ragverdict.judges.base import JudgeError, JudgeScore
from ragverdict.judges.llm_judge import faithfulness_prompt, output_schema


def _ex(i: int) -> Example:
    return Example(id=str(i), split="test", task="QA", generator="g", source=f"src {i}",
                   response=f"resp {i}", hallucinated=False, span_types=(), span_texts=(),
                   numeric=False, source_id="s")


def _ok(content: str, finish: str = "stop", provider: str | None = None) -> httpx.Response:
    body: dict = {
        "model": "deepseek/deepseek-v4.1-flash-20260801",
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 1300, "completion_tokens": 60, "cost": 0.00016},
    }
    if provider is not None:
        body["provider"] = provider
    return httpx.Response(200, json=body)


VALID = JudgeScore(score=0.5, reasoning="one claim unsupported").model_dump_json(exclude={"confidence"})


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_chat_body_uses_claude_rubric_and_schema() -> None:
    body = chat_body(DEEPSEEK_FLASH, "resp", "ctx")
    system, user = faithfulness_prompt("resp", "ctx")
    assert body["model"] == "deepseek/deepseek-v4.1-flash"
    assert body["messages"] == [{"role": "system", "content": system}, {"role": "user", "content": user}]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == output_schema(JudgeScore)
    assert body["max_tokens"] == 512 and body["temperature"] == 0.0
    assert body["reasoning"] == {"enabled": False}
    assert body["usage"] == {"include": True}
    assert body["provider"] == {"require_parameters": True}


def test_glm_body_omits_temperature_and_enables_reasoning() -> None:
    body = chat_body(GLM_FLASH, "r", "c")
    assert "temperature" not in body
    assert body["reasoning"] == {"effort": "low"} and body["max_tokens"] == 4096


def test_json_object_mode_puts_schema_in_system_prompt() -> None:
    cfg = ChatJudgeConfig("m", 100, json_mode="json_object")
    body = chat_body(cfg, "r", "c")
    assert body["response_format"] == {"type": "json_object"}
    assert json.dumps(output_schema(JudgeScore)) in body["messages"][0]["content"]


def test_parse_valid_and_fenced() -> None:
    assert parse_chat_response(_ok(VALID).json()).score == 0.5
    fenced = f"```json\n{VALID}\n```"
    assert parse_chat_response(_ok(fenced).json()).score == 0.5


def test_parse_truncated_raises() -> None:
    with pytest.raises(JudgeError, match="truncated"):
        parse_chat_response(_ok('{"score": 0.5, "rea', finish="length").json())


def test_parse_invalid_json_raises() -> None:
    with pytest.raises(JudgeError, match="invalid JSON"):
        parse_chat_response(_ok("not json").json())


def test_parse_missing_choices_raises() -> None:
    with pytest.raises(JudgeError, match="unexpected response shape"):
        parse_chat_response({"error": {"message": "x"}})


def test_run_records_usage_cost_and_model(tmp_path: Path) -> None:
    preds = run_chat_judge([_ex(1), _ex(2)], DEEPSEEK_FLASH, PredictionStore(tmp_path), "deepseek",
                           api_key="k", client=_client(lambda r: _ok(VALID)), backoff_s=0)
    assert [p.score for p in preds] == [0.5, 0.5]
    p = preds[0]
    assert p.input_tokens == 1300 and p.output_tokens == 60 and p.cost_usd == pytest.approx(0.00016)
    assert p.served_model == "deepseek/deepseek-v4.1-flash-20260801"
    assert p.reasoning == "one claim unsupported" and p.latency_s is not None


def test_run_records_provider_in_served_model(tmp_path: Path) -> None:
    preds = run_chat_judge([_ex(1)], DEEPSEEK_FLASH, PredictionStore(tmp_path), "deepseek",
                           api_key="k", client=_client(lambda r: _ok(VALID, provider="Fireworks")),
                           backoff_s=0)
    assert preds[0].served_model == "deepseek/deepseek-v4.1-flash-20260801 via Fireworks"


def test_run_records_invalid_output_as_error_row(tmp_path: Path) -> None:
    preds = run_chat_judge([_ex(1)], DEEPSEEK_FLASH, PredictionStore(tmp_path), "deepseek",
                           api_key="k", client=_client(lambda r: _ok("nope")), backoff_s=0)
    assert preds[0].score is None and "invalid JSON" in (preds[0].error or "")
    assert preds[0].cost_usd == pytest.approx(0.00016)  # a failed call still costs money
    assert preds[0].error_kind == "judge"


def test_run_retries_429(tmp_path: Path) -> None:
    responses = [httpx.Response(429, text="slow"), _ok(VALID)]
    preds = run_chat_judge([_ex(1)], DEEPSEEK_FLASH, PredictionStore(tmp_path), "deepseek",
                           api_key="k", client=_client(lambda r: responses.pop(0)), backoff_s=0)
    assert preds[0].score == 0.5


def test_run_non_retryable_http_error_is_error_row(tmp_path: Path) -> None:
    preds = run_chat_judge([_ex(1)], DEEPSEEK_FLASH, PredictionStore(tmp_path), "deepseek",
                           api_key="k", client=_client(lambda r: httpx.Response(400, text="bad")),
                           backoff_s=0)
    assert preds[0].score is None and "HTTP 400" in (preds[0].error or "")
    assert preds[0].error_kind == "transport"


def test_run_sends_bearer_key(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok(VALID)

    run_chat_judge([_ex(1)], DEEPSEEK_FLASH, PredictionStore(tmp_path), "deepseek", api_key="k",
                   client=_client(handler), backoff_s=0)
    assert seen[0].headers["Authorization"] == "Bearer k"
    assert str(seen[0].url) == "https://openrouter.ai/api/v1/chat/completions"
