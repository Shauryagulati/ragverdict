"""JevJudge tests over httpx.MockTransport — no network."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from ragverdict.judges.base import Judge, JudgeError, JudgeTransportError
from ragverdict.judges.jev_judge import FAITHFULNESS_QUESTION, JevJudge

Handler = Callable[[httpx.Request], httpx.Response]


def _ok(p: float) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "typesafe/jev-1.13-20260917",
            "answers": {"q": {"type": "noul", "noul": p}},
            "usage": {"input_tokens": 300, "output_tokens": 20, "cost": 0.0000126},
        },
    )


def _judge(handler: Handler, **kwargs: object) -> JevJudge:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return JevJudge(api_key="test-key", client=client, backoff_s=0.0, **kwargs)  # type: ignore[arg-type]


def test_is_a_judge() -> None:
    assert isinstance(_judge(lambda r: _ok(0.5)), Judge)


def test_request_shape() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok(0.9)

    _judge(handler).faithfulness("the response", "the source")
    request = seen[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/systemone"
    assert request.headers["Authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert body["model"] == "typesafe/jev-1.13"
    assert body["state"] == {"source": "the source", "response": "the response"}
    assert body["questions"] == {"q": {"type": "noul", "instructions": FAITHFULNESS_QUESTION}}


def test_faithfulness_maps_probability() -> None:
    score = _judge(lambda r: _ok(0.83)).faithfulness("r", "s")
    assert score.score == pytest.approx(0.83)
    assert score.confidence == pytest.approx(0.83)
    assert "typesafe/jev-1.13" in score.reasoning and "0.83" in score.reasoning

    low = _judge(lambda r: _ok(0.2)).faithfulness("r", "s")
    assert low.confidence == pytest.approx(0.8)


def test_inverted_question_is_reoriented_to_p_supported() -> None:
    judge = _judge(lambda r: _ok(0.9), faithfulness_question="Does it add unsupported info?",
                   question_means_unsupported=True)
    assert judge.faithfulness("r", "s").score == pytest.approx(0.1)
    assert judge.faithfulness_answer("r", "s").p_yes == pytest.approx(0.1)


def test_boundary_probabilities() -> None:
    assert _judge(lambda r: _ok(0.5)).refusal("r", "q").is_refusal is True
    assert _judge(lambda r: _ok(0.49)).refusal("r", "q").is_refusal is False
    assert _judge(lambda r: _ok(0.0)).faithfulness("r", "s").confidence == 1.0
    assert _judge(lambda r: _ok(1.0)).pushback("r", "p").handled_correctly is True


def test_relevance_and_pushback_states() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return _ok(0.7)

    judge = _judge(handler)
    judge.relevance("resp", "the query")
    judge.pushback("resp", "the premise")
    assert bodies[0]["state"] == {"query": "the query", "response": "resp"}
    assert bodies[1]["state"] == {"false_premise": "the premise", "response": "resp"}


def test_401_is_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="bad key")

    with pytest.raises(JudgeError, match="HTTP 401"):
        _judge(handler).faithfulness("r", "s")
    assert calls == 1


def test_429_then_success_is_retried() -> None:
    responses = [httpx.Response(429, text="slow down"), _ok(0.6)]
    judge = _judge(lambda r: responses.pop(0))
    assert judge.faithfulness("r", "s").score == pytest.approx(0.6)
    assert responses == []


def test_retry_exhaustion_raises() -> None:
    with pytest.raises(JudgeError, match="after 3 attempts: HTTP 503"):
        _judge(lambda r: httpx.Response(503, text="down")).faithfulness("r", "s")


def test_network_error_is_retried() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("boom", request=request)
        return _ok(0.4)

    assert _judge(handler).faithfulness("r", "s").score == pytest.approx(0.4)
    assert attempts == 2


def test_malformed_body_raises() -> None:
    with pytest.raises(JudgeTransportError, match="unexpected response shape"):
        _judge(lambda r: httpx.Response(200, json={"answers": {}})).faithfulness("r", "s")


def test_non_json_body_raises_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json", headers={"content-type": "application/json"})

    with pytest.raises(JudgeTransportError, match="non-JSON body"):
        _judge(handler).faithfulness("r", "s")


def test_non_dict_body_raises_transport_error() -> None:
    with pytest.raises(JudgeTransportError, match="unexpected response shape"):
        _judge(lambda r: httpx.Response(200, json=[1, 2, 3])).faithfulness("r", "s")


def test_out_of_range_probability_raises() -> None:
    with pytest.raises(JudgeTransportError, match="out of range"):
        _judge(lambda r: _ok(1.5)).faithfulness("r", "s")


def test_usage_accounting() -> None:
    judge = _judge(lambda r: _ok(0.5))
    judge.faithfulness("r", "s")
    answer = judge.faithfulness_answer("r", "s")
    assert judge.calls == 2
    assert judge.input_tokens == 600
    assert judge.cost_usd == pytest.approx(0.0000252)
    assert judge.served_models == {"typesafe/jev-1.13-20260917"}
    assert answer.input_tokens == 300 and answer.served_model == "typesafe/jev-1.13-20260917"


def test_missing_openrouter_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(JudgeError, match="OPENROUTER_API_KEY"):
        JevJudge()


def test_typesafe_base_url_uses_typesafe_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JudgeError, match="TYPESAFE_API_KEY"):
        JevJudge(base_url="https://api.typesafe.ai", model="jev-1.13.0")


def test_malformed_usage_type_is_ignored() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "typesafe/jev-1.13-20260917",
                "answers": {"q": {"type": "noul", "noul": 0.7}},
                "usage": "n/a",  # Non-dict truthy value
            },
        )

    answer = _judge(handler).faithfulness_answer("r", "s")
    assert answer.p_yes == pytest.approx(0.7)
    assert answer.input_tokens == 0
    assert answer.cost_usd == 0.0
