"""Prediction store and runners with fake judges/clients — no network."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock

import anthropic
import pytest

from ragverdict.bench.predict import (
    BudgetError,
    ClaudeBatchError,
    ClaudeBatchRunner,
    Prediction,
    PredictionStore,
    check_budget,
    claude_cost,
    custom_id,
    parse_custom_id,
    run_claude_live,
    run_jev,
    total_spend,
)
from ragverdict.bench.ragtruth import Example
from ragverdict.judges.base import JudgeError, JudgeScore, JudgeTransportError
from ragverdict.judges.jev_judge import JevAnswer
from ragverdict.judges.llm_judge import LLMJudge


def _ex(i: int, hallucinated: bool = False) -> Example:
    return Example(id=str(i), split="test", task="QA", generator="g", source=f"src {i}",
                   response=f"resp {i}", hallucinated=hallucinated, span_types=(), span_texts=(),
                   numeric=False)


class FakeJev:
    def __init__(
        self, fail_ids: set[str] | None = None, error_cls: type[JudgeError] = JudgeTransportError
    ) -> None:
        self.fail_ids = fail_ids or set()
        self.calls: list[str] = []
        self.error_cls = error_cls

    def faithfulness_answer(self, response_text: str, retrieved_context: str) -> JevAnswer:
        eid = response_text.split()[-1]
        self.calls.append(eid)
        if eid in self.fail_ids:
            raise self.error_cls("jev boom")
        return JevAnswer(p_yes=0.25, input_tokens=100, cost_usd=0.0000042,
                         served_model="typesafe/jev-1.13-20260917", latency_s=0.3)


def test_custom_id_roundtrip() -> None:
    assert custom_id("123", 2) == "123-r2"
    assert parse_custom_id("123-r2") == ("123", 2)


def test_store_roundtrip_and_last_row_wins(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    store.append(Prediction(run="x", example_id="1", repeat=0, score=None, error="first"))
    store.append(Prediction(run="x", example_id="1", repeat=0, score=0.5))
    loaded = store.load("x")
    assert loaded[("1", 0)].score == 0.5
    assert store.load("missing") == {}


def test_load_skips_malformed_trailing_line(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    store.append(Prediction(run="x", example_id="1", repeat=0, score=0.5))
    path = store.path("x")
    with path.open("a") as fh:
        fh.write('{"run": "x", "example_id": "2", "repeat": 0, "sc')  # truncated, no newline
    loaded = store.load("x")
    assert loaded.keys() == {("1", 0)}


def test_load_raises_on_malformed_middle_line(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    store.append(Prediction(run="x", example_id="1", repeat=0, score=0.5))
    path = store.path("x")
    with path.open("a") as fh:
        # malformed JSON, but NOT the last line — a real partial write only ever
        # truncates the final line, so a broken line earlier means real corruption.
        fh.write('{"run": "x", "example_id": "2", "repeat": 0, "sc\n')
        fh.write(json.dumps({"run": "x", "example_id": "3", "repeat": 0, "score": 0.1}) + "\n")
    with pytest.raises(ValueError) as excinfo:
        store.load("x")
    assert str(path) in str(excinfo.value)
    assert "line 2" in str(excinfo.value)


def test_load_raises_on_unknown_field_row(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    store.append(Prediction(run="x", example_id="1", repeat=0, score=0.5))
    path = store.path("x")
    with path.open("a") as fh:
        # valid JSON, but doesn't match the Prediction schema — not the last line.
        fh.write(json.dumps({"run": "x", "example_id": "2", "repeat": 0, "score": 0.3,
                             "totally_unknown_field": "oops"}) + "\n")
        fh.write(json.dumps({"run": "x", "example_id": "3", "repeat": 0, "score": 0.1}) + "\n")
    with pytest.raises(ValueError) as excinfo:
        store.load("x")
    assert str(path) in str(excinfo.value)
    assert "line 2" in str(excinfo.value)


def test_load_tolerates_rows_missing_optional_fields(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    path = store.path("x")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        # a row written before error_kind existed — no error_kind key at all.
        fh.write(json.dumps({"run": "x", "example_id": "1", "repeat": 0, "score": 0.5}) + "\n")
    loaded = store.load("x")
    assert loaded[("1", 0)].score == 0.5
    assert loaded[("1", 0)].error_kind is None


def test_total_spend_sums_every_row_including_superseded(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    store.append(Prediction(run="x", example_id="1", repeat=0, score=None,
                            error="boom", error_kind="transport", cost_usd=0.01))
    store.append(Prediction(run="x", example_id="1", repeat=0, score=0.5, cost_usd=0.02))
    store.append(Prediction(run="x", example_id="2", repeat=0, score=0.3, cost_usd=0.03))
    assert total_spend(store, "x") == pytest.approx(0.06)
    assert total_spend(store, "missing") == 0.0


def test_run_jev_records_predictions_and_errors(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    preds = run_jev([_ex(1), _ex(2)], FakeJev(fail_ids={"2"}), store, "jev", workers=2)
    by_id = {p.example_id: p for p in preds}
    assert by_id["1"].score == 0.25 and by_id["1"].cost_usd == pytest.approx(0.0000042)
    assert by_id["1"].served_model == "typesafe/jev-1.13-20260917"
    assert by_id["2"].score is None and by_id["2"].error == "jev boom"
    assert by_id["2"].error_kind == "transport"


def test_run_jev_skips_cached_successes_and_retries_cached_errors(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    run_jev([_ex(1), _ex(2)], FakeJev(fail_ids={"2"}), store, "jev")
    fake = FakeJev()
    preds = run_jev([_ex(1), _ex(2)], fake, store, "jev")
    assert fake.calls == ["2"]
    assert all(p.error is None for p in preds)


def test_run_jev_does_not_retry_judge_kind_errors(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    run_jev([_ex(1), _ex(2)], FakeJev(fail_ids={"2"}, error_cls=JudgeError), store, "jev")
    stored = store.load("jev")
    assert stored[("2", 0)].error_kind == "judge"
    fake = FakeJev()
    run_jev([_ex(1), _ex(2)], fake, store, "jev")
    assert fake.calls == []  # example 2's judge-kind failure is final


def test_run_jev_repeats(tmp_path: Path) -> None:
    preds = run_jev([_ex(1)], FakeJev(), PredictionStore(tmp_path), "flip", repeats=3)
    assert sorted(p.repeat for p in preds) == [0, 1, 2]


def test_claude_cost() -> None:
    assert claude_cost("claude-sonnet-5", 1_000_000, 100_000, batch=False) == pytest.approx(3.0)
    assert claude_cost("claude-sonnet-5", 1_000_000, 100_000, batch=True) == pytest.approx(1.5)
    # 1M cache reads at 0.1x of $2 = $0.20; 1M cache writes at 1.25x = $2.50
    assert claude_cost("claude-sonnet-5", 0, 0, batch=False, cache_read_tokens=1_000_000) == pytest.approx(0.2)
    assert claude_cost("claude-sonnet-5", 0, 0, batch=False, cache_write_tokens=1_000_000) == pytest.approx(2.5)
    with pytest.raises(KeyError):
        claude_cost("unknown-model", 1, 1, batch=True)


def test_check_budget() -> None:
    assert check_budget(100, 0.01, 2.0) == pytest.approx(1.0)
    with pytest.raises(BudgetError, match="exceeds"):
        check_budget(1000, 0.01, 2.0)


# ---------- Claude batch ----------

def _message(score: float) -> Mock:
    return Mock(
        content=[Mock(type="text", text=JudgeScore(score=score, reasoning="r").model_dump_json())],
        stop_reason="end_turn",
        usage=Mock(input_tokens=1000, output_tokens=100, cache_read_input_tokens=500,
                   cache_creation_input_tokens=0),
    )


class FakeBatches:
    """Minimal stand-in for client.messages.batches."""

    def __init__(self, outcomes: list[dict[str, str]]) -> None:
        # outcomes[k] maps custom_id -> "ok" | "errored" | "expired" for the k-th submitted batch
        self.outcomes = outcomes
        self.submitted: list[list[dict[str, Any]]] = []
        self.polls = 0

    def create(self, *, requests: list[dict[str, Any]]) -> SimpleNamespace:
        self.submitted.append(requests)
        return SimpleNamespace(id=f"batch_{len(self.submitted)}")

    def retrieve(self, batch_id: str) -> SimpleNamespace:
        self.polls += 1
        return SimpleNamespace(processing_status="ended" if self.polls % 2 == 0 else "in_progress")

    def results(self, batch_id: str) -> list[SimpleNamespace]:
        k = int(batch_id.split("_")[1]) - 1
        out = []
        for req in self.submitted[k]:
            status = self.outcomes[k].get(req["custom_id"], "ok")
            if status == "ok":
                result = SimpleNamespace(type="succeeded", message=_message(0.5))
            elif status == "errored":
                result = SimpleNamespace(type="errored", error="overloaded")
            else:
                result = SimpleNamespace(type="expired")
            out.append(SimpleNamespace(custom_id=req["custom_id"], result=result))
        return out


def _batch_runner(tmp_path: Path, batches: FakeBatches) -> tuple[ClaudeBatchRunner, LLMJudge]:
    judge = LLMJudge(client=MagicMock(spec=anthropic.Anthropic), model="claude-sonnet-5",
                     thinking="disabled")
    client = SimpleNamespace(messages=SimpleNamespace(batches=batches))
    return ClaudeBatchRunner(judge, PredictionStore(tmp_path), client=client, poll_s=0,
                             sleep=lambda s: None), judge


def test_batch_submits_shipped_requests_and_parses(tmp_path: Path) -> None:
    batches = FakeBatches([{}])
    runner, judge = _batch_runner(tmp_path, batches)
    preds = runner.run([_ex(1), _ex(2)], "claude")
    request = batches.submitted[0][0]
    assert request["custom_id"] == "1-r0"
    assert request["params"] == judge.faithfulness_request("resp 1", "src 1")
    assert {p.example_id: p.score for p in preds} == {"1": 0.5, "2": 0.5}
    assert preds[0].cache_read_tokens == 500
    assert preds[0].cost_usd == pytest.approx(
        claude_cost("claude-sonnet-5", 1000, 100, batch=True, cache_read_tokens=500)
    )


def test_failed_requests_retried_once_then_recorded(tmp_path: Path) -> None:
    batches = FakeBatches([{"1-r0": "errored", "2-r0": "expired"}, {"2-r0": "expired"}])
    runner, _ = _batch_runner(tmp_path, batches)
    preds = {p.example_id: p for p in runner.run([_ex(1), _ex(2)], "claude")}
    assert [r["custom_id"] for r in batches.submitted[1]] == ["1-r0", "2-r0"]
    assert len(batches.submitted) == 2
    assert preds["1"].score == 0.5
    assert preds["2"].score is None and "expired" in (preds["2"].error or "")


def test_resume_uses_logged_batch(tmp_path: Path) -> None:
    batches = FakeBatches([{}])
    runner, _ = _batch_runner(tmp_path, batches)
    batches.create(requests=[{"custom_id": "1-r0", "params": {}}])  # submitted, then "crash"
    log = tmp_path / "batches" / "claude.json"
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps([{"batch_id": "batch_1", "ingested": False}]))
    preds = runner.run([_ex(1)], "claude")
    assert len(batches.submitted) == 1  # no resubmission
    assert preds[0].score == 0.5


def test_resumed_batch_is_marked_ingested(tmp_path: Path) -> None:
    batches = FakeBatches([{}])
    runner, _ = _batch_runner(tmp_path, batches)
    batches.create(requests=[{"custom_id": "1-r0", "params": {}}])
    log_path = tmp_path / "batches" / "claude.json"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(json.dumps([{"batch_id": "batch_1", "ingested": False}]))
    runner.run([_ex(1)], "claude")
    logged = json.loads(log_path.read_text())
    assert logged[0]["ingested"] is True


def test_batch_results_outside_by_id_are_recorded_not_retried(tmp_path: Path) -> None:
    batches = FakeBatches([{"9-r0": "errored"}])
    runner, judge = _batch_runner(tmp_path, batches)
    # a prior process submitted a batch covering examples "1" and "9"; this run only
    # asks about "1" (e.g. "9" was dropped from the requested set after the crash).
    batches.create(requests=[
        {"custom_id": "1-r0", "params": judge.faithfulness_request("resp 1", "src 1")},
        {"custom_id": "9-r0", "params": judge.faithfulness_request("resp 9", "src 9")},
    ])
    log = tmp_path / "batches" / "claude.json"
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps([{"batch_id": "batch_1", "ingested": False}]))
    preds = runner.run([_ex(1)], "claude")
    assert {p.example_id for p in preds} == {"1"}
    all_rows = runner.store.load("claude")
    assert ("9", 0) in all_rows
    assert all_rows[("9", 0)].error is not None and all_rows[("9", 0)].error_kind == "transport"
    assert len(batches.submitted) == 1  # "9" was recorded, not retried (not in by_id)


def test_resume_continues_to_new_pending_items(tmp_path: Path) -> None:
    batches = FakeBatches([{}, {}])
    runner, judge = _batch_runner(tmp_path, batches)
    batches.create(requests=[{"custom_id": "1-r0", "params": judge.faithfulness_request("resp 1", "src 1")}])
    log = tmp_path / "batches" / "claude.json"
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps([{"batch_id": "batch_1", "ingested": False}]))
    preds = runner.run([_ex(1), _ex(2)], "claude")
    assert {p.example_id: p.score for p in preds} == {"1": 0.5, "2": 0.5}
    assert len(batches.submitted) == 2
    assert batches.submitted[1][0]["custom_id"] == "2-r0"


def test_stale_submitting_marker_raises_and_does_not_resubmit(tmp_path: Path) -> None:
    batches = FakeBatches([{}])
    runner, _ = _batch_runner(tmp_path, batches)
    log = tmp_path / "batches" / "claude.json"
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps([{"batch_id": None, "submitting": True, "n_requests": 2}]))
    with pytest.raises(ClaudeBatchError, match=r"batches\.list"):
        runner.run([_ex(1)], "claude")
    assert len(batches.submitted) == 0


def test_log_has_real_batch_id_before_polling_starts(tmp_path: Path) -> None:
    batches = FakeBatches([{}])
    runner, _ = _batch_runner(tmp_path, batches)

    def boom(batch_id: str) -> SimpleNamespace:
        raise RuntimeError("simulated crash during poll")

    batches.retrieve = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="simulated crash"):
        runner.run([_ex(1)], "claude")
    log = json.loads((tmp_path / "batches" / "claude.json").read_text())
    assert log[-1]["batch_id"] == "batch_1"
    assert log[-1]["ingested"] is False


def test_batch_skips_cached(tmp_path: Path) -> None:
    batches = FakeBatches([{}])
    runner, _ = _batch_runner(tmp_path, batches)
    runner.run([_ex(1)], "claude")
    runner.run([_ex(1)], "claude")
    assert len(batches.submitted) == 1


def test_run_claude_live_records_latency_and_tokens(tmp_path: Path) -> None:
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.return_value = _message(0.75)
    judge = LLMJudge(client=client, model="claude-sonnet-5", thinking="disabled")
    preds = run_claude_live([_ex(1), _ex(2)], judge, PredictionStore(tmp_path), "live")
    assert [p.score for p in preds] == [0.75, 0.75]
    assert preds[1].input_tokens == 1000 and preds[1].output_tokens == 100
    assert preds[1].cache_read_tokens == 500
    assert preds[0].latency_s is not None
    assert preds[0].cost_usd == pytest.approx(
        claude_cost("claude-sonnet-5", 1000, 100, batch=False, cache_read_tokens=500)
    )
