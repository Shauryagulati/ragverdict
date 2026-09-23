"""Run judges over examples and cache every prediction to append-only JSONL.

Reruns never recompute a cached success, so a crashed or repeated run never pays
twice; cached errors are retried. Claude goes through the Batch API (50% price)
using exactly the request LLMJudge sends live. Submitted batch ids are logged
before waiting so an interrupted run resumes the same batch.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import anthropic

from ragverdict.bench.ragtruth import Example
from ragverdict.judges.base import JudgeError, JudgeScore
from ragverdict.judges.jev_judge import JevAnswer
from ragverdict.judges.llm_judge import LLMJudge, parse_judge_message

# Standard list prices, $ per million tokens (input, output), as of 2026-09-22.
# The Batch API bills 50% of these.
CLAUDE_PRICES: dict[str, tuple[float, float]] = {"claude-sonnet-5": (2.00, 10.00)}


@dataclass
class Prediction:
    run: str
    example_id: str
    repeat: int
    score: float | None  # Jev: P(supported). Claude: supported-claim fraction.
    input_tokens: int = 0  # uncached input only (Anthropic reports cache tokens separately)
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float | None = None
    served_model: str = ""
    reasoning: str = ""
    error: str | None = None
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class PredictionStore:
    """`<root>/raw/<run>.jsonl`, one Prediction per line; the last row per key wins."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()

    def path(self, run: str) -> Path:
        return self.root / "raw" / f"{run}.jsonl"

    def load(self, run: str) -> dict[tuple[str, int], Prediction]:
        path = self.path(run)
        if not path.exists():
            return {}
        rows: dict[tuple[str, int], Prediction] = {}
        with path.open() as fh:
            for line in fh:
                if line.strip():
                    pred = Prediction(**json.loads(line))
                    rows[(pred.example_id, pred.repeat)] = pred
        return rows

    def append(self, pred: Prediction) -> None:
        path = self.path(pred.run)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as fh:
                fh.write(json.dumps(asdict(pred)) + "\n")


def custom_id(example_id: str, repeat: int) -> str:
    return f"{example_id}-r{repeat}"  # batch ids allow only [a-zA-Z0-9_-]


def parse_custom_id(cid: str) -> tuple[str, int]:
    example_id, repeat = cid.rsplit("-r", 1)
    return example_id, int(repeat)


def claude_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    batch: bool,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Dollar cost of one call. Cache reads bill at 0.1x input, 5-min cache writes at 1.25x."""
    price_in, price_out = CLAUDE_PRICES[model]
    cost = (
        input_tokens * price_in
        + cache_read_tokens * price_in * 0.1
        + cache_write_tokens * price_in * 1.25
        + output_tokens * price_out
    ) / 1_000_000
    return cost / 2 if batch else cost


class BudgetError(Exception):
    """Raised when a run's estimated cost exceeds the allowed spend."""


def check_budget(n_calls: int, cost_per_call: float, max_spend_usd: float) -> float:
    estimate = n_calls * cost_per_call
    if estimate > max_spend_usd:
        raise BudgetError(
            f"estimated ${estimate:.2f} for {n_calls} calls exceeds max spend ${max_spend_usd:.2f}"
        )
    return estimate


def pending(
    examples: Sequence[Example], done: dict[tuple[str, int], Prediction], repeats: int
) -> list[tuple[Example, int]]:
    return [
        (ex, r)
        for ex in examples
        for r in range(repeats)
        if (ex.id, r) not in done or done[(ex.id, r)].error is not None
    ]


def collect(
    store: PredictionStore, run: str, examples: Sequence[Example], repeats: int
) -> list[Prediction]:
    done = store.load(run)
    return [done[(ex.id, r)] for ex in examples for r in range(repeats) if (ex.id, r) in done]


class _JevLike(Protocol):
    def faithfulness_answer(self, response_text: str, retrieved_context: str) -> JevAnswer: ...


def run_jev(
    examples: Sequence[Example],
    judge: _JevLike,
    store: PredictionStore,
    run: str,
    *,
    repeats: int = 1,
    workers: int = 8,
) -> list[Prediction]:
    def one(item: tuple[Example, int]) -> None:
        ex, repeat = item
        try:
            answer = judge.faithfulness_answer(ex.response, ex.source)
        except JudgeError as exc:
            store.append(Prediction(run=run, example_id=ex.id, repeat=repeat, score=None, error=str(exc)))
            return
        store.append(
            Prediction(
                run=run,
                example_id=ex.id,
                repeat=repeat,
                score=answer.p_yes,
                input_tokens=answer.input_tokens,
                cost_usd=answer.cost_usd,
                latency_s=answer.latency_s,
                served_model=answer.served_model,
            )
        )

    todo = pending(examples, store.load(run), repeats)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))
    return collect(store, run, examples, repeats)


def run_claude_live(
    examples: Sequence[Example], judge: LLMJudge, store: PredictionStore, run: str
) -> list[Prediction]:
    """Sequential live calls — used only to measure real latency."""
    for ex, repeat in pending(examples, store.load(run), 1):
        before_in, before_out = judge.input_tokens, judge.output_tokens
        before_read, before_write = judge.cache_read_tokens, judge.cache_creation_tokens
        started = time.perf_counter()
        try:
            score: JudgeScore | None = judge.faithfulness(ex.response, ex.source)
            error = None
        except JudgeError as exc:
            score, error = None, str(exc)
        latency = time.perf_counter() - started
        tokens_in = judge.input_tokens - before_in
        tokens_out = judge.output_tokens - before_out
        cache_read = judge.cache_read_tokens - before_read
        cache_write = judge.cache_creation_tokens - before_write
        store.append(
            Prediction(
                run=run,
                example_id=ex.id,
                repeat=repeat,
                score=score.score if score else None,
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                cost_usd=claude_cost(
                    judge.model, tokens_in, tokens_out, batch=False,
                    cache_read_tokens=cache_read, cache_write_tokens=cache_write,
                ),
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                latency_s=latency,
                served_model=judge.model,
                reasoning=score.reasoning if score else "",
                error=error,
            )
        )
    return collect(store, run, examples, 1)


class ClaudeBatchRunner:
    def __init__(
        self,
        judge: LLMJudge,
        store: PredictionStore,
        *,
        client: Any = None,
        poll_s: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.judge = judge
        self.store = store
        self.client: Any = client if client is not None else anthropic.Anthropic()
        self.poll_s = poll_s
        self.sleep = sleep

    def run(self, examples: Sequence[Example], run: str, *, repeats: int = 1) -> list[Prediction]:
        by_id = {ex.id: ex for ex in examples}
        batch_id = self._unfinished_batch(run)
        if batch_id is None:
            todo = pending(examples, self.store.load(run), repeats)
            if not todo:
                return collect(self.store, run, examples, repeats)
            batch_id = self._submit(run, todo)
        failed = self._finish(run, batch_id, by_id, record_failures=False)
        if failed:
            retry_id = self._submit(run, [(by_id[eid], r) for eid, r in failed])
            self._finish(run, retry_id, by_id, record_failures=True)
        return collect(self.store, run, examples, repeats)

    # ---------- internals ----------

    def _log_path(self, run: str) -> Path:
        return self.store.root / "batches" / f"{run}.json"

    def _read_log(self, run: str) -> list[dict[str, Any]]:
        path = self._log_path(run)
        return json.loads(path.read_text()) if path.exists() else []

    def _write_log(self, run: str, entries: list[dict[str, Any]]) -> None:
        path = self._log_path(run)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries, indent=2))

    def _unfinished_batch(self, run: str) -> str | None:
        for entry in self._read_log(run):
            if not entry["ingested"]:
                return str(entry["batch_id"])
        return None

    def _submit(self, run: str, items: Sequence[tuple[Example, int]]) -> str:
        requests = [
            {
                "custom_id": custom_id(ex.id, repeat),
                "params": self.judge.faithfulness_request(ex.response, ex.source),
            }
            for ex, repeat in items
        ]
        batch = self.client.messages.batches.create(requests=requests)
        self._write_log(run, [*self._read_log(run), {"batch_id": batch.id, "ingested": False}])
        return str(batch.id)

    def _finish(
        self, run: str, batch_id: str, by_id: dict[str, Example], *, record_failures: bool
    ) -> list[tuple[str, int]]:
        while self.client.messages.batches.retrieve(batch_id).processing_status != "ended":
            self.sleep(self.poll_s)
        failed: list[tuple[str, int]] = []
        for item in self.client.messages.batches.results(batch_id):
            example_id, repeat = parse_custom_id(item.custom_id)
            if example_id not in by_id:
                continue
            result = item.result
            if result.type == "succeeded":
                self.store.append(self._parse(run, example_id, repeat, result.message))
                continue
            reason = f"batch {result.type}: {getattr(result, 'error', '')}".strip()
            if record_failures:
                self.store.append(
                    Prediction(run=run, example_id=example_id, repeat=repeat, score=None, error=reason)
                )
            else:
                failed.append((example_id, repeat))
        entries = self._read_log(run)
        for entry in entries:
            if entry["batch_id"] == batch_id:
                entry["ingested"] = True
        self._write_log(run, entries)
        return failed

    def _parse(self, run: str, example_id: str, repeat: int, message: Any) -> Prediction:
        usage = message.usage
        tokens_in, tokens_out = int(usage.input_tokens), int(usage.output_tokens)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        cost = claude_cost(self.judge.model, tokens_in, tokens_out, batch=True,
                           cache_read_tokens=cache_read, cache_write_tokens=cache_write)
        try:
            score = parse_judge_message(message, JudgeScore)
        except JudgeError as exc:
            return Prediction(run=run, example_id=example_id, repeat=repeat, score=None,
                              input_tokens=tokens_in, output_tokens=tokens_out, cost_usd=cost,
                              cache_read_tokens=cache_read, cache_write_tokens=cache_write,
                              served_model=self.judge.model, error=str(exc))
        return Prediction(run=run, example_id=example_id, repeat=repeat, score=score.score,
                          input_tokens=tokens_in, output_tokens=tokens_out, cost_usd=cost,
                          cache_read_tokens=cache_read, cache_write_tokens=cache_write,
                          served_model=self.judge.model, reasoning=score.reasoning)
