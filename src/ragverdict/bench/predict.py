"""Run judges over examples and cache every prediction to append-only JSONL.

Reruns never recompute a cached success, so a crashed or repeated run never pays
twice; cached transport errors (network/HTTP/API failures) are retried, but a
judge-output failure (invalid JSON, truncation, refusal, an out-of-range score)
is final — retrying it won't fix a bad answer, so `pending()` only retries rows
whose `error_kind` is "transport" (or unset, for legacy rows written before this
distinction existed). Claude goes through the Batch API (50% price) using
exactly the request LLMJudge sends live. Submitted batch ids are logged before
waiting so an interrupted run resumes the same batch; log writes are atomic
(temp file + os.replace) and a batch's submission is logged as a "submitting"
marker before `batches.create` is called, so a crash mid-submit is detected on
the next run instead of silently risking a duplicate submission.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import anthropic

from ragverdict.bench.ragtruth import Example
from ragverdict.judges.base import JudgeError, JudgeScore, JudgeTransportError
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
    error_kind: str | None = None  # "transport" (retryable) | "judge" (final) | None (no error)
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    supported_claims: int = 0
    total_claims: int = 0
    response_id: str = ""  # provider response id (Anthropic message.id / OpenRouter data["id"])


def _iter_predictions(path: Path) -> Iterator[Prediction]:
    """Yield every well-formed row in `path`, skipping only a malformed trailing line.

    A crash mid-write can leave a truncated final line; that row never completed
    and carries no information, so it's skipped rather than crashing the load. Any
    other malformed line — malformed JSON, or a row whose fields don't match
    `Prediction` — is real corruption, not a partial write, and must not be
    silently dropped: a dropped row would be silently re-billed (retried) and
    missed by `total_spend`. So it raises a clear ValueError naming the file and
    line number instead.
    """
    if not path.exists():
        return
    lines = path.read_text().splitlines()
    last_index = len(lines) - 1
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        is_last = index == last_index
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            if is_last:
                continue
            raise ValueError(
                f"malformed JSON in {path} at line {index + 1}: {exc}"
            ) from exc
        try:
            yield Prediction(**data)
        except TypeError as exc:
            if is_last:
                continue
            raise ValueError(
                f"row in {path} at line {index + 1} doesn't match the Prediction schema: {exc}"
            ) from exc


class PredictionStore:
    """`<root>/raw/<run>.jsonl`, one Prediction per line; the last row per key wins."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()

    def path(self, run: str) -> Path:
        return self.root / "raw" / f"{run}.jsonl"

    def load(self, run: str) -> dict[tuple[str, int], Prediction]:
        rows: dict[tuple[str, int], Prediction] = {}
        for pred in _iter_predictions(self.path(run)):
            rows[(pred.example_id, pred.repeat)] = pred
        return rows

    def append(self, pred: Prediction) -> None:
        path = self.path(pred.run)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as fh:
                fh.write(json.dumps(asdict(pred)) + "\n")


def total_spend(store: PredictionStore, run: str) -> float:
    """Total cost_usd over every raw row of `run`, including superseded/retried attempts.

    Unlike `collect()`, this doesn't dedupe by (example_id, repeat) — a retried call still
    cost money the first time, so the budget must count it too.
    """
    return sum(pred.cost_usd for pred in _iter_predictions(store.path(run)))


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


class ClaudeBatchError(Exception):
    """Raised when a batch run needs human intervention before continuing.

    Never auto-resubmits in this state — an ambiguous or in-flight submission could
    otherwise be double-billed.
    """


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
    def needs_retry(pred: Prediction) -> bool:
        # Judge-output failures are final — retrying won't fix a bad answer. Only
        # transport failures (and legacy rows with no error_kind) get retried.
        return pred.error is not None and pred.error_kind != "judge"

    return [
        (ex, r)
        for ex in examples
        for r in range(repeats)
        if (ex.id, r) not in done or needs_retry(done[(ex.id, r)])
    ]


def collect(
    store: PredictionStore, run: str, examples: Sequence[Example], repeats: int
) -> list[Prediction]:
    done = store.load(run)
    return [done[(ex.id, r)] for ex in examples for r in range(repeats) if (ex.id, r) in done]


class _JevLike(Protocol):
    faithfulness_question: str
    question_means_unsupported: bool

    def faithfulness_answer(self, response_text: str, retrieved_context: str) -> JevAnswer: ...
    def ask(self, state: dict[str, str], question: str) -> JevAnswer: ...


def _reorient_to_supported(answer: JevAnswer) -> JevAnswer:
    """Same re-orientation `JevJudge.faithfulness_answer` applies when the configured
    question means "is it unsupported?" — P(yes) becomes P(supported) = 1 - P(yes)."""
    return JevAnswer(
        p_yes=1.0 - answer.p_yes,
        input_tokens=answer.input_tokens,
        cost_usd=answer.cost_usd,
        served_model=answer.served_model,
        latency_s=answer.latency_s,
        response_id=answer.response_id,
    )


def run_jev(
    examples: Sequence[Example],
    judge: _JevLike,
    store: PredictionStore,
    run: str,
    *,
    repeats: int = 1,
    workers: int = 8,
    state_builder: Callable[[Example], dict[str, str]] | None = None,
) -> list[Prediction]:
    def one(item: tuple[Example, int]) -> None:
        ex, repeat = item
        try:
            if state_builder is not None:
                answer = judge.ask(state_builder(ex), judge.faithfulness_question)
                if judge.question_means_unsupported:
                    answer = _reorient_to_supported(answer)
            else:
                answer = judge.faithfulness_answer(ex.response, ex.source)
        except JudgeError as exc:
            kind = "transport" if isinstance(exc, JudgeTransportError) else "judge"
            store.append(
                Prediction(
                    run=run, example_id=ex.id, repeat=repeat, score=None,
                    error=str(exc), error_kind=kind,
                )
            )
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
                response_id=answer.response_id,
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
        error_kind: str | None = None
        try:
            score: JudgeScore | None = judge.faithfulness(ex.response, ex.source)
            error = None
        except JudgeError as exc:
            score, error = None, str(exc)
            error_kind = "transport" if isinstance(exc, JudgeTransportError) else "judge"
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
                error_kind=error_kind,
                supported_claims=score.supported_claims if score else 0,
                total_claims=score.total_claims if score else 0,
                response_id=judge.last_response_id,
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
        resumed_batch_id = self._unfinished_batch(run)
        if resumed_batch_id is not None:
            self._run_batch_to_completion(run, resumed_batch_id, by_id)
        # Normal path: whatever's still missing — either because there was no batch to
        # resume, or because the resumed batch (plus its one retry) didn't cover
        # everything — gets a fresh batch.
        todo = pending(examples, self.store.load(run), repeats)
        if todo:
            new_batch_id = self._submit(run, todo)
            self._run_batch_to_completion(run, new_batch_id, by_id)
        return collect(self.store, run, examples, repeats)

    # ---------- internals ----------

    def _run_batch_to_completion(
        self, run: str, batch_id: str, by_id: dict[str, Example]
    ) -> None:
        failed = self._finish(run, batch_id, by_id, record_failures=False)
        if failed:
            retry_id = self._submit(run, [(by_id[eid], r) for eid, r in failed])
            self._finish(run, retry_id, by_id, record_failures=True)

    def _log_path(self, run: str) -> Path:
        return self.store.root / "batches" / f"{run}.json"

    def _read_log(self, run: str) -> list[dict[str, Any]]:
        path = self._log_path(run)
        return json.loads(path.read_text()) if path.exists() else []

    def _write_log(self, run: str, entries: list[dict[str, Any]]) -> None:
        path = self._log_path(run)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(entries, indent=2))
        os.replace(tmp, path)

    def _unfinished_batch(self, run: str) -> str | None:
        for entry in self._read_log(run):
            if entry.get("submitting") and entry.get("batch_id") is None:
                raise ClaudeBatchError(
                    f"run {run!r} has an incomplete batch submission logged "
                    f"(n_requests={entry.get('n_requests')}) — the process may have crashed "
                    "between submitting and logging the batch id. Check "
                    "client.messages.batches.list() for a matching batch before rerunning; "
                    "resubmitting blindly risks paying for the same requests twice."
                )
            if entry.get("batch_id") is not None and not entry.get("ingested", False):
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
        entries = self._read_log(run)
        entries.append({"batch_id": None, "submitting": True, "n_requests": len(requests)})
        self._write_log(run, entries)
        batch = self.client.messages.batches.create(requests=requests)
        entries[-1] = {"batch_id": batch.id, "ingested": False}
        self._write_log(run, entries)
        return str(batch.id)

    def _finish(
        self, run: str, batch_id: str, by_id: dict[str, Example], *, record_failures: bool
    ) -> list[tuple[str, int]]:
        while self.client.messages.batches.retrieve(batch_id).processing_status != "ended":
            self.sleep(self.poll_s)
        failed: list[tuple[str, int]] = []
        for item in self.client.messages.batches.results(batch_id):
            example_id, repeat = parse_custom_id(item.custom_id)
            result = item.result
            if result.type == "succeeded":
                self.store.append(self._parse(run, example_id, repeat, result.message))
                continue
            reason = f"batch {result.type}: {getattr(result, 'error', '')}".strip()
            if record_failures or example_id not in by_id:
                self.store.append(
                    Prediction(
                        run=run, example_id=example_id, repeat=repeat, score=None,
                        error=reason, error_kind="transport",
                    )
                )
            else:
                failed.append((example_id, repeat))
        entries = self._read_log(run)
        for entry in entries:
            if entry.get("batch_id") == batch_id:
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
        response_id = str(getattr(message, "id", "") or "")
        try:
            score = parse_judge_message(message, JudgeScore)
        except JudgeError as exc:
            return Prediction(run=run, example_id=example_id, repeat=repeat, score=None,
                              input_tokens=tokens_in, output_tokens=tokens_out, cost_usd=cost,
                              cache_read_tokens=cache_read, cache_write_tokens=cache_write,
                              served_model=self.judge.model, error=str(exc), error_kind="judge",
                              response_id=response_id)
        return Prediction(run=run, example_id=example_id, repeat=repeat, score=score.score,
                          input_tokens=tokens_in, output_tokens=tokens_out, cost_usd=cost,
                          cache_read_tokens=cache_read, cache_write_tokens=cache_write,
                          served_model=self.judge.model, reasoning=score.reasoning,
                          supported_claims=score.supported_claims, total_claims=score.total_claims,
                          response_id=response_id)
