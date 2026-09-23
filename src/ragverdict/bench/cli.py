"""`ragverdict bench ragtruth ...` commands."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from ragverdict.bench import metrics
from ragverdict.bench.openrouter_llm import (
    DEEPSEEK_FLASH,
    DEEPSEEK_FLASH_PILOT,
    GLM_FLASH,
    GLM_FLASH_PILOT,
    ChatJudgeConfig,
    run_chat_judge,
)
from ragverdict.bench.predict import (
    BudgetError,
    ClaudeBatchRunner,
    PredictionStore,
    check_budget,
    pending,
    run_claude_live,
    run_jev,
)
from ragverdict.bench.ragtruth import Example, ensure_downloaded, load_examples
from ragverdict.bench.runs import (
    PARAPHRASES,
    RUNS,
    FrozenConfig,
    RunSpec,
    load_frozen,
    select_examples,
)
from ragverdict.bench.summary import UNTUNED_JEV_THRESHOLD, build_summary
from ragverdict.judges.jev_judge import JevJudge
from ragverdict.judges.llm_judge import LLMJudge

FROZEN_PATH = Path("bench/frozen_config.json")  # run bench commands from the repo root
# Runs that don't pin their own paraphrase — they use the frozen one — so their invocation
# metadata records which jev_paraphrase/frozen_sha256 they actually ran under (for
# `_frozen_mismatch`). Registration itself gates every test-split run (see `_preflight`),
# not just these two.
_FROZEN_GATED_RUNS = ("jev", "jev-flip")
console = Console()


def _data_dir() -> Path:
    return ensure_downloaded()


@click.group(name="bench")
def bench() -> None:
    """Benchmarks for ragverdict judges."""


@bench.group(name="ragtruth")
def ragtruth() -> None:
    """Jev vs Claude vs cascade on RAGTruth (human-labeled hallucinations)."""


def _preflight(specs: list[RunSpec]) -> list[str]:
    """Everything that must be true before any run executes: credentials, a readable and
    registered frozen config (for every test-split run — pre-registration must be locked
    before results can be reported). Budget is checked separately by the caller, after this —
    nothing here needs network or disk access to the dataset."""
    problems: list[str] = []
    if any(spec.judge in ("jev", "chat") for spec in specs) and not os.environ.get(
        "OPENROUTER_API_KEY"
    ):
        problems.append("OPENROUTER_API_KEY is not set (required for jev/chat runs)")
    if any(spec.judge in ("claude_batch", "claude_live") for spec in specs) and not os.environ.get(
        "ANTHROPIC_API_KEY"
    ):
        problems.append("ANTHROPIC_API_KEY is not set (required for claude runs)")

    # Every test-split run is held to pre-registration — not just the runs that read the
    # frozen paraphrase — so results can never be reported from an unregistered config.
    # Train-split runs (tune-*) are exploratory and allowed before registration.
    gated = [spec for spec in specs if spec.split == "test"]
    if gated:
        names = ", ".join(spec.name for spec in gated)
        if not FROZEN_PATH.exists():
            problems.append(f"{FROZEN_PATH} is not readable (required by {names})")
        else:
            try:
                frozen, _sha = load_frozen(FROZEN_PATH)
            except Exception as exc:  # malformed frozen config: fail closed, not a crash
                problems.append(f"could not load {FROZEN_PATH}: {exc}")
            else:
                if not frozen.registered:
                    problems.append(
                        f"frozen config is not registered (registered=false in {FROZEN_PATH}); "
                        f"complete pre-registration before running {names}"
                    )
    return problems


@ragtruth.command()
@click.argument("names", nargs=-1, required=True)
@click.option("--out", type=click.Path(path_type=Path), default=Path("bench_results"), show_default=True)
@click.option("--max-spend-usd", type=float, required=True, help="Abort if the estimate exceeds this.")
def run(names: tuple[str, ...], out: Path, max_spend_usd: float) -> None:
    """Execute named runs (see ragverdict/bench/runs.py), caching predictions under --out."""
    unknown = [n for n in names if n not in RUNS]
    if unknown:
        click.echo(f"error: unknown run(s): {', '.join(unknown)}; known: {', '.join(RUNS)}", err=True)
        sys.exit(2)
    specs = [RUNS[n] for n in names]
    problems = _preflight(specs)
    if problems:
        for problem in problems:
            click.echo(f"error: {problem}", err=True)
        sys.exit(2)
    data_dir = _data_dir()
    store = PredictionStore(out)
    plans = [(spec, select_examples(spec, data_dir)) for spec in specs]
    try:
        estimate = sum(
            check_budget(len(exs) * spec.repeats, spec.cost_per_call, max_spend_usd)
            for spec, exs in plans
        )
        check_budget(1, estimate, max_spend_usd)
    except BudgetError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(2)
    click.echo(f"estimated spend ${estimate:.2f} (cap ${max_spend_usd:.2f})")
    for spec, exs in plans:
        n_total = len(exs) * spec.repeats
        n_computed = len(pending(exs, store.load(spec.name), spec.repeats))
        n_cached = n_total - n_computed
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        preds = _execute(spec, exs, store, data_dir)
        wall_clock = time.perf_counter() - started
        _record_invocation(out, spec, started_at, wall_clock, n_computed, n_cached)
        errors = sum(p.error is not None for p in preds)
        cost = sum(p.cost_usd for p in preds)
        click.echo(
            f"{spec.name}: {len(preds)} predictions, {errors} errors, ${cost:.4f} billed, "
            f"{wall_clock:.1f}s wall clock"
        )


def _load_meta_record(meta_path: Path, spec: RunSpec) -> dict[str, Any]:
    """`{run, invocations}`. Tolerates a legacy flat meta file (`{"run", "n", "wall_clock_s"}`,
    written before the invocations format existed) by converting it into one invocation
    instead of raising."""
    if not meta_path.exists():
        return {"run": spec.name, "invocations": []}
    raw: dict[str, Any] = json.loads(meta_path.read_text())
    if "invocations" in raw:
        return raw
    return {
        "run": raw.get("run", spec.name),
        "invocations": [{
            "started_at": raw.get("started_at", ""),
            "wall_clock_s": raw.get("wall_clock_s", 0.0),
            "n_computed": raw.get("n", 0),
            "n_cached": raw.get("n_cached", 0),
        }],
    }


def _record_invocation(
    out: Path, spec: RunSpec, started_at: str, wall_clock_s: float, n_computed: int, n_cached: int
) -> None:
    """Append one invocation to `<out>/meta/<run>.json` (never overwrite prior invocations —
    build_summary's wall-clock number is a sum over every invocation that computed anything)."""
    meta_path = out / "meta" / f"{spec.name}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = _load_meta_record(meta_path, spec)
    invocation: dict[str, Any] = {
        "started_at": started_at,
        "wall_clock_s": wall_clock_s,
        "n_computed": n_computed,
        "n_cached": n_cached,
    }
    if spec.name in _FROZEN_GATED_RUNS:
        frozen, sha = load_frozen(FROZEN_PATH)
        invocation["jev_paraphrase"] = frozen.jev_paraphrase
        invocation["frozen_sha256"] = sha
    record["invocations"].append(invocation)
    meta_path.write_text(json.dumps(record, indent=2))


def _pilot_state(ex: Example) -> dict[str, str]:
    """The 2026-09-19 pilot's exact state keys — only meaningful for QA examples, which are
    the only ones with question/passages populated. An empty question or passages means this
    was called on a non-QA (or malformed) example, which would silently produce a garbage
    judgment rather than a clear failure, so it's rejected instead."""
    if not ex.question or not ex.passages:
        raise ValueError(
            f"example {ex.id!r} has no question/passages (task={ex.task!r}); the pilot state "
            "is only defined for QA examples"
        )
    return {"question": ex.question, "context": ex.passages, "answer": ex.response}


_CHAT_CONFIGS: dict[tuple[str, str], ChatJudgeConfig] = {
    ("deepseek", "ragverdict"): DEEPSEEK_FLASH,
    ("deepseek", "pilot"): DEEPSEEK_FLASH_PILOT,
    ("glm", "ragverdict"): GLM_FLASH,
    ("glm", "pilot"): GLM_FLASH_PILOT,
}


def _execute(spec: RunSpec, exs: list[Any], store: PredictionStore, data_dir: Path) -> list[Any]:
    if spec.judge == "jev":
        paraphrase = spec.paraphrase or load_frozen(FROZEN_PATH)[0].jev_paraphrase
        question, inverted = PARAPHRASES[paraphrase]
        judge = JevJudge(faithfulness_question=question, question_means_unsupported=inverted)
        state_builder = _pilot_state if spec.jev_state == "pilot" else None
        return run_jev(exs, judge, store, spec.name, repeats=spec.repeats,
                       state_builder=state_builder)
    if spec.judge == "chat":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise click.UsageError("OPENROUTER_API_KEY is not set")
        cfg = _CHAT_CONFIGS[(spec.chat or "", spec.chat_prompt)]
        return run_chat_judge(exs, cfg, store, spec.name, api_key=api_key)
    llm = LLMJudge(model="claude-sonnet-5", thinking=spec.thinking)
    if spec.judge == "claude_live":
        return run_claude_live(exs, llm, store, spec.name)
    return ClaudeBatchRunner(llm, store).run(exs, spec.name, repeats=spec.repeats)


@ragtruth.command()
@click.option("--out", type=click.Path(path_type=Path), default=Path("bench_results"), show_default=True)
def tune(out: Path) -> None:
    """Report train-split results for each tuning run (to fill bench/frozen_config.json)."""
    data_dir = _data_dir()
    store = PredictionStore(out)
    for pid in "ABC":
        spec = RUNS[f"tune-{pid}"]
        exs = select_examples(spec, data_dir)
        preds = {eid: p for (eid, rep), p in store.load(spec.name).items() if p.score is not None}
        scored = [e for e in exs if e.id in preds]
        if not scored:
            click.echo(f"tune-{pid}: no predictions yet")
            continue
        p_sup = [float(preds[e.id].score) for e in scored]  # type: ignore[arg-type]
        labels = [e.hallucinated for e in scored]
        auroc = metrics.auroc([1 - p for p in p_sup], labels)
        threshold, f1 = metrics.best_f1_threshold(p_sup, labels)
        click.echo(f"tune-{pid}: n={len(scored)} AUROC={auroc:.3f} best-F1={f1:.3f} @ threshold {threshold:.3f}")
        for half_width in (0.10, 0.15, 0.20, 0.25, 0.30):
            lo, hi = max(0.0, threshold - half_width), min(1.0, threshold + half_width)
            inside = [i for i, p in enumerate(p_sup) if lo < p < hi]
            if inside:
                acc = sum((p_sup[i] < threshold) == labels[i] for i in inside) / len(inside)
                click.echo(f"    band ({lo:.2f},{hi:.2f}): {len(inside)} in band, Jev accuracy {acc:.2f}")


def _frozen_mismatch(out: Path, frozen: FrozenConfig, sha: str) -> str | None:
    """None if every recorded jev_paraphrase/frozen_sha256 for the frozen-gated runs matches
    the frozen config being summarized against; otherwise a message naming the mismatch."""
    for run_name in _FROZEN_GATED_RUNS:
        meta_path = out / "meta" / f"{run_name}.json"
        if not meta_path.exists():
            continue
        record = json.loads(meta_path.read_text())
        for inv in record.get("invocations", []):
            recorded_paraphrase = inv.get("jev_paraphrase")
            recorded_sha = inv.get("frozen_sha256")
            if recorded_paraphrase is None and recorded_sha is None:
                continue  # invocation predates this recording (or isn't frozen-gated)
            if recorded_paraphrase != frozen.jev_paraphrase or recorded_sha != sha:
                return (
                    f"run {run_name!r} was executed under a different frozen config "
                    f"(recorded jev_paraphrase={recorded_paraphrase!r}, "
                    f"frozen_sha256={recorded_sha!r}; current jev_paraphrase="
                    f"{frozen.jev_paraphrase!r}, frozen_sha256={sha!r}); rerun {run_name!r} "
                    "under the current frozen config, or summarize against the one it ran under"
                )
    return None


@ragtruth.command()
@click.option("--out", type=click.Path(path_type=Path), default=Path("bench_results"), show_default=True)
def summarize(out: Path) -> None:
    """Compute summary.json from cached predictions. Makes no API calls."""
    data_dir = _data_dir()
    frozen, sha = load_frozen(FROZEN_PATH)
    mismatch = _frozen_mismatch(out, frozen, sha)
    if mismatch:
        click.echo(f"error: {mismatch}", err=True)
        sys.exit(2)
    summary = build_summary(
        {"test": load_examples(data_dir, split="test", quality="all")},
        PredictionStore(out), frozen, sha,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _print_table(summary)
    click.echo(f"wrote {out / 'summary.json'}")


def _print_table(summary: dict[str, Any]) -> None:
    table = Table(title=f"RAGTruth test (n={summary['n_test']})")
    for col in ("Judge", "n", "AUROC [95% CI]", "F1 [95% CI]", "$ / 1k", "p50 latency"):
        table.add_column(col)
    for name, block in summary["judges"].items():
        auroc, f1 = block["auroc"], block["f1"]
        latency = block.get("latency_s", {}).get("p50")
        table.add_row(
            name,
            str(block["n"]),
            f"{auroc['value']:.3f} [{auroc['ci95'][0]:.3f}, {auroc['ci95'][1]:.3f}]",
            f"{f1['value']:.3f} [{f1['ci95'][0]:.3f}, {f1['ci95'][1]:.3f}]",
            f"${block['cost_usd_per_1k']:.3f}",
            f"{latency:.2f}s" if latency is not None else "—",
        )
    cascade = summary.get("cascade") or {}
    frozen_band = cascade.get("frozen_band")
    if frozen_band:
        table.add_row("cascade", str(cascade["n"]), "—", f"{frozen_band['f1']:.3f}",
                      f"${frozen_band['cost_usd_per_1k']:.3f}", "—")
    console.print(table)


@ragtruth.command()
@click.option("--out", type=click.Path(path_type=Path), default=Path("bench_results"), show_default=True)
@click.option("--docs", type=click.Path(path_type=Path), default=Path("docs/bench"), show_default=True)
@click.option("--audit", type=click.Path(path_type=Path), default=Path("docs/bench/data/audit.json"))
def report(out: Path, docs: Path, audit: Path) -> None:
    """Render charts and the results-page data from summary.json + cached predictions."""
    from ragverdict.bench.charts import render_all
    from ragverdict.bench.page import export

    summary = json.loads((out / "summary.json").read_text())
    for path in render_all(summary, docs):
        click.echo(f"wrote {path}")
    examples = load_examples(_data_dir(), split="test")
    result = export(examples, PredictionStore(out), summary, UNTUNED_JEV_THRESHOLD,
                    docs / "data", audit)
    click.echo(f"wrote {result}")
