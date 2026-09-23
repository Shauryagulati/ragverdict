"""`ragverdict bench ragtruth ...` commands."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from ragverdict.bench import metrics
from ragverdict.bench.openrouter_llm import DEEPSEEK_FLASH, GLM_FLASH, run_chat_judge
from ragverdict.bench.predict import (
    BudgetError,
    ClaudeBatchRunner,
    PredictionStore,
    check_budget,
    run_claude_live,
    run_jev,
)
from ragverdict.bench.ragtruth import ensure_downloaded, load_examples
from ragverdict.bench.runs import PARAPHRASES, RUNS, RunSpec, load_frozen, select_examples
from ragverdict.bench.summary import build_summary
from ragverdict.judges.jev_judge import JevJudge
from ragverdict.judges.llm_judge import LLMJudge

FROZEN_PATH = Path("bench/frozen_config.json")  # run bench commands from the repo root
console = Console()


def _data_dir() -> Path:
    return ensure_downloaded()


@click.group(name="bench")
def bench() -> None:
    """Benchmarks for ragverdict judges."""


@bench.group(name="ragtruth")
def ragtruth() -> None:
    """Jev vs Claude vs cascade on RAGTruth (human-labeled hallucinations)."""


@ragtruth.command()
@click.argument("names", nargs=-1, required=True)
@click.option("--out", type=click.Path(path_type=Path), default=Path("bench_results"), show_default=True)
@click.option("--max-spend-usd", type=float, required=True, help="Abort if the estimate exceeds this.")
def run(names: tuple[str, ...], out: Path, max_spend_usd: float) -> None:
    """Execute named runs (see ragverdict/bench/runs.py), caching predictions under --out."""
    unknown = [n for n in names if n not in RUNS]
    if unknown:
        click.echo(f"error: unknown run(s): {', '.join(unknown)}; known: {', '.join(RUNS)}")
        sys.exit(2)
    data_dir = _data_dir()
    store = PredictionStore(out)
    plans = [(RUNS[n], select_examples(RUNS[n], data_dir)) for n in names]
    try:
        estimate = sum(
            check_budget(len(exs) * spec.repeats, spec.cost_per_call, max_spend_usd)
            for spec, exs in plans
        )
        check_budget(1, estimate, max_spend_usd)
    except BudgetError as exc:
        click.echo(f"error: {exc}")
        sys.exit(2)
    click.echo(f"estimated spend ${estimate:.2f} (cap ${max_spend_usd:.2f})")
    for spec, exs in plans:
        started = time.perf_counter()
        preds = _execute(spec, exs, store, data_dir)
        wall_clock = time.perf_counter() - started
        meta = out / "meta" / f"{spec.name}.json"
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps({"run": spec.name, "n": len(preds), "wall_clock_s": wall_clock}))
        errors = sum(p.error is not None for p in preds)
        cost = sum(p.cost_usd for p in preds)
        click.echo(
            f"{spec.name}: {len(preds)} predictions, {errors} errors, ${cost:.4f} billed, "
            f"{wall_clock:.1f}s wall clock"
        )


def _execute(spec: RunSpec, exs: list[Any], store: PredictionStore, data_dir: Path) -> list[Any]:
    if spec.judge == "jev":
        paraphrase = spec.paraphrase or load_frozen(FROZEN_PATH)[0].jev_paraphrase
        question, inverted = PARAPHRASES[paraphrase]
        judge = JevJudge(faithfulness_question=question, question_means_unsupported=inverted)
        return run_jev(exs, judge, store, spec.name, repeats=spec.repeats)
    if spec.judge == "chat":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise click.UsageError("OPENROUTER_API_KEY is not set")
        cfg = {"deepseek": DEEPSEEK_FLASH, "glm": GLM_FLASH}[spec.chat or ""]
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


@ragtruth.command()
@click.option("--out", type=click.Path(path_type=Path), default=Path("bench_results"), show_default=True)
def summarize(out: Path) -> None:
    """Compute summary.json from cached predictions. Makes no API calls."""
    data_dir = _data_dir()
    frozen, sha = load_frozen(FROZEN_PATH)
    summary = build_summary(
        {"test": load_examples(data_dir, split="test")},
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
    cascade = summary.get("cascade", {}).get("frozen_band")
    if cascade:
        table.add_row("cascade", str(summary["cascade"]["n"]), "—", f"{cascade['f1']:.3f}",
                      f"${cascade['cost_usd_per_1k']:.3f}", "—")
    console.print(table)
