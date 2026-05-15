"""Reporter — Rich Live table during the run, JSON + Markdown at the end."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.table import Table

from rag_eval.config import Config
from rag_eval.evaluators.base import TestResult, Verdict

_VERDICT_STYLE: dict[Verdict, str] = {
    Verdict.PASS: "bold green",
    Verdict.WEAK: "bold yellow",
    Verdict.FAIL: "bold red",
    Verdict.ERROR: "bold magenta",
}


class Reporter:
    def __init__(self, *, out_dir: Path, config: Config) -> None:
        self.out_dir = out_dir
        self.config = config
        self.console = Console()
        self._test_count = 0
        self._table: Table | None = None
        self._live: Live | None = None

    def start(self, *, test_count: int) -> None:
        self._test_count = test_count
        self.console.rule("[bold]rag-eval[/bold]")
        if test_count == 0:
            self.console.print("[yellow]No tests defined in config — nothing to run.[/yellow]")
            return
        self.console.print(f"Running [bold]{test_count}[/bold] test(s)…\n")

        self._table = _build_results_table()
        self._live = Live(
            self._table,
            console=self.console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.__enter__()

    def on_result(self, result: TestResult) -> None:
        style = _VERDICT_STYLE.get(result.verdict, "white")
        verdict = f"[{style}]{result.verdict.value}[/{style}]"
        detail = result.detail if len(result.detail) < 100 else result.detail[:97] + "…"
        if self._table is not None:
            self._table.add_row(
                result.name,
                result.evaluator,
                verdict,
                f"{result.duration_ms}ms",
                detail,
            )
        else:
            # No-table fallback (test_count == 0 path).
            self.console.print(
                f"  {verdict}  [bold]{result.name}[/bold] ({result.evaluator}) {detail}"
            )

    def finalize(self, results: list[TestResult]) -> tuple[Table, int]:
        if self._live is not None:
            self._live.__exit__(None, None, None)
            self._live = None

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(results)
        self._write_markdown(results)

        summary = Table(title="Summary", title_style="bold")
        summary.add_column("Verdict")
        summary.add_column("Count", justify="right")
        counts: dict[Verdict, int] = {v: 0 for v in Verdict}
        for r in results:
            counts[r.verdict] += 1
        for verdict in Verdict:
            style = _VERDICT_STYLE[verdict]
            summary.add_row(f"[{style}]{verdict.value}[/{style}]", str(counts[verdict]))
        self.console.print()
        self.console.print(summary)
        self.console.print(f"\nReports written to [bold]{self.out_dir}/[/bold]")

        exit_code = self._exit_code(counts)
        return summary, exit_code

    def _exit_code(self, counts: dict[Verdict, int]) -> int:
        if self._test_count == 0:
            return 0
        # All tests errored — typically infrastructure failure (judge unreachable).
        if counts[Verdict.ERROR] == self._test_count:
            return 3
        if counts[Verdict.FAIL] > 0 or counts[Verdict.ERROR] > 0:
            return 1
        return 0

    def _write_json(self, results: list[TestResult]) -> None:
        payload: dict[str, Any] = {
            "tests": [_result_to_dict(r) for r in results],
            "summary": {v.value: sum(1 for r in results if r.verdict == v) for v in Verdict},
        }
        (self.out_dir / "report.json").write_text(json.dumps(payload, indent=2, default=str))

    def _write_markdown(self, results: list[TestResult]) -> None:
        lines: list[str] = ["# rag-eval report", ""]
        if not results:
            lines.append("_No tests ran._")
        else:
            lines.append("| Test | Evaluator | Verdict | Latency | Detail |")
            lines.append("| --- | --- | --- | ---: | --- |")
            for r in results:
                detail = r.detail.replace("|", "\\|").replace("\n", " ")
                lines.append(
                    f"| {r.name} | {r.evaluator} | **{r.verdict.value}** | "
                    f"{r.duration_ms}ms | {detail} |"
                )
        (self.out_dir / "report.md").write_text("\n".join(lines) + "\n")


def _build_results_table() -> Table:
    table = Table(show_lines=False)
    table.add_column("Test", style="bold")
    table.add_column("Evaluator", style="dim")
    table.add_column("Verdict")
    table.add_column("Latency", justify="right")
    table.add_column("Detail", overflow="fold")
    return table


def _result_to_dict(r: TestResult) -> dict[str, Any]:
    d = dataclasses.asdict(r)
    d["verdict"] = r.verdict.value
    return d
