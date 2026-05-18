from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from rag_eval.cli import cli


def test_cli_version() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "rag-eval" in result.output


def test_cli_empty_tests_exits_zero(tmp_path: Path) -> None:
    """Empty tests list runs cleanly."""
    config = tmp_path / "empty.yaml"
    config.write_text(
        "adapter:\n"
        "  type: python\n"
        "  module: examples.demo_rag.adapter\n"
        "  class: DemoAdapter\n"
        "tests: []\n"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli, ["run", str(config), "--out-dir", str(tmp_path / "report")]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "report" / "report.json").exists()
    assert (tmp_path / "report" / "report.md").exists()


def test_cli_full_demo_runs_evaluators(repo_root: Path, tmp_path: Path) -> None:
    """Run the bundled demo config and confirm the evaluators wired through end-to-end.

    Uses --no-judge so the test does not hit the Anthropic API in CI.
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(repo_root / "examples" / "demo_rag" / "config.yaml"),
            "--out-dir",
            str(tmp_path / "report"),
            "--no-judge",
        ],
    )
    # Exit code may be 0 or 1 depending on demo quality — we just want clean wiring.
    assert result.exit_code in (0, 1), result.output

    payload = json.loads((tmp_path / "report" / "report.json").read_text())
    test_names = {t["name"] for t in payload["tests"]}
    assert test_names == {
        "tool_coverage_all",
        "direct_retrieval_basics",
        "hallucination_guardrail",
        "citation_audit_basics",
        "edge_cases_battery",
    }
    by_name = {t["name"]: t for t in payload["tests"]}
    assert by_name["tool_coverage_all"]["verdict"] == "PASS"
    # citation_audit in --no-judge mode runs the dangling check only — should PASS.
    assert by_name["citation_audit_basics"]["verdict"] == "PASS"
    # edge_cases assert per-case: three hard-assertion kinds (long_input, multi_turn,
    # empty_input) should PASS; contradiction is expected to FAIL because the demo's
    # substring-RAG doesn't push back on false premises — that's *exactly* the failure
    # mode the contradiction kind is designed to catch. Demonstrates the evaluator
    # doing its job on the bundled reference adapter.
    ec_cases = {c["kind"]: c for c in by_name["edge_cases_battery"]["artifacts"]["cases"]}
    assert ec_cases["long_input"]["verdict"] == "PASS"
    assert ec_cases["multi_turn"]["verdict"] == "PASS"
    assert ec_cases["empty_input"]["verdict"] == "PASS"
    assert ec_cases["contradiction"]["verdict"] == "FAIL"
