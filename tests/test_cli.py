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
    # Demo is all-green by design: the bundled DemoAdapter is a well-behaved
    # reference agent (it pushes back on false premises), so the run exits 0.
    assert result.exit_code == 0, result.output

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
    # All four edge-case kinds PASS on the bundled reference adapter. contradiction
    # passes because DemoAdapter's groundedness guard makes it push back on the false
    # premise ("Acme's acquisition of XYZ Corp") instead of confabulating — even under
    # the --no-judge heuristic. The tool's ability to *catch* an agent that does NOT
    # push back is proven in test_regression_smoke.py against CompliantAdapter.
    ec_cases = {c["kind"]: c for c in by_name["edge_cases_battery"]["artifacts"]["cases"]}
    assert ec_cases["long_input"]["verdict"] == "PASS"
    assert ec_cases["multi_turn"]["verdict"] == "PASS"
    assert ec_cases["empty_input"]["verdict"] == "PASS"
    assert ec_cases["contradiction"]["verdict"] == "PASS"
