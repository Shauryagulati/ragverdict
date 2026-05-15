from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from rag_eval.cli import cli


def test_cli_version() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "rag-eval" in result.output


def test_cli_empty_tests_exits_zero(repo_root: Path, tmp_path: Path) -> None:
    """Day 1 verification: empty tests list runs cleanly."""
    runner = CliRunner()
    # CliRunner doesn't change cwd by default — invoke with --out-dir into tmp_path.
    result = runner.invoke(
        cli,
        [
            "run",
            str(repo_root / "examples" / "demo_rag" / "config.yaml"),
            "--out-dir",
            str(tmp_path / "report"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "report" / "report.json").exists()
    assert (tmp_path / "report" / "report.md").exists()
