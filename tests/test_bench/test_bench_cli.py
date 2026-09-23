"""`ragverdict bench ragtruth` wiring, with runners monkeypatched — no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from ragverdict.bench import cli as bench_cli
from ragverdict.cli import cli


def test_bench_group_registered() -> None:
    result = CliRunner().invoke(cli, ["bench", "ragtruth", "--help"])
    assert result.exit_code == 0
    assert "run" in result.output and "summarize" in result.output and "tune" in result.output


def test_run_refuses_over_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench_cli, "_data_dir", lambda: tmp_path)
    monkeypatch.setattr(bench_cli, "select_examples", lambda spec, d: [object()] * 1000)
    result = CliRunner().invoke(
        cli, ["bench", "ragtruth", "run", "claude", "--out", str(tmp_path), "--max-spend-usd", "0.10"]
    )
    assert result.exit_code == 2
    assert "exceeds max spend" in result.output


def test_unknown_run_name(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli, ["bench", "ragtruth", "run", "nope", "--out", str(tmp_path), "--max-spend-usd", "1"]
    )
    assert result.exit_code == 2
    assert "unknown run" in result.output


def test_summarize_writes_summary_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench_cli, "_data_dir", lambda: tmp_path)
    monkeypatch.setattr(bench_cli, "load_examples", lambda d, split: [])
    monkeypatch.setattr(bench_cli, "load_frozen", lambda p: (object(), "sha"))
    monkeypatch.setattr(bench_cli, "build_summary", lambda *a: {"n_test": 0, "judges": {}})
    monkeypatch.setattr(bench_cli, "_print_table", lambda summary: None)
    result = CliRunner().invoke(cli, ["bench", "ragtruth", "summarize", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / "summary.json").read_text())["n_test"] == 0
