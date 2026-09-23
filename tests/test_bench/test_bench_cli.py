"""`ragverdict bench ragtruth` wiring, with runners monkeypatched — no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from ragverdict.bench import cli as bench_cli
from ragverdict.bench.predict import Prediction
from ragverdict.bench.ragtruth import Example
from ragverdict.bench.runs import FrozenConfig
from ragverdict.cli import cli


def test_bench_group_registered() -> None:
    result = CliRunner().invoke(cli, ["bench", "ragtruth", "--help"])
    assert result.exit_code == 0
    assert "run" in result.output and "summarize" in result.output and "tune" in result.output


def test_run_refuses_over_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")  # satisfy the preflight
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


def test_run_missing_openrouter_key_executes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run claude deepseek` needs OPENROUTER_API_KEY for deepseek; without it, nothing runs —
    not even the claude leg, which would otherwise have started spending money."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    calls: list[object] = []
    monkeypatch.setattr(bench_cli, "_execute", lambda *a, **kw: calls.append(a) or [])
    result = CliRunner().invoke(
        cli,
        ["bench", "ragtruth", "run", "claude", "deepseek", "--out", str(tmp_path), "--max-spend-usd", "100"],
    )
    assert result.exit_code == 2
    assert "OPENROUTER_API_KEY" in result.output
    assert calls == []


def test_run_missing_anthropic_key_for_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = CliRunner().invoke(
        cli, ["bench", "ragtruth", "run", "claude", "--out", str(tmp_path), "--max-spend-usd", "100"]
    )
    assert result.exit_code == 2
    assert "ANTHROPIC_API_KEY" in result.output


def test_run_refuses_jev_when_frozen_config_not_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`jev` uses the frozen paraphrase, so it refuses to run until pre-registration is done —
    the real bench/frozen_config.json placeholder ships with registered=false."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    result = CliRunner().invoke(
        cli, ["bench", "ragtruth", "run", "jev", "--out", str(tmp_path), "--max-spend-usd", "100"]
    )
    assert result.exit_code == 2
    assert "registered" in result.output


def test_run_records_invocation_with_computed_and_cached_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end (network mocked out): `run jev` writes a meta invocation with
    n_computed/n_cached and, since jev uses the frozen paraphrase, the paraphrase id and the
    frozen config's sha256 too."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(json.dumps({
        "jev_model": "typesafe/jev-1.13", "jev_paraphrase": "A", "jev_threshold": 0.5,
        "claude_model": "claude-sonnet-5", "claude_rule": "score<1.0",
        "cascade_band": [0.3, 0.7], "cascade_band_sweep": [[0.4, 0.6]],
        "dataset_commit": "abc", "bootstrap_resamples": 200, "bootstrap_seed": 0,
        "registered": True,
    }))
    monkeypatch.setattr(bench_cli, "FROZEN_PATH", frozen_path)
    monkeypatch.setattr(bench_cli, "_data_dir", lambda: tmp_path)
    exs = [
        Example(id=str(i), split="test", task="QA", generator="g", source="s", response="r",
                hallucinated=False, span_types=(), span_texts=(), numeric=False)
        for i in range(3)
    ]
    monkeypatch.setattr(bench_cli, "select_examples", lambda spec, d: exs)

    def fake_execute(spec: object, exs_: list[Example], store: object, data_dir: object) -> list[Prediction]:
        for e in exs_:
            store.append(  # type: ignore[attr-defined]
                Prediction(run=spec.name, example_id=e.id, repeat=0, score=0.9, cost_usd=0.00005)  # type: ignore[attr-defined]
            )
        return [store.load(spec.name)[(e.id, 0)] for e in exs_]  # type: ignore[attr-defined]

    monkeypatch.setattr(bench_cli, "_execute", fake_execute)
    out = tmp_path / "out"
    result = CliRunner().invoke(
        cli, ["bench", "ragtruth", "run", "jev", "--out", str(out), "--max-spend-usd", "1"]
    )
    assert result.exit_code == 0, result.output
    meta = json.loads((out / "meta" / "jev.json").read_text())
    inv = meta["invocations"][0]
    assert inv["n_computed"] == 3 and inv["n_cached"] == 0
    assert inv["jev_paraphrase"] == "A" and inv["frozen_sha256"]

    # A second invocation over the same (now fully cached) examples must record n_computed=0.
    result2 = CliRunner().invoke(
        cli, ["bench", "ragtruth", "run", "jev", "--out", str(out), "--max-spend-usd", "1"]
    )
    assert result2.exit_code == 0, result2.output
    meta2 = json.loads((out / "meta" / "jev.json").read_text())
    assert len(meta2["invocations"]) == 2
    assert meta2["invocations"][1]["n_computed"] == 0 and meta2["invocations"][1]["n_cached"] == 3


def test_summarize_writes_summary_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench_cli, "_data_dir", lambda: tmp_path)
    monkeypatch.setattr(bench_cli, "load_examples", lambda d, split, quality="good": [])
    monkeypatch.setattr(bench_cli, "load_frozen", lambda p: (object(), "sha"))
    monkeypatch.setattr(bench_cli, "build_summary", lambda *a: {"n_test": 0, "judges": {}})
    monkeypatch.setattr(bench_cli, "_print_table", lambda summary: None)
    result = CliRunner().invoke(cli, ["bench", "ragtruth", "summarize", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / "summary.json").read_text())["n_test"] == 0


def test_summarize_refuses_on_frozen_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If `jev` was executed under a different frozen paraphrase/config than the one
    `summarize` is about to read, refuse rather than silently mixing results."""
    monkeypatch.setattr(bench_cli, "_data_dir", lambda: tmp_path)
    monkeypatch.setattr(bench_cli, "load_examples", lambda d, split, quality="good": [])
    current = FrozenConfig(
        jev_model="typesafe/jev-1.13", jev_paraphrase="A", jev_threshold=0.5,
        claude_model="claude-sonnet-5", claude_rule="score<1.0", cascade_band=(0.3, 0.7),
        cascade_band_sweep=[(0.4, 0.6)], dataset_commit="abc",
        bootstrap_resamples=200, bootstrap_seed=0, registered=True,
    )
    monkeypatch.setattr(bench_cli, "load_frozen", lambda p: (current, "current-sha"))
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "jev.json").write_text(json.dumps({
        "run": "jev",
        "invocations": [
            {"started_at": "t", "wall_clock_s": 1.0, "n_computed": 1, "n_cached": 0,
             "jev_paraphrase": "B", "frozen_sha256": "stale-sha"},
        ],
    }))
    result = CliRunner().invoke(cli, ["bench", "ragtruth", "summarize", "--out", str(tmp_path)])
    assert result.exit_code == 2
    assert "different frozen config" in result.output
    assert not (tmp_path / "summary.json").exists()
