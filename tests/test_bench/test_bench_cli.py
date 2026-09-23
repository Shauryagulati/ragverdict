"""`ragverdict bench ragtruth` wiring, with runners monkeypatched — no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from ragverdict.bench import cli as bench_cli
from ragverdict.bench.predict import Prediction, PredictionStore
from ragverdict.bench.ragtruth import Example
from ragverdict.bench.runs import RUNS, FrozenConfig
from ragverdict.cli import cli


def _write_registered_frozen(path: Path, *, registered: bool = True) -> None:
    path.write_text(json.dumps({
        "jev_model": "typesafe/jev-1.13", "jev_paraphrase": "A", "jev_threshold": 0.5,
        "claude_model": "claude-sonnet-5", "claude_rule": "score<1.0",
        "cascade_band": [0.3, 0.7], "cascade_band_sweep": [[0.4, 0.6]],
        "dataset_commit": "abc", "bootstrap_resamples": 200, "bootstrap_seed": 0,
        "registered": registered,
    }))


def test_bench_group_registered() -> None:
    result = CliRunner().invoke(cli, ["bench", "ragtruth", "--help"])
    assert result.exit_code == 0
    assert "run" in result.output and "summarize" in result.output and "tune" in result.output


def test_run_refuses_over_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")  # satisfy the preflight
    frozen_path = tmp_path / "frozen.json"
    _write_registered_frozen(frozen_path)
    monkeypatch.setattr(bench_cli, "FROZEN_PATH", frozen_path)  # a tmp placeholder, not the repo file
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
    frozen_path = tmp_path / "frozen.json"
    _write_registered_frozen(frozen_path)
    monkeypatch.setattr(bench_cli, "FROZEN_PATH", frozen_path)  # tmp placeholder, not the repo file
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
    frozen_path = tmp_path / "frozen.json"
    _write_registered_frozen(frozen_path)
    monkeypatch.setattr(bench_cli, "FROZEN_PATH", frozen_path)  # tmp placeholder, not the repo file
    result = CliRunner().invoke(
        cli, ["bench", "ragtruth", "run", "claude", "--out", str(tmp_path), "--max-spend-usd", "100"]
    )
    assert result.exit_code == 2
    assert "ANTHROPIC_API_KEY" in result.output


def test_run_refuses_jev_when_frozen_config_not_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every test-split run — not just `jev`/`jev-flip` — refuses until pre-registration is
    done. Uses a tmp placeholder rather than the real bench/frozen_config.json so this test
    doesn't depend on the repo file's registered flag."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    frozen_path = tmp_path / "frozen.json"
    _write_registered_frozen(frozen_path, registered=False)
    monkeypatch.setattr(bench_cli, "FROZEN_PATH", frozen_path)
    result = CliRunner().invoke(
        cli, ["bench", "ragtruth", "run", "jev", "--out", str(tmp_path), "--max-spend-usd", "100"]
    )
    assert result.exit_code == 2
    assert "registered" in result.output


def test_run_refuses_non_jev_test_split_run_when_not_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The registration gate covers every test-split run, not only the ones that read the
    frozen paraphrase — e.g. `deepseek` (a fixed-paraphrase-free chat run) is gated too."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    frozen_path = tmp_path / "frozen.json"
    _write_registered_frozen(frozen_path, registered=False)
    monkeypatch.setattr(bench_cli, "FROZEN_PATH", frozen_path)
    result = CliRunner().invoke(
        cli, ["bench", "ragtruth", "run", "deepseek", "--out", str(tmp_path), "--max-spend-usd", "100"]
    )
    assert result.exit_code == 2
    assert "registered" in result.output


def test_run_allows_train_split_tune_run_when_not_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Train-split (tune-*) runs are exploratory and allowed before pre-registration."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    frozen_path = tmp_path / "frozen.json"
    _write_registered_frozen(frozen_path, registered=False)
    monkeypatch.setattr(bench_cli, "FROZEN_PATH", frozen_path)
    monkeypatch.setattr(bench_cli, "_data_dir", lambda: tmp_path)
    monkeypatch.setattr(bench_cli, "select_examples", lambda spec, d: [])
    monkeypatch.setattr(bench_cli, "_execute", lambda *a, **kw: [])
    result = CliRunner().invoke(
        cli, ["bench", "ragtruth", "run", "tune-A", "--out", str(tmp_path), "--max-spend-usd", "100"]
    )
    assert result.exit_code == 0, result.output
    assert "registered" not in result.output


def test_run_records_invocation_with_computed_and_cached_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end (network mocked out): `run jev` writes a meta invocation with
    n_computed/n_cached and, since jev uses the frozen paraphrase, the paraphrase id and the
    frozen config's sha256 too."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    frozen_path = tmp_path / "frozen.json"
    _write_registered_frozen(frozen_path)
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


def test_record_invocation_tolerates_legacy_flat_meta_file(tmp_path: Path) -> None:
    """A meta file written before the invocations format existed (`{"run","n","wall_clock_s"}`)
    is converted into one invocation instead of raising a KeyError."""
    meta_path = tmp_path / "meta" / "claude.json"
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(json.dumps({"run": "claude", "n": 5, "wall_clock_s": 12.5}))
    bench_cli._record_invocation(tmp_path, RUNS["claude"], "2026-09-22T00:00:00Z", 3.0, 2, 1)
    record = json.loads(meta_path.read_text())
    assert len(record["invocations"]) == 2
    legacy = record["invocations"][0]
    assert legacy["n_computed"] == 5 and legacy["wall_clock_s"] == 12.5
    fresh = record["invocations"][1]
    assert fresh["n_computed"] == 2 and fresh["n_cached"] == 1


# ---------- _execute dispatch (pilot-replication arm) ----------

def test_pilot_state_rejects_empty_question() -> None:
    ex = Example(id="1", split="test", task="QA", generator="g", source="s", response="resp",
                hallucinated=False, span_types=(), span_texts=(), numeric=False,
                question="", passages="ctx")
    with pytest.raises(ValueError, match="question/passages"):
        bench_cli._pilot_state(ex)


def test_pilot_state_rejects_empty_passages() -> None:
    ex = Example(id="1", split="test", task="QA", generator="g", source="s", response="resp",
                hallucinated=False, span_types=(), span_texts=(), numeric=False,
                question="q?", passages="")
    with pytest.raises(ValueError, match="question/passages"):
        bench_cli._pilot_state(ex)


def test_execute_jev_pilot_state_passes_state_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    captured: dict[str, object] = {}

    def fake_run_jev(exs: object, judge: object, store: object, name: str, *,
                     repeats: int = 1, state_builder: object = None) -> list[object]:
        captured["state_builder"] = state_builder
        return []

    monkeypatch.setattr(bench_cli, "run_jev", fake_run_jev)
    ex = Example(id="1", split="test", task="QA", generator="g", source="s", response="resp",
                hallucinated=False, span_types=(), span_texts=(), numeric=False,
                question="q?", passages="ctx")
    bench_cli._execute(RUNS["jev-pilot-state"], [ex], PredictionStore(tmp_path), tmp_path)
    builder = captured["state_builder"]
    assert builder is not None
    assert builder(ex) == {"question": "q?", "context": "ctx", "answer": "resp"}  # type: ignore[operator]


def test_execute_jev_standard_run_passes_no_state_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    frozen_path = tmp_path / "frozen.json"
    _write_registered_frozen(frozen_path)
    monkeypatch.setattr(bench_cli, "FROZEN_PATH", frozen_path)
    captured: dict[str, object] = {}

    def fake_run_jev(exs: object, judge: object, store: object, name: str, *,
                     repeats: int = 1, state_builder: object = None) -> list[object]:
        captured["state_builder"] = state_builder
        return []

    monkeypatch.setattr(bench_cli, "run_jev", fake_run_jev)
    bench_cli._execute(RUNS["jev"], [], PredictionStore(tmp_path), tmp_path)
    assert captured["state_builder"] is None


def test_execute_chat_pilot_prompt_selects_pilot_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    captured: dict[str, object] = {}

    def fake_run_chat_judge(exs: object, cfg: object, store: object, name: str, *,
                            api_key: str) -> list[object]:
        captured["cfg"] = cfg
        return []

    monkeypatch.setattr(bench_cli, "run_chat_judge", fake_run_chat_judge)
    bench_cli._execute(RUNS["deepseek-pilot"], [], PredictionStore(tmp_path), tmp_path)
    cfg = captured["cfg"]
    assert cfg.prompt == "pilot"  # type: ignore[attr-defined]
    assert cfg.model == "deepseek/deepseek-v4.1-flash"  # type: ignore[attr-defined]


def test_execute_chat_ragverdict_prompt_selects_default_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    captured: dict[str, object] = {}

    def fake_run_chat_judge(exs: object, cfg: object, store: object, name: str, *,
                            api_key: str) -> list[object]:
        captured["cfg"] = cfg
        return []

    monkeypatch.setattr(bench_cli, "run_chat_judge", fake_run_chat_judge)
    bench_cli._execute(RUNS["deepseek"], [], PredictionStore(tmp_path), tmp_path)
    assert captured["cfg"].prompt == "ragverdict"  # type: ignore[attr-defined]


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
