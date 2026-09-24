"""Pure helpers of scripts/bench_extras.py against hand-computed expectations."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ragverdict.bench.predict import Prediction

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "bench_extras.py"
_spec = importlib.util.spec_from_file_location("bench_extras", _PATH)
assert _spec is not None and _spec.loader is not None
bx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bx)

T, F = True, False


def test_flag_thresholds_groups_ties_and_starts_at_flag_nothing() -> None:
    ops = bx.flag_thresholds([0.2, 0.8, 0.2, 0.5])
    assert [n for _, n in ops] == [0, 2, 3, 4]
    assert [t for t, _ in ops][:3] == pytest.approx([0.2, 0.35, 0.65])


def test_oracle_f1_picks_best_threshold() -> None:
    # flag 1: F1 2/3; flag 2: 1/2; flag 3: 4/5; flag 4: 2/3
    best = bx.oracle_f1([0.1, 0.2, 0.3, 0.9], [T, F, T, F])
    assert best["f1"] == pytest.approx(0.8)
    assert best["threshold"] == pytest.approx(0.6)
    assert best["n_flagged"] == 3


def test_oracle_f1_flags_ties_together() -> None:
    # the two 0.5s can only be flagged together: F1 = 2*1/(2+1)
    best = bx.oracle_f1([0.5, 0.5, 0.9], [T, F, F])
    assert best["f1"] == pytest.approx(2 / 3)
    assert best["n_flagged"] == 2


def test_matched_flag_threshold_nearest_count_ties_to_smaller() -> None:
    p = [0.1, 0.2, 0.2, 0.4]  # achievable counts: 0, 1, 3, 4
    assert bx.matched_flag_threshold(p, 2) == (pytest.approx(0.15), 1)
    assert bx.matched_flag_threshold(p, 0) == (0.1, 0)
    assert bx.matched_flag_threshold(p, 4)[1] == 4


def test_jitter_stats() -> None:
    stats = bx.jitter_stats([[0.1, 0.1, 0.1], [0.2, 0.5, 0.3], [0.0, 0.1, 0.05], [0.4, 0.4, 0.6]])
    # ranges: 0, 0.3, 0.1, 0.2
    assert stats["n"] == 4
    assert stats["n_changed"] == 3
    assert stats["median"] == pytest.approx(0.15)
    assert stats["p90"] == pytest.approx(0.27)  # sorted[2] + 0.7 * (sorted[3] - sorted[2])
    assert stats["max"] == pytest.approx(0.3)


def test_base_rate_precision() -> None:
    # 0.8*0.05 / (0.8*0.05 + 0.1*0.95) = 0.04 / 0.135
    assert bx.base_rate_precision(0.8, 0.1, 0.05) == pytest.approx(0.04 / 0.135)
    assert bx.base_rate_precision(0.0, 0.0, 0.1) is None


def test_disagreement_and_agreement() -> None:
    a, b, y = [T, F, T, F], [F, T, T, F], [T, F, F, F]
    d = bx.disagreement(a, b, y)
    assert (d["n"], d["a_right"], d["b_right"]) == (2, 2, 0)
    assert d["b_flags_a_passes"] == {"n": 1, "n_human_clean": 1}
    assert d["a_flags_b_passes"] == {"n": 1, "n_human_clean": 0}
    g = bx.agreement(a, b, y)
    assert (g["n_agree"], g["accuracy_when_agree"]) == (2, 0.5)
    assert (g["n_both_flag"], g["precision_when_both_flag"]) == (1, 0.0)
    assert (g["n_both_pass"], g["clean_rate_when_both_pass"]) == (1, 1.0)


def test_recall_by_skips_negatives_and_none_keys() -> None:
    out = bx.recall_by(["x", "x", "y", None], {"j": [T, F, T, T]}, [T, T, T, F])
    assert out == {"x": {"n_pos": 2, "j": 0.5}, "y": {"n_pos": 1, "j": 1.0}}


def test_claude_standard_uncached_prices_cache_reads_as_input() -> None:
    p = Prediction(run="claude", example_id="1", repeat=0, score=1.0, input_tokens=100,
                   cache_read_tokens=900, output_tokens=50)
    # (1000 * 2 + 50 * 10) / 1e6
    assert bx.claude_standard_uncached_usd(p, 2.0, 10.0) == pytest.approx(0.0025)
