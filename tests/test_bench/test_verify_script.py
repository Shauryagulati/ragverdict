"""The independent verifier agrees with build_summary on a synthetic dataset."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ragverdict.bench.predict import Prediction, PredictionStore
from ragverdict.bench.ragtruth import load_examples
from ragverdict.bench.runs import load_frozen
from ragverdict.bench.summary import build_summary

REPO = Path(__file__).resolve().parents[2]


def _dataset(root: Path) -> None:
    sources = [{"source_id": "s", "task_type": "QA", "source": "x", "source_info": {}, "prompt": "p"}]
    rows = []
    for i in range(14):
        labels = [{"start": 0, "end": 1, "text": "2022", "meta": "", "label_type": "Evident Conflict",
                   "implicit_true": False, "due_to_null": False}] if i % 3 == 0 else []
        rows.append({"id": str(i), "source_id": "s", "model": "m", "temperature": 0.7, "split": "test",
                     "quality": "good", "response": f"r{i}", "labels": labels})
    (root / "source_info.jsonl").write_text("\n".join(json.dumps(s) for s in sources) + "\n")
    (root / "response.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _full_store(out: Path) -> PredictionStore:
    """jev (tuned), jev-para-A, claude and deepseek over the 14-example dataset."""
    store = PredictionStore(out)
    for i in range(14):
        hallucinated = i % 3 == 0
        if i == 12:
            jev_score = 0.5  # exactly the frozen jev_threshold: exercises the `<` boundary
        elif i == 13:
            jev_score = 0.0  # 1 - score == 1.0 exactly: exercises the ECE last-bin boundary
        else:
            jev_score = (0.2 + 0.01 * i) if hallucinated else (0.6 + 0.02 * i)
        store.append(Prediction(run="jev", example_id=str(i), repeat=0,
                                score=jev_score, cost_usd=0.00005))
        # untuned headline Jev: a different score profile so its threshold (0.5) matters
        store.append(Prediction(run="jev-para-A", example_id=str(i), repeat=0,
                                score=0.45 if (hallucinated or i in (2, 4)) else 0.8,
                                cost_usd=0.00004))
        claude = (0.5, 1, 2) if (hallucinated or i == 1) else (1.0, 2, 2)
        if i == 3:
            # hallucinated; score 1.0 but the claim counts show an unsupported claim, so the
            # pre-registered LLM rule flags it (the old `score < 1.0` rule would not)
            claude = (1.0, 2, 3)
        store.append(Prediction(run="claude", example_id=str(i), repeat=0, score=claude[0],
                                supported_claims=claude[1], total_claims=claude[2],
                                cost_usd=0.002))
        if i != 5:  # deepseek never scored id 5 -> the headline cohort drops it
            store.append(Prediction(run="deepseek", example_id=str(i), repeat=0,
                                    score=0.5 if i % 2 == 0 else 1.0, supported_claims=1,
                                    total_claims=1 if i % 2 else 2, cost_usd=0.0004))
    return store


# A fixed fixture config (not the repo's pre-registered bench/frozen_config.json), so the
# boundary examples above stay exactly on the Jev threshold (0.5) whatever gets frozen.
FROZEN_PATH = Path(__file__).parent / "fixtures" / "verify_frozen_config.json"


def _summarize(data: Path, store: PredictionStore) -> dict[str, Any]:
    frozen, sha = load_frozen(FROZEN_PATH)
    return build_summary({"test": load_examples(data, split="test")}, store, frozen, sha)


def _verify(data: Path, out: Path, summary: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    (out / "summary.json").write_text(json.dumps(summary))
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "verify_bench.py"), "--out", str(out),
         "--data", str(data), "--frozen", str(FROZEN_PATH)],
        capture_output=True, text=True,
    )


def test_verifier_matches_summary(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _dataset(data)
    out = tmp_path / "out"
    summary = _summarize(data, _full_store(out))
    result = _verify(data, out, summary)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VERIFIED" in result.stdout
    for check in ("claude.f1", "claude.inconsistency", "jev_untuned.f1", "headline.n",
                  "headline.claude.f1", "headline.h1_diff", "headline.h1_verdict",
                  "h9.cascade_f1", "h9.single.jev_tuned", "h9.single.claude", "h9.diff",
                  "h9.verdict"):
        assert check in result.stdout, result.stdout
    assert summary["headline"]["cohort"]["n"] == 13
    assert summary["hypotheses"]["H9"] is not None


@pytest.mark.parametrize("tamper", ["h1_verdict", "h9_cascade_f1", "h9_verdict"])
def test_verifier_flags_tampered_hypotheses(tmp_path: Path, tamper: str) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _dataset(data)
    out = tmp_path / "out"
    summary = _summarize(data, _full_store(out))
    h1, h9 = summary["headline"]["h1"], summary["hypotheses"]["H9"]
    if tamper == "h1_verdict":
        h1["verdict"] = "jev_better" if h1["verdict"] != "jev_better" else "jev_worse"
    elif tamper == "h9_cascade_f1":
        h9["cascade_f1"] += 0.1
    else:
        h9["verdict"] = "confirmed" if h9["verdict"] != "confirmed" else "refuted"
    result = _verify(data, out, summary)
    assert result.returncode == 1 and "MISMATCH" in result.stdout, result.stdout


def test_verifier_flags_a_tampered_summary(tmp_path: Path) -> None:
    """The verifier must fail when summary.json disagrees with the raw predictions."""
    data = tmp_path / "data"
    data.mkdir()
    _dataset(data)
    out = tmp_path / "out"
    store = PredictionStore(out)
    for i in range(14):
        hallucinated = i % 3 == 0
        store.append(Prediction(run="claude", example_id=str(i), repeat=0,
                                score=1.0, supported_claims=1 if hallucinated else 2,
                                total_claims=2, cost_usd=0.002))
    summary = _summarize(data, store)
    assert summary["judges"]["claude"]["f1"]["value"] == 1.0  # claims catch every positive
    summary["judges"]["claude"]["f1"]["value"] = 0.0  # what the old score<1 rule would report
    result = _verify(data, out, summary)
    assert result.returncode == 1 and "MISMATCH" in result.stdout
