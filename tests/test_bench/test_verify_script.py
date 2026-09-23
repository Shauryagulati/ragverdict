"""The independent verifier agrees with build_summary on a synthetic dataset."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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


def test_verifier_matches_summary(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _dataset(data)
    out = tmp_path / "out"
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
        store.append(Prediction(run="claude", example_id=str(i), repeat=0,
                                score=0.5 if (hallucinated or i == 1) else 1.0, cost_usd=0.002))
    frozen_path = REPO / "bench" / "frozen_config.json"
    frozen, sha = load_frozen(frozen_path)
    summary = build_summary({"test": load_examples(data, split="test")}, store, frozen, sha)
    (out / "summary.json").write_text(json.dumps(summary))
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "verify_bench.py"), "--out", str(out),
         "--data", str(data), "--frozen", str(frozen_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VERIFIED" in result.stdout
