"""RunSpec / select_examples / RUNS wiring for the pilot-replication arm."""

from __future__ import annotations

import json
from pathlib import Path

from ragverdict.bench.runs import RUNS, select_examples

_SOURCES = [
    {"source_id": "s1", "task_type": "QA", "source": "MARCO", "source_info": {}, "prompt": "Q"},
    {"source_id": "s2", "task_type": "Summary", "source": "CNN", "source_info": "x", "prompt": "S"},
]
_RESPONSES = [
    {"id": "1", "source_id": "s1", "model": "g", "temperature": 0.7, "split": "test",
     "quality": "good", "response": "r1", "labels": []},
    {"id": "2", "source_id": "s2", "model": "g", "temperature": 0.7, "split": "test",
     "quality": "good", "response": "r2", "labels": []},
]


def _write_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "response.jsonl").write_text("\n".join(json.dumps(r) for r in _RESPONSES) + "\n")
    (root / "source_info.jsonl").write_text("\n".join(json.dumps(s) for s in _SOURCES) + "\n")
    return root


def test_jev_pilot_state_run_is_registered() -> None:
    spec = RUNS["jev-pilot-state"]
    assert spec.judge == "jev" and spec.paraphrase == "P"
    assert spec.jev_state == "pilot" and spec.task_filter == "QA"
    assert spec.split == "test" and spec.per_task is None


def test_deepseek_pilot_and_glm_pilot_runs_are_registered() -> None:
    ds = RUNS["deepseek-pilot"]
    assert ds.judge == "chat" and ds.chat == "deepseek" and ds.chat_prompt == "pilot"
    assert ds.task_filter == "QA"
    glm = RUNS["glm-pilot"]
    assert glm.judge == "chat" and glm.chat == "glm" and glm.chat_prompt == "pilot"
    assert glm.task_filter == "QA"


def test_existing_runs_default_to_standard_state_and_ragverdict_prompt() -> None:
    assert RUNS["jev"].jev_state == "standard" and RUNS["jev"].task_filter is None
    assert RUNS["deepseek"].chat_prompt == "ragverdict"


def test_select_examples_task_filter_restricts_to_qa(tmp_path: Path) -> None:
    exs = select_examples(RUNS["jev-pilot-state"], _write_dataset(tmp_path))
    assert [e.id for e in exs] == ["1"]
    assert all(e.task == "QA" for e in exs)


def test_select_examples_without_task_filter_keeps_every_task(tmp_path: Path) -> None:
    exs = select_examples(RUNS["jev"], _write_dataset(tmp_path))
    assert {e.id for e in exs} == {"1", "2"}
