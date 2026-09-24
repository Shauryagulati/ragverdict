"""RAGTruth loader — fixtures are tiny hand-written JSONL files; no network."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from ragverdict.bench.ragtruth import (
    DEFAULT_CACHE,
    DatasetError,
    Example,
    ensure_downloaded,
    is_numeric_span,
    load_examples,
    stratified_sample,
)

SOURCES = [
    {"source_id": "s1", "task_type": "QA", "source": "MARCO", "source_info": {}, "prompt": "Q prompt"},
    {"source_id": "s2", "task_type": "Summary", "source": "CNN", "source_info": "x", "prompt": "S prompt"},
]


def _span(
    label_type: str, text: str, implicit_true: bool = False, due_to_null: bool = False
) -> dict[str, object]:
    return {"start": 0, "end": 1, "text": text, "meta": "", "label_type": label_type,
            "implicit_true": implicit_true, "due_to_null": due_to_null}


RESPONSES = [
    {"id": "1", "source_id": "s1", "model": "gpt-4-0613", "temperature": 0.7, "split": "test",
     "quality": "good", "response": "clean answer", "labels": []},
    {"id": "2", "source_id": "s1", "model": "llama-2-7b-chat", "temperature": 0.7, "split": "test",
     "quality": "good", "response": "wrong year", "labels": [_span("Evident Conflict", "in 2022")]},
    {"id": "3", "source_id": "s2", "model": "mistral-7B-instruct", "temperature": 0.7, "split": "test",
     "quality": "good", "response": "implicit only",
     "labels": [_span("Subtle Baseless Info", "true but unstated", implicit_true=True)]},
    {"id": "4", "source_id": "s2", "model": "gpt-3.5-turbo-0613", "temperature": 0.7, "split": "test",
     "quality": "incorrect_refusal", "response": "refused", "labels": []},
    {"id": "5", "source_id": "s2", "model": "gpt-4-0613", "temperature": 0.7, "split": "train",
     "quality": "good", "response": "subtle add", "labels": [_span("Subtle Baseless Info", "a new fact")]},
]


def _write_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "response.jsonl").write_text("\n".join(json.dumps(r) for r in RESPONSES) + "\n")
    (root / "source_info.jsonl").write_text("\n".join(json.dumps(s) for s in SOURCES) + "\n")
    return root


def test_load_filters_quality_and_split(tmp_path: Path) -> None:
    examples = load_examples(_write_dataset(tmp_path), split="test")
    assert [e.id for e in examples] == ["1", "2", "3"]


def test_quality_all_keeps_refusals_and_pilot_label(tmp_path: Path) -> None:
    by_id = {e.id: e for e in load_examples(_write_dataset(tmp_path), split="test", quality="all")}
    assert sorted(by_id) == ["1", "2", "3", "4"]
    assert by_id["4"].quality == "incorrect_refusal"
    assert by_id["3"].hallucinated is False and by_id["3"].hallucinated_any_span is True
    assert by_id["2"].hallucinated_any_span is True and by_id["1"].hallucinated_any_span is False


def test_labels_ignore_implicit_true_spans(tmp_path: Path) -> None:
    by_id = {e.id: e for e in load_examples(_write_dataset(tmp_path), split="test")}
    assert by_id["1"].hallucinated is False
    assert by_id["2"].hallucinated is True
    assert by_id["3"].hallucinated is False  # only span is implicit_true
    assert by_id["3"].span_types == ()


def test_example_fields_and_slices(tmp_path: Path) -> None:
    e = {x.id: x for x in load_examples(_write_dataset(tmp_path), split="test")}["2"]
    assert e.task == "QA" and e.generator == "llama-2-7b-chat" and e.source == "Q prompt"
    assert e.source_id == "s1"
    assert e.span_types == ("Evident Conflict",) and e.numeric is True
    assert e.severity == "evident" and e.kind == "conflict"
    assert e.n_chars == len("Q prompt") + len("wrong year")
    train = load_examples(_write_dataset(tmp_path), split="train")[0]
    assert train.severity == "subtle" and train.kind == "baseless" and train.numeric is False


def test_clean_example_has_no_severity(tmp_path: Path) -> None:
    e = load_examples(_write_dataset(tmp_path), split="test")[0]
    assert e.severity is None and e.kind is None


def test_question_and_passages_empty_for_non_qa(tmp_path: Path) -> None:
    by_id = {e.id: e for e in load_examples(_write_dataset(tmp_path), split="test")}
    assert by_id["3"].task == "Summary"
    assert by_id["3"].question == "" and by_id["3"].passages == ""


# ---------- convention_dependent + QA question/passages ----------

_CONVENTION_SOURCES = [
    {"source_id": "cs1", "task_type": "QA", "source": "MARCO",
     "source_info": {"question": "q?", "passages": "p1\np2"}, "prompt": "Q prompt"},
]

_CONVENTION_RESPONSES = [
    # clean only by convention: the single span is implicit_true
    {"id": "c1", "source_id": "cs1", "model": "g", "temperature": 0.7, "split": "test",
     "quality": "good", "response": "r1",
     "labels": [_span("Subtle Baseless Info", "true unstated", implicit_true=True)]},
    # hallucinated only by convention: the single counted span is due_to_null
    {"id": "c2", "source_id": "cs1", "model": "g", "temperature": 0.7, "split": "test",
     "quality": "good", "response": "r2",
     "labels": [_span("Evident Conflict", "null field", due_to_null=True)]},
    # ordinary hallucination: counted span is not due_to_null
    {"id": "c3", "source_id": "cs1", "model": "g", "temperature": 0.7, "split": "test",
     "quality": "good", "response": "r3",
     "labels": [_span("Evident Conflict", "just wrong")]},
    # ordinary clean: no spans at all
    {"id": "c4", "source_id": "cs1", "model": "g", "temperature": 0.7, "split": "test",
     "quality": "good", "response": "r4", "labels": []},
    # mixed: one implicit_true span plus one real (non-due_to_null) span -> hallucinated,
    # and the counted span isn't all due_to_null, so not convention_dependent
    {"id": "c5", "source_id": "cs1", "model": "g", "temperature": 0.7, "split": "test",
     "quality": "good", "response": "r5",
     "labels": [_span("Subtle Baseless Info", "unstated", implicit_true=True),
                _span("Evident Conflict", "wrong")]},
]


def _write_convention_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "response.jsonl").write_text(
        "\n".join(json.dumps(r) for r in _CONVENTION_RESPONSES) + "\n"
    )
    (root / "source_info.jsonl").write_text(
        "\n".join(json.dumps(s) for s in _CONVENTION_SOURCES) + "\n"
    )
    return root


def test_convention_dependent_clean_only_by_implicit_true(tmp_path: Path) -> None:
    by_id = {e.id: e for e in load_examples(_write_convention_dataset(tmp_path), split="test")}
    assert by_id["c1"].hallucinated is False
    assert by_id["c1"].convention_dependent is True


def test_convention_dependent_hallucinated_only_by_due_to_null(tmp_path: Path) -> None:
    by_id = {e.id: e for e in load_examples(_write_convention_dataset(tmp_path), split="test")}
    assert by_id["c2"].hallucinated is True
    assert by_id["c2"].convention_dependent is True


def test_convention_dependent_false_for_ordinary_and_mixed_cases(tmp_path: Path) -> None:
    by_id = {e.id: e for e in load_examples(_write_convention_dataset(tmp_path), split="test")}
    assert by_id["c3"].convention_dependent is False  # ordinary hallucination
    assert by_id["c4"].convention_dependent is False  # ordinary clean
    assert by_id["c5"].convention_dependent is False  # hallucinated, mixed spans


def test_question_and_passages_populated_for_qa(tmp_path: Path) -> None:
    by_id = {e.id: e for e in load_examples(_write_convention_dataset(tmp_path), split="test")}
    assert by_id["c1"].question == "q?" and by_id["c1"].passages == "p1\np2"


@pytest.mark.parametrize(("text", "expected"), [
    ("$5.2M", True), ("in 2022", True), ("on March 3", True), ("the CEO resigned", False),
    ("Mayor of Springfield", False),
    ("it may help", False), ("in May 2020", True), ("they march on", False),
])
def test_is_numeric_span(text: str, expected: bool) -> None:
    assert is_numeric_span(text) is expected


def _examples(n_per_task: int) -> list[Example]:
    out = []
    for task in ("QA", "Summary", "Data2txt"):
        for i in range(n_per_task):
            out.append(Example(id=f"{task}-{i}", split="test", task=task, generator="g", source="s",
                               response="r", hallucinated=i % 2 == 0, span_types=(), span_texts=(),
                               numeric=False))
    return out


def test_stratified_sample_is_per_task_and_deterministic() -> None:
    pool = _examples(10)
    a = stratified_sample(pool, per_task=4, seed=7)
    b = stratified_sample(list(reversed(pool)), per_task=4, seed=7)
    assert [e.id for e in a] == [e.id for e in b]
    assert {t: sum(e.task == t for e in a) for t in ("QA", "Summary", "Data2txt")} == {
        "QA": 4, "Summary": 4, "Data2txt": 4}


def test_stratified_sample_too_large_raises() -> None:
    with pytest.raises(DatasetError, match="only 2"):
        stratified_sample(_examples(2), per_task=3, seed=1)


def _transport(payloads: dict[str, bytes]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, content=payloads[name])
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_ensure_downloaded_verifies_checksums(tmp_path: Path) -> None:
    payloads = {"response.jsonl": b"r\n", "source_info.jsonl": b"s\n"}
    files = {k: hashlib.sha256(v).hexdigest() for k, v in payloads.items()}
    out = ensure_downloaded(tmp_path / "c", files=files, commit="abc", client=_transport(payloads))
    assert (out / "response.jsonl").read_bytes() == b"r\n"


def test_ensure_downloaded_rejects_bad_checksum(tmp_path: Path) -> None:
    payloads = {"response.jsonl": b"tampered\n", "source_info.jsonl": b"s\n"}
    files = {"response.jsonl": "0" * 64, "source_info.jsonl": hashlib.sha256(b"s\n").hexdigest()}
    with pytest.raises(DatasetError, match="checksum mismatch"):
        ensure_downloaded(tmp_path / "c", files=files, commit="abc", client=_transport(payloads))
    assert not (tmp_path / "c" / "response.jsonl").exists()


def test_ensure_downloaded_rechecks_existing_files(tmp_path: Path) -> None:
    cache = tmp_path / "c"
    cache.mkdir()
    (cache / "response.jsonl").write_bytes(b"corrupted")
    files = {"response.jsonl": hashlib.sha256(b"original").hexdigest()}
    with pytest.raises(DatasetError, match="checksum mismatch"):
        ensure_downloaded(cache, files=files, commit="abc", client=_transport({}))


@pytest.mark.skipif(not (DEFAULT_CACHE / "response.jsonl").exists(), reason="RAGTruth not downloaded")
def test_real_test_split_counts() -> None:
    examples = load_examples(ensure_downloaded(), split="test")
    assert len(examples) == 2675
    assert sum(e.hallucinated for e in examples) == 894
    assert {e.task for e in examples} == {"QA", "Summary", "Data2txt"}
    everything = load_examples(ensure_downloaded(), split="test", quality="all")
    assert len(everything) == 2700
    qa = [e for e in everything if e.task == "QA"]
    # the 2026-09-19 pilot's QA cohort: 900 responses, 160 with any span
    assert len(qa) == 900 and sum(e.hallucinated_any_span for e in qa) == 160


@pytest.mark.skipif(not (DEFAULT_CACHE / "response.jsonl").exists(), reason="RAGTruth not downloaded")
def test_real_convention_dependent_counts() -> None:
    """120 test/good rows are convention_dependent: 49 clean only because every span is
    implicit_true, 71 hallucinated only because every counted span is due_to_null."""
    examples = load_examples(ensure_downloaded(), split="test")
    dependent = [e for e in examples if e.convention_dependent]
    implicit_only = [e for e in dependent if not e.hallucinated]
    due_to_null_only = [e for e in dependent if e.hallucinated]
    assert len(dependent) == 120
    assert len(implicit_only) == 49
    assert len(due_to_null_only) == 71
