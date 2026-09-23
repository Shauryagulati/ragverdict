"""RAGTruth: pinned download, checksum verification, filtering, labels, slice metadata.

Label rule: a response is hallucinated iff it has at least one annotated span with
`implicit_true == false` (implicit_true spans are correct-but-unstated information).
Only `quality == "good"` responses are kept.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

RAGTRUTH_COMMIT = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
RAGTRUTH_FILES = {
    "response.jsonl": "e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073",
    "source_info.jsonl": "0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b",
}
_RAW_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{commit}/dataset/{name}"
DEFAULT_CACHE = Path.home() / ".cache" / "ragverdict" / "ragtruth" / RAGTRUTH_COMMIT

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
)
_NUMERIC_RE = re.compile(rf"\d|\b({_MONTHS})\b", re.IGNORECASE)


class DatasetError(Exception):
    """Raised when the dataset is missing, corrupted, or can't be sampled as asked."""


@dataclass(frozen=True)
class Example:
    id: str
    split: str
    task: str  # "QA" | "Summary" | "Data2txt"
    generator: str  # model that wrote the response
    source: str  # the prompt the generator saw — what every judge is given
    response: str
    hallucinated: bool
    span_types: tuple[str, ...]
    span_texts: tuple[str, ...]
    numeric: bool  # some counted span contains a number or month name
    source_id: str = ""  # ~6 responses share a source; bootstrap resamples by this
    quality: str = "good"  # RAGTruth quality flag: good | incorrect_refusal | truncated
    hallucinated_any_span: bool = False  # 2026-09-19 pilot's rule: any span, incl. implicit_true

    @property
    def severity(self) -> str | None:
        if not self.hallucinated:
            return None
        return "evident" if any(t.startswith("Evident") for t in self.span_types) else "subtle"

    @property
    def kind(self) -> str | None:
        if not self.hallucinated:
            return None
        return "conflict" if any(t.endswith("Conflict") for t in self.span_types) else "baseless"

    @property
    def n_chars(self) -> int:
        return len(self.source) + len(self.response)


def is_numeric_span(text: str) -> bool:
    return bool(_NUMERIC_RE.search(text))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_downloaded(
    cache_dir: Path = DEFAULT_CACHE,
    *,
    files: dict[str, str] = RAGTRUTH_FILES,
    commit: str = RAGTRUTH_COMMIT,
    client: httpx.Client | None = None,
) -> Path:
    """Download any missing files, then verify every file's sha256. Returns `cache_dir`."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    http = client or httpx.Client(timeout=120.0, follow_redirects=True)
    for name, expected in files.items():
        path = cache_dir / name
        if not path.exists():
            response = http.get(_RAW_URL.format(commit=commit, name=name))
            if response.status_code != 200:
                raise DatasetError(f"download failed for {name}: HTTP {response.status_code}")
            tmp = path.with_suffix(".part")
            tmp.write_bytes(response.content)
            tmp.rename(path)
        actual = _sha256(path)
        if actual != expected:
            path.unlink()
            raise DatasetError(f"checksum mismatch for {name}: expected {expected}, got {actual}")
    return cache_dir


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_examples(
    data_dir: Path, *, split: str, quality: Literal["good", "all"] = "good"
) -> list[Example]:
    """Load one split in file order. quality="good" (default) drops incorrect_refusal and
    truncated responses; quality="all" keeps them (the 2026-09-19 pilot kept everything)."""
    sources = {s["source_id"]: s for s in _read_jsonl(data_dir / "source_info.jsonl")}
    examples: list[Example] = []
    for row in _read_jsonl(data_dir / "response.jsonl"):
        if row["split"] != split or (quality == "good" and row["quality"] != "good"):
            continue
        source = sources[row["source_id"]]
        spans = [s for s in row["labels"] if not s.get("implicit_true")]
        texts = tuple(str(s["text"]) for s in spans)
        examples.append(
            Example(
                id=str(row["id"]),
                split=split,
                task=str(source["task_type"]),
                generator=str(row["model"]),
                source=str(source["prompt"]),
                response=str(row["response"]),
                hallucinated=bool(spans),
                span_types=tuple(str(s["label_type"]) for s in spans),
                span_texts=texts,
                numeric=any(is_numeric_span(t) for t in texts),
                source_id=str(row["source_id"]),
                quality=str(row["quality"]),
                hallucinated_any_span=bool(row["labels"]),
            )
        )
    return examples


def _id_key(e: Example) -> tuple[bool, int, str]:
    """Numeric ids (RAGTruth) sort numerically; anything else sorts after, by string."""
    return (not e.id.isdigit(), int(e.id) if e.id.isdigit() else 0, e.id)


def stratified_sample(examples: Sequence[Example], *, per_task: int, seed: int) -> list[Example]:
    """`per_task` examples from each task type; independent of input order."""
    rng = random.Random(seed)
    chosen: list[Example] = []
    for task in sorted({e.task for e in examples}):
        pool = sorted((e for e in examples if e.task == task), key=_id_key)
        if len(pool) < per_task:
            raise DatasetError(f"task {task} has only {len(pool)} examples, asked for {per_task}")
        chosen.extend(rng.sample(pool, per_task))
    return chosen
