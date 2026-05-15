"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval.config import Thresholds


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def default_thresholds() -> Thresholds:
    return Thresholds()
