"""Instantiate the user's RagAdapter from config."""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

from ragverdict.adapters.base import RagAdapter
from ragverdict.adapters.http import HttpAdapter
from ragverdict.config import AdapterConfig, HttpAdapterConfig, PythonAdapterConfig


class AdapterLoadError(Exception):
    """Raised when an adapter cannot be loaded or instantiated."""


def load_adapter(config: AdapterConfig) -> RagAdapter:
    if isinstance(config, PythonAdapterConfig):
        return _load_python(config.module, config.cls)
    if isinstance(config, HttpAdapterConfig):
        return HttpAdapter(
            endpoint=config.endpoint,
            headers=config.headers,
            timeout_s=config.timeout_s,
        )
    raise AdapterLoadError(f"unknown adapter type: {type(config).__name__}")


def _load_python(module_name: str, class_name: str) -> RagAdapter:
    # User adapters live in their project tree, not in this package — make cwd
    # importable so `module: examples.demo_rag.adapter` resolves when the user
    # runs `ragverdict` from their project root.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise AdapterLoadError(f"cannot import module {module_name!r}: {exc}") from exc
    cls: Any = getattr(module, class_name, None)
    if cls is None:
        raise AdapterLoadError(f"module {module_name!r} has no attribute {class_name!r}")
    try:
        instance = cls()
    except Exception as exc:
        raise AdapterLoadError(f"failed to instantiate {module_name}.{class_name}: {exc}") from exc
    if not isinstance(instance, RagAdapter):
        raise AdapterLoadError(
            f"{module_name}.{class_name} is not a RagAdapter subclass"
        )
    return instance
