from __future__ import annotations

from pathlib import Path

import pytest

from ragverdict.config import (
    Config,
    ConfigError,
    HttpAdapterConfig,
    PythonAdapterConfig,
    load_config,
)


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(body)
    return p


def test_load_minimal_python_adapter(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        adapter:
          type: python
          module: my.module
          class: MyAdapter
        tests: []
        """,
    )
    cfg = load_config(path)
    assert isinstance(cfg, Config)
    assert isinstance(cfg.adapter, PythonAdapterConfig)
    assert cfg.adapter.module == "my.module"
    assert cfg.adapter.cls == "MyAdapter"
    assert cfg.judge.model == "claude-sonnet-4-6"
    assert cfg.tests == []


def test_load_http_adapter(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        adapter:
          type: http
          endpoint: https://rag.example.com/query
          headers:
            Authorization: Bearer XYZ
        tests: []
        """,
    )
    cfg = load_config(path)
    assert isinstance(cfg.adapter, HttpAdapterConfig)
    assert cfg.adapter.endpoint == "https://rag.example.com/query"
    assert cfg.adapter.headers["Authorization"] == "Bearer XYZ"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_bad_yaml_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "adapter: {type: python\n  oops")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)


def test_validation_error_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        adapter:
          type: python
          # missing required fields: module, class
        tests: []
        """,
    )
    with pytest.raises(ConfigError, match="validation failed"):
        load_config(path)
