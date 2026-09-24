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


def test_judge_defaults_unchanged(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, """
        adapter: {type: python, module: m, class: C}
    """))
    assert cfg.judge.provider == "anthropic"
    assert cfg.judge.model == "claude-sonnet-4-6"
    assert cfg.judge.thinking == "model_default"
    assert cfg.judge.jev_model == "typesafe/jev-1.13"
    assert cfg.judge.jev_base_url == "https://openrouter.ai/api"
    assert cfg.judge.cascade_band == (0.3, 0.7)


def test_cascade_judge_config_parses(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, """
        adapter: {type: python, module: m, class: C}
        judge:
          provider: cascade
          model: claude-sonnet-5
          thinking: disabled
          cascade_band: [0.2, 0.8]
    """))
    assert cfg.judge.provider == "cascade"
    assert cfg.judge.thinking == "disabled"
    assert cfg.judge.cascade_band == (0.2, 0.8)


@pytest.mark.parametrize("band", ["[0.8, 0.2]", "[-0.1, 0.5]", "[0.5, 1.5]"])
def test_invalid_cascade_band_rejected(tmp_path: Path, band: str) -> None:
    with pytest.raises(ConfigError, match="cascade_band"):
        load_config(_write(tmp_path, f"""
            adapter: {{type: python, module: m, class: C}}
            judge: {{provider: cascade, cascade_band: {band}}}
        """))


def test_unknown_provider_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, """
            adapter: {type: python, module: m, class: C}
            judge: {provider: openai}
        """))


def test_demo_rag_jev_example_config_validates() -> None:
    path = Path(__file__).parent.parent / "examples" / "demo_rag" / "config.jev.yaml"
    cfg = load_config(path)
    assert cfg.judge.provider == "cascade"
    assert cfg.judge.model == "claude-sonnet-4-6"
    assert cfg.judge.jev_model == "typesafe/jev-1.13"
    assert cfg.judge.jev_base_url == "https://openrouter.ai/api"
    assert cfg.judge.cascade_band == (0.1, 0.6)
    assert cfg.thresholds.faithfulness_pass == 0.5
    assert cfg.thresholds.faithfulness_weak == 0.2
