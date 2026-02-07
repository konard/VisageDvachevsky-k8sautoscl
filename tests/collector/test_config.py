"""Tests for collector config parsing."""

from pathlib import Path

import pytest

from k8s_ml_predictive_autoscaling.collector.config import (
    CollectorConfig,
    load_config,
    parse_duration,
)


def test_parse_duration_parses_supported_units() -> None:
    assert parse_duration("30s").total_seconds() == 30
    assert parse_duration("5m").total_seconds() == 300
    assert parse_duration("2h").total_seconds() == 7200


def test_parse_duration_rejects_invalid_unit() -> None:
    with pytest.raises(ValueError):
        parse_duration("10d")


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(
        """
        prometheus:
          base_url: http://localhost:9090
        collection:
          output_dir: data/raw
          lookback_hours: 12
          chunk_hours: 2
          default_step: 15s
        metrics:
          - name: cpu
            promql: cpu_query
        """,
        encoding="utf-8",
    )

    config = load_config(config_yaml)
    assert isinstance(config, CollectorConfig)
    assert config.prometheus.base_url == "http://localhost:9090"
    assert config.collection.lookback_hours == 12
    assert config.metrics[0].name == "cpu"
