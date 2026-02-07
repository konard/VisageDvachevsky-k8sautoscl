"""Tests for the historical collector routine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from k8s_ml_predictive_autoscaling.collector.collect_historical import HistoricalCollector
from k8s_ml_predictive_autoscaling.collector.config import CollectorConfig


class StubPrometheusClient:
    def __init__(self, payload: list[dict[str, Any]]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, datetime, datetime]] = []

    def query_range(
        self,
        query: str,
        *,
        start: datetime,
        end: datetime,
        step: timedelta,
    ) -> list[dict[str, Any]]:
        self.calls.append((query, start, end))
        return self.payload


def config_for_tests(tmp_path: Path) -> CollectorConfig:
    return CollectorConfig.model_validate(
        {
            "prometheus": {"base_url": "http://localhost:9090"},
            "collection": {
                "output_dir": str(tmp_path),
                "lookback_hours": 1,
                "chunk_hours": 1,
                "default_step": "30s",
            },
            "metrics": [
                {
                    "name": "cpu",
                    "promql": "cpu_query",
                    "output_prefix": "cpu_metrics",
                }
            ],
        }
    )


def test_collector_persists_samples(tmp_path: Path) -> None:
    payload = [
        {
            "metric": {"pod": "demo"},
            "values": [
                [datetime(2024, 1, 1, tzinfo=UTC).timestamp(), "0.5"],
                [datetime(2024, 1, 1, 0, 5, tzinfo=UTC).timestamp(), "0.7"],
            ],
        }
    ]
    client = StubPrometheusClient(payload)
    config = config_for_tests(tmp_path)
    collector = HistoricalCollector(config, client)  # type: ignore[arg-type]

    outputs = collector.collect()

    assert len(outputs) == 1
    saved = outputs[0]
    assert saved.exists()
    content = saved.read_text(encoding="utf-8")
    assert "timestamp" in content
    assert "0.5" in content


def test_collect_invokes_prometheus_with_chunks(tmp_path: Path) -> None:
    payload: list[dict[str, Any]] = []
    client = StubPrometheusClient(payload)
    config = config_for_tests(tmp_path)
    collector = HistoricalCollector(config, client)  # type: ignore[arg-type]

    collector.collect()

    assert client.calls  # at least one call executed
