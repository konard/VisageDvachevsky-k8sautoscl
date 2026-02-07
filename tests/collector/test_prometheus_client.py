"""Tests for the Prometheus HTTP client wrapper."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from k8s_ml_predictive_autoscaling.collector.prometheus_client import (
    PrometheusClient,
    PrometheusQueryError,
)


def _build_client(handler: httpx.MockTransport) -> PrometheusClient:
    client = httpx.Client(transport=handler, base_url="http://prometheus:9090")
    return PrometheusClient("http://prometheus:9090", client=client)


def test_query_range_returns_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = {
            "status": "success",
            "data": {"result": [{"metric": {}, "values": [[1, "0.5"]]}]},
        }
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    client = _build_client(transport)
    result = client.query_range(
        "up",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
        step=timedelta(minutes=1),
    )
    assert result


def test_query_range_raises_on_error_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = {"status": "error", "error": "bad query"}
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    client = _build_client(transport)
    with pytest.raises(PrometheusQueryError):
        client.query_range(
            "bad",
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
            step=timedelta(minutes=1),
        )
