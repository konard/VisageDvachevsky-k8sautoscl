"""Tests for the async Prometheus feeder."""

from __future__ import annotations

import time

import httpx
import pytest

from k8s_ml_predictive_autoscaling.planner.prometheus_feeder import (
    MetricSnapshot,
    PrometheusFeeder,
)

PROM_BASE = "http://prometheus:9090"

# ---------------------------------------------------------------------------
# Mock transports
# ---------------------------------------------------------------------------


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=PROM_BASE,
    )


def _prom_instant_response(metric_name: str, value: float) -> dict:
    """Build a Prometheus /api/v1/query success response."""
    return {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {
                    "metric": {"__name__": metric_name},
                    "value": [time.time(), str(value)],
                }
            ],
        },
    }


def _prom_range_response(
    values: list[tuple[float, float]],
) -> dict:
    """Build a Prometheus /api/v1/query_range success response."""
    return {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [
                {
                    "metric": {},
                    "values": [[ts, str(v)] for ts, v in values],
                }
            ],
        },
    }


def _make_instant_handler(responses: dict[str, float]):
    """Create a handler that returns different values per PromQL query."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/query":
            query = str(request.url.params.get("query", ""))
            for keyword, value in responses.items():
                if keyword in query:
                    return httpx.Response(
                        200, json=_prom_instant_response(keyword, value)
                    )
            # default empty result
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": {"resultType": "vector", "result": []},
                },
            )
        return httpx.Response(404)

    return handler


def _make_range_handler(
    series: dict[str, list[tuple[float, float]]],
):
    """Create a handler returning range data per query keyword."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/query_range":
            query = str(request.url.params.get("query", ""))
            for keyword, points in series.items():
                if keyword in query:
                    return httpx.Response(
                        200, json=_prom_range_response(points)
                    )
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": {"resultType": "matrix", "result": []},
                },
            )
        return httpx.Response(404)

    return handler


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFetchLatest:

    @pytest.mark.asyncio
    async def test_returns_metric_snapshot(self) -> None:
        handler = _make_instant_handler({
            "cpu": 72.5,
            "mem": 45.0,
            "http_requests_total": 120.0,
            "duration_seconds": 80.0,
        })
        feeder = PrometheusFeeder(PROM_BASE, client=_mock_client(handler))

        snap = await feeder.fetch_latest()
        assert isinstance(snap, MetricSnapshot)
        assert snap.cpu_usage == 72.5
        assert snap.mem_util_percent == 45.0
        assert snap.timestamp > 0

    @pytest.mark.asyncio
    async def test_missing_metric_defaults_to_zero(self) -> None:
        """If a query returns no results, that metric defaults to 0.0."""
        handler = _make_instant_handler({"cpu": 50.0})
        feeder = PrometheusFeeder(PROM_BASE, client=_mock_client(handler))

        snap = await feeder.fetch_latest()
        assert snap.cpu_usage == 50.0
        assert snap.mem_util_percent == 0.0  # no match for mem query
        assert snap.disk_io_percent == 0.0


class TestFetchHistorical:

    @pytest.mark.asyncio
    async def test_returns_observations_list(self) -> None:
        now = time.time()
        ts_list = [now - 600, now - 300, now]
        range_series = {
            "cpu": [(t, 50.0 + i * 10) for i, t in enumerate(ts_list)],
            "mem": [(t, 30.0) for t in ts_list],
            "http_requests_total": [(t, 100.0) for t in ts_list],
            "duration_seconds": [(t, 50.0) for t in ts_list],
        }
        handler = _make_range_handler(range_series)
        feeder = PrometheusFeeder(PROM_BASE, client=_mock_client(handler))

        observations = await feeder.fetch_historical(
            duration_minutes=15, step_seconds=300
        )
        assert len(observations) == 3
        assert observations[0]["cpu_usage"] == 50.0
        assert observations[2]["cpu_usage"] == 70.0
        # Check time features are present
        assert "hour" in observations[0]
        assert "day_of_week" in observations[0]
        assert "is_weekend" in observations[0]
        assert "minute_of_day" in observations[0]

    @pytest.mark.asyncio
    async def test_empty_when_no_cpu_data(self) -> None:
        """If CPU series is empty, return empty list."""
        handler = _make_range_handler({})
        feeder = PrometheusFeeder(PROM_BASE, client=_mock_client(handler))

        observations = await feeder.fetch_historical()
        assert observations == []


class TestSnapshotToObservation:

    def test_converts_with_time_features(self) -> None:
        # Fixed timestamp: 2024-01-15 14:30:00 UTC (Monday)
        snap = MetricSnapshot(
            cpu_usage=65.0,
            mem_util_percent=40.0,
            net_in=100.0,
            net_out=50.0,
            disk_io_percent=5.0,
            timestamp=1705325400.0,  # 2024-01-15 14:30 UTC
        )
        obs = PrometheusFeeder.snapshot_to_observation(snap)

        assert obs["cpu_usage"] == 65.0
        assert obs["mem_util_percent"] == 40.0
        assert obs["net_in"] == 100.0
        assert obs["net_out"] == 50.0
        assert obs["disk_io_percent"] == 5.0
        assert isinstance(obs["hour"], int)
        assert isinstance(obs["day_of_week"], int)
        assert obs["is_weekend"] in (0, 1)
        assert 0 <= obs["minute_of_day"] <= 1439


class TestFeederLifecycle:

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        handler = _make_instant_handler({"cpu": 50.0})
        async with PrometheusFeeder(PROM_BASE, client=_mock_client(handler)) as feeder:
            snap = await feeder.fetch_latest()
            assert snap.cpu_usage == 50.0
