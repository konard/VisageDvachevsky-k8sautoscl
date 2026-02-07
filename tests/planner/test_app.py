"""Tests for the planner FastAPI application."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from k8s_ml_predictive_autoscaling.planner.config import PlannerSettings
from k8s_ml_predictive_autoscaling.planner.predictor_client import (
    PredictionResult,
    PredictorClient,
)
from k8s_ml_predictive_autoscaling.planner.prometheus_feeder import (
    MetricSnapshot,
    PrometheusFeeder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _test_settings() -> PlannerSettings:
    return PlannerSettings(
        use_k8s_client=False,  # MockExecutor
        backfill_on_startup=False,
        loop_interval_seconds=3600,  # Very long so loop doesn't run during tests
        predictor_url="http://mock-predictor:8000",
        prometheus_url="http://mock-prometheus:9090",
    )


def _mock_predictor() -> AsyncMock:
    mock = AsyncMock(spec=PredictorClient)
    mock.health.return_value = True
    mock.ingest.return_value = {
        "status": "accepted", "buffer_size": 24, "ready_for_prediction": True,
    }
    mock.predict.return_value = PredictionResult(
        prediction_normalized=0.55,
        prediction_original_scale=55.0,
        target_metric="cpu_usage",
        forecast_horizon_steps=3,
        forecast_horizon_minutes=15,
        model_type="lstm_onnx",
        buffer_size=24,
        inference_ms=0.3,
    )
    mock.close = AsyncMock()
    return mock


def _mock_feeder() -> AsyncMock:
    mock = AsyncMock(spec=PrometheusFeeder)
    mock.fetch_latest.return_value = MetricSnapshot(
        cpu_usage=55.0, mem_util_percent=40.0,
        net_in=100.0, net_out=50.0, disk_io_percent=5.0,
        timestamp=1700000000.0,
    )
    mock.snapshot_to_observation = PrometheusFeeder.snapshot_to_observation
    mock.fetch_historical.return_value = []
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """Create a TestClient with mocked external dependencies."""
    settings = _test_settings()
    predictor = _mock_predictor()
    feeder = _mock_feeder()

    # Patch the PredictorClient and PrometheusFeeder constructors
    with (
        patch(
            "k8s_ml_predictive_autoscaling.planner.app.PredictorClient",
            return_value=predictor,
        ),
        patch(
            "k8s_ml_predictive_autoscaling.planner.app.PrometheusFeeder",
            return_value=feeder,
        ),
    ):
        from k8s_ml_predictive_autoscaling.planner.app import create_app

        app = create_app(settings)
        with TestClient(app) as client:
            yield client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthEndpoints:

    def test_health_returns_ok(self, test_client: TestClient) -> None:
        resp = test_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ready_returns_status(self, test_client: TestClient) -> None:
        resp = test_client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "cycle_count" in data
        assert "degraded" in data


class TestStatusEndpoint:

    def test_status_returns_loop_info(self, test_client: TestClient) -> None:
        resp = test_client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "cycle_count" in data
        assert "degraded" in data
        assert "current_replicas" in data
        assert "consecutive_failures" in data


class TestOverrideEndpoint:

    def test_override_applies_replicas(self, test_client: TestClient) -> None:
        resp = test_client.post(
            "/override",
            json={"desired_replicas": 5, "reason": "test override"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "applied"
        assert data["desired_replicas"] == 5

    def test_override_validates_input(self, test_client: TestClient) -> None:
        resp = test_client.post(
            "/override", json={"desired_replicas": 0},
        )
        assert resp.status_code == 422  # Validation error


class TestMetricsEndpoint:

    def test_metrics_returns_prometheus_text(self, test_client: TestClient) -> None:
        resp = test_client.get("/metrics")
        assert resp.status_code == 200
        body = resp.text
        assert "planner_" in body or "python_" in body or "process_" in body
