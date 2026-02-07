"""Tests for the async predictor HTTP client."""

from __future__ import annotations

import httpx
import pytest

from k8s_ml_predictive_autoscaling.planner.predictor_client import (
    PredictorClient,
    PredictorNotReadyError,
    PredictorUnavailableError,
)

BASE = "http://test-predictor"

# ---------------------------------------------------------------------------
# Mock transport helpers
# ---------------------------------------------------------------------------


def _mock_client(handler) -> httpx.AsyncClient:
    """Build an AsyncClient with MockTransport and a base_url."""
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=BASE,
    )


def _health_ok(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/health":
        return httpx.Response(200, json={"status": "ok"})
    return httpx.Response(404)


def _ingest_ok(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/ingest" and request.method == "POST":
        return httpx.Response(
            202,
            json={"status": "accepted", "buffer_size": 5, "ready_for_prediction": False},
        )
    return httpx.Response(404)


def _predict_ok(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/predict" and request.method == "POST":
        return httpx.Response(
            200,
            json={
                "prediction_normalized": 0.42,
                "prediction_original_scale": 42.0,
                "target_metric": "cpu_usage",
                "forecast_horizon_steps": 3,
                "forecast_horizon_minutes": 15,
                "model_type": "lstm_onnx",
                "buffer_size": 24,
                "inference_ms": 0.3,
                "total_ms": 1.0,
            },
        )
    return httpx.Response(404)


def _predict_425(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/predict" and request.method == "POST":
        return httpx.Response(
            425,
            json={"detail": "Not enough data: 5/24 observations buffered."},
        )
    return httpx.Response(404)


def _server_error(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, json={"detail": "internal error"})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPredictorClientHealth:

    @pytest.mark.asyncio
    async def test_health_ok(self) -> None:
        pc = PredictorClient(BASE, client=_mock_client(_health_ok))
        assert await pc.health() is True

    @pytest.mark.asyncio
    async def test_health_failure(self) -> None:
        pc = PredictorClient(BASE, client=_mock_client(_server_error))
        assert await pc.health() is False


class TestPredictorClientIngest:

    @pytest.mark.asyncio
    async def test_ingest_success(self) -> None:
        pc = PredictorClient(BASE, client=_mock_client(_ingest_ok))
        result = await pc.ingest({"cpu_usage": 50.0, "mem_util_percent": 30.0})
        assert result["status"] == "accepted"
        assert result["buffer_size"] == 5

    @pytest.mark.asyncio
    async def test_ingest_server_error_raises(self) -> None:
        pc = PredictorClient(BASE, client=_mock_client(_server_error))
        with pytest.raises(PredictorUnavailableError, match="Failed to ingest"):
            await pc.ingest({"cpu_usage": 50.0})


class TestPredictorClientPredict:

    @pytest.mark.asyncio
    async def test_predict_success(self) -> None:
        pc = PredictorClient(BASE, client=_mock_client(_predict_ok))
        result = await pc.predict()
        assert result.prediction_original_scale == 42.0
        assert result.target_metric == "cpu_usage"
        assert result.forecast_horizon_minutes == 15
        assert result.inference_ms == 0.3

    @pytest.mark.asyncio
    async def test_predict_not_ready_raises(self) -> None:
        pc = PredictorClient(BASE, client=_mock_client(_predict_425))
        with pytest.raises(PredictorNotReadyError, match="buffer not ready"):
            await pc.predict()

    @pytest.mark.asyncio
    async def test_predict_server_error_raises(self) -> None:
        pc = PredictorClient(BASE, client=_mock_client(_server_error))
        with pytest.raises(PredictorUnavailableError, match="Failed to get prediction"):
            await pc.predict()


class TestPredictorClientLifecycle:

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Verify async context manager does not raise."""
        async with PredictorClient(BASE, client=_mock_client(_health_ok)) as pc:
            assert await pc.health() is True
