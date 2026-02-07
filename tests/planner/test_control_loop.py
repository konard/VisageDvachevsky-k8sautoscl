"""Tests for the planner control loop."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from k8s_ml_predictive_autoscaling.executor.mock_executor import MockExecutor
from k8s_ml_predictive_autoscaling.planner.config import PlannerSettings
from k8s_ml_predictive_autoscaling.planner.control_loop import PlannerControlLoop
from k8s_ml_predictive_autoscaling.planner.predictor_client import (
    PredictionResult,
    PredictorClient,
    PredictorNotReadyError,
    PredictorUnavailableError,
)
from k8s_ml_predictive_autoscaling.planner.prometheus_feeder import (
    MetricSnapshot,
    PrometheusFeeder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(**overrides) -> PlannerSettings:
    """Create PlannerSettings with sensible test defaults."""
    defaults = {
        "loop_interval_seconds": 5,
        "backfill_on_startup": False,
        "max_consecutive_failures": 3,
        "predictor_url": "http://predictor:8000",
        "prometheus_url": "http://prometheus:9090",
    }
    defaults.update(overrides)
    return PlannerSettings(**defaults)


def _mock_predictor(
    *,
    healthy: bool = True,
    prediction_cpu: float = 55.0,
    not_ready: bool = False,
    unavailable: bool = False,
) -> AsyncMock:
    """Build a mock PredictorClient."""
    mock = AsyncMock(spec=PredictorClient)
    mock.health.return_value = healthy
    mock.ingest.return_value = {
        "status": "accepted",
        "buffer_size": 24,
        "ready_for_prediction": True,
    }
    if not_ready:
        mock.predict.side_effect = PredictorNotReadyError("buffer warming")
    elif unavailable:
        mock.predict.side_effect = PredictorUnavailableError("connection refused")
    else:
        mock.predict.return_value = PredictionResult(
            prediction_normalized=prediction_cpu / 100.0,
            prediction_original_scale=prediction_cpu,
            target_metric="cpu_usage",
            forecast_horizon_steps=3,
            forecast_horizon_minutes=15,
            model_type="lstm_onnx",
            buffer_size=24,
            inference_ms=0.3,
        )
    return mock


def _mock_feeder(
    *,
    cpu: float = 55.0,
    historical_count: int = 0,
) -> AsyncMock:
    """Build a mock PrometheusFeeder."""
    mock = AsyncMock(spec=PrometheusFeeder)
    mock.fetch_latest.return_value = MetricSnapshot(
        cpu_usage=cpu,
        mem_util_percent=40.0,
        net_in=100.0,
        net_out=50.0,
        disk_io_percent=5.0,
        timestamp=1700000000.0,
    )
    mock.snapshot_to_observation = PrometheusFeeder.snapshot_to_observation
    mock.fetch_historical.return_value = [
        {"cpu_usage": 50.0 + i, "mem_util_percent": 40.0,
         "net_in": 100.0, "net_out": 50.0, "disk_io_percent": 5.0,
         "hour": 12, "day_of_week": 2, "is_weekend": 0, "minute_of_day": 720}
        for i in range(historical_count)
    ]
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunOnce:

    @pytest.mark.asyncio
    async def test_hold_in_dead_zone(self) -> None:
        """CPU at 55% is in the dead zone -- should hold."""
        settings = _settings()
        predictor = _mock_predictor(prediction_cpu=55.0)
        feeder = _mock_feeder(cpu=55.0)
        executor = MockExecutor(initial_replicas=3)

        loop = PlannerControlLoop(settings, predictor, feeder, executor)
        decision = await loop.run_once()

        assert decision is not None
        assert decision.direction == "hold"
        assert decision.should_execute is False
        assert await executor.get_current_replicas() == 3
        assert loop.status.cycle_count == 1

    @pytest.mark.asyncio
    async def test_scale_up_on_high_cpu(self) -> None:
        """CPU at 80% should trigger scale-up."""
        settings = _settings()
        predictor = _mock_predictor(prediction_cpu=80.0)
        feeder = _mock_feeder(cpu=80.0)
        executor = MockExecutor(initial_replicas=3)

        loop = PlannerControlLoop(settings, predictor, feeder, executor)
        decision = await loop.run_once()

        assert decision is not None
        assert decision.direction == "up"
        assert decision.should_execute is True
        assert await executor.get_current_replicas() > 3
        assert len(executor.history) == 1

    @pytest.mark.asyncio
    async def test_predictor_not_ready_returns_none(self) -> None:
        """When predictor buffer is warming, return None (hold)."""
        settings = _settings()
        predictor = _mock_predictor(not_ready=True)
        feeder = _mock_feeder()
        executor = MockExecutor(initial_replicas=3)

        loop = PlannerControlLoop(settings, predictor, feeder, executor)
        decision = await loop.run_once()

        assert decision is None
        assert await executor.get_current_replicas() == 3
        assert len(executor.history) == 0

    @pytest.mark.asyncio
    async def test_predictor_unavailable_increments_failures(self) -> None:
        """Connection failure should increment consecutive_failures."""
        settings = _settings()
        predictor = _mock_predictor(unavailable=True)
        feeder = _mock_feeder()
        executor = MockExecutor(initial_replicas=3)

        loop = PlannerControlLoop(settings, predictor, feeder, executor)
        decision = await loop.run_once()

        assert decision is None
        assert loop._scaling_state.consecutive_failures == 1
        assert loop.status.last_error is not None

    @pytest.mark.asyncio
    async def test_degraded_after_max_failures(self) -> None:
        """After max_consecutive_failures, planner enters degraded mode."""
        settings = _settings(max_consecutive_failures=2)
        predictor = _mock_predictor(unavailable=True)
        feeder = _mock_feeder()
        executor = MockExecutor(initial_replicas=3)

        loop = PlannerControlLoop(settings, predictor, feeder, executor)
        await loop.run_once()
        assert not loop.status.degraded
        await loop.run_once()
        assert loop.status.degraded


class TestBackfill:

    @pytest.mark.asyncio
    async def test_backfill_ingests_historical_data(self) -> None:
        settings = _settings()
        predictor = _mock_predictor()
        feeder = _mock_feeder(historical_count=10)
        executor = MockExecutor()

        loop = PlannerControlLoop(settings, predictor, feeder, executor)
        count = await loop.backfill()

        assert count == 10
        assert predictor.ingest.call_count == 10
        assert loop.status.backfill_done is True

    @pytest.mark.asyncio
    async def test_backfill_handles_predictor_unavailable(self) -> None:
        settings = _settings()
        predictor = _mock_predictor()
        # Make ingest fail after 3 calls
        predictor.ingest.side_effect = [
            {"status": "accepted", "buffer_size": 1, "ready_for_prediction": False},
            {"status": "accepted", "buffer_size": 2, "ready_for_prediction": False},
            {"status": "accepted", "buffer_size": 3, "ready_for_prediction": False},
            PredictorUnavailableError("gone"),
        ]
        feeder = _mock_feeder(historical_count=10)
        executor = MockExecutor()

        loop = PlannerControlLoop(settings, predictor, feeder, executor)
        count = await loop.backfill()

        assert count == 3  # stopped at the error


class TestLoopLifecycle:

    @pytest.mark.asyncio
    async def test_start_and_stop(self) -> None:
        """Verify start/stop doesn't hang or raise."""
        settings = _settings(loop_interval_seconds=60)
        predictor = _mock_predictor()
        feeder = _mock_feeder()
        executor = MockExecutor()

        loop = PlannerControlLoop(settings, predictor, feeder, executor)
        task = loop.start()
        assert task is not None

        # Let it run briefly
        await asyncio.sleep(0.05)
        await loop.stop()
        assert task.done()
