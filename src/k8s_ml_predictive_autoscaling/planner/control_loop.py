"""Async control loop -- the central planner orchestrator.

Each iteration:
  1. Fetch latest metrics from Prometheus
  2. Ingest into the predictor buffer
  3. Request a CPU prediction
  4. Compute a scaling decision
  5. Execute via the executor (K8s or mock)

Failure in any phase causes a hold (no scaling action) and increments
the consecutive failure counter. After ``max_consecutive_failures``
successive failures the planner enters degraded mode.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from ..executor.base import ScalingExecutor
from .algorithm import ScalingConfig, ScalingDecision, ScalingState, compute_scaling_decision
from .config import PlannerSettings
from .metrics import (
    CONSECUTIVE_FAILURES,
    COOLDOWN_ACTIVE,
    CURRENT_REPLICAS,
    DEGRADED,
    DESIRED_REPLICAS,
    FALLBACK_TOTAL,
    LOOP_DURATION,
    LOOP_ERRORS_TOTAL,
    PREDICTED_CPU,
    SCALING_DECISIONS_TOTAL,
    SCALING_EXECUTIONS_TOTAL,
)
from .predictor_client import (
    PredictorClient,
    PredictorNotReadyError,
    PredictorUnavailableError,
)
from .prometheus_feeder import PrometheusFeeder

logger = logging.getLogger(__name__)


@dataclass
class LoopStatus:
    """Snapshot of the control loop's latest cycle for /status endpoint."""

    cycle_count: int = 0
    last_decision: ScalingDecision | None = None
    last_error: str | None = None
    degraded: bool = False
    backfill_done: bool = False


class PlannerControlLoop:
    """Async control loop that drives the planner.

    Args:
        settings: Planner configuration.
        predictor: Async client for the predictor service.
        feeder: Async Prometheus feeder.
        executor: Scaling executor (K8s or mock).
    """

    def __init__(
        self,
        settings: PlannerSettings,
        predictor: PredictorClient,
        feeder: PrometheusFeeder,
        executor: ScalingExecutor,
    ) -> None:
        self.settings = settings
        self.predictor = predictor
        self.feeder = feeder
        self.executor = executor

        # Algorithm state
        self._scaling_config = ScalingConfig(
            target_utilization=settings.target_utilization,
            min_replicas=settings.min_replicas,
            max_replicas=settings.max_replicas,
            safety_margin=settings.safety_margin,
            smoothing_alpha=settings.smoothing_alpha,
            scale_up_threshold_factor=settings.scale_up_threshold_factor,
            scale_down_threshold_factor=settings.scale_down_threshold_factor,
            max_scale_up_step=settings.max_scale_up_step,
            max_scale_down_step=settings.max_scale_down_step,
            scale_up_cooldown_seconds=settings.scale_up_cooldown_seconds,
            scale_down_cooldown_seconds=settings.scale_down_cooldown_seconds,
            scale_down_stabilization_count=settings.scale_down_stabilization_count,
        )
        self._scaling_state = ScalingState()

        # Observable status
        self.status = LoopStatus()

        # Cancellation
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Backfill -- warm the predictor buffer from historical Prometheus data
    # ------------------------------------------------------------------

    async def backfill(self) -> int:
        """Backfill the predictor buffer with historical observations.

        Returns:
            Number of observations ingested.
        """
        logger.info("Starting predictor backfill from Prometheus history")
        try:
            observations = await self.feeder.fetch_historical(
                duration_minutes=120, step_seconds=300,
            )
            count = 0
            for obs in observations:
                try:
                    await self.predictor.ingest(obs)
                    count += 1
                except PredictorUnavailableError:
                    logger.warning("Predictor unavailable during backfill")
                    break
            logger.info("Backfill complete: %d observations ingested", count)
            self.status.backfill_done = True
            return count
        except Exception as exc:
            logger.error("Backfill failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Single cycle
    # ------------------------------------------------------------------

    async def run_once(self) -> ScalingDecision | None:
        """Execute a single control loop iteration.

        Returns:
            The scaling decision, or None if an error prevented a decision.
        """
        now = time.monotonic()
        start = time.perf_counter()

        try:
            # Phase 1: Fetch latest metrics from Prometheus
            try:
                snapshot = await self.feeder.fetch_latest()
            except Exception as exc:
                LOOP_ERRORS_TOTAL.labels(phase="fetch").inc()
                raise RuntimeError(f"Prometheus fetch failed: {exc}") from exc

            # Phase 2: Ingest into predictor
            try:
                observation = self.feeder.snapshot_to_observation(snapshot)
                await self.predictor.ingest(observation)
            except PredictorUnavailableError as exc:
                LOOP_ERRORS_TOTAL.labels(phase="ingest").inc()
                raise RuntimeError(f"Predictor ingest failed: {exc}") from exc

            # Phase 3: Get prediction
            try:
                prediction = await self.predictor.predict()
                predicted_cpu = prediction.prediction_original_scale
            except PredictorNotReadyError:
                LOOP_ERRORS_TOTAL.labels(phase="predict").inc()
                logger.info(
                    "Predictor buffer warming up, holding current replicas"
                )
                FALLBACK_TOTAL.inc()
                self._scaling_state.consecutive_failures = 0
                return None
            except PredictorUnavailableError as exc:
                LOOP_ERRORS_TOTAL.labels(phase="predict").inc()
                raise RuntimeError(f"Prediction failed: {exc}") from exc

            # Phase 4: Sync current replicas from executor
            current = await self.executor.get_current_replicas()
            self._scaling_state.current_replicas = current
            CURRENT_REPLICAS.set(current)

            # Phase 5: Compute scaling decision
            decision = compute_scaling_decision(
                predicted_cpu=predicted_cpu,
                state=self._scaling_state,
                config=self._scaling_config,
                now=now,
            )

            # Update metrics
            SCALING_DECISIONS_TOTAL.labels(direction=decision.direction).inc()
            PREDICTED_CPU.set(decision.smoothed_cpu)
            DESIRED_REPLICAS.set(decision.desired_replicas)

            # Phase 6: Execute if needed
            if decision.should_execute:
                try:
                    ok = await self.executor.scale(
                        decision.desired_replicas, reason=decision.reason
                    )
                    if ok:
                        SCALING_EXECUTIONS_TOTAL.labels(direction=decision.direction).inc()
                        CURRENT_REPLICAS.set(decision.desired_replicas)
                        logger.info(
                            "Scaled %s: %d -> %d (%s)",
                            decision.direction,
                            current,
                            decision.desired_replicas,
                            decision.reason,
                        )
                    else:
                        logger.warning("Executor returned False for scale")
                except Exception as exc:
                    LOOP_ERRORS_TOTAL.labels(phase="execute").inc()
                    logger.error("Executor error: %s", exc)
                    # Don't propagate -- the decision was valid
            else:
                logger.debug(
                    "Hold: %s (predicted=%.1f%%, smoothed=%.1f%%)",
                    decision.reason,
                    decision.predicted_cpu,
                    decision.smoothed_cpu,
                )

            # Success: reset failure counter
            self._scaling_state.consecutive_failures = 0
            CONSECUTIVE_FAILURES.set(0)
            if self.status.degraded:
                self.status.degraded = False
                DEGRADED.set(0)
                logger.info("Planner recovered from degraded state")

            self.status.last_decision = decision
            self.status.last_error = None
            self.status.cycle_count += 1
            return decision

        except Exception as exc:
            # Global failure handler
            self._scaling_state.consecutive_failures += 1
            CONSECUTIVE_FAILURES.set(self._scaling_state.consecutive_failures)
            FALLBACK_TOTAL.inc()

            if (
                self._scaling_state.consecutive_failures
                >= self.settings.max_consecutive_failures
            ):
                self.status.degraded = True
                DEGRADED.set(1)
                logger.error(
                    "Planner DEGRADED: %d consecutive failures",
                    self._scaling_state.consecutive_failures,
                )

            self.status.last_error = str(exc)
            logger.error("Control loop error: %s", exc)
            return None

        finally:
            elapsed = time.perf_counter() - start
            LOOP_DURATION.observe(elapsed)

    # ------------------------------------------------------------------
    # Continuous loop
    # ------------------------------------------------------------------

    async def run_forever(self) -> None:
        """Run the control loop continuously until cancelled."""
        logger.info(
            "Starting planner control loop (interval=%ds)",
            self.settings.loop_interval_seconds,
        )

        if self.settings.backfill_on_startup:
            await self.backfill()

        while True:
            await self.run_once()
            await asyncio.sleep(self.settings.loop_interval_seconds)

    def start(self) -> asyncio.Task[None]:
        """Start the control loop as a background task.

        Returns:
            The asyncio.Task running the loop.
        """
        self._task = asyncio.create_task(self.run_forever(), name="planner-loop")
        return self._task

    async def stop(self) -> None:
        """Cancel the background task and wait for it to finish."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Planner control loop stopped")
