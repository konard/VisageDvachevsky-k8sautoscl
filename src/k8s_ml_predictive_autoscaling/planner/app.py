"""FastAPI planner service -- ML-driven autoscaling orchestrator.

Endpoints:
    GET  /health      - Liveness probe
    GET  /ready       - Readiness probe
    GET  /status      - Full control loop status JSON
    GET  /metrics     - Prometheus metrics
    POST /override    - Manual replica override
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from ..executor.mock_executor import MockExecutor
from .config import PlannerSettings, get_planner_settings
from .control_loop import PlannerControlLoop
from .predictor_client import PredictorClient
from .prometheus_feeder import PrometheusFeeder

logger = logging.getLogger(__name__)


# -- Request models -----------------------------------------------------------


class OverrideRequest(BaseModel):
    """Manual replica override."""

    desired_replicas: int = Field(..., ge=1, le=100)
    reason: str = Field(default="manual override")


# -- Application factory ------------------------------------------------------


def create_app(settings: PlannerSettings | None = None) -> FastAPI:
    """Factory for the planner FastAPI application."""

    settings = settings or get_planner_settings()

    # Mutable state shared between lifespan and endpoints
    state: dict[str, Any] = {
        "loop": None,
        "predictor": None,
        "feeder": None,
        "executor": None,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[type-arg]
        """Initialize clients, executor, and start the control loop."""
        logger.info("Planner starting (env=%s)", settings.environment)

        # Build components
        predictor = PredictorClient(
            settings.predictor_url,
            timeout=settings.predictor_timeout_seconds,
        )
        feeder = PrometheusFeeder(
            settings.prometheus_url,
            timeout=settings.prometheus_timeout_seconds,
        )

        # Choose executor based on configuration
        if settings.use_k8s_client:
            try:
                from ..executor.k8s_executor import K8sExecutor

                executor = K8sExecutor(
                    deployment_name=settings.target_deployment,
                    namespace=settings.target_namespace,
                )
                logger.info(
                    "Using K8s executor: %s/%s",
                    settings.target_namespace,
                    settings.target_deployment,
                )
            except ImportError:
                logger.warning(
                    "kubernetes package not installed, falling back to MockExecutor"
                )
                executor = MockExecutor(initial_replicas=settings.min_replicas)
        else:
            executor = MockExecutor(initial_replicas=settings.min_replicas)
            logger.info("Using MockExecutor (PLANNER_USE_K8S_CLIENT=false)")

        # Build and start control loop
        loop = PlannerControlLoop(settings, predictor, feeder, executor)
        loop.start()

        state["loop"] = loop
        state["predictor"] = predictor
        state["feeder"] = feeder
        state["executor"] = executor

        logger.info("Planner control loop started")
        yield

        # Shutdown
        await loop.stop()
        await predictor.close()
        await feeder.close()
        logger.info("Planner shutdown complete")

    app = FastAPI(
        title=settings.service_name,
        version="0.1.0",
        description="ML-driven predictive autoscaling planner",
        lifespan=lifespan,
    )

    def _get_loop() -> PlannerControlLoop:
        loop = state["loop"]
        if loop is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Planner not initialized",
            )
        return loop

    # -- Endpoints -------------------------------------------------------------

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        """Liveness probe -- always OK if the process is running."""
        return {"status": "ok"}

    @app.get("/ready", tags=["system"])
    def ready() -> dict[str, Any]:
        """Readiness probe -- checks loop is running and not degraded."""
        loop = _get_loop()
        is_ready = loop.status.cycle_count > 0 and not loop.status.degraded
        return {
            "status": "ready" if is_ready else "not_ready",
            "cycle_count": loop.status.cycle_count,
            "degraded": loop.status.degraded,
            "backfill_done": loop.status.backfill_done,
        }

    @app.get("/status", tags=["system"])
    async def get_status() -> dict[str, Any]:
        """Detailed planner status with latest decision."""
        loop = _get_loop()
        executor = state["executor"]

        current_replicas = (
            await executor.get_current_replicas() if executor else None
        )

        result: dict[str, Any] = {
            "cycle_count": loop.status.cycle_count,
            "degraded": loop.status.degraded,
            "backfill_done": loop.status.backfill_done,
            "last_error": loop.status.last_error,
            "current_replicas": current_replicas,
            "consecutive_failures": loop._scaling_state.consecutive_failures,
        }

        if loop.status.last_decision is not None:
            d = loop.status.last_decision
            result["last_decision"] = {
                "direction": d.direction,
                "desired_replicas": d.desired_replicas,
                "predicted_cpu": round(d.predicted_cpu, 2),
                "smoothed_cpu": round(d.smoothed_cpu, 2),
                "reason": d.reason,
                "should_execute": d.should_execute,
            }

        return result

    @app.post("/override", tags=["control"])
    async def override(request: OverrideRequest) -> dict[str, Any]:
        """Manually override the replica count (for debugging/emergencies)."""
        executor = state["executor"]
        if executor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Executor not initialized",
            )

        ok = await executor.scale(
            request.desired_replicas, reason=request.reason
        )
        return {
            "status": "applied" if ok else "failed",
            "desired_replicas": request.desired_replicas,
            "reason": request.reason,
        }

    @app.get(settings.metrics_path, tags=["system"], response_class=PlainTextResponse)
    def metrics() -> PlainTextResponse:
        """Prometheus metrics endpoint."""
        data = generate_latest()
        return PlainTextResponse(content=data, media_type=CONTENT_TYPE_LATEST)

    return app


def get_app() -> FastAPI:
    """Convenience wrapper for uvicorn entrypoint."""
    return create_app()
