"""FastAPI predictor service -- ML inference for workload forecasting.

Endpoints:
    GET  /health      - Liveness probe
    GET  /ready       - Readiness probe (model loaded + buffer filled)
    POST /predict     - Generate prediction from buffered observations
    POST /ingest      - Ingest new metric observation
    POST /predict/batch - Predict from pre-built sequences
    GET  /metrics     - Prometheus metrics
    GET  /model/info  - Model and engine metadata
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from .config import PredictorSettings, get_predictor_settings
from .engine import PredictionEngine
from .metrics import (
    BUFFER_SIZE,
    INGEST_COUNTER,
    MODEL_LOAD_TIME,
    ONNX_INFERENCE_LATENCY,
    PREDICTION_COUNTER,
    PREDICTION_LATENCY,
    PREDICTION_VALUE,
)

logger = logging.getLogger(__name__)


# -- Request/Response models ---------------------------------------------------


class MetricObservation(BaseModel):
    """Single metric observation from Prometheus or a monitoring system."""

    cpu_usage: float = Field(..., description="CPU utilization percentage")
    mem_util_percent: float = Field(..., description="Memory utilization percentage")
    net_in: float = Field(default=0.0, description="Network inbound")
    net_out: float = Field(default=0.0, description="Network outbound")
    disk_io_percent: float = Field(default=0.0, description="Disk I/O percentage")
    hour: int = Field(default=0, ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(default=0, ge=0, le=6, description="Day of week (0=Mon)")
    is_weekend: int = Field(default=0, ge=0, le=1, description="Weekend flag")
    minute_of_day: int = Field(default=0, ge=0, le=1439, description="Minute of day")


class BatchPredictionRequest(BaseModel):
    """Pre-built feature sequences for batch prediction."""

    sequences: list[list[list[float]]] = Field(
        ..., description="Sequences of shape (batch, seq_len, features)"
    )


class PredictionResponse(BaseModel):
    """Prediction result."""

    prediction_normalized: float
    prediction_original_scale: float
    target_metric: str
    forecast_horizon_steps: int
    forecast_horizon_minutes: int
    model_type: str
    buffer_size: int
    inference_ms: float
    total_ms: float


class BatchPredictionResponse(BaseModel):
    """Batch prediction result."""

    predictions_normalized: list[float]
    predictions_original_scale: list[float]
    batch_size: int
    target_metric: str
    model_type: str
    inference_ms: float
    total_ms: float


# -- Application factory -------------------------------------------------------


def create_app(settings: PredictorSettings | None = None) -> FastAPI:
    """Factory for the predictor FastAPI application."""

    settings = settings or get_predictor_settings()

    # Mutable state shared between lifespan and endpoints
    state: dict[str, Any] = {"engine": None, "load_error": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[type-arg]
        """Load the prediction engine on startup."""
        try:
            start = time.perf_counter()
            state["engine"] = PredictionEngine(
                onnx_model_path=settings.onnx_model_path,
                scaler_path=settings.scaler_path,
                metadata_path=settings.metadata_path,
                sequence_length=settings.sequence_length,
            )
            load_time = time.perf_counter() - start
            MODEL_LOAD_TIME.set(load_time)
            logger.info("Prediction engine loaded in %.3fs", load_time)
        except Exception as exc:
            state["load_error"] = str(exc)
            logger.error("Failed to load prediction engine: %s", exc)
        yield

    app = FastAPI(
        title=settings.service_name,
        version="0.1.0",
        description="ML-powered workload prediction service using ONNX Runtime",
        lifespan=lifespan,
    )

    def _get_engine() -> PredictionEngine:
        engine = state["engine"]
        if engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model not loaded: {state['load_error'] or 'initializing'}",
            )
        return engine

    # -- Endpoints --------------------------------------------------------------

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        """Liveness probe -- always returns OK if the process is running."""
        return {"status": "ok"}

    @app.get("/ready", tags=["system"])
    def ready() -> dict[str, Any]:
        """Readiness probe -- checks model loaded and buffer state."""
        eng = _get_engine()
        is_ready = eng.is_ready
        return {
            "status": "ready" if is_ready else "warming",
            "model_loaded": True,
            "buffer_size": eng.buffer_size,
            "buffer_required": eng.sequence_length,
            "ready_for_prediction": is_ready,
        }

    @app.post("/ingest", tags=["data"], status_code=status.HTTP_202_ACCEPTED)
    def ingest(observation: MetricObservation) -> dict[str, Any]:
        """Ingest a new metric observation into the sliding window."""
        eng = _get_engine()
        eng.ingest(observation.model_dump())
        INGEST_COUNTER.inc()
        BUFFER_SIZE.set(eng.buffer_size)
        return {
            "status": "accepted",
            "buffer_size": eng.buffer_size,
            "ready_for_prediction": eng.is_ready,
        }

    @app.post("/predict", tags=["inference"], response_model=PredictionResponse)
    def predict() -> PredictionResponse:
        """Generate a prediction from the buffered observations."""
        eng = _get_engine()

        if not eng.is_ready:
            raise HTTPException(
                status_code=status.HTTP_425_TOO_EARLY,
                detail=(
                    f"Not enough data: {eng.buffer_size}/{eng.sequence_length} "
                    f"observations buffered. Ingest more data first."
                ),
            )

        start = time.perf_counter()
        try:
            result = eng.predict()
            latency = time.perf_counter() - start

            PREDICTION_LATENCY.observe(latency)
            ONNX_INFERENCE_LATENCY.observe(result["inference_ms"] / 1000)
            PREDICTION_COUNTER.labels(status="success").inc()
            PREDICTION_VALUE.set(result["prediction_original_scale"])

            return PredictionResponse(**result)
        except Exception as exc:
            PREDICTION_COUNTER.labels(status="error").inc()
            logger.error("Prediction failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction error: {exc}",
            )

    @app.post(
        "/predict/batch",
        tags=["inference"],
        response_model=BatchPredictionResponse,
    )
    def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
        """Batch prediction from pre-built feature sequences."""
        eng = _get_engine()

        if len(request.sequences) > settings.max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Batch size {len(request.sequences)} exceeds max {settings.max_batch_size}",
            )

        start = time.perf_counter()
        try:
            sequences = np.array(request.sequences, dtype=np.float32)
            result = eng.predict_from_sequence(sequences)
            latency = time.perf_counter() - start

            PREDICTION_LATENCY.observe(latency)
            PREDICTION_COUNTER.labels(status="success").inc()

            return BatchPredictionResponse(**result)
        except Exception as exc:
            PREDICTION_COUNTER.labels(status="error").inc()
            logger.error("Batch prediction failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch prediction error: {exc}",
            )

    @app.get("/model/info", tags=["system"])
    def model_info() -> dict[str, Any]:
        """Return model and engine metadata."""
        eng = _get_engine()
        return {
            "model_type": "lstm_onnx",
            "target_metric": eng.target_metric,
            "forecast_horizon_steps": eng.forecast_horizon,
            "forecast_horizon_minutes": eng.forecast_horizon * 5,
            "sequence_length": eng.sequence_length,
            "n_features": eng.n_features,
            "feature_columns": eng.feature_columns,
            "buffer_size": eng.buffer_size,
            "is_ready": eng.is_ready,
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


app = create_app()
