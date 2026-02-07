"""Async HTTP client for the predictor inference service.

Wraps /ingest, /predict, and /health endpoints with typed responses
and structured error handling for the control loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class PredictorUnavailableError(RuntimeError):
    """Raised when the predictor service cannot be reached or returns 5xx."""


class PredictorNotReadyError(RuntimeError):
    """Raised when the predictor returns 425 (buffer not filled)."""


@dataclass
class PredictionResult:
    """Typed wrapper around the predictor /predict response."""

    prediction_normalized: float
    prediction_original_scale: float
    target_metric: str
    forecast_horizon_steps: int
    forecast_horizon_minutes: int
    model_type: str
    buffer_size: int
    inference_ms: float


class PredictorClient:
    """Async client for the predictor FastAPI service.

    Args:
        base_url: Predictor service URL (e.g. ``http://predictor:8000``).
        timeout: Request timeout in seconds.
        client: Optional pre-built httpx.AsyncClient (for testing).
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
        )
        self._owns_client = client is None

    async def health(self) -> bool:
        """Check if the predictor is alive.

        Returns:
            True if /health returns 200.
        """
        try:
            resp = await self._client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def ingest(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Send a metric observation to the predictor buffer.

        Args:
            observation: Dict matching MetricObservation schema.

        Returns:
            Response payload (status, buffer_size, ready_for_prediction).

        Raises:
            PredictorUnavailableError: On connection or server errors.
        """
        try:
            resp = await self._client.post("/ingest", json=observation)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise PredictorUnavailableError(
                f"Failed to ingest observation: {exc}"
            ) from exc

    async def predict(self) -> PredictionResult:
        """Request a prediction from the buffered observations.

        Returns:
            Typed PredictionResult.

        Raises:
            PredictorNotReadyError: If buffer is not yet filled (425).
            PredictorUnavailableError: On connection or server errors.
        """
        try:
            resp = await self._client.post("/predict")

            if resp.status_code == 425:
                raise PredictorNotReadyError(
                    f"Predictor buffer not ready: {resp.json().get('detail', '')}"
                )

            resp.raise_for_status()
            data = resp.json()

            return PredictionResult(
                prediction_normalized=data["prediction_normalized"],
                prediction_original_scale=data["prediction_original_scale"],
                target_metric=data["target_metric"],
                forecast_horizon_steps=data["forecast_horizon_steps"],
                forecast_horizon_minutes=data["forecast_horizon_minutes"],
                model_type=data["model_type"],
                buffer_size=data["buffer_size"],
                inference_ms=data["inference_ms"],
            )
        except (PredictorNotReadyError, PredictorUnavailableError):
            raise
        except httpx.HTTPError as exc:
            raise PredictorUnavailableError(
                f"Failed to get prediction: {exc}"
            ) from exc

    async def close(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> PredictorClient:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()
