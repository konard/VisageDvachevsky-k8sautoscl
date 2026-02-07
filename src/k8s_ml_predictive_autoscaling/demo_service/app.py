"""Demo FastAPI service with metrics endpoints."""

from __future__ import annotations

import random
import secrets
import time

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from ..logging import get_logger, log_structured
from ..metrics import ACTIVE_JOBS, REQUEST_COUNTER, REQUEST_LATENCY
from ..settings import Settings, get_settings

LOGGER = get_logger(__name__)


class SyntheticWorkload(BaseModel):
    payload_size: int = 128
    cpu_hint: float = 0.05


def create_app(settings: Settings) -> FastAPI:
    """Factory for the demo FastAPI application."""

    if not settings.api_token:
        msg = (
            "AUTOSCALER_API_TOKEN must be configured to start the demo service. "
            "Set it via environment variables or .env."
        )
        raise RuntimeError(msg)

    app = FastAPI(title=settings.service_name, version="0.1.0")
    token_value = settings.api_token.get_secret_value()

    @app.get("/health", tags=["system"], status_code=status.HTTP_200_OK)
    def health() -> dict[str, str]:
        log_structured(LOGGER, "health", status="ok")
        return {"status": "ok"}

    @app.post(
        "/workload",
        tags=["workload"],
        status_code=status.HTTP_202_ACCEPTED,
    )
    def handle_workload(
        body: SyntheticWorkload,
        api_key: str | None = Header(default=None, alias=settings.api_key_header),
    ) -> dict[str, str | int]:
        if not api_key or not secrets.compare_digest(api_key, token_value):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API token",
                headers={"WWW-Authenticate": "API-Key"},
            )
        start = time.perf_counter()
        simulated_latency = max(random.gauss(body.cpu_hint, 0.01), 0.005)
        time.sleep(simulated_latency)

        REQUEST_LATENCY.observe(time.perf_counter() - start)
        ACTIVE_JOBS.inc()
        ACTIVE_JOBS.dec()
        REQUEST_COUNTER.labels(endpoint="/workload", method="POST", status=202).inc()

        log_structured(
            LOGGER,
            "synthetic workload accepted",
            payload_size=body.payload_size,
            cpu_hint=body.cpu_hint,
        )
        return {"status": "queued", "payload_size": body.payload_size}

    @app.get(settings.metrics_path, tags=["system"], response_class=PlainTextResponse)
    def metrics() -> PlainTextResponse:
        data = generate_latest()
        return PlainTextResponse(content=data, media_type=CONTENT_TYPE_LATEST)

    return app


def get_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    return create_app(resolved)


app = create_app(get_settings())


def run() -> None:  # pragma: no cover - thin wrapper for uvicorn
    import uvicorn

    uvicorn.run(
        "k8s_ml_predictive_autoscaling.demo_service.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
