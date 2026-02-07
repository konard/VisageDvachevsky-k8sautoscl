"""Smoke tests for the demo FastAPI application."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute
from pydantic import SecretStr

os.environ.setdefault("AUTOSCALER_API_TOKEN", "unit-test-token")

from k8s_ml_predictive_autoscaling.demo_service.app import (  # noqa: E402
    SyntheticWorkload,
    create_app,
)
from k8s_ml_predictive_autoscaling.settings import Settings  # noqa: E402

SECRET_TOKEN = SecretStr("unit-test-token")
SETTINGS = Settings(api_token=SECRET_TOKEN)
APP = create_app(SETTINGS)


def _get_endpoint(path: str) -> Callable[..., Any]:
    for route in APP.routes:
        if isinstance(route, APIRoute) and route.path == path:
            return route.endpoint
    raise AssertionError(f"Route {path} not found")


def test_health_endpoint_returns_ok() -> None:
    handler = _get_endpoint("/health")
    result = handler()
    assert result == {"status": "ok"}


def test_metrics_endpoint_exposes_prometheus_text() -> None:
    handler = _get_endpoint(SETTINGS.metrics_path)
    response = handler()
    assert response.status_code == 200
    assert "demo_service_requests_total" in response.body.decode()


def test_workload_endpoint_requires_api_key() -> None:
    handler = _get_endpoint("/workload")
    with pytest.raises(HTTPException):
        handler(SyntheticWorkload(payload_size=32, cpu_hint=0.05), api_key=None)


def test_workload_endpoint_accepts_valid_api_key() -> None:
    handler = _get_endpoint("/workload")
    token = SECRET_TOKEN.get_secret_value()
    response = handler(
        SyntheticWorkload(payload_size=64, cpu_hint=0.05),
        api_key=token,
    )
    assert response["status"] == "queued"


def test_create_app_requires_token() -> None:
    settings = Settings(api_token=None)
    with pytest.raises(RuntimeError):
        create_app(settings)
