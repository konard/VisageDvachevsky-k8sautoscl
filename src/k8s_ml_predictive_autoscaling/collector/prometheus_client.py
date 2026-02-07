"""HTTP client for Prometheus query_range API."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

import httpx

from ..logging import get_logger

LOGGER = get_logger(__name__)


class PrometheusQueryError(RuntimeError):
    """Raised when Prometheus responds with an error payload."""


class PrometheusClient:
    """Thin wrapper around Prometheus HTTP API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 10.0,
        verify_ssl: bool = True,
        client: httpx.Client | None = None,
    ) -> None:
        self._client = client or httpx.Client(
            base_url=base_url,
            timeout=timeout_seconds,
            verify=verify_ssl,
        )
        self._owns_client = client is None

    def query(self, query: str) -> list[dict[str, Any]]:
        """Execute an instant query and return the raw result entries.

        This is useful for fetching the latest value of a metric
        without specifying a time range (e.g., in the planner control loop).
        """
        params = {"query": query}
        LOGGER.debug("Prometheus instant query", extra={"query": query})
        response = self._client.get("/api/v1/query", params=params)
        response.raise_for_status()
        payload = cast(dict[str, Any], response.json())
        if payload.get("status") != "success":
            raise PrometheusQueryError(payload.get("error", "unknown error"))
        data = cast(dict[str, Any], payload.get("data", {}))
        result = cast(list[dict[str, Any]], data.get("result", []))
        return result

    def query_range(
        self,
        query: str,
        *,
        start: datetime,
        end: datetime,
        step: timedelta,
    ) -> list[dict[str, Any]]:
        """Execute a query_range request and return the raw data entries."""

        params = {
            "query": query,
            "start": f"{start.timestamp():.3f}",
            "end": f"{end.timestamp():.3f}",
            "step": f"{int(step.total_seconds())}",
        }
        LOGGER.debug("Prometheus query", extra={"query": query, "params": params})
        response = self._client.get("/api/v1/query_range", params=params)
        response.raise_for_status()
        payload = cast(dict[str, Any], response.json())
        if payload.get("status") != "success":  # pragma: no cover - defensive
            raise PrometheusQueryError(payload.get("error", "unknown error"))
        data = cast(dict[str, Any], payload.get("data", {}))
        result = cast(list[dict[str, Any]], data.get("result", []))
        return result

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "PrometheusClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def to_utc(dt: datetime) -> datetime:
    """Ensure datetime carries timezone info for logging clarity."""

    return dt.astimezone(UTC)


__all__ = ["PrometheusClient", "PrometheusQueryError", "to_utc"]
