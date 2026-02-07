"""Async Prometheus feeder -- fetches metrics and builds MetricObservation dicts.

Responsibilities:
  1. Instant PromQL queries for latest metric values.
  2. Range query for historical backfill on startup.
  3. Transforms raw Prometheus data into the dict format expected by
     the predictor /ingest endpoint.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class PrometheusQueryError(RuntimeError):
    """Raised when Prometheus responds with an error or unexpected shape."""


@dataclass
class MetricSnapshot:
    """A single metric observation assembled from Prometheus queries."""

    cpu_usage: float
    mem_util_percent: float
    net_in: float
    net_out: float
    disk_io_percent: float
    timestamp: float  # unix epoch


# Default PromQL expressions for each metric
DEFAULT_QUERIES: dict[str, str] = {
    "cpu_usage": (
        '100 * (1 - avg(rate(process_cpu_seconds_total{job="demo-services"}[1m])))'
    ),
    "mem_util_percent": (
        'avg(process_resident_memory_bytes{job="demo-services"}) '
        "/ 1073741824 * 100"  # approx % of 1 GB
    ),
    "net_in": 'sum(rate(http_requests_total{job="demo-services"}[1m]))',
    "net_out": 'sum(rate(http_request_duration_seconds_count{job="demo-services"}[1m]))',
    "disk_io_percent": "0",  # placeholder -- demo service has no disk metric
}


class PrometheusFeeder:
    """Builds metric observations from live Prometheus data.

    Args:
        base_url: Prometheus server URL.
        timeout: HTTP request timeout seconds.
        queries: Override default PromQL per metric.
        client: Optional pre-built httpx.AsyncClient.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        queries: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
        )
        self._owns_client = client is None
        self._queries = queries or dict(DEFAULT_QUERIES)

    # ------------------------------------------------------------------
    # Low-level query helpers
    # ------------------------------------------------------------------

    async def _instant_query(self, promql: str) -> float:
        """Execute an instant query and return a single scalar value."""
        resp = await self._client.get(
            "/api/v1/query", params={"query": promql}
        )
        resp.raise_for_status()
        payload = resp.json()

        if payload.get("status") != "success":
            raise PrometheusQueryError(
                f"Prometheus error: {payload.get('error', 'unknown')}"
            )

        results = payload.get("data", {}).get("result", [])
        if not results:
            logger.debug("No results for query: %s", promql)
            return 0.0

        # Extract scalar value from first time series
        value = results[0].get("value", [0, "0"])
        try:
            return float(value[1])
        except (IndexError, ValueError, TypeError):
            return 0.0

    async def _range_query(
        self, promql: str, start: float, end: float, step: int = 60
    ) -> list[tuple[float, float]]:
        """Execute a range query and return list of (timestamp, value)."""
        resp = await self._client.get(
            "/api/v1/query_range",
            params={
                "query": promql,
                "start": f"{start:.3f}",
                "end": f"{end:.3f}",
                "step": str(step),
            },
        )
        resp.raise_for_status()
        payload = resp.json()

        if payload.get("status") != "success":
            raise PrometheusQueryError(
                f"Prometheus range error: {payload.get('error', 'unknown')}"
            )

        results = payload.get("data", {}).get("result", [])
        if not results:
            return []

        points: list[tuple[float, float]] = []
        for ts, val in results[0].get("values", []):
            try:
                points.append((float(ts), float(val)))
            except (ValueError, TypeError):
                continue
        return points

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_latest(self) -> MetricSnapshot:
        """Fetch latest values for all metrics via instant queries.

        Returns:
            MetricSnapshot with current metric values.
        """
        values: dict[str, float] = {}
        for metric_name, promql in self._queries.items():
            try:
                values[metric_name] = await self._instant_query(promql)
            except (httpx.HTTPError, PrometheusQueryError) as exc:
                logger.warning("Failed to query %s: %s", metric_name, exc)
                values[metric_name] = 0.0

        return MetricSnapshot(
            cpu_usage=values.get("cpu_usage", 0.0),
            mem_util_percent=values.get("mem_util_percent", 0.0),
            net_in=values.get("net_in", 0.0),
            net_out=values.get("net_out", 0.0),
            disk_io_percent=values.get("disk_io_percent", 0.0),
            timestamp=time.time(),
        )

    async def fetch_historical(
        self,
        duration_minutes: int = 120,
        step_seconds: int = 300,
    ) -> list[dict[str, Any]]:
        """Fetch historical data for backfilling the predictor buffer.

        Performs range queries for each metric, aligns them by timestamp,
        and returns a list of dicts suitable for POST /ingest.

        Args:
            duration_minutes: How far back to query.
            step_seconds: Resolution in seconds (300 = 5 min).

        Returns:
            List of observation dicts ordered by time.
        """
        end = time.time()
        start = end - duration_minutes * 60

        # Fetch all metrics in parallel-ish (sequential for simplicity)
        series: dict[str, list[tuple[float, float]]] = {}
        for metric_name, promql in self._queries.items():
            try:
                series[metric_name] = await self._range_query(
                    promql, start, end, step_seconds
                )
            except (httpx.HTTPError, PrometheusQueryError) as exc:
                logger.warning(
                    "Failed to fetch historical %s: %s", metric_name, exc
                )
                series[metric_name] = []

        # Use cpu_usage timestamps as the alignment reference
        cpu_series = series.get("cpu_usage", [])
        if not cpu_series:
            logger.warning("No historical CPU data from Prometheus")
            return []

        # Build lookup tables for each metric (timestamp -> value)
        lookups: dict[str, dict[float, float]] = {}
        for name, points in series.items():
            lookups[name] = {ts: val for ts, val in points}

        observations: list[dict[str, Any]] = []
        for ts, cpu_val in cpu_series:
            dt = datetime.fromtimestamp(ts)
            obs = {
                "cpu_usage": cpu_val,
                "mem_util_percent": lookups.get("mem_util_percent", {}).get(ts, 0.0),
                "net_in": lookups.get("net_in", {}).get(ts, 0.0),
                "net_out": lookups.get("net_out", {}).get(ts, 0.0),
                "disk_io_percent": lookups.get("disk_io_percent", {}).get(ts, 0.0),
                "hour": dt.hour,
                "day_of_week": dt.weekday(),
                "is_weekend": 1 if dt.weekday() >= 5 else 0,
                "minute_of_day": dt.hour * 60 + dt.minute,
            }
            observations.append(obs)

        logger.info(
            "Fetched %d historical observations (%.0f min, step %ds)",
            len(observations),
            duration_minutes,
            step_seconds,
        )
        return observations

    @staticmethod
    def snapshot_to_observation(snapshot: MetricSnapshot) -> dict[str, Any]:
        """Convert a MetricSnapshot to a dict for POST /ingest.

        Adds time-based features from the snapshot timestamp.
        """
        dt = datetime.fromtimestamp(snapshot.timestamp)
        return {
            "cpu_usage": snapshot.cpu_usage,
            "mem_util_percent": snapshot.mem_util_percent,
            "net_in": snapshot.net_in,
            "net_out": snapshot.net_out,
            "disk_io_percent": snapshot.disk_io_percent,
            "hour": dt.hour,
            "day_of_week": dt.weekday(),
            "is_weekend": 1 if dt.weekday() >= 5 else 0,
            "minute_of_day": dt.hour * 60 + dt.minute,
        }

    async def close(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> PrometheusFeeder:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()
