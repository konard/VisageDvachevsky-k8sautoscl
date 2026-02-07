"""Prometheus helpers shared by demo applications."""

from prometheus_client import Counter, Gauge, Histogram

REQUEST_LATENCY = Histogram(
    "demo_service_request_latency_seconds",
    "Latency of synthetic operations in the demo service.",
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5),
)
REQUEST_COUNTER = Counter(
    "demo_service_requests_total",
    "Number of inference-like requests served by the demo service.",
    labelnames=("endpoint", "method", "status"),
)
ACTIVE_JOBS = Gauge(
    "demo_service_active_jobs",
    "Synthetic gauge representing amount of queued work.",
)


__all__ = ["REQUEST_LATENCY", "REQUEST_COUNTER", "ACTIVE_JOBS"]
