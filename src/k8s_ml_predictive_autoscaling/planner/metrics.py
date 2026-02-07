"""Prometheus metrics for the planner service.

Exposes operational counters and gauges consumed by Grafana dashboards:
  - Scaling decisions and execution counts
  - Predicted CPU and desired replica gauges
  - Control loop timing and error rates
  - Degradation state
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# -- Scaling decisions -------------------------------------------------------

SCALING_DECISIONS_TOTAL = Counter(
    "planner_scaling_decisions_total",
    "Total scaling decisions by direction",
    ["direction"],  # up, down, hold
)

SCALING_EXECUTIONS_TOTAL = Counter(
    "planner_scaling_executions_total",
    "Actual scaling actions applied (subset of decisions)",
    ["direction"],  # up, down
)

# -- Current state -----------------------------------------------------------

PREDICTED_CPU = Gauge(
    "planner_predicted_cpu_percent",
    "Latest smoothed CPU prediction from the ML model",
)

DESIRED_REPLICAS = Gauge(
    "planner_desired_replicas",
    "Desired replica count from the latest decision",
)

CURRENT_REPLICAS = Gauge(
    "planner_current_replicas",
    "Current replica count as reported by the executor",
)

# -- Control loop ------------------------------------------------------------

LOOP_DURATION = Histogram(
    "planner_loop_duration_seconds",
    "Duration of a single control loop iteration",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)

LOOP_ERRORS_TOTAL = Counter(
    "planner_loop_errors_total",
    "Errors encountered during control loop iterations",
    ["phase"],  # fetch, ingest, predict, decide, execute
)

# -- Resilience --------------------------------------------------------------

CONSECUTIVE_FAILURES = Gauge(
    "planner_consecutive_failures",
    "Number of consecutive control loop failures",
)

DEGRADED = Gauge(
    "planner_degraded",
    "Whether the planner is in degraded mode (1=degraded, 0=healthy)",
)

FALLBACK_TOTAL = Counter(
    "planner_fallback_total",
    "Times the planner fell back to holding current replicas",
)

# -- Cooldowns ---------------------------------------------------------------

COOLDOWN_ACTIVE = Gauge(
    "planner_cooldown_active",
    "Whether a cooldown is currently blocking scaling (per direction)",
    ["direction"],  # up, down
)
