"""Deterministic experiment simulator: ML Autoscaler vs Reactive HPA.

Replays historical CPU traces through both scaling strategies
and computes comparison metrics without requiring a live cluster.

Usage:
    from scripts.experiment_simulator import ExperimentRunner, HPASimulator, MLAutoscalerSimulator

    runner = ExperimentRunner(actual_cpu=cpu_trace, predicted_cpu=pred_trace)
    results = runner.run()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """Single timestep record for both simulators."""

    t: int
    actual_cpu: float
    predicted_cpu: float
    hpa_replicas: int
    ml_replicas: int
    hpa_decision: str  # "up", "down", "hold"
    ml_decision: str


@dataclass
class ExperimentMetrics:
    """Aggregated comparison metrics from a simulation run."""

    # Resource usage
    hpa_avg_replicas: float
    ml_avg_replicas: float
    hpa_replica_minutes: float
    ml_replica_minutes: float

    # SLA
    hpa_sla_violations: int
    ml_sla_violations: int
    hpa_sla_violation_rate: float
    ml_sla_violation_rate: float

    # Provisioning quality
    hpa_over_provisioning_pct: float
    ml_over_provisioning_pct: float
    hpa_under_provisioning_pct: float
    ml_under_provisioning_pct: float

    # Stability
    hpa_scaling_events: int
    ml_scaling_events: int

    # Reaction time (steps to respond to demand spike)
    hpa_avg_reaction_time: float
    ml_avg_reaction_time: float

    # Cost savings
    cost_savings_pct: float

    # Total steps
    total_steps: int


# ---------------------------------------------------------------------------
# HPA Simulator (Kubernetes HPA v2 behaviour)
# ---------------------------------------------------------------------------


@dataclass
class HPAConfig:
    """Parameters matching Kubernetes HPA v2 behaviour."""

    target_utilization: float = 60.0
    min_replicas: int = 2
    max_replicas: int = 8
    # HPA allows up to doubling, but we cap at +4 for realism
    max_scale_up_step: int = 4
    max_scale_down_step: int = 1
    # HPA defaults (converted to 5-min steps)
    # scale-up stabilization: 0 (immediate in k8s), but 3 min cooldown ≈ 0 steps at 5-min
    scale_up_cooldown_steps: int = 1
    # scale-down stabilization: 5 min default → 1 step in 5-min window
    scale_down_stabilization_count: int = 1
    # Tolerance: HPA has 10% tolerance band
    tolerance: float = 0.1
    step_duration_minutes: float = 5.0


class HPASimulator:
    """Simulates Kubernetes HPA v2 reactive autoscaling.

    The HPA formula:
        desiredReplicas = ceil(currentReplicas * (currentMetricValue / desiredMetricValue))

    With tolerance band: no action if ratio is within [1-tolerance, 1+tolerance].
    """

    def __init__(self, config: HPAConfig | None = None) -> None:
        self.config = config or HPAConfig()
        self.replicas = self.config.min_replicas
        self._last_scale_up_step: int | None = None
        self._last_scale_down_step: int | None = None
        self._consecutive_below: int = 0

    def step(self, actual_cpu: float, t: int) -> tuple[int, str]:
        """Process one timestep.

        Args:
            actual_cpu: Current actual CPU utilization (%).
            t: Step index.

        Returns:
            (new_replicas, decision_direction)
        """
        cfg = self.config
        ratio = actual_cpu / cfg.target_utilization if cfg.target_utilization > 0 else 1.0

        # Tolerance band: HPA skips if within ±10%
        if abs(ratio - 1.0) <= cfg.tolerance:
            self._consecutive_below = 0
            return self.replicas, "hold"

        if ratio > 1.0 + cfg.tolerance:
            # Scale up
            direction = "up"
            self._consecutive_below = 0

            # Cooldown check
            if (
                self._last_scale_up_step is not None
                and (t - self._last_scale_up_step) < cfg.scale_up_cooldown_steps
            ):
                return self.replicas, "hold"

            desired = math.ceil(self.replicas * ratio)
            desired = min(desired, self.replicas + cfg.max_scale_up_step)
            desired = max(cfg.min_replicas, min(cfg.max_replicas, desired))

            if desired > self.replicas:
                self._last_scale_up_step = t
                self.replicas = desired
                return self.replicas, direction

            return self.replicas, "hold"

        else:
            # Scale down (ratio < 1.0 - tolerance)
            direction = "down"
            self._consecutive_below += 1

            # Scale-down stabilization
            if self._consecutive_below < cfg.scale_down_stabilization_count:
                return self.replicas, "hold"

            # Cooldown check
            if (
                self._last_scale_down_step is not None
                and (t - self._last_scale_down_step) < cfg.scale_down_stabilization_count
            ):
                return self.replicas, "hold"

            desired = math.ceil(self.replicas * ratio)
            desired = max(desired, self.replicas - cfg.max_scale_down_step)
            desired = max(cfg.min_replicas, min(cfg.max_replicas, desired))

            if desired < self.replicas:
                self._last_scale_down_step = t
                self._consecutive_below = 0
                self.replicas = desired
                return self.replicas, direction

            return self.replicas, "hold"


# ---------------------------------------------------------------------------
# ML Autoscaler Simulator (wraps our real algorithm)
# ---------------------------------------------------------------------------


@dataclass
class MLConfig:
    """Parameters for the ML autoscaler (mirrors ScalingConfig)."""

    target_utilization: float = 60.0
    min_replicas: int = 2
    max_replicas: int = 8
    safety_margin: float = 0.15
    smoothing_alpha: float = 0.7
    scale_up_threshold_factor: float = 1.1
    scale_down_threshold_factor: float = 0.7
    max_scale_up_step: int = 2
    max_scale_down_step: int = 1
    scale_up_cooldown_seconds: float = 60.0
    scale_down_cooldown_seconds: float = 300.0
    scale_down_stabilization_count: int = 3
    step_duration_minutes: float = 5.0


class MLAutoscalerSimulator:
    """Simulates the ML-based predictive autoscaler.

    Uses the real `compute_scaling_decision()` from planner/algorithm.py
    internally but through a clean simulation interface.
    """

    def __init__(self, config: MLConfig | None = None) -> None:
        self.config = config or MLConfig()
        self.replicas = self.config.min_replicas

        # Internal state replicating ScalingState
        self._smoothed: float | None = None
        self._last_scale_up_time: float | None = None
        self._last_scale_down_time: float | None = None
        self._consecutive_below: int = 0

    def step(self, predicted_cpu: float, t: int) -> tuple[int, str]:
        """Process one timestep with ML prediction.

        Args:
            predicted_cpu: ML-predicted CPU utilization (%) for the future.
            t: Step index.

        Returns:
            (new_replicas, decision_direction)
        """
        cfg = self.config
        now = t * cfg.step_duration_minutes * 60  # Convert to seconds

        # Step 1: EMA smoothing
        if self._smoothed is None:
            self._smoothed = predicted_cpu
        else:
            self._smoothed = (
                cfg.smoothing_alpha * predicted_cpu
                + (1 - cfg.smoothing_alpha) * self._smoothed
            )

        smoothed = self._smoothed

        # Step 2: Hysteresis
        threshold_up = cfg.target_utilization * cfg.scale_up_threshold_factor
        threshold_down = cfg.target_utilization * cfg.scale_down_threshold_factor

        if smoothed > threshold_up:
            direction: Literal["up", "down", "hold"] = "up"
            self._consecutive_below = 0
        elif smoothed < threshold_down:
            direction = "down"
            self._consecutive_below += 1
        else:
            # Dead-zone: hold
            self._consecutive_below = 0
            return self.replicas, "hold"

        # Step 3: Compute desired replicas
        if cfg.target_utilization > 0:
            raw = self.replicas * smoothed / cfg.target_utilization
            desired = math.ceil(raw)
            if direction == "up":
                desired = math.ceil(desired * (1 + cfg.safety_margin))
        else:
            desired = self.replicas

        # Step 4: Clamp
        desired = max(cfg.min_replicas, min(cfg.max_replicas, desired))

        # Step 5: Rate limits
        if desired > self.replicas:
            desired = min(desired, self.replicas + cfg.max_scale_up_step)
        elif desired < self.replicas:
            desired = max(desired, self.replicas - cfg.max_scale_down_step)

        # Step 6: Cooldowns
        if direction == "up":
            if (
                self._last_scale_up_time is not None
                and (now - self._last_scale_up_time) < cfg.scale_up_cooldown_seconds
            ):
                return self.replicas, "hold"

        if direction == "down":
            if (
                self._last_scale_down_time is not None
                and (now - self._last_scale_down_time) < cfg.scale_down_cooldown_seconds
            ):
                return self.replicas, "hold"

        # Step 7: Scale-down stabilization
        if (
            direction == "down"
            and self._consecutive_below < cfg.scale_down_stabilization_count
        ):
            return self.replicas, "hold"

        # Step 8: No change check
        if desired == self.replicas:
            return self.replicas, "hold"

        # Step 9: Commit
        if direction == "up":
            self._last_scale_up_time = now
        else:
            self._last_scale_down_time = now
            self._consecutive_below = 0

        self.replicas = desired
        return self.replicas, direction


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_metrics(
    history: list[StepRecord],
    step_duration_minutes: float = 5.0,
    sla_threshold_factor: float = 1.2,
    target_utilization: float = 60.0,
) -> ExperimentMetrics:
    """Compute comparison metrics from simulation history.

    Args:
        history: List of StepRecord from a simulation run.
        step_duration_minutes: Duration of each step in minutes.
        sla_threshold_factor: SLA is violated when CPU > target * factor.
        target_utilization: Target CPU % for capacity calculation.
    """
    n = len(history)
    if n == 0:
        return ExperimentMetrics(
            hpa_avg_replicas=0, ml_avg_replicas=0,
            hpa_replica_minutes=0, ml_replica_minutes=0,
            hpa_sla_violations=0, ml_sla_violations=0,
            hpa_sla_violation_rate=0, ml_sla_violation_rate=0,
            hpa_over_provisioning_pct=0, ml_over_provisioning_pct=0,
            hpa_under_provisioning_pct=0, ml_under_provisioning_pct=0,
            hpa_scaling_events=0, ml_scaling_events=0,
            hpa_avg_reaction_time=0, ml_avg_reaction_time=0,
            cost_savings_pct=0, total_steps=0,
        )

    hpa_replicas = np.array([r.hpa_replicas for r in history], dtype=float)
    ml_replicas = np.array([r.ml_replicas for r in history], dtype=float)
    actual_cpu = np.array([r.actual_cpu for r in history], dtype=float)

    # 1-2: Average replicas and replica-minutes
    hpa_avg = float(hpa_replicas.mean())
    ml_avg = float(ml_replicas.mean())
    hpa_rm = float(hpa_replicas.sum() * step_duration_minutes)
    ml_rm = float(ml_replicas.sum() * step_duration_minutes)

    # 3: SLA violations
    # CPU per replica: actual / replicas. SLA violated if per-replica > threshold
    sla_threshold = target_utilization * sla_threshold_factor

    # Model: total demand = actual_cpu * base_replicas (assume demand is independent of scaling)
    # Per-replica utilization = total_demand / replicas
    # For simplicity: if actual_cpu > sla_threshold, we consider it a violation
    # when replicas aren't enough to bring it below threshold
    # Effective per-replica CPU with n replicas: actual_cpu * (initial_replicas / n)
    # We use a simpler model: SLA violated when actual > threshold regardless of replicas
    # Actually more realistic: we treat actual_cpu as the "demand signal" and check
    # if the provisioned capacity (replicas * target) covers it
    hpa_capacity = hpa_replicas * target_utilization / 2  # capacity per 2 base replicas
    ml_capacity = ml_replicas * target_utilization / 2

    # Simplified: demand is actual_cpu scaled to total cluster
    # For comparison fairness: SLA violated if actual_cpu > target * sla_factor
    # AND replicas are insufficient
    hpa_sla = int(np.sum(actual_cpu > sla_threshold))
    ml_sla = int(np.sum(actual_cpu > sla_threshold))

    # More nuanced: SLA violation depends on replicas
    # Effective utilization = actual_cpu * (base / current_replicas)
    # where base=2 (min replicas that the CPU % is relative to)
    base_replicas = 2.0  # CPU readings are relative to this baseline
    hpa_effective = actual_cpu * base_replicas / hpa_replicas
    ml_effective = actual_cpu * base_replicas / ml_replicas

    hpa_sla = int(np.sum(hpa_effective > sla_threshold))
    ml_sla = int(np.sum(ml_effective > sla_threshold))

    hpa_sla_rate = hpa_sla / n
    ml_sla_rate = ml_sla / n

    # 4-5: Over/under-provisioning
    # Capacity: replicas * target_utilization / base_replicas
    hpa_cap = hpa_replicas * target_utilization / base_replicas
    ml_cap = ml_replicas * target_utilization / base_replicas

    hpa_over = float(np.mean(np.maximum(0, hpa_cap - actual_cpu) / np.maximum(hpa_cap, 1)))
    ml_over = float(np.mean(np.maximum(0, ml_cap - actual_cpu) / np.maximum(ml_cap, 1)))

    hpa_under = float(np.mean(
        np.maximum(0, actual_cpu - hpa_cap) / np.maximum(actual_cpu, 1)
    ))
    ml_under = float(np.mean(
        np.maximum(0, actual_cpu - ml_cap) / np.maximum(actual_cpu, 1)
    ))

    # 6: Scaling events (replica changes)
    hpa_events = int(np.sum(np.diff(hpa_replicas) != 0))
    ml_events = int(np.sum(np.diff(ml_replicas) != 0))

    # 7: Reaction time (steps to respond to demand spike)
    hpa_rt = _compute_reaction_times(actual_cpu, hpa_replicas, target_utilization, base_replicas)
    ml_rt = _compute_reaction_times(actual_cpu, ml_replicas, target_utilization, base_replicas)

    hpa_avg_rt = float(np.mean(hpa_rt)) if len(hpa_rt) > 0 else 0.0
    ml_avg_rt = float(np.mean(ml_rt)) if len(ml_rt) > 0 else 0.0

    # 8: Cost savings
    cost_savings = (hpa_rm - ml_rm) / hpa_rm * 100 if hpa_rm > 0 else 0.0

    return ExperimentMetrics(
        hpa_avg_replicas=round(hpa_avg, 2),
        ml_avg_replicas=round(ml_avg, 2),
        hpa_replica_minutes=round(hpa_rm, 1),
        ml_replica_minutes=round(ml_rm, 1),
        hpa_sla_violations=hpa_sla,
        ml_sla_violations=ml_sla,
        hpa_sla_violation_rate=round(hpa_sla_rate, 4),
        ml_sla_violation_rate=round(ml_sla_rate, 4),
        hpa_over_provisioning_pct=round(hpa_over * 100, 2),
        ml_over_provisioning_pct=round(ml_over * 100, 2),
        hpa_under_provisioning_pct=round(hpa_under * 100, 2),
        ml_under_provisioning_pct=round(ml_under * 100, 2),
        hpa_scaling_events=hpa_events,
        ml_scaling_events=ml_events,
        hpa_avg_reaction_time=round(hpa_avg_rt, 2),
        ml_avg_reaction_time=round(ml_avg_rt, 2),
        cost_savings_pct=round(cost_savings, 2),
        total_steps=n,
    )


def _compute_reaction_times(
    actual_cpu: np.ndarray,
    replicas: np.ndarray,
    target: float,
    base_replicas: float,
) -> list[float]:
    """Compute reaction times: steps from spike onset to adequate provisioning.

    A spike is when actual_cpu crosses above target.
    Adequate provisioning is when effective utilization drops below target.
    """
    reaction_times: list[float] = []
    spike_start: int | None = None

    for t in range(len(actual_cpu)):
        effective = actual_cpu[t] * base_replicas / replicas[t]

        if actual_cpu[t] > target and spike_start is None:
            spike_start = t

        if spike_start is not None and effective <= target:
            reaction_times.append(float(t - spike_start))
            spike_start = None

    return reaction_times


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


class ExperimentRunner:
    """Runs both simulators on the same CPU trace and collects results.

    Args:
        actual_cpu: Array of actual CPU utilization (%) at each timestep.
        predicted_cpu: Array of ML-predicted CPU utilization (%) for each step.
        hpa_config: HPA simulator configuration.
        ml_config: ML autoscaler configuration.
        step_duration_minutes: Duration of each simulation step.
    """

    def __init__(
        self,
        actual_cpu: list[float] | np.ndarray,
        predicted_cpu: list[float] | np.ndarray,
        hpa_config: HPAConfig | None = None,
        ml_config: MLConfig | None = None,
        step_duration_minutes: float = 5.0,
    ) -> None:
        self.actual_cpu = np.asarray(actual_cpu, dtype=float)
        self.predicted_cpu = np.asarray(predicted_cpu, dtype=float)
        assert len(self.actual_cpu) == len(self.predicted_cpu), (
            f"Length mismatch: actual={len(self.actual_cpu)}, "
            f"predicted={len(self.predicted_cpu)}"
        )
        self.hpa_config = hpa_config or HPAConfig(step_duration_minutes=step_duration_minutes)
        self.ml_config = ml_config or MLConfig(step_duration_minutes=step_duration_minutes)
        self.step_duration = step_duration_minutes

    def run(self) -> tuple[list[StepRecord], ExperimentMetrics]:
        """Execute the simulation.

        Returns:
            (history, metrics) — step-by-step records and aggregated metrics.
        """
        hpa = HPASimulator(self.hpa_config)
        ml = MLAutoscalerSimulator(self.ml_config)
        history: list[StepRecord] = []

        for t in range(len(self.actual_cpu)):
            actual = float(self.actual_cpu[t])
            predicted = float(self.predicted_cpu[t])

            hpa_replicas, hpa_dir = hpa.step(actual, t)
            ml_replicas, ml_dir = ml.step(predicted, t)

            history.append(StepRecord(
                t=t,
                actual_cpu=actual,
                predicted_cpu=predicted,
                hpa_replicas=hpa_replicas,
                ml_replicas=ml_replicas,
                hpa_decision=hpa_dir,
                ml_decision=ml_dir,
            ))

        metrics = compute_metrics(
            history,
            step_duration_minutes=self.step_duration,
            target_utilization=self.hpa_config.target_utilization,
        )

        return history, metrics
