"""Pure scaling algorithm -- no I/O, fully testable.

Implements the predictive autoscaling decision logic:
  1. EMA smoothing of predictions
  2. Hysteresis dead-zone to prevent oscillation
  3. Replica computation with safety margin
  4. Rate limiting (+2/-1 per cycle)
  5. Cooldown enforcement
  6. Scale-down stabilization
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ScalingConfig:
    """Subset of PlannerSettings needed by the algorithm."""

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


@dataclass
class ScalingState:
    """Mutable state persisted across control-loop cycles."""

    current_replicas: int = 2
    previous_smoothed_prediction: float | None = None
    last_scale_up_time: float | None = None
    last_scale_down_time: float | None = None
    consecutive_below_threshold: int = 0
    consecutive_failures: int = 0


@dataclass
class ScalingDecision:
    """Result of a single scaling computation."""

    desired_replicas: int
    direction: Literal["up", "down", "hold"]
    reason: str
    predicted_cpu: float
    smoothed_cpu: float
    should_execute: bool


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def apply_smoothing(
    current: float, previous: float | None, alpha: float
) -> float:
    """Exponential moving average.

    Args:
        current: New observed/predicted value.
        previous: Previous smoothed value (None on first call).
        alpha: Weight for the new value (0..1).
    """
    if previous is None:
        return current
    return alpha * current + (1.0 - alpha) * previous


def compute_desired_replicas(
    current_replicas: int,
    smoothed_cpu: float,
    target_utilization: float,
    safety_margin: float,
    direction: Literal["up", "down"],
) -> int:
    """Compute raw desired replicas from utilization ratio.

    Formula: ceil(current * smoothed / target) * (1 + margin for scale-up).
    """
    if target_utilization <= 0:
        return current_replicas

    raw = current_replicas * smoothed_cpu / target_utilization
    desired = math.ceil(raw)

    # Safety margin only for scale-up
    if direction == "up":
        desired = math.ceil(desired * (1.0 + safety_margin))

    return desired


def apply_rate_limits(
    current: int, desired: int, max_up: int, max_down: int
) -> int:
    """Clamp the change per cycle to avoid large jumps."""
    if desired > current:
        return min(desired, current + max_up)
    if desired < current:
        return max(desired, current - max_down)
    return desired


def check_cooldown(
    last_event_time: float | None,
    cooldown_seconds: float,
    now: float,
) -> bool:
    """Return True if still in cooldown (action blocked)."""
    if last_event_time is None:
        return False
    return (now - last_event_time) < cooldown_seconds


# ---------------------------------------------------------------------------
# Main decision function
# ---------------------------------------------------------------------------


def compute_scaling_decision(
    predicted_cpu: float,
    state: ScalingState,
    config: ScalingConfig,
    now: float,
) -> ScalingDecision:
    """Compute a full scaling decision.

    This is the core algorithm. It mutates ``state`` in-place
    (smoothed prediction, cooldown counters, stabilization counters).

    Args:
        predicted_cpu: Raw prediction from the ML model (CPU %).
        state: Mutable scaling state.
        config: Algorithm parameters.
        now: Current monotonic time (injectable for testing).

    Returns:
        A ScalingDecision indicating what to do.
    """

    # -- Step 1: EMA smoothing -------------------------------------------------
    smoothed = apply_smoothing(
        predicted_cpu, state.previous_smoothed_prediction, config.smoothing_alpha
    )
    state.previous_smoothed_prediction = smoothed

    # -- Step 2: Hysteresis -- determine direction -----------------------------
    threshold_up = config.target_utilization * config.scale_up_threshold_factor
    threshold_down = config.target_utilization * config.scale_down_threshold_factor

    if smoothed > threshold_up:
        direction: Literal["up", "down", "hold"] = "up"
        state.consecutive_below_threshold = 0
    elif smoothed < threshold_down:
        direction = "down"
        state.consecutive_below_threshold += 1
    else:
        # Inside dead-zone -- hold
        state.consecutive_below_threshold = 0
        return ScalingDecision(
            desired_replicas=state.current_replicas,
            direction="hold",
            reason=f"in dead-zone [{threshold_down:.1f}%, {threshold_up:.1f}%]",
            predicted_cpu=predicted_cpu,
            smoothed_cpu=smoothed,
            should_execute=False,
        )

    # -- Step 3: Compute desired replicas --------------------------------------
    desired = compute_desired_replicas(
        current_replicas=state.current_replicas,
        smoothed_cpu=smoothed,
        target_utilization=config.target_utilization,
        safety_margin=config.safety_margin,
        direction=direction,
    )

    # -- Step 4: Clamp to [min, max] -------------------------------------------
    desired = max(config.min_replicas, min(config.max_replicas, desired))

    # -- Step 5: Rate limits ---------------------------------------------------
    desired = apply_rate_limits(
        state.current_replicas,
        desired,
        config.max_scale_up_step,
        config.max_scale_down_step,
    )

    # -- Step 6: Check cooldowns -----------------------------------------------
    if direction == "up" and check_cooldown(
        state.last_scale_up_time, config.scale_up_cooldown_seconds, now
    ):
        return ScalingDecision(
            desired_replicas=state.current_replicas,
            direction="hold",
            reason="scale-up cooldown active",
            predicted_cpu=predicted_cpu,
            smoothed_cpu=smoothed,
            should_execute=False,
        )

    if direction == "down" and check_cooldown(
        state.last_scale_down_time, config.scale_down_cooldown_seconds, now
    ):
        return ScalingDecision(
            desired_replicas=state.current_replicas,
            direction="hold",
            reason="scale-down cooldown active",
            predicted_cpu=predicted_cpu,
            smoothed_cpu=smoothed,
            should_execute=False,
        )

    # -- Step 7: Scale-down stabilization --------------------------------------
    if (
        direction == "down"
        and state.consecutive_below_threshold < config.scale_down_stabilization_count
    ):
        count = state.consecutive_below_threshold
        need = config.scale_down_stabilization_count
        return ScalingDecision(
            desired_replicas=state.current_replicas,
            direction="hold",
            reason=f"scale-down stabilization {count}/{need}",
            predicted_cpu=predicted_cpu,
            smoothed_cpu=smoothed,
            should_execute=False,
        )

    # -- Step 8: No change needed? ---------------------------------------------
    if desired == state.current_replicas:
        return ScalingDecision(
            desired_replicas=state.current_replicas,
            direction="hold",
            reason="desired == current",
            predicted_cpu=predicted_cpu,
            smoothed_cpu=smoothed,
            should_execute=False,
        )

    # -- Step 9: Commit decision -----------------------------------------------
    if direction == "up":
        state.last_scale_up_time = now
    else:
        state.last_scale_down_time = now
        state.consecutive_below_threshold = 0  # reset after actual scale-down

    reason = (
        f"{direction} {state.current_replicas} -> {desired} "
        f"(smoothed={smoothed:.1f}%, target={config.target_utilization:.0f}%)"
    )

    return ScalingDecision(
        desired_replicas=desired,
        direction=direction,
        reason=reason,
        predicted_cpu=predicted_cpu,
        smoothed_cpu=smoothed,
        should_execute=True,
    )
