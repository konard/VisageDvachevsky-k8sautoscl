"""Tests for the scaling algorithm -- pure functions, no I/O."""

from __future__ import annotations

import pytest

from k8s_ml_predictive_autoscaling.planner.algorithm import (
    ScalingConfig,
    ScalingDecision,
    ScalingState,
    apply_rate_limits,
    apply_smoothing,
    check_cooldown,
    compute_desired_replicas,
    compute_scaling_decision,
)


# ---------------------------------------------------------------------------
# Helper defaults
# ---------------------------------------------------------------------------

def _config(**overrides) -> ScalingConfig:
    return ScalingConfig(**overrides)


def _state(**overrides) -> ScalingState:
    return ScalingState(**overrides)


# ---------------------------------------------------------------------------
# apply_smoothing
# ---------------------------------------------------------------------------


class TestSmoothing:

    def test_first_observation_returns_current(self) -> None:
        assert apply_smoothing(75.0, None, 0.7) == 75.0

    def test_dampens_spike(self) -> None:
        # prev=50, new=100, alpha=0.7 => 0.7*100 + 0.3*50 = 85
        result = apply_smoothing(100.0, 50.0, 0.7)
        assert abs(result - 85.0) < 1e-9

    def test_alpha_zero_keeps_previous(self) -> None:
        result = apply_smoothing(100.0, 50.0, 0.0)
        assert abs(result - 50.0) < 1e-9

    def test_alpha_one_takes_current(self) -> None:
        result = apply_smoothing(100.0, 50.0, 1.0)
        assert abs(result - 100.0) < 1e-9


# ---------------------------------------------------------------------------
# compute_desired_replicas
# ---------------------------------------------------------------------------


class TestDesiredReplicas:

    def test_basic_scale_up(self) -> None:
        # R=3, CPU=75%, target=60% => ceil(3*75/60) = ceil(3.75) = 4
        # with margin: ceil(4*1.15) = ceil(4.6) = 5
        result = compute_desired_replicas(3, 75.0, 60.0, 0.15, "up")
        assert result == 5

    def test_low_load_scale_down(self) -> None:
        # R=4, CPU=25%, target=60% => ceil(4*25/60) = ceil(1.67) = 2
        # no margin for down
        result = compute_desired_replicas(4, 25.0, 60.0, 0.15, "down")
        assert result == 2

    def test_high_load_scale_up(self) -> None:
        # R=3, CPU=95%, target=60% => ceil(3*95/60) = ceil(4.75) = 5
        # with margin: ceil(5*1.15) = ceil(5.75) = 6
        result = compute_desired_replicas(3, 95.0, 60.0, 0.15, "up")
        assert result == 6

    def test_no_margin_on_scale_down(self) -> None:
        # R=5, CPU=30%, target=60% => ceil(5*30/60) = ceil(2.5) = 3
        result = compute_desired_replicas(5, 30.0, 60.0, 0.15, "down")
        assert result == 3

    def test_zero_target_returns_current(self) -> None:
        result = compute_desired_replicas(3, 50.0, 0.0, 0.15, "up")
        assert result == 3


# ---------------------------------------------------------------------------
# apply_rate_limits
# ---------------------------------------------------------------------------


class TestRateLimits:

    def test_caps_scale_up(self) -> None:
        # current=3, desired=7, max_up=2 => 5
        assert apply_rate_limits(3, 7, 2, 1) == 5

    def test_caps_scale_down(self) -> None:
        # current=5, desired=2, max_down=1 => 4
        assert apply_rate_limits(5, 2, 2, 1) == 4

    def test_no_change(self) -> None:
        assert apply_rate_limits(3, 3, 2, 1) == 3

    def test_within_limits(self) -> None:
        # current=3, desired=4, max_up=2 => 4 (within limit)
        assert apply_rate_limits(3, 4, 2, 1) == 4


# ---------------------------------------------------------------------------
# check_cooldown
# ---------------------------------------------------------------------------


class TestCooldown:

    def test_no_previous_event_no_cooldown(self) -> None:
        assert check_cooldown(None, 60.0, 100.0) is False

    def test_within_cooldown_returns_true(self) -> None:
        assert check_cooldown(90.0, 60.0, 100.0) is True

    def test_after_cooldown_returns_false(self) -> None:
        assert check_cooldown(30.0, 60.0, 100.0) is False

    def test_exact_boundary_returns_false(self) -> None:
        # At exactly cooldown time, cooldown is expired
        assert check_cooldown(40.0, 60.0, 100.0) is False


# ---------------------------------------------------------------------------
# compute_scaling_decision -- full pipeline
# ---------------------------------------------------------------------------


class TestScalingDecision:

    def test_hold_in_dead_zone(self) -> None:
        """Prediction within [42%, 66%] should hold."""
        state = _state(current_replicas=3)
        config = _config()
        # 55% is in dead-zone [42, 66]
        decision = compute_scaling_decision(55.0, state, config, now=100.0)
        assert decision.direction == "hold"
        assert decision.should_execute is False
        assert "dead-zone" in decision.reason

    def test_scale_up_above_threshold(self) -> None:
        """Prediction above 66% should trigger scale-up."""
        state = _state(current_replicas=3)
        config = _config()
        # 80% > 66% threshold
        decision = compute_scaling_decision(80.0, state, config, now=100.0)
        assert decision.direction == "up"
        assert decision.should_execute is True
        assert decision.desired_replicas > 3

    def test_scale_down_requires_stabilization(self) -> None:
        """First prediction below 42% should NOT scale down (needs 3 consecutive)."""
        state = _state(current_replicas=5)
        config = _config()
        # 30% < 42% threshold, but first time
        decision = compute_scaling_decision(30.0, state, config, now=100.0)
        assert decision.direction == "hold"
        assert "stabilization 1/3" in decision.reason
        assert decision.should_execute is False

    def test_scale_down_after_stabilization(self) -> None:
        """After 3 consecutive below-threshold predictions, scale down."""
        state = _state(current_replicas=5, consecutive_below_threshold=2)
        config = _config()
        # 3rd time below threshold
        decision = compute_scaling_decision(30.0, state, config, now=100.0)
        assert decision.direction == "down"
        assert decision.should_execute is True
        assert decision.desired_replicas < 5

    def test_cooldown_blocks_rapid_scale_up(self) -> None:
        """Second scale-up within 60s should be blocked."""
        state = _state(current_replicas=3, last_scale_up_time=95.0)
        config = _config()
        # Try to scale up 5 seconds later
        decision = compute_scaling_decision(80.0, state, config, now=100.0)
        assert decision.direction == "hold"
        assert "cooldown" in decision.reason
        assert decision.should_execute is False

    def test_cooldown_allows_after_expiry(self) -> None:
        """Scale-up after cooldown period should be allowed."""
        state = _state(current_replicas=3, last_scale_up_time=30.0)
        config = _config()
        # 70 seconds later -- past 60s cooldown
        decision = compute_scaling_decision(80.0, state, config, now=100.0)
        assert decision.direction == "up"
        assert decision.should_execute is True

    def test_clamp_respects_max(self) -> None:
        """Should not exceed max_replicas."""
        state = _state(current_replicas=7)
        config = _config(max_replicas=8)
        # Very high prediction
        decision = compute_scaling_decision(99.0, state, config, now=100.0)
        assert decision.desired_replicas <= 8

    def test_clamp_respects_min(self) -> None:
        """Should not go below min_replicas."""
        state = _state(current_replicas=3, consecutive_below_threshold=5)
        config = _config(min_replicas=2)
        decision = compute_scaling_decision(5.0, state, config, now=100.0)
        assert decision.desired_replicas >= 2

    def test_rate_limit_caps_large_jump(self) -> None:
        """Large scale-up should be capped to +2 per cycle."""
        state = _state(current_replicas=2)
        config = _config(max_scale_up_step=2, max_replicas=8)
        # Very high prediction -- would want many replicas
        decision = compute_scaling_decision(95.0, state, config, now=100.0)
        assert decision.desired_replicas <= 2 + 2  # max +2

    def test_smoothing_applied_to_prediction(self) -> None:
        """Smoothed CPU should differ from raw when there's a previous value."""
        state = _state(current_replicas=3, previous_smoothed_prediction=50.0)
        config = _config(smoothing_alpha=0.7)
        decision = compute_scaling_decision(80.0, state, config, now=100.0)
        # smoothed = 0.7*80 + 0.3*50 = 71
        assert abs(decision.smoothed_cpu - 71.0) < 1e-9

    def test_state_updated_after_scale_up(self) -> None:
        """State should record the scale-up time."""
        state = _state(current_replicas=3)
        config = _config()
        decision = compute_scaling_decision(80.0, state, config, now=100.0)
        assert decision.should_execute is True
        assert state.last_scale_up_time == 100.0

    def test_consecutive_below_resets_on_scale_down(self) -> None:
        """Consecutive below counter should reset after actual scale-down."""
        state = _state(current_replicas=5, consecutive_below_threshold=2)
        config = _config()
        compute_scaling_decision(30.0, state, config, now=100.0)
        # After successful scale-down, counter should be 0
        assert state.consecutive_below_threshold == 0

    def test_hold_when_desired_equals_current(self) -> None:
        """If computed desired == current, direction is hold."""
        state = _state(current_replicas=2, consecutive_below_threshold=5)
        config = _config(min_replicas=2)
        # Very low CPU, desired would be 1, but clamped to min=2 == current
        decision = compute_scaling_decision(10.0, state, config, now=100.0)
        assert decision.direction == "hold"
        assert decision.should_execute is False
