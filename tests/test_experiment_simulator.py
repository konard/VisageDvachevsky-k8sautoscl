"""Tests for the experiment simulator (HPA vs ML autoscaler)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.experiment_simulator import (
    ExperimentMetrics,
    ExperimentRunner,
    HPAConfig,
    HPASimulator,
    MLAutoscalerSimulator,
    MLConfig,
    StepRecord,
    compute_metrics,
)


# ---------------------------------------------------------------------------
# HPASimulator tests
# ---------------------------------------------------------------------------


class TestHPASimulator:
    def test_starts_at_min_replicas(self):
        hpa = HPASimulator(HPAConfig(min_replicas=3))
        assert hpa.replicas == 3

    def test_scales_up_on_high_cpu(self):
        hpa = HPASimulator(HPAConfig(target_utilization=60.0, tolerance=0.1))
        # CPU=80% → ratio=1.33, well above 1.1 tolerance
        replicas, direction = hpa.step(80.0, t=0)
        assert direction == "up"
        assert replicas > 2

    def test_holds_in_tolerance_band(self):
        hpa = HPASimulator(HPAConfig(target_utilization=60.0, tolerance=0.1))
        # CPU=63% → ratio=1.05, within ±10% tolerance
        replicas, direction = hpa.step(63.0, t=0)
        assert direction == "hold"
        assert replicas == 2

    def test_respects_max_replicas(self):
        hpa = HPASimulator(HPAConfig(
            target_utilization=60.0, min_replicas=2, max_replicas=4
        ))
        # Push CPU very high multiple times
        for t in range(20):
            hpa.step(95.0, t=t)
        assert hpa.replicas <= 4

    def test_respects_min_replicas(self):
        hpa = HPASimulator(HPAConfig(
            target_utilization=60.0, min_replicas=2, max_replicas=8
        ))
        # First scale up then try to scale way down
        hpa.step(90.0, t=0)
        for t in range(1, 30):
            hpa.step(10.0, t=t)
        assert hpa.replicas >= 2

    def test_scale_down_requires_stabilization(self):
        cfg = HPAConfig(
            target_utilization=60.0,
            scale_down_stabilization_count=3,
        )
        hpa = HPASimulator(cfg)
        # Scale up first
        hpa.step(90.0, t=0)
        initial = hpa.replicas
        # First below-threshold step should hold (stabilization)
        _, dir1 = hpa.step(20.0, t=1)
        assert dir1 == "hold"
        assert hpa.replicas == initial

    def test_max_scale_up_step_respected(self):
        cfg = HPAConfig(
            target_utilization=60.0,
            min_replicas=2,
            max_replicas=8,
            max_scale_up_step=2,
        )
        hpa = HPASimulator(cfg)
        # Extreme CPU: should want many replicas but capped by step
        _, _ = hpa.step(99.0, t=0)
        assert hpa.replicas <= 2 + cfg.max_scale_up_step


# ---------------------------------------------------------------------------
# MLAutoscalerSimulator tests
# ---------------------------------------------------------------------------


class TestMLAutoscalerSimulator:
    def test_starts_at_min_replicas(self):
        ml = MLAutoscalerSimulator(MLConfig(min_replicas=3))
        assert ml.replicas == 3

    def test_holds_in_dead_zone(self):
        ml = MLAutoscalerSimulator(MLConfig(target_utilization=60.0))
        # 50% → smoothed ≈ 50%, between 42% and 66% → dead zone
        _, direction = ml.step(50.0, t=0)
        assert direction == "hold"

    def test_scales_up_above_threshold(self):
        cfg = MLConfig(
            target_utilization=60.0,
            scale_up_threshold_factor=1.1,  # threshold = 66%
            smoothing_alpha=1.0,  # No smoothing for predictability
        )
        ml = MLAutoscalerSimulator(cfg)
        _, direction = ml.step(75.0, t=0)
        assert direction == "up"
        assert ml.replicas > 2

    def test_scale_down_stabilization(self):
        cfg = MLConfig(
            target_utilization=60.0,
            smoothing_alpha=1.0,
            scale_down_stabilization_count=3,
        )
        ml = MLAutoscalerSimulator(cfg)
        # First scale up
        ml.step(80.0, t=0)
        initial = ml.replicas

        # Try to scale down — needs 3 consecutive below-threshold
        _, d1 = ml.step(30.0, t=1)
        assert d1 == "hold"  # 1/3
        _, d2 = ml.step(30.0, t=2)
        assert d2 == "hold"  # 2/3

    def test_ema_smoothing_dampens(self):
        cfg = MLConfig(smoothing_alpha=0.5)
        ml = MLAutoscalerSimulator(cfg)
        # First step: smoothed = predicted (no history)
        ml.step(50.0, t=0)
        assert ml._smoothed == 50.0

        # Second step: EMA applies
        ml.step(70.0, t=1)
        expected = 0.5 * 70.0 + 0.5 * 50.0  # = 60.0
        assert abs(ml._smoothed - expected) < 0.01

    def test_respects_bounds(self):
        cfg = MLConfig(
            target_utilization=60.0,
            min_replicas=2,
            max_replicas=5,
            smoothing_alpha=1.0,
        )
        ml = MLAutoscalerSimulator(cfg)
        for t in range(30):
            ml.step(99.0, t=t)
        assert ml.replicas <= 5

    def test_cooldown_blocks_rapid_scaling(self):
        cfg = MLConfig(
            target_utilization=60.0,
            smoothing_alpha=1.0,
            scale_up_cooldown_seconds=600.0,  # 10 minutes
            step_duration_minutes=5.0,
        )
        ml = MLAutoscalerSimulator(cfg)
        # First scale-up
        _, d1 = ml.step(80.0, t=0)
        assert d1 == "up"
        # Immediate next step: should be blocked by cooldown
        _, d2 = ml.step(85.0, t=1)
        assert d2 == "hold"


# ---------------------------------------------------------------------------
# ExperimentRunner tests
# ---------------------------------------------------------------------------


class TestExperimentRunner:
    def test_runs_on_simple_data(self):
        actual = [50.0] * 20
        predicted = [55.0] * 20
        runner = ExperimentRunner(actual, predicted)
        history, metrics = runner.run()
        assert len(history) == 20
        assert metrics.total_steps == 20
        assert all(isinstance(r, StepRecord) for r in history)

    def test_history_records_correct_cpu(self):
        actual = [30.0, 50.0, 70.0]
        predicted = [35.0, 55.0, 75.0]
        runner = ExperimentRunner(actual, predicted)
        history, _ = runner.run()
        assert history[0].actual_cpu == 30.0
        assert history[1].predicted_cpu == 55.0
        assert history[2].actual_cpu == 70.0

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError, match="Length mismatch"):
            ExperimentRunner([1.0, 2.0], [1.0])

    def test_metrics_has_all_fields(self):
        actual = list(np.random.uniform(30, 80, 100))
        predicted = list(np.random.uniform(30, 80, 100))
        runner = ExperimentRunner(actual, predicted)
        _, metrics = runner.run()

        assert isinstance(metrics.hpa_avg_replicas, float)
        assert isinstance(metrics.ml_avg_replicas, float)
        assert isinstance(metrics.cost_savings_pct, float)
        assert metrics.hpa_scaling_events >= 0
        assert metrics.ml_scaling_events >= 0


# ---------------------------------------------------------------------------
# Metrics computation tests
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_empty_history(self):
        metrics = compute_metrics([])
        assert metrics.total_steps == 0
        assert metrics.cost_savings_pct == 0

    def test_constant_replicas_no_events(self):
        history = [
            StepRecord(t=i, actual_cpu=50.0, predicted_cpu=50.0,
                       hpa_replicas=3, ml_replicas=3,
                       hpa_decision="hold", ml_decision="hold")
            for i in range(10)
        ]
        metrics = compute_metrics(history)
        assert metrics.hpa_scaling_events == 0
        assert metrics.ml_scaling_events == 0
        assert metrics.hpa_avg_replicas == 3.0
        assert metrics.ml_avg_replicas == 3.0

    def test_cost_savings_positive_when_ml_uses_fewer(self):
        history = [
            StepRecord(t=i, actual_cpu=50.0, predicted_cpu=50.0,
                       hpa_replicas=5, ml_replicas=3,
                       hpa_decision="hold", ml_decision="hold")
            for i in range(10)
        ]
        metrics = compute_metrics(history)
        assert metrics.cost_savings_pct > 0

    def test_scaling_events_counted(self):
        history = [
            StepRecord(t=0, actual_cpu=50.0, predicted_cpu=50.0,
                       hpa_replicas=2, ml_replicas=2,
                       hpa_decision="hold", ml_decision="hold"),
            StepRecord(t=1, actual_cpu=80.0, predicted_cpu=80.0,
                       hpa_replicas=4, ml_replicas=3,
                       hpa_decision="up", ml_decision="up"),
            StepRecord(t=2, actual_cpu=80.0, predicted_cpu=80.0,
                       hpa_replicas=4, ml_replicas=3,
                       hpa_decision="hold", ml_decision="hold"),
            StepRecord(t=3, actual_cpu=30.0, predicted_cpu=30.0,
                       hpa_replicas=3, ml_replicas=2,
                       hpa_decision="down", ml_decision="down"),
        ]
        metrics = compute_metrics(history)
        assert metrics.hpa_scaling_events == 2  # 2→4, 4→3
        assert metrics.ml_scaling_events == 2  # 2→3, 3→2
