"""Tests for the planner configuration."""

from __future__ import annotations

import os

import pytest

from k8s_ml_predictive_autoscaling.planner.config import PlannerSettings


class TestPlannerSettings:

    def test_defaults(self) -> None:
        settings = PlannerSettings()
        assert settings.target_utilization == 60.0
        assert settings.min_replicas == 2
        assert settings.max_replicas == 8
        assert settings.loop_interval_seconds == 60
        assert settings.use_k8s_client is True
        assert settings.smoothing_alpha == 0.7
        assert settings.scale_up_cooldown_seconds == 60
        assert settings.scale_down_cooldown_seconds == 300

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PLANNER_TARGET_UTILIZATION", "75.0")
        monkeypatch.setenv("PLANNER_MIN_REPLICAS", "3")
        monkeypatch.setenv("PLANNER_MAX_REPLICAS", "10")
        monkeypatch.setenv("PLANNER_USE_K8S_CLIENT", "false")

        settings = PlannerSettings()
        assert settings.target_utilization == 75.0
        assert settings.min_replicas == 3
        assert settings.max_replicas == 10
        assert settings.use_k8s_client is False

    def test_predictor_and_prometheus_urls(self) -> None:
        settings = PlannerSettings(
            predictor_url="http://custom-predictor:9000",
            prometheus_url="http://custom-prom:9091",
        )
        assert settings.predictor_url == "http://custom-predictor:9000"
        assert settings.prometheus_url == "http://custom-prom:9091"
