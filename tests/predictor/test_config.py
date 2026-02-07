"""Tests for predictor service configuration."""

from __future__ import annotations

from pathlib import Path

from k8s_ml_predictive_autoscaling.predictor.config import PredictorSettings


def test_default_settings() -> None:
    """PredictorSettings should have sensible defaults."""
    settings = PredictorSettings()
    assert settings.environment == "local"
    assert settings.log_level == "INFO"
    assert settings.service_name == "k8s-ml-predictor"
    assert settings.sequence_length == 24
    assert settings.target_metric == "cpu_usage"
    assert settings.max_batch_size == 32
    assert settings.metrics_path == "/metrics"


def test_settings_from_env(monkeypatch: "pytest.MonkeyPatch") -> None:
    """PredictorSettings should read from PREDICTOR_ env vars."""
    monkeypatch.setenv("PREDICTOR_ENVIRONMENT", "prod")
    monkeypatch.setenv("PREDICTOR_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("PREDICTOR_SEQUENCE_LENGTH", "48")
    monkeypatch.setenv("PREDICTOR_TARGET_METRIC", "mem_util_percent")
    monkeypatch.setenv("PREDICTOR_MAX_BATCH_SIZE", "64")

    settings = PredictorSettings()
    assert settings.environment == "prod"
    assert settings.log_level == "DEBUG"
    assert settings.sequence_length == 48
    assert settings.target_metric == "mem_util_percent"
    assert settings.max_batch_size == 64


def test_settings_model_paths() -> None:
    """Model paths should be Path objects."""
    settings = PredictorSettings()
    assert isinstance(settings.onnx_model_path, Path)
    assert isinstance(settings.scaler_path, Path)
    assert isinstance(settings.metadata_path, Path)
