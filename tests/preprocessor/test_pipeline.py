"""Integration tests for the preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from k8s_ml_predictive_autoscaling.preprocessor.config import (
    FeatureConfig,
    PreprocessorConfig,
    SlidingWindowConfig,
)
from k8s_ml_predictive_autoscaling.preprocessor.pipeline import PreprocessingPipeline


def _write_sample_raw(tmp_path: Path, metrics: list[str] | None = None) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    timestamps = pd.date_range("2024-01-01", periods=200, freq="1min", tz="UTC")
    values = {
        "cpu_metrics": 0.5,
        "memory_metrics": 200.0,
        "request_rate": 1.0,
    }
    metrics = metrics or ["cpu_metrics", "memory_metrics"]
    rows = []
    for ts in timestamps:
        for metric in metrics:
            rows.append({"timestamp": ts, "metric": metric, "value": values.get(metric, 1.0)})
    frame = pd.DataFrame(rows)
    frame.to_csv(raw_dir / "metrics.csv", index=False)


def test_pipeline_generates_processed_datasets(tmp_path: Path) -> None:
    _write_sample_raw(tmp_path)
    processed_dir = tmp_path / "processed"
    config = PreprocessorConfig(
        input_glob=str(tmp_path / "raw" / "*.csv"),
        output_dir=processed_dir,
        metrics=["cpu_metrics", "memory_metrics"],
        scaler_features=["cpu_metrics", "memory_metrics"],
        resample_rule="1min",
        features=FeatureConfig(lags=[1], rolling_windows=[2]),
        sliding_window=SlidingWindowConfig(
            sequence_length=5,
            forecast_steps=[2],
            target_metric="cpu_metrics",
        ),
    )
    pipeline = PreprocessingPipeline(config)
    outputs = pipeline.run()

    assert (processed_dir / "train.csv").exists()
    assert (processed_dir / "validation.csv").exists()
    assert (processed_dir / "test.csv").exists()
    assert (processed_dir / "scaler.pkl").exists()
    assert (processed_dir / "sequences_train.npz").exists()
    assert "train" in outputs


def test_pipeline_raises_error_when_metrics_missing(tmp_path: Path) -> None:
    _write_sample_raw(tmp_path, metrics=["cpu_metrics"])
    processed_dir = tmp_path / "processed"
    config = PreprocessorConfig(
        input_glob=str(tmp_path / "raw" / "*.csv"),
        output_dir=processed_dir,
        metrics=["cpu_metrics", "memory_metrics"],
        scaler_features=["cpu_metrics", "memory_metrics"],
        resample_rule="1min",
        features=FeatureConfig(lags=[1], rolling_windows=[2]),
        sliding_window=SlidingWindowConfig(
            sequence_length=5,
            forecast_steps=[2],
            target_metric="cpu_metrics",
        ),
    )
    pipeline = PreprocessingPipeline(config)
    with pytest.raises(ValueError, match="missing required metrics"):
        pipeline.run()
