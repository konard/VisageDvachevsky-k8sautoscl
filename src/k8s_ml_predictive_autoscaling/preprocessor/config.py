"""Configuration schema for preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, PositiveInt, field_validator

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


class FeatureConfig(BaseModel):
    enable_time_features: bool = True
    lags: list[int] = Field(default_factory=lambda: [1, 5, 15])
    rolling_windows: list[int] = Field(default_factory=lambda: [3, 5])


class AnomalyConfig(BaseModel):
    enabled: bool = True
    zscore_threshold: float = 3.0


class SlidingWindowConfig(BaseModel):
    sequence_length: PositiveInt = 30
    forecast_steps: list[int] = Field(default_factory=lambda: [5, 15, 30])
    stride: PositiveInt = 1
    target_metric: str = "cpu_metrics"


class DatasetSplitConfig(BaseModel):
    train: float = 0.7
    validation: float = 0.15
    test: float = 0.15

    @field_validator("train", "validation", "test")
    @classmethod
    def _range(cls, value: float) -> float:
        if value <= 0 or value >= 1:
            raise ValueError("Split ratios must be between 0 and 1")
        return value

    @property
    def total(self) -> float:
        return self.train + self.validation + self.test


class PreprocessorConfig(BaseModel):
    input_glob: str = Field(default="data/raw/*.csv")
    timestamp_column: str = Field(default="timestamp")
    metric_column: str = Field(default="metric")
    value_column: str = Field(default="value")
    output_dir: Path = Field(default=Path("data/processed"))
    resample_rule: str = Field(default="1min")
    interpolation_method: InterpolationMethod = Field(default="time")
    scaler_features: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(
        default_factory=lambda: [
            "cpu_metrics",
            "memory_metrics",
            "request_rate",
        ]
    )
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    sliding_window: SlidingWindowConfig = Field(default_factory=SlidingWindowConfig)
    splits: DatasetSplitConfig = Field(default_factory=DatasetSplitConfig)

    @field_validator("output_dir", mode="before")
    @classmethod
    def _expand_output_dir(cls, value: Any) -> Path:
        return Path(value).expanduser()


def load_config(path: Path | None = None) -> PreprocessorConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        raise ValueError(f"Preprocessor config is empty: {config_path}")
    return PreprocessorConfig.model_validate(data)


__all__ = [
    "PreprocessorConfig",
    "FeatureConfig",
    "AnomalyConfig",
    "SlidingWindowConfig",
    "DatasetSplitConfig",
    "DEFAULT_CONFIG_PATH",
    "InterpolationMethod",
    "load_config",
]
InterpolationMethod = Literal[
    "linear",
    "time",
    "index",
    "values",
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "barycentric",
    "polynomial",
    "krogh",
    "piecewise_polynomial",
    "spline",
    "pchip",
    "akima",
    "cubicspline",
    "from_derivatives",
]
