"""Configuration helpers for Prometheus data collection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, PositiveInt, field_validator

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


class MetricConfig(BaseModel):
    """Single metric definition with PromQL query and optional overrides."""

    name: str
    promql: str
    output_prefix: str | None = Field(default=None, description="File prefix override.")
    step: str | None = Field(default=None, description="Prometheus query step, e.g. 30s.")

    def resolve_step(self, default_step: str) -> timedelta:
        return parse_duration(self.step or default_step)

    def resolved_prefix(self) -> str:
        return self.output_prefix or self.name


class PrometheusSettings(BaseModel):
    base_url: str = Field(default="http://localhost:9090")
    timeout_seconds: PositiveInt = Field(default=10)
    verify_ssl: bool = Field(default=True)


class CollectionSettings(BaseModel):
    output_dir: Path = Field(default=Path("data/raw"))
    lookback_hours: PositiveInt = Field(default=24)
    chunk_hours: PositiveInt = Field(default=6)
    default_step: str = Field(default="30s")

    @field_validator("output_dir", mode="before")
    @classmethod
    def _expand_output_dir(cls, value: Any) -> Path:
        return Path(value).expanduser()

    @property
    def lookback_delta(self) -> timedelta:
        return timedelta(hours=self.lookback_hours)

    @property
    def chunk_delta(self) -> timedelta:
        return timedelta(hours=self.chunk_hours)


class CollectorConfig(BaseModel):
    prometheus: PrometheusSettings = Field(default_factory=PrometheusSettings)
    collection: CollectionSettings = Field(default_factory=CollectionSettings)
    metrics: list[MetricConfig]

    @field_validator("metrics")
    @classmethod
    def _ensure_metrics(cls, value: list[MetricConfig]) -> list[MetricConfig]:
        if not value:
            msg = "At least one metric should be defined in the collector config"
            raise ValueError(msg)
        return value


@dataclass(slots=True)
class CollectorConfigBundle:
    config: CollectorConfig
    path: Path


def parse_duration(value: str) -> timedelta:
    """Parse duration strings like `30s`, `5m`, `1h`."""

    units = {
        "s": 1,
        "m": 60,
        "h": 3600,
    }
    value = value.strip().lower()
    if not value:
        raise ValueError("Duration cannot be empty")

    suffix = value[-1]
    if suffix not in units:
        raise ValueError(f"Unsupported duration unit: {suffix}")

    amount = float(value[:-1]) if value[:-1] else 0.0
    if amount <= 0:
        raise ValueError("Duration must be positive")
    seconds = amount * units[suffix]
    return timedelta(seconds=seconds)


def load_config(path: Path | None = None) -> CollectorConfig:
    """Load YAML config into a CollectorConfig instance."""

    config_path = path or DEFAULT_CONFIG_PATH
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        raise ValueError(f"Collector config is empty: {config_path}")
    return CollectorConfig.model_validate(data)


__all__ = [
    "CollectorConfig",
    "CollectorConfigBundle",
    "CollectionSettings",
    "MetricConfig",
    "PrometheusSettings",
    "DEFAULT_CONFIG_PATH",
    "load_config",
    "parse_duration",
]
