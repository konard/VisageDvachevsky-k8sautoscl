"""Planner service configuration."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class PlannerSettings(BaseSettings):
    """Configuration for the resource planner service."""

    environment: Literal["local", "dev", "prod"] = Field(
        default="local", description="Deployment environment."
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    service_name: str = Field(default="k8s-ml-planner")
    metrics_path: str = Field(default="/metrics")

    # Service discovery
    predictor_url: str = Field(
        default="http://predictor:8000",
        description="URL of the predictor inference service.",
    )
    prometheus_url: str = Field(
        default="http://prometheus:9090",
        description="URL of the Prometheus server.",
    )

    # Target deployment
    target_deployment: str = Field(
        default="demo-service",
        description="Name of the K8s Deployment to scale.",
    )
    target_namespace: str = Field(
        default="predictive-autoscaling",
        description="Namespace of the target Deployment.",
    )

    # Scaling bounds
    target_utilization: float = Field(
        default=60.0, description="Target CPU utilization threshold (%)."
    )
    min_replicas: int = Field(default=2, ge=1, description="Minimum replica count.")
    max_replicas: int = Field(default=8, ge=1, description="Maximum replica count.")

    # Algorithm tuning
    safety_margin: float = Field(
        default=0.15, ge=0.0, le=1.0,
        description="Over-provisioning safety margin for scale-up (0.15 = 15%).",
    )
    smoothing_alpha: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="EMA smoothing factor (higher = more weight to new prediction).",
    )
    scale_up_threshold_factor: float = Field(
        default=1.1, description="Hysteresis: scale up when predicted > target * factor."
    )
    scale_down_threshold_factor: float = Field(
        default=0.7, description="Hysteresis: scale down when predicted < target * factor."
    )
    max_scale_up_step: int = Field(
        default=2, ge=1, description="Max replicas to add per cycle."
    )
    max_scale_down_step: int = Field(
        default=1, ge=1, description="Max replicas to remove per cycle."
    )

    # Cooldowns
    scale_up_cooldown_seconds: int = Field(
        default=60, ge=0, description="Minimum seconds between consecutive scale-ups."
    )
    scale_down_cooldown_seconds: int = Field(
        default=300, ge=0, description="Minimum seconds between consecutive scale-downs."
    )
    scale_down_stabilization_count: int = Field(
        default=3, ge=1,
        description="Consecutive below-threshold predictions required before scale-down.",
    )

    # Control loop
    loop_interval_seconds: int = Field(
        default=60, ge=5, description="Seconds between control loop iterations."
    )
    backfill_on_startup: bool = Field(
        default=True,
        description="Backfill predictor buffer with historical data on startup.",
    )

    # Resilience
    max_consecutive_failures: int = Field(
        default=5, ge=1, description="Consecutive failures before entering degraded state."
    )

    # Runtime mode
    use_k8s_client: bool = Field(
        default=True,
        description="Use real K8s client (False = MockExecutor for Docker Compose).",
    )

    # Timeouts
    predictor_timeout_seconds: float = Field(default=5.0, gt=0)
    prometheus_timeout_seconds: float = Field(default=10.0, gt=0)

    model_config = {
        "env_file": ".env",
        "env_prefix": "PLANNER_",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_planner_settings() -> PlannerSettings:
    """Cache settings to avoid re-parsing on every injection."""
    return PlannerSettings()
