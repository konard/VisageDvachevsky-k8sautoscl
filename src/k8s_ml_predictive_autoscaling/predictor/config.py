"""Predictor service configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class PredictorSettings(BaseSettings):
    """Configuration for the predictor inference service."""

    environment: Literal["local", "dev", "prod"] = Field(
        default="local", description="Deployment environment."
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    service_name: str = Field(default="k8s-ml-predictor")
    metrics_path: str = Field(default="/metrics")

    # Model paths
    onnx_model_path: Path = Field(
        default=Path("models/onnx/lstm_forecaster.onnx"),
        description="Path to the ONNX model file.",
    )
    scaler_path: Path = Field(
        default=Path("data/processed/scaler.pkl"),
        description="Path to the fitted StandardScaler.",
    )
    metadata_path: Path = Field(
        default=Path("data/processed/metadata.json"),
        description="Path to dataset metadata JSON.",
    )

    # Model parameters
    sequence_length: int = Field(default=24, description="LSTM input sequence length.")
    target_metric: str = Field(default="cpu_usage", description="Primary target metric.")

    # Inference settings
    max_batch_size: int = Field(default=32, description="Max batch size for inference.")

    model_config = {
        "env_file": ".env",
        "env_prefix": "PREDICTOR_",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_predictor_settings() -> PredictorSettings:
    """Cache settings to avoid re-parsing on every injection."""
    return PredictorSettings()
