"""Application-wide configuration helpers."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Typed settings leveraging environment variables for overrides."""

    environment: Literal["local", "dev", "prod"] = Field(
        default="local", description="Deployment environment descriptor."
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    service_name: str = Field(default="k8s-ml-predictive-autoscaling-demo")
    metrics_path: str = Field(default="/metrics")
    api_token: SecretStr | None = Field(
        default=None,
        description="Shared API token used for authenticating write operations.",
    )
    api_key_header: str = Field(
        default="X-API-Key",
        min_length=3,
        description="HTTP header name used to transport the API token.",
    )

    model_config = {
        "env_file": ".env",
        "env_prefix": "AUTOSCALER_",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @field_validator("api_token")
    @classmethod
    def _normalize_token(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        if not value.get_secret_value().strip():
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    """Cache Settings to avoid re-parsing env on every injection."""

    return Settings()
