"""Proxy module exposing synthetic helpers for backwards compatibility."""

from k8s_ml_predictive_autoscaling.synthetic.patterns import (
    PatternConfig,
    generate_profile,
    windowed,
)

__all__ = ["PatternConfig", "generate_profile", "windowed"]
