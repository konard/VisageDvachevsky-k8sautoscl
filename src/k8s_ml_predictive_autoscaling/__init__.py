"""Core package for the k8s-ml-predictive-autoscaling project."""

from importlib import metadata

__all__ = ["__version__"]


try:
    __version__ = metadata.version("k8s-ml-predictive-autoscaling")
except metadata.PackageNotFoundError:  # pragma: no cover - during local dev without install
    __version__ = "0.1.0"
