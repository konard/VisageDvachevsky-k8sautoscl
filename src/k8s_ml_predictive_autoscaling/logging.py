"""Opinionated logging configuration used across services."""

import logging
from typing import Any

from .settings import get_settings


def configure_logging(extra_handlers: list[logging.Handler] | None = None) -> None:
    """Configure root logger with structured format suitable for containers."""

    settings = get_settings()
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(level=settings.log_level, format=fmt)

    if extra_handlers:
        root = logging.getLogger()
        for handler in extra_handlers:
            root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Helper for retrieving configured loggers."""

    configure_logging()
    return logging.getLogger(name)


def log_structured(logger: logging.Logger, message: str, **context: Any) -> None:
    """Uniform structured log helper."""

    extras = " ".join(f"{key}={value}" for key, value in context.items())
    logger.info("%s %s", message, extras)
