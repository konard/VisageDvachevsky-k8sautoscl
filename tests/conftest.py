"""Pytest configuration and shared fixtures."""

import os

os.environ.setdefault("AUTOSCALER_API_TOKEN", "unit-test-token")
os.environ.setdefault("AUTOSCALER_API_KEY_HEADER", "X-API-Key")
