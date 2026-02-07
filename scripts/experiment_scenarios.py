"""Load scenarios for experiment simulation.

Generates 4 distinct workload patterns from real data and synthetic profiles,
each producing (actual_cpu, predicted_cpu) trace pairs for the simulator.

Usage:
    from scripts.experiment_scenarios import get_all_scenarios, Scenario
    scenarios = get_all_scenarios("data/processed/test.csv", "data/processed/scaler.pkl")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


@dataclass
class Scenario:
    """A named experiment scenario with CPU traces."""

    name: str
    description: str
    actual_cpu: np.ndarray  # shape (N,) — actual CPU % (original scale)
    predicted_cpu: np.ndarray  # shape (N,) — predicted CPU % (original scale)
    duration_hours: float

    @property
    def steps(self) -> int:
        return len(self.actual_cpu)


def _load_test_data(
    test_csv_path: str | Path,
    scaler_path: str | Path,
    metadata_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Load test.csv and inverse-transform CPU columns to original scale.

    Returns:
        (actual_cpu, target_cpu_t3, scaler_mean, scaler_scale)
    """
    df = pd.read_csv(test_csv_path)
    scaler = joblib.load(scaler_path)

    # Determine CPU index in scaler
    if metadata_path is not None:
        with open(metadata_path) as f:
            meta = json.load(f)
        scaler_cols = meta["scaler_feature_columns"]
    else:
        # Fallback: cpu_usage is always first
        scaler_cols = ["cpu_usage"]

    cpu_idx = scaler_cols.index("cpu_usage")
    cpu_mean = scaler.mean_[cpu_idx]
    cpu_scale = scaler.scale_[cpu_idx]

    # Inverse transform: original = scaled * scale + mean
    actual = df["cpu_usage"].values * cpu_scale + cpu_mean
    target = df["target_cpu_usage_t+3"].values * cpu_scale + cpu_mean

    return actual, target, cpu_mean, cpu_scale


def _add_prediction_noise(
    actual: np.ndarray,
    target_t3: np.ndarray,
    noise_std: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate predicted CPU by shifting target_t+3 with small noise.

    The LSTM model achieves R²=0.89, so predictions are close to reality
    but not perfect. We simulate this by using target_t+3 (the actual
    future value) with small Gaussian noise added.
    """
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, noise_std, size=len(target_t3))
    predicted = target_t3 + noise
    return np.clip(predicted, 0, 100)


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------


def scenario_stable(
    actual_full: np.ndarray,
    target_full: np.ndarray,
) -> Scenario:
    """Scenario 1: Stable workload (~45-55% CPU).

    Select a segment with low variance from the real data.
    """
    window = 50
    n = len(actual_full)
    best_start = 0
    best_var = float("inf")

    # Find the most stable 300-step window
    target_len = 300
    for start in range(0, n - target_len, window):
        segment = actual_full[start : start + target_len]
        var = float(np.var(segment))
        if var < best_var:
            best_var = var
            best_start = start

    actual = actual_full[best_start : best_start + target_len].copy()
    predicted = _add_prediction_noise(
        actual, target_full[best_start : best_start + target_len], noise_std=1.5, seed=100
    )

    return Scenario(
        name="stable_workload",
        description="Stable workload with low variance (~45-55% CPU). "
        "Tests steady-state behavior and resource efficiency.",
        actual_cpu=actual,
        predicted_cpu=predicted,
        duration_hours=target_len * 5 / 60,
    )


def scenario_gradual_ramp(
    actual_full: np.ndarray,
    target_full: np.ndarray,
) -> Scenario:
    """Scenario 2: Gradual ramp from low to high load.

    Synthetic ramp: 30% → 85% over 400 steps with realistic noise.
    """
    n_steps = 400
    rng = np.random.RandomState(200)

    # Create ramp
    t = np.linspace(0, 1, n_steps)
    base_ramp = 30 + 55 * t  # 30% → 85%

    # Add realistic noise from actual data distribution
    noise = rng.normal(0, 3, size=n_steps)
    actual = np.clip(base_ramp + noise, 10, 100)

    # Predicted: ramp shifted forward by 3 steps (15 min lookahead)
    predicted_base = np.roll(base_ramp, -3)
    predicted_base[-3:] = base_ramp[-1]  # Extend last value
    predicted_noise = rng.normal(0, 2.5, size=n_steps)
    predicted = np.clip(predicted_base + predicted_noise, 10, 100)

    return Scenario(
        name="gradual_ramp",
        description="Gradual load increase from 30% to 85% CPU over ~33 hours. "
        "Tests proactive scaling during sustained growth.",
        actual_cpu=actual,
        predicted_cpu=predicted,
        duration_hours=n_steps * 5 / 60,
    )


def scenario_spike_pattern(
    actual_full: np.ndarray,
    target_full: np.ndarray,
) -> Scenario:
    """Scenario 3: Spike pattern with sudden load increases.

    3 spikes: 40% → 85% → 40%, each lasting ~30 steps.
    This is where ML prediction shines — it can anticipate spikes.
    """
    n_steps = 350
    rng = np.random.RandomState(300)

    # Base: moderate load with 3 spikes
    actual = np.full(n_steps, 42.0)
    noise = rng.normal(0, 3, size=n_steps)

    # Spike 1: steps 50-80 (sharp)
    for i in range(50, 80):
        progress = (i - 50) / 30
        if progress < 0.2:
            actual[i] = 42 + (85 - 42) * (progress / 0.2)  # Sharp rise
        elif progress < 0.7:
            actual[i] = 85  # Sustained peak
        else:
            actual[i] = 85 - (85 - 42) * ((progress - 0.7) / 0.3)  # Gradual decline

    # Spike 2: steps 140-185 (gradual buildup)
    for i in range(130, 185):
        progress = (i - 130) / 55
        if progress < 0.4:
            actual[i] = 42 + (90 - 42) * (progress / 0.4)
        elif progress < 0.7:
            actual[i] = 90
        else:
            actual[i] = 90 - (90 - 42) * ((progress - 0.7) / 0.3)

    # Spike 3: steps 250-290 (double spike)
    for i in range(250, 270):
        progress = (i - 250) / 20
        actual[i] = 42 + (78 - 42) * math.sin(progress * math.pi)
    for i in range(275, 300):
        progress = (i - 275) / 25
        actual[i] = 42 + (92 - 42) * math.sin(progress * math.pi)

    actual = np.clip(actual + noise, 10, 100)

    # Predicted: ML model sees 3 steps ahead (the model's lookahead)
    predicted = np.roll(actual, -3)
    predicted[-3:] = actual[-1]
    pred_noise = rng.normal(0, 3.0, size=n_steps)
    predicted = np.clip(predicted + pred_noise, 10, 100)

    return Scenario(
        name="spike_pattern",
        description="Three load spikes (42% → 85-92% → 42%) testing reactive vs "
        "predictive response to sudden demand changes.",
        actual_cpu=actual,
        predicted_cpu=predicted,
        duration_hours=n_steps * 5 / 60,
    )


def scenario_diurnal_cycle(
    actual_full: np.ndarray,
    target_full: np.ndarray,
) -> Scenario:
    """Scenario 4: Diurnal (24-hour) cycle.

    Realistic day/night pattern: low at night, peak during business hours.
    Uses synthetic.patterns.generate_profile() scaled to CPU range.
    """
    # Generate 48 hours (576 steps at 5-min intervals)
    n_steps = 576
    rng = np.random.RandomState(400)

    # Create diurnal pattern manually (to avoid import issues)
    actual = np.zeros(n_steps)
    for i in range(n_steps):
        minute = (i * 5) % (24 * 60)
        hour = minute / 60

        # Hourly factor based on real traffic patterns
        if 0 <= hour < 6:
            factor = 0.3 + 0.05 * math.sin(math.pi * hour / 6)
        elif 6 <= hour < 9:
            factor = 0.35 + 0.65 * ((hour - 6) / 3)
        elif 9 <= hour < 12:
            factor = 1.0
        elif 12 <= hour < 14:
            factor = 0.85  # Lunch dip
        elif 14 <= hour < 18:
            factor = 1.1  # Afternoon peak
        elif 18 <= hour < 20:
            factor = 0.8
        elif 20 <= hour < 22:
            factor = 0.7
        else:
            factor = 0.45

        # Scale to CPU range (20-85%)
        cpu_base = 20 + factor * 65
        actual[i] = cpu_base

    # Add noise
    noise = rng.normal(0, 4, size=n_steps)
    actual = np.clip(actual + noise, 10, 100)

    # Predicted: ML sees ahead by 3 steps
    predicted = np.roll(actual, -3)
    predicted[-3:] = actual[-1]
    pred_noise = rng.normal(0, 2.5, size=n_steps)
    predicted = np.clip(predicted + pred_noise, 10, 100)

    return Scenario(
        name="diurnal_cycle",
        description="48-hour diurnal pattern (night low ~25%, business peak ~85%) "
        "testing ML's ability to anticipate daily traffic patterns.",
        actual_cpu=actual,
        predicted_cpu=predicted,
        duration_hours=n_steps * 5 / 60,
    )


# ---------------------------------------------------------------------------
# Convenience: load all scenarios
# ---------------------------------------------------------------------------


def get_all_scenarios(
    test_csv_path: str | Path = "data/processed/test.csv",
    scaler_path: str | Path = "data/processed/scaler.pkl",
    metadata_path: str | Path = "data/processed/metadata.json",
) -> list[Scenario]:
    """Load and return all 4 scenarios.

    Args:
        test_csv_path: Path to test.csv (normalized data).
        scaler_path: Path to scaler.pkl (for inverse transform).
        metadata_path: Path to metadata.json.

    Returns:
        List of 4 Scenario objects.
    """
    actual_full, target_full, _, _ = _load_test_data(
        test_csv_path, scaler_path, metadata_path
    )

    return [
        scenario_stable(actual_full, target_full),
        scenario_gradual_ramp(actual_full, target_full),
        scenario_spike_pattern(actual_full, target_full),
        scenario_diurnal_cycle(actual_full, target_full),
    ]
