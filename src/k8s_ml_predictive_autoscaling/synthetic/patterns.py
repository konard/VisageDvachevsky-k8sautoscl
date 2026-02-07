"""Deterministic helpers to craft synthetic demand patterns based on real web traffic research."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(slots=True)
class PatternConfig:
    """Configuration bag for realistic synthetic load generation.

    Based on research of real web traffic patterns (2025):
    - Peak hours show 2-3x baseline load
    - Business hours (9-18) have higher traffic
    - Lunch dip (12-14) shows ~20% reduction
    - Weekends have ~40-50% less traffic
    - Mobile traffic dominates (63%)
    - Typical production: 200-300 req/s avg, 800 req/s peaks
    """

    minutes: int = 24 * 60 * 7  # 1 week by default for better pattern coverage
    baseline: float = 0.2  # Lower baseline for more realistic contrast
    peak_multiplier: float = 2.5  # 2-3x increase during peak hours
    lunch_dip: float = 0.2  # 20% reduction during lunch
    weekend_drop: float = 0.45  # 45% reduction on weekends
    night_drop: float = 0.7  # 70% reduction at night (0-6am)
    spike_probability: float = 0.02  # 2% chance of traffic spike
    spike_intensity: float = 0.8  # Intensity of sudden spikes
    gradual_spike_probability: float = 0.01  # 1% chance of gradual surge
    flash_crowd_probability: float = 0.005  # 0.5% chance of "flash crowd" event
    seed: int | None = None


def generate_profile(config: PatternConfig | None = None) -> list[float]:
    """Generate a normalized load profile based on real website traffic patterns.

    Implements:
    - Realistic hourly patterns with morning/evening peaks
    - Lunch dip (12-14h)
    - Night-time reduction (0-6h)
    - Weekend reduction
    - Multiple spike types: instant, gradual, flash-crowds
    - Small random noise to simulate real variance
    """

    config = config or PatternConfig()
    rng = random.Random(config.seed)
    values: list[float] = []
    flash_crowd_active = 0  # Duration counter for flash crowd events
    gradual_surges: list[tuple[int, int, int]] = []  # (start, duration, peak)

    # First pass: generate base profile with instant events
    for minute in range(config.minutes):
        hour = (minute % (24 * 60)) / 60  # 0-24
        weekday = (minute // (24 * 60)) % 7  # 0=Mon, 6=Sun

        # Base hourly pattern (realistic web traffic curve)
        hourly_factor = _calculate_hourly_factor(hour, config)

        # Weekly pattern (weekends are quieter)
        weekly_modifier = 1.0
        if weekday >= 5:  # Weekend
            weekly_modifier = 1 - config.weekend_drop

        # Combine base load with patterns
        value = config.baseline * hourly_factor * weekly_modifier

        # Add realistic noise (±5%)
        value *= 1 + (rng.random() - 0.5) * 0.1

        # Flash crowd event (sustained spike, 30-120 min)
        if flash_crowd_active > 0:
            value *= 1.5 + rng.random() * 0.5
            flash_crowd_active -= 1
        elif rng.random() < config.flash_crowd_probability:
            flash_crowd_active = rng.randint(30, 120)
            value *= 2.0

        # Instant traffic spike
        if rng.random() < config.spike_probability:
            value += rng.random() * config.spike_intensity

        # Schedule gradual surge (don't apply yet)
        if rng.random() < config.gradual_spike_probability:
            surge_duration = rng.randint(60, 180)  # 1-3 hours
            if minute + surge_duration < config.minutes:
                surge_peak = minute + surge_duration // 2
                gradual_surges.append((minute, surge_duration, surge_peak))

        values.append(max(value, 0.01))  # Ensure non-negative with minimum floor

    # Second pass: apply gradual surges
    for start_minute, duration, peak_minute in gradual_surges:
        for offset in range(duration):
            idx = start_minute + offset
            if idx < len(values):
                # Bell curve for gradual surge
                distance = abs(idx - peak_minute)
                surge_multiplier = math.exp(-((distance / (duration / 4)) ** 2))
                values[idx] *= 1 + surge_multiplier * 0.6

    return values


def _calculate_hourly_factor(hour: float, config: PatternConfig) -> float:
    """Calculate traffic multiplier based on hour of day.

    Peak hours (real web traffic research):
    - Night (0-6h): very low (30% of baseline)
    - Morning ramp (6-9h): gradual increase
    - Morning peak (9-12h): high traffic (2-3x baseline)
    - Lunch dip (12-14h): small decrease
    - Afternoon peak (14-18h): highest traffic
    - Evening (18-20h): moderate
    - Late evening (20-22h): secondary peak (home users)
    - Night decline (22-24h): gradual decrease
    """
    if 0 <= hour < 6:
        # Deep night - minimal traffic
        return (1 - config.night_drop) + 0.1 * math.sin(math.pi * hour / 6)

    elif 6 <= hour < 9:
        # Morning ramp-up
        progress = (hour - 6) / 3
        return 0.4 + progress * (config.peak_multiplier - 0.4)

    elif 9 <= hour < 12:
        # Morning peak (business hours)
        return config.peak_multiplier * (0.9 + 0.1 * math.sin(math.pi * (hour - 9) / 3))

    elif 12 <= hour < 14:
        # Lunch dip
        dip_progress = (hour - 12) / 2
        return config.peak_multiplier * (1 - config.lunch_dip * math.sin(math.pi * dip_progress))

    elif 14 <= hour < 18:
        # Afternoon peak (highest traffic)
        return config.peak_multiplier * 1.1

    elif 18 <= hour < 20:
        # Evening transition
        progress = (hour - 18) / 2
        return config.peak_multiplier * (1.1 - 0.3 * progress)

    elif 20 <= hour < 22:
        # Late evening secondary peak (home users, entertainment)
        return config.peak_multiplier * 0.85

    else:  # 22-24
        # Night decline
        progress = (hour - 22) / 2
        return config.peak_multiplier * 0.85 * (1 - progress * 0.7)


def windowed(values: Sequence[float], size: int) -> Iterable[list[float]]:
    """Yield sliding windows from the generated profile."""

    if size <= 0:
        raise ValueError("Window size must be positive")
    for index in range(0, len(values) - size + 1):
        yield list(values[index : index + size])


__all__ = ["PatternConfig", "generate_profile", "windowed"]
