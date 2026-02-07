"""Tests for synthetic load pattern helpers."""

from k8s_ml_predictive_autoscaling.synthetic import PatternConfig, generate_profile, windowed


def test_generate_profile_has_expected_length() -> None:
    config = PatternConfig(minutes=60, seed=42)
    values = generate_profile(config)
    assert len(values) == 60
    assert all(v >= 0 for v in values)


def test_windowed_produces_correct_number_of_windows() -> None:
    values = list(range(10))
    windows = list(windowed(values, size=3))
    assert len(windows) == 8
    assert windows[0] == [0, 1, 2]
