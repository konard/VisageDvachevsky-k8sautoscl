"""Feature engineering helpers for temporal datasets."""

from __future__ import annotations

from typing import cast

import pandas as pd


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Append hour/day/weekend indicators derived from datetime index.

    Args:
        frame: Input dataframe indexed by datetime.

    Returns:
        Copy of the dataframe enriched with hour/day/weekend features.
    """

    enriched = frame.copy()
    dt_index = cast(pd.DatetimeIndex, enriched.index)
    enriched["hour"] = dt_index.hour
    enriched["day_of_week"] = dt_index.dayofweek
    enriched["is_weekend"] = (enriched["day_of_week"] >= 5).astype(int)
    enriched["minute_of_day"] = dt_index.hour * 60 + dt_index.minute
    return enriched


def add_lag_features(frame: pd.DataFrame, columns: list[str], lags: list[int]) -> pd.DataFrame:
    """Append lag features for the requested columns.

    Args:
        frame: Input dataframe indexed by datetime.
        columns: Base metric columns for which lags are created.
        lags: Lag offsets (in samples).

    Returns:
        Copy of the dataframe with additional lag columns.
    """

    enriched = frame.copy()
    for column in columns:
        if column not in enriched.columns:
            continue
        for lag in lags:
            enriched[f"{column}_lag_{lag}"] = enriched[column].shift(lag)
    return enriched


def add_rolling_features(
    frame: pd.DataFrame, columns: list[str], windows: list[int]
) -> pd.DataFrame:
    """Append rolling mean statistics for provided windows.

    Args:
        frame: Input dataframe indexed by datetime.
        columns: Metric columns to aggregate.
        windows: Window sizes (in samples) for rolling means.

    Returns:
        Copy of the dataframe with rolling mean columns.
    """

    enriched = frame.copy()
    for column in columns:
        if column not in enriched.columns:
            continue
        for window in windows:
            enriched[f"{column}_rolling_mean_{window}"] = enriched[column].rolling(window).mean()
    return enriched


__all__ = ["add_time_features", "add_lag_features", "add_rolling_features"]
