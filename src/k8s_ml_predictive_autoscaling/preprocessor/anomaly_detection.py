"""Simple heuristics for removing anomalous rows from time-series data."""

from __future__ import annotations

import pandas as pd


def filter_zscore(frame: pd.DataFrame, columns: list[str], threshold: float) -> pd.DataFrame:
    """Drop rows where any specified column exceeds the Z-score threshold.

    Args:
        frame: Input dataframe with numeric columns.
        columns: Columns to evaluate for anomalies.
        threshold: Absolute Z-score threshold for filtering.

    Returns:
        Filtered dataframe without extreme rows.
    """

    if not columns:
        return frame
    present = [column for column in columns if column in frame.columns]
    if not present:
        return frame
    subset = frame[present]
    std = subset.std(ddof=0).replace(0, 1.0)
    standardized = (subset - subset.mean()) / std
    standardized = standardized.fillna(0.0)
    mask = (standardized.abs() <= threshold).all(axis=1)
    return frame.loc[mask]


__all__ = ["filter_zscore"]
