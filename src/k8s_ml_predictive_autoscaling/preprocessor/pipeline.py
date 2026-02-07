"""CLI entry-point for the preprocessing pipeline."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..logging import get_logger
from .anomaly_detection import filter_zscore
from .config import InterpolationMethod, PreprocessorConfig, load_config
from .feature_engineering import add_lag_features, add_rolling_features, add_time_features

LOGGER = get_logger(__name__)


class PreprocessingPipeline:
    """Transforms raw Prometheus extracts into ML-ready datasets."""

    def __init__(self, config: PreprocessorConfig) -> None:
        self.config = config
        self.scaler = StandardScaler()

    def run(self) -> dict[str, Path]:
        LOGGER.info("Starting preprocessing pipeline")
        frame = self._load_raw()
        frame = self._resample(frame)
        if self.config.anomaly.enabled:
            frame = filter_zscore(
                frame,
                self.config.metrics,
                self.config.anomaly.zscore_threshold,
            )
        frame = self._engineer_features(frame)
        dataset = self._build_targets(frame)
        dataset = dataset.dropna()
        features = self._determine_scaler_features(dataset)
        dataset[features] = self.scaler.fit_transform(dataset[features])
        splits = self._split(dataset)
        outputs = self._persist_splits(splits)
        outputs.update(self._persist_sequences(dataset, splits))
        scaler_path = self.config.output_dir / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        outputs["scaler"] = scaler_path
        LOGGER.info(
            "Preprocessing finished",
            extra={"outputs": {k: str(v) for k, v in outputs.items()}},
        )
        return outputs

    def _load_raw(self) -> pd.DataFrame:
        files = sorted(glob.glob(self.config.input_glob))
        if not files:
            raise FileNotFoundError(f"No files matched glob: {self.config.input_glob}")
        frames = []
        for path in files:
            df = pd.read_csv(path, parse_dates=[self.config.timestamp_column])
            df = df[
                [
                    self.config.timestamp_column,
                    self.config.metric_column,
                    self.config.value_column,
                ]
            ]
            df = df[df[self.config.metric_column].isin(self.config.metrics)]
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)
        combined[self.config.timestamp_column] = pd.to_datetime(
            combined[self.config.timestamp_column],
            utc=True,
        )
        combined.sort_values(by=self.config.timestamp_column, inplace=True)
        pivot = combined.pivot_table(
            index=self.config.timestamp_column,
            columns=self.config.metric_column,
            values=self.config.value_column,
        )
        pivot = pivot.sort_index()
        self._ensure_required_metrics(pivot)
        return pivot

    def _resample(self, frame: pd.DataFrame) -> pd.DataFrame:
        resampled = frame.resample(self.config.resample_rule).mean()
        method: InterpolationMethod = self.config.interpolation_method
        resampled = resampled.interpolate(method=method)
        resampled = resampled.ffill()
        resampled = resampled.bfill()
        return resampled

    def _engineer_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy()
        if self.config.features.enable_time_features:
            enriched = add_time_features(enriched)
        enriched = add_lag_features(
            enriched,
            self.config.metrics,
            self.config.features.lags,
        )
        enriched = add_rolling_features(
            enriched, self.config.metrics, self.config.features.rolling_windows
        )
        return enriched

    def _build_targets(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy()
        target_metric = self.config.sliding_window.target_metric
        for horizon in self.config.sliding_window.forecast_steps:
            enriched[f"target_{target_metric}_t+{horizon}"] = enriched[target_metric].shift(
                -horizon
            )
        return enriched

    def _determine_scaler_features(self, frame: pd.DataFrame) -> list[str]:
        if self.config.scaler_features:
            return [col for col in self.config.scaler_features if col in frame.columns]
        return [col for col in self.config.metrics if col in frame.columns]

    def _split(self, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if abs(self.config.splits.total - 1.0) > 1e-6:
            raise ValueError("Dataset splits must sum to 1.0")
        total = len(frame)
        train_end = int(total * self.config.splits.train)
        val_end = train_end + int(total * self.config.splits.validation)
        splits = {
            "train": frame.iloc[:train_end],
            "validation": frame.iloc[train_end:val_end],
            "test": frame.iloc[val_end:],
        }
        return splits

    def _ensure_required_metrics(self, frame: pd.DataFrame) -> None:
        missing = [metric for metric in self.config.metrics if metric not in frame.columns]
        if not missing:
            return
        msg = (
            "Raw dataset is missing required metrics: "
            f"{', '.join(missing)}. Ensure collector exports match "
            "preprocessor.metrics or adjust the preprocessing config."
        )
        raise ValueError(msg)

    def _persist_splits(self, splits: dict[str, pd.DataFrame]) -> dict[str, Path]:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {}
        for name, data in splits.items():
            path = self.config.output_dir / f"{name}.csv"
            data.to_csv(path, index_label="timestamp")
            outputs[name] = path
        return outputs

    def _persist_sequences(
        self, dataset: pd.DataFrame, splits: dict[str, pd.DataFrame]
    ) -> dict[str, Path]:
        outputs: dict[str, Path] = {}
        target_cols = [c for c in dataset.columns if c.startswith("target_")]
        feature_cols = [c for c in dataset.columns if c not in target_cols]
        main_target = (
            f"target_{self.config.sliding_window.target_metric}_t+"
            f"{self.config.sliding_window.forecast_steps[0]}"
        )
        for name, data in splits.items():
            sequences, targets, timestamps = build_sequences(
                data,
                feature_cols,
                main_target,
                self.config.sliding_window.sequence_length,
                self.config.sliding_window.stride,
            )
            path = self.config.output_dir / f"sequences_{name}.npz"
            np.savez_compressed(
                path,
                sequences=sequences,
                targets=targets,
                timestamps=np.array([ts.isoformat() for ts in timestamps]),
                feature_columns=np.array(feature_cols),
                target_column=main_target,
            )
            outputs[f"sequences_{name}"] = path
        return outputs


def build_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    sequence_length: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    """Generate sliding windows for sequence models.

    Args:
        frame: Dataset slice with datetime index.
        feature_columns: Columns used as input features.
        target_column: Target column name (t + horizon).
        sequence_length: Number of timesteps per sequence.
        stride: Step size between neighboring windows.

    Returns:
        Tuple with stacked sequences, target values, and timestamps.
    """

    if target_column not in frame.columns:
        raise KeyError(f"Target column {target_column} not found in frame")
    values = frame[feature_columns].to_numpy(dtype=float)
    targets = frame[target_column].to_numpy(dtype=float)
    timestamps = frame.index.to_list()
    sequences: list[np.ndarray] = []
    y: list[float] = []
    stamps: list[pd.Timestamp] = []
    for start in range(0, len(frame) - sequence_length + 1, stride):
        end = start + sequence_length
        sequences.append(values[start:end])
        y.append(targets[end - 1])
        stamps.append(timestamps[end - 1])
    if not sequences:
        return np.empty((0, sequence_length, len(feature_columns))), np.empty((0,)), []
    return np.stack(sequences), np.array(y), stamps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to preprocessing YAML config",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    pipeline = PreprocessingPipeline(config)
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
