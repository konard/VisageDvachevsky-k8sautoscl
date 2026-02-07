"""ONNX inference engine for time series prediction.

Handles model loading, session management, feature preprocessing,
and prediction generation with proper scaling.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class PredictionEngine:
    """ONNX Runtime-based inference engine.

    Manages a sliding window of recent metric observations,
    applies feature engineering and scaling, then runs the
    ONNX model for prediction.
    """

    def __init__(
        self,
        onnx_model_path: Path,
        scaler_path: Path,
        metadata_path: Path,
        sequence_length: int = 24,
    ) -> None:
        self.sequence_length = sequence_length
        self._load_metadata(metadata_path)
        self._load_scaler(scaler_path)
        self._load_model(onnx_model_path)

        # Sliding window buffer for real-time observations
        self._buffer: deque[dict[str, float]] = deque(maxlen=sequence_length + 30)
        self._ready = False

    def _load_metadata(self, path: Path) -> None:
        """Load dataset metadata (feature columns, target info)."""
        with open(path) as f:
            self._metadata: dict[str, Any] = json.load(f)

        self.feature_columns: list[str] = self._metadata["feature_columns"]
        self.target_metric: str = self._metadata["target_metric"]
        self.forecast_horizon: int = self._metadata["forecast_horizon"]
        self.n_features: int = len(self.feature_columns)

        # Separate feature groups for engineering
        self.base_metrics = [
            "cpu_usage", "mem_util_percent", "net_in", "net_out", "disk_io_percent"
        ]
        self.lag_steps = [1, 3, 6, 12]
        self.rolling_windows = [3, 6, 12]
        self.time_features = ["hour", "day_of_week", "is_weekend", "minute_of_day"]
        self.scaler_feature_columns: list[str] = self._metadata.get(
            "scaler_feature_columns", []
        )

        logger.info(
            "Metadata loaded: %d features, target=%s, horizon=%d",
            self.n_features, self.target_metric, self.forecast_horizon,
        )

    def _load_scaler(self, path: Path) -> None:
        """Load the fitted StandardScaler."""
        self._scaler = joblib.load(path)
        logger.info("Scaler loaded from %s", path)

    def _load_model(self, path: Path) -> None:
        """Initialize ONNX Runtime session."""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = 2

        self._session = ort.InferenceSession(str(path), sess_options)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        # Verify expected input shape
        expected_shape = self._session.get_inputs()[0].shape
        logger.info(
            "ONNX model loaded from %s, input=%s, expected_shape=%s",
            path, self._input_name, expected_shape,
        )

    @property
    def is_ready(self) -> bool:
        """Whether the engine has enough buffered data for prediction."""
        return len(self._buffer) >= self.sequence_length

    @property
    def buffer_size(self) -> int:
        """Current number of observations in the buffer."""
        return len(self._buffer)

    def ingest(self, observation: dict[str, float]) -> None:
        """Add a new metric observation to the sliding window.

        Args:
            observation: Dict with keys matching base_metrics + time info.
                Required keys: cpu_usage, mem_util_percent, net_in, net_out,
                disk_io_percent, hour, minute_of_day
        """
        self._buffer.append(observation)
        if self.is_ready and not self._ready:
            self._ready = True
            logger.info("Engine ready: buffer filled (%d observations)", len(self._buffer))

    def _engineer_features(self, observations: list[dict[str, float]]) -> np.ndarray:
        """Build the full feature vector for each timestep.

        Takes raw observations and produces the 44-feature matrix
        expected by the ONNX model.
        """
        n = len(observations)
        feature_matrix = np.zeros((n, self.n_features), dtype=np.float32)

        # Build arrays for base metrics
        base_arrays: dict[str, np.ndarray] = {}
        for metric in self.base_metrics:
            base_arrays[metric] = np.array(
                [obs.get(metric, 0.0) for obs in observations], dtype=np.float32
            )

        col_idx = {name: i for i, name in enumerate(self.feature_columns)}

        for t in range(n):
            # Base metrics
            for metric in self.base_metrics:
                if metric in col_idx:
                    feature_matrix[t, col_idx[metric]] = base_arrays[metric][t]

            # Lag features
            for metric in self.base_metrics:
                for lag in self.lag_steps:
                    col_name = f"{metric}_lag_{lag}"
                    if col_name in col_idx:
                        src_idx = t - lag
                        if src_idx >= 0:
                            feature_matrix[t, col_idx[col_name]] = base_arrays[metric][src_idx]
                        else:
                            feature_matrix[t, col_idx[col_name]] = base_arrays[metric][0]

            # Rolling mean features
            for metric in self.base_metrics:
                for window in self.rolling_windows:
                    col_name = f"{metric}_rolling_mean_{window}"
                    if col_name in col_idx:
                        start = max(0, t - window + 1)
                        feature_matrix[t, col_idx[col_name]] = base_arrays[metric][start:t + 1].mean()

            # Time features (normalized to [0,1])
            obs = observations[t]
            if "hour" in col_idx:
                feature_matrix[t, col_idx["hour"]] = obs.get("hour", 0) / 23.0
            if "day_of_week" in col_idx:
                feature_matrix[t, col_idx["day_of_week"]] = obs.get("day_of_week", 0) / 6.0
            if "is_weekend" in col_idx:
                feature_matrix[t, col_idx["is_weekend"]] = float(obs.get("is_weekend", 0))
            if "minute_of_day" in col_idx:
                feature_matrix[t, col_idx["minute_of_day"]] = obs.get("minute_of_day", 0) / 1439.0

        return feature_matrix

    def _apply_scaling(self, features: np.ndarray) -> np.ndarray:
        """Apply StandardScaler to the non-time feature columns."""
        scaled = features.copy()

        # Only scale the columns that the scaler was fitted on
        scaler_indices = []
        for col_name in self.scaler_feature_columns:
            if col_name in self.feature_columns:
                scaler_indices.append(self.feature_columns.index(col_name))

        if scaler_indices:
            scaler_data = features[:, scaler_indices]
            scaled[:, scaler_indices] = self._scaler.transform(scaler_data)

        return scaled

    def predict(self) -> dict[str, Any]:
        """Generate a prediction using the current buffer.

        Returns:
            Dict with prediction value, confidence info, and timing.

        Raises:
            RuntimeError: If buffer doesn't have enough data.
        """
        if not self.is_ready:
            raise RuntimeError(
                f"Not enough data: have {len(self._buffer)}, "
                f"need {self.sequence_length}"
            )

        start_time = time.perf_counter()

        # Get the most recent sequence_length observations
        recent = list(self._buffer)[-self.sequence_length:]

        # Feature engineering
        features = self._engineer_features(recent)

        # Apply scaling
        features_scaled = self._apply_scaling(features)

        # Reshape for ONNX: (1, seq_length, n_features)
        input_tensor = features_scaled.reshape(1, self.sequence_length, self.n_features)

        # Run inference
        inference_start = time.perf_counter()
        outputs = self._session.run(
            [self._output_name], {self._input_name: input_tensor}
        )
        inference_ms = (time.perf_counter() - inference_start) * 1000

        prediction_value = float(outputs[0].flat[0])

        # Inverse transform the prediction to original scale
        cpu_idx = self.scaler_feature_columns.index(self.target_metric)
        original_value = (
            prediction_value * self._scaler.scale_[cpu_idx]
            + self._scaler.mean_[cpu_idx]
        )

        total_ms = (time.perf_counter() - start_time) * 1000

        return {
            "prediction_normalized": prediction_value,
            "prediction_original_scale": original_value,
            "target_metric": self.target_metric,
            "forecast_horizon_steps": self.forecast_horizon,
            "forecast_horizon_minutes": self.forecast_horizon * 5,
            "model_type": "lstm_onnx",
            "buffer_size": len(self._buffer),
            "inference_ms": round(inference_ms, 3),
            "total_ms": round(total_ms, 3),
        }

    def predict_from_sequence(self, sequence: np.ndarray) -> dict[str, Any]:
        """Direct prediction from a pre-built feature sequence.

        Args:
            sequence: Array of shape (seq_length, n_features) or
                      (batch, seq_length, n_features).

        Returns:
            Prediction result dict.
        """
        start_time = time.perf_counter()

        if sequence.ndim == 2:
            sequence = sequence.reshape(1, *sequence.shape)

        input_tensor = sequence.astype(np.float32)

        inference_start = time.perf_counter()
        outputs = self._session.run(
            [self._output_name], {self._input_name: input_tensor}
        )
        inference_ms = (time.perf_counter() - inference_start) * 1000

        predictions = outputs[0].flatten().tolist()

        # Inverse transform
        cpu_idx = self.scaler_feature_columns.index(self.target_metric)
        original_predictions = [
            p * self._scaler.scale_[cpu_idx] + self._scaler.mean_[cpu_idx]
            for p in predictions
        ]

        total_ms = (time.perf_counter() - start_time) * 1000

        return {
            "predictions_normalized": predictions,
            "predictions_original_scale": original_predictions,
            "batch_size": len(predictions),
            "target_metric": self.target_metric,
            "model_type": "lstm_onnx",
            "inference_ms": round(inference_ms, 3),
            "total_ms": round(total_ms, 3),
        }
