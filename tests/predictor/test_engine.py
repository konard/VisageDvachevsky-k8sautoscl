"""Tests for the PredictionEngine (ONNX inference)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from k8s_ml_predictive_autoscaling.predictor.engine import PredictionEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    # Base metrics (5)
    "cpu_usage", "mem_util_percent", "net_in", "net_out", "disk_io_percent",
    # Lag features (20)
    "cpu_usage_lag_1", "cpu_usage_lag_3", "cpu_usage_lag_6", "cpu_usage_lag_12",
    "mem_util_percent_lag_1", "mem_util_percent_lag_3", "mem_util_percent_lag_6",
    "mem_util_percent_lag_12",
    "net_in_lag_1", "net_in_lag_3", "net_in_lag_6", "net_in_lag_12",
    "net_out_lag_1", "net_out_lag_3", "net_out_lag_6", "net_out_lag_12",
    "disk_io_percent_lag_1", "disk_io_percent_lag_3", "disk_io_percent_lag_6",
    "disk_io_percent_lag_12",
    # Rolling means (15)
    "cpu_usage_rolling_mean_3", "cpu_usage_rolling_mean_6", "cpu_usage_rolling_mean_12",
    "mem_util_percent_rolling_mean_3", "mem_util_percent_rolling_mean_6",
    "mem_util_percent_rolling_mean_12",
    "net_in_rolling_mean_3", "net_in_rolling_mean_6", "net_in_rolling_mean_12",
    "net_out_rolling_mean_3", "net_out_rolling_mean_6", "net_out_rolling_mean_12",
    "disk_io_percent_rolling_mean_3", "disk_io_percent_rolling_mean_6",
    "disk_io_percent_rolling_mean_12",
    # Time features (4)
    "hour", "day_of_week", "is_weekend", "minute_of_day",
]

SCALER_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in
                          ("hour", "day_of_week", "is_weekend", "minute_of_day")]

N_FEATURES = len(FEATURE_COLUMNS)
SEQ_LEN = 12  # shorter for faster tests


def _make_metadata(tmp_path: Path) -> Path:
    """Write a fake metadata.json and return its path."""
    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "scaler_feature_columns": SCALER_FEATURE_COLUMNS,
        "target_metric": "cpu_usage",
        "forecast_horizon": 3,
    }
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(meta))
    return path


def _make_scaler(tmp_path: Path) -> Path:
    """Write a fake StandardScaler and return its path."""
    import joblib
    from sklearn.preprocessing import StandardScaler

    n_scaler_cols = len(SCALER_FEATURE_COLUMNS)
    scaler = StandardScaler()
    # Fit on some dummy data so mean_ and scale_ are populated
    rng = np.random.default_rng(42)
    dummy = rng.standard_normal((100, n_scaler_cols)).astype(np.float32)
    scaler.fit(dummy)
    path = tmp_path / "scaler.pkl"
    joblib.dump(scaler, path)
    return path


def _make_dummy_onnx(tmp_path: Path) -> Path:
    """Create a minimal ONNX model that accepts (batch, seq_len, n_features)."""
    import onnx
    from onnx import TensorProto, helper

    # A trivial model: just a ReduceMean over axes [1,2] -> scalar per batch
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, SEQ_LEN, N_FEATURES])
    Y = helper.make_tensor_value_info("prediction", TensorProto.FLOAT, [None, 1])

    reduce_node = helper.make_node(
        "ReduceMean",
        inputs=["input"],
        outputs=["prediction"],
        axes=[1, 2],
        keepdims=True,
    )
    # Reshape to (batch, 1)
    reshape_shape = helper.make_tensor("shape", TensorProto.INT64, [2], [-1, 1])
    reshape_node = helper.make_node(
        "Reshape",
        inputs=["prediction", "shape"],
        outputs=["output"],
    )
    Y_out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])

    graph = helper.make_graph(
        [reduce_node, reshape_node],
        "test_model",
        [X],
        [Y_out],
        initializer=[reshape_shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    path = tmp_path / "model.onnx"
    onnx.save(model, str(path))
    return path


@pytest.fixture
def engine(tmp_path: Path) -> PredictionEngine:
    """Build a PredictionEngine backed by a tiny dummy ONNX model."""
    metadata_path = _make_metadata(tmp_path)
    scaler_path = _make_scaler(tmp_path)
    onnx_path = _make_dummy_onnx(tmp_path)
    return PredictionEngine(
        onnx_model_path=onnx_path,
        scaler_path=scaler_path,
        metadata_path=metadata_path,
        sequence_length=SEQ_LEN,
    )


def _observation(cpu: float = 0.5, hour: int = 12) -> dict[str, float]:
    """Build a minimal metric observation dict."""
    return {
        "cpu_usage": cpu,
        "mem_util_percent": 40.0,
        "net_in": 100.0,
        "net_out": 50.0,
        "disk_io_percent": 10.0,
        "hour": hour,
        "day_of_week": 2,
        "is_weekend": 0,
        "minute_of_day": hour * 60,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEngineProperties:
    """Tests for basic engine properties."""

    def test_initial_state(self, engine: PredictionEngine) -> None:
        assert engine.buffer_size == 0
        assert engine.is_ready is False
        assert engine.sequence_length == SEQ_LEN
        assert engine.n_features == N_FEATURES
        assert engine.target_metric == "cpu_usage"

    def test_feature_columns(self, engine: PredictionEngine) -> None:
        assert len(engine.feature_columns) == N_FEATURES
        assert "cpu_usage" in engine.feature_columns
        assert "hour" in engine.feature_columns

    def test_forecast_horizon(self, engine: PredictionEngine) -> None:
        assert engine.forecast_horizon == 3


class TestIngest:
    """Tests for the ingest (sliding window) functionality."""

    def test_ingest_adds_to_buffer(self, engine: PredictionEngine) -> None:
        engine.ingest(_observation())
        assert engine.buffer_size == 1

    def test_buffer_fills_up(self, engine: PredictionEngine) -> None:
        for i in range(SEQ_LEN):
            engine.ingest(_observation(cpu=float(i)))
        assert engine.is_ready is True
        assert engine.buffer_size == SEQ_LEN

    def test_buffer_respects_maxlen(self, engine: PredictionEngine) -> None:
        # Fill past maxlen (seq_len + 30)
        maxlen = SEQ_LEN + 30
        for i in range(maxlen + 10):
            engine.ingest(_observation(cpu=float(i)))
        assert engine.buffer_size == maxlen


class TestPredict:
    """Tests for the predict() method."""

    def test_predict_raises_when_not_ready(self, engine: PredictionEngine) -> None:
        engine.ingest(_observation())
        with pytest.raises(RuntimeError, match="Not enough data"):
            engine.predict()

    def test_predict_returns_valid_result(self, engine: PredictionEngine) -> None:
        for i in range(SEQ_LEN):
            engine.ingest(_observation(cpu=30.0 + i, hour=i % 24))

        result = engine.predict()

        assert "prediction_normalized" in result
        assert "prediction_original_scale" in result
        assert "target_metric" in result
        assert result["target_metric"] == "cpu_usage"
        assert result["model_type"] == "lstm_onnx"
        assert result["buffer_size"] == SEQ_LEN
        assert result["forecast_horizon_steps"] == 3
        assert result["forecast_horizon_minutes"] == 15
        assert isinstance(result["inference_ms"], float)
        assert isinstance(result["total_ms"], float)
        assert result["inference_ms"] > 0
        assert result["total_ms"] >= result["inference_ms"]

    def test_predict_with_extra_buffer(self, engine: PredictionEngine) -> None:
        """Predict should use most recent seq_len observations."""
        for i in range(SEQ_LEN + 10):
            engine.ingest(_observation(cpu=float(i)))

        result = engine.predict()
        assert result["buffer_size"] == SEQ_LEN + 10
        assert "prediction_normalized" in result


class TestPredictFromSequence:
    """Tests for predict_from_sequence()."""

    def test_single_sequence(self, engine: PredictionEngine) -> None:
        seq = np.random.randn(SEQ_LEN, N_FEATURES).astype(np.float32)
        result = engine.predict_from_sequence(seq)

        assert result["batch_size"] == 1
        assert len(result["predictions_normalized"]) == 1
        assert len(result["predictions_original_scale"]) == 1
        assert result["model_type"] == "lstm_onnx"
        assert result["inference_ms"] > 0

    def test_batch_sequences(self, engine: PredictionEngine) -> None:
        batch_size = 4
        seq = np.random.randn(batch_size, SEQ_LEN, N_FEATURES).astype(np.float32)
        result = engine.predict_from_sequence(seq)

        assert result["batch_size"] == batch_size
        assert len(result["predictions_normalized"]) == batch_size
        assert len(result["predictions_original_scale"]) == batch_size


class TestFeatureEngineering:
    """Tests for the internal feature engineering logic."""

    def test_feature_matrix_shape(self, engine: PredictionEngine) -> None:
        observations = [_observation(cpu=float(i)) for i in range(SEQ_LEN)]
        features = engine._engineer_features(observations)

        assert features.shape == (SEQ_LEN, N_FEATURES)
        assert features.dtype == np.float32

    def test_time_features_normalized(self, engine: PredictionEngine) -> None:
        """Time features should be in [0, 1] range."""
        observations = [_observation(hour=23, cpu=50.0)] * SEQ_LEN
        features = engine._engineer_features(observations)

        col_idx = {name: i for i, name in enumerate(engine.feature_columns)}
        hour_val = features[0, col_idx["hour"]]
        assert 0.0 <= hour_val <= 1.0
        assert abs(hour_val - 23.0 / 23.0) < 1e-5

    def test_lag_features_populated(self, engine: PredictionEngine) -> None:
        """Lag features should be populated from prior observations."""
        observations = [_observation(cpu=float(i * 10)) for i in range(SEQ_LEN)]
        features = engine._engineer_features(observations)

        col_idx = {name: i for i, name in enumerate(engine.feature_columns)}
        # At t=5, cpu_usage_lag_1 should equal observation[4].cpu_usage = 40.0
        lag1_val = features[5, col_idx["cpu_usage_lag_1"]]
        assert abs(lag1_val - 40.0) < 1e-5

    def test_rolling_mean_features(self, engine: PredictionEngine) -> None:
        """Rolling mean should average over the window."""
        observations = [_observation(cpu=10.0 * i) for i in range(SEQ_LEN)]
        features = engine._engineer_features(observations)

        col_idx = {name: i for i, name in enumerate(engine.feature_columns)}
        # At t=3, rolling_mean_3 of cpu_usage = mean(10, 20, 30) = 20.0
        rm3 = features[3, col_idx["cpu_usage_rolling_mean_3"]]
        assert abs(rm3 - 20.0) < 1e-5
