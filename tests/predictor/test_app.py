"""Tests for the predictor FastAPI application."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from k8s_ml_predictive_autoscaling.predictor.config import PredictorSettings


# ---------------------------------------------------------------------------
# Helpers -- Build a real PredictionEngine backed by a tiny ONNX model
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "cpu_usage", "mem_util_percent", "net_in", "net_out", "disk_io_percent",
    "cpu_usage_lag_1", "cpu_usage_lag_3", "cpu_usage_lag_6", "cpu_usage_lag_12",
    "mem_util_percent_lag_1", "mem_util_percent_lag_3", "mem_util_percent_lag_6",
    "mem_util_percent_lag_12",
    "net_in_lag_1", "net_in_lag_3", "net_in_lag_6", "net_in_lag_12",
    "net_out_lag_1", "net_out_lag_3", "net_out_lag_6", "net_out_lag_12",
    "disk_io_percent_lag_1", "disk_io_percent_lag_3", "disk_io_percent_lag_6",
    "disk_io_percent_lag_12",
    "cpu_usage_rolling_mean_3", "cpu_usage_rolling_mean_6", "cpu_usage_rolling_mean_12",
    "mem_util_percent_rolling_mean_3", "mem_util_percent_rolling_mean_6",
    "mem_util_percent_rolling_mean_12",
    "net_in_rolling_mean_3", "net_in_rolling_mean_6", "net_in_rolling_mean_12",
    "net_out_rolling_mean_3", "net_out_rolling_mean_6", "net_out_rolling_mean_12",
    "disk_io_percent_rolling_mean_3", "disk_io_percent_rolling_mean_6",
    "disk_io_percent_rolling_mean_12",
    "hour", "day_of_week", "is_weekend", "minute_of_day",
]

SCALER_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in
                          ("hour", "day_of_week", "is_weekend", "minute_of_day")]

N_FEATURES = len(FEATURE_COLUMNS)
SEQ_LEN = 6  # very short for fast tests


def _setup_test_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create minimal ONNX model, scaler, and metadata in tmp_path."""
    import joblib
    import onnx
    from onnx import TensorProto, helper
    from sklearn.preprocessing import StandardScaler

    # metadata
    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "scaler_feature_columns": SCALER_FEATURE_COLUMNS,
        "target_metric": "cpu_usage",
        "forecast_horizon": 3,
    }
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(json.dumps(meta))

    # scaler
    n_scaler_cols = len(SCALER_FEATURE_COLUMNS)
    scaler = StandardScaler()
    rng = np.random.default_rng(42)
    scaler.fit(rng.standard_normal((100, n_scaler_cols)).astype(np.float32))
    scaler_path = tmp_path / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    # ONNX model -- ReduceMean
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, SEQ_LEN, N_FEATURES])
    reduce_node = helper.make_node(
        "ReduceMean", inputs=["input"], outputs=["reduced"], axes=[1, 2], keepdims=True,
    )
    reshape_shape = helper.make_tensor("shape", TensorProto.INT64, [2], [-1, 1])
    reshape_node = helper.make_node(
        "Reshape", inputs=["reduced", "shape"], outputs=["output"],
    )
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])
    graph = helper.make_graph(
        [reduce_node, reshape_node], "test", [X], [Y], initializer=[reshape_shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx_path = tmp_path / "model.onnx"
    onnx.save(model, str(onnx_path))

    return onnx_path, scaler_path, meta_path


@pytest.fixture
def test_client(tmp_path: Path) -> Generator[TestClient, None, None]:
    """Create a TestClient with a real PredictionEngine.

    Uses context manager to ensure lifespan events (startup/shutdown) fire.
    """
    onnx_path, scaler_path, meta_path = _setup_test_artifacts(tmp_path)

    settings = PredictorSettings(
        onnx_model_path=onnx_path,
        scaler_path=scaler_path,
        metadata_path=meta_path,
        sequence_length=SEQ_LEN,
    )

    from k8s_ml_predictive_autoscaling.predictor.app import create_app

    app = create_app(settings)
    with TestClient(app) as client:
        yield client


def _observation_payload(cpu: float = 50.0, hour: int = 12) -> dict:
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


class TestHealthEndpoints:

    def test_health_returns_ok(self, test_client: TestClient) -> None:
        resp = test_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ready_returns_warming(self, test_client: TestClient) -> None:
        resp = test_client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_loaded"] is True
        assert data["status"] == "warming"
        assert data["ready_for_prediction"] is False


class TestIngestEndpoint:

    def test_ingest_returns_202(self, test_client: TestClient) -> None:
        resp = test_client.post("/ingest", json=_observation_payload())
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["buffer_size"] == 1

    def test_ingest_increments_buffer(self, test_client: TestClient) -> None:
        for i in range(3):
            resp = test_client.post("/ingest", json=_observation_payload(cpu=float(i)))
        data = resp.json()
        assert data["buffer_size"] == 3

    def test_ingest_validates_payload(self, test_client: TestClient) -> None:
        resp = test_client.post("/ingest", json={"bad_field": 1.0})
        assert resp.status_code == 422  # Validation error


class TestPredictEndpoint:

    def test_predict_returns_425_when_not_ready(self, test_client: TestClient) -> None:
        # Ingest fewer observations than required
        test_client.post("/ingest", json=_observation_payload())
        resp = test_client.post("/predict")
        assert resp.status_code == 425

    def test_predict_returns_200_when_ready(self, test_client: TestClient) -> None:
        for i in range(SEQ_LEN):
            test_client.post("/ingest", json=_observation_payload(cpu=30.0 + i, hour=i % 24))

        resp = test_client.post("/predict")
        assert resp.status_code == 200

        data = resp.json()
        assert "prediction_normalized" in data
        assert "prediction_original_scale" in data
        assert data["target_metric"] == "cpu_usage"
        assert data["model_type"] == "lstm_onnx"
        assert data["forecast_horizon_steps"] == 3
        assert data["forecast_horizon_minutes"] == 15
        assert data["buffer_size"] == SEQ_LEN
        assert data["inference_ms"] > 0

    def test_ready_becomes_true_after_filling_buffer(
        self, test_client: TestClient
    ) -> None:
        for i in range(SEQ_LEN):
            test_client.post("/ingest", json=_observation_payload(cpu=float(i)))

        resp = test_client.get("/ready")
        data = resp.json()
        assert data["status"] == "ready"
        assert data["ready_for_prediction"] is True


class TestBatchPredictEndpoint:

    def test_batch_predict_returns_200(self, test_client: TestClient) -> None:
        batch = np.random.randn(2, SEQ_LEN, N_FEATURES).astype(np.float32).tolist()
        resp = test_client.post("/predict/batch", json={"sequences": batch})
        assert resp.status_code == 200

        data = resp.json()
        assert data["batch_size"] == 2
        assert len(data["predictions_normalized"]) == 2
        assert len(data["predictions_original_scale"]) == 2

    def test_batch_predict_rejects_too_large(self, test_client: TestClient) -> None:
        # Default max_batch_size is 32, send 33
        batch = np.random.randn(33, SEQ_LEN, N_FEATURES).astype(np.float32).tolist()
        resp = test_client.post("/predict/batch", json={"sequences": batch})
        assert resp.status_code == 400


class TestModelInfoEndpoint:

    def test_model_info_returns_metadata(self, test_client: TestClient) -> None:
        resp = test_client.get("/model/info")
        assert resp.status_code == 200

        data = resp.json()
        assert data["model_type"] == "lstm_onnx"
        assert data["target_metric"] == "cpu_usage"
        assert data["sequence_length"] == SEQ_LEN
        assert data["n_features"] == N_FEATURES
        assert isinstance(data["feature_columns"], list)


class TestMetricsEndpoint:

    def test_metrics_returns_prometheus_text(self, test_client: TestClient) -> None:
        resp = test_client.get("/metrics")
        assert resp.status_code == 200
        # Prometheus format
        body = resp.text
        assert "predictor_" in body or "python_" in body or "process_" in body
